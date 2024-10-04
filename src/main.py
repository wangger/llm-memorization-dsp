import torch
import uuid
import numpy as np
import transformers
from torch.utils.data import DataLoader
from accelerate import Accelerator
import argparse
from torch.utils.tensorboard import SummaryWriter
import utils as ut
from sklearn.model_selection import train_test_split
torch.backends.cudnn.enabled = True
from utils import seed_everything
from transformers import AutoTokenizer
import os
import copy
import time
from model_config import LLM_dict, exclude_keywords_dict, lr_keywords_dict
from generator_utils import build_generator, DynamicSoftEmbedding, is_target_module
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description='Tune a dynamic soft-prompt, generate sequences by appending it to given prompts')
    parser.add_argument('--seed', type=int, default=42, help='random seed for the run')

    # Architecture related
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers from the target LLM to serve as the generator')
    parser.add_argument('--is_init_from_pretrain', type=str, default='True', help='whether to initialize the generator from pretrained LLM')
    parser.add_argument('--is_zero_init', type=str, default='True', help='whether to initialize the generator w/ zero init')
    parser.add_argument('--is_constant_input', type=str, default='False', help='whether the input to the generator is constant')
    parser.add_argument('--embed_idx', type=int, default=-2, help='the index for to retrieve from the hidden status')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large', 'gpt2', 'gpt2XL'], help='indicate which of 125M-1.3B-2.7B (small-medium-large) models of gpt-neo to use')

    # Training setting
    parser.add_argument('--num_epochs', type=int, default=15, help='number of epochs to train the soft-prompt')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate to train attention params to generate the soft-prompt')
    parser.add_argument('--embed_lr', type=float, default=1e-4, help='learning rate to train embedding params to generate the soft-prompt')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay to train the soft-prompt')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of steps for gradient accumulation')
    parser.add_argument('--num_beams', type=int, default=1, help='beam size for beam decoding')
    parser.add_argument('--len_prompt', type=int, default=20, help='size of the soft prompt in number of tokens')
    parser.add_argument('--aligned', type=int, default=1, help='compute loss only over suffix if set')
    parser.add_argument('--prefix_size', type=int, default=50, help='size of prefix we provide to the model')
    parser.add_argument('--suffix_size', type=int, default=50, help='size of suffix we generate from the model')
    parser.add_argument('--test_set_size', type=int, default=1000, help='size of the evaluation dataset')

    # Dataset related
    parser.add_argument('--dataset_dir', type=str, default='../datasets', help="root dir path of the data files")
    parser.add_argument('--log_dir', type=str, default='../logs/', help="root dir path of the log files")
    parser.add_argument('--result_dir', type=str, default='../results/', help="root dir path of the result files")
    parser.add_argument('--train_preprefix', type=str, default='train_preprefix.npy',
                        help="path to binary train_preprefix file")
    parser.add_argument('--train_prefix', type=str, default='train_prefix.npy',
                        help="path to binary train_prefix file")
    parser.add_argument('--train_suffix', type=str, default='train_suffix.npy',
                        help="path to binary train_suffix file")
    parser.add_argument('--test_prefix', type=str, default='pile_test_ppl.npy',
                        help="path to binary test_prefix file")    
    args = parser.parse_args()

    if args.is_init_from_pretrain =='True':
        args.is_init_from_pretrain = True
    else:
        args.is_init_from_pretrain = False

    if args.is_zero_init =='True':
        args.is_zero_init = True
    else:
        args.is_zero_init = False

    if args.is_constant_input =='True':
        args.is_constant_input = True
    else:
        args.is_constant_input = False

    start_time = time.time()
    accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=args.gradient_accumulation_steps)

    if args.model_size in LLM_dict:
        MODEL_PATH = LLM_dict[args.model_size]
    else:
        raise Exception("Not supported model of LLM!")

    GENERATOR_PATH = MODEL_PATH

    # prepare datasets & dataloaders
    accelerator.print('Loading dataset..')
    DATASET_PATH = args.dataset_dir
    prefixes = np.concatenate((ut.load_prompts(f'{DATASET_PATH}/{args.train_preprefix}'), ut.load_prompts(f'{DATASET_PATH}/{args.train_prefix}')), axis=1)[:, -args.prefix_size:]
    suffixes = ut.load_prompts(f'{DATASET_PATH}/{args.train_suffix}')[:, :args.suffix_size]

    # Set random seed
    seed_everything(args.seed)

    # sample a random training/test set
    prefix_tr, prefix_test, suffix_tr, suffix_test = train_test_split(prefixes, suffixes, test_size=args.test_set_size)
    model_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left", device_map="auto")

    # get the pad_token_id
    accelerator.print("The pad_token before assignment is ", model_tokenizer.pad_token)
    if model_tokenizer.pad_token is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token
    model_pad_token_id = model_tokenizer.pad_token_id
    accelerator.print("The pad_token after assignment is ", model_pad_token_id)

    if args.len_prompt <= args.prefix_size:
        if not args.is_constant_input:
            train_ds = torch.cat([torch.tensor(prefix_tr, dtype=torch.int64)[:, -args.len_prompt:], torch.tensor(prefix_tr, dtype=torch.int64), torch.tensor(suffix_tr, dtype=torch.int64)], dim=1)

            test_ds = torch.cat([torch.tensor(prefix_test, dtype=torch.int64)[:, -args.len_prompt:], torch.tensor(prefix_test, dtype=torch.int64), torch.tensor(suffix_test, dtype=torch.int64)], dim=1)
        else:
            train_ds = torch.cat([torch.tensor(range(0, args.len_prompt), dtype=torch.int64).repeat(len(prefix_tr), 1), torch.tensor(prefix_tr, dtype=torch.int64), torch.tensor(suffix_tr, dtype=torch.int64)], dim=1)

            test_ds = torch.cat([torch.tensor(range(0, args.len_prompt), dtype=torch.int64).repeat(len(prefix_test), 1), torch.tensor(prefix_test, dtype=torch.int64), torch.tensor(suffix_test, dtype=torch.int64)], dim=1)      

    else:
        if not args.is_constant_input:
            min_repeat = args.len_prompt//args.prefix_size
            if args.len_prompt % args.prefix_size == 0:
                n_repeat = min_repeat
            else:
                n_repeat = min_repeat + 1
            
            prompt_tr = prefix_tr
            prompt_test = prefix_test
            for _ in range(n_repeat-1):
                prompt_tr = np.concatenate((prompt_tr, prefix_tr), axis=1)
                prompt_test = np.concatenate((prompt_test, prefix_test), axis=1)

            assert prompt_tr.shape[1] == n_repeat * args.prefix_size, "Error in duplicating prefixes for prompting"
            prompt_tr = prompt_tr[:, -args.len_prompt:]
            prompt_test = prompt_test[:, -args.len_prompt:]

            train_ds = torch.cat([torch.tensor(prompt_tr, dtype=torch.int64), torch.tensor(prefix_tr, dtype=torch.int64), torch.tensor(suffix_tr, dtype=torch.int64)],dim=1)
            test_ds = torch.cat([torch.tensor(prompt_test, dtype=torch.int64), torch.tensor(prefix_test, dtype=torch.int64), torch.tensor(suffix_test, dtype=torch.int64)],dim=1)
        else:
            train_ds = torch.cat([torch.tensor(range(0, args.len_prompt), dtype=torch.int64).repeat(len(prefix_tr), 1), torch.tensor(prefix_tr, dtype=torch.int64), torch.tensor(suffix_tr, dtype=torch.int64)], dim=1)

            test_ds = torch.cat([torch.tensor(range(0, args.len_prompt), dtype=torch.int64).repeat(len(prefix_test), 1), torch.tensor(prefix_test, dtype=torch.int64), torch.tensor(suffix_test, dtype=torch.int64)], dim=1)    

    # dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.bs)

    # load model
    accelerator.print('Loading model..')
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(accelerator.device)
    accelerator.print("The verified device is ", model.device, accelerator.device)
    device = model.device

    accelerator.print('Build generator..')
    generator = build_generator(target_mdoel=model.transformer, generator_name=GENERATOR_PATH, num_layers=args.num_layers, is_init_from_pretrain=args.is_init_from_pretrain, device=device, is_zero_init=args.is_zero_init)

    # freeze model params and add soft-prompting "layer"
    accelerator.print('Prepare Embedding..')
    for p in model.parameters():
        p.requires_grad=False
    
    ori_wte = copy.deepcopy(model.get_input_embeddings())
    dynamic_soft_prompt = DynamicSoftEmbedding(generator, ori_wte, args.len_prompt, args.embed_idx, device)
    model.set_input_embeddings(dynamic_soft_prompt)

    # Exclude the last layer norm for optimizer
    exclude_keywords_list = exclude_keywords_dict[GENERATOR_PATH]
    for module_name, module in dynamic_soft_prompt.generator.named_modules():
        if is_target_module(exclude_keywords_list, module_name):
            assert isinstance(module, nn.LayerNorm) or isinstance(module, nn.Embedding), "error for excluding"
            for name, p in module.named_parameters():
                p.requires_grad=False

    lr_keywords_list = lr_keywords_dict[GENERATOR_PATH]    # usually for embed params
    embed_params = []
    block_params = []

    # Split the learnable parameters to groups for different lr
    embed_params_name = []
    block_params_name = []
    for module_name, module in dynamic_soft_prompt.generator.named_modules():
        if is_target_module(lr_keywords_list, module_name):
            assert isinstance(module, nn.Embedding), "error for lr adaptation"
            for name, p in module.named_parameters():
                embed_params.append(p)
                embed_params_name.append(module_name + '.' + name)
    
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.Linear):
            for name, p in module.named_parameters():
                block_params.append(p)
                block_params_name.append(module_name + '.' + name)

    optimizer = torch.optim.AdamW(
        params=[
            {'params': filter(lambda p: p.requires_grad,block_params)},
            {'params': filter(lambda p: p.requires_grad, embed_params), 'lr': args.embed_lr}
            ], lr=args.lr, weight_decay=args.weight_decay
        )

    # accelerator version of things
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    # creating tensorboard logger
    if accelerator.is_main_process:
        setting_part = f"Dynamic_Attack_modelSize:{args.model_size}_numLayers:{args.num_layers}_embedIdx_{str(args.embed_idx)}_lenPrompt:{args.len_prompt}_prefixSize:{args.prefix_size}_nEpochs:{args.num_epochs}_lr:{args.lr}_embed_lr:{args.embed_lr}_aligned:{args.aligned}"

        if args.is_init_from_pretrain:
            setting_part += "_init_from_pretrain"

        UUID = uuid.uuid1().hex
        id_part = f"tensorboard_dynamic_promptLearnAttack_id:{UUID}"
        file_name = os.path.join(setting_part, id_part)
        writer = SummaryWriter(args.log_dir + file_name)
        accelerator.print("The setting part is ", setting_part)
        accelerator.print("The id part is ", id_part)
        accelerator.print("The file name is ", file_name)
    
    else:
        writer = None
        file_name = None

    # training the prompt
    len_prompt_tr = args.len_prompt
    len_prompt_test = args.len_prompt

    # Save the best model and the last model
    best_loss = float('inf')
    best_generator = None
    best_epoch = None

    accelerator.print('Start Training..')
    for ep in range(args.num_epochs):
        model.train()
        tr_loss = []
        tol_data = 0
        dynamic_soft_prompt.set_n_tokens(len_prompt_tr)
        for batch in train_loader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    if args.aligned:
                        labels = torch.clone(batch)
                        # predicting only the last args.suffix_size tokens
                        # so ignore everything else in loss calculation
                        labels[:, :labels.shape[1]-args.suffix_size] = -100
                    else:
                        labels=batch

                outputs = model(input_ids=batch, labels=labels)
                cur_bs = labels.shape[0]
                tr_loss.append((outputs.loss.detach()*len(batch)).unsqueeze(0).cpu())

                accelerator.backward(outputs.loss)
                optimizer.step()
                optimizer.zero_grad()

                tol_data += cur_bs

        with torch.inference_mode():
            tr_loss = tr_loss[:len(train_loader.dataset)]
            tr_loss = (torch.sum(torch.cat(tr_loss)) / len(train_loader.dataset)).item()
            tr_plp = np.exp(tr_loss)

            dynamic_soft_prompt.set_n_tokens(len_prompt_test)
            test_loss = ut.evaluate(model, test_loader, args)
            test_plp = np.exp(test_loss)
            if accelerator.is_main_process:
                # Save the best model
                if test_loss < best_loss:
                    best_loss = test_loss
                    cur_generator = dynamic_soft_prompt.generator
                    best_generator = copy.deepcopy(cur_generator)
                    best_epoch = ep + 1

                writer.add_scalar('Train/Loss', tr_loss, ep)
                writer.add_scalar('Train/PLP', tr_plp, ep)
                writer.add_scalar('Test/Loss', test_loss, ep)
                writer.add_scalar('Test/PLP', test_plp, ep)
                accelerator.print(f'EP:{ep+1} Tr. Loss/PLP:{tr_loss:.3f}/{tr_plp:.3f}', end=' --- ')
                accelerator.print(f'Test Loss/PLP:{test_loss:.3f}/{test_plp:.3f}', end='\r\n')

    # Evaluate the Last generator
    accelerator.print("*"*20 + "Evaluate the Model with last generator" + "*"*20)   
    dynamic_soft_prompt.set_n_tokens(len_prompt_test)

    exact_rate_last, fract_rate_last, test_loss_last, test_plp_last = ut.evaluate_model_pipeline(model, accelerator, test_loader, writer, suffix_test, model_pad_token_id, args, suffix='_last')

    # Evaluate the Best generator
    accelerator.print("*"*20 + "Evaluate the Model with best generator" + "*"*20)
    best_dynamic_soft_prompt = DynamicSoftEmbedding(best_generator, ori_wte, len_prompt_test, args.embed_idx, device)
    model.set_input_embeddings(best_dynamic_soft_prompt)
    exact_rate_best, fract_rate_best, test_loss_best, test_plp_best = ut.evaluate_model_pipeline(model, accelerator, test_loader, writer, suffix_test, model_pad_token_id, args, suffix='_best')

    # Save the two generators
    if accelerator.is_main_process:
        # last
        save_name = 'last_generator'
        last_path = os.path.join(args.log_dir, file_name, save_name)
        last_generator = dynamic_soft_prompt.generator
        last_generator.save_pretrained(last_path)

        # best
        save_name = 'best_generator'
        best_path = os.path.join(args.log_dir, file_name, save_name)
        best_generator.save_pretrained(best_path)

    # Write the result to the final file
    if accelerator.is_main_process:
        result_name = f'lenPrompt:{str(args.len_prompt)}_prefixSize:{args.prefix_size}_model_{args.model_size}_numLayers_{str(args.num_layers)}_embedIdx_{str(args.embed_idx)}_aligned:{args.aligned}'

        result_dir = os.path.join(args.result_dir, 'dynamic_prompt_attack')
        result_dir = os.path.join(result_dir, result_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        result_path = os.path.join(result_dir, 'tol_result.csv')
        result_cols = ['seed', 'UUID', 'Is_best', 'is_zero_init', 'is_constant_input', 'model_size', 'num_layers', 'embed_idx', 'len_prompt', 'lr', 'embed_lr', 'best_epoch','num_epochs',  'Exact Extract Rate', 'Frac Extract Rate', 'Test Loss', 'Test PLP', 'is_init_from_pretrain', 'batch size', 'gradient_accumulation_steps', 'aligned', 'prefix_size', 'suffix_size', 'numBeams']

        result_vals_best = [args.seed, UUID, True, str(args.is_zero_init), str(args.is_constant_input), args.model_size, args.num_layers, args.embed_idx, args.len_prompt, args.lr, args.embed_lr, best_epoch, args.num_epochs, exact_rate_best, fract_rate_best, test_loss_best, test_plp_best, str(args.is_init_from_pretrain), args.bs, args.gradient_accumulation_steps, str(args.aligned), args.prefix_size, args.suffix_size, args.num_beams]

        result_vals_last = [args.seed, UUID, False, str(args.is_zero_init), str(args.is_constant_input), args.model_size, args.num_layers, args.embed_idx, args.len_prompt, args.lr, args.embed_lr, best_epoch, args.num_epochs, exact_rate_last, fract_rate_last, test_loss_last, test_plp_last, str(args.is_init_from_pretrain), args.bs, args.gradient_accumulation_steps, str(args.aligned), args.prefix_size, args.suffix_size, args.num_beams]

        ut.write_results_to_csv(result_path, result_cols, result_vals_best)
        ut.write_results_to_csv(result_path, result_cols, result_vals_last)

    end_time = time.time()
    elaps_time = (end_time - start_time)/3600
    accelerator.print('The totol execution time is {} hours'.format(elaps_time))

if __name__ == "__main__":
    main()