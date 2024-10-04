import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import os
from pandas import DataFrame

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_prompts(prompt_path):
    """Loads prompts from the file pointed to prompt_path"""
    return np.load(prompt_path).astype(np.int64)

def compute_reconstruct_rate(generations, answers, args):
    """ compute fractional and exact reconstruction rates """
    reconstruct_success = generations == answers
    frac_reconstruct_rate = reconstruct_success[:, -args.suffix_size:].sum()/(args.suffix_size*args.test_set_size)
    exact_reconstruct_rate = np.all(reconstruct_success, axis=1).sum()/args.test_set_size

    return frac_reconstruct_rate, exact_reconstruct_rate


def generate_suffixes(model, data_loader, args, use_cache=True, pad_token_id=50256):
    """ generate suffixes from the supplied data loader """
    with torch.inference_mode():
        generations = []
        for batch in tqdm(data_loader):
            input_ids = batch[:, :-args.suffix_size]
            generated_tokens = model.generate(
                inputs=input_ids,
                max_new_tokens=args.suffix_size,
                do_sample=False,
                num_beams=args.num_beams,
                use_cache=use_cache,
                pad_token_id=pad_token_id
                )
            generations.extend(generated_tokens[:, -args.suffix_size:].cpu().numpy())
    return generations

def evaluate(model, data_loader, args):
    """ get inference loss on supplied data loader """
    model.eval()
    with torch.inference_mode():
        loss = 0
        for batch in data_loader:
            with torch.no_grad():
                if args.aligned:
                    labels = torch.clone(batch)
                    # predicting only the last args.suffix_size tokens,
                    # so ignore everything else in loss calculation
                    labels[:, :labels.shape[1]-args.suffix_size] = -100
                else:
                    labels=batch

            outputs = model(input_ids=batch, labels=labels)
            loss += (outputs.loss.item()*len(batch))
        return loss/len(data_loader.dataset)


def generate_suffixes_distributed(model, data_loader, args, accelerator, use_cache=True, pad_token_id=50256):
    """ generate suffixes from the supplied data loader (for distributed training) """
    with torch.inference_mode():
        generations = []
        for batch in tqdm(data_loader):
            # get a batch, and have the model generate new tokens
            input_ids = batch[:, :-args.suffix_size]
            generated_tokens = model.generate(
                inputs=input_ids,
                max_new_tokens=args.suffix_size,
                do_sample=False,
                num_beams=args.num_beams,
                use_cache=use_cache,
                )
            generations.extend(accelerator.gather(generated_tokens[:, -args.suffix_size:].contiguous()).cpu().numpy())
    # to match batch sizes, distributed training pad the last batch
    # we get rid of the extra samples by truncating
    return generations[:args.test_set_size]



def evaluate_distributed(model, data_loader, args, accelerator):
    """ get inference loss on supplied data loader (for distributed training) """
    model.eval()
    with torch.inference_mode():
        loss = []
        for batch in data_loader:
            with torch.no_grad():
                if args.aligned:
                    labels = torch.clone(batch)
                    # predicting only the last args.suffix_size tokens,
                    # so ignore everything else in loss calculation
                    labels[:, :labels.shape[1]-args.suffix_size] = -100
                else:
                    labels=batch
            outputs = model(input_ids=batch, labels=labels)
            loss.append(accelerator.gather(outputs.loss*len(batch)).cpu())
        # to match batch sizes, distributed training pad the last batch
        # we get rid of the extra samples by truncating
        loss = torch.cat(loss)[:args.test_set_size]
        return (torch.sum(loss) / args.test_set_size).item()


def evaluate_model_pipeline(model, accelerator, test_loader, writer, suffix_test, model_pad_token_id, args, suffix=''):
    """
    Args:
        suffix: indicate whether it is for best or last model
    """

    # generate suffixes
    accelerator.print('Start testing..')
    accelerator.print('Generating suffixes..')

    generations_test = generate_suffixes(model, test_loader, args, use_cache=False, pad_token_id=model_pad_token_id)
    generations_test = np.stack(generations_test, axis=0)

    # always measure the final loss over suffix tokens
    accelerator.print('Start Evaluation..')
    args.aligned = True
    test_loss = evaluate(model, test_loader, args)

    # log results
    accelerator.print('Log Results..')
    if accelerator.is_main_process:
        # measure  fractional and exact extract rates
        fract_rate, exact_rate = compute_reconstruct_rate(generations_test, suffix_test, args)
        accelerator.print(f'Exact/Fract extract rate:{exact_rate:.3f}/{fract_rate:.3f}')
        test_plp = np.exp(test_loss)
        accelerator.print(f'Test Loss/PLP:{test_loss:.3f}/{test_plp:.3f}')
        # log the results
        writer.add_scalar('Memorization/Fract_Rate' + suffix, fract_rate, 0)
        writer.add_scalar('Memorization/Exact_Rate' + suffix, exact_rate, 0)
        writer.add_scalar('Test_Final/Loss' + suffix, test_loss, 0)
        writer.add_scalar('Test_Final/PLP' + suffix, np.exp(test_loss), 0)
        writer.flush()
        writer.close()
        return exact_rate, fract_rate, test_loss, test_plp
    
    else:
        return None, None, None, None

def write_results_to_csv(path, cols, vals):
    result_dict = {}
    for idx in range(len(cols)):
        col = cols[idx]
        val = vals[idx]
        result_dict[col] = [val]
    
    result_dict_df = DataFrame(result_dict)
    if os.path.exists(path):
        result_dict_df.to_csv(path, mode='a+', index=False, header=False)
    else:
        result_dict_df.to_csv(path, mode='a+', index=False, header=True)