import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, List
from datasets import Dataset
from model_config import init_keywords_dict, lr_keywords_dict
from collections import OrderedDict
import copy


def is_target_module(keyword_list, module_name):
    for keyword in keyword_list:
        if keyword in module_name:
            return True
    
    return False

def build_generator_pythia(target_mdoel, generator_name, num_layers=2, is_init_from_pretrain=True, device='cuda', is_zero_init=True):
    # load config
    ori_generator = target_mdoel
    config = AutoConfig.from_pretrained(generator_name)

    # Modify config 
    config.num_hidden_layers = num_layers
    config.use_cache = False
    new_generator = AutoModel.from_config(config)
    new_generator.to(device)

    if is_init_from_pretrain:   # Copy weights
        new_generator.load_state_dict(ori_generator.state_dict(), strict=False)
    else:
        embed_state_dict = OrderedDict()
        embed_list = lr_keywords_dict[generator_name]
        for name, param in ori_generator.state_dict().items():
            if is_target_module(embed_list, name):
                embed_state_dict[name] = copy.deepcopy(param)
        new_generator.load_state_dict(embed_state_dict, strict=False)

    # zero out related parameters
    if is_zero_init:
        keyword_list = init_keywords_dict[generator_name]

        for module_name, module in new_generator.named_modules():
            if is_target_module(keyword_list, module_name):
                assert isinstance(module, nn.Linear), "error for zering out"
                for name, val in module.named_parameters():
                    setattr(module, name, nn.Parameter(torch.zeros_like(val)))

    return new_generator

def build_generator(target_mdoel, generator_name, num_layers=2, is_init_from_pretrain=True, device='cuda', is_zero_init=True):
    """
        target_mdoel: AutoModel

    """
    ori_generator = target_mdoel

    config = AutoConfig.from_pretrained(generator_name)
    config.num_layers = num_layers
    config.attention_layers = config.attention_layers[:num_layers]
    config.attention_types[0][1] = int(config.num_layers/len(config.attention_types[0][0]))
    new_generator = AutoModel.from_config(config)
    new_generator.to(device)

    if is_init_from_pretrain:   # Copy weights
        new_generator.load_state_dict(ori_generator.state_dict(), strict=False)
    else:
        embed_state_dict = OrderedDict()
        embed_list = lr_keywords_dict[generator_name]
        for name, param in ori_generator.state_dict().items():
            if is_target_module(embed_list, name):
                embed_state_dict[name] = copy.deepcopy(param)

        new_generator.load_state_dict(embed_state_dict, strict=False)
    # zero out related parameters
    if is_zero_init:
        keyword_list = init_keywords_dict[generator_name]

        for module_name, module in new_generator.named_modules():
            if is_target_module(keyword_list, module_name):
                assert isinstance(module, nn.Linear), "error for zering out"
                for name, val in module.named_parameters():
                    setattr(module, name, nn.Parameter(torch.zeros_like(val)))
        
        new_generator.wpe.weight = nn.Parameter(torch.zeros_like(new_generator.wpe.weight))

    return new_generator

class DynamicSoftEmbedding(nn.Module):
    def __init__(self,
                generator,
                wte: nn.Embedding,
                n_tokens: int,
                embed_idx: int =-2,
                device = 'cuda'):
        """appends learned embedding to 

        Args:
            generator: pretrained model (base, AutoModel)
            wte (nn.Embedding): original word embedding (LUT) for LLM
            n_tokens (int): number of tokens for dynamic soft embedding.
            embed_idx: Specify which output hidden states are used
            device: important for the attention mask in forward
        """
        super(DynamicSoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.device = device
        self.generator = generator
        self.embed_idx = embed_idx

    def set_n_tokens(self, n_tokens):
        self.n_tokens = n_tokens

    def forward(self, tokens):
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        ori_embedding_tokens = tokens[:, :self.n_tokens] 

        dynamic_soft_embedding = self.generator(input_ids=ori_embedding_tokens, output_hidden_states=True)['hidden_states'][self.embed_idx]
        return torch.cat(tensors=[dynamic_soft_embedding, input_embedding], dim=1)


def token_ids_to_raw_strings(ori_ds_part: torch.Tensor, tokenizer) -> List[str]:
    """
    It is only used for the experiments for GPT-2 settings.
    """
    str_ds_part = tokenizer.batch_decode(ori_ds_part, skip_special_tokens=True)
    return str_ds_part


def build_fix_dict_from_tokens(ori_ds_part: torch.Tensor) -> Dict:
    """
    It is only used for the experiments for GPT-2 settings.
    """
    fix_dict = {}
    fix_dict['input_ids'] = ori_ds_part
    attention_mask = torch.ones_like(input=ori_ds_part, dtype=torch.int8)
    fix_dict['attention_mask'] = attention_mask

    return fix_dict

def convert_Dataset_to_dict(ori_ds_part: Dataset) -> Dict:
    dict_part = {}
    dict_part['input_ids'] = ori_ds_part['input_ids']
    dict_part['attention_mask'] = ori_ds_part['attention_mask']

    return dict_part

def build_generator_dict(str_ds_part: List[str], tokenizer)->Dict:
    input_dict = {}
    tokenzied_ds = tokenizer(str_ds_part, padding=True, return_tensors='pt', return_attention_mask=True)   

    for key, val in tokenzied_ds.items():
        input_dict[key] = val
    
    return input_dict

def build_input_dataset(inputs_dict_list) -> Dataset:
    key_list = ['input_ids', 'attention_mask']
    merge_dict = {}
    for key in key_list:
        concat_list = [single_dict[key] for single_dict in inputs_dict_list]
        merge_val = torch.cat(concat_list, dim=-1)
        merge_dict[key] = merge_val
    
    merge_ds = Dataset.from_dict(mapping=merge_dict)
    return merge_ds