LLM_dict ={
    'small': 'EleutherAI/gpt-neo-125M',
    'medium': 'EleutherAI/gpt-neo-1.3B',
    'large': 'EleutherAI/gpt-neo-2.7B',
    'gpt2': 'gpt2',
    'gpt2XL': 'gpt2-xl',
    'pythia_ts': 'EleutherAI/pythia-160m',
    'pythia_s': 'EleutherAI/pythia-410m',
    'pythia_m': 'EleutherAI/pythia-1.4b',
    'pythia_l': 'EleutherAI/pythia-2.8b',
    'pythia_xl': 'EleutherAI/pythia-6.9b',
    'pythia_xxl': 'EleutherAI/pythia-12b'
}

# Zero Init (attention block)
init_keywords_dict = {
    'EleutherAI/gpt-neo-125M': ['attention.out_proj', 'mlp.c_proj'],
    'EleutherAI/gpt-neo-1.3B': ['attention.out_proj', 'mlp.c_proj'],
    'EleutherAI/gpt-neo-2.7B': ['attention.out_proj', 'mlp.c_proj'],
    'EleutherAI/pythia-160m': ['attention.dense', 'mlp.dense_4h_to_h'],
    'EleutherAI/pythia-410m': ['attention.dense', 'mlp.dense_4h_to_h'],
    'EleutherAI/pythia-1.4b': ['attention.dense', 'mlp.dense_4h_to_h'],
    'EleutherAI/pythia-2.8b': ['attention.dense', 'mlp.dense_4h_to_h'],
    'EleutherAI/pythia-6.9b': ['attention.dense', 'mlp.dense_4h_to_h'],
    'EleutherAI/pythia-12b': ['attention.dense', 'mlp.dense_4h_to_h']
}

# Param not updated (usually last/final ln)
exclude_keywords_dict = {
    'EleutherAI/gpt-neo-125M': ['ln_f'],
    'EleutherAI/gpt-neo-1.3B': ['ln_f'],
    'EleutherAI/gpt-neo-2.7B': ['ln_f'],
    'EleutherAI/pythia-160m': ['final_layer_norm'],
    'EleutherAI/pythia-410m': ['final_layer_norm'],
    'EleutherAI/pythia-1.4b': ['final_layer_norm'],
    'EleutherAI/pythia-2.8b': ['final_layer_norm'],
    'EleutherAI/pythia-6.9b': ['final_layer_norm'],
    'EleutherAI/pythia-12b': ['final_layer_norm']
}

# Param with different lr (Learable Embedding)
lr_keywords_dict = {
    'EleutherAI/gpt-neo-125M': ['wte', 'wpe'],
    'EleutherAI/gpt-neo-1.3B': ['wte', 'wpe'],
    'EleutherAI/gpt-neo-2.7B': ['wte', 'wpe'],
    'EleutherAI/pythia-160m': ['embed_in'],
    'EleutherAI/pythia-410m': ['embed_in'],
    'EleutherAI/pythia-1.4b': ['embed_in'],
    'EleutherAI/pythia-2.8b': ['embed_in'],
    'EleutherAI/pythia-6.9b': ['embed_in'],
    'EleutherAI/pythia-12b': ['embed_in']
}

pythia_filter = [158, 398, 488, 702, 719, 1096, 1201, 1630, 1695, 1766, 1767, 1769, 2227, 2645, 2715, 2792, 2815, 3249, 3273, 4285, 4321, 4578, 4579, 4580, 4631, 4685, 4939, 4996, 5086, 5241, 5387, 5458, 5509, 5661, 5712, 5713, 5774, 6045, 6046, 6047, 6097, 6190, 6222, 6523, 6583, 6627, 6992, 6993, 6994, 7946, 7957, 8273, 8369, 8373, 8441, 8695, 8829, 9294, 10875, 11104, 11490, 11752, 12439, 12937, 13007, 13278, 13294, 13505, 13549, 13793, 13895, 13937, 13955, 14234, 14398, 14467, 14494, 14773, 14819]

# index for filtering the dataset
data_filter_dict = {
    'EleutherAI/gpt-neo-125M': [],  
    'EleutherAI/gpt-neo-1.3B': [],
    'EleutherAI/gpt-neo-2.7B': [],
    'EleutherAI/pythia-160m': pythia_filter,
    'EleutherAI/pythia-410m': pythia_filter,
    'EleutherAI/pythia-1.4b': pythia_filter,
    'EleutherAI/pythia-2.8b': pythia_filter,
    'EleutherAI/pythia-6.9b': pythia_filter,
    'EleutherAI/pythia-12b': pythia_filter
}

# name of function to build generator w/ our methods
from generator_utils import build_generator, build_generator_pythia
build_generator_func_dict = {
    'EleutherAI/gpt-neo-125M': build_generator,  
    'EleutherAI/gpt-neo-1.3B': build_generator,
    'EleutherAI/gpt-neo-2.7B': build_generator,
    'EleutherAI/pythia-160m': build_generator_pythia,
    'EleutherAI/pythia-410m': build_generator_pythia,
    'EleutherAI/pythia-1.4b': build_generator_pythia,
    'EleutherAI/pythia-2.8b': build_generator_pythia,
    'EleutherAI/pythia-6.9b': build_generator_pythia,
    'EleutherAI/pythia-12b': build_generator_pythia
}

# name of the attribute of the decoder part of the model (excluding the LM header)
decode_attr_dict = {
    'EleutherAI/gpt-neo-125M': 'transformer',  
    'EleutherAI/gpt-neo-1.3B': 'transformer',
    'EleutherAI/gpt-neo-2.7B': 'transformer',
    'EleutherAI/pythia-160m': 'gpt_neox',
    'EleutherAI/pythia-410m': 'gpt_neox',
    'EleutherAI/pythia-1.4b': 'gpt_neox',
    'EleutherAI/pythia-2.8b': 'gpt_neox',
    'EleutherAI/pythia-6.9b': 'gpt_neox',
    'EleutherAI/pythia-12b': 'gpt_neox'
}