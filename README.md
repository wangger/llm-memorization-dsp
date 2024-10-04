# Unlocking Memorization in Large Language Models with Dynamic Soft Prompting

## Introduction
This repository contains the source code to extract memorized training data from large language models (LLMs) using dynamic soft prompting. The implementation is based on the repo (https://github.com/amazon-science/controlling-llm-memorization). 

<div align="center">
	<img src="./materials/Github_Image.png" alt="ali_pay" width="600" />
</div>

## Execution
To run the code on GPT-Neo (125M), the command is as follows.

```bash
CUDA_VISIBLE_DEVICES="0" accelerate launch --mixed_precision=fp16 main.py --seed 42 --num_epochs 15 --embed_idx -2 --num_layers 2 --lr 7e-6 --embed_lr 2e-4 --bs 128 --gradient_accumulation_steps 1 --len_prompt 50 --model_size small --is_init_from_pretrain False --prefix_size 50 --suffix_size 50 --is_zero_init True --is_constant_input False --aligned 1 --test_set_size 1000
```

To run the code on GPT-Neo (1.3B), the command is as follows.

```bash
CUDA_VISIBLE_DEVICES="0" accelerate launch --mixed_precision=fp16 main.py --seed 42 --num_epochs 15 --embed_idx -2 --num_layers 2 --lr 5e-6 --embed_lr 1e-4 --bs 128 --gradient_accumulation_steps 4 --len_prompt 50 --model_size medium --is_init_from_pretrain False --prefix_size 50 --suffix_size 50 --is_zero_init True --is_constant_input False --aligned 1 --test_set_size 1000
```

To run the code on GPT-Neo (2.7B), the command is as follows.

```bash
CUDA_VISIBLE_DEVICES="0" accelerate launch --mixed_precision=fp16 main.py --seed 42 --num_epochs 15 --embed_idx -2 --num_layers 2 --lr 1e-6 --embed_lr 1e-4 --bs 128 --gradient_accumulation_steps 8 --len_prompt 50 --model_size large --is_init_from_pretrain False --prefix_size 50 --suffix_size 50 --is_zero_init True --is_constant_input False --aligned 1 --test_set_size 1000
```


## Citation
If you find the repo useful, please kindly star this repository and cite our papers:

```
@inproceedings{wang2024unlocking,
    title     = {Unlocking Memorization in Large Language Models with Dynamic Soft Prompting},
    author    = {Wang, Zhepeng and Bao, Runxue and Wu, Yawen and Taylor, Jackson and Xiao, Cao and Zheng, Feng and Jiang, Weiwen and Gao, Shangqian and Zhang, Yanfu},
    booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
    year      = {2024}
}
```