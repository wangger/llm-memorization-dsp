# Run on GPT-Neo (125M)
CUDA_VISIBLE_DEVICES="0" accelerate launch --mixed_precision=fp16 main.py --seed 42 --num_epochs 15 --embed_idx -2 --num_layers 2 --lr 7e-6 --embed_lr 2e-4 --bs 128 --gradient_accumulation_steps 1 --len_prompt 50 --model_size small --is_init_from_pretrain False --prefix_size 50 --suffix_size 50 --is_zero_init True --is_constant_input False --aligned 1 --test_set_size 1000

# Run on GPT-Neo (1.3B)
CUDA_VISIBLE_DEVICES="0" accelerate launch --mixed_precision=fp16 main.py --seed 42 --num_epochs 15 --embed_idx -2 --num_layers 2 --lr 5e-6 --embed_lr 1e-4 --bs 128 --gradient_accumulation_steps 4 --len_prompt 50 --model_size medium --is_init_from_pretrain False --prefix_size 50 --suffix_size 50 --is_zero_init True --is_constant_input False --aligned 1 --test_set_size 1000

# Run on GPT-Neo (2.7B)
CUDA_VISIBLE_DEVICES="0" accelerate launch --mixed_precision=fp16 main.py --seed 42 --num_epochs 15 --embed_idx -2 --num_layers 2 --lr 1e-6 --embed_lr 1e-4 --bs 128 --gradient_accumulation_steps 8 --len_prompt 50 --model_size large --is_init_from_pretrain False --prefix_size 50 --suffix_size 50 --is_zero_init True --is_constant_input False --aligned 1 --test_set_size 1000