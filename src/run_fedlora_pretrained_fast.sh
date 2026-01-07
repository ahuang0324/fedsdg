#!/bin/bash
# FedLoRA with Pretrained ViT - 优化版（更快训练）
# 通过减小 batch size 和 local epochs 加速训练

export HF_ENDPOINT=https://hf-mirror.com

python3 federated_main.py \
    --alg fedlora \
    --model vit \
    --model_variant pretrained \
    --dataset cifar \
    --image_size 224 \
    --epochs 80 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 3 \
    --local_bs 16 \
    --lr 0.0001 \
    --optimizer adam \
    --lora_r 8 \
    --lora_alpha 8 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 100 \
    --gpu 0 \
    --log_subdir fedlora_pretrained_vit_cifar_E80_bs16_le3_fast

# 优化说明：
# --local_bs 16: 减小 batch size（32 -> 16），减少 GPU 计算量
# --local_ep 3: 减少本地训练轮数（5 -> 3），加快每轮速度
# 预期每轮时间：~60-80 秒（相比原来的 128 秒）
# 总训练时间：约 1.5-2 小时（vs 原来的 3 小时）
