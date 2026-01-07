#!/bin/bash
# FedLoRA with Pretrained ViT - 使用 128x128 图像（速度与性能的折中）

export HF_ENDPOINT=https://hf-mirror.com

python3 federated_main.py \
    --alg fedlora \
    --model vit \
    --model_variant pretrained \
    --dataset cifar \
    --image_size 128 \
    --epochs 80 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.0001 \
    --optimizer adam \
    --lora_r 8 \
    --lora_alpha 8 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 100 \
    --gpu 0 \
    --log_subdir fedlora_pretrained_vit_cifar_E80_img128

# 优化说明：
# --image_size 128: 使用 128x128 而非 224x224
# 计算量减少约 3 倍（128^2 vs 224^2）
# 预期每轮时间：~40-50 秒
# 准确率可能略有下降（预期 65-80% vs 70-85%）
