#!/bin/bash
# FedLoRA with Pretrained ViT - 最优配置（平衡速度与性能）
# 综合优化：减小 batch size + 减少 local epochs

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
    --local_bs 20 \
    --lr 0.00015 \
    --optimizer adam \
    --lora_r 8 \
    --lora_alpha 8 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 100 \
    --gpu 0 \
    --log_subdir fedlora_pretrained_vit_cifar_E80_optimal

# 最优配置说明：
# --local_bs 20: 适中的 batch size（平衡速度和收敛性）
# --local_ep 3: 减少本地训练轮数（5 -> 3）
# --lr 0.00015: 略微提高学习率以补偿训练轮数减少
#
# 预期效果：
# - 每轮时间：~70-90 秒（比原来快 30-40%）
# - 总训练时间：约 1.5-2 小时
# - 准确率：预期仍能达到 70-80%
