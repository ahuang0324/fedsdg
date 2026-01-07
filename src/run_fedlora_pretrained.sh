#!/bin/bash
# FedLoRA with Pretrained ViT (timm)
# 使用预训练 ViT 模型进行联邦学习，提升基准性能

# 关键配置说明：
# --model_variant pretrained: 使用预训练模型（而非从零训练）
# --image_size 224: 预训练模型需要 224x224 输入（CIFAR-10 会自动 resize）
# --lr 0.0001: 预训练模型微调通常需要更小的学习率
# --lora_r 8, --lora_alpha 8: 较小的缩放因子，避免破坏预训练权重

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
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.0001 \
    --optimizer adam \
    --lora_r 8 \
    --lora_alpha 8 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 100 \
    --gpu 0 \
    --log_subdir fedlora_pretrained_vit_cifar_E80_lr0.0001

# 预期效果：
# - 初始准确率应该在 40-60%（预训练权重的迁移能力）
# - 最终准确率应该达到 70-85%（远高于从零训练的 20%）
# - 训练更稳定，Loss 不会发散
