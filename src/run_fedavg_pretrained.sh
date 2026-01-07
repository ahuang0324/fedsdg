#!/bin/bash
# FedAvg with Pretrained ViT (timm)
# 使用预训练 ViT 模型进行全量参数联邦学习（对比 FedLoRA）

# 关键配置说明：
# --alg fedavg: 使用 FedAvg 算法（全量参数训练）
# --model_variant pretrained: 使用预训练模型
# --image_size 224: 预训练模型需要 224x224 输入
# --lr 0.0001: 预训练模型微调通常需要更小的学习率
# 注意：FedAvg 会训练所有参数，通信开销和计算量都比 FedLoRA 大

export HF_ENDPOINT=https://hf-mirror.com

python3 federated_main.py \
    --alg fedavg \
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
    --dirichlet_alpha 100 \
    --gpu 2 \
    --log_subdir fedavg_pretrained_vit_cifar_E80_lr0.0001

# 预期效果：
# - 训练所有参数（5.7M），而非仅 LoRA 参数（~200K）
# - 通信开销更大（传输完整模型参数）
# - 准确率可能略高于 FedLoRA（75-90%）
# - 每轮时间可能更长（~150-180秒，因为梯度计算量更大）
#
# 对比目的：
# - 验证 FedLoRA 的参数效率优势
# - 对比准确率差异（FedLoRA vs FedAvg）
# - 对比通信开销（200K vs 5.7M 参数）
