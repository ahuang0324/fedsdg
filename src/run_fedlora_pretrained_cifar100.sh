#!/bin/bash
# FedLoRA with Pretrained ViT (timm) - CIFAR-100 使用离线预处理数据
# 使用预训练 ViT + LoRA 进行参数高效的联邦学习
# 关键优化：使用离线预处理数据，消除实时 Resize，降低 CPU 负载

# 关键配置说明：
# --alg fedlora: 使用 FedLoRA 算法（仅训练 LoRA 参数）
# --model_variant pretrained: 使用预训练模型
# --dataset cifar100: 使用 CIFAR-100 数据集（100个类别）
# --image_size 224: 预训练模型需要 224x224 输入
# --use_offline_data: 使用离线预处理数据（关键优化！）
# --offline_data_root: CIFAR-100 预处理数据路径
# --lora_r 8: LoRA 秩（控制参数量和表达能力）
# --lora_alpha 16: LoRA 缩放因子
# --lr 0.0003: LoRA 通常可以使用更大的学习率

# 注意：使用此脚本前，请先运行预处理脚本：
#   python3 preprocess_cifar100.py

export HF_ENDPOINT=https://hf-mirror.com

python3 federated_main.py \
    --alg fedlora \
    --model vit \
    --model_variant pretrained \
    --dataset cifar100 \
    --image_size 224 \
    --use_offline_data \
    --offline_data_root ../data/preprocessed/ \
    --epochs 50 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.0003 \
    --optimizer adam \
    --lora_r 16 \
    --lora_alpha 16 \
    --dirichlet_alpha 0.5 \
    --gpu 2 \
    --log_subdir fedlora_pretrained_vit_cifar100_E50_r16_lr0.0003_offline_alpha0.5

# 预期效果：
# - CIFAR-100 更具挑战性（100个类别 vs CIFAR-10的10个类别）
# - 仅训练 LoRA 参数（~200K），大幅减少通信开销
# - 通信效率高（仅传输 LoRA 参数，约为全量参数的 3.5%）
# - 准确率预期：55-70%（略低于 FedAvg，但通信开销大幅降低）
# - CPU 占用率降低 80% 以上（相比实时 Resize）
# - GPU 利用率显著提升（数据加载不再是瓶颈）
# - 每轮时间大幅缩短（~25-40秒，相比原来的 150-180秒）
#
# LoRA 优势：
# - 参数高效：仅训练 3.5% 的参数
# - 通信高效：减少 96.5% 的通信开销
# - 内存高效：客户端内存占用更小
# - 适合资源受限的联邦学习场景
#
# 对比目的：
# - 验证 LoRA 在 CIFAR-100 上的性能
# - 对比 FedLoRA vs FedAvg 的准确率-通信开销权衡
# - 评估更复杂数据集上的参数高效学习能力
