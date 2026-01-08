#!/bin/bash
# FedLoRA with Pretrained ViT (timm) - 使用离线预处理数据
# 使用预训练 ViT + LoRA 进行参数高效的联邦学习
# 关键优化：使用离线预处理数据，消除实时 Resize，降低 CPU 负载

# 关键配置说明：
# --alg fedlora: 使用 FedLoRA 算法（仅训练 LoRA 参数）
# --model_variant pretrained: 使用预训练模型
# --image_size 224: 预训练模型需要 224x224 输入
# --use_offline_data: 使用离线预处理数据（关键优化！）
# --lora_r 8: LoRA 秩（低秩维度）
# --lora_alpha 16: LoRA 缩放参数
# --lr 0.0003: FedLoRA 通常需要较大的学习率

# 注意：使用此脚本前，请先运行预处理脚本：
#   bash run_preprocess.sh
# 或手动运行：
#   python3 preprocess_data.py --image_size 224

export HF_ENDPOINT=https://hf-mirror.com

python federated_main.py \
    --alg fedlora \
    --model vit \
    --model_variant pretrained \
    --dataset cifar \
    --image_size 224 \
    --use_offline_data \
    --epochs 80 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.0003 \
    --optimizer adam \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 0.5 \
    --gpu 2 \
    --log_subdir fedlora_pretrained_vit_cifar_E80_lr0.0003_offline_alpha0.5

# 预期效果：
# - 仅训练 LoRA 参数（~200K），通信开销小
# - 准确率目标：75-90%（与 FedAvg 相当）
# - CPU 占用率降低 80% 以上（相比实时 Resize）
# - GPU 利用率显著提升（数据加载不再是瓶颈）
# - 每轮时间大幅缩短（~20-30秒，相比原来的 100-120秒）
# - 参数效率：仅 3.5% 的参数量（200K vs 5.7M）
#
# 优势：
# - 通信开销小（仅传输 LoRA 参数）
# - 训练速度快（参数量少 + 离线预处理）
# - 内存占用低（多进程共享内存映射数据）
