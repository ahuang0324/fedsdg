#!/bin/bash
# FedAvg with Pretrained ViT (timm) - 使用离线预处理数据
# 使用预训练 ViT 模型进行全量参数联邦学习（对比 FedLoRA）
# 关键优化：使用离线预处理数据，消除实时 Resize，降低 CPU 负载

# 关键配置说明：
# --alg fedavg: 使用 FedAvg 算法（全量参数训练）
# --model_variant pretrained: 使用预训练模型
# --image_size 224: 预训练模型需要 224x224 输入
# --use_offline_data: 使用离线预处理数据（关键优化！）
# --lr 0.0001: 预训练模型微调通常需要更小的学习率

# 注意：使用此脚本前，请先运行预处理脚本：
#   bash run_preprocess.sh
# 或手动运行：
#   python3 preprocess_data.py --image_size 224

export HF_ENDPOINT=https://hf-mirror.com

python3 federated_main.py \
    --alg fedavg \
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
    --lr 0.0001 \
    --optimizer adam \
    --dirichlet_alpha 0.5 \
    --gpu 2 \
    --log_subdir fedavg_pretrained_vit_cifar_E80_lr0.0001_offline_alpha0.5

# 预期效果：
# - 训练所有参数（5.7M），而非仅 LoRA 参数（~200K）
# - 通信开销更大（传输完整模型参数）
# - 准确率可能略高于 FedLoRA（75-90%）
# - CPU 占用率降低 80% 以上（相比实时 Resize）
# - GPU 利用率显著提升（数据加载不再是瓶颈）
# - 每轮时间大幅缩短（~30-50秒，相比原来的 150-180秒）
#
# 对比目的：
# - 验证离线预处理的性能提升
# - 对比 FedLoRA vs FedAvg 的准确率差异
# - 对比通信开销（200K vs 5.7M 参数）
