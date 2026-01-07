#!/bin/bash
# FedSDG 对比实验脚本
# 在不同 Non-IID 程度下对比 FedSDG 与 FedLoRA 的性能

echo "=========================================="
echo "FedSDG vs FedLoRA 对比实验"
echo "=========================================="
echo ""
echo "实验设计："
echo "  - 数据集: CIFAR-10"
echo "  - 模型: ViT (手写版本)"
echo "  - 对比算法: FedSDG vs FedLoRA"
echo "  - Non-IID 程度: α ∈ {0.1, 0.5, 1.0}"
echo "  - 训练轮次: 50"
echo ""
echo "预期结果："
echo "  - α=0.1 (强 Non-IID): FedSDG 应显著优于 FedLoRA"
echo "  - α=0.5 (中等 Non-IID): FedSDG 应略优于 FedLoRA"
echo "  - α=1.0 (弱 Non-IID): FedSDG 与 FedLoRA 性能接近"
echo ""
echo "=========================================="
echo ""

# 实验 1: α=0.1 (强 Non-IID)
echo "【实验 1/3】α=0.1 (强 Non-IID) - FedSDG"
echo "=========================================="
python3 federated_main.py \
    --alg fedsdg \
    --model vit \
    --dataset cifar \
    --num_classes 10 \
    --epochs 50 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.001 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 0.1 \
    --gpu 0 \
    --log_subdir comparison/fedsdg_cifar10_alpha0.1

echo ""
echo "【实验 1/3】α=0.1 (强 Non-IID) - FedLoRA"
echo "=========================================="
python3 federated_main.py \
    --alg fedlora \
    --model vit \
    --dataset cifar \
    --num_classes 10 \
    --epochs 50 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.001 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 0.1 \
    --gpu 0 \
    --log_subdir comparison/fedlora_cifar10_alpha0.1

echo ""
echo "=========================================="
echo "实验 1/3 完成"
echo "=========================================="
echo ""

# 实验 2: α=0.5 (中等 Non-IID)
echo "【实验 2/3】α=0.5 (中等 Non-IID) - FedSDG"
echo "=========================================="
python3 federated_main.py \
    --alg fedsdg \
    --model vit \
    --dataset cifar \
    --num_classes 10 \
    --epochs 50 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.001 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 0.5 \
    --gpu 0 \
    --log_subdir comparison/fedsdg_cifar10_alpha0.5

echo ""
echo "【实验 2/3】α=0.5 (中等 Non-IID) - FedLoRA"
echo "=========================================="
python3 federated_main.py \
    --alg fedlora \
    --model vit \
    --dataset cifar \
    --num_classes 10 \
    --epochs 50 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.001 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 0.5 \
    --gpu 0 \
    --log_subdir comparison/fedlora_cifar10_alpha0.5

echo ""
echo "=========================================="
echo "实验 2/3 完成"
echo "=========================================="
echo ""

# 实验 3: α=1.0 (弱 Non-IID)
echo "【实验 3/3】α=1.0 (弱 Non-IID) - FedSDG"
echo "=========================================="
python3 federated_main.py \
    --alg fedsdg \
    --model vit \
    --dataset cifar \
    --num_classes 10 \
    --epochs 50 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.001 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 1.0 \
    --gpu 0 \
    --log_subdir comparison/fedsdg_cifar10_alpha1.0

echo ""
echo "【实验 3/3】α=1.0 (弱 Non-IID) - FedLoRA"
echo "=========================================="
python3 federated_main.py \
    --alg fedlora \
    --model vit \
    --dataset cifar \
    --num_classes 10 \
    --epochs 50 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.001 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 1.0 \
    --gpu 0 \
    --log_subdir comparison/fedlora_cifar10_alpha1.0

echo ""
echo "=========================================="
echo "所有对比实验完成！"
echo "=========================================="
echo ""
echo "结果分析："
echo "  1. 查看 TensorBoard 对比曲线："
echo "     tensorboard --logdir=../logs/comparison/"
echo ""
echo "  2. 对比指标："
echo "     - 测试准确率 (Test Accuracy)"
echo "     - 训练损失 (Train Loss)"
echo "     - 收敛速度"
echo ""
echo "  3. 预期发现："
echo "     - 强 Non-IID (α=0.1): FedSDG 准确率 > FedLoRA"
echo "     - 中等 Non-IID (α=0.5): FedSDG 准确率 ≥ FedLoRA"
echo "     - 弱 Non-IID (α=1.0): FedSDG 准确率 ≈ FedLoRA"
echo ""
echo "  4. 通信量验证："
echo "     - FedSDG 和 FedLoRA 的通信量应完全一致"
echo "     - 查看实验总结文件中的通信量统计"
echo ""
