#!/bin/bash
# FedSDG 正式训练脚本 - CIFAR-10
# 使用手写 ViT 模型，从零训练

echo "=========================================="
echo "FedSDG 训练 - CIFAR-10 (从零训练)"
echo "=========================================="
echo ""
echo "训练配置："
echo "  - 算法: FedSDG (双路架构 + 门控机制)"
echo "  - 模型: ViT (手写版本，从零训练)"
echo "  - 数据集: CIFAR-10"
echo "  - 训练轮次: 50"
echo "  - 客户端数量: 100"
echo "  - 参与率: 10%"
echo "  - LoRA 秩: 8"
echo "  - Dirichlet Alpha: 0.1 (强 Non-IID)"
echo ""
echo "FedSDG 特点："
echo "  - 全局分支 (lora_A, lora_B): 参与服务器聚合"
echo "  - 私有分支 (lora_A_private, lora_B_private): 仅本地更新"
echo "  - 门控参数 (lambda_k): 自动学习全局/私有权重"
echo "  - 通信量与 FedLoRA 一致 (~0.2MB/轮)"
echo ""
echo "=========================================="
echo ""

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
    --log_subdir fedsdg_cifar10_E50_alpha0.1

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "结果文件位置："
echo "  - TensorBoard 日志: logs/fedsdg_cifar10_E50_alpha0.1/"
echo "  - 实验总结: save/summaries/cifar_vit_fedsdg_E50_summary.txt"
echo "  - 最终模型: save/models/cifar_vit_final.pth"
echo ""
echo "查看 TensorBoard："
echo "  tensorboard --logdir=../logs/fedsdg_cifar10_E50_alpha0.1"
echo ""
