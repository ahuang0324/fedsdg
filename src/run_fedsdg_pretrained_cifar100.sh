#!/bin/bash
# FedSDG 正式训练脚本 - CIFAR-100 (预训练模型 + 离线数据)
# 使用 timm 预训练 ViT 模型，配合离线预处理的 224x224 数据

echo "=========================================="
echo "FedSDG 训练 - CIFAR-100 (预训练 + 离线数据)"
echo "=========================================="
echo ""
echo "训练配置："
echo "  - 算法: FedSDG (双路架构 + 门控机制)"
echo "  - 模型: ViT-Tiny (timm 预训练，ImageNet-21k)"
echo "  - 数据集: CIFAR-100 (离线预处理 224x224)"
echo "  - 训练轮次: 50"
echo "  - 客户端数量: 100"
echo "  - 参与率: 10%"
echo "  - LoRA 秩: 8"
echo "  - 学习率: 0.0005 (预训练模型用较小学习率)"
echo "  - Dirichlet Alpha: 0.5 (中等 Non-IID)"
echo ""
echo "FedSDG 特点："
echo "  - 全局分支 (lora_A, lora_B): 参与服务器聚合"
echo "  - 私有分支 (lora_A_private, lora_B_private): 仅本地更新"
echo "  - 门控参数 (lambda_k): 自动学习全局/私有权重"
echo "  - 通信量与 FedLoRA 一致 (~0.2MB/轮)"
echo ""
echo "数据要求："
echo "  - 离线数据路径: ../data/preprocessed/cifar100_224x224/"
echo "  - 需要包含: train_images.npy, train_labels.npy"
echo "  - 需要包含: test_images.npy, test_labels.npy"
echo ""
echo "=========================================="
echo ""

# 检查离线数据是否存在
if [ ! -f "../data/preprocessed/cifar100_224x224/train_images.npy" ]; then
    echo "❌ 错误: 离线数据不存在！"
    echo "请先运行数据预处理脚本："
    echo "  python3 preprocess_cifar100.py"
    exit 1
fi

echo "✓ 离线数据检查通过"
echo ""

python3 federated_main.py \
    --alg fedsdg \
    --model vit \
    --model_variant pretrained \
    --dataset cifar100 \
    --num_classes 100 \
    --image_size 224 \
    --use_offline_data \
    --offline_data_root ../data/preprocessed/ \
    --epochs 50 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 5 \
    --local_bs 16 \
    --lr 0.0005 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 0.5 \
    --gpu 1 \
    --log_subdir fedsdg_pretrained_vit_cifar100_E50_lr0.0005_alpha0.5_fix

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "结果文件位置："
echo "  - TensorBoard 日志: logs/fedsdg_pretrained_vit_cifar100_E50_lr0.0005_alpha0.5_fix/"
echo "  - 实验总结: save/summaries/cifar100_vit_pretrained_fedsdg_E50_summary.txt"
echo "  - 最终模型: save/models/cifar100_vit_pretrained_final.pth"
echo ""
echo "查看 TensorBoard："
echo "  tensorboard --logdir=../logs/fedsdg_pretrained_vit_cifar100_E50_lr0.0005_alpha0.5_fix"
echo ""
echo "与 FedLoRA 对比："
echo "  - FedSDG 通过私有分支学习客户端特定模式"
echo "  - 预期在强 Non-IID 场景下性能优于 FedLoRA"
echo "  - 通信量保持一致 (~0.2MB/轮)"
echo ""
