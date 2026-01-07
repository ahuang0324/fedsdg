#!/bin/bash
# FedSDG 测试脚本
# 使用 CIFAR-10 数据集进行快速测试（5轮训练）

echo "=========================================="
echo "FedSDG 功能测试"
echo "=========================================="
echo ""
echo "测试配置："
echo "  - 算法: FedSDG"
echo "  - 模型: ViT (手写版本)"
echo "  - 数据集: CIFAR-10"
echo "  - 训练轮次: 5"
echo "  - LoRA 秩: 8"
echo "  - Alpha: 0.1 (强 Non-IID)"
echo ""
echo "预期结果："
echo "  - 通信量应与 FedLoRA 一致 (~0.2MB/轮)"
echo "  - 私有参数不参与服务器聚合"
echo "  - 每个客户端维护独立的私有状态"
echo ""
echo "=========================================="
echo ""

python3 federated_main.py \
    --alg fedsdg \
    --model vit \
    --dataset cifar \
    --num_classes 10 \
    --epochs 5 \
    --num_users 10 \
    --frac 0.3 \
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.001 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 0.1 \
    --gpu -1 \
    --log_subdir fedsdg_test_cifar10_alpha0.1

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "请检查以下内容："
echo "  1. 通信量统计是否显示与 FedLoRA 一致"
echo "  2. 是否显示 'FedSDG 客户端私有状态管理已初始化'"
echo "  3. 训练是否正常进行"
echo "  4. TensorBoard 日志是否正确生成"
echo ""
