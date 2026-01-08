#!/bin/bash
# FedSDG 服务端对齐度加权聚合算法 - 端到端功能测试
#
# 本脚本用于测试 FedSDG 服务端聚合算法的两种模式：
# 1. fedavg: 传统的 FedAvg 均匀加权聚合
# 2. alignment: 基于对齐度加权的 FedSDG 聚合算法
#
# 测试场景：CIFAR-10 数据集，强 Non-IID 分布 (alpha=0.1)
# 预期结果：在强 Non-IID 场景下，alignment 聚合应该比 fedavg 表现更好

echo "============================================================"
echo "FedSDG 服务端对齐度加权聚合算法 - 端到端功能测试"
echo "============================================================"
echo ""

# 基础配置
DATASET="cifar"
MODEL="vit"
MODEL_VARIANT="pretrained"
IMAGE_SIZE=224
NUM_USERS=10
FRAC=0.5
LOCAL_EP=2
LOCAL_BS=32
EPOCHS=5  # 快速测试，仅运行5轮
LORA_R=8
LORA_ALPHA=16
DIRICHLET_ALPHA=0.1  # 强 Non-IID

# GPU 配置
GPU=3

echo "测试配置:"
echo "  数据集: $DATASET"
echo "  模型: $MODEL ($MODEL_VARIANT)"
echo "  客户端数量: $NUM_USERS"
echo "  参与率: $FRAC"
echo "  本地轮次: $LOCAL_EP"
echo "  全局轮次: $EPOCHS"
echo "  Dirichlet Alpha: $DIRICHLET_ALPHA (强 Non-IID)"
echo ""

# 测试1: FedSDG + FedAvg 聚合
echo "============================================================"
echo "测试1: FedSDG + FedAvg 均匀加权聚合"
echo "============================================================"
python federated_main.py \
    --alg fedsdg \
    --model $MODEL \
    --model_variant $MODEL_VARIANT \
    --image_size $IMAGE_SIZE \
    --dataset $DATASET \
    --num_users $NUM_USERS \
    --frac $FRAC \
    --local_ep $LOCAL_EP \
    --local_bs $LOCAL_BS \
    --epochs $EPOCHS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --dirichlet_alpha $DIRICHLET_ALPHA \
    --server_agg_method fedavg \
    --gpu $GPU \
    --log_subdir fedsdg_alignment_test/fedavg

echo ""
echo "============================================================"
echo "测试2: FedSDG + Alignment 对齐度加权聚合"
echo "============================================================"
python federated_main.py \
    --alg fedsdg \
    --model $MODEL \
    --model_variant $MODEL_VARIANT \
    --image_size $IMAGE_SIZE \
    --dataset $DATASET \
    --num_users $NUM_USERS \
    --frac $FRAC \
    --local_ep $LOCAL_EP \
    --local_bs $LOCAL_BS \
    --epochs $EPOCHS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --dirichlet_alpha $DIRICHLET_ALPHA \
    --server_agg_method alignment \
    --gpu $GPU \
    --log_subdir fedsdg_alignment_test/alignment

echo ""
echo "============================================================"
echo "测试完成！"
echo "============================================================"
echo ""
echo "结果保存位置:"
echo "  - TensorBoard 日志: ../logs/fedsdg_alignment_test/"
echo "  - 实验报告: ../save/summaries/"
echo ""
echo "查看 TensorBoard:"
echo "  tensorboard --logdir ../logs/fedsdg_alignment_test/"
