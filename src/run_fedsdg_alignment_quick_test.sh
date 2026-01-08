#!/bin/bash
# FedSDG 服务端对齐度加权聚合算法 - 快速功能测试
#
# 本脚本用于快速验证 FedSDG 服务端聚合算法的代码正确性
# 使用最小配置进行测试，仅验证代码能够正常运行

echo "============================================================"
echo "FedSDG 服务端对齐度加权聚合算法 - 快速功能测试"
echo "============================================================"
echo ""

# 最小配置（用于快速验证）
DATASET="cifar"
MODEL="vit"
MODEL_VARIANT="pretrained"  # 使用预训练的 ViT
IMAGE_SIZE=224  # 预训练 ViT 需要 224x224 输入
NUM_USERS=5
FRAC=0.6
LOCAL_EP=1
LOCAL_BS=16
EPOCHS=2  # 仅运行2轮
LORA_R=4
LORA_ALPHA=8
DIRICHLET_ALPHA=0.5

# GPU 配置（使用 GPU 进行测试）
GPU=3

echo "快速测试配置:"
echo "  数据集: $DATASET"
echo "  模型: $MODEL ($MODEL_VARIANT)"
echo "  客户端数量: $NUM_USERS"
echo "  参与率: $FRAC"
echo "  本地轮次: $LOCAL_EP"
echo "  全局轮次: $EPOCHS"
echo "  设备: CPU (GPU=$GPU)"
echo ""

# 测试1: FedSDG + FedAvg 聚合
echo "============================================================"
echo "测试1: FedSDG + FedAvg 均匀加权聚合"
echo "============================================================"
python federated_main.py \
    --alg fedsdg \
    --model $MODEL \
    --model_variant $MODEL_VARIANT \
    --use_offline_data \
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
    --log_subdir fedsdg_quick_test/fedavg

if [ $? -ne 0 ]; then
    echo "❌ 测试1失败！"
    exit 1
fi
echo "✓ 测试1通过"
echo ""

# 测试2: FedSDG + Alignment 对齐度加权聚合
echo "============================================================"
echo "测试2: FedSDG + Alignment 对齐度加权聚合"
echo "============================================================"
python federated_main.py \
    --alg fedsdg \
    --model $MODEL \
    --model_variant $MODEL_VARIANT \
    --use_offline_data \
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
    --log_subdir fedsdg_quick_test/alignment

if [ $? -ne 0 ]; then
    echo "❌ 测试2失败！"
    exit 1
fi
echo "✓ 测试2通过"
echo ""

echo "============================================================"
echo "✓ 所有快速测试通过！"
echo "============================================================"
echo ""
echo "FedSDG 服务端聚合算法功能验证成功："
echo "  1. fedavg 模式：传统的 FedAvg 均匀加权聚合"
echo "  2. alignment 模式：基于对齐度加权的 FedSDG 聚合算法"
echo ""
echo "结果保存位置:"
echo "  - TensorBoard 日志: ../logs/fedsdg_quick_test/"
echo "  - 实验报告: ../save/summaries/"
