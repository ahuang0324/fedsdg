#!/bin/bash
# FedSDG 正式训练脚本 - CIFAR-100
# 使用手写 ViT 模型，从零训练

# ==================== 参数配置 ====================
ALG="fedsdg"
MODEL="vit"
MODEL_VARIANT="scratch"
DATASET="cifar100"
NUM_CLASSES=100
EPOCHS=80
NUM_USERS=100
FRAC=0.1
LOCAL_EP=5
LOCAL_BS=128
LR=0.001
LORA_R=8
LORA_ALPHA=16
DIRICHLET_ALPHA=0.5
GPU=0
LOG_SUBDIR="fedsdg_cifar100_E${EPOCHS}_alpha${DIRICHLET_ALPHA}_bs${LOCAL_BS}_lr${LR}"
# =================================================

echo "=========================================="
echo "FedSDG 训练 - CIFAR-100 (从零训练)"
echo "=========================================="
echo ""
echo "训练配置："
echo "  - 算法: ${ALG} (双路架构 + 门控机制)"
echo "  - 模型: ViT (手写版本，从零训练)"
echo "  - 数据集: ${DATASET}"
echo "  - 训练轮次: ${EPOCHS}"
echo "  - 客户端数量: ${NUM_USERS}"
echo "  - 参与率: $(echo "scale=0; ${FRAC} * 100" | bc)%"
echo "  - 本地 Epoch: ${LOCAL_EP}"
echo "  - 本地 Batch Size: ${LOCAL_BS}"
echo "  - LoRA 秩: ${LORA_R}"
echo "  - LoRA Alpha: ${LORA_ALPHA}"
echo "  - 学习率: ${LR}"
echo "  - Dirichlet Alpha: ${DIRICHLET_ALPHA}"
echo "  - GPU: ${GPU}"
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
    --alg ${ALG} \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --num_classes ${NUM_CLASSES} \
    --epochs ${EPOCHS} \
    --num_users ${NUM_USERS} \
    --frac ${FRAC} \
    --local_ep ${LOCAL_EP} \
    --local_bs ${LOCAL_BS} \
    --lr ${LR} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha ${DIRICHLET_ALPHA} \
    --gpu ${GPU} \
    --log_subdir ${LOG_SUBDIR}

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "结果文件位置："
echo "  - TensorBoard 日志: ../logs/${LOG_SUBDIR}/"
echo "  - 实验总结: ../save/summaries/${DATASET}_${MODEL}_${ALG}_E${EPOCHS}_summary.txt"
echo "  - 最终模型: ../save/models/${DATASET}_${MODEL}_final.pth"
echo ""
echo "查看 TensorBoard："
echo "  tensorboard --logdir=../logs/${LOG_SUBDIR}"
echo ""
