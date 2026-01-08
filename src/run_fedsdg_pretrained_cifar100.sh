#!/bin/bash
# FedSDG 正式训练脚本 - CIFAR-100 (预训练模型 + 离线数据)
# 使用 timm 预训练 ViT 模型，配合离线预处理的 224x224 数据

# ==================== 参数配置 ====================
ALG="fedsdg"
MODEL="vit"
MODEL_VARIANT="pretrained"
DATASET="cifar100"
NUM_CLASSES=100
IMAGE_SIZE=224
EPOCHS=100
NUM_USERS=100
FRAC=0.1
LOCAL_EP=5
LOCAL_BS=128
LR=0.001
LORA_R=8
LORA_ALPHA=16
DIRICHLET_ALPHA=0.1
AGG_METHOD=alignment
GPU=3


# ========== FedSDG 专用参数（根据 proposal 设计）==========
LR_GATE=0.01           # ηm: 门控参数学习率
LAMBDA1=0.01          # λ₁: L1 门控稀疏性惩罚 0.001 好像有点小 尝试一下0.01
LAMBDA2=0.0001         # λ₂: L2 私有参数正则化
GRAD_CLIP=1.0          # 梯度裁剪范数
GATE_PENALTY_TYPE="unilateral"  # 门控惩罚类型: unilateral(单边,推向0) 或 bilateral(双边,推向0或1)
# =========================================================

LOG_SUBDIR="${ALG}_${MODEL_VARIANT}_${MODEL}_${DATASET}_E${EPOCHS}_lr${LR}_lrgate${LR_GATE}_alpha${DIRICHLET_ALPHA}_bs${LOCAL_BS}_l1${LAMBDA1}_gate${GATE_PENALTY_TYPE}_${AGG_METHOD}"
OFFLINE_DATA_ROOT="../data/preprocessed/"
# =================================================

echo "=========================================="
echo "FedSDG 训练 - CIFAR-100 (预训练 + 离线数据)"
echo "=========================================="
echo ""
echo "训练配置："
echo "  - 算法: ${ALG} (双路架构 + 门控机制)"
echo "  - 模型: ViT-Tiny (timm 预训练，ImageNet-21k)"
echo "  - 数据集: ${DATASET} (离线预处理 ${IMAGE_SIZE}x${IMAGE_SIZE})"
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
echo "FedSDG 专用参数（根据 proposal 设计）："
echo "  - 门控学习率 (ηm): ${LR_GATE}"
echo "  - L1 门控惩罚 (λ₁): ${LAMBDA1}"
echo "  - L2 私有惩罚 (λ₂): ${LAMBDA2}"
echo "  - 门控惩罚类型: ${GATE_PENALTY_TYPE}"
echo "  - 梯度裁剪范数: ${GRAD_CLIP}"
echo ""
echo "FedSDG 特点："
echo "  - 全局分支 (lora_A, lora_B): 参与服务器聚合，学习率 ${LR}"
echo "  - 私有分支 (lora_A_private, lora_B_private): 仅本地更新，学习率 ${LR}"
echo "  - 门控参数 (lambda_k): 自动学习全局/私有权重，学习率 ${LR_GATE}"
echo "  - 通信量与 FedLoRA 一致 (~0.2MB/轮)"
echo ""
echo "数据要求："
echo "  - 离线数据路径: ${OFFLINE_DATA_ROOT}cifar100_${IMAGE_SIZE}x${IMAGE_SIZE}/"
echo "  - 需要包含: train_images.npy, train_labels.npy"
echo "  - 需要包含: test_images.npy, test_labels.npy"
echo ""
echo "=========================================="
echo ""

# 检查离线数据是否存在
if [ ! -f "${OFFLINE_DATA_ROOT}cifar100_${IMAGE_SIZE}x${IMAGE_SIZE}/train_images.npy" ]; then
    echo "❌ 错误: 离线数据不存在！"
    echo "请先运行数据预处理脚本："
    echo "  python3 preprocess_cifar100.py"
    exit 1
fi

echo "✓ 离线数据检查通过"
echo ""

python3 federated_main.py \
    --alg ${ALG} \
    --model ${MODEL} \
    --model_variant ${MODEL_VARIANT} \
    --dataset ${DATASET} \
    --num_classes ${NUM_CLASSES} \
    --image_size ${IMAGE_SIZE} \
    --use_offline_data \
    --offline_data_root ${OFFLINE_DATA_ROOT} \
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
    --lr_gate ${LR_GATE} \
    --lambda1 ${LAMBDA1} \
    --lambda2 ${LAMBDA2} \
    --gate_penalty_type ${GATE_PENALTY_TYPE} \
    --server_agg_method ${AGG_METHOD} \
    --grad_clip ${GRAD_CLIP} \
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
echo "与 FedLoRA 对比："
echo "  - FedSDG 通过私有分支学习客户端特定模式"
echo "  - 预期在强 Non-IID 场景下性能优于 FedLoRA"
echo "  - 通信量保持一致 (~0.2MB/轮)"
echo ""
