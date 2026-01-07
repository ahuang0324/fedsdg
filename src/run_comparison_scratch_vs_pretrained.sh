#!/bin/bash
# 对比实验：从零训练 vs 预训练 ViT
# 用于验证预训练模型的性能提升

echo "=========================================="
echo "对比实验：FedLoRA 从零训练 vs 预训练"
echo "=========================================="

# 实验 1：从零训练（Baseline）
echo ""
echo "[实验 1/2] 从零训练 ViT (Scratch)"
echo "预期准确率：15-25%"
echo "------------------------------------------"

python3 federated_main.py \
    --alg fedlora \
    --model vit \
    --model_variant scratch \
    --dataset cifar \
    --image_size 32 \
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
    --dirichlet_alpha 100 \
    --gpu 0 \
    --log_subdir comparison/fedlora_scratch_E80

echo ""
echo "[实验 1/2] 完成"
echo ""
sleep 5

# 实验 2：预训练模型
echo ""
echo "[实验 2/2] 预训练 ViT (Pretrained)"
echo "预期准确率：70-85%"
echo "------------------------------------------"

python3 federated_main.py \
    --alg fedlora \
    --model vit \
    --model_variant pretrained \
    --dataset cifar \
    --image_size 224 \
    --epochs 80 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.0001 \
    --optimizer adam \
    --lora_r 8 \
    --lora_alpha 8 \
    --lora_train_mlp_head 1 \
    --dirichlet_alpha 100 \
    --gpu 0 \
    --log_subdir comparison/fedlora_pretrained_E80

echo ""
echo "[实验 2/2] 完成"
echo ""
echo "=========================================="
echo "对比实验完成！"
echo "=========================================="
echo ""
echo "查看结果："
echo "tensorboard --logdir=../logs/comparison"
echo ""
