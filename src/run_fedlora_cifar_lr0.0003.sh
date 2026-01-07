#!/bin/bash
# FedLoRA 修复版本 - 使用更小的学习率
# 修复了 LoRA 初始化问题，并降低学习率以避免训练崩溃

# 每次需要修改 gpu 日志输出目录 
python3 federated_main.py \
    --alg fedlora \
    --model vit \
    --dataset cifar \
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
    --gpu 1 \
    --log_subdir fedlora_vit_cifar_E80_lr0.0003_fixed
