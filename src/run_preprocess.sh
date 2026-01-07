#!/bin/bash
# 离线数据预处理脚本
# 功能：预处理 CIFAR-10 数据集，消除训练时的实时 Resize 操作

echo "========================================================================"
echo "                    CIFAR-10 离线数据预处理                              "
echo "========================================================================"
echo ""
echo "此脚本将预处理 CIFAR-10 数据集并保存为 numpy memmap 格式"
echo "预期效果："
echo "  - CPU 占用率降低 80% 以上"
echo "  - GPU 利用率显著提升"
echo "  - 训练速度大幅加快"
echo "  - 多进程共享内存，节省系统内存"
echo ""
echo "========================================================================"
echo ""

# 预处理 224x224 数据（用于预训练模型）
echo "步骤 1/2: 预处理 224x224 数据（预训练模型）..."
python3 preprocess_data.py --image_size 224

echo ""
echo "========================================================================"
echo ""

# 预处理 128x128 数据（可选，用于更快的训练）
echo "步骤 2/2: 预处理 128x128 数据（可选）..."
read -p "是否预处理 128x128 数据？(y/n): " choice
if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    python3 preprocess_data.py --image_size 128
else
    echo "跳过 128x128 数据预处理"
fi

echo ""
echo "========================================================================"
echo "                           预处理完成！                                  "
echo "========================================================================"
echo ""
echo "使用方法："
echo "  在训练脚本中添加参数: --use_offline_data"
echo ""
echo "示例："
echo "  python3 federated_main.py \\"
echo "    --alg fedlora \\"
echo "    --model vit \\"
echo "    --model_variant pretrained \\"
echo "    --dataset cifar \\"
echo "    --image_size 224 \\"
echo "    --use_offline_data \\"
echo "    --epochs 80 \\"
echo "    --num_users 100 \\"
echo "    --frac 0.1 \\"
echo "    --local_ep 5 \\"
echo "    --local_bs 32 \\"
echo "    --lr 0.0001"
echo ""
echo "========================================================================"
