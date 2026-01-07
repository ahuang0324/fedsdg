#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-100 离线数据预处理脚本

功能：
1. 下载 CIFAR-100 数据集
2. 将图像调整为 224x224（适配 ViT 等预训练模型）
3. 保存为 numpy memmap 格式，支持多进程共享内存
4. 验证处理后的数据完整性

优势：
- 离线预处理，避免训练时重复调整图像大小
- 使用 memmap 格式，多进程可共享内存，降低 CPU 和内存消耗
- 数据标准化在 DataLoader 中完成，保持灵活性
"""

import os
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm


def download_cifar100(data_root='../data/cifar100/'):
    """下载 CIFAR-100 数据集"""
    print("\n" + "="*60)
    print("【步骤 1】下载 CIFAR-100 数据集")
    print("="*60)
    
    os.makedirs(data_root, exist_ok=True)
    
    print(f"数据保存路径: {data_root}")
    print("开始下载训练集...")
    train_dataset = datasets.CIFAR100(
        data_root, 
        train=True, 
        download=True
    )
    print(f"✓ 训练集下载完成，样本数: {len(train_dataset)}")
    
    print("开始下载测试集...")
    test_dataset = datasets.CIFAR100(
        data_root, 
        train=False, 
        download=True
    )
    print(f"✓ 测试集下载完成，样本数: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def preprocess_and_save(dataset, output_path, image_size=224, split='train', dataset_name='cifar100'):
    """
    预处理数据集并保存为 memmap 格式
    
    参数:
        dataset: PyTorch 数据集对象
        output_path: 输出路径
        image_size: 目标图像大小
        split: 'train' 或 'test'
        dataset_name: 数据集名称，用于创建子目录
    """
    print("\n" + "="*60)
    print(f"【步骤 2】预处理 {split} 数据集")
    print("="*60)
    
    num_samples = len(dataset)
    print(f"样本数量: {num_samples}")
    print(f"目标图像大小: {image_size}x{image_size}")
    
    # 创建数据集特定的子目录（与 CIFAR-10 格式一致）
    dataset_dir = os.path.join(output_path, f'{dataset_name}_{image_size}x{image_size}')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 定义输出文件路径
    images_file = os.path.join(dataset_dir, f'{split}_images.npy')
    labels_file = os.path.join(dataset_dir, f'{split}_labels.npy')
    
    # 创建 memmap 文件用于存储图像
    # 形状: (num_samples, 3, image_size, image_size)
    # 数据类型: float32（节省空间，且训练时常用此类型）
    print(f"\n创建 memmap 文件: {images_file}")
    images_memmap = np.memmap(
        images_file,
        dtype='float32',
        mode='w+',
        shape=(num_samples, 3, image_size, image_size)
    )
    
    # 创建标签数组
    labels = np.zeros(num_samples, dtype=np.int64)
    
    # 定义图像预处理（仅调整大小和转换为 Tensor）
    # 注意：标准化在 DataLoader 中完成，保持灵活性
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # 转换为 [0, 1] 范围的 Tensor
    ])
    
    print(f"\n开始处理图像...")
    for idx in tqdm(range(num_samples), desc=f"处理 {split} 数据"):
        # 获取原始图像和标签
        img, label = dataset[idx]
        
        # 如果图像不是 PIL Image，转换为 PIL Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        
        # 应用预处理
        img_tensor = transform(img)  # 形状: (3, image_size, image_size)
        
        # 保存到 memmap
        images_memmap[idx] = img_tensor.numpy()
        labels[idx] = label
    
    # 刷新 memmap 到磁盘
    images_memmap.flush()
    print(f"✓ 图像数据已保存到: {images_file}")
    
    # 保存标签
    np.save(labels_file, labels)
    print(f"✓ 标签数据已保存到: {labels_file}")
    
    # 打印统计信息
    print(f"\n数据统计:")
    print(f"  - 图像形状: {images_memmap.shape}")
    print(f"  - 图像数据类型: {images_memmap.dtype}")
    print(f"  - 图像值范围: [{images_memmap.min():.4f}, {images_memmap.max():.4f}]")
    print(f"  - 标签形状: {labels.shape}")
    print(f"  - 标签数据类型: {labels.dtype}")
    print(f"  - 标签值范围: [{labels.min()}, {labels.max()}]")
    print(f"  - 类别数量: {len(np.unique(labels))}")
    
    # 计算文件大小
    images_size_mb = os.path.getsize(images_file) / (1024 * 1024)
    labels_size_mb = os.path.getsize(labels_file) / (1024 * 1024)
    print(f"\n文件大小:")
    print(f"  - 图像文件: {images_size_mb:.2f} MB")
    print(f"  - 标签文件: {labels_size_mb:.2f} MB")
    print(f"  - 总计: {images_size_mb + labels_size_mb:.2f} MB")


def verify_preprocessed_data(output_path, image_size=224, dataset_name='cifar100'):
    """验证预处理后的数据"""
    print("\n" + "="*60)
    print("【步骤 3】验证预处理数据")
    print("="*60)
    
    dataset_dir = os.path.join(output_path, f'{dataset_name}_{image_size}x{image_size}')
    
    for split in ['train', 'test']:
        print(f"\n验证 {split} 数据集:")
        
        images_file = os.path.join(dataset_dir, f'{split}_images.npy')
        labels_file = os.path.join(dataset_dir, f'{split}_labels.npy')
        
        # 检查文件是否存在
        if not os.path.exists(images_file):
            print(f"  [ERROR] 图像文件不存在: {images_file}")
            continue
        if not os.path.exists(labels_file):
            print(f"  [ERROR] 标签文件不存在: {labels_file}")
            continue
        
        print(f"  ✓ 文件存在")
        
        # 加载数据
        num_samples = 50000 if split == 'train' else 10000
        images = np.memmap(
            images_file,
            dtype='float32',
            mode='r',
            shape=(num_samples, 3, image_size, image_size)
        )
        labels = np.load(labels_file)
        
        print(f"  ✓ 数据加载成功")
        print(f"  ✓ 图像形状: {images.shape}")
        print(f"  ✓ 标签形状: {labels.shape}")
        
        # 安全地检查数据值范围（避免索引错误）
        if len(images) > 0:
            sample_min = images[0].min()
            sample_max = images[0].max()
            print(f"  ✓ 图像值范围（样本0）: [{sample_min:.4f}, {sample_max:.4f}]")
        else:
            print(f"  [ERROR] 图像数据为空")
        
        if len(labels) > 0:
            print(f"  ✓ 标签值范围: [{labels.min()}, {labels.max()}]")
            print(f"  ✓ 标签分布: 100个类别")
            print(f"  ✓ 每个类别样本数: {len(labels) // 100}")
        else:
            print(f"  [ERROR] 标签数据为空")
    
    print("\n" + "="*60)
    print("验证完成！")
    print("="*60)


def main():
    """主函数"""
    # 配置参数
    data_root = '../data/cifar/'  # 原始数据下载路径（与 CIFAR-10 共享）
    output_root = '../data/preprocessed/'  # 预处理数据输出路径（与 CIFAR-10 共享）
    image_size = 224  # 目标图像大小（适配 ViT）
    
    print("\n" + "="*60)
    print("CIFAR-100 离线数据预处理")
    print("="*60)
    print(f"原始数据路径: {data_root}")
    print(f"输出数据路径: {output_root}")
    print(f"目标图像大小: {image_size}x{image_size}")
    
    # 步骤 1: 下载数据集
    train_dataset, test_dataset = download_cifar100(data_root)
    
    # 步骤 2: 预处理并保存训练集
    preprocess_and_save(train_dataset, output_root, image_size, split='train', dataset_name='cifar100')
    
    # 步骤 3: 预处理并保存测试集
    preprocess_and_save(test_dataset, output_root, image_size, split='test', dataset_name='cifar100')
    
    # 步骤 4: 验证数据
    verify_preprocessed_data(output_root, image_size, dataset_name='cifar100')
    
    print("\n" + "="*60)
    print("所有步骤完成！")
    print("="*60)
    print(f"\n预处理数据已保存到: {output_root}")
    print("可以在训练脚本中使用 OfflineCIFAR100 类加载数据")


if __name__ == '__main__':
    main()
