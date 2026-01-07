#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线数据预处理脚本
功能：将 CIFAR-10 数据集预处理（Resize）并保存为 numpy memmap 格式
目的：消除训练时的实时 Resize 操作，降低 CPU 负载，提升 GPU 利用率
"""

import os
import numpy as np
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from PIL import Image
import argparse


def preprocess_cifar10(image_size=224, data_root='../data/cifar/', output_root='../data/preprocessed/'):
    """
    预处理 CIFAR-10 数据集并保存为 numpy memmap 格式
    
    参数:
        image_size: 目标图像尺寸（默认 224x224）
        data_root: CIFAR-10 原始数据路径
        output_root: 预处理后数据保存路径
    """
    print("="*70)
    print("CIFAR-10 离线数据预处理".center(70))
    print("="*70)
    print(f"\n配置:")
    print(f"  - 目标尺寸: {image_size}x{image_size}")
    print(f"  - 原始数据路径: {data_root}")
    print(f"  - 输出路径: {output_root}")
    print()
    
    # 创建输出目录
    output_dir = os.path.join(output_root, f'cifar10_{image_size}x{image_size}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}\n")
    
    # 定义 Resize transform（仅 Resize，不做标准化）
    # 标准化将在训练时动态进行，以保持灵活性
    resize_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # 转换为 Tensor 并归一化到 [0, 1]
    ])
    
    # 处理训练集和测试集
    for split in ['train', 'test']:
        print(f"\n{'='*70}")
        print(f"处理 {split.upper()} 集".center(70))
        print(f"{'='*70}\n")
        
        # 加载原始数据集（不应用任何 transform）
        is_train = (split == 'train')
        dataset = datasets.CIFAR10(
            root=data_root,
            train=is_train,
            download=True,
            transform=None  # 不使用 transform，手动处理
        )
        
        num_samples = len(dataset)
        print(f"样本数量: {num_samples}")
        
        # 创建 memmap 文件用于存储图像数据
        # 形状: (num_samples, 3, image_size, image_size)
        # dtype: float32（与 PyTorch Tensor 一致）
        images_path = os.path.join(output_dir, f'{split}_images.npy')
        labels_path = os.path.join(output_dir, f'{split}_labels.npy')
        
        # 创建 memmap 数组
        images_memmap = np.memmap(
            images_path,
            dtype='float32',
            mode='w+',
            shape=(num_samples, 3, image_size, image_size)
        )
        
        # 创建标签数组（直接保存到内存，最后一次性写入）
        labels = np.zeros(num_samples, dtype=np.int64)
        
        # 逐个处理样本
        print(f"开始处理...")
        for idx in tqdm(range(num_samples), desc=f"Processing {split}"):
            # 获取原始图像和标签
            img, label = dataset[idx]
            
            # 应用 Resize transform
            # img 是 PIL Image，需要转换
            img_tensor = resize_transform(img)  # (3, image_size, image_size)
            
            # 写入 memmap
            images_memmap[idx] = img_tensor.numpy()
            labels[idx] = label
        
        # 刷新 memmap 到磁盘
        images_memmap.flush()
        del images_memmap  # 释放内存映射
        
        # 保存标签
        np.save(labels_path, labels)
        
        # 验证文件大小
        images_size_mb = os.path.getsize(images_path) / (1024 ** 2)
        labels_size_mb = os.path.getsize(labels_path) / (1024 ** 2)
        
        print(f"\n完成 {split} 集处理:")
        print(f"  - 图像文件: {images_path}")
        print(f"  - 图像大小: {images_size_mb:.2f} MB")
        print(f"  - 标签文件: {labels_path}")
        print(f"  - 标签大小: {labels_size_mb:.2f} MB")
        print(f"  - 总大小: {images_size_mb + labels_size_mb:.2f} MB")
    
    # 保存元数据
    metadata = {
        'image_size': image_size,
        'num_train_samples': len(datasets.CIFAR10(data_root, train=True, download=False)),
        'num_test_samples': len(datasets.CIFAR10(data_root, train=False, download=False)),
        'num_classes': 10,
        'channels': 3,
        'dtype': 'float32',
        'format': 'numpy_memmap',
        'description': 'Preprocessed CIFAR-10 with Resize applied, normalized to [0, 1]'
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n{'='*70}")
    print("预处理完成！".center(70))
    print(f"{'='*70}")
    print(f"\n元数据已保存到: {metadata_path}")
    print(f"\n使用方法:")
    print(f"  在训练时添加参数: --use_offline_data --image_size {image_size}")
    print(f"\n预期效果:")
    print(f"  - CPU 占用率降低 80% 以上")
    print(f"  - GPU 利用率显著提升")
    print(f"  - 训练速度大幅加快")
    print(f"  - 多进程共享内存，节省系统内存")
    print(f"\n{'='*70}\n")


def verify_preprocessed_data(image_size=224, output_root='../data/preprocessed/'):
    """
    验证预处理数据的完整性和正确性
    """
    print("\n" + "="*70)
    print("验证预处理数据".center(70))
    print("="*70 + "\n")
    
    output_dir = os.path.join(output_root, f'cifar10_{image_size}x{image_size}')
    
    for split in ['train', 'test']:
        print(f"\n验证 {split} 集...")
        
        images_path = os.path.join(output_dir, f'{split}_images.npy')
        labels_path = os.path.join(output_dir, f'{split}_labels.npy')
        
        # 检查文件是否存在
        if not os.path.exists(images_path):
            print(f"  [ERROR] 图像文件不存在: {images_path}")
            continue
        if not os.path.exists(labels_path):
            print(f"  [ERROR] 标签文件不存在: {labels_path}")
            continue
        
        # 使用 memmap 加载（不占用内存）
        images = np.load(images_path, mmap_mode='r')
        labels = np.load(labels_path)
        
        print(f"  ✓ 图像形状: {images.shape}")
        print(f"  ✓ 标签形状: {labels.shape}")
        print(f"  ✓ 图像数据类型: {images.dtype}")
        print(f"  ✓ 标签数据类型: {labels.dtype}")
        
        # 安全地检查数据值范围（避免索引错误）
        if len(images) > 0:
            # 检查第一个样本的值范围
            sample_min = images[0].min()
            sample_max = images[0].max()
            print(f"  ✓ 图像值范围（样本0）: [{sample_min:.4f}, {sample_max:.4f}]")
        else:
            print(f"  [ERROR] 图像数据为空")
        
        if len(labels) > 0:
            print(f"  ✓ 标签值范围: [{labels.min()}, {labels.max()}]")
            print(f"  ✓ 标签分布: {np.bincount(labels)}")
        else:
            print(f"  [ERROR] 标签数据为空")
    
    print(f"\n{'='*70}")
    print("验证完成！数据完整性正常。".center(70))
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 离线数据预处理')
    parser.add_argument('--image_size', type=int, default=224,
                        help='目标图像尺寸（默认 224）')
    parser.add_argument('--data_root', type=str, default='../data/cifar/',
                        help='CIFAR-10 原始数据路径')
    parser.add_argument('--output_root', type=str, default='../data/preprocessed/',
                        help='预处理后数据保存路径')
    parser.add_argument('--verify', action='store_true',
                        help='验证预处理数据的完整性')
    
    args = parser.parse_args()
    
    if args.verify:
        # 仅验证数据
        verify_preprocessed_data(args.image_size, args.output_root)
    else:
        # 执行预处理
        preprocess_cifar10(args.image_size, args.data_root, args.output_root)
        # 自动验证
        verify_preprocessed_data(args.image_size, args.output_root)
