#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线数据集类
功能：使用内存映射加载预处理好的 CIFAR-10/CIFAR-100 数据
优势：
  1. 零 CPU 开销的 Resize（已离线完成）
  2. 多进程共享内存（mmap_mode='r'）
  3. 快速数据加载，提升 GPU 利用率
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class OfflineCIFAR10(Dataset):
    """
    离线预处理的 CIFAR-10 数据集
    
    使用 numpy memmap 加载预处理好的数据，避免实时 Resize 操作
    支持多进程共享内存，大幅降低 CPU 负载和内存占用
    
    参数:
        root: 预处理数据的根目录（例如 '../data/preprocessed/'）
        train: True 表示训练集，False 表示测试集
        image_size: 图像尺寸（必须与预处理时一致）
        transform: 额外的数据增强（可选，通常仅需要 Normalize）
        use_imagenet_norm: 是否使用 ImageNet 标准化（预训练模型需要）
    """
    
    def __init__(self, root='../data/preprocessed/', train=True, image_size=224,
                 transform=None, use_imagenet_norm=True):
        self.root = root
        self.train = train
        self.image_size = image_size
        self.use_imagenet_norm = use_imagenet_norm
        
        # 构建数据路径
        data_dir = os.path.join(root, f'cifar10_{image_size}x{image_size}')
        split = 'train' if train else 'test'
        
        self.images_path = os.path.join(data_dir, f'{split}_images.npy')
        self.labels_path = os.path.join(data_dir, f'{split}_labels.npy')
        
        # 检查文件是否存在
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(
                f"预处理图像文件不存在: {self.images_path}\n"
                f"请先运行预处理脚本: python preprocess_data.py --image_size {image_size}"
            )
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(
                f"预处理标签文件不存在: {self.labels_path}\n"
                f"请先运行预处理脚本: python preprocess_data.py --image_size {image_size}"
            )
        
        # 使用内存映射加载数据（mmap_mode='r' 表示只读，支持多进程共享）
        # 关键优化：多个进程可以共享同一块物理内存，而不需要为每个进程复制数据
        print(f"[OfflineCIFAR10] 加载预处理数据（内存映射模式）...")
        print(f"  - 图像文件: {self.images_path}")
        print(f"  - 标签文件: {self.labels_path}")
        
        # 确定样本数量
        num_samples = 50000 if train else 10000
        
        # 图像文件使用 memmap 加载（这是原始 memmap 文件，不是标准 .npy）
        self.images = np.memmap(
            self.images_path,
            dtype='float32',
            mode='r',
            shape=(num_samples, 3, image_size, image_size)
        )
        
        # 标签文件使用标准 numpy 加载
        self.labels = np.load(self.labels_path, allow_pickle=True)
        
        print(f"  - 样本数量: {len(self.labels)}")
        print(f"  - 图像形状: {self.images.shape}")
        print(f"  - 内存映射: 启用（多进程共享内存）")
        
        # 设置 transform
        if transform is not None:
            self.transform = transform
        else:
            # 默认 transform：仅标准化（Resize 已在预处理时完成）
            if use_imagenet_norm:
                # 预训练模型：使用 ImageNet 标准化
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                print(f"  - 标准化: ImageNet 标准化（预训练模型）")
            else:
                # 从零训练：使用简单标准化
                normalize = transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
                print(f"  - 标准化: 标准标准化（从零训练）")
            
            # 注意：这里不需要 ToTensor()，因为数据已经是 numpy array 格式
            # 我们会在 __getitem__ 中手动转换为 Tensor
            self.transform = normalize
        
        # 用于兼容性的属性（与 torchvision.datasets.CIFAR10 保持一致）
        self.targets = self.labels.tolist()
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        关键优化：
          1. 从 memmap 读取数据（零拷贝，直接访问磁盘映射内存）
          2. 转换为 Tensor（快速操作）
          3. 应用标准化（轻量级操作）
          4. 无需 Resize（已在预处理时完成）
        """
        # 从 memmap 读取图像（已经是 float32，范围 [0, 1]）
        # 形状: (3, image_size, image_size)
        image = self.images[idx]
        label = int(self.labels[idx])
        
        # 复制数组以避免只读警告，然后转换为 Tensor
        image = torch.from_numpy(image.copy())
        
        # 应用标准化
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def __repr__(self):
        split = "Train" if self.train else "Test"
        return (f"OfflineCIFAR10(split={split}, size={len(self)}, "
                f"image_size={self.image_size}x{self.image_size}, "
                f"imagenet_norm={self.use_imagenet_norm})")


class OfflineCIFAR100(Dataset):
    """
    离线预处理的 CIFAR-100 数据集
    
    使用 numpy memmap 加载预处理好的数据，避免实时 Resize 操作
    支持多进程共享内存，大幅降低 CPU 负载和内存占用
    
    参数:
        root: 预处理数据的根目录（例如 '../data/preprocessed_cifar100/'）
        train: True 表示训练集，False 表示测试集
        image_size: 图像尺寸（必须与预处理时一致）
        transform: 额外的数据增强（可选，通常仅需要 Normalize）
        use_imagenet_norm: 是否使用 ImageNet 标准化（预训练模型需要）
    """
    
    def __init__(self, root='../data/preprocessed/', train=True, image_size=224,
                 transform=None, use_imagenet_norm=True):
        self.root = root
        self.train = train
        self.image_size = image_size
        self.use_imagenet_norm = use_imagenet_norm
        
        # 构建数据路径（与 CIFAR-10 格式一致）
        data_dir = os.path.join(root, f'cifar100_{image_size}x{image_size}')
        split = 'train' if train else 'test'
        
        self.images_path = os.path.join(data_dir, f'{split}_images.npy')
        self.labels_path = os.path.join(data_dir, f'{split}_labels.npy')
        
        # 检查文件是否存在
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(
                f"预处理图像文件不存在: {self.images_path}\n"
                f"请先运行预处理脚本: python preprocess_cifar100.py"
            )
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(
                f"预处理标签文件不存在: {self.labels_path}\n"
                f"请先运行预处理脚本: python preprocess_cifar100.py"
            )
        
        # 使用内存映射加载数据（mmap_mode='r' 表示只读，支持多进程共享）
        print(f"[OfflineCIFAR100] 加载预处理数据（内存映射模式）...")
        print(f"  - 图像文件: {self.images_path}")
        print(f"  - 标签文件: {self.labels_path}")
        
        # 确定样本数量
        num_samples = 50000 if train else 10000
        
        # 图像文件使用 memmap 加载
        self.images = np.memmap(
            self.images_path,
            dtype='float32',
            mode='r',
            shape=(num_samples, 3, image_size, image_size)
        )
        
        # 标签文件使用标准 numpy 加载
        self.labels = np.load(self.labels_path, allow_pickle=True)
        
        print(f"  - 样本数量: {len(self.labels)}")
        print(f"  - 图像形状: {self.images.shape}")
        print(f"  - 内存映射: 启用（多进程共享内存）")
        
        # 设置 transform
        if transform is not None:
            self.transform = transform
        else:
            # 默认 transform：仅标准化（Resize 已在预处理时完成）
            if use_imagenet_norm:
                # 预训练模型：使用 ImageNet 标准化
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                print(f"  - 标准化: ImageNet 标准化（预训练模型）")
            else:
                # 从零训练：使用简单标准化
                normalize = transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
                print(f"  - 标准化: 标准标准化（从零训练）")
            
            self.transform = normalize
        
        # 用于兼容性的属性
        self.targets = self.labels.tolist()
        # CIFAR-100 有 100 个类别
        self.num_classes = 100
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        关键优化：
          1. 从 memmap 读取数据（零拷贝，直接访问磁盘映射内存）
          2. 转换为 Tensor（快速操作）
          3. 应用标准化（轻量级操作）
          4. 无需 Resize（已在预处理时完成）
        """
        # 从 memmap 读取图像（已经是 float32，范围 [0, 1]）
        image = self.images[idx]
        label = int(self.labels[idx])
        
        # 复制数组以避免只读警告，然后转换为 Tensor
        image = torch.from_numpy(image.copy())
        
        # 应用标准化
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def __repr__(self):
        split = "Train" if self.train else "Test"
        return (f"OfflineCIFAR100(split={split}, size={len(self)}, "
                f"image_size={self.image_size}x{self.image_size}, "
                f"imagenet_norm={self.use_imagenet_norm})")


def get_offline_cifar10(root='../data/preprocessed/', image_size=224, use_imagenet_norm=True):
    """
    便捷函数：获取离线预处理的 CIFAR-10 训练集和测试集
    
    参数:
        root: 预处理数据的根目录
        image_size: 图像尺寸
        use_imagenet_norm: 是否使用 ImageNet 标准化
    
    返回:
        train_dataset, test_dataset
    """
    train_dataset = OfflineCIFAR10(
        root=root,
        train=True,
        image_size=image_size,
        use_imagenet_norm=use_imagenet_norm
    )
    
    test_dataset = OfflineCIFAR10(
        root=root,
        train=False,
        image_size=image_size,
        use_imagenet_norm=use_imagenet_norm
    )
    
    return train_dataset, test_dataset


def get_offline_cifar100(root='../data/preprocessed/', image_size=224, use_imagenet_norm=True):
    """
    便捷函数：获取离线预处理的 CIFAR-100 训练集和测试集
    
    参数:
        root: 预处理数据的根目录
        image_size: 图像尺寸
        use_imagenet_norm: 是否使用 ImageNet 标准化
    
    返回:
        train_dataset, test_dataset
    """
    train_dataset = OfflineCIFAR100(
        root=root,
        train=True,
        image_size=image_size,
        use_imagenet_norm=use_imagenet_norm
    )
    
    test_dataset = OfflineCIFAR100(
        root=root,
        train=False,
        image_size=image_size,
        use_imagenet_norm=use_imagenet_norm
    )
    
    return train_dataset, test_dataset


if __name__ == '__main__':
    """
    测试脚本：验证 OfflineCIFAR10 数据集的正确性
    """
    print("="*70)
    print("测试 OfflineCIFAR10 数据集".center(70))
    print("="*70 + "\n")
    
    # 测试不同尺寸
    for image_size in [224, 128]:
        print(f"\n{'='*70}")
        print(f"测试 {image_size}x{image_size} 数据集".center(70))
        print(f"{'='*70}\n")
        
        try:
            # 加载数据集
            train_dataset, test_dataset = get_offline_cifar10(
                root='../data/preprocessed/',
                image_size=image_size,
                use_imagenet_norm=True
            )
            
            print(f"\n训练集: {train_dataset}")
            print(f"测试集: {test_dataset}")
            
            # 测试数据加载
            print(f"\n测试数据加载...")
            img, label = train_dataset[0]
            print(f"  - 图像形状: {img.shape}")
            print(f"  - 图像数据类型: {img.dtype}")
            print(f"  - 图像值范围: [{img.min():.4f}, {img.max():.4f}]")
            print(f"  - 标签: {label} ({train_dataset.classes[label]})")
            
            # 测试批量加载
            from torch.utils.data import DataLoader
            print(f"\n测试批量加载（batch_size=32, num_workers=4）...")
            loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
            
            import time
            start_time = time.time()
            for i, (images, labels) in enumerate(loader):
                if i >= 10:  # 仅测试 10 个 batch
                    break
            elapsed = time.time() - start_time
            
            print(f"  - 加载 10 个 batch 耗时: {elapsed:.4f} 秒")
            print(f"  - 平均每个 batch: {elapsed/10:.4f} 秒")
            print(f"  ✓ 批量加载测试通过")
            
        except FileNotFoundError as e:
            print(f"\n[WARNING] 预处理数据不存在: {image_size}x{image_size}")
            print(f"  请运行: python preprocess_data.py --image_size {image_size}")
        except Exception as e:
            print(f"\n[ERROR] 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("测试完成".center(70))
    print(f"{'='*70}\n")
