#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-100 预处理数据验证脚本

功能：
1. 检查预处理数据文件是否存在
2. 验证数据加载是否正确
3. 检查数据形状、类型和值范围
4. 验证标签分布
5. 测试 DataLoader 性能
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader


def verify_file_existence(output_root, image_size=224):
    """检查文件是否存在"""
    print("\n" + "="*60)
    print("【1】文件存在性检查")
    print("="*60)
    
    all_passed = True
    dataset_dir = os.path.join(output_root, f'cifar100_{image_size}x{image_size}')
    
    for split in ['train', 'test']:
        print(f"\n检查 {split} 数据集:")
        
        images_path = os.path.join(dataset_dir, f'{split}_images.npy')
        labels_path = os.path.join(dataset_dir, f'{split}_labels.npy')
        
        if os.path.exists(images_path):
            size_mb = os.path.getsize(images_path) / (1024 * 1024)
            print(f"  ✓ 图像文件存在: {images_path}")
            print(f"    大小: {size_mb:.2f} MB")
        else:
            print(f"  ❌ 图像文件不存在: {images_path}")
            all_passed = False
        
        if os.path.exists(labels_path):
            size_mb = os.path.getsize(labels_path) / (1024 * 1024)
            print(f"  ✓ 标签文件存在: {labels_path}")
            print(f"    大小: {size_mb:.2f} MB")
        else:
            print(f"  ❌ 标签文件不存在: {labels_path}")
            all_passed = False
    
    return all_passed


def verify_data_loading(output_root, image_size=224):
    """验证数据加载"""
    print("\n" + "="*60)
    print("【2】数据加载检查")
    print("="*60)
    
    all_passed = True
    dataset_dir = os.path.join(output_root, f'cifar100_{image_size}x{image_size}')
    
    for split in ['train', 'test']:
        print(f"\n检查 {split} 数据集:")
        
        images_path = os.path.join(dataset_dir, f'{split}_images.npy')
        labels_path = os.path.join(dataset_dir, f'{split}_labels.npy')
        
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"  ⚠️  跳过（文件不存在）")
            all_passed = False
            continue
        
        try:
            expected_samples = 50000 if split == 'train' else 10000
            
            images = np.memmap(
                images_path,
                dtype='float32',
                mode='r',
                shape=(expected_samples, 3, image_size, image_size)
            )
            
            labels = np.load(labels_path, allow_pickle=True)
            
            print(f"  ✓ 数据加载成功（图像: memmap, 标签: numpy array）")
        except Exception as e:
            print(f"  ❌ 数据加载失败: {e}")
            all_passed = False
            continue
        
        print(f"\n  数据形状:")
        print(f"    - 图像: {images.shape}")
        print(f"    - 标签: {labels.shape}")
        
        if images.shape != (expected_samples, 3, image_size, image_size):
            print(f"  ❌ 图像形状不正确，期望: ({expected_samples}, 3, {image_size}, {image_size})")
            all_passed = False
        else:
            print(f"  ✓ 图像形状正确")
        
        if labels.shape != (expected_samples,):
            print(f"  ❌ 标签形状不正确，期望: ({expected_samples},)")
            all_passed = False
        else:
            print(f"  ✓ 标签形状正确")
        
        print(f"\n  数据类型:")
        print(f"    - 图像: {images.dtype}")
        print(f"    - 标签: {labels.dtype}")
        
        if images.dtype != np.float32:
            print(f"  ⚠️  图像数据类型不是 float32")
        
        if len(images) > 0:
            img_min = images[0].min()
            img_max = images[0].max()
            print(f"\n  数据值范围（样本0）:")
            print(f"    - 图像: [{img_min:.4f}, {img_max:.4f}]")
            
            if img_min < 0 or img_max > 1:
                print(f"  ⚠️  图像值超出 [0, 1] 范围")
        
        if len(labels) > 0:
            label_min = labels.min()
            label_max = labels.max()
            print(f"    - 标签: [{label_min}, {label_max}]")
            
            if label_min != 0 or label_max != 99:
                print(f"  ⚠️  标签值不在 [0, 99] 范围（CIFAR-100 有 100 个类别）")
            else:
                print(f"  ✓ 标签值范围正确（0-99，共100个类别）")
            
            unique_labels = np.unique(labels)
            print(f"\n  标签统计:")
            print(f"    - 唯一标签数: {len(unique_labels)}")
            print(f"    - 每个类别平均样本数: {len(labels) // 100}")
            
            if len(unique_labels) != 100:
                print(f"  ⚠️  唯一标签数不等于 100")
    
    return all_passed


def verify_dataset_class(output_root, image_size=224):
    """验证 OfflineCIFAR100 类"""
    print("\n" + "="*60)
    print("【3】OfflineCIFAR100 类检查")
    print("="*60)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from offline_dataset import OfflineCIFAR100
        
        print("\n加载训练集...")
        train_dataset = OfflineCIFAR100(
            root=output_root,
            train=True,
            image_size=image_size,
            use_imagenet_norm=True
        )
        print(f"  ✓ 训练集加载成功: {len(train_dataset)} 个样本")
        
        print("\n加载测试集...")
        test_dataset = OfflineCIFAR100(
            root=output_root,
            train=False,
            image_size=image_size,
            use_imagenet_norm=True
        )
        print(f"  ✓ 测试集加载成功: {len(test_dataset)} 个样本")
        
        print("\n测试数据访问...")
        img, label = train_dataset[0]
        print(f"  ✓ 样本 0:")
        print(f"    - 图像形状: {img.shape}")
        print(f"    - 图像类型: {img.dtype}")
        print(f"    - 图像值范围: [{img.min():.4f}, {img.max():.4f}]")
        print(f"    - 标签: {label} (类型: {type(label).__name__})")
        
        if img.shape != (3, image_size, image_size):
            print(f"  ❌ 图像形状不正确")
            return False
        
        if not isinstance(img, torch.Tensor):
            print(f"  ❌ 图像不是 Tensor 类型")
            return False
        
        print(f"  ✓ 数据格式正确")
        
        return True
        
    except Exception as e:
        print(f"\n❌ OfflineCIFAR100 类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dataloader_performance(output_root, image_size=224):
    """验证 DataLoader 性能"""
    print("\n" + "="*60)
    print("【4】DataLoader 性能测试")
    print("="*60)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from offline_dataset import OfflineCIFAR100
        
        train_dataset = OfflineCIFAR100(
            root=output_root,
            train=True,
            image_size=image_size,
            use_imagenet_norm=True
        )
        
        print("\n测试配置:")
        print("  - batch_size: 32")
        print("  - num_workers: 4")
        print("  - pin_memory: True")
        print("  - 测试批次数: 50")
        
        loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        import time
        start_time = time.time()
        
        for i, (images, labels) in enumerate(loader):
            if i >= 50:
                break
        
        elapsed = time.time() - start_time
        
        print(f"\n性能结果:")
        print(f"  - 总耗时: {elapsed:.4f} 秒")
        print(f"  - 平均每批次: {elapsed/50:.4f} 秒")
        print(f"  - 吞吐量: {50*32/elapsed:.2f} 样本/秒")
        print(f"  ✓ DataLoader 测试通过")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DataLoader 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    output_root = '../data/preprocessed/'
    image_size = 224
    
    print("\n" + "="*60)
    print("CIFAR-100 预处理数据验证")
    print("="*60)
    print(f"数据路径: {output_root}")
    print(f"图像尺寸: {image_size}x{image_size}")
    
    results = []
    
    results.append(("文件存在性", verify_file_existence(output_root, image_size)))
    results.append(("数据加载", verify_data_loading(output_root, image_size)))
    results.append(("Dataset类", verify_dataset_class(output_root, image_size)))
    results.append(("DataLoader性能", verify_dataloader_performance(output_root, image_size)))
    
    print("\n" + "="*60)
    print("验证结果汇总")
    print("="*60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有验证通过！数据可以用于训练。")
    else:
        print("❌ 部分验证失败，请检查预处理步骤。")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
