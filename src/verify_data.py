#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立数据验证脚本
功能：全面检测预处理后的 CIFAR-10 数据是否正确
包括：文件完整性、数据形状、数值范围、标签分布、数据一致性等
"""

import os
import numpy as np
import argparse
from tqdm import tqdm


def verify_preprocessed_data(image_size=224, output_root='../data/preprocessed/', 
                            check_samples=10, verbose=True):
    """
    全面验证预处理数据的完整性和正确性
    
    参数:
        image_size: 图像尺寸
        output_root: 预处理数据根目录
        check_samples: 随机检查的样本数量
        verbose: 是否显示详细信息
    
    返回:
        bool: 验证是否通过
    """
    print("\n" + "="*80)
    print("CIFAR-10 预处理数据验证工具".center(80))
    print("="*80 + "\n")
    
    output_dir = os.path.join(output_root, f'cifar10_{image_size}x{image_size}')
    
    if not os.path.exists(output_dir):
        print(f"❌ 错误：数据目录不存在: {output_dir}")
        print(f"   请先运行预处理脚本: python preprocess_data.py --image_size {image_size}")
        return False
    
    print(f"📁 数据目录: {output_dir}\n")
    
    all_passed = True
    
    for split in ['train', 'test']:
        print(f"\n{'='*80}")
        print(f"验证 {split.upper()} 集".center(80))
        print(f"{'='*80}\n")
        
        images_path = os.path.join(output_dir, f'{split}_images.npy')
        labels_path = os.path.join(output_dir, f'{split}_labels.npy')
        
        # ========== 1. 文件存在性检查 ==========
        print("【1】文件存在性检查")
        if not os.path.exists(images_path):
            print(f"  ❌ 图像文件不存在: {images_path}")
            all_passed = False
            continue
        else:
            print(f"  ✓ 图像文件存在: {images_path}")
        
        if not os.path.exists(labels_path):
            print(f"  ❌ 标签文件不存在: {labels_path}")
            all_passed = False
            continue
        else:
            print(f"  ✓ 标签文件存在: {labels_path}")
        
        # 文件大小
        images_size_mb = os.path.getsize(images_path) / (1024 ** 2)
        labels_size_mb = os.path.getsize(labels_path) / (1024 ** 2)
        print(f"  ✓ 图像文件大小: {images_size_mb:.2f} MB")
        print(f"  ✓ 标签文件大小: {labels_size_mb:.2f} MB")
        
        # ========== 2. 数据加载检查 ==========
        print("\n【2】数据加载检查")
        try:
            # 图像文件是 memmap 格式，需要知道形状和数据类型
            expected_samples = 50000 if split == 'train' else 10000
            
            # 尝试加载图像 memmap
            images = np.memmap(
                images_path,
                dtype='float32',
                mode='r',
                shape=(expected_samples, 3, image_size, image_size)
            )
            
            # 标签文件是标准 numpy 数组
            labels = np.load(labels_path, allow_pickle=True)
            
            print(f"  ✓ 数据加载成功（图像: memmap, 标签: numpy array）")
        except Exception as e:
            print(f"  ❌ 数据加载失败: {e}")
            all_passed = False
            continue
        
        # ========== 3. 数据形状检查 ==========
        print("\n【3】数据形状检查")
        expected_samples = 50000 if split == 'train' else 10000
        expected_shape = (expected_samples, 3, image_size, image_size)
        
        if images.shape == expected_shape:
            print(f"  ✓ 图像形状正确: {images.shape}")
        else:
            print(f"  ❌ 图像形状错误: {images.shape}, 期望: {expected_shape}")
            all_passed = False
        
        if labels.shape == (expected_samples,):
            print(f"  ✓ 标签形状正确: {labels.shape}")
        else:
            print(f"  ❌ 标签形状错误: {labels.shape}, 期望: ({expected_samples},)")
            all_passed = False
        
        # ========== 4. 数据类型检查 ==========
        print("\n【4】数据类型检查")
        if images.dtype == np.float32:
            print(f"  ✓ 图像数据类型正确: {images.dtype}")
        else:
            print(f"  ❌ 图像数据类型错误: {images.dtype}, 期望: float32")
            all_passed = False
        
        if labels.dtype == np.int64:
            print(f"  ✓ 标签数据类型正确: {labels.dtype}")
        else:
            print(f"  ⚠ 标签数据类型: {labels.dtype} (期望: int64, 但可能兼容)")
        
        # ========== 5. 数据值范围检查 ==========
        print("\n【5】数据值范围检查")
        
        if len(images) == 0:
            print(f"  ❌ 图像数据为空")
            all_passed = False
        else:
            # 检查多个样本的值范围
            sample_indices = np.random.choice(len(images), min(check_samples, len(images)), replace=False)
            min_vals = []
            max_vals = []
            
            for idx in sample_indices:
                min_vals.append(images[idx].min())
                max_vals.append(images[idx].max())
            
            overall_min = min(min_vals)
            overall_max = max(max_vals)
            
            # 图像应该在 [0, 1] 范围内（经过 ToTensor 归一化）
            if 0.0 <= overall_min and overall_max <= 1.0:
                print(f"  ✓ 图像值范围正确: [{overall_min:.4f}, {overall_max:.4f}] (检查了 {len(sample_indices)} 个样本)")
            else:
                print(f"  ❌ 图像值范围异常: [{overall_min:.4f}, {overall_max:.4f}], 期望: [0.0, 1.0]")
                all_passed = False
            
            if verbose:
                print(f"     样本值范围详情:")
                for i, idx in enumerate(sample_indices[:5]):  # 只显示前5个
                    print(f"       样本 {idx}: [{min_vals[i]:.4f}, {max_vals[i]:.4f}]")
        
        if len(labels) == 0:
            print(f"  ❌ 标签数据为空")
            all_passed = False
        else:
            label_min = labels.min()
            label_max = labels.max()
            
            # CIFAR-10 标签应该在 [0, 9] 范围内
            if label_min == 0 and label_max == 9:
                print(f"  ✓ 标签值范围正确: [{label_min}, {label_max}]")
            else:
                print(f"  ❌ 标签值范围错误: [{label_min}, {label_max}], 期望: [0, 9]")
                all_passed = False
        
        # ========== 6. 标签分布检查 ==========
        print("\n【6】标签分布检查")
        if len(labels) > 0:
            label_counts = np.bincount(labels)
            print(f"  ✓ 标签分布:")
            
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
            
            for i, count in enumerate(label_counts):
                percentage = count / len(labels) * 100
                print(f"     类别 {i} ({class_names[i]:>10s}): {count:>5d} 样本 ({percentage:>5.2f}%)")
            
            # 检查类别平衡性（CIFAR-10 应该是平衡的）
            expected_count = len(labels) // 10
            tolerance = expected_count * 0.1  # 允许 10% 的偏差
            
            is_balanced = all(abs(count - expected_count) <= tolerance for count in label_counts)
            if is_balanced:
                print(f"  ✓ 类别分布平衡（每类约 {expected_count} 个样本）")
            else:
                print(f"  ⚠ 类别分布不完全平衡（期望每类约 {expected_count} 个样本）")
        
        # ========== 7. 数据一致性检查 ==========
        print("\n【7】数据一致性检查")
        if len(images) == len(labels):
            print(f"  ✓ 图像和标签数量一致: {len(images)} 个样本")
        else:
            print(f"  ❌ 图像和标签数量不一致: 图像 {len(images)}, 标签 {len(labels)}")
            all_passed = False
        
        # ========== 8. 随机样本抽查 ==========
        print("\n【8】随机样本抽查")
        if len(images) > 0 and len(labels) > 0:
            print(f"  正在检查 {check_samples} 个随机样本...")
            sample_indices = np.random.choice(len(images), min(check_samples, len(images)), replace=False)
            
            issues = []
            for idx in tqdm(sample_indices, desc="  检查样本", leave=False):
                img = images[idx]
                label = labels[idx]
                
                # 检查图像形状
                if img.shape != (3, image_size, image_size):
                    issues.append(f"样本 {idx}: 形状错误 {img.shape}")
                
                # 检查是否有 NaN 或 Inf
                if np.isnan(img).any():
                    issues.append(f"样本 {idx}: 包含 NaN")
                if np.isinf(img).any():
                    issues.append(f"样本 {idx}: 包含 Inf")
                
                # 检查标签范围
                if not (0 <= label <= 9):
                    issues.append(f"样本 {idx}: 标签超出范围 {label}")
            
            if len(issues) == 0:
                print(f"  ✓ 所有抽查样本正常")
            else:
                print(f"  ❌ 发现 {len(issues)} 个问题:")
                for issue in issues[:10]:  # 最多显示 10 个
                    print(f"     - {issue}")
                all_passed = False
        
        print(f"\n{'-'*80}")
    
    # ========== 9. 元数据检查 ==========
    print(f"\n{'='*80}")
    print("【9】元数据检查".center(80))
    print(f"{'='*80}\n")
    
    metadata_path = os.path.join(output_dir, 'metadata.txt')
    if os.path.exists(metadata_path):
        print(f"✓ 元数据文件存在: {metadata_path}")
        print(f"\n元数据内容:")
        with open(metadata_path, 'r') as f:
            for line in f:
                print(f"  {line.rstrip()}")
    else:
        print(f"⚠ 元数据文件不存在: {metadata_path}")
    
    # ========== 最终结果 ==========
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ 验证通过！数据完整性和正确性正常。".center(80))
    else:
        print("❌ 验证失败！发现数据问题，请检查上述错误信息。".center(80))
    print(f"{'='*80}\n")
    
    return all_passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 预处理数据验证工具')
    parser.add_argument('--image_size', type=int, default=224,
                        help='图像尺寸（默认 224）')
    parser.add_argument('--output_root', type=str, default='../data/preprocessed/',
                        help='预处理数据根目录')
    parser.add_argument('--check_samples', type=int, default=10,
                        help='随机检查的样本数量（默认 10）')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细信息')
    
    args = parser.parse_args()
    
    # 运行验证
    passed = verify_preprocessed_data(
        image_size=args.image_size,
        output_root=args.output_root,
        check_samples=args.check_samples,
        verbose=args.verbose
    )
    
    # 返回退出码
    exit(0 if passed else 1)
