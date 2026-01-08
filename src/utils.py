#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from sampling import dirichlet_partition
from offline_dataset import OfflineCIFAR10, OfflineCIFAR100


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        # 检查是否使用离线预处理数据
        if hasattr(args, 'use_offline_data') and args.use_offline_data:
            # 使用离线预处理数据（零 CPU 开销的 Resize）
            print("\n" + "="*70)
            print("[Offline Data] 使用离线预处理数据模式".center(70))
            print("="*70)
            print(f"[Offline Data] 数据路径: {args.offline_data_root}")
            print(f"[Offline Data] 图像尺寸: {args.image_size}x{args.image_size}")
            
            # 判断是否使用 ImageNet 标准化
            use_imagenet_norm = (hasattr(args, 'model_variant') and 
                                args.model_variant == 'pretrained')
            
            if use_imagenet_norm:
                print(f"[Offline Data] 标准化: ImageNet 标准化（预训练模型）")
            else:
                print(f"[Offline Data] 标准化: 标准标准化（从零训练）")
            
            print("="*70 + "\n")
            
            # 加载离线数据集
            train_dataset = OfflineCIFAR10(
                root=args.offline_data_root,
                train=True,
                image_size=args.image_size,
                use_imagenet_norm=use_imagenet_norm
            )
            
            test_dataset = OfflineCIFAR10(
                root=args.offline_data_root,
                train=False,
                image_size=args.image_size,
                use_imagenet_norm=use_imagenet_norm
            )
            
            print(f"[Offline Data] 训练集: {len(train_dataset)} 样本")
            print(f"[Offline Data] 测试集: {len(test_dataset)} 样本")
            print(f"[Offline Data] 优势: 零 Resize 开销 + 多进程共享内存 + 低 CPU 负载\n")
        else:
            # 使用传统的在线数据加载（实时 Resize）
            data_dir = '../data/cifar/'
            
            # 根据 model_variant 选择不同的数据预处理策略
            if hasattr(args, 'model_variant') and args.model_variant == 'pretrained':
                # 预训练模型：使用 ImageNet 标准化 + Resize 到指定尺寸
                # ImageNet 均值和方差（RGB 顺序）
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                
                transform_list = [transforms.ToTensor()]
                
                # 如果需要 resize（通常预训练模型需要 224x224）
                if hasattr(args, 'image_size') and args.image_size != 32:
                    transform_list.insert(0, transforms.Resize(args.image_size))
                    print(f"[Data] CIFAR-10 将被 resize 到 {args.image_size}x{args.image_size}")
                
                transform_list.append(transforms.Normalize(imagenet_mean, imagenet_std))
                apply_transform = transforms.Compose(transform_list)
                print(f"[Data] 使用 ImageNet 标准化（预训练模型模式）")
            else:
                # 从零训练：使用简单的标准化
                apply_transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                print(f"[Data] 使用标准标准化（从零训练模式）")

            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                          transform=apply_transform)

    elif args.dataset == 'cifar100':
        # 检查是否使用离线预处理数据
        if hasattr(args, 'use_offline_data') and args.use_offline_data:
            # 使用离线预处理数据（零 CPU 开销的 Resize）
            print("\n" + "="*70)
            print("[Offline Data] 使用离线预处理数据模式 (CIFAR-100)".center(70))
            print("="*70)
            # 使用统一的预处理数据根目录
            offline_root = args.offline_data_root if hasattr(args, 'offline_data_root') else '../data/preprocessed/'
            print(f"[Offline Data] 数据路径: {offline_root}")
            print(f"[Offline Data] 图像尺寸: {args.image_size}x{args.image_size}")
            
            # 判断是否使用 ImageNet 标准化
            use_imagenet_norm = (hasattr(args, 'model_variant') and 
                                args.model_variant == 'pretrained')
            
            if use_imagenet_norm:
                print(f"[Offline Data] 标准化: ImageNet 标准化（预训练模型）")
            else:
                print(f"[Offline Data] 标准化: 标准标准化（从零训练）")
            
            print("="*70 + "\n")
            
            # 加载离线数据集
            train_dataset = OfflineCIFAR100(
                root=offline_root,
                train=True,
                image_size=args.image_size,
                use_imagenet_norm=use_imagenet_norm
            )
            
            test_dataset = OfflineCIFAR100(
                root=offline_root,
                train=False,
                image_size=args.image_size,
                use_imagenet_norm=use_imagenet_norm
            )
            
            print(f"[Offline Data] 训练集: {len(train_dataset)} 样本")
            print(f"[Offline Data] 测试集: {len(test_dataset)} 样本")
            print(f"[Offline Data] 类别数: 100")
            print(f"[Offline Data] 优势: 零 Resize 开销 + 多进程共享内存 + 低 CPU 负载\n")
        else:
            # 使用传统的在线数据加载（实时 Resize）
            data_dir = '../data/cifar100/'
            
            # 根据 model_variant 选择不同的数据预处理策略
            if hasattr(args, 'model_variant') and args.model_variant == 'pretrained':
                # 预训练模型：使用 ImageNet 标准化 + Resize 到指定尺寸
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                
                transform_list = [transforms.ToTensor()]
                
                # 如果需要 resize（通常预训练模型需要 224x224）
                if hasattr(args, 'image_size') and args.image_size != 32:
                    transform_list.insert(0, transforms.Resize(args.image_size))
                    print(f"[Data] CIFAR-100 将被 resize 到 {args.image_size}x{args.image_size}")
                
                transform_list.append(transforms.Normalize(imagenet_mean, imagenet_std))
                apply_transform = transforms.Compose(transform_list)
                print(f"[Data] 使用 ImageNet 标准化（预训练模型模式）")
            else:
                # 从零训练：使用简单的标准化
                apply_transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                print(f"[Data] 使用标准标准化（从零训练模式）")

            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                          transform=apply_transform)

    elif args.dataset in ('mnist', 'fmnist'):
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    else:
        raise ValueError('Error: unrecognized dataset')

    if args.unequal:
        raise NotImplementedError()

    alpha = float(args.dirichlet_alpha)
    user_groups = dirichlet_partition(train_dataset, args.num_users, alpha=alpha, min_size=20)
    print(f"Partitioning complete: Alpha: {alpha}")

    # 提示（用于后续画热力图验证 Non-IID 程度）：
    # 1) 取出训练集标签 targets（MNIST/CIFAR10 通常是 dataset.targets）
    # 2) 对每个 client 的 indices（user_groups[client_id]）做标签计数直方图
    # 3) 将每个 client 的类别直方图堆叠成矩阵，即可绘制 heatmap
    # 注意：alpha 越小越异构，alpha 越大（如 100）越接近 IID。

    # 额外支持：自动绘制“客户端-类别分布”热力图，并输出到 ../save/objects/ 下
    # - 行：client_id（0..num_users-1）
    # - 列：class_id（0..num_classes-1）
    # - 值：该客户端该类别的样本数
    try:
        targets = train_dataset.targets
        if hasattr(targets, 'cpu'):
            targets = targets.cpu()
        targets = np.asarray(targets)
    except Exception:
        targets = None

    if targets is not None:
        num_classes = int(np.max(targets)) + 1
        dist = np.zeros((args.num_users, num_classes), dtype=np.int64)
        for client_id, idxs in user_groups.items():
            idxs = np.asarray(idxs, dtype=np.int64)
            dist[client_id] = np.bincount(targets[idxs], minlength=num_classes)

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            path_project = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            objects_dir = os.path.join(path_project, 'save', 'objects')
            os.makedirs(objects_dir, exist_ok=True)
            out_path = os.path.join(
                objects_dir,
                f"client_class_heatmap_{args.dataset}_{args.num_users}_alpha{args.dirichlet_alpha}.png",
            )

            plt.figure(figsize=(max(8, num_classes * 0.8), max(6, args.num_users * 0.12)))
            plt.imshow(dist, aspect='auto', interpolation='nearest')
            plt.colorbar(label='num_samples')
            plt.xlabel('class_id')
            plt.ylabel('client_id')
            plt.title(f"Client-Class Distribution Heatmap (alpha={args.dirichlet_alpha})")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()

            print(f"Heatmap saved to: {out_path}")
        except ImportError:
            print('matplotlib not installed; skip client-class heatmap plotting.')

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_lora(w, global_state_dict):
    """
    FedLoRA 专用的选择性聚合函数
    仅聚合包含 'lora_' 关键词的参数（LoRA 低秩矩阵 A 和 B）以及 mlp_head 参数
    其他冻结的骨干权重保持不变
    
    参数:
        w: 客户端上传的 state_dict 列表 (每个元素是一个 state_dict)
        global_state_dict: 全局模型的完整 state_dict（用于保留冻结权重）
    
    返回:
        更新后的全局 state_dict（仅 LoRA 参数被聚合更新）
    """
    # 1. 深拷贝全局 state_dict 作为基础（保留所有冻结权重）
    w_avg = copy.deepcopy(global_state_dict)
    
    # 2. 仅对 LoRA 相关参数进行 FedAvg 聚合
    # 识别所有需要聚合的 LoRA 参数键
    # 注意：需要同时匹配 'mlp_head'（手写 ViT）和 'head'（timm ViT）
    # FedSDG: 排除私有参数（_private）和门控参数（lambda_k）
    lora_keys = [key for key in w[0].keys() 
                 if ('lora_' in key or 'mlp_head' in key or 'head' in key) 
                 and '_private' not in key 
                 and 'lambda_k' not in key]
    
    if len(lora_keys) == 0:
        print("  [WARNING] 未找到任何 LoRA 参数，请检查模型是否正确注入 LoRA")
        return w_avg
    
    # 3. 对每个 LoRA 参数键进行平均聚合
    for key in lora_keys:
        # 初始化为第一个客户端的参数
        w_avg[key] = copy.deepcopy(w[0][key])
        # 累加其他客户端的参数
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        # 计算平均值
        w_avg[key] = torch.div(w_avg[key], len(w))
    
    print(f"  [FedLoRA] 已聚合 {len(lora_keys)} 个 LoRA 参数键")
    
    # 4. 验证返回的是完整的 state_dict（包含所有键）
    # 这确保 load_state_dict() 可以正常工作（strict=True）
    assert len(w_avg) == len(global_state_dict), \
        f"聚合后的 state_dict 键数量不匹配：{len(w_avg)} vs {len(global_state_dict)}"
    
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Algorithm : {args.alg}')  # 新增：显示算法类型
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    # Dirichlet 划分的异构程度由 alpha 控制：
    # - alpha -> 0：极端异构（每个客户端更偏向少数类别）
    # - alpha 很大（如 100）：统计意义上接近 IID
    print(f'    Dirichlet Alpha : {args.dirichlet_alpha}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}')
    
    # 新增：显示 LoRA 参数（如果使用 FedLoRA 或 FedSDG）
    if args.alg in ('fedlora', 'fedsdg'):
        print(f'\n    LoRA parameters:')
        print(f'    LoRA rank (r)      : {args.lora_r}')
        print(f'    LoRA alpha         : {args.lora_alpha}')
        print(f'    Train mlp_head     : {bool(args.lora_train_mlp_head)}')
        if args.alg == 'fedsdg':
            print(f'\n    FedSDG specific:')
            print(f'    Dual-path mode     : Enabled (Global + Private branches)')
            print(f'    Private params     : Not communicated (client-local only)')
    print()
    return


def get_communication_stats(model, alg):
    """
    计算联邦学习中的通信量统计
    
    参数:
        model: PyTorch 模型
        alg: 算法类型 ('fedavg' 或 'fedlora')
    
    返回:
        dict: 包含通信量统计的字典
            - total_params: 模型总参数量
            - trainable_params: 可训练参数量
            - comm_params: 每轮通信的参数量
            - total_size_mb: 模型总大小（MB）
            - trainable_size_mb: 可训练参数大小（MB）
            - comm_size_mb: 每轮通信大小（MB）
            - compression_ratio: 压缩比（FedLoRA vs FedAvg）
    """
    # 统计总参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 统计可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算通信参数量
    if alg == 'fedavg':
        # FedAvg: 通信所有参数
        comm_params = total_params
    elif alg in ('fedlora', 'fedsdg'):
        # FedLoRA 和 FedSDG: 仅通信全局 LoRA 参数（不包括私有参数）
        # FedSDG 的私有参数（_private 和 lambda_k）不参与通信
        # 需要精确计算：排除私有参数和门控参数
        comm_params = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                # 排除私有参数和门控参数
                if '_private' not in name and 'lambda_k' not in name:
                    comm_params += p.numel()
    else:
        comm_params = total_params
    
    # 转换为 MB（假设 float32，每个参数 4 字节）
    bytes_per_param = 4
    total_size_mb = (total_params * bytes_per_param) / (1024 ** 2)
    trainable_size_mb = (trainable_params * bytes_per_param) / (1024 ** 2)
    comm_size_mb = (comm_params * bytes_per_param) / (1024 ** 2)
    
    # 计算压缩比
    compression_ratio = (comm_params / total_params) * 100 if total_params > 0 else 100
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'comm_params': comm_params,
        'total_size_mb': total_size_mb,
        'trainable_size_mb': trainable_size_mb,
        'comm_size_mb': comm_size_mb,
        'compression_ratio': compression_ratio
    }


def print_communication_profile(comm_stats, args):
    """
    打印通信量统计表格
    
    参数:
        comm_stats: get_communication_stats() 返回的统计字典
        args: 命令行参数
    """
    print("\n" + "="*70)
    print("COMMUNICATION PROFILE".center(70))
    print("="*70)
    
    print(f"\n{'Metric':<35} {'Value':>20} {'Unit':>10}")
    print("-"*70)
    
    # 模型参数统计
    print(f"{'Total Parameters':<35} {comm_stats['total_params']:>20,} {'params':>10}")
    print(f"{'Trainable Parameters':<35} {comm_stats['trainable_params']:>20,} {'params':>10}")
    print(f"{'Communication Parameters':<35} {comm_stats['comm_params']:>20,} {'params':>10}")
    
    print("-"*70)
    
    # 大小统计（MB）
    print(f"{'Total Model Size':<35} {comm_stats['total_size_mb']:>20.2f} {'MB':>10}")
    print(f"{'Trainable Size':<35} {comm_stats['trainable_size_mb']:>20.2f} {'MB':>10}")
    print(f"{'Communication per Round (1-way)':<35} {comm_stats['comm_size_mb']:>20.2f} {'MB':>10}")
    
    # 计算双向通信（上传 + 下载）
    comm_per_round_2way = comm_stats['comm_size_mb'] * 2
    print(f"{'Communication per Round (2-way)':<35} {comm_per_round_2way:>20.2f} {'MB':>10}")
    
    print("-"*70)
    
    # 总通信量估算
    total_rounds = args.epochs
    estimated_total_volume = comm_per_round_2way * total_rounds
    print(f"{'Total Rounds':<35} {total_rounds:>20} {'rounds':>10}")
    print(f"{'Estimated Total Volume':<35} {estimated_total_volume:>20.2f} {'MB':>10}")
    print(f"{'Estimated Total Volume':<35} {estimated_total_volume/1024:>20.2f} {'GB':>10}")
    
    print("-"*70)
    
    # 效率指标
    print(f"{'Compression Ratio':<35} {comm_stats['compression_ratio']:>20.2f} {'%':>10}")
    
    if args.alg == 'fedlora':
        bandwidth_saving = 100 - comm_stats['compression_ratio']
        print(f"{'Bandwidth Saving vs FedAvg':<35} {bandwidth_saving:>20.2f} {'%':>10}")
    
    print("="*70)
    
    # 算法特定说明
    if args.alg == 'fedavg':
        print("\n[FedAvg] Communicating ALL model parameters each round")
    elif args.alg == 'fedlora':
        print("\n[FedLoRA] Communicating ONLY LoRA parameters + classification head")
        print(f"[FedLoRA] Parameter Efficiency: {comm_stats['compression_ratio']:.2f}% of full model")
    elif args.alg == 'fedsdg':
        print("\n[FedSDG] Communicating ONLY Global LoRA parameters (lora_A, lora_B) + classification head")
        print(f"[FedSDG] Private parameters (lora_A_private, lora_B_private, lambda_k) stay local")
        print(f"[FedSDG] Communication Efficiency: {comm_stats['compression_ratio']:.2f}% of full model (same as FedLoRA)")
    
    print("="*70 + "\n")
