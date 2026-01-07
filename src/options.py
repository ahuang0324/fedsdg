#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    
    # algorithm selection
    parser.add_argument('--alg', type=str, default='fedavg', choices=['fedavg', 'fedlora', 'fedsdg'],
                        help='federated learning algorithm: fedavg (full model), fedlora (LoRA-based), or fedsdg (LoRA with dual-path)')
    
    # LoRA-specific arguments (仅在 alg=fedlora 时使用)
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank (low-rank dimension), default=8')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA scaling parameter alpha, default=16')
    parser.add_argument('--lora_train_mlp_head', type=int, default=1,
                        help='Whether to train mlp_head in FedLoRA (1=True, 0=False), default=1')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn', 'vit'], help='model name')
    
    # pretrained model arguments (预训练模型相关参数)
    parser.add_argument('--model_variant', type=str, default='scratch', choices=['scratch', 'pretrained'],
                        help='Model variant: scratch (手写 ViT) or pretrained (timm 预训练 ViT), default=scratch')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to local pretrained weights (optional). If None, will download from timm.')
    parser.add_argument('--image_size', type=int, default=32,
                        help='Input image size. Use 224 for pretrained models, 32 for CIFAR from scratch.')
    
    # offline data preprocessing arguments (离线数据预处理参数)
    parser.add_argument('--use_offline_data', action='store_true',
                        help='Use offline preprocessed data (eliminates real-time Resize, reduces CPU load)')
    parser.add_argument('--offline_data_root', type=str, default='../data/preprocessed/',
                        help='Root directory for offline preprocessed data')
    
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['mnist', 'fmnist', 'cifar', 'cifar10', 'cifar100'],
                        help="name of dataset: mnist, fmnist, cifar/cifar10, cifar100")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID to use (>=0). Set to -1 for CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5,
                        help='Dirichlet concentration parameter for Non-IID partitioning. '
                             'Smaller -> more heterogeneous; larger (e.g. 100) -> approx IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log_subdir', type=str, default='fedavg_run_cnn_cifar_IID',
                        help='TensorBoard log subdirectory name (under ../logs)')
    args = parser.parse_args()
    
    # 验证 FedLoRA 和 FedSDG 仅支持 ViT 模型
    if args.alg in ('fedlora', 'fedsdg') and args.model != 'vit':
        raise ValueError(f"{args.alg.upper()} (--alg {args.alg}) currently only supports ViT model (--model vit)")
    
    # 验证预训练模型配置
    if args.model_variant == 'pretrained' and args.model != 'vit':
        raise ValueError("Pretrained variant (--model_variant pretrained) only supports ViT model (--model vit)")
    
    # 预训练模型建议使用 224x224 分辨率
    if args.model_variant == 'pretrained' and args.image_size == 32:
        print("[WARNING] Using pretrained ViT with image_size=32. Consider using --image_size 224 for better performance.")
    
    # 自动设置 num_classes（如果用户未明确指定）
    if args.dataset == 'cifar100':
        args.num_classes = 100
        print(f"[Config] 自动设置 num_classes=100 (CIFAR-100)")
    elif args.dataset in ('cifar', 'cifar10'):
        if args.num_classes != 10:
            print(f"[Config] 覆盖 num_classes={args.num_classes} -> 10 (CIFAR-10)")
        args.num_classes = 10
        # 统一使用 'cifar' 作为内部标识
        if args.dataset == 'cifar10':
            args.dataset = 'cifar'
            print(f"[Config] 数据集名称标准化: cifar10 -> cifar")
    elif args.dataset in ('mnist', 'fmnist'):
        if args.num_classes != 10:
            print(f"[Config] 覆盖 num_classes={args.num_classes} -> 10 (MNIST/Fashion-MNIST)")
        args.num_classes = 10
    
    return args
