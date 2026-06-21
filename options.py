# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command line argument parser for federated learning experiments.

Supports loading configuration from YAML files:
    python main.py --config configs/fedsdg.yaml
    python main.py --config configs/fedsdg.yaml --dataset cifar --epochs 50
    
Priority: Command line args > Config file > Default values

三层参数传递 （优先级由高到低）：
- 命令行参数
- 配置文件
- 代码中的默认值（硬编码）


参数配置策略：
1. 配置文件 (YAML): 存放核心算法参数、默认超参数、模型架构配置
   - 优点：便于复现实验、版本控制、参数组织清晰
   - 适合：算法核心参数（lambda1, lambda2）、训练默认配置、数据集预设
   
2. 命令行参数: 用于快速调整实验变量、环境相关参数
   - 优点：灵活，便于批量实验
   - 适合：训练轮数、GPU选择、数据集切换、随机种子
   
3. 混合使用（推荐）: 基础配置在文件，实验变量在命令行
   - 示例: python main.py --config configs/fedsdg.yaml --epochs 150 --gpu 2
   - 说明: 命令行参数会覆盖配置文件中的对应值（无冲突，自动处理）
   - 也就是说 配置文件作为基本的默认参数，然后可以使用命令行进行灵活的调整覆盖。
"""

import argparse
import os
import sys
from typing import Dict, Any, Optional

# Try to import yaml, provide helpful message if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# =============================================================================
# Default Values Constants (硬编码默认值常量定义)
# =============================================================================
# 这些是程序层面的绝对兜底默认值，当不使用配置文件时会使用这些值
# 建议：尽量通过配置文件设置参数，这些默认值仅作为最后的保障

# Algorithm defaults
DEFAULT_ALG = 'fedavg'
DEFAULT_SERVER_AGG_METHOD = 'fedavg'
DEFAULT_GATE_PENALTY_TYPE = 'bilateral'

# Dataset defaults
DEFAULT_DATASET = 'mnist'  # 统一使用 mnist 作为默认数据集（最简单、最轻量）
DEFAULT_DATASET_FALLBACK = 'cifar100'  # flatten_config 中的兜底值（用于数据集特定配置提取，因为配置文件通常是针对 cifar100）
DEFAULT_NUM_CLASSES = 10
DEFAULT_IMAGE_SIZE = 32

# Federated Learning defaults
DEFAULT_EPOCHS = 10
DEFAULT_NUM_USERS = 100
DEFAULT_FRAC = 0.1
DEFAULT_LOCAL_EP = 10
DEFAULT_LOCAL_BS = 10
DEFAULT_DIRICHLET_ALPHA = 0.5

# Training defaults
DEFAULT_LR = 0.01
DEFAULT_MOMENTUM = 0.5
DEFAULT_OPTIMIZER = 'sgd'

# LoRA defaults
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_TRAIN_MLP_HEAD = 1

# FedSDG defaults
DEFAULT_LAMBDA1 = 1e-3
DEFAULT_LAMBDA2 = 1e-4
DEFAULT_LR_GATE = 1e-2
DEFAULT_GRAD_CLIP = 1.0

# Model defaults
DEFAULT_MODEL = 'vit'
DEFAULT_MODEL_VARIANT = 'pretrained'
DEFAULT_NUM_CHANNELS = 1
DEFAULT_NUM_FILTERS = 32
DEFAULT_KERNEL_NUM = 9
DEFAULT_KERNEL_SIZES = '3,4,5'
DEFAULT_NORM = 'batch_norm'
DEFAULT_MAX_POOL = 'True'

# Evaluation defaults
DEFAULT_TEST_FRAC = 0.2

# Checkpoint defaults
DEFAULT_ENABLE_CHECKPOINT = 1
DEFAULT_SAVE_FREQUENCY = 5
DEFAULT_SAVE_CLIENT_WEIGHTS = 1
DEFAULT_MAX_CHECKPOINTS = -1

# System defaults
DEFAULT_GPU = -1  # -1 means CPU
DEFAULT_SEED = 1
DEFAULT_VERBOSE = 1
DEFAULT_STOPPING_ROUNDS = 10
DEFAULT_UNEQUAL = 0
DEFAULT_USE_OFFLINE = False  # Default to online data loading (torchvision)
DEFAULT_OFFLINE_DATA_ROOT = './datasets/preprocessed'


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for config file support. "
            "Install with: pip install pyyaml"
        )
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        #  YAML 格式的数据转换为 Python 对象（如字典、列表、字符串等）的函数。
        # YAML禁止使用TAB 只能使用空格来对齐表示层级结构
        # 然后python会解析为嵌套字典
    return config


def flatten_config(config: dict[str, any], dataset: str = None) -> dict[str, Any]:
    """
    Flatten nested config structure to argument format.
    
    Args:
        config: Nested config dictionary
        dataset: Dataset name to use for dataset-specific settings
        
    Returns:
        Flat dictionary suitable for argparse

    功能描述：
        将嵌套的配置结构（即多层字典）展平（Flatten）为命令行参数风格的格式。
    Args (输入参数)：
        config: 嵌套的配置字典（比如你之前看到的包含 datasets: { cifar100: {...} } 的多层字典）。
        dataset: 数据集名称，用于提取该数据集特有的设置（例如指定为 cifar100，函数就会去 datasets 下面找对应的参数）。
    Returns (返回值)：
        返回一个扁平的字典，该字典的格式适合 argparse（Python 命令行解析工具）使用。
    """
    flat = {}
    
    # Algorithm
    if 'algorithm' in config:
        flat['alg'] = config['algorithm']
    
    # Dataset (can be overridden by command line)
    if dataset:
        flat['dataset'] = dataset
    elif 'dataset' in config:
        flat['dataset'] = config['dataset']
    
    # Get dataset-specific settings
    # 使用 DEFAULT_DATASET_FALLBACK 作为兜底值（用于从 datasets 嵌套结构中提取配置）
    dataset_name = flat.get('dataset', DEFAULT_DATASET_FALLBACK)
    if 'datasets' in config and dataset_name in config['datasets']:
        ds_config = config['datasets'][dataset_name]
        for key, value in ds_config.items():
            flat[key] = value
    
    # LoRA settings
    if 'lora' in config:
        lora = config['lora']
        if 'r' in lora:
            flat['lora_r'] = lora['r']
        if 'alpha' in lora:
            flat['lora_alpha'] = lora['alpha']
        if 'train_mlp_head' in lora:
            flat['lora_train_mlp_head'] = 1 if lora['train_mlp_head'] else 0
    
    # FedSDG settings
    if 'fedsdg' in config:
        sdg = config['fedsdg']
        if 'server_agg_method' in sdg:
            flat['server_agg_method'] = sdg['server_agg_method']
        if 'lambda1' in sdg:
            flat['lambda1'] = sdg['lambda1']
        if 'lambda2' in sdg:
            flat['lambda2'] = sdg['lambda2']
        if 'gate_penalty_type' in sdg:
            flat['gate_penalty_type'] = sdg['gate_penalty_type']
        if 'lr_gate' in sdg:
            flat['lr_gate'] = sdg['lr_gate']
        if 'grad_clip' in sdg:
            flat['grad_clip'] = sdg['grad_clip']
    
    # Federated settings
    if 'federated' in config:
        fed = config['federated']
        if 'num_users' in fed:
            flat['num_users'] = fed['num_users']
        if 'frac' in fed:
            flat['frac'] = fed['frac']
        if 'dirichlet_alpha' in fed:
            flat['dirichlet_alpha'] = fed['dirichlet_alpha']
    
    # Training settings
    if 'training' in config:
        train = config['training']
        if 'epochs' in train:
            flat['epochs'] = train['epochs']
        if 'local_ep' in train:
            flat['local_ep'] = train['local_ep']
        if 'local_bs' in train:
            flat['local_bs'] = train['local_bs']
        if 'lr' in train:
            flat['lr'] = train['lr']
        if 'optimizer' in train:
            flat['optimizer'] = train['optimizer']
        if 'momentum' in train:
            flat['momentum'] = train['momentum']
    
    # Evaluation settings
    if 'evaluation' in config:
        evl = config['evaluation']
        if 'test_frac' in evl:
            flat['test_frac'] = evl['test_frac']
    
    # System settings
    if 'system' in config:
        sys_cfg = config['system']
        if 'gpu' in sys_cfg:
            flat['gpu'] = sys_cfg['gpu']
        if 'seed' in sys_cfg:
            flat['seed'] = sys_cfg['seed']
        if 'verbose' in sys_cfg:
            flat['verbose'] = sys_cfg['verbose']
    
    # Checkpoint settings
    if 'checkpoint' in config:
        ckpt = config['checkpoint']
        if 'enable' in ckpt:
            flat['enable_checkpoint'] = 1 if ckpt['enable'] else 0
        if 'save_frequency' in ckpt:
            flat['save_frequency'] = ckpt['save_frequency']
        if 'save_client_weights' in ckpt:
            flat['save_client_weights'] = 1 if ckpt['save_client_weights'] else 0
    
    return flat


def args_parser():
    """
    Parse command line arguments with optional config file support.
    
    Returns:
        Parsed arguments namespace
    """
    # First pass: check for --config argument
    # 先检查是否是使用配置文件进行启动的
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None,
                            help='Path to YAML config file (e.g., configs/fedsdg.yaml)')
    pre_args, remaining = pre_parser.parse_known_args()
    
    # Load config file if specified
    # 加载配置文件
    config_defaults = {}
    if pre_args.config:
        try:
            config = load_config(pre_args.config)
            # Check if dataset is specified in remaining args
            temp_parser = argparse.ArgumentParser(add_help=False)
            temp_parser.add_argument('--dataset', type=str, default=None)
            temp_args, _ = temp_parser.parse_known_args(remaining)
            
            config_defaults = flatten_config(config, dataset=temp_args.dataset)
            print(f"[Config] Loaded configuration from: {pre_args.config}")
        except Exception as e:
            print(f"[Config] Warning: Failed to load config file: {e}")
    
    # Main parser
    parser = argparse.ArgumentParser(
        description='Federated Learning Experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file argument
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')

    # ==========================================================================
    # Federated Learning Arguments
    # ==========================================================================
    parser.add_argument('--epochs', type=int, 
                        default=config_defaults.get('epochs', DEFAULT_EPOCHS),
                        help="Number of global communication rounds")
    parser.add_argument('--num_users', type=int, 
                        default=config_defaults.get('num_users', DEFAULT_NUM_USERS),
                        help="Total number of clients")
    parser.add_argument('--frac', type=float, 
                        default=config_defaults.get('frac', DEFAULT_FRAC),
                        help='Fraction of clients per round')
    parser.add_argument('--local_ep', type=int, 
                        default=config_defaults.get('local_ep', DEFAULT_LOCAL_EP),
                        help="Number of local epochs per round")
    parser.add_argument('--local_bs', type=int, 
                        default=config_defaults.get('local_bs', DEFAULT_LOCAL_BS),
                        help="Local batch size")
    parser.add_argument('--lr', type=float, 
                        default=config_defaults.get('lr', DEFAULT_LR),
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, 
                        default=config_defaults.get('momentum', DEFAULT_MOMENTUM),
                        help='SGD momentum')
    
    # ==========================================================================
    # Algorithm Selection
    # ==========================================================================
    parser.add_argument('--alg', type=str, 
                        default=config_defaults.get('alg', DEFAULT_ALG),
                        choices=['fedavg', 'fedlora', 'fedsdg'],
                        help='Federated learning algorithm')
    
    # ==========================================================================
    # LoRA Arguments (FedLoRA/FedSDG)
    # ==========================================================================
    parser.add_argument('--lora_r', type=int, 
                        default=config_defaults.get('lora_r', DEFAULT_LORA_R),
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, 
                        default=config_defaults.get('lora_alpha', DEFAULT_LORA_ALPHA),
                        help='LoRA scaling factor')
    parser.add_argument('--lora_train_mlp_head', type=int, 
                        default=config_defaults.get('lora_train_mlp_head', DEFAULT_LORA_TRAIN_MLP_HEAD),
                        help='Train classification head (1=True, 0=False)')
    
    # ==========================================================================
    # FedSDG Arguments
    # ==========================================================================
    parser.add_argument('--server_agg_method', type=str, 
                        default=config_defaults.get('server_agg_method', DEFAULT_SERVER_AGG_METHOD),
                        choices=['fedavg', 'alignment'],
                        help='FedSDG server aggregation method')
    parser.add_argument('--lambda1', type=float, 
                        default=config_defaults.get('lambda1', DEFAULT_LAMBDA1),
                        help='FedSDG: Gate sparsity penalty (L1)')
    parser.add_argument('--lambda2', type=float, 
                        default=config_defaults.get('lambda2', DEFAULT_LAMBDA2),
                        help='FedSDG: Private parameter regularization (L2)')
    parser.add_argument('--gate_penalty_type', type=str, 
                        default=config_defaults.get('gate_penalty_type', DEFAULT_GATE_PENALTY_TYPE),
                        choices=['unilateral', 'bilateral'],
                        help='FedSDG: Gate penalty type')
    parser.add_argument('--lr_gate', type=float, 
                        default=config_defaults.get('lr_gate', DEFAULT_LR_GATE),
                        help='FedSDG: Gate parameter learning rate')
    parser.add_argument('--grad_clip', type=float, 
                        default=config_defaults.get('grad_clip', DEFAULT_GRAD_CLIP),
                        help='FedSDG: Gradient clipping norm (0=disabled)')
    
    # ==========================================================================
    # Evaluation Arguments
    # ==========================================================================
    parser.add_argument('--test_frac', type=float, 
                        default=config_defaults.get('test_frac', DEFAULT_TEST_FRAC),
                        help='Fraction of clients for local evaluation')
    
    # ==========================================================================
    # Checkpoint Arguments
    # ==========================================================================
    parser.add_argument('--enable_checkpoint', type=int, 
                        default=config_defaults.get('enable_checkpoint', DEFAULT_ENABLE_CHECKPOINT),
                        help='Enable checkpoint saving (1=True, 0=False)')
    parser.add_argument('--save_frequency', type=int, 
                        default=config_defaults.get('save_frequency', DEFAULT_SAVE_FREQUENCY),
                        help='Checkpoint save frequency')
    parser.add_argument('--save_client_weights', type=int, 
                        default=config_defaults.get('save_client_weights', DEFAULT_SAVE_CLIENT_WEIGHTS),
                        help='Save client weights (1=True, 0=False)')
    parser.add_argument('--max_checkpoints', type=int, default=DEFAULT_MAX_CHECKPOINTS,
                        help='Max checkpoints to keep (-1=unlimited)')

    # ==========================================================================
    # Model Arguments
    # ==========================================================================
    parser.add_argument('--model', type=str, 
                        default=config_defaults.get('model', DEFAULT_MODEL),
                        choices=['mlp', 'cnn', 'vit'],
                        help='Model architecture')
    parser.add_argument('--model_variant', type=str, 
                        default=config_defaults.get('model_variant', DEFAULT_MODEL_VARIANT),
                        choices=['scratch', 'pretrained'],
                        help='Model variant')
    parser.add_argument('--image_size', type=int, 
                        default=config_defaults.get('image_size', DEFAULT_IMAGE_SIZE),
                        help='Input image size')
    
    # ==========================================================================
    # Offline Data Arguments
    # ==========================================================================
    use_offline_default = config_defaults.get('use_offline', DEFAULT_USE_OFFLINE)
    parser.add_argument('--use_offline_data', action='store_true',
                        default=use_offline_default,
                        help='Use offline preprocessed data')
    parser.add_argument('--offline_data_root', type=str, 
                        default=DEFAULT_OFFLINE_DATA_ROOT,
                        help='Offline data directory')
    
    # ==========================================================================
    # Other Model Arguments
    # ==========================================================================
    parser.add_argument('--kernel_num', type=int, default=DEFAULT_KERNEL_NUM)
    parser.add_argument('--kernel_sizes', type=str, default=DEFAULT_KERNEL_SIZES)
    parser.add_argument('--num_channels', type=int, 
                        default=config_defaults.get('num_channels', DEFAULT_NUM_CHANNELS))
    parser.add_argument('--norm', type=str, default=DEFAULT_NORM)
    parser.add_argument('--num_filters', type=int, default=DEFAULT_NUM_FILTERS)
    parser.add_argument('--max_pool', type=str, default=DEFAULT_MAX_POOL)

    # ==========================================================================
    # Dataset & System Arguments
    # ==========================================================================
    parser.add_argument('--dataset', type=str, 
                        default=config_defaults.get('dataset', DEFAULT_DATASET),
                        choices=['mnist', 'fmnist', 'cifar', 'cifar10', 'cifar100'],
                        help="Dataset name")
    parser.add_argument('--num_classes', type=int, 
                        default=config_defaults.get('num_classes', DEFAULT_NUM_CLASSES),
                        help="Number of classes")
    parser.add_argument('--gpu', type=int, 
                        default=config_defaults.get('gpu', DEFAULT_GPU),
                        help="GPU ID (-1 for CPU)")
    parser.add_argument('--optimizer', type=str, 
                        default=config_defaults.get('optimizer', DEFAULT_OPTIMIZER),
                        help="Optimizer type")
    parser.add_argument('--dirichlet_alpha', type=float, 
                        default=config_defaults.get('dirichlet_alpha', DEFAULT_DIRICHLET_ALPHA),
                        help='Dirichlet alpha for Non-IID')
    parser.add_argument('--unequal', type=int, default=DEFAULT_UNEQUAL,
                        help='Unequal data splits')
    parser.add_argument('--stopping_rounds', type=int, default=DEFAULT_STOPPING_ROUNDS,
                        help='Early stopping rounds')
    parser.add_argument('--verbose', type=int, 
                        default=config_defaults.get('verbose', DEFAULT_VERBOSE),
                        help='Verbosity level')
    parser.add_argument('--seed', type=int, 
                        default=config_defaults.get('seed', DEFAULT_SEED),
                        help='Random seed')
    
    args = parser.parse_args()
    
    # ==========================================================================
    # Validation
    # ==========================================================================
    if args.alg in ('fedlora', 'fedsdg') and args.model != 'vit':
        raise ValueError(
            f"{args.alg.upper()} requires ViT model. "
            f"Use --model vit or switch to --alg fedavg"
        )
    
    if args.model_variant == 'pretrained' and args.model != 'vit':
        raise ValueError("Pretrained variant only supports ViT")
    
    if args.model_variant == 'pretrained' and args.image_size == 32:
        print("[Warning] Pretrained ViT with image_size=32. Consider --image_size 224")
    
    # ==========================================================================
    # Auto-configure num_classes
    # ==========================================================================
    if args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset in ('cifar', 'cifar10'):
        args.num_classes = 10
        if args.dataset == 'cifar10':
            args.dataset = 'cifar'
    elif args.dataset in ('mnist', 'fmnist'):
        args.num_classes = 10
    
    # Print config summary if using config file
    if pre_args.config:
        print(f"[Config] Algorithm: {args.alg}")
        print(f"[Config] Dataset: {args.dataset} ({args.num_classes} classes)")
        print(f"[Config] Model: {args.model} ({args.model_variant})")
        if args.alg in ('fedlora', 'fedsdg'):
            print(f"[Config] LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        if args.alg == 'fedsdg':
            print(f"[Config] FedSDG: λ1={args.lambda1}, λ2={args.lambda2}, "
                  f"agg={args.server_agg_method}")
    
    return args
