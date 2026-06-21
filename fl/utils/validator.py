# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数验证模块

提供配置参数和数据集的验证功能，将验证逻辑从 main.py 中解耦。

Usage:
    from fl.utils import validate_args, validate_dataset
    
    # 验证配置
    validate_args(args)
    
    # 验证数据集
    validate_dataset(train_dataset, test_dataset, user_groups, user_groups_test)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch


class ValidationError(ValueError):
    """验证错误异常"""
    pass


def validate_args(args) -> None:
    """
    验证配置参数的有效性
    
    Args:
        args: 配置对象（ConfigAdapter 或 argparse Namespace）
        
    Raises:
        ValidationError: 参数无效时抛出
    """
    # 基本参数验证
    _validate_positive_int(args.epochs, "epochs", "训练轮次")
    _validate_positive_int(args.num_users, "num_users", "客户端数量")
    _validate_positive_int(args.local_ep, "local_ep", "本地训练轮次")
    _validate_positive_int(args.local_bs, "local_bs", "本地批次大小")
    _validate_positive_float(args.lr, "lr", "学习率")
    _validate_range(args.frac, "frac", "客户端参与率", 0, 1, include_upper=True)
    
    # 枚举验证
    _validate_choice(args.dataset, "dataset", 
                     ['mnist', 'fmnist', 'cifar', 'cifar10', 'cifar100', 'femnist', 'tinyimagenet', 'domainnet', 'mini_domainnet', 'officehome', 'stanford_cars', 'imagenet_r','pathmnist'])
    _validate_choice(args.model, "model", ['mlp', 'cnn', 'vit'])
    _validate_choice(args.alg, "alg", 
                     ['fedavg', 'fedprox_avg', 'fedlora', 'fedprox_lora', 'fedsdg', 'feddpa', 'fedsalora', 'pf2lora', 'fedtp', 'lorafair', 'fedalt', 'local_only', 'fedrep', 'ditto'])
    
    # 算法特定验证
    if args.alg in ('fedlora', 'fedprox_lora', 'fedsdg', 'feddpa', 'fedsalora', 'pf2lora', 'fedtp', 'lorafair', 'fedalt', 'local_only', 'fedrep', 'ditto'):
        _validate_lora_args(args)
    
    if args.alg == 'fedrep':
        _validate_fedrep_args(args)
    
    if args.alg == 'ditto':
        _validate_ditto_args(args)
    
    if args.alg == 'fedsdg':
        _validate_fedsdg_args(args)
    
    if args.alg == 'feddpa':
        _validate_feddpa_args(args)
    
    if args.alg == 'fedtp':
        _validate_fedtp_args(args)
    
    if args.alg in ('fedprox_avg', 'fedprox_lora'):
        _validate_fedprox_args(args)
    
    if args.alg == 'lorafair':
        _validate_lorafair_args(args)
    
    if args.alg == 'fedalt':
        _validate_fedalt_args(args)
    
    # ViT 类型验证
    if args.model == 'vit':
        vit_type = getattr(args, 'vit_type', 'tiny')
        _validate_choice(vit_type, "vit_type", ['tiny', 'base'])
    
    # 预训练模型验证
    if getattr(args, 'model_variant', None) == 'pretrained':
        _validate_pretrained_args(args)
    
    # Dirichlet alpha 验证
    _validate_positive_float(args.dirichlet_alpha, "dirichlet_alpha", "Dirichlet alpha")
    
    # GPU 验证
    _validate_gpu(args)


def validate_dataset(
    train_dataset, 
    val_dataset,
    test_dataset, 
    user_groups_train: Dict[int, Any], 
    user_groups_val: Dict[int, Any],
    user_groups_test: Dict[int, Any]
) -> None:
    """
    验证数据集和用户分组的有效性（Train/Val/Test三划分）
    
    Args:
        train_dataset: 训练数据集（官方train的80%）
        val_dataset: 验证数据集（官方train的20%，与train_dataset是同一对象）
        test_dataset: 测试数据集（官方test）
        user_groups_train: 训练数据用户分组 {user_id: indices}
        user_groups_val: 验证数据用户分组 {user_id: indices}
        user_groups_test: 测试数据用户分组 {user_id: indices}
        
    Raises:
        ValidationError: 数据无效时抛出
    """
    if len(train_dataset) == 0:
        raise ValidationError("训练数据集为空")
    
    if len(test_dataset) == 0:
        raise ValidationError("测试数据集为空")
    
    if len(user_groups_train) == 0:
        raise ValidationError("训练用户分组为空")
    
    if len(user_groups_val) == 0:
        raise ValidationError("验证用户分组为空")
    
    if len(user_groups_test) == 0:
        raise ValidationError("测试用户分组为空")
    
    # 检查空用户（训练集）
    empty_users = [uid for uid, idxs in user_groups_train.items() if len(idxs) == 0]
    if empty_users:
        from .console_logger import cprint
        cprint(f"[警告] 以下用户没有训练数据: {empty_users[:10]}...")
    
    # 打印数据集信息
    _print_dataset_info(train_dataset, val_dataset, test_dataset, 
                        user_groups_train, user_groups_val, user_groups_test)


def validate_model_for_algorithm(model_name: str, alg: str) -> None:
    """
    验证模型与算法的兼容性
    
    Args:
        model_name: 模型名称
        alg: 算法名称
        
    Raises:
        ValidationError: 不兼容时抛出
    """
    if alg in ('fedlora', 'fedprox_lora', 'fedsdg', 'feddpa', 'fedsalora', 'pf2lora', 'fedtp', 'lorafair', 'local_only', 'fedrep', 'ditto') and model_name != 'vit':
        raise ValidationError(
            f"{alg.upper()} 目前仅支持 ViT 模型，但当前模型为 '{model_name}'。"
            f"请使用 model=vit 或切换到 algorithm=fedavg"
        )


# =============================================================================
# 私有辅助函数
# =============================================================================

def _validate_positive_int(value: int, name: str, desc: str) -> None:
    """验证正整数"""
    if not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{desc} ({name}) 必须为正整数，当前值: {value}")


def _validate_positive_float(value: float, name: str, desc: str) -> None:
    """验证正数"""
    if value <= 0:
        raise ValidationError(f"{desc} ({name}) 必须为正数，当前值: {value}")


def _validate_non_negative(value: float, name: str, desc: str) -> None:
    """验证非负数"""
    if value < 0:
        raise ValidationError(f"{desc} ({name}) 必须为非负数，当前值: {value}")


def _validate_range(
    value: float, 
    name: str, 
    desc: str, 
    lower: float, 
    upper: float, 
    include_upper: bool = False
) -> None:
    """验证范围"""
    if include_upper:
        if not (lower < value <= upper):
            raise ValidationError(
                f"{desc} ({name}) 必须在 ({lower}, {upper}] 范围内，当前值: {value}"
            )
    else:
        if not (lower < value < upper):
            raise ValidationError(
                f"{desc} ({name}) 必须在 ({lower}, {upper}) 范围内，当前值: {value}"
            )


def _validate_choice(value: Any, name: str, choices: List[Any]) -> None:
    """验证枚举值"""
    if value not in choices:
        raise ValidationError(
            f"不支持的 {name}: {value}，有效选项: {choices}"
        )


def _validate_lora_args(args) -> None:
    """验证 LoRA 相关参数"""
    if args.model != 'vit':
        raise ValidationError(
            f"{args.alg.upper()} 目前仅支持 ViT 模型，但当前模型为 '{args.model}'。"
            f"请使用 model=vit 或切换到 algorithm=fedavg"
        )
    
    lora_r = getattr(args, 'lora_r', None)
    lora_alpha = getattr(args, 'lora_alpha', None)
    
    if lora_r is not None and lora_r <= 0:
        raise ValidationError(f"LoRA 秩 (lora_r) 必须为正整数，当前值: {lora_r}")
    
    if lora_alpha is not None and lora_alpha <= 0:
        raise ValidationError(f"LoRA Alpha (lora_alpha) 必须为正整数，当前值: {lora_alpha}")


def _validate_fedsdg_args(args) -> None:
    """验证 FedSDG 特定参数"""
    # 门控粒度验证
    gate_granularity = getattr(args, 'gate_granularity', 'fine')
    if gate_granularity not in ['fine', 'coarse']:
        raise ValidationError(f"gate_granularity must be 'fine' or 'coarse', got {gate_granularity}")
    
    # 固定门控值范围验证
    fix_gate = getattr(args, 'fix_gate', False)
    if fix_gate:
        fixed_gate_value = getattr(args, 'fixed_gate_value', 0.5)
        if not (0.0 <= fixed_gate_value <= 1.0):
            raise ValidationError(f"fixed_gate_value must be in [0.0, 1.0], got {fixed_gate_value}")
    
    # 正则化参数验证
    lambda1 = getattr(args, 'lambda1', 0)
    lambda2 = getattr(args, 'lambda2', 0)
    server_agg_method = getattr(args, 'server_agg_method', 'fedavg')
    
    if lambda1 < 0:
        raise ValidationError(f"lambda1 must be non-negative, current value: {lambda1}")
    
    if lambda2 < 0:
        raise ValidationError(f"lambda2 must be non-negative, current value: {lambda2}")
    
    valid_agg_methods = ['fedavg', 'alignment']
    if server_agg_method not in valid_agg_methods:
        raise ValidationError(f"server_agg_method must be one of {valid_agg_methods}, got {server_agg_method}")


def _validate_feddpa_args(args) -> None:
    """验证 FedDPA 特定参数"""
    train_mix_ratio = getattr(args, 'train_mix_ratio', 0.5)
    inference_scale_factor = getattr(args, 'inference_scale_factor', 0.5)
    anchor_count = getattr(args, 'anchor_count', 5)
    server_agg_method = getattr(args, 'server_agg_method', 'fedavg')
    
    if not (0.0 <= train_mix_ratio <= 1.0):
        raise ValidationError(f"train_mix_ratio 必须在 [0, 1] 范围内，当前值: {train_mix_ratio}")
    
    if not (0.0 <= inference_scale_factor <= 1.0):
        raise ValidationError(f"inference_scale_factor 必须在 [0, 1] 范围内，当前值: {inference_scale_factor}")
    
    if anchor_count <= 0:
        raise ValidationError(f"anchor_count 必须为正整数，当前值: {anchor_count}")
    
    valid_agg_methods = ['fedavg', 'alignment']
    if server_agg_method not in valid_agg_methods:
        raise ValidationError(
            f"不支持的聚合方法: {server_agg_method}，有效选项: {valid_agg_methods}"
        )


def _validate_fedalt_args(args) -> None:
    """验证 FedALT-adapted 特定参数"""
    # 门控粒度验证
    gate_granularity = getattr(args, 'gate_granularity', 'fine')
    if gate_granularity not in ['fine', 'coarse']:
        raise ValidationError(f"gate_granularity must be 'fine' or 'coarse', got {gate_granularity}")
    
    # 固定门控值范围验证
    fix_gate = getattr(args, 'fix_gate', False)
    if fix_gate:
        fixed_gate_value = getattr(args, 'fixed_gate_value', 0.5)
        if not (0.0 <= fixed_gate_value <= 1.0):
            raise ValidationError(f"fixed_gate_value must be in [0.0, 1.0], got {fixed_gate_value}")
    
    # 正则化参数验证
    lambda1 = getattr(args, 'lambda1', 0)
    lambda2 = getattr(args, 'lambda2', 0)
    server_agg_method = getattr(args, 'server_agg_method', 'fedavg')
    
    if lambda1 < 0:
        raise ValidationError(f"FedALT lambda1 must be non-negative, current value: {lambda1}")
    
    if lambda2 < 0:
        raise ValidationError(f"FedALT lambda2 must be non-negative, current value: {lambda2}")
    
    valid_agg_methods = ['fedavg', 'alignment']
    if server_agg_method not in valid_agg_methods:
        raise ValidationError(
            f"不支持的聚合方法: {server_agg_method}，有效选项: {valid_agg_methods}"
        )
    
    # lr_gate 验证
    lr_gate = getattr(args, 'lr_gate', 0.1)
    if lr_gate <= 0:
        raise ValidationError(f"FedALT lr_gate must be positive, current value: {lr_gate}")


def _validate_fedtp_args(args) -> None:
    """验证 FedTP (Two-Phase LoRA) 特定参数"""
    phase1_epochs = getattr(args, 'phase1_epochs', 50)
    total_epochs = getattr(args, 'epochs', 80)
    server_agg_method = getattr(args, 'server_agg_method', 'fedavg')
    
    if not isinstance(phase1_epochs, int) or phase1_epochs <= 0:
        raise ValidationError(f"FedTP phase1_epochs 必须为正整数，当前值: {phase1_epochs}")
    
    if phase1_epochs >= total_epochs:
        raise ValidationError(
            f"FedTP phase1_epochs ({phase1_epochs}) 必须小于总训练轮次 epochs ({total_epochs})。"
            f"Phase 2 至少需要 1 轮用于本地微调。"
        )
    
    valid_agg_methods = ['fedavg', 'alignment']
    if server_agg_method not in valid_agg_methods:
        raise ValidationError(
            f"不支持的聚合方法: {server_agg_method}，有效选项: {valid_agg_methods}"
        )


def _validate_fedprox_args(args) -> None:
    """验证 FedProx 特定参数"""
    mu = getattr(args, 'mu', 0)
    
    if mu < 0:
        raise ValidationError(f"FedProx mu 必须为非负数，当前值: {mu}")


def _validate_lorafair_args(args) -> None:
    """验证 LoRA-FAIR 特定参数"""
    residual_mu = getattr(args, 'residual_mu', 0.1)
    server_agg_method = getattr(args, 'server_agg_method', 'fedavg')
    
    if residual_mu <= 0:
        raise ValidationError(f"LoRA-FAIR residual_mu 必须为正数，当前值: {residual_mu}")
    
    valid_agg_methods = ['fedavg', 'alignment']
    if server_agg_method not in valid_agg_methods:
        raise ValidationError(
            f"不支持的聚合方法: {server_agg_method}，有效选项: {valid_agg_methods}"
        )


def _validate_fedrep_args(args) -> None:
    """验证 FedRep 特定参数"""
    rep_epochs = getattr(args, 'fedrep_rep_epochs', 1)
    head_epochs = getattr(args, 'fedrep_head_epochs', 5)
    
    if rep_epochs <= 0:
        raise ValidationError(f"FedRep rep_epochs 必须为正整数，当前值: {rep_epochs}")
    
    if head_epochs <= 0:
        raise ValidationError(f"FedRep head_epochs 必须为正整数，当前值: {head_epochs}")


def _validate_ditto_args(args) -> None:
    """验证 Ditto 特定参数"""
    lambda_ditto = getattr(args, 'lambda_ditto', 0.1)
    ditto_reg_target = getattr(args, 'ditto_reg_target', 'server')
    
    if lambda_ditto < 0:
        raise ValidationError(f"Ditto lambda_ditto 必须为非负数，当前值: {lambda_ditto}")
    
    if ditto_reg_target not in ('server', 'local'):
        raise ValidationError(
            f"Ditto ditto_reg_target 必须为 'server' 或 'local'，当前值: '{ditto_reg_target}'"
        )


def _validate_pretrained_args(args) -> None:
    """验证预训练模型参数"""
    if args.model != 'vit':
        raise ValidationError("预训练模型 (pretrained) 目前仅支持 ViT 模型")
    
    image_size = getattr(args, 'image_size', 224)
    if image_size < 32:
        raise ValidationError(f"图像尺寸 (image_size) 必须 >= 32，当前值: {image_size}")
    
    if image_size == 32:
        print("[警告] 预训练 ViT 使用 image_size=32 可能效果不佳，建议使用 --image_size 224")


def _validate_gpu(args) -> None:
    """验证 GPU 设置"""
    gpu = getattr(args, 'gpu', None)
    
    if gpu is not None and gpu >= 0:
        if not torch.cuda.is_available():
            print(f"[警告] 指定了 GPU {gpu}，但 CUDA 不可用，将使用 CPU")
        elif gpu >= torch.cuda.device_count():
            raise ValidationError(
                f"指定的 GPU {gpu} 不存在，可用 GPU 数量: {torch.cuda.device_count()}"
            )


def _print_dataset_info(
    train_dataset, val_dataset, test_dataset, 
    user_groups_train: Dict, user_groups_val: Dict, user_groups_test: Dict
) -> None:
    """打印数据集信息（Train/Val/Test三划分）"""
    from .console_logger import cprint
    
    try:
        sample_data, sample_label = train_dataset[0]
        sample_shape = sample_data.shape
    except Exception:
        sample_shape = "unknown"
    
    train_counts = [len(idxs) for idxs in user_groups_train.values()]
    val_counts = [len(idxs) for idxs in user_groups_val.values()]
    test_counts = [len(idxs) for idxs in user_groups_test.values()]
    
    cprint(f"\n[数据集信息]")
    cprint(f"  样本形状: {sample_shape}")
    cprint(f"  用户数量: {len(user_groups_train)}")
    cprint(f"  Train 样本: total={sum(train_counts)}, per_user: avg={np.mean(train_counts):.1f}, "
           f"min={min(train_counts)}, max={max(train_counts)}")
    cprint(f"  Val 样本: total={sum(val_counts)}, per_user: avg={np.mean(val_counts):.1f}, "
           f"min={min(val_counts)}, max={max(val_counts)}")
    cprint(f"  Test 样本: total={sum(test_counts)}, per_user: avg={np.mean(test_counts):.1f}, "
           f"min={min(test_counts)}, max={max(test_counts)}")
