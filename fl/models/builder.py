# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型构建工厂 - 统一的模型创建接口

提供模型构建和 LoRA 注入的工厂函数，将模型创建逻辑从 main.py 中解耦。

Usage:
    from fl.models import get_model
    
    # 一站式获取模型（包含 LoRA 注入）
    model = get_model(args, train_dataset, device)
    
    # 或分步操作
    model = build_model(args, train_dataset, device)
    if args.alg in ('fedlora', 'fedsdg'):
        model = inject_lora_to_model(model, args)
"""

from typing import Optional
import torch
import torch.nn as nn

from .mlp import MLP
from .cnn import CNNMnist, CNNFashion_Mnist, CNNCifar
from .vit import ViT, get_pretrained_vit
from .lora import inject_lora, inject_lora_timm, GlobalGateModule
from .lora_dpa import inject_lora_dpa, inject_lora_dpa_timm


class ModelBuildError(ValueError):
    """模型构建错误"""
    pass


def get_model_device(model):
    """安全获取模型设备，处理边界情况"""
    try:
        device = next(model.parameters()).device
        # 处理设备字符串格式不一致的问题
        if isinstance(device, str):
            return torch.device(device)
        return device
    except StopIteration:
        # 模型没有参数，使用CPU作为默认设备
        return torch.device('cpu')


def inject_lora_with_gate_granularity(model, args, model_variant, gate_granularity='fine'):
    """支持门控粒度的LoRA注入 - 完善设备处理"""
    from fl.utils.console_logger import cprint
    
    is_fedsdg = getattr(args, 'alg', '') == 'fedsdg'
    
    if is_fedsdg and gate_granularity == 'coarse':
        # 获取门控参数
        fix_gate = getattr(args, 'fix_gate', False)
        gate_init_value = getattr(args, 'gate_init_value', 0.0)
        fixed_gate_value = getattr(args, 'fixed_gate_value', 0.5)
        
        # 安全获取设备
        device = get_model_device(model)
        
        # 创建全局门控模块
        if fix_gate:
            # 固定模式：将 fixed_gate_value 转为 logit 值
            import math
            if fixed_gate_value <= 0:
                init_logit = -100.0
            elif fixed_gate_value >= 1:
                init_logit = 100.0
            else:
                init_logit = math.log(fixed_gate_value / (1 - fixed_gate_value))
            gate_module = GlobalGateModule(init_logit)
            # 冻结门控参数
            gate_module.lambda_k_global.requires_grad = False
            cprint(f"[FedSDG-Coarse] 固定门控模式: fixed_gate_value={fixed_gate_value} → logit={init_logit:.4f}")
        else:
            gate_module = GlobalGateModule(gate_init_value)
        gate_module.to(device)
        
        # 使用唯一名称避免冲突
        gate_module_name = 'fedsdg_global_gate'
        
        # 检查并处理命名冲突
        if hasattr(model, gate_module_name):
            delattr(model, gate_module_name)
        
        # 注册为子模块
        model.add_module(gate_module_name, gate_module)
        
        # 为所有LoRA层设置全局门控引用
        lora_layers = []
        for module in model.modules():
            if hasattr(module, 'lambda_k_logit') or hasattr(module, 'global_gate_ref'):
                # 这是一个LoRALayer
                if hasattr(module, 'set_global_gate'):
                    module.set_global_gate(gate_module.lambda_k_global)
                    lora_layers.append(module)
        
        # 保存模块名称供后续使用
        model._fedsdg_global_gate_name = gate_module_name
        
        cprint(f"[FedSDG-Coarse] 注入全局门控模块，覆盖 {len(lora_layers)} 个LoRA层，设备: {device}")
    
    return model


def build_model(args, train_dataset, device: str) -> nn.Module:
    """
    根据配置构建模型
    
    Args:
        args: 配置对象（需要包含 model, dataset, num_classes 等属性）
        train_dataset: 训练数据集（用于获取输入维度）
        device: 计算设备 ('cuda' 或 'cpu')
        
    Returns:
        构建好的模型（已移动到目标设备，处于训练模式）
        
    Raises:
        ModelBuildError: 模型类型无效或构建失败时抛出
    """
    model: Optional[nn.Module] = None
    
    if args.model == 'cnn':
        model = _build_cnn(args)
    elif args.model == 'vit':
        model = _build_vit(args, train_dataset)
    elif args.model == 'mlp':
        model = _build_mlp(args, train_dataset)
    else:
        raise ModelBuildError(f"不支持的模型类型: {args.model}，有效选项: ['cnn', 'vit', 'mlp']")
    
    if model is None:
        raise ModelBuildError("模型构建失败")
    
    # 移动到设备并设置为训练模式
    model.to(device)
    model.train()
    
    return model


def _build_cnn(args) -> nn.Module:
    """
    构建 CNN 模型
    
    根据数据集类型选择对应的 CNN 架构。
    """
    dataset_to_model = {
        'mnist': CNNMnist,
        'fmnist': CNNFashion_Mnist,
        'cifar': CNNCifar,
        'cifar10': CNNCifar,
        'cifar100': CNNCifar,
        'tinyimagenet': CNNCifar,  # 复用 CIFAR CNN 架构
    }
    
    if args.dataset not in dataset_to_model:
        raise ModelBuildError(
            f"CNN 不支持数据集: {args.dataset}，"
            f"有效选项: {list(dataset_to_model.keys())}"
        )
    
    return dataset_to_model[args.dataset](args=args)


def _build_vit(args, train_dataset) -> nn.Module:
    """
    构建 ViT 模型
    
    支持两种模式：
    - pretrained: 使用预训练权重（timm）
    - scratch: 从零训练（手写 ViT）
    """
    model_variant = getattr(args, 'model_variant', 'scratch')
    
    if model_variant == 'pretrained':
        # 预训练 ViT（使用 timm，自动管理权重缓存）
        vit_type = getattr(args, 'vit_type', 'tiny')
        return get_pretrained_vit(
            num_classes=args.num_classes,
            image_size=getattr(args, 'image_size', 224),
            vit_type=vit_type
        )
    else:
        # 从零训练的 ViT
        try:
            sample_data = train_dataset[0][0]
            img_size = sample_data.shape[-1]
            channels = sample_data.shape[0]
        except (IndexError, AttributeError) as e:
            raise ModelBuildError(f"无法从数据集获取图像尺寸: {e}")
        
        return ViT(
            image_size=img_size,
            patch_size=4,
            num_classes=args.num_classes,
            dim=128,
            depth=6,
            heads=8,
            mlp_dim=256,
            channels=channels,
        )


def _build_mlp(args, train_dataset) -> nn.Module:
    """
    构建 MLP 模型
    
    自动计算输入维度。
    """
    try:
        sample_data = train_dataset[0][0]
        img_size = sample_data.shape
        len_in = 1
        for x in img_size:
            len_in *= x
    except (IndexError, AttributeError) as e:
        raise ModelBuildError(f"无法从数据集获取输入维度: {e}")
    
    return MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)


def inject_lora_to_model(model: nn.Module, args) -> nn.Module:
    """
    为模型注入 LoRA 层
    
    根据算法类型（FedLoRA、FedSDG 或 FedDPA）和模型变体（pretrained 或 scratch）
    选择合适的 LoRA 注入方式。
    
    Args:
        model: 原始模型
        args: 配置对象（需要包含 alg, lora_r, lora_alpha 等属性）
            - lora_r: LoRA 秩
            - lora_alpha: LoRA 缩放因子
            - lora_dropout: LoRA Dropout 概率（可选，默认 0.0）
            - lora_train_mlp_head: 是否训练分类头（可选，默认 True）
        
    Returns:
        注入 LoRA 后的模型
    """
    is_fedsdg = (args.alg in ('fedsdg', 'fedtp', 'pf2lora', 'fedalt'))
    is_feddpa = (args.alg == 'feddpa')
    
    # 确定算法名称
    if args.alg == 'fedsdg':
        alg_name = "FedSDG"
    elif args.alg == 'fedtp':
        alg_name = "FedTP"
    elif is_feddpa:
        alg_name = "FedDPA"
    elif args.alg == 'fedprox_lora':
        alg_name = "FedProx+LoRA"
    elif args.alg == 'local_only':
        alg_name = "Local-Only LoRA"
    elif args.alg == 'fedrep':
        alg_name = "FedRep"
    elif args.alg == 'ditto':
        alg_name = "Ditto"
    elif args.alg == 'fedsalora':
        alg_name = "FedSA-LoRA"
    elif args.alg == 'pf2lora':
        alg_name = "PF2LoRA"
    elif args.alg == 'lorafair':
        alg_name = "LoRA-FAIR"
    elif args.alg == 'fedalt':
        alg_name = "FedALT"
    else:
        alg_name = "FedLoRA"
    
    model_variant = getattr(args, 'model_variant', 'scratch')
    lora_dropout = getattr(args, 'lora_dropout', 0.0)
    lora_r_private = getattr(args, 'lora_r_private', None)  # 私有分支独立秩（None 则等于 lora_r）
    
    # FedDPA 特有参数
    if is_feddpa:
        train_mix_ratio = getattr(args, 'train_mix_ratio', 0.5)
        return _inject_lora_dpa(model, args, model_variant, lora_dropout, train_mix_ratio)
    
    # FedSDG / FedTP 门控配置
    # FedTP 复用 FedSDG 的双路 LoRA 架构，但始终固定门控
    fix_gate = getattr(args, 'fix_gate', False) if is_fedsdg else False
    fixed_gate_value = getattr(args, 'fixed_gate_value', 0.5) if is_fedsdg else 0.5
    gate_init_value = getattr(args, 'gate_init_value', 0.0) if is_fedsdg else 0.0
    gate_granularity = getattr(args, 'gate_granularity', 'fine') if is_fedsdg else 'fine'
    # FedTP 不使用 Dynamic Alignment
    use_dynamic_alignment = getattr(args, 'use_dynamic_alignment', False) if (is_fedsdg and args.alg not in ('fedtp', 'pf2lora')) else False
    da_floor_gamma = getattr(args, 'da_floor_gamma', 0.1) if is_fedsdg else 0.1
    da_target_mode = getattr(args, 'da_target_mode', 'floor') if is_fedsdg else 'floor'
    use_da_scale = getattr(args, 'use_da_scale', False) if (is_fedsdg and args.alg not in ('fedtp', 'pf2lora')) else False
    da_detach_private_rms = getattr(args, 'da_detach_private_rms', True) if is_fedsdg else True
    
    if model_variant == 'pretrained':
        # 预训练模型：使用 timm 专用的 LoRA 注入函数
        model = inject_lora_timm(
            model, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha,
            lora_dropout=lora_dropout,
            train_head=bool(getattr(args, 'lora_train_mlp_head', True)),
            is_fedsdg=is_fedsdg,
            fix_gate=fix_gate,
            fixed_gate_value=fixed_gate_value,
            gate_init_value=gate_init_value,
            gate_granularity=gate_granularity,
            use_dynamic_alignment=use_dynamic_alignment,
            da_floor_gamma=da_floor_gamma,
            da_target_mode=da_target_mode,
            use_da_scale=use_da_scale,
            da_detach_private_rms=da_detach_private_rms,
            r_private=lora_r_private
        )
    else:
        # 从零训练模型：使用手写 ViT 的 LoRA 注入函数
        print("\n" + "="*60)
        print(f"[{alg_name}] 注入 {alg_name} 到手写 ViT 模型...")
        print("="*60)
        model = inject_lora(
            model, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha,
            lora_dropout=lora_dropout,
            train_mlp_head=bool(getattr(args, 'lora_train_mlp_head', True)),
            is_fedsdg=is_fedsdg,
            fix_gate=fix_gate,
            fixed_gate_value=fixed_gate_value,
            gate_init_value=gate_init_value,
            gate_granularity=gate_granularity,
            use_dynamic_alignment=use_dynamic_alignment,
            da_floor_gamma=da_floor_gamma,
            da_target_mode=da_target_mode,
            use_da_scale=use_da_scale,
            da_detach_private_rms=da_detach_private_rms,
            r_private=lora_r_private
        )
        print("="*60 + "\n")
    
    # FedSDG coarse 模式：注入全局门控模块并设置引用
    # FedTP 不使用 coarse 门控
    if is_fedsdg and args.alg not in ('fedtp', 'pf2lora') and gate_granularity == 'coarse':
        model = inject_lora_with_gate_granularity(model, args, model_variant, gate_granularity)
    
    # FedSDG DA-Scale 模式：注入全局 DA-Scale 模块并设置引用
    # FedTP 不使用 DA-Scale
    if is_fedsdg and args.alg not in ('fedtp', 'pf2lora') and use_dynamic_alignment and use_da_scale:
        from .lora import GlobalDAScaleModule, LoRALayer
        from fl.utils.console_logger import cprint
        
        device = get_model_device(model)
        da_scale_module = GlobalDAScaleModule(init_value=0.0)
        da_scale_module.to(device)
        
        # 注册为子模块（参数名: fedsdg_global_da_scale.da_scale_logit）
        da_scale_module_name = 'fedsdg_global_da_scale'
        if hasattr(model, da_scale_module_name):
            delattr(model, da_scale_module_name)
        model.add_module(da_scale_module_name, da_scale_module)
        
        # 为所有 DA LoRA 层设置全局 DA-Scale 引用
        da_lora_layers = []
        for module in model.modules():
            if isinstance(module, LoRALayer) and module.use_dynamic_alignment:
                module.set_global_da_scale(da_scale_module.da_scale_logit)
                da_lora_layers.append(module)
        
        cprint(f"[FedSDG-DAScale] 注入全局 DA-Scale 模块，覆盖 {len(da_lora_layers)} 个 DA LoRA 层，设备: {device}")
    
    return model


def _inject_lora_dpa(model: nn.Module, args, model_variant: str, 
                     lora_dropout: float, train_mix_ratio: float) -> nn.Module:
    """
    FedDPA 专用的 LoRA 注入函数
    
    Args:
        model: 原始模型
        args: 配置对象
        model_variant: 模型变体 ('pretrained' 或 'scratch')
        lora_dropout: Dropout 概率
        train_mix_ratio: 训练时的固定混合比例
    
    Returns:
        注入 FedDPA LoRA 后的模型
    """
    # 读取 DA 参数
    use_dynamic_alignment = getattr(args, 'use_dynamic_alignment', False)
    da_floor_gamma = getattr(args, 'da_floor_gamma', 0.1)
    da_target_mode = getattr(args, 'da_target_mode', 'floor')
    da_detach_private_rms = getattr(args, 'da_detach_private_rms', True)
    
    da_kwargs = dict(
        use_dynamic_alignment=use_dynamic_alignment,
        da_floor_gamma=da_floor_gamma,
        da_target_mode=da_target_mode,
        da_detach_private_rms=da_detach_private_rms,
    )
    
    if model_variant == 'pretrained':
        model = inject_lora_dpa_timm(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=lora_dropout,
            train_head=bool(getattr(args, 'lora_train_mlp_head', True)),
            train_mix_ratio=train_mix_ratio,
            **da_kwargs
        )
    else:
        print("\n" + "="*60)
        print(f"[FedDPA] 注入 FedDPA 到手写 ViT 模型...")
        print("="*60)
        model = inject_lora_dpa(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=lora_dropout,
            train_mlp_head=bool(getattr(args, 'lora_train_mlp_head', True)),
            train_mix_ratio=train_mix_ratio,
            **da_kwargs
        )
        print("="*60 + "\n")
    
    return model


def get_model(args, train_dataset, device: str) -> nn.Module:
    """
    一站式模型获取接口（包含 LoRA 注入）
    
    这是推荐使用的主接口，会根据算法类型自动处理 LoRA 注入。
    
    Args:
        args: 配置对象
        train_dataset: 训练数据集
        device: 计算设备
        
    Returns:
        准备好的模型（如需要，已注入 LoRA）
        
    Example:
        >>> model = get_model(args, train_dataset, 'cuda')
        >>> print(model)
    """
    # 构建基础模型
    model = build_model(args, train_dataset, device)
    
    # 根据算法注入 LoRA
    # local_only: 每个客户端独立维护自己的 LoRA 参数，不进行聚合
    # fedrep: 使用 LoRA 作为 Backbone，Head 单独训练
    # ditto: 使用 LoRA 作为可训练参数，全局模型和个性化模型都使用 LoRA
    if args.alg in ('fedlora', 'fedprox_lora', 'fedsdg', 'feddpa', 'local_only', 'fedrep', 'ditto', 'fedsalora', 'fedtp', 'pf2lora', 'lorafair', 'fedalt'):
        model = inject_lora_to_model(model, args)
    
    return model


def get_model_info(model: nn.Module) -> dict:
    """
    获取模型信息摘要
    
    Args:
        model: PyTorch 模型
        
    Returns:
        包含模型信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 统计 LoRA 参数
    lora_params = 0
    private_params = 0
    gate_params = 0
    
    for name, p in model.named_parameters():
        if 'lora_' in name:
            if '_private' in name:
                private_params += p.numel()
            else:
                lora_params += p.numel()
        if 'lambda_k' in name:
            gate_params += p.numel()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'lora_params': lora_params,
        'private_params': private_params,
        'gate_params': gate_params,
        'total_size_mb': total_params * 4 / (1024 ** 2),
        'trainable_size_mb': trainable_params * 4 / (1024 ** 2),
    }
