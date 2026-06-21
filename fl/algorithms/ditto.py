# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ditto 算法实现

核心思想：
- 每个客户端维护两个模型：全局模型 w 和个性化模型 v
- 个性化模型通过正则化向全局模型靠拢

损失函数：
    L(v) = L_task(v) + (λ/2) ||v - w||²

训练流程（每轮本地训练）：
1. Step 1: 在全局模型 w 上训练（标准训练，用于聚合）
2. Step 2: 在个性化模型 v 上训练（带正则化）

聚合：
- 聚合全局模型 w（全部 LoRA + Head 参数）

评估：
- 使用个性化模型 v 评估（local_test_acc）

内存优化：
- 使用单模型实例 + state_dict 切换，避免 deepcopy
- 仅存储参数字典，不复制完整模型结构

参考论文:
    Li et al., "Ditto: Fair and Robust Federated Learning 
    Through Personalization", ICML 2021
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, Optional

from ..models.lora import get_lora_state_dict, _is_head_param


def ditto_update_weights(
    args,
    model: nn.Module,
    trainloader: DataLoader,
    device: str,
    global_round: int,
    personal_state: Optional[Dict[str, torch.Tensor]] = None,
    gpu_transform=None,
    verbose: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, Dict[str, Any]]:
    """
    Ditto 本地训练函数（内存优化版本）
    
    使用单模型实例 + state_dict 切换，避免 deepcopy
    
    训练流程:
    1. Step 1: 在全局模型上训练（标准训练，用于聚合）
    2. Step 2: 在个性化模型上训练（带正则化）
    
    Args:
        args: 配置对象，需要包含:
            - local_ep: 本地训练 epoch 数
            - lr: 学习率
            - lambda_ditto: 正则化系数（默认 0.1）
            - ditto_reg_target: 正则化参考点 'server' 或 'local'（默认 'server'）
        model: 全局模型（已加载 Server 下发的权重）
        trainloader: 训练数据加载器
        device: 设备 ('cuda' 或 'cpu')
        global_round: 当前全局轮次
        personal_state: 个性化模型状态（首次为 None）
        gpu_transform: GPU 数据增强（可选）
        verbose: 是否打印详细信息
    
    Returns:
        global_w: 全局模型权重（用于聚合，包含 LoRA + Head）
        personal_w: 个性化模型权重（本地保存）
        avg_loss: 平均损失
        train_metrics: 训练指标字典
    """
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 获取配置
    lambda_ditto = getattr(args, 'lambda_ditto', 0.1)
    reg_target = getattr(args, 'ditto_reg_target', 'server')  # 'server' or 'local'
    local_ep = args.local_ep
    lr = args.lr
    lr_head = getattr(args, 'lr_head', None) or args.lr  # 获取head学习率
    
    # 损失记录
    step1_losses = []
    step2_losses = []
    step2_task_losses = []
    step2_reg_losses = []
    
    # 训练准确率统计（Step 2 最后阶段）
    last_correct = 0
    last_total = 0
    
    # ========================================
    # 保存正则化参考点 w_reference
    # ========================================
    # 选项 A: 使用 Server 下发的 w^t（原论文方式）
    # 选项 B: 使用本地训练后的 w^{t+1}_local
    
    if reg_target == 'server':
        # 在 Step 1 训练前保存 w^t
        w_reference = {
            name: param.clone().detach() 
            for name, param in model.named_parameters() 
            if param.requires_grad
        }
    
    # ========================================
    # Step 1: 训练全局模型（标准训练，用于聚合）
    # ========================================
    if verbose:
        print(f'  [Ditto] Step 1: Training global model for {local_ep} epoch(s), lr={lr}')
    
    model.train()
    
    # 参数分组：支持head和backbone不同学习率
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if _is_head_param(name):
                head_params.append(param)
            else:
                backbone_params.append(param)
    
    # 构建参数组
    param_groups = [
        {'params': backbone_params, 'lr': lr},
        {'params': head_params, 'lr': lr_head},
    ]
    
    optimizer_global = torch.optim.Adam(param_groups, lr=lr, weight_decay=1e-4)
    
    for epoch in range(local_ep):
        epoch_loss = []
        for batch_idx, (images, labels) in enumerate(trainloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # GPU Transform
            if gpu_transform is not None:
                images = gpu_transform(images)
            
            optimizer_global.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer_global.step()
            
            epoch_loss.append(loss.item())
        
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
        step1_losses.append(avg_epoch_loss)
    
    # 提取全局模型权重（用于聚合）
    global_w = get_lora_state_dict(model)
    
    if reg_target == 'local':
        # 选项 B: 在 Step 1 训练后保存 w^{t+1}_local
        w_reference = {
            name: param.clone().detach() 
            for name, param in model.named_parameters() 
            if param.requires_grad
        }
    
    # ========================================
    # Step 2: 训练个性化模型（带正则化）
    # ========================================
    if verbose:
        print(f'  [Ditto] Step 2: Training personal model for {local_ep} epoch(s), λ={lambda_ditto}')
    
    # 切换到个性化模型状态
    if personal_state is not None:
        # 加载历史个性化状态
        model.load_state_dict(personal_state, strict=False)
        if verbose:
            print(f'  [Ditto] Loaded personal state with {len(personal_state)} parameters')
    else:
        # 首次训练：继承 Step 1 结果（已经是 global_w 状态）
        if verbose:
            print(f'  [Ditto] First-time training, initializing from global model')
    
    model.train()
    
    # 参数分组：支持head和backbone不同学习率
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if _is_head_param(name):
                head_params.append(param)
            else:
                backbone_params.append(param)
    
    # 构建参数组
    param_groups = [
        {'params': backbone_params, 'lr': lr},
        {'params': head_params, 'lr': lr_head},
    ]
    
    optimizer_personal = torch.optim.Adam(param_groups, lr=lr, weight_decay=1e-4)
    
    for epoch in range(local_ep):
        epoch_task_loss = []
        epoch_reg_loss = []
        is_last_epoch = (epoch == local_ep - 1)
        
        for batch_idx, (images, labels) in enumerate(trainloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # GPU Transform
            if gpu_transform is not None:
                images = gpu_transform(images)
            
            optimizer_personal.zero_grad()
            logits = model(images)
            
            # 任务损失
            task_loss = criterion(logits, labels)
            
            # 正则化损失: (λ/2) ||v - w||²
            reg_loss = torch.tensor(0.0, device=device)
            for name, param in model.named_parameters():
                if param.requires_grad and name in w_reference:
                    reg_loss = reg_loss + ((param - w_reference[name]) ** 2).sum()
            
            # 总损失
            loss = task_loss + (lambda_ditto / 2) * reg_loss
            loss.backward()
            optimizer_personal.step()
            
            epoch_task_loss.append(task_loss.item())
            epoch_reg_loss.append(reg_loss.item())
            
            # 最后一个 epoch 统计训练准确率
            if is_last_epoch:
                with torch.no_grad():
                    _, pred = torch.max(logits, 1)
                    last_correct += (pred == labels).sum().item()
                    last_total += labels.size(0)
        
        avg_task_loss = sum(epoch_task_loss) / len(epoch_task_loss) if epoch_task_loss else 0.0
        avg_reg_loss = sum(epoch_reg_loss) / len(epoch_reg_loss) if epoch_reg_loss else 0.0
        step2_losses.append(avg_task_loss + (lambda_ditto / 2) * avg_reg_loss)
        step2_task_losses.append(avg_task_loss)
        step2_reg_losses.append(avg_reg_loss)
    
    # 提取个性化模型权重（本地保存）
    personal_w = get_lora_state_dict(model)
    
    # ========================================
    # 构建返回值
    # ========================================
    all_losses = step1_losses + step2_losses
    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
    train_acc = last_correct / last_total if last_total > 0 else 0.0
    
    train_metrics = {
        'train_acc': train_acc,
        'task_loss': sum(step2_task_losses) / len(step2_task_losses) if step2_task_losses else 0.0,
        'reg_loss': sum(step2_reg_losses) / len(step2_reg_losses) if step2_reg_losses else 0.0,
        'step1_loss': sum(step1_losses) / len(step1_losses) if step1_losses else 0.0,
        'step2_loss': sum(step2_losses) / len(step2_losses) if step2_losses else 0.0,
        'lambda_ditto': lambda_ditto,
        'reg_target': reg_target,
    }
    
    if verbose:
        print(f'  [Ditto] Training complete: loss={avg_loss:.4f}, acc={train_acc*100:.2f}%')
        print(f'           Global params: {len(global_w)}, Personal params: {len(personal_w)}')
        print(f'           Reg loss: {train_metrics["reg_loss"]:.4f}')
    
    return global_w, personal_w, avg_loss, train_metrics


def get_ditto_aggregation_keys(
    client_weights: list,
) -> Tuple[list, list, list]:
    """
    获取 Ditto 聚合相关的参数键
    
    Ditto 聚合策略:
    - 聚合: 全部 LoRA + Head 参数（与 FedLoRA 相同）
    - 不聚合: 无（个性化模型不聚合，但也不在 client_weights 中）
    
    Args:
        client_weights: 客户端权重列表（全局模型权重）
        
    Returns:
        aggregated_keys: 参与聚合的参数键
        align_keys: 用于对齐度计算的参数键 (LoRA)
        excluded_keys: 被排除的参数键
    """
    if not client_weights:
        return [], [], []
    
    all_keys = list(client_weights[0].keys())
    aggregated_keys = []
    align_keys = []
    excluded_keys = []
    
    for key in all_keys:
        # Ditto 聚合全部 LoRA + Head（与 FedLoRA 相同）
        # 排除 FedSDG 的私有参数和门控参数
        if '_private' in key or 'lambda_k' in key:
            excluded_keys.append(key)
        else:
            aggregated_keys.append(key)
            # LoRA 参数用于对齐度计算
            if 'lora_' in key:
                align_keys.append(key)
    
    return aggregated_keys, align_keys, excluded_keys
