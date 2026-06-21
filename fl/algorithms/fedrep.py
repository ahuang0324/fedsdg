# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedRep (Federated Representation Learning) 算法实现

核心思想：
- 模型分为 Backbone (表示层/LoRA参数) 和 Head (分类头)
- Backbone 参与联邦聚合，学习共享表示
- Head 由每个客户端本地维护，学习个性化分类器

训练流程（每轮本地训练，遵循原论文 Algorithm 1）：
1. Step 1: 冻结 Backbone，训练 Head (fedrep_head_epochs 个 epoch)
   - 让 Head 适应新聚合的全局 Backbone
2. Step 2: 冻结 Head，训练 Backbone (fedrep_rep_epochs 个 epoch)
   - 用适配好的 Head 指导 Backbone 学习更好的表示

聚合：
- 只聚合 Backbone (LoRA 参数)
- Head 参数作为 local_private_states 管理

评估：
- 不进行 Global Model Evaluation（全局模型没有训练好的 Head）
- 以 Local Personalization Evaluation 为主

参考论文:
    Collins et al., "Exploiting Shared Representations for Personalized 
    Federated Learning", ICML 2021
    
    Algorithm 1 原文顺序:
    for j = 1, ..., ν do      // Fix φ (Backbone), update h (Head)
    for j = 1, ..., τ do      // Fix h (Head), update φ (Backbone)
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, List, Optional

from ..models.lora import get_lora_state_dict
from ..utils.console_logger import cprint


# =============================================================================
# Helper Functions
# =============================================================================

def _is_head_param(name: str) -> bool:
    """
    判断参数是否属于 Head（分类头）
    
    支持的 Head 命名:
    - mlp_head: 手写 ViT
    - head: timm ViT
    """
    return 'mlp_head' in name or ('head' in name and 'lora_' not in name)


def _is_backbone_param(name: str) -> bool:
    """
    判断参数是否属于 Backbone (LoRA 参数)
    
    Backbone 参数包括:
    - lora_A, lora_B: LoRA 低秩矩阵
    """
    return 'lora_' in name and '_private' not in name and 'lambda_k' not in name


def _set_requires_grad(model: nn.Module, backbone: bool, head: bool) -> None:
    """
    设置模型参数的 requires_grad
    
    Args:
        model: 模型实例
        backbone: Backbone (LoRA) 参数是否需要梯度
        head: Head 参数是否需要梯度
    """
    for name, param in model.named_parameters():
        if _is_head_param(name):
            param.requires_grad = head
        elif _is_backbone_param(name):
            param.requires_grad = backbone
        # 其他参数（如原始冻结的预训练参数）保持 requires_grad=False


def _get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """获取当前可训练的参数列表"""
    return [p for p in model.parameters() if p.requires_grad]


def get_head_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    提取 Head 参数
    
    Args:
        model: 模型实例
        
    Returns:
        Head 参数字典 {name: tensor}
    """
    head_dict = {}
    for name, param in model.named_parameters():
        if _is_head_param(name):
            head_dict[name] = param.data.detach().clone()
    return head_dict


def get_backbone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    提取 Backbone (LoRA) 参数
    
    Args:
        model: 模型实例
        
    Returns:
        Backbone 参数字典 {name: tensor}
    """
    backbone_dict = {}
    for name, param in model.named_parameters():
        if _is_backbone_param(name):
            backbone_dict[name] = param.data.detach().clone()
    return backbone_dict


# =============================================================================
# FedRep Local Training
# =============================================================================

def fedrep_update_weights(
    args,
    model: nn.Module,
    trainloader: DataLoader,
    device: str,
    global_round: int,
    gpu_transform=None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, Dict[str, Any]]:
    """
    FedRep 本地训练函数（遵循原论文 Algorithm 1）
    
    执行两阶段训练:
    1. Step 1: 冻结 Backbone，训练 Head (ν 轮)
       - 让 Head 适应新聚合的全局 Backbone
    2. Step 2: 冻结 Head，训练 Backbone (τ 轮)
       - 用适配好的 Head 指导 Backbone 学习
    
    Args:
        args: 配置对象，需要包含:
            - fedrep_head_epochs: Step 1 epoch 数 (ν，原论文建议 5-10)
            - fedrep_rep_epochs: Step 2 epoch 数 (τ，原论文建议 1)
            - lr: Backbone 学习率
            - lr_head: Head 学习率（可选，默认使用 lr）
        model: 本地模型（已加载全局 Backbone + 本地 Head）
        trainloader: 训练数据加载器
        device: 设备 ('cuda' 或 'cpu')
        global_round: 当前全局轮次
        gpu_transform: GPU 数据增强（可选，用于 FEMNIST）
    
    Returns:
        backbone_w: Backbone 权重（用于聚合）
        head_w: Head 权重（本地保存）
        avg_loss: 平均损失
        train_metrics: 训练指标字典
    """
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 获取配置（原论文符号：ν=head_epochs, τ=rep_epochs）
    head_epochs = getattr(args, 'fedrep_head_epochs', 5)   # ν: Head 训练轮数
    rep_epochs = getattr(args, 'fedrep_rep_epochs', 1)     # τ: Backbone 训练轮数
    lr_backbone = args.lr
    lr_head = getattr(args, 'lr_head', None) or args.lr    # 如果未设置，使用相同学习率
    
    # 损失记录
    all_losses = []
    step1_losses = []  # Head 训练损失
    step2_losses = []  # Backbone 训练损失
    
    # 训练准确率统计（在 Step 2 最后阶段统计）
    last_correct = 0
    last_total = 0
    
    # ========================================
    # Step 1: 训练 Head（让 Head 适应新的全局 Backbone）
    # 原论文: for j = 1, ..., ν do  // Fix φ, update h
    # ========================================
    # 仅首轮打印详细信息（避免日志过多）
    if global_round == 0:
        cprint(f'  [FedRep] Step 1: Training Head for {head_epochs} epoch(s), lr={lr_head}')
    
    # 设置梯度：Backbone 冻结，Head 可训练
    _set_requires_grad(model, backbone=False, head=True)
    
    # 创建 Head 优化器
    head_params = _get_trainable_params(model)
    if len(head_params) == 0:
        raise RuntimeError("[FedRep] No trainable head parameters found!")
    
    head_optimizer = torch.optim.Adam(head_params, lr=lr_head, weight_decay=1e-4)
    
    # Step 1 训练循环
    for epoch in range(head_epochs):
        epoch_loss = []
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # GPU Transform（用于 FEMNIST）
            if gpu_transform is not None:
                images = gpu_transform(images)
            
            head_optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            head_optimizer.step()
            
            epoch_loss.append(loss.item())
        
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        step1_losses.append(avg_epoch_loss)
        all_losses.append(avg_epoch_loss)
    
    # ========================================
    # Step 2: 训练 Backbone（用适配好的 Head 指导 Backbone 学习）
    # 原论文: for j = 1, ..., τ do  // Fix h, update φ
    # ========================================
    if global_round == 0:
        cprint(f'  [FedRep] Step 2: Training Backbone for {rep_epochs} epoch(s), lr={lr_backbone}')
    
    # 设置梯度：Backbone 可训练，Head 冻结
    _set_requires_grad(model, backbone=True, head=False)
    
    # 创建 Backbone 优化器
    backbone_params = _get_trainable_params(model)
    if len(backbone_params) == 0:
        raise RuntimeError("[FedRep] No trainable backbone parameters found!")
    
    backbone_optimizer = torch.optim.Adam(backbone_params, lr=lr_backbone, weight_decay=1e-4)
    
    # Step 2 训练循环
    for epoch in range(rep_epochs):
        epoch_loss = []
        is_last_epoch = (epoch == rep_epochs - 1)
        
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # GPU Transform
            if gpu_transform is not None:
                images = gpu_transform(images)
            
            backbone_optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            backbone_optimizer.step()
            
            epoch_loss.append(loss.item())
            
            # 最后一个 epoch 统计训练准确率
            if is_last_epoch:
                with torch.no_grad():
                    _, pred = torch.max(logits, 1)
                    last_correct += (pred == labels).sum().item()
                    last_total += labels.size(0)
        
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        step2_losses.append(avg_epoch_loss)
        all_losses.append(avg_epoch_loss)
    
    # ========================================
    # 提取权重和构建返回值
    # ========================================
    
    # 计算平均损失
    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
    
    # 计算训练准确率
    train_acc = last_correct / last_total if last_total > 0 else 0.0
    
    # 提取 Backbone 权重（用于聚合）
    backbone_w = get_backbone_state_dict(model)
    
    # 提取 Head 权重（本地保存）
    head_w = get_head_state_dict(model)
    
    # 构建训练指标
    train_metrics = {
        'train_acc': train_acc,
        'task_loss': avg_loss,
        'step1_loss': sum(step1_losses) / len(step1_losses) if step1_losses else 0.0,  # Head 训练损失
        'step2_loss': sum(step2_losses) / len(step2_losses) if step2_losses else 0.0,  # Backbone 训练损失
        'head_epochs': head_epochs,   # ν
        'rep_epochs': rep_epochs,     # τ
    }
    
    # 仅首轮打印完成信息
    if global_round == 0:
        cprint(f'  [FedRep] Training complete: loss={avg_loss:.4f}, acc={train_acc*100:.2f}%')
        cprint(f'           Backbone params: {len(backbone_w)}, Head params: {len(head_w)}')
    
    return backbone_w, head_w, avg_loss, train_metrics


# =============================================================================
# FedRep Aggregation Keys
# =============================================================================

def get_fedrep_aggregation_keys(
    client_weights: List[Dict[str, torch.Tensor]]
) -> Tuple[List[str], List[str], List[str]]:
    """
    获取 FedRep 聚合相关的参数键
    
    FedRep 聚合策略:
    - 聚合: LoRA 参数 (lora_A, lora_B)
    - 不聚合: Head 参数 (mlp_head, head)
    
    Args:
        client_weights: 客户端权重列表
        
    Returns:
        aggregated_keys: 参与聚合的参数键 (LoRA)
        align_keys: 用于对齐度计算的参数键 (LoRA)
        excluded_keys: 被排除的参数键 (Head)
    """
    if not client_weights:
        return [], [], []
    
    all_keys = list(client_weights[0].keys())
    aggregated_keys = []
    align_keys = []
    excluded_keys = []
    
    for key in all_keys:
        if _is_backbone_param(key):
            # LoRA 参数：聚合 + 用于对齐度计算
            aggregated_keys.append(key)
            align_keys.append(key)
        elif _is_head_param(key):
            # Head 参数：不聚合
            excluded_keys.append(key)
    
    return aggregated_keys, align_keys, excluded_keys
