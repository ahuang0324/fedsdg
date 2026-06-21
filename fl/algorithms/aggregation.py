# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一的服务端聚合模块

提供联邦学习服务端聚合的统一入口，支持所有算法（FedAvg、FedLoRA、FedSDG）。

核心设计：
- 两个维度分离：
  1. algorithm: 决定聚合哪些参数（fedavg/fedlora/fedsdg）
  2. agg_method: 决定如何聚合（fedavg/alignment）

使用方式：
    from fl.algorithms import server_aggregate
    
    new_state, info = server_aggregate(
        client_weights=local_weights,
        global_state_dict=global_state_cache,
        algorithm='fedlora',           # 聚合哪些参数
        agg_method='alignment',        # 如何聚合
        alignment_strategy='loo_mean',
        weight_transform='relu_normalize',
    )
"""

import copy
import math
import torch
from typing import List, Dict, Tuple, Any, Optional, Set

from .alignment_strategies import (
    AlignmentStrategy,
    create_alignment_strategy,
)


# ============================================================================
# Helper Functions
# ============================================================================

def _flatten_params(
    state_dict: Dict[str, torch.Tensor], 
    keys: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    展平参数到 1D 向量（float32）。
    
    Args:
        state_dict: 模型 state_dict
        keys: 要展平的参数键名列表
        device: 目标设备
        
    Returns:
        展平后的参数向量 [D]，float32
    """
    tensors = []
    for k in keys:
        if k in state_dict:
            tensors.append(state_dict[k].flatten().float())
    
    if tensors:
        return torch.cat(tensors).to(device)
    else:
        return torch.empty(0, dtype=torch.float32, device=device)


def _is_head_key(key: str) -> bool:
    """
    判断参数键是否属于分类头（Head）
    
    精确匹配规则，避免误伤注意力相关参数：
    - mlp_head: 手写 ViT 的分类头
    - head: timm ViT 的分类头（但排除 lora_ 前缀和注意力相关）
    
    排除的误匹配模式：
    - attention.head, attn.head: 注意力头
    - head_dim: 注意力头维度
    - multihead, multi_head: 多头注意力
    - num_heads: 注意力头数量
    """
    # 精确匹配 mlp_head（手写 ViT）
    if 'mlp_head' in key:
        return True
    
    # 排除 LoRA 参数（lora_A, lora_B 等）
    if 'lora_' in key:
        return False
    
    # 排除注意力相关的 head
    attention_patterns = [
        'attention.head', 'attn.head',  # 注意力头
        'head_dim',                      # 注意力头维度
        'multihead', 'multi_head',       # 多头注意力
        'num_heads', 'n_heads',          # 注意力头数量
    ]
    key_lower = key.lower()
    for pattern in attention_patterns:
        if pattern in key_lower:
            return False
    
    # timm ViT: 分类头通常是 'head.weight', 'head.bias'
    # 需要确保是顶层的 head，而不是嵌套在其他模块中的
    if '.head.' in key or key.startswith('head.'):
        return True
    
    return False


def _get_aggregation_keys(
    client_weights: List[Dict[str, torch.Tensor]],
    algorithm: str,
    head_mode: str = 'global'
) -> Tuple[List[str], List[str], List[str]]:
    """
    根据算法类型确定聚合哪些参数。
    
    Args:
        client_weights: 客户端权重列表
        algorithm: 算法类型 ('fedavg' | 'fedprox_avg' | 'fedlora' | 'fedprox_lora' | 'fedsdg' | 'local_only' | 'fedrep')
        head_mode: Head 参数模式（仅对 fedsdg 生效）
            - 'global': Head 参与聚合（默认）
            - 'private': Head 不参与聚合
        
    Returns:
        aggregated_keys: 参与聚合的参数键
        align_keys: 用于计算对齐度的参数键（仅 LoRA 参数）
        excluded_keys: 被排除的参数键
    """
    if not client_weights:
        return [], [], []
    
    all_keys = list(client_weights[0].keys())
    aggregated_keys = []
    align_keys = []
    excluded_keys = []
    
    if algorithm in ('fedavg', 'fedprox_avg'):
        # FedAvg / FedProx: 聚合所有参数
        aggregated_keys = all_keys
        # 对齐度计算使用所有参数
        align_keys = all_keys
    
    elif algorithm in ('fedlora', 'fedprox_lora', 'fedsdg', 'feddpa', 'fedsalora', 'pf2lora', 'fedtp', 'lorafair', 'fedalt'):
        # FedLoRA / FedProx+LoRA / FedSDG / FedDPA / FedSA-LoRA / PF2LoRA: 仅聚合 LoRA 参数和分类头
        # FedDPA 与 FedSDG 使用相同的聚合逻辑：排除 _private 参数
        # FedSA-LoRA: public_w 仅包含 lora_A + Head（lora_B 已在参数提取时分离）
        # PF2LoRA: public_w 仅包含 Shared LoRA (A+B) + Head（Private LoRA 已分离）
        for key in all_keys:
            if '_private' in key or 'lambda_k' in key:
                # 私有参数和门控参数不聚合
                excluded_keys.append(key)
            elif 'lora_' in key:
                # LoRA 参数：聚合 + 用于对齐度计算
                align_keys.append(key)
                aggregated_keys.append(key)
            elif 'da_scale_logit' in key:
                # 全局 DA-Scale 参数：聚合（不用于对齐度计算）
                aggregated_keys.append(key)
            elif _is_head_key(key):
                # 分类头：根据 head_mode 决定是否聚合
                if algorithm in ('fedsdg', 'feddpa') and head_mode == 'private':
                    # FedSDG/FedDPA + head_mode='private': Head 不聚合
                    excluded_keys.append(key)
                else:
                    # 其他情况：Head 聚合但不用于对齐度计算
                    aggregated_keys.append(key)
    
    elif algorithm == 'local_only':
        # Local-Only: 不聚合任何参数
        # 所有 LoRA 参数都是"私有"的（每个客户端独立维护）
        aggregated_keys = []
        align_keys = []
        excluded_keys = all_keys  # 所有参数都被排除
    
    elif algorithm == 'fedrep':
        # FedRep: 聚合 LoRA 参数，不聚合 Head
        # Backbone (LoRA) 参与聚合，Head 由客户端本地维护
        for key in all_keys:
            if 'lora_' in key and '_private' not in key and 'lambda_k' not in key:
                # LoRA 参数：聚合 + 用于对齐度计算
                aggregated_keys.append(key)
                align_keys.append(key)
            elif _is_head_key(key):
                # Head 参数：不聚合
                excluded_keys.append(key)
    
    elif algorithm == 'ditto':
        # Ditto: 聚合全局模型（与 FedLoRA 相同）
        # LoRA + Head 都参与聚合，个性化模型保存在本地
        for key in all_keys:
            if '_private' in key or 'lambda_k' in key:
                # 私有参数和门控参数不聚合（来自 FedSDG，Ditto 不使用）
                excluded_keys.append(key)
            elif 'lora_' in key:
                # LoRA 参数：聚合 + 用于对齐度计算
                align_keys.append(key)
                aggregated_keys.append(key)
            elif _is_head_key(key):
                # 分类头：聚合但不用于对齐度计算
                aggregated_keys.append(key)
    
    else:
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. "
            f"Available: 'fedavg', 'fedprox_avg', 'fedlora', 'fedprox_lora', 'fedsdg', 'feddpa', 'fedsalora', 'pf2lora', 'fedtp', 'lorafair', 'fedalt', 'local_only', 'fedrep', 'ditto'"
        )
    
    return aggregated_keys, align_keys, excluded_keys


# ============================================================================
# Aggregation Methods
# ============================================================================

def _aggregate_uniform(
    client_weights: List[Dict[str, torch.Tensor]],
    aggregated_keys: List[str],
    w_avg: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    均匀加权平均聚合（FedAvg 方式）。
    
    Args:
        client_weights: 客户端权重列表
        aggregated_keys: 需要聚合的参数键
        w_avg: 初始化的聚合结果（会被修改）
        
    Returns:
        w_avg: 聚合后的权重
        stats: 聚合统计信息
    """
    M = len(client_weights)
    uniform_weight = 1.0 / M
    
    for key in aggregated_keys:
        # 获取目标设备和数据类型（从 w_avg 中获取，确保与全局模型一致）
        target_device = w_avg[key].device
        target_dtype = w_avg[key].dtype
        
        # 初始化为第一个客户端的权重（确保设备和数据类型一致）
        w_avg[key] = client_weights[0][key].to(device=target_device, dtype=target_dtype).clone()
        
        # 累加其他客户端的权重
        for i in range(1, M):
            w_avg[key] += client_weights[i][key].to(device=target_device, dtype=target_dtype)
        
        # 计算平均值
        w_avg[key] = torch.div(w_avg[key], M)
    
    stats = {
        'weights': [uniform_weight] * M,
        'n_eff': float(M),
        'entropy': math.log(M) if M > 0 else 0.0,
        'max_weight': uniform_weight,
        'diff_to_uniform': 0.0,
        'kl_from_uniform': 0.0,
    }
    
    return w_avg, stats


def _aggregate_alignment(
    client_weights: List[Dict[str, torch.Tensor]],
    global_state_dict: Dict[str, torch.Tensor],
    aggregated_keys: List[str],
    align_keys: List[str],
    w_avg: Dict[str, torch.Tensor],
    epsilon: float = 1e-8,
    alignment_strategy: str = 'loo_mean',
    weight_transform: str = 'relu_normalize',
    softmax_temperature: float = 1.0,
    lambda_smooth: float = 0.0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    基于对齐度的加权聚合。
    
    Args:
        client_weights: 客户端权重列表
        global_state_dict: 全局模型 state_dict
        aggregated_keys: 需要聚合的参数键
        align_keys: 用于计算对齐度的参数键
        w_avg: 初始化的聚合结果（会被修改）
        epsilon: 数值稳定性参数
        alignment_strategy: 对齐度计算策略
        weight_transform: 权重转换方式
        softmax_temperature: softmax 温度
        lambda_smooth: 权重平滑系数
        
    Returns:
        w_avg: 聚合后的权重
        stats: 聚合统计信息
    """
    M = len(client_weights)
    
    # 创建对齐度计算策略
    strategy = create_alignment_strategy(alignment_strategy)
    
    # 计算对齐度权重
    weights, weight_stats = _compute_alignment_weights(
        client_weights=client_weights,
        global_state_dict=global_state_dict,
        align_keys=align_keys,
        epsilon=epsilon,
        strategy=strategy,
        weight_transform=weight_transform,
        softmax_temperature=softmax_temperature,
        lambda_smooth=lambda_smooth,
    )
    
    # 加权聚合
    for key in aggregated_keys:
        weighted_sum = torch.zeros_like(client_weights[0][key], dtype=torch.float32)
        for i, client_w in enumerate(client_weights):
            weighted_sum += weights[i] * client_w[key].float()
        w_avg[key] = weighted_sum.to(client_weights[0][key].dtype)
    
    # 构建统计信息
    stats = {
        'weights': weights,
        'weight_stats': weight_stats,
        'alignment_strategy': strategy.get_strategy_name(),
        'weight_transform': weight_transform,
        'n_eff': weight_stats.get('n_eff', 0.0),
        'entropy': weight_stats.get('entropy', 0.0),
        'max_weight': weight_stats.get('max', 0.0),
        'diff_to_uniform': weight_stats.get('diff_to_uniform', 0.0),
        'kl_from_uniform': weight_stats.get('kl_from_uniform', 0.0),
    }
    
    return w_avg, stats


def _compute_alignment_weights(
    client_weights: List[Dict[str, torch.Tensor]],
    global_state_dict: Dict[str, torch.Tensor],
    align_keys: List[str],
    epsilon: float = 1e-8,
    strategy: Optional[AlignmentStrategy] = None,
    weight_transform: str = 'relu_normalize',
    softmax_temperature: float = 1.0,
    lambda_smooth: float = 0.0,
) -> Tuple[List[float], Dict[str, Any]]:
    """
    计算对齐度权重。
    
    Args:
        client_weights: 客户端权重列表
        global_state_dict: 全局模型 state_dict
        align_keys: 用于对齐度计算的参数键名
        epsilon: 数值稳定性参数
        strategy: 对齐度计算策略
        weight_transform: 权重转换方式
        softmax_temperature: softmax 温度
        lambda_smooth: 权重平滑系数
        
    Returns:
        weights: 最终聚合权重 List[float]
        weight_stats: 统计信息字典
    """
    M = len(client_weights)
    
    # 默认策略
    if strategy is None:
        strategy = create_alignment_strategy('loo_mean')
    
    strategy_name = strategy.get_strategy_name()
    
    # 边界情况
    if M == 0:
        return [], _make_empty_stats(strategy_name, weight_transform)
    
    if M == 1:
        return [1.0], _make_single_client_stats(strategy_name, weight_transform)
    
    # 推断 device
    device = next(iter(global_state_dict.values())).device
    
    # Flatten 参数
    global_flat = _flatten_params(global_state_dict, align_keys, device)
    
    if global_flat.numel() == 0:
        return [1.0 / M] * M, _make_fallback_stats(M, strategy_name, weight_transform)
    
    # 计算 deltas
    deltas = []
    norm_deltas_list = []
    for client_w in client_weights:
        client_flat = _flatten_params(client_w, align_keys, device)
        delta = client_flat - global_flat
        deltas.append(delta)
        norm_deltas_list.append(torch.linalg.norm(delta.float(), ord=2))
    
    # 统一 stack
    delta_stack = torch.stack(deltas)  # [M, D]
    norm_deltas = torch.stack(norm_deltas_list)  # [M]
    
    # 确保所有 CUDA 操作完成，避免多进程序列化问题
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 计算平均更新方向的范数
    delta_mean = delta_stack.mean(dim=0)
    norm_mean = torch.linalg.norm(delta_mean.float(), ord=2).item()
    
    # 策略计算余弦相似度
    cos_raw, strategy_stats = strategy.compute_cosine_similarities(
        delta_stack, norm_deltas, epsilon
    )
    
    # 权重转换
    fallback_uniform = False
    alpha_raw = None
    logits = None
    
    if weight_transform == 'relu_normalize':
        alpha_raw = torch.clamp(cos_raw, min=0.0)
        sum_alpha = alpha_raw.sum()
        
        if sum_alpha < 1e-6:
            weights = torch.ones(M, device=device, dtype=torch.float32) / M
            fallback_uniform = True
        else:
            weights = alpha_raw / (sum_alpha + epsilon)
    
    elif weight_transform == 'softmax':
        logits = cos_raw / softmax_temperature
        logits = logits - logits.max()
        weights = torch.softmax(logits, dim=0)
    
    else:
        raise ValueError(f"Unknown weight_transform: '{weight_transform}'")
    
    # 权重平滑
    if lambda_smooth > 0 and not fallback_uniform:
        uniform = torch.ones(M, device=device, dtype=torch.float32) / M
        weights = (1 - lambda_smooth) * weights + lambda_smooth * uniform
    
    # 确保所有 CUDA 操作完成，避免多进程序列化问题
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 将所有 tensor 移到 CPU 并 detach，避免多进程序列化问题
    weights_cpu = weights.detach().cpu()
    cos_raw_cpu = cos_raw.detach().cpu()
    alpha_raw_cpu = alpha_raw.detach().cpu() if alpha_raw is not None else None
    logits_cpu = logits.detach().cpu() if logits is not None else None
    norm_deltas_cpu = norm_deltas.detach().cpu()
    
    # 计算监控指标（在 CPU 上）
    entropy = -torch.sum(weights_cpu * torch.log(weights_cpu + epsilon)).item()
    n_eff = (1.0 / torch.sum(weights_cpu ** 2)).item()
    
    # 计算与均匀分布的差异
    uniform_weights = torch.ones_like(weights_cpu) / M
    diff_to_uniform = torch.sum(torch.abs(weights_cpu - uniform_weights)).item() / 2.0
    
    mask = weights_cpu > epsilon
    if mask.any():
        kl_from_uniform = torch.sum(weights_cpu[mask] * torch.log(weights_cpu[mask] / uniform_weights[mask])).item()
    else:
        kl_from_uniform = 0.0
    
    # 构建统计信息（所有值都在 CPU 上）
    weight_stats = {
        'weights': weights_cpu.tolist(),
        'mean': weights_cpu.mean().item(),
        'std': weights_cpu.std(unbiased=False).item(),
        'min': weights_cpu.min().item(),
        'max': weights_cpu.max().item(),
        'num_zero': int((weights_cpu < 1e-6).sum().item()),
        'entropy': entropy,
        'n_eff': n_eff,
        'diff_to_uniform': diff_to_uniform,
        'kl_from_uniform': kl_from_uniform,
        'cos_raw': cos_raw_cpu.tolist(),
        'alpha_raw': alpha_raw_cpu.tolist() if alpha_raw_cpu is not None else None,
        'sum_alpha_raw': alpha_raw_cpu.sum().item() if alpha_raw_cpu is not None else None,
        'logits': logits_cpu.tolist() if logits_cpu is not None else None,
        'softmax_temperature': softmax_temperature if weight_transform == 'softmax' else None,
        'norm_mean': norm_mean,
        'norm_delta_mean': norm_deltas_cpu.mean().item(),
        'norm_delta_std': norm_deltas_cpu.std(unbiased=False).item(),
        'norm_delta_min': norm_deltas_cpu.min().item(),
        'norm_delta_max': norm_deltas_cpu.max().item(),
        'strategy': strategy_name,
        'weight_transform': weight_transform,
        'lambda_smooth': lambda_smooth,
        'fallback_uniform': fallback_uniform,
        'num_params_aligned': global_flat.numel(),
    }
    
    # 合并策略特有统计（确保所有 tensor 都在 CPU 上）
    for key, val in strategy_stats.items():
        if key not in weight_stats:
            if isinstance(val, torch.Tensor):
                # 确保 tensor 在 CPU 上
                if val.is_cuda:
                    val = val.detach().cpu()
                weight_stats[key] = val.item() if val.numel() == 1 else val.tolist()
            else:
                weight_stats[key] = val
    
    return weights_cpu.tolist(), weight_stats


# ============================================================================
# Stats Helper Functions
# ============================================================================

def _make_empty_stats(strategy_name: str, weight_transform: str) -> Dict[str, Any]:
    """M=0 时的统计"""
    return {
        'weights': [],
        'cos_raw': [],
        'alpha_raw': None,
        'logits': None,
        'entropy': 0.0,
        'n_eff': 0.0,
        'diff_to_uniform': 0.0,
        'kl_from_uniform': 0.0,
        'mean': 0.0,
        'std': 0.0,
        'min': 0.0,
        'max': 0.0,
        'num_zero': 0,
        'norm_mean': 0.0,
        'norm_delta_mean': 0.0,
        'norm_delta_std': 0.0,
        'norm_delta_min': 0.0,
        'norm_delta_max': 0.0,
        'strategy': strategy_name,
        'weight_transform': weight_transform,
        'fallback_uniform': False,
        'num_params_aligned': 0,
    }


def _make_single_client_stats(strategy_name: str, weight_transform: str) -> Dict[str, Any]:
    """M=1 时的统计"""
    return {
        'weights': [1.0],
        'mean': 1.0,
        'std': 0.0,
        'min': 1.0,
        'max': 1.0,
        'num_zero': 0,
        'cos_raw': [1.0],
        'alpha_raw': [1.0] if weight_transform == 'relu_normalize' else None,
        'logits': [0.0] if weight_transform == 'softmax' else None,
        'entropy': 0.0,
        'n_eff': 1.0,
        'diff_to_uniform': 1.0,
        'kl_from_uniform': 0.0,
        'norm_mean': 0.0,
        'norm_delta_mean': 0.0,
        'norm_delta_std': 0.0,
        'norm_delta_min': 0.0,
        'norm_delta_max': 0.0,
        'strategy': strategy_name,
        'weight_transform': weight_transform,
        'fallback_uniform': False,
    }


def _make_fallback_stats(M: int, strategy_name: str, weight_transform: str) -> Dict[str, Any]:
    """参数为空时的 fallback 统计"""
    uniform_w = 1.0 / M
    return {
        'weights': [uniform_w] * M,
        'mean': uniform_w,
        'std': 0.0,
        'min': uniform_w,
        'max': uniform_w,
        'num_zero': 0,
        'cos_raw': [1.0] * M,
        'alpha_raw': [1.0] * M if weight_transform == 'relu_normalize' else None,
        'logits': None,
        'entropy': math.log(M),
        'n_eff': float(M),
        'diff_to_uniform': 0.0,
        'kl_from_uniform': 0.0,
        'norm_mean': 0.0,
        'norm_delta_mean': 0.0,
        'norm_delta_std': 0.0,
        'norm_delta_min': 0.0,
        'norm_delta_max': 0.0,
        'strategy': strategy_name,
        'weight_transform': weight_transform,
        'fallback_uniform': True,
        'num_params_aligned': 0,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def _apply_lorafair_residual_correction(
    w_avg: Dict[str, torch.Tensor],
    client_weights: List[Dict[str, torch.Tensor]],
    aggregated_keys: List[str],
    residual_mu: float = 0.1,
) -> Dict[str, Any]:
    """
    LoRA-FAIR 服务端残差修正：修正聚合后的 lora_B 矩阵。
    
    本代码库的 LoRA 形状约定（与 HuggingFace PEFT 不同）：
    - lora_A: [d_in, r]    （forward: x @ A @ B）
    - lora_B: [r, d_out]
    
    核心思想：
    - 理想全局更新: ΔW = (1/M) Σ A_k @ B_k    形状 [d_in, d_out]
    - FedAvg 聚合: Ā @ B̄ ≠ ΔW（聚合偏差）
    - 修正: 找到 ΔB 使得 Ā @ (B̄+ΔB) ≈ ΔW
    - 令 R = ΔW - Ā@B̄，求解 min_ΔB ||Ā@ΔB - R||² + μ||ΔB||²
    - 闭式解: ΔB = (Ā^T Ā + μI)^{-1} Ā^T R
    
    Args:
        w_avg: 聚合后的全局状态字典（将被原地修改）
        client_weights: 客户端权重列表
        aggregated_keys: 参与聚合的参数键
        residual_mu: 正则化系数 μ
        
    Returns:
        stats: 修正统计信息
    """
    M = len(client_weights)
    if M == 0:
        return {'lorafair_corrected_layers': 0, 'lorafair_avg_delta_norm': 0.0}
    
    # 收集所有 LoRA 层的 (lora_A_key, lora_B_key) 对
    lora_a_keys = [k for k in aggregated_keys if 'lora_A' in k]
    lora_pairs = []
    for a_key in lora_a_keys:
        b_key = a_key.replace('lora_A', 'lora_B')
        if b_key in aggregated_keys:
            lora_pairs.append((a_key, b_key))
    
    if not lora_pairs:
        return {'lorafair_corrected_layers': 0, 'lorafair_avg_delta_norm': 0.0}
    
    total_delta_norm = 0.0
    corrected_count = 0
    
    for a_key, b_key in lora_pairs:
        # 获取聚合后的 Ā 和 B̄（本代码库约定）
        A_bar = w_avg[a_key].float()  # [d_in, r]
        B_bar = w_avg[b_key].float()  # [r, d_out]
        
        # 计算理想全局更新: ΔW = (1/M) Σ (A_k @ B_k)
        # ΔW 形状: [d_in, d_out]
        device = A_bar.device
        delta_W = torch.zeros(A_bar.shape[0], B_bar.shape[1], device=device, dtype=torch.float32)
        for i in range(M):
            A_k = client_weights[i][a_key].float().to(device)  # [d_in, r]
            B_k = client_weights[i][b_key].float().to(device)  # [r, d_out]
            delta_W += A_k @ B_k
        delta_W /= M
        
        # 计算残差: R = ΔW - Ā @ B̄    形状 [d_in, d_out]
        residual = delta_W - A_bar @ B_bar
        
        # 闭式解: ΔB = (Ā^T Ā + μI)^{-1} @ Ā^T @ R
        ATA = A_bar.t() @ A_bar  # [r, d_in] @ [d_in, r] = [r, r]
        reg = residual_mu * torch.eye(ATA.shape[0], device=device, dtype=torch.float32)
        ATA_reg_inv = torch.linalg.inv(ATA + reg)  # [r, r]
        delta_B = ATA_reg_inv @ A_bar.t() @ residual  # [r, r] @ [r, d_in] @ [d_in, d_out] = [r, d_out]
        
        # 应用修正: B' = B̄ + ΔB
        w_avg[b_key] = (B_bar + delta_B).to(w_avg[b_key].dtype)
        
        total_delta_norm += delta_B.norm().item()
        corrected_count += 1
    
    avg_delta_norm = total_delta_norm / corrected_count if corrected_count > 0 else 0.0
    
    return {
        'lorafair_corrected_layers': corrected_count,
        'lorafair_avg_delta_norm': avg_delta_norm,
    }


def server_aggregate(
    client_weights: List[Dict[str, torch.Tensor]],
    global_state_dict: Dict[str, torch.Tensor],
    algorithm: str,
    agg_method: str = 'fedavg',
    epsilon: float = 1e-8,
    alignment_strategy: str = 'loo_mean',
    weight_transform: str = 'relu_normalize',
    softmax_temperature: float = 1.0,
    lambda_smooth: float = 0.0,
    head_mode: str = 'global',
    residual_mu: float = 0.1,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    统一的服务端聚合函数（支持 FedAvg、FedLoRA、FedSDG）。
    
    两个维度的选择：
    1. algorithm: 决定聚合哪些参数
       - 'fedavg': 所有参数
       - 'fedlora': LoRA 参数 + 分类头
       - 'fedsdg': 全局 LoRA 参数 + 分类头（排除私有和门控）
    
    2. agg_method: 决定如何聚合
       - 'fedavg': 均匀加权平均
       - 'alignment': 基于对齐度的加权平均
    
    Args:
        client_weights: 客户端权重列表
        global_state_dict: 完整的全局模型 state_dict
        algorithm: 算法类型，决定聚合哪些参数
        agg_method: 聚合方法 ('fedavg' | 'alignment')
        epsilon: 数值稳定性参数
        alignment_strategy: 对齐度计算策略 ('loo_mean' | 'standard_mean')
        weight_transform: 权重转换方式 ('relu_normalize' | 'softmax')
        softmax_temperature: softmax 温度
        lambda_smooth: 权重平滑系数
        head_mode: Head 参数模式（仅对 fedsdg 生效）
            - 'global': Head 参与聚合（默认）
            - 'private': Head 不参与聚合
        
    Returns:
        new_global_state: 聚合后的全局 state_dict
        aggregation_info: 聚合统计信息
    """
    # 确定聚合哪些参数（提前计算，用于选择性拷贝）
    aggregated_keys, align_keys, excluded_keys = _get_aggregation_keys(
        client_weights, algorithm, head_mode=head_mode
    )
    
    # 初始化聚合结果：选择性拷贝，仅对需要聚合的 key 做 clone()
    # 未参与聚合的 key 保持原始引用，减少不必要的内存拷贝
    aggregated_set = set(aggregated_keys)
    w_avg = {}
    for k, v in global_state_dict.items():
        if k in aggregated_set:
            w_avg[k] = v.clone()
        else:
            w_avg[k] = v
    
    # 初始化聚合信息
    aggregation_info = {
        'algorithm': algorithm,
        'agg_method': agg_method,
        'num_clients': len(client_weights),
        'aggregated_keys': aggregated_keys,
        'align_keys': align_keys,
        'excluded_keys': excluded_keys,
    }
    
    # 边界情况
    if len(client_weights) == 0 or len(aggregated_keys) == 0:
        aggregation_info['weights'] = []
        return w_avg, aggregation_info
    
    # 根据聚合方法执行聚合
    if agg_method == 'alignment':
        w_avg, stats = _aggregate_alignment(
            client_weights=client_weights,
            global_state_dict=global_state_dict,
            aggregated_keys=aggregated_keys,
            align_keys=align_keys,
            w_avg=w_avg,
            epsilon=epsilon,
            alignment_strategy=alignment_strategy,
            weight_transform=weight_transform,
            softmax_temperature=softmax_temperature,
            lambda_smooth=lambda_smooth,
        )
    else:
        # 默认使用均匀加权
        w_avg, stats = _aggregate_uniform(
            client_weights=client_weights,
            aggregated_keys=aggregated_keys,
            w_avg=w_avg,
        )
    
    # 合并统计信息
    aggregation_info.update(stats)
    
    # LoRA-FAIR: 在基础聚合之后应用残差修正
    if algorithm == 'lorafair':
        lorafair_stats = _apply_lorafair_residual_correction(
            w_avg=w_avg,
            client_weights=client_weights,
            aggregated_keys=aggregated_keys,
            residual_mu=residual_mu,
        )
        aggregation_info.update(lorafair_stats)
    
    return w_avg, aggregation_info
