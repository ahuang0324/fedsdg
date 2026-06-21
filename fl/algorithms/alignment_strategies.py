# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对齐度计算策略模块

提供服务端聚合时计算客户端更新对齐度的策略。

可用策略：
- StandardMeanStrategy: 标准平均策略
- LOOMeanStrategy: Leave-One-Out 平均策略（推荐）

使用方式：
    from fl.algorithms.alignment_strategies import create_alignment_strategy
    
    strategy = create_alignment_strategy('loo_mean')
    cos_similarities, stats = strategy.compute_cosine_similarities(delta_stack, norm_deltas)
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Tuple


class AlignmentStrategy(ABC):
    """
    对齐度计算策略的抽象基类。
    
    设计原则：
    - 只计算余弦相似度，返回 torch.Tensor
    - 不做权重转换（ReLU、归一化等），由外层聚合器决定
    - 纯 torch 计算，不触发 GPU 同步（不调用 .item()）
    """
    
    @abstractmethod
    def compute_cosine_similarities(
        self,
        delta_stack: torch.Tensor,  # [M, D] float32
        norm_deltas: torch.Tensor,  # [M] float32
        epsilon: float = 1e-8
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算客户端余弦相似度。
        
        Args:
            delta_stack: 客户端更新向量堆叠 [M, param_dim]，float32
            norm_deltas: 每个 delta 的 L2 范数 [M]，float32
            epsilon: 数值稳定性参数
            
        Returns:
            cos_similarities: 余弦相似度张量 [M]（可能包含负值，未经 ReLU）
            strategy_stats: 策略特定的统计信息 Dict[str, torch.Tensor]（不做 .item()）
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """返回策略名称（用于日志和统计）"""
        pass


class StandardMeanStrategy(AlignmentStrategy):
    """
    标准平均策略：使用所有客户端的平均更新方向。
    
    计算：cos_k = <delta_k, mean(all deltas)> / (||delta_k|| * ||mean||)
    """
    
    def compute_cosine_similarities(
        self,
        delta_stack: torch.Tensor,
        norm_deltas: torch.Tensor,
        epsilon: float = 1e-8
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 转 float32 确保精度
        delta_stack = delta_stack.float()
        norm_deltas = norm_deltas.float()
        
        # 计算全局平均
        delta_mean = delta_stack.mean(dim=0)  # [D]
        norm_mean = torch.linalg.norm(delta_mean, ord=2)  # scalar tensor
        
        # 向量化计算余弦相似度
        # numerator[i] = dot(delta_stack[i], delta_mean)
        numerator = torch.matmul(delta_stack, delta_mean)  # [M]
        denominator = norm_deltas * norm_mean + epsilon    # [M]
        cos_similarities = numerator / denominator         # [M]
        
        strategy_stats = {
            'norm_mean': norm_mean,
        }
        return cos_similarities, strategy_stats
    
    def get_strategy_name(self) -> str:
        return 'standard_mean'


class LOOMeanStrategy(AlignmentStrategy):
    """
    Leave-One-Out (LOO) 平均策略：为每个客户端计算排除自身后的平均方向。
    
    优势：
    - 消除自相关偏差（客户端 k 不再与包含自身的平均方向对齐）
    - 更真实地反映客户端更新与"其他客户端共识"的对齐度
    - 可能放大客户端间的差异，使权重分布更不均匀
    
    计算：cos_k = <delta_k, mean(all deltas except k)> / (||delta_k|| * ||mean_loo||)
    
    优化实现：使用 delta_sum - delta_k 避免重复计算，复杂度 O(M) 而非 O(M²)
    """
    
    def compute_cosine_similarities(
        self,
        delta_stack: torch.Tensor,
        norm_deltas: torch.Tensor,
        epsilon: float = 1e-8
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        M = delta_stack.shape[0]
        
        # 转 float32 确保精度
        delta_stack = delta_stack.float()
        norm_deltas = norm_deltas.float()
        
        if M == 1:
            # 边界情况：回退到标准平均
            delta_mean = delta_stack.mean(dim=0)
            norm_mean = torch.linalg.norm(delta_mean, ord=2)
            numerator = torch.matmul(delta_stack, delta_mean)
            denominator = norm_deltas * norm_mean + epsilon
            cos_similarities = numerator / denominator
            
            strategy_stats = {
                'norm_loo_mean': norm_mean,
                'norm_loo_std': torch.tensor(0.0, device=delta_stack.device),
                'norm_loo_min': norm_mean,
                'norm_loo_max': norm_mean,
            }
            return cos_similarities, strategy_stats
        
        # LOO mean 向量化计算
        delta_sum = delta_stack.sum(dim=0)  # [D]
        
        # delta_mean_loo[i] = (delta_sum - delta_stack[i]) / (M - 1)
        # 广播: delta_sum [1, D] - delta_stack [M, D] -> [M, D]
        delta_mean_loo = (delta_sum.unsqueeze(0) - delta_stack) / (M - 1)  # [M, D]
        
        # 每个 LOO mean 的范数
        norm_loo_means = torch.linalg.norm(delta_mean_loo, ord=2, dim=1)  # [M]
        
        # 向量化计算余弦相似度
        # numerator[i] = dot(delta_stack[i], delta_mean_loo[i])
        numerator = (delta_stack * delta_mean_loo).sum(dim=1)  # [M]
        denominator = norm_deltas * norm_loo_means + epsilon   # [M]
        cos_similarities = numerator / denominator             # [M]
        
        strategy_stats = {
            'norm_loo_mean': norm_loo_means.mean(),
            'norm_loo_std': norm_loo_means.std(unbiased=False),
            'norm_loo_min': norm_loo_means.min(),
            'norm_loo_max': norm_loo_means.max(),
        }
        return cos_similarities, strategy_stats
    
    def get_strategy_name(self) -> str:
        return 'loo_mean'


def create_alignment_strategy(strategy_name: str = 'loo_mean') -> AlignmentStrategy:
    """
    策略工厂函数：根据名称创建对应的对齐度计算策略。
    
    Args:
        strategy_name: 策略名称
            - 'loo_mean': Leave-One-Out 平均策略（推荐，默认）
            - 'standard_mean': 标准平均策略（原始实现）
    
    Returns:
        AlignmentStrategy 实例
    """
    strategy_map = {
        'standard_mean': StandardMeanStrategy,
        'loo_mean': LOOMeanStrategy,
    }
    
    strategy_name_lower = strategy_name.lower()
    if strategy_name_lower not in strategy_map:
        available = ', '.join(strategy_map.keys())
        raise ValueError(
            f"Unknown alignment strategy: '{strategy_name}'. "
            f"Available strategies: {available}"
        )
    
    return strategy_map[strategy_name_lower]()
