# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
算法基类和接口定义

设计原则：
- 每个算法实现独立的算法类
- 算法类负责封装算法特定的逻辑（训练、评估、状态管理）
- 通过接口统一，消除硬编码的 if/else 判断

Usage:
    from fl.algorithms import get_algorithm
    
    alg = get_algorithm('fedsdg', args)
    optimizer = alg.create_optimizer(model)
    local_weights, extra_info = alg.train_client(model, ...)
    metrics = alg.evaluate(model, ...)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn


class Algorithm(ABC):
    """
    联邦学习算法基类
    
    所有算法需要实现以下接口：
    - create_optimizer: 创建优化器（算法特定的参数分组）
    - train_client: 客户端训练逻辑（算法特定的损失函数）
    - evaluate_client: 客户端评估逻辑（算法特定的模型状态）
    - aggregate: 服务端聚合逻辑（已统一，但可以覆盖）
    - get_state_dict: 获取需要上传的权重
    - initialize_state: 初始化算法特定的状态
    - update_state: 更新算法特定的状态
    """
    
    def __init__(self, args):
        self.args = args
        self.name = args.alg
    
    @abstractmethod
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        创建优化器（算法特定的参数分组和配置）
        
        Args:
            model: 模型实例
            
        Returns:
            优化器实例
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self, 
        model: nn.Module, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        criterion: nn.Module
    ) -> torch.Tensor:
        """
        计算损失（算法特定的正则化项）
        
        Args:
            model: 模型实例
            logits: 模型输出
            labels: 真实标签
            criterion: 基础损失函数
            
        Returns:
            总损失值
        """
        pass
    
    @abstractmethod
    def get_state_dict(
        self, 
        model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        获取需要上传的权重和私有状态
        
        Args:
            model: 模型实例
            
        Returns:
            (public_state_dict, private_state_dict)
            - public_state_dict: 用于服务端聚合的权重
            - private_state_dict: 客户端私有的权重（如 FedSDG），如果不存在则返回 None
        """
        pass
    
    @abstractmethod
    def prepare_model_for_training(
        self, 
        model: nn.Module, 
        global_state: Dict[str, torch.Tensor],
        private_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> nn.Module:
        """
        准备模型用于训练（注入全局状态和私有状态）
        
        Args:
            model: 模型实例
            global_state: 全局模型状态
            private_state: 客户端私有状态（可选）
            
        Returns:
            准备好的模型实例
        """
        pass
    
    @abstractmethod
    def prepare_model_for_evaluation(
        self, 
        model: nn.Module,
        private_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> nn.Module:
        """
        准备模型用于评估（可能需要禁用某些分支）
        
        Args:
            model: 模型实例
            private_state: 客户端私有状态（可选）
            
        Returns:
            准备好的模型实例
        """
        pass
    
    def initialize_state(self) -> Dict[str, Any]:
        """
        初始化算法特定的训练状态
        
        Returns:
            状态字典
        """
        return {}
    
    def update_state(
        self, 
        state: Dict[str, Any], 
        client_idx: int, 
        private_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """
        更新算法特定的训练状态
        
        Args:
            state: 状态字典
            client_idx: 客户端索引
            private_state: 客户端私有状态（可选）
        """
        pass
    
    def get_aggregation_keys(
        self, 
        client_weights: List[Dict[str, torch.Tensor]]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        获取聚合相关的参数键
        
        Args:
            client_weights: 客户端权重列表
            
        Returns:
            (aggregated_keys, align_keys, excluded_keys)
        """
        if not client_weights:
            return [], [], []
        return list(client_weights[0].keys()), list(client_weights[0].keys()), []
    
    def requires_grad_clip(self) -> bool:
        """是否需要梯度裁剪"""
        return False
    
    def get_grad_clip_value(self) -> float:
        """获取梯度裁剪值"""
        return 0.0
    
    def requires_private_state(self) -> bool:
        """是否需要管理私有状态"""
        return False


# =============================================================================
# 算法注册机制
# =============================================================================

_ALGORITHM_REGISTRY: Dict[str, type] = {}


def register_algorithm(name: str, algorithm_class: type):
    """
    注册算法类
    
    Args:
        name: 算法名称
        algorithm_class: 算法类
    """
    _ALGORITHM_REGISTRY[name] = algorithm_class


def get_algorithm(name: str, args) -> Algorithm:
    """
    获取算法实例
    
    Args:
        name: 算法名称
        args: 配置对象
        
    Returns:
        算法实例
        
    Raises:
        ValueError: 如果算法未注册
    """
    if name not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"算法 '{name}' 未注册。"
            f"可用算法: {list(_ALGORITHM_REGISTRY.keys())}"
        )
    return _ALGORITHM_REGISTRY[name](args)


def get_available_algorithms() -> List[str]:
    """获取已注册的算法列表"""
    return list(_ALGORITHM_REGISTRY.keys())
