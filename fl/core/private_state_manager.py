# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
私有状态管理器

统一管理 FedSDG、Local-Only、FedRep、Ditto 等算法的客户端私有状态。

Unified interface for managing client-private states across algorithms.

Usage:
    manager = PrivateStateManager(algorithm='fedsdg')
    manager.save(client_idx=0, state_dict=private_w)
    state = manager.load(client_idx=0)
"""

from typing import Dict, Optional, Set
import torch


class PrivateStateManager:
    """
    统一的客户端私有状态管理器
    
    职责：
    1. 保存客户端私有状态（自动转移到 CPU）
    2. 加载客户端私有状态
    3. 判断算法是否需要私有状态管理
    
    设计原则：
    - 使用能力集（算法名集合）而非 Literal 类型，便于扩展
    - 自动转移到 CPU 以节省 GPU 内存
    - 支持检查点保存/恢复
    
    Attributes:
        algorithm: 算法名称
        enabled: 是否启用私有状态管理
        states: 所有客户端的私有状态字典
    """
    
    # 需要私有状态管理的算法集合
    # 使用集合而非 Literal 类型，便于扩展新算法
    ALGORITHMS_WITH_PRIVATE_STATE: Set[str] = {
        'fedsdg',      # 私有 LoRA 参数 + lambda_k
        'feddpa',      # 私有 LoRA 参数（无门控）
        'local_only',  # 完整 LoRA 参数（不聚合）
        'fedrep',      # 私有 Head 参数
        'ditto',       # 个性化模型参数
        'fedsalora',   # lora_B 参数（客户端特有知识）
        'pf2lora',     # Private LoRA 参数（自动秩学习）
        'fedtp',       # Phase 2: 私有 LoRA + Head 参数
        'fedalt',      # Individual LoRA + gate (FedALT-adapted)
    }
    
    def __init__(self, algorithm: str):
        """
        初始化私有状态管理器
        
        Args:
            algorithm: 算法名称（如 'fedsdg', 'fedavg' 等）
        """
        self.algorithm = algorithm.lower()
        self._states: Dict[int, Dict[str, torch.Tensor]] = {}
        self._enabled = self.algorithm in self.ALGORITHMS_WITH_PRIVATE_STATE
    
    @property
    def enabled(self) -> bool:
        """是否启用私有状态管理"""
        return self._enabled
    
    @property
    def states(self) -> Optional[Dict[int, Dict[str, torch.Tensor]]]:
        """
        获取所有私有状态（用于检查点保存、评估等）
        
        Returns:
            私有状态字典，如果未启用则返回 None
        """
        return self._states if self._enabled else None
    
    def save(
        self, 
        client_idx: int, 
        state_dict: Dict[str, torch.Tensor],
        to_cpu: bool = True
    ) -> None:
        """
        保存客户端私有状态
        
        Args:
            client_idx: 客户端索引
            state_dict: 状态字典（通常由 get_xxx_state_dict() 返回）
            to_cpu: 是否转移到 CPU（默认 True，节省 GPU 内存）
        
        Note:
            - 自动创建副本，不影响原始张量
            - 默认自动转移到 CPU
        """
        if not self._enabled:
            return
        
        private_state = {}
        for key, value in state_dict.items():
            if to_cpu:
                # detach + clone + cpu：确保完全独立的副本
                private_state[key] = value.detach().clone().cpu()
            else:
                private_state[key] = value.detach().clone()
        
        self._states[client_idx] = private_state
    
    def load(self, client_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        加载客户端私有状态
        
        Args:
            client_idx: 客户端索引
            
        Returns:
            私有状态字典，如果不存在则返回 None
        """
        if not self._enabled:
            return None
        return self._states.get(client_idx)
    
    def restore_all(self, states: Optional[Dict[int, Dict[str, torch.Tensor]]]) -> None:
        """
        恢复所有私有状态（用于检查点加载）
        
        Args:
            states: 私有状态字典（从检查点加载）
        """
        if states is not None and self._enabled:
            self._states = states
    
    def clear(self) -> None:
        """清空所有私有状态"""
        self._states.clear()
    
    def __len__(self) -> int:
        """返回已保存的客户端数量"""
        return len(self._states)
    
    def __contains__(self, client_idx: int) -> bool:
        """检查客户端是否有保存的状态"""
        return client_idx in self._states
    
    def __repr__(self) -> str:
        return (
            f"PrivateStateManager("
            f"algorithm='{self.algorithm}', "
            f"enabled={self._enabled}, "
            f"num_clients={len(self._states)})"
        )
