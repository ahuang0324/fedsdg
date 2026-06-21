# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
算法策略模式

Algorithm strategy abstractions used by the federated trainer.

每个算法策略封装：
1. 模型准备时是否需要注入私有状态
2. 训练后如何处理返回值（提取公共/私有权重）
3. 是否需要聚合
4. 聚合时使用哪些参数键

Usage:
    strategy = AlgorithmStrategyFactory.create(args.alg)
    
    # 在训练循环中
    private_state = strategy.get_private_state_for_model(private_state_manager, client_idx)
    public_w, private_w = strategy.process_training_output(output, client_idx, private_state_manager)
    if strategy.requires_aggregation:
        local_weights.append(public_w)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .private_state_manager import PrivateStateManager


class AlgorithmStrategy(ABC):
    """
    算法策略抽象基类
    
    定义联邦学习算法的核心行为：
    1. 获取模型准备时的私有状态
    2. 处理训练输出（提取公共/私有权重）
    3. 判断是否需要聚合
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """算法名称"""
        pass
    
    @property
    @abstractmethod
    def requires_aggregation(self) -> bool:
        """是否需要服务端聚合"""
        pass
    
    @property
    @abstractmethod
    def needs_private_state_for_model(self) -> bool:
        """模型准备时是否需要注入私有状态"""
        pass
    
    @property
    def needs_personal_state_for_training(self) -> bool:
        """训练时是否需要传入个性化状态（如 Ditto）"""
        return False
    
    def get_private_state_for_model(
        self,
        private_state_manager: 'PrivateStateManager',
        client_idx: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        获取模型准备时需要注入的私有状态
        
        Args:
            private_state_manager: 私有状态管理器
            client_idx: 客户端索引
            
        Returns:
            私有状态字典，或 None（如果不需要）
        """
        if self.needs_private_state_for_model:
            return private_state_manager.load(client_idx)
        return None
    
    def get_personal_state_for_training(
        self,
        private_state_manager: 'PrivateStateManager',
        client_idx: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        获取训练时需要传入的个性化状态（如 Ditto）
        
        Args:
            private_state_manager: 私有状态管理器
            client_idx: 客户端索引
            
        Returns:
            个性化状态字典，或 None（如果不需要）
        """
        if self.needs_personal_state_for_training:
            return private_state_manager.load(client_idx)
        return None
    
    @abstractmethod
    def process_training_output(
        self,
        output: Tuple,
        client_idx: int,
        private_state_manager: 'PrivateStateManager'
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float, Dict[str, Any]]:
        """
        处理训练输出
        
        Args:
            output: LocalUpdate.update_weights 的返回值
            client_idx: 客户端索引
            private_state_manager: 私有状态管理器
            
        Returns:
            (public_weights, loss, train_metrics)
            - public_weights: 用于聚合的公共权重，None 表示不参与聚合
            - loss: 训练损失
            - train_metrics: 训练指标
        """
        pass
    
    def get_model_pool_flags(self) -> Dict[str, bool]:
        """
        获取 ModelPool.prepare_model 需要的标志
        
        Returns:
            包含 is_fedsdg, is_local_only, is_fedrep, is_ditto 的字典
        """
        return {
            'is_fedsdg': False,
            'is_local_only': False,
            'is_fedrep': False,
            'is_ditto': False,
        }


class FedAvgStrategy(AlgorithmStrategy):
    """FedAvg/FedLoRA 策略"""
    
    @property
    def name(self) -> str:
        return 'fedavg'
    
    @property
    def requires_aggregation(self) -> bool:
        return True
    
    @property
    def needs_private_state_for_model(self) -> bool:
        return False
    
    def process_training_output(
        self,
        output: Tuple,
        client_idx: int,
        private_state_manager: 'PrivateStateManager'
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float, Dict[str, Any]]:
        # FedAvg/FedLoRA: 返回 (w, loss, train_metrics)
        w, loss, train_metrics = output
        return w, loss, train_metrics


class FedLoRAStrategy(FedAvgStrategy):
    """FedLoRA 策略（与 FedAvg 相同，仅名称不同）"""
    
    @property
    def name(self) -> str:
        return 'fedlora'


class FedProxAvgStrategy(FedAvgStrategy):
    """FedProx (基于 FedAvg) 策略 - 与 FedAvg 逻辑相同，proximal term 在训练时计算"""
    
    @property
    def name(self) -> str:
        return 'fedprox_avg'


class FedProxLoRAStrategy(FedAvgStrategy):
    """FedProx+LoRA 策略 - 与 FedLoRA 逻辑相同，proximal term 在训练时计算"""
    
    @property
    def name(self) -> str:
        return 'fedprox_lora'


class FedSDGStrategy(AlgorithmStrategy):
    """FedSDG 策略"""
    
    @property
    def name(self) -> str:
        return 'fedsdg'
    
    @property
    def requires_aggregation(self) -> bool:
        return True
    
    @property
    def needs_private_state_for_model(self) -> bool:
        return True
    
    def process_training_output(
        self,
        output: Tuple,
        client_idx: int,
        private_state_manager: 'PrivateStateManager'
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float, Dict[str, Any]]:
        # FedSDG: 返回 (public_w, private_w, loss, train_metrics)
        public_w, private_w, loss, train_metrics = output
        
        # 保存私有状态
        private_state_manager.save(client_idx, private_w)
        
        return public_w, loss, train_metrics
    
    def get_model_pool_flags(self) -> Dict[str, bool]:
        return {
            'is_fedsdg': True,
            'is_local_only': False,
            'is_fedrep': False,
            'is_ditto': False,
        }


class LocalOnlyStrategy(AlgorithmStrategy):
    """Local-Only 策略"""
    
    @property
    def name(self) -> str:
        return 'local_only'
    
    @property
    def requires_aggregation(self) -> bool:
        return False  # 不进行聚合
    
    @property
    def needs_private_state_for_model(self) -> bool:
        return True
    
    def process_training_output(
        self,
        output: Tuple,
        client_idx: int,
        private_state_manager: 'PrivateStateManager'
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float, Dict[str, Any]]:
        # Local-Only: 返回 (lora_w, loss, train_metrics)
        lora_w, loss, train_metrics = output
        
        # 保存 LoRA 状态
        private_state_manager.save(client_idx, lora_w)
        
        # 不返回公共权重（不参与聚合）
        return None, loss, train_metrics
    
    def get_model_pool_flags(self) -> Dict[str, bool]:
        return {
            'is_fedsdg': False,
            'is_local_only': True,
            'is_fedrep': False,
            'is_ditto': False,
        }


class FedRepStrategy(AlgorithmStrategy):
    """FedRep 策略"""
    
    @property
    def name(self) -> str:
        return 'fedrep'
    
    @property
    def requires_aggregation(self) -> bool:
        return True
    
    @property
    def needs_private_state_for_model(self) -> bool:
        return True
    
    def process_training_output(
        self,
        output: Tuple,
        client_idx: int,
        private_state_manager: 'PrivateStateManager'
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float, Dict[str, Any]]:
        # FedRep: 返回 (backbone_w, head_w, loss, train_metrics)
        backbone_w, head_w, loss, train_metrics = output
        
        # 保存 Head 状态
        private_state_manager.save(client_idx, head_w)
        
        # 返回 Backbone 权重用于聚合
        return backbone_w, loss, train_metrics
    
    def get_model_pool_flags(self) -> Dict[str, bool]:
        return {
            'is_fedsdg': False,
            'is_local_only': False,
            'is_fedrep': True,
            'is_ditto': False,
        }


class DittoStrategy(AlgorithmStrategy):
    """Ditto 策略"""
    
    @property
    def name(self) -> str:
        return 'ditto'
    
    @property
    def requires_aggregation(self) -> bool:
        return True
    
    @property
    def needs_private_state_for_model(self) -> bool:
        return False  # 模型准备时不注入，而是在训练时传入
    
    @property
    def needs_personal_state_for_training(self) -> bool:
        return True  # 训练时需要传入个性化状态
    
    def process_training_output(
        self,
        output: Tuple,
        client_idx: int,
        private_state_manager: 'PrivateStateManager'
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float, Dict[str, Any]]:
        # Ditto: 返回 (global_w, personal_w, loss, train_metrics)
        global_w, personal_w, loss, train_metrics = output
        
        # 保存个性化模型状态
        private_state_manager.save(client_idx, personal_w)
        
        # 返回全局模型权重用于聚合
        return global_w, loss, train_metrics
    
    def get_model_pool_flags(self) -> Dict[str, bool]:
        return {
            'is_fedsdg': False,
            'is_local_only': False,
            'is_fedrep': False,
            'is_ditto': True,
        }


class FedDPAStrategy(AlgorithmStrategy):
    """FedDPA 策略"""
    
    @property
    def name(self) -> str:
        return 'feddpa'
    
    @property
    def requires_aggregation(self) -> bool:
        return True  # Global LoRA 参与聚合
    
    @property
    def needs_private_state_for_model(self) -> bool:
        return True  # 需要注入 Private LoRA 状态
    
    def process_training_output(
        self,
        output: Tuple,
        client_idx: int,
        private_state_manager: 'PrivateStateManager'
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float, Dict[str, Any]]:
        # FedDPA: 返回 (global_w, private_w, loss, train_metrics)
        global_w, private_w, loss, train_metrics = output
        
        # 保存 Private 状态
        private_state_manager.save(client_idx, private_w)
        
        return global_w, loss, train_metrics
    
    def get_model_pool_flags(self) -> Dict[str, bool]:
        return {
            'is_fedsdg': False,
            'is_feddpa': True,
            'is_local_only': False,
            'is_fedrep': False,
            'is_ditto': False,
        }


class FedSALoRAStrategy(AlgorithmStrategy):
    """FedSA-LoRA 策略
    
    选择性聚合：仅 lora_A + Head 参与聚合，lora_B 本地保留。
    训练逻辑与 FedLoRA 完全一致（标准单路 LoRA，无额外正则化）。
    """
    
    @property
    def name(self) -> str:
        return 'fedsalora'
    
    @property
    def requires_aggregation(self) -> bool:
        return True  # lora_A 参与聚合
    
    @property
    def needs_private_state_for_model(self) -> bool:
        return True  # 需要注入客户端的 lora_B 矩阵
    
    def process_training_output(
        self,
        output: Tuple,
        client_idx: int,
        private_state_manager: 'PrivateStateManager'
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float, Dict[str, Any]]:
        # FedSA-LoRA: 返回 (public_w, private_w, loss, train_metrics)
        public_w, private_w, loss, train_metrics = output
        
        # 保存 lora_B 私有状态
        private_state_manager.save(client_idx, private_w)
        
        return public_w, loss, train_metrics
    
    def get_model_pool_flags(self) -> Dict[str, bool]:
        return {
            'is_fedsdg': False,
            'is_fedsalora': True,
            'is_local_only': False,
            'is_fedrep': False,
            'is_ditto': False,
        }


class PF2LoRAStrategy(AlgorithmStrategy):
    """PF2LoRA 策略
    
    双路加性 LoRA + 自动秩学习：
    - Shared LoRA (A_s, B_s) + Head 参与聚合
    - Private LoRA (A_p, B_p) 本地保留
    - 纯加性组合（gate 固定为 1.0，无门控机制）
    - 基于重要性分数的秩剪枝在 local_trainer 中执行
    """
    
    @property
    def name(self) -> str:
        return 'pf2lora'
    
    @property
    def requires_aggregation(self) -> bool:
        return True  # Shared LoRA 参与聚合
    
    @property
    def needs_private_state_for_model(self) -> bool:
        return True  # 需要注入客户端的 Private LoRA
    
    def process_training_output(
        self,
        output: Tuple,
        client_idx: int,
        private_state_manager: 'PrivateStateManager'
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float, Dict[str, Any]]:
        # PF2LoRA: 返回 (public_w, private_w, loss, train_metrics)
        public_w, private_w, loss, train_metrics = output
        
        # 保存 Private LoRA 私有状态
        private_state_manager.save(client_idx, private_w)
        
        return public_w, loss, train_metrics
    
    def get_model_pool_flags(self) -> Dict[str, bool]:
        return {
            'is_fedsdg': True,  # 复用 FedSDG 的双路架构（gate=1.0）
            'is_local_only': False,
            'is_fedrep': False,
            'is_ditto': False,
        }


class FedALTStrategy(AlgorithmStrategy):
    """FedALT-adapted 策略
    
    复用 FedSDG 双路 LoRA 架构，但反转训练和聚合角色：
    - Public 分支 (lora_A/B): Global LoRA，训练时冻结（聚合后的 RoW 参考）
    - Private 分支 (lora_A/B_private): Individual LoRA，每轮从聚合初始化 → 本地训练 → 上传聚合
    - Gate: Mixer，本地保留，跨轮次演化（唯一持久个性化组件）
    
    关键设计: Individual LoRA 每轮从聚合结果出发（类似 FedAvg），
    确保联邦知识传递。训练后的 Individual 提供评估时的个性化信号。
    """
    
    @property
    def name(self) -> str:
        return 'fedalt'
    
    @property
    def requires_aggregation(self) -> bool:
        return True
    
    @property
    def needs_private_state_for_model(self) -> bool:
        return True  # 需要注入 Individual LoRA + gate
    
    def process_training_output(
        self,
        output: Tuple,
        client_idx: int,
        private_state_manager: 'PrivateStateManager'
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float, Dict[str, Any]]:
        # FedALT: 返回 (public_w, private_w, loss, train_metrics)
        # public_w: Individual LoRA (renamed to public keys) + Head → 聚合
        # private_w: Individual LoRA (original keys) + gate → 本地保存
        public_w, private_w, loss, train_metrics = output
        
        # 保存私有状态 (Individual LoRA + gate)
        private_state_manager.save(client_idx, private_w)
        
        return public_w, loss, train_metrics
    
    def get_model_pool_flags(self) -> Dict[str, bool]:
        return {
            'is_fedsdg': True,  # 复用 FedSDG 的双路模型加载逻辑
            'is_local_only': False,
            'is_fedrep': False,
            'is_ditto': False,
        }


class LoRAFAIRStrategy(FedAvgStrategy):
    """LoRA-FAIR 策略
    
    标准单路 LoRA 训练，服务端聚合后应用残差修正。
    客户端行为与 FedLoRA 完全相同（无私有状态、无额外计算）。
    核心差异在服务端 aggregation.py 中的 ΔB 修正逻辑。
    """
    
    @property
    def name(self) -> str:
        return 'lorafair'


class FedTPStrategy(AlgorithmStrategy):
    """FedTP (Two-Phase LoRA) 策略
    
    两阶段训练：
    - Phase 1: 等价于 FedLoRA（全局 LoRA 聚合，私有冻结）
    - Phase 2: 等价于 Local-Only（私有 LoRA 本地训练，不聚合）
    
    Phase 由 trainer 通过 set_phase() 设置。
    """
    
    def __init__(self):
        self._current_phase = 1
        self._phase1_epochs = 50
    
    @property
    def name(self) -> str:
        return 'fedtp'
    
    @property
    def current_phase(self) -> int:
        return self._current_phase
    
    @property
    def requires_aggregation(self) -> bool:
        return self._current_phase == 1  # Phase 1 聚合，Phase 2 不聚合
    
    @property
    def needs_private_state_for_model(self) -> bool:
        return self._current_phase == 2  # Phase 2 需要注入私有状态
    
    def set_phase(self, epoch: int, phase1_epochs: int):
        """根据当前 epoch 设置 phase"""
        self._phase1_epochs = phase1_epochs
        self._current_phase = 1 if epoch < phase1_epochs else 2
    
    def process_training_output(
        self,
        output: Tuple,
        client_idx: int,
        private_state_manager: 'PrivateStateManager'
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float, Dict[str, Any]]:
        if self._current_phase == 1:
            # Phase 1: 返回全局权重用于聚合
            # update_weights 返回 (public_w, None, loss, train_metrics)
            public_w, _, loss, train_metrics = output
            return public_w, loss, train_metrics
        else:
            # Phase 2: 保存私有状态，不返回公共权重
            # update_weights 返回 (None, private_w, loss, train_metrics)
            _, private_w, loss, train_metrics = output
            private_state_manager.save(client_idx, private_w)
            return None, loss, train_metrics
    
    def get_model_pool_flags(self) -> Dict[str, bool]:
        return {
            'is_fedsdg': True,  # 复用 FedSDG 的双路架构
            'is_local_only': False,
            'is_fedrep': False,
            'is_ditto': False,
        }


class AlgorithmStrategyFactory:
    """算法策略工厂"""
    
    _strategies = {
        'fedavg': FedAvgStrategy,
        'fedlora': FedLoRAStrategy,
        'fedprox_avg': FedProxAvgStrategy,
        'fedprox_lora': FedProxLoRAStrategy,
        'fedsdg': FedSDGStrategy,
        'feddpa': FedDPAStrategy,
        'local_only': LocalOnlyStrategy,
        'fedrep': FedRepStrategy,
        'ditto': DittoStrategy,
        'fedsalora': FedSALoRAStrategy,
        'pf2lora': PF2LoRAStrategy,
        'fedtp': FedTPStrategy,
        'lorafair': LoRAFAIRStrategy,
        'fedalt': FedALTStrategy,
    }
    
    @classmethod
    def create(cls, algorithm: str) -> AlgorithmStrategy:
        """
        创建算法策略实例
        
        Args:
            algorithm: 算法名称
            
        Returns:
            AlgorithmStrategy 实例
            
        Raises:
            ValueError: 如果算法不支持
        """
        if algorithm not in cls._strategies:
            raise ValueError(f"Unsupported algorithm: {algorithm}. "
                           f"Supported: {list(cls._strategies.keys())}")
        return cls._strategies[algorithm]()
    
    @classmethod
    def register(cls, name: str, strategy_class: type) -> None:
        """
        注册新的算法策略
        
        Args:
            name: 算法名称
            strategy_class: 策略类
        """
        cls._strategies[name] = strategy_class
    
    @classmethod
    def supported_algorithms(cls) -> List[str]:
        """返回支持的算法列表"""
        return list(cls._strategies.keys())
