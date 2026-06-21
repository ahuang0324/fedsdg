# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习核心训练模块

提供 FederatedTrainer 类，封装完整的联邦学习训练流程。

Usage:
    from fl.core import FederatedTrainer
    
    trainer = FederatedTrainer(args, hydra_cfg=cfg, hydra_run_dir=run_dir)
    trainer.run()
    
    # 或单独使用模型池
    from fl.core import ModelPool
"""

from .trainer import FederatedTrainer, TrainingState
from .model_pool import ModelPool
from .private_state_manager import PrivateStateManager
from .trainer_components import TrainerComponentsFactory
from .algorithm_strategy import (
    AlgorithmStrategy, AlgorithmStrategyFactory,
    FedAvgStrategy, FedLoRAStrategy, FedSDGStrategy,
    LocalOnlyStrategy, FedRepStrategy, DittoStrategy,
    FedSALoRAStrategy, PF2LoRAStrategy, FedTPStrategy,
    LoRAFAIRStrategy,
)

__all__ = [
    # 训练器
    'FederatedTrainer',
    'TrainingState',
    # 模型池（内存优化）
    'ModelPool',
    # 私有状态管理器
    'PrivateStateManager',
    # 组件工厂
    'TrainerComponentsFactory',
    # 算法策略
    'AlgorithmStrategy', 'AlgorithmStrategyFactory',
    'FedAvgStrategy', 'FedLoRAStrategy', 'FedSDGStrategy',
    'LocalOnlyStrategy', 'FedRepStrategy', 'DittoStrategy',
    'FedSALoRAStrategy', 'PF2LoRAStrategy',
    'FedTPStrategy', 'LoRAFAIRStrategy',
]
