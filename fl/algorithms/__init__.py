# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习服务端聚合算法模块

核心设计：
- 统一入口：server_aggregate() 支持所有算法和聚合方式
- 两个维度分离：
  1. algorithm: 决定聚合哪些参数 (fedavg/fedlora/fedsdg)
  2. agg_method: 决定如何聚合 (fedavg/alignment)

使用方式：
    from fl.algorithms import server_aggregate
    
    # FedAvg + 均匀聚合
    new_state, info = server_aggregate(
        client_weights, global_state_dict,
        algorithm='fedavg', agg_method='fedavg'
    )
    
    # FedLoRA + 对齐度聚合
    new_state, info = server_aggregate(
        client_weights, global_state_dict,
        algorithm='fedlora', agg_method='alignment',
        alignment_strategy='loo_mean'
    )
    
    # FedSDG + 对齐度聚合
    new_state, info = server_aggregate(
        client_weights, global_state_dict,
        algorithm='fedsdg', agg_method='alignment',
        alignment_strategy='loo_mean'
    )
"""

# 统一聚合入口
from .aggregation import server_aggregate

# 对齐策略（供高级用户使用）
from .alignment_strategies import (
    AlignmentStrategy,
    StandardMeanStrategy,
    LOOMeanStrategy,
    create_alignment_strategy,
)

# FedDPA 推理工具
from .feddpa import FedDPAInference

# FedRep 训练函数和辅助函数
from .fedrep import (
    fedrep_update_weights,
    get_head_state_dict,
    get_backbone_state_dict,
    get_fedrep_aggregation_keys,
)

# Ditto 训练函数和辅助函数
from .ditto import (
    ditto_update_weights,
    get_ditto_aggregation_keys,
)

__all__ = [
    # 统一聚合入口
    'server_aggregate',
    
    # 对齐策略
    'AlignmentStrategy',
    'StandardMeanStrategy',
    'LOOMeanStrategy',
    'create_alignment_strategy',
    
    # FedDPA 推理工具
    'FedDPAInference',
    
    # FedRep 训练函数
    'fedrep_update_weights',
    'get_head_state_dict',
    'get_backbone_state_dict',
    'get_fedrep_aggregation_keys',
    
    # Ditto 训练函数
    'ditto_update_weights',
    'get_ditto_aggregation_keys',
]
