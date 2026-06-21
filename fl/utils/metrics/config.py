# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics configuration module.

Defines configuration dataclass for metrics collection.
"""

from dataclasses import dataclass


@dataclass
class MetricsConfig:
    """
    Metrics collection configuration.
    
    Attributes:
        eval_freq: Evaluation frequency (every N rounds)
        eval_at_last: Force evaluation at last round
        eval_start_epoch: Epoch to start evaluation. Negative means last N epochs.
                          e.g., -10 with epochs=60 means evaluate from epoch 50.
                          0 means evaluate from the beginning (default).
        local_stats: Enable local statistics (std, percentiles)
        drift: Enable drift diagnostics
        fedsdg_gate: Enable FedSDG gate metrics
        fedsdg_reg: Enable FedSDG regularization metrics
    """
    eval_freq: int = 5
    eval_at_last: bool = True
    eval_start_epoch: int = 0
    local_stats: bool = True
    drift: bool = True
    fedsdg_gate: bool = True
    fedsdg_reg: bool = True
    
    @classmethod
    def from_hydra_config(cls, hydra_cfg) -> 'MetricsConfig':
        """
        Create MetricsConfig from Hydra configuration.
        
        Args:
            hydra_cfg: Hydra DictConfig object
            
        Returns:
            MetricsConfig instance
        """
        training_cfg = getattr(hydra_cfg, 'training', None) if hydra_cfg else None
        eval_cfg = getattr(training_cfg, 'evaluation', None) if training_cfg else None
        metrics_cfg = getattr(training_cfg, 'metrics', None) if training_cfg else None
        
        return cls(
            eval_freq=getattr(eval_cfg, 'eval_freq', 5) if eval_cfg else 5,
            eval_at_last=getattr(eval_cfg, 'eval_at_last', True) if eval_cfg else True,
            eval_start_epoch=getattr(eval_cfg, 'eval_start_epoch', 0) if eval_cfg else 0,
            local_stats=getattr(metrics_cfg, 'local_stats', True) if metrics_cfg else True,
            drift=getattr(metrics_cfg, 'drift', True) if metrics_cfg else True,
            fedsdg_gate=getattr(metrics_cfg, 'fedsdg_gate', True) if metrics_cfg else True,
            fedsdg_reg=getattr(metrics_cfg, 'fedsdg_reg', True) if metrics_cfg else True,
        )
