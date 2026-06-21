# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics collection module for federated learning.

Provides modular metrics collectors for different evaluation scenarios:
- GlobalMetricsCollector: Global model evaluation metrics
- LocalMetricsCollector: Local/personalized model metrics with statistics
- ClientMetricsCollector: Client training metrics aggregation
- FedSDGMetricsCollector: FedSDG-specific metrics (gate, regularization)
- DriftMetricsCollector: Client drift and global update diagnostics

Usage:
    from fl.utils.metrics import MetricsManager
    
    # Create from Hydra config
    metrics_manager = MetricsManager.from_hydra_config(hydra_cfg, algorithm='fedsdg')
    
    # Check if should evaluate this round
    if metrics_manager.should_evaluate(epoch, total_epochs):
        # Collect metrics
        local_metrics = metrics_manager.local.collect(client_results)
"""

from .config import MetricsConfig
from .base import BaseMetricsCollector
from .local_metrics import LocalMetricsCollector
from .client_metrics import ClientMetricsCollector
from .fedsdg_metrics import FedSDGMetricsCollector
from .drift_metrics import DriftMetricsCollector
from .manager import MetricsManager

__all__ = [
    'MetricsConfig',
    'BaseMetricsCollector',
    'LocalMetricsCollector',
    'ClientMetricsCollector',
    'FedSDGMetricsCollector',
    'DriftMetricsCollector',
    'MetricsManager',
]
