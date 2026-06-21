# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Client training metrics collector module.

Aggregates training metrics from participating clients.
"""

from typing import Dict, List, Optional
import numpy as np

from .base import BaseMetricsCollector
from .config import MetricsConfig


class ClientMetricsCollector(BaseMetricsCollector):
    """
    Collector for client training metrics.
    
    Computes:
    - client/train_task_loss_mean/var/min/max: Task loss statistics
    - local/train_acc_avg: Average training accuracy
    """
    
    def __init__(self, config: MetricsConfig):
        super().__init__(config)
    
    def collect(
        self,
        train_metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Collect aggregated training metrics from all participating clients.
        
        Args:
            train_metrics_list: List of train_metrics dicts from each client
                Each dict contains: train_acc, task_loss, (reg_loss, lambda_values for FedSDG)
                
        Returns:
            Dictionary of aggregated metrics
        """
        if not train_metrics_list:
            return {}
        
        # Extract values
        train_accs = [m['train_acc'] for m in train_metrics_list if 'train_acc' in m]
        task_losses = [m['task_loss'] for m in train_metrics_list if 'task_loss' in m]
        
        metrics = {}
        
        # Training accuracy
        if train_accs:
            metrics['local/train_acc_avg'] = float(np.mean(train_accs))
        
        # Task loss statistics
        if task_losses:
            metrics.update({
                'client/train_task_loss_mean': float(np.mean(task_losses)),
                'client/train_task_loss_var': float(np.var(task_losses)),
                'client/train_task_loss_min': float(np.min(task_losses)),
                'client/train_task_loss_max': float(np.max(task_losses)),
            })
        
        return metrics
    
    def collect_from_losses(
        self,
        local_losses: List[float]
    ) -> Dict[str, float]:
        """
        Collect loss statistics from a list of losses.
        
        Args:
            local_losses: List of average losses from each client
            
        Returns:
            Dictionary of loss statistics
        """
        if not local_losses:
            return {}
        
        return {
            'client/train_task_loss_mean': float(np.mean(local_losses)),
            'client/train_task_loss_var': float(np.var(local_losses)),
            'client/train_task_loss_min': float(np.min(local_losses)),
            'client/train_task_loss_max': float(np.max(local_losses)),
        }
