# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedSDG-specific metrics collector module.

Collects gate and regularization metrics for FedSDG algorithm.
"""

from typing import Dict, List, Optional
import numpy as np

from .base import BaseMetricsCollector
from .config import MetricsConfig


class FedSDGMetricsCollector(BaseMetricsCollector):
    """
    Collector for FedSDG-specific metrics.
    
    Computes:
    - gate/lambda_mean/std/min/max/p10/p50/p90: Gate statistics
    - fedsdg/train_reg_loss_mean/var/min/max: Regularization loss statistics
    """
    
    def __init__(self, config: MetricsConfig):
        super().__init__(config)
    
    def collect(
        self,
        train_metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Collect FedSDG-specific metrics from training results.
        
        Args:
            train_metrics_list: List of train_metrics dicts from each client
                Each dict should contain: reg_loss, lambda_values
                
        Returns:
            Dictionary of FedSDG metrics
        """
        metrics = {}
        
        # Collect gate metrics
        if self.config.fedsdg_gate:
            gate_metrics = self._collect_gate_stats(train_metrics_list)
            metrics.update(gate_metrics)
        
        # Collect regularization loss metrics
        if self.config.fedsdg_reg:
            reg_metrics = self._collect_reg_loss_stats(train_metrics_list)
            metrics.update(reg_metrics)
        
        return metrics
    
    def _collect_gate_stats(
        self,
        train_metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Collect gate (lambda) statistics across clients.
        
        For each client, first average lambda values across layers,
        then compute statistics across clients.
        
        Args:
            train_metrics_list: List of train_metrics dicts
            
        Returns:
            Dictionary of gate statistics
        """
        # Extract lambda values from each client
        client_lambda_values = []
        for m in train_metrics_list:
            if 'lambda_values' in m and m['lambda_values']:
                client_lambda_values.append(m['lambda_values'])
        
        if not client_lambda_values:
            return {}
        
        # Compute per-client mean lambda (average across layers)
        client_means = [float(np.mean(lambdas)) for lambdas in client_lambda_values]
        
        if not client_means:
            return {}
        
        metrics = {
            'gate/lambda_mean': float(np.mean(client_means)),
            'gate/lambda_std': float(np.std(client_means)),
            'gate/lambda_min': float(np.min(client_means)),
            'gate/lambda_max': float(np.max(client_means)),
        }
        
        # Percentiles (only if enough clients)
        if len(client_means) >= 3:
            metrics.update({
                'gate/lambda_p10': float(np.percentile(client_means, 10)),
                'gate/lambda_p50': float(np.percentile(client_means, 50)),
                'gate/lambda_p90': float(np.percentile(client_means, 90)),
            })
        
        return metrics
    
    def _collect_reg_loss_stats(
        self,
        train_metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Collect regularization loss statistics.
        
        分别收集门控正则化损失和私有参数正则化损失。
        
        Args:
            train_metrics_list: List of train_metrics dicts
            
        Returns:
            Dictionary of regularization loss statistics
        """
        metrics = {}
        
        # 收集总和（保持向后兼容）
        reg_losses = [m['reg_loss'] for m in train_metrics_list if 'reg_loss' in m]
        if reg_losses:
            metrics.update({
                'fedsdg/train_reg_loss_mean': float(np.mean(reg_losses)),
                'fedsdg/train_reg_loss_var': float(np.var(reg_losses)),
                'fedsdg/train_reg_loss_min': float(np.min(reg_losses)),
                'fedsdg/train_reg_loss_max': float(np.max(reg_losses)),
            })
        
        # 分别收集门控和私有参数的正则化损失
        reg_loss_gates = [m['reg_loss_gate'] for m in train_metrics_list if 'reg_loss_gate' in m]
        if reg_loss_gates:
            metrics.update({
                'fedsdg/train_reg_loss_gate_mean': float(np.mean(reg_loss_gates)),
                'fedsdg/train_reg_loss_gate_var': float(np.var(reg_loss_gates)),
                'fedsdg/train_reg_loss_gate_min': float(np.min(reg_loss_gates)),
                'fedsdg/train_reg_loss_gate_max': float(np.max(reg_loss_gates)),
            })
        
        reg_loss_privates = [m['reg_loss_private'] for m in train_metrics_list if 'reg_loss_private' in m]
        if reg_loss_privates:
            metrics.update({
                'fedsdg/train_reg_loss_private_mean': float(np.mean(reg_loss_privates)),
                'fedsdg/train_reg_loss_private_var': float(np.var(reg_loss_privates)),
                'fedsdg/train_reg_loss_private_min': float(np.min(reg_loss_privates)),
                'fedsdg/train_reg_loss_private_max': float(np.max(reg_loss_privates)),
            })
        
        return metrics
    
    def collect_gate_stats_from_values(
        self,
        client_lambda_values: List[List[float]]
    ) -> Dict[str, float]:
        """
        Collect gate statistics from raw lambda values.
        
        Args:
            client_lambda_values: List of lambda value lists for each client
                [[client_0 lambdas], [client_1 lambdas], ...]
                
        Returns:
            Dictionary of gate statistics
        """
        if not self.config.fedsdg_gate:
            return {}
        
        if not client_lambda_values:
            return {}
        
        # Compute per-client mean lambda
        client_means = [float(np.mean(lambdas)) for lambdas in client_lambda_values if lambdas]
        
        if not client_means:
            return {}
        
        metrics = {
            'gate/lambda_mean': float(np.mean(client_means)),
            'gate/lambda_std': float(np.std(client_means)),
            'gate/lambda_min': float(np.min(client_means)),
            'gate/lambda_max': float(np.max(client_means)),
        }
        
        if len(client_means) >= 3:
            metrics.update({
                'gate/lambda_p10': float(np.percentile(client_means, 10)),
                'gate/lambda_p50': float(np.percentile(client_means, 50)),
                'gate/lambda_p90': float(np.percentile(client_means, 90)),
            })
        
        return metrics
