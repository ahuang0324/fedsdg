# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local/personalized metrics collector module.

Collects metrics for local model evaluation on client test subsets.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from .base import BaseMetricsCollector
from .config import MetricsConfig


class LocalMetricsCollector(BaseMetricsCollector):
    """
    Collector for local/personalized model metrics.
    
    Computes:
    - local/test_acc_avg: Average accuracy across clients
    - local/test_loss_avg: Average loss across clients
    - local/test_acc_std: Standard deviation of accuracies
    - local/test_acc_p10/p50/p90: Percentiles of accuracies
    """
    
    def __init__(self, config: MetricsConfig):
        super().__init__(config)
    
    def collect(
        self, 
        client_results: Dict[int, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Collect local test metrics from client results.
        
        Args:
            client_results: Dictionary mapping client_id to (accuracy, loss) tuple
            
        Returns:
            Dictionary of metrics
        """
        if not client_results:
            return {}
        
        accs = [acc for acc, _ in client_results.values()]
        losses = [loss for _, loss in client_results.values()]
        
        metrics = {
            'local/test_acc_avg': float(np.mean(accs)),
            'local/test_loss_avg': float(np.mean(losses)),
        }
        
        # Statistical metrics (configurable)
        if self.config.local_stats and len(accs) > 1:
            metrics.update({
                'local/test_acc_std': float(np.std(accs)),
                'local/test_acc_p10': float(np.percentile(accs, 10)),
                'local/test_acc_p50': float(np.percentile(accs, 50)),
                'local/test_acc_p90': float(np.percentile(accs, 90)),
            })
        
        return metrics
    
    def collect_from_list(
        self, 
        accuracies: List[float],
        losses: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Collect metrics from lists of accuracies and losses.
        
        Args:
            accuracies: List of client accuracies
            losses: Optional list of client losses
            
        Returns:
            Dictionary of metrics
        """
        if not accuracies:
            return {}
        
        metrics = {
            'local/test_acc_avg': float(np.mean(accuracies)),
        }
        
        if losses:
            metrics['local/test_loss_avg'] = float(np.mean(losses))
        
        # Statistical metrics
        if self.config.local_stats and len(accuracies) > 1:
            metrics.update({
                'local/test_acc_std': float(np.std(accuracies)),
                'local/test_acc_p10': float(np.percentile(accuracies, 10)),
                'local/test_acc_p50': float(np.percentile(accuracies, 50)),
                'local/test_acc_p90': float(np.percentile(accuracies, 90)),
            })
        
        return metrics
