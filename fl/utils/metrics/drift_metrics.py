# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drift diagnostics metrics collector module.

Computes client drift and global update magnitude metrics.
"""

from typing import Dict, List, Optional
import numpy as np
import torch

from .base import BaseMetricsCollector
from .config import MetricsConfig


class DriftMetricsCollector(BaseMetricsCollector):
    """
    Collector for drift and update diagnostics.
    
    Computes:
    - diagnostics/drift_norm_avg/std/p90: Client drift statistics
    - diagnostics/update_norm_global: Global update magnitude
    """
    
    def __init__(self, config: MetricsConfig):
        super().__init__(config)
    
    def collect(self, **kwargs) -> Dict[str, float]:
        """
        Collect drift metrics (placeholder for interface compliance).
        
        Use collect_client_drift() and collect_global_update() directly.
        """
        return {}
    
    def collect_client_drift(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        global_state: Dict[str, torch.Tensor],
        comm_keys: List[str]
    ) -> Dict[str, float]:
        """
        Compute client drift magnitudes.
        
        drift_k = ||w_k,end - w^(t)||_2 (only for comm_keys)
        
        Args:
            client_weights: List of client weight dicts after local training
            global_state: Global model state dict before training
            comm_keys: List of parameter names to consider
            
        Returns:
            Dictionary of drift statistics
        """
        if not self.config.drift:
            return {}
        
        if not client_weights or not comm_keys:
            return {}
        
        drift_norms = []
        for w in client_weights:
            drift_sq_sum = 0.0
            for key in comm_keys:
                if key in w and key in global_state:
                    delta = w[key].float().cpu() - global_state[key].float().cpu()
                    drift_sq_sum += (delta ** 2).sum().item()
            drift_norms.append(np.sqrt(drift_sq_sum))
        
        if not drift_norms:
            return {}
        
        metrics = {
            'diagnostics/drift_norm_avg': float(np.mean(drift_norms)),
            'diagnostics/drift_norm_std': float(np.std(drift_norms)),
        }
        
        if len(drift_norms) >= 3:
            metrics['diagnostics/drift_norm_p90'] = float(np.percentile(drift_norms, 90))
        
        return metrics
    
    def collect_global_update(
        self,
        new_state: Dict[str, torch.Tensor],
        old_state: Dict[str, torch.Tensor],
        comm_keys: List[str]
    ) -> Dict[str, float]:
        """
        Compute global update magnitude.
        
        update = ||w^(t+1) - w^(t)||_2 (only for comm_keys)
        
        Args:
            new_state: Global model state dict after aggregation
            old_state: Global model state dict before aggregation
            comm_keys: List of parameter names to consider
            
        Returns:
            Dictionary with update norm
        """
        if not self.config.drift:
            return {}
        
        if not comm_keys:
            return {}
        
        update_sq_sum = 0.0
        for key in comm_keys:
            if key in new_state and key in old_state:
                delta = new_state[key].float().cpu() - old_state[key].float().cpu()
                update_sq_sum += (delta ** 2).sum().item()
        
        return {
            'diagnostics/update_norm_global': float(np.sqrt(update_sq_sum)),
        }
