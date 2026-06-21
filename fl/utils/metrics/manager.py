# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics manager module.

Provides unified interface for all metrics collectors.
"""

from typing import Dict, List, Tuple, Optional, Any
import torch

from .config import MetricsConfig
from .local_metrics import LocalMetricsCollector
from .client_metrics import ClientMetricsCollector
from .fedsdg_metrics import FedSDGMetricsCollector
from .drift_metrics import DriftMetricsCollector


class MetricsManager:
    """
    Unified metrics manager for federated learning.
    
    Provides a single interface to manage all metrics collectors
    and coordinate metric collection.
    
    Attributes:
        config: MetricsConfig instance
        algorithm: Algorithm name ('fedavg', 'fedlora', 'fedsdg', etc.)
        local: LocalMetricsCollector instance
        client: ClientMetricsCollector instance
        drift: DriftMetricsCollector instance
        fedsdg: FedSDGMetricsCollector instance (only for FedSDG)
    """
    
    def __init__(self, config: MetricsConfig, algorithm: str):
        """
        Initialize the metrics manager.
        
        Args:
            config: MetricsConfig instance
            algorithm: Algorithm name
        """
        self.config = config
        self.algorithm = algorithm
        
        # Initialize collectors
        self.local = LocalMetricsCollector(config)
        self.client = ClientMetricsCollector(config)
        self.drift = DriftMetricsCollector(config)
        
        # FedSDG-specific collector
        self.fedsdg = FedSDGMetricsCollector(config) if algorithm == 'fedsdg' else None
    
    def should_evaluate(self, epoch: int, total_epochs: int) -> bool:
        """
        Check if evaluation should be performed this round.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            True if evaluation should be performed
        """
        return self.local.should_evaluate(epoch, total_epochs)
    
    def collect_round_metrics(
        self,
        train_metrics_list: List[Dict[str, float]],
        local_losses: List[float],
        client_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
        global_state_before: Optional[Dict[str, torch.Tensor]] = None,
        global_state_after: Optional[Dict[str, torch.Tensor]] = None,
        comm_keys: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Collect all round-level metrics.
        
        Args:
            train_metrics_list: List of train_metrics from each client
            local_losses: List of average losses from each client
            client_weights: List of client weights (for drift calculation)
            global_state_before: Global state before aggregation
            global_state_after: Global state after aggregation
            comm_keys: List of communication parameter keys
            
        Returns:
            Dictionary of all collected metrics
        """
        metrics = {}
        
        # Client training metrics
        client_metrics = self.client.collect(train_metrics_list)
        metrics.update(client_metrics)
        
        # FedSDG-specific metrics
        if self.fedsdg is not None:
            fedsdg_metrics = self.fedsdg.collect(train_metrics_list)
            metrics.update(fedsdg_metrics)
        
        # Drift metrics (if data provided)
        if client_weights and global_state_before and comm_keys:
            drift_metrics = self.drift.collect_client_drift(
                client_weights, global_state_before, comm_keys
            )
            metrics.update(drift_metrics)
        
        # Global update metrics (if data provided)
        if global_state_before and global_state_after and comm_keys:
            update_metrics = self.drift.collect_global_update(
                global_state_after, global_state_before, comm_keys
            )
            metrics.update(update_metrics)
        
        return metrics
    
    def collect_eval_metrics(
        self,
        client_results: Dict[int, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Collect evaluation metrics from client test results.
        
        Args:
            client_results: Dictionary mapping client_id to (accuracy, loss)
            
        Returns:
            Dictionary of local evaluation metrics
        """
        return self.local.collect(client_results)
    
    @classmethod
    def from_hydra_config(cls, hydra_cfg, algorithm: str) -> 'MetricsManager':
        """
        Create MetricsManager from Hydra configuration.
        
        Args:
            hydra_cfg: Hydra DictConfig object
            algorithm: Algorithm name
            
        Returns:
            MetricsManager instance
        """
        config = MetricsConfig.from_hydra_config(hydra_cfg)
        return cls(config, algorithm)
