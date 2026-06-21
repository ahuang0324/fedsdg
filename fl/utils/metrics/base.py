# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base metrics collector module.

Defines abstract base class for all metrics collectors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from .config import MetricsConfig


class BaseMetricsCollector(ABC):
    """
    Abstract base class for metrics collectors.
    
    All metrics collectors should inherit from this class and implement
    the collect() method.
    """
    
    def __init__(self, config: MetricsConfig):
        """
        Initialize the metrics collector.
        
        Args:
            config: MetricsConfig instance
        """
        self.config = config
    
    @abstractmethod
    def collect(self, **kwargs) -> Dict[str, float]:
        """
        Collect metrics.
        
        Returns:
            Dictionary mapping metric names to values
        """
        pass
    
    def should_evaluate(self, epoch: int, total_epochs: int) -> bool:
        """
        Check if evaluation should be performed this round.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            True if evaluation should be performed
        """
        # Always evaluate at last epoch if configured
        if self.config.eval_at_last and epoch == total_epochs - 1:
            return True
        
        # Check eval_start_epoch: skip evaluation before start epoch
        start = self.config.eval_start_epoch
        if start < 0:
            # Negative: last N epochs (e.g., -10 with 60 epochs → start at epoch 50)
            start = max(0, total_epochs + start)
        if epoch < start:
            return False
        
        # Evaluate every eval_freq epochs (relative to start)
        return (epoch - start) % self.config.eval_freq == 0
