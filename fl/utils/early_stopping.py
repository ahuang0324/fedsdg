# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Early Stopping for Federated Learning.

Monitors validation performance and stops training when no improvement.
"""

from .console_logger import cprint


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.
    
    Usage:
        early_stopper = EarlyStopping(patience=20, min_delta=0.001, mode='max')
        
        for epoch in range(max_epochs):
            val_acc = validate(...)
            
            if early_stopper.should_stop(val_acc, epoch):
                print(f"Early stopping at epoch {epoch}")
                best_epoch = early_stopper.get_best_epoch()
                break
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics like accuracy, 'min' for metrics like loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.stopped = False
        self.stop_epoch = None
    
    def should_stop(self, current_score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current validation metric value
            epoch: Current epoch number
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            cprint(f"[EarlyStopping] Initial best: {current_score:.4f} at epoch {epoch}")
            return False
        
        # Check improvement
        if self.mode == 'max':
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta
        
        if improved:
            cprint(f"[EarlyStopping] New best: {current_score:.4f} at epoch {epoch} "
                   f"(improved by {abs(current_score - self.best_score):.4f})")
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            cprint(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.stopped = True
                self.stop_epoch = epoch
                cprint(f"\n{'='*70}")
                cprint(f"[EarlyStopping] Triggered!")
                cprint(f"  Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                cprint(f"  Stopped at: epoch {epoch}")
                cprint(f"{'='*70}\n")
                return True
        
        return False
    
    def get_best_epoch(self) -> int:
        """Get the epoch with best validation score."""
        return self.best_epoch
    
    def get_best_score(self) -> float:
        """Get the best validation score."""
        return self.best_score
    
    def state_dict(self) -> dict:
        """Get state dict for checkpoint."""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
            'stopped': self.stopped,
            'stop_epoch': self.stop_epoch,
        }
    
    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.best_score = state['best_score']
        self.best_epoch = state['best_epoch']
        self.counter = state['counter']
        self.stopped = state['stopped']
        self.stop_epoch = state['stop_epoch']
