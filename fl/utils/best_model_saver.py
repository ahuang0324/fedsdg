# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Best Model Saver - Save Best Model Based on Validation Metric

简化设计：
1. 仅保存最佳模型（根据验证集指标）
2. 用于训练结束后的最终测试评估
3. 不实现断点续训（简化系统复杂度）
"""

import os
import torch
import time
from typing import Dict, Any, Optional

from .console_logger import cprint


class BestModelSaver:
    """
    最佳模型保存器
    
    特点:
    1. 监控验证集指标（如 val_acc_avg）
    2. 自动保存最佳模型
    3. 轻量级设计（仅保存模型权重和私有状态）
    
    Usage:
        saver = BestModelSaver(
            save_dir='outputs/exp_xxx',
            monitor_metric='val_acc_avg',
            mode='max'
        )
        
        for epoch in range(100):
            val_acc = validate(...)
            saver.save_if_best(epoch, model, val_acc, private_states)
        
        # 训练结束后加载最佳模型
        best_checkpoint = saver.load_best()
        model.load_state_dict(best_checkpoint['model_state_dict'])
    """
    
    def __init__(
        self,
        save_dir: str,
        monitor_metric: str = 'val_acc_avg',
        mode: str = 'max'
    ):
        """
        Args:
            save_dir: 保存目录
            monitor_metric: 监控的指标名称
            mode: 'max' 表示越大越好，'min' 表示越小越好
        """
        self.save_dir = save_dir
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.checkpoint_path = os.path.join(save_dir, 'checkpoint_best.pt')
        
        self.best_score = None
        self.best_epoch = None
        
        cprint(f"[BestModelSaver] 已初始化")
        cprint(f"  保存路径: {self.checkpoint_path}")
        cprint(f"  监控指标: {monitor_metric} ({mode})")
    
    def save_if_best(
        self,
        epoch: int,
        model: torch.nn.Module,
        current_score: float,
        local_private_states: Optional[Dict] = None,
        extra_info: Optional[Dict] = None
    ) -> bool:
        """
        如果当前模型更好，则保存
        
        Args:
            epoch: 当前轮次
            model: 全局模型
            current_score: 当前验证指标值
            local_private_states: FedSDG私有状态（如需要）
            extra_info: 额外信息（可选）
        
        Returns:
            True if saved, False otherwise
        """
        # 判断是否需要更新
        is_better = self._is_better(current_score)
        
        if is_better:
            self.best_score = current_score
            self.best_epoch = epoch
            
            # 构建checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self._to_cpu(model.state_dict()),
                'local_private_states': self._to_cpu(local_private_states) if local_private_states else None,
                'best_score': float(self.best_score),
                'monitor_metric': self.monitor_metric,
                'timestamp': time.time(),
            }
            
            if extra_info:
                checkpoint['extra_info'] = extra_info
            
            # 保存
            torch.save(checkpoint, self.checkpoint_path)
            
            cprint(f"[BestModelSaver] 保存最佳模型: "
                  f"{self.monitor_metric}={current_score:.4f} (epoch {epoch})")
            
            return True
        
        return False
    
    def load_best(self) -> Optional[Dict]:
        """加载最佳模型checkpoint"""
        if not os.path.exists(self.checkpoint_path):
            cprint(f"[BestModelSaver] 未找到 checkpoint: {self.checkpoint_path}")
            return None
        
        # PyTorch 2.6+ 默认 weights_only=True，但 checkpoint 包含 numpy 对象，需要设置为 False
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        cprint(f"[BestModelSaver] 加载最佳模型 (epoch {checkpoint['epoch']})")
        cprint(f"  最佳 {checkpoint['monitor_metric']}: {checkpoint['best_score']:.4f}")
        
        return checkpoint
    
    def get_best_epoch(self) -> Optional[int]:
        """获取最佳轮次"""
        return self.best_epoch
    
    def get_best_score(self) -> Optional[float]:
        """获取最佳分数"""
        return self.best_score
    
    def _is_better(self, current_score: float) -> bool:
        """判断当前分数是否更好"""
        if self.best_score is None:
            return True
        
        if self.mode == 'max':
            return current_score > self.best_score
        else:
            return current_score < self.best_score
    
    @staticmethod
    def _to_cpu(state: Any) -> Any:
        """将state移到CPU（节省GPU内存）"""
        if state is None:
            return None
        
        if isinstance(state, dict):
            return {k: BestModelSaver._to_cpu(v) for k, v in state.items()}
        elif isinstance(state, torch.Tensor):
            return state.cpu().clone()
        else:
            return state
