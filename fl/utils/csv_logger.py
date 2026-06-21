# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV Logger for training metrics.

Generates metrics.csv for easy plotting and analysis.

Usage:
    from fl.utils.csv_logger import CSVLogger
    
    csv_logger = CSVLogger(log_dir='outputs/xxx')
    
    for epoch in range(100):
        metrics = {'round': epoch, 'val_acc_avg': 0.75, ...}
        csv_logger.log(metrics)
    
    csv_logger.close()
"""

import csv
import os
from typing import Dict, List, Optional, Any

from .console_logger import cprint


class CSVLogger:
    """
    CSV日志器，用于记录训练过程指标
    
    特点:
    1. 每轮追加一行
    2. 自动处理列顺序（预定义顺序 + 其他列）
    3. 缺失值用空字符串填充
    4. 支持动态添加新列
    """
    
    # 预定义列顺序（保证可读性和一致性）
    COLUMN_ORDER = [
        # 基本信息
        'round', 'epoch',
        # 训练指标
        'train_loss_avg', 'train_acc_avg',
        # 验证指标（过程监控和早停）
        'val_acc_avg', 'val_acc_std', 'val_acc_p10', 'val_acc_p50', 'val_acc_p90',
        'val_loss_avg',
        'global_val_acc', 'global_val_loss',
        # 测试指标（论文报告，计算最后N轮平均值）
        'test_acc_avg', 'test_acc_std', 'test_acc_p10', 'test_acc_p50', 'test_acc_p90',
        'test_loss_avg',
        'global_test_acc', 'global_test_loss',
        # 通信统计
        'comm_volume_round_mb', 'comm_volume_cumulative_mb', 'comm_volume_cumulative_gb',
        # FedSDG 门控值
        'gate_mean', 'gate_std', 'gate_min', 'gate_max',
        'gate_p10', 'gate_p50', 'gate_p90',
        'gate_attn_mean', 'gate_mlp_mean',
        # FedSDG 正则化损失
        'reg_loss_gate_mean', 'reg_loss_private_mean',
        # 时间
        'round_time_sec', 'cumulative_time_min',
    ]
    
    def __init__(self, log_dir: str, filename: str = "metrics.csv"):
        """
        初始化 CSV Logger
        
        Args:
            log_dir: 日志目录路径
            filename: CSV 文件名（默认 metrics.csv）
        """
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, filename)
        self.fieldnames: Optional[List[str]] = None
        self._initialized = False
        self._all_keys: set = set()
        
        cprint(f"[CSVLogger] Initialized: {self.csv_path}")
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """
        记录一行指标
        
        Args:
            metrics: 指标字典，必须包含 'round' 键
        """
        if 'round' not in metrics:
            raise ValueError("metrics must contain 'round' key")
        
        # 添加 epoch (= round + 1) 如果不存在
        if 'epoch' not in metrics:
            metrics = metrics.copy()
            metrics['epoch'] = metrics['round'] + 1
        
        # 首次调用时初始化
        if not self._initialized:
            self._initialize_csv(metrics)
        else:
            # 检查是否有新列（动态添加）
            new_keys = set(metrics.keys()) - self._all_keys
            if new_keys:
                self._add_new_columns(new_keys)
        
        # 写入一行（缺失值填充为空）
        row = {field: metrics.get(field, '') for field in self.fieldnames}
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
    
    def _initialize_csv(self, first_metrics: Dict[str, Any]) -> None:
        """初始化CSV文件（写入表头）"""
        # 按预定义顺序 + 其他列（字母排序）
        available_cols = set(first_metrics.keys())
        ordered_cols = [c for c in self.COLUMN_ORDER if c in available_cols]
        other_cols = sorted(available_cols - set(ordered_cols))
        self.fieldnames = ordered_cols + other_cols
        self._all_keys = set(self.fieldnames)
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        
        self._initialized = True
        cprint(f"[CSVLogger] CSV initialized with {len(self.fieldnames)} columns")
    
    def _add_new_columns(self, new_keys: set) -> None:
        """
        动态添加新列（需要重写整个文件）
        
        注意: 这是一个代价较高的操作，应尽量避免
        """
        import pandas as pd
        
        # 读取现有数据
        df = pd.read_csv(self.csv_path)
        
        # 添加新列
        for key in new_keys:
            df[key] = ''
        
        # 更新 fieldnames
        self.fieldnames = list(df.columns)
        self._all_keys = set(self.fieldnames)
        
        # 重写文件
        df.to_csv(self.csv_path, index=False)
        
        cprint(f"[CSVLogger] Added new columns: {new_keys}")
    
    def close(self) -> None:
        """关闭日志（当前实现无需显式关闭，但提供接口）"""
        if self._initialized:
            cprint(f"[CSVLogger] Closed: {self.csv_path}")
    
    def get_path(self) -> str:
        """获取 CSV 文件路径"""
        return self.csv_path
    
    def get_columns(self) -> Optional[List[str]]:
        """获取当前列名列表"""
        return self.fieldnames.copy() if self.fieldnames else None
