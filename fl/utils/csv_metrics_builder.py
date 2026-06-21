# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV 指标构建器

Builds CSV rows from internal training and evaluation metrics.

Usage:
    builder = CSVMetricsBuilder(
        comm_stats=comm_stats,
        algorithm='fedsdg',
        num_users=100,
        frac=0.1
    )
    csv_row = builder.build(
        epoch=0,
        loss_avg=0.5,
        train_acc=0.7,
        round_metrics=round_metrics,
        round_time=10.5,
        metrics=metrics
    )
    csv_logger.log(csv_row)
"""

from typing import Dict, List, Any, Optional
import numpy as np


class CSVMetricsBuilder:
    """
    CSV 指标构建器
    
    职责：
    1. 将内部指标格式转换为 CSV 列格式
    2. 管理累计时间
    3. 提取客户端统计量（std, p10, p50, p90）
    
    设计原则：
    - 无状态构建（除累计时间外）
    - 与 trainer 解耦
    - 支持 FedSDG 特有指标
    """
    
    def __init__(
        self,
        comm_stats: Dict[str, Any],
        algorithm: str,
        num_users: int,
        frac: float
    ):
        """
        初始化 CSV 指标构建器
        
        Args:
            comm_stats: 通信统计字典（包含 comm_size_mb）
            algorithm: 算法名称
            num_users: 总客户端数
            frac: 参与率
        """
        self.comm_stats = comm_stats
        self.algorithm = algorithm
        self.num_users = num_users
        self.frac = frac
        self._cumulative_time: float = 0.0
    
    def build(
        self,
        epoch: int,
        loss_avg: float,
        train_acc: Optional[float],
        round_metrics: Dict[str, Any],
        round_time: float,
        cumulative_comm_mb: float,
        metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        构建一行 CSV 指标
        
        Args:
            epoch: 当前轮次（0-indexed）
            loss_avg: 平均训练损失
            train_acc: 训练准确率
            round_metrics: 轮次评估指标
            round_time: 本轮时间（秒）
            cumulative_comm_mb: 累计通信量（MB）
            metrics: 额外指标字典（用于 FedSDG 特有指标）
        
        Returns:
            CSV 行字典
        """
        # 累加时间
        self._cumulative_time += round_time
        
        # 计算本轮通信量
        num_clients_per_round = max(int(self.num_users * self.frac), 1)
        round_comm_mb = 2 * num_clients_per_round * self.comm_stats['comm_size_mb']
        
        csv_row = {
            'round': epoch + 1,  # 1-indexed for readability
            'train_loss_avg': loss_avg,
            'train_acc_avg': train_acc,
            'comm_volume_round_mb': round_comm_mb,
            'comm_volume_cumulative_mb': cumulative_comm_mb,
            'comm_volume_cumulative_gb': cumulative_comm_mb / 1024,
            'round_time_sec': round_time,
            'cumulative_time_min': self._cumulative_time / 60,
        }
        
        # Val 集指标
        self._add_val_metrics(csv_row, round_metrics)
        
        # Test 集指标
        self._add_test_metrics(csv_row, round_metrics)
        
        # FedSDG 特有指标
        if metrics and self.algorithm == 'fedsdg':
            self._add_fedsdg_metrics(csv_row, metrics)
        
        return csv_row
    
    def _add_val_metrics(self, csv_row: Dict[str, Any], round_metrics: Dict[str, Any]) -> None:
        """添加 Val 集指标"""
        if round_metrics.get('local_val_acc') is not None:
            csv_row['val_acc_avg'] = round_metrics['local_val_acc']
            csv_row['val_loss_avg'] = round_metrics['local_val_loss']
            csv_row['global_val_acc'] = round_metrics.get('val_acc')
            csv_row['global_val_loss'] = round_metrics.get('val_loss')
            
            # Val 集客户端统计量
            if round_metrics.get('val_client_results'):
                stats = self._extract_client_stats(round_metrics['val_client_results'])
                csv_row['val_acc_std'] = stats['std']
                csv_row['val_acc_p10'] = stats['p10']
                csv_row['val_acc_p50'] = stats['p50']
                csv_row['val_acc_p90'] = stats['p90']
    
    def _add_test_metrics(self, csv_row: Dict[str, Any], round_metrics: Dict[str, Any]) -> None:
        """添加 Test 集指标"""
        if round_metrics.get('local_test_acc') is not None:
            csv_row['test_acc_avg'] = round_metrics['local_test_acc']
            csv_row['test_loss_avg'] = round_metrics['local_test_loss']
            csv_row['global_test_acc'] = round_metrics.get('test_acc')
            csv_row['global_test_loss'] = round_metrics.get('test_loss')
            
            # Test 集客户端统计量
            if round_metrics.get('test_client_results'):
                stats = self._extract_client_stats(round_metrics['test_client_results'])
                csv_row['test_acc_std'] = stats['std']
                csv_row['test_acc_p10'] = stats['p10']
                csv_row['test_acc_p50'] = stats['p50']
                csv_row['test_acc_p90'] = stats['p90']
    
    def _add_fedsdg_metrics(self, csv_row: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """添加 FedSDG 特有指标"""
        # 门控值统计
        if 'gate/lambda_mean' in metrics:
            csv_row['gate_mean'] = metrics['gate/lambda_mean']
        if 'gate/lambda_std' in metrics:
            csv_row['gate_std'] = metrics['gate/lambda_std']
        if 'gate/lambda_min' in metrics:
            csv_row['gate_min'] = metrics['gate/lambda_min']
        if 'gate/lambda_max' in metrics:
            csv_row['gate_max'] = metrics['gate/lambda_max']
        if 'gate/lambda_p10' in metrics:
            csv_row['gate_p10'] = metrics['gate/lambda_p10']
        if 'gate/lambda_p50' in metrics:
            csv_row['gate_p50'] = metrics['gate/lambda_p50']
        if 'gate/lambda_p90' in metrics:
            csv_row['gate_p90'] = metrics['gate/lambda_p90']
        if 'gate/attn_mean' in metrics:
            csv_row['gate_attn_mean'] = metrics['gate/attn_mean']
        if 'gate/mlp_mean' in metrics:
            csv_row['gate_mlp_mean'] = metrics['gate/mlp_mean']
        
        # 正则化损失统计
        if 'fedsdg/train_reg_loss_gate_mean' in metrics:
            csv_row['reg_loss_gate_mean'] = metrics['fedsdg/train_reg_loss_gate_mean']
        if 'fedsdg/train_reg_loss_private_mean' in metrics:
            csv_row['reg_loss_private_mean'] = metrics['fedsdg/train_reg_loss_private_mean']
    
    def _extract_client_stats(self, client_results) -> Dict[str, float]:
        """
        从客户端结果中提取统计量
        
        Args:
            client_results: 客户端结果（dict 或 list）
        
        Returns:
            包含 std, p10, p50, p90 的字典
        """
        if isinstance(client_results, dict):
            accs = [v[0] if isinstance(v, (tuple, list)) else v.get('acc', v) for v in client_results.values()]
        else:
            accs = [r[0] if isinstance(r, (tuple, list)) else r.get('acc', r) for r in client_results]
        
        if not accs:
            return {'std': 0.0, 'p10': 0.0, 'p50': 0.0, 'p90': 0.0}
        
        return {
            'std': float(np.std(accs)),
            'p10': float(np.percentile(accs, 10)),
            'p50': float(np.percentile(accs, 50)),
            'p90': float(np.percentile(accs, 90)),
        }
    
    def reset_cumulative_time(self) -> None:
        """重置累计时间（用于重新开始训练）"""
        self._cumulative_time = 0.0
    
    def set_cumulative_time(self, time_sec: float) -> None:
        """设置累计时间（用于恢复检查点）"""
        self._cumulative_time = time_sec
    
    @property
    def cumulative_time(self) -> float:
        """累计时间（秒）"""
        return self._cumulative_time
