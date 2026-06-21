# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通信量追踪器

Tracks per-round and cumulative communication volume.

Usage:
    tracker = CommTracker(comm_size_per_client_mb=0.5)
    stats = tracker.record_round(num_clients=10)
    print(f"本轮: {stats.round_comm_mb} MB, 累计: {stats.cumulative_comm_mb} MB")
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RoundCommStats:
    """单轮通信统计"""
    round_comm_mb: float           # 本轮通信量（MB）
    cumulative_comm_mb: float      # 累计通信量（MB）
    cumulative_comm_gb: float      # 累计通信量（GB）
    num_clients: int               # 参与客户端数


class CommTracker:
    """
    通信量追踪器（有状态）
    
    职责：
    1. 记录每轮通信量
    2. 维护累计通信量
    3. 提供统一的通信统计接口
    
    设计原则：
    - 封装通信量计算逻辑（上传 + 下载 = 2x）
    - 支持检查点保存/恢复
    - 与 CSVMetricsBuilder 解耦
    
    Usage:
        tracker = CommTracker(comm_size_per_client_mb=0.5)
        stats = tracker.record_round(num_clients=10)
        print(f"本轮: {stats.round_comm_mb} MB, 累计: {stats.cumulative_comm_mb} MB")
    """
    
    def __init__(self, comm_size_per_client_mb: float):
        """
        初始化通信追踪器
        
        Args:
            comm_size_per_client_mb: 每个客户端单向通信量（MB）
        """
        self._comm_size_per_client = comm_size_per_client_mb
        self._cumulative_mb: float = 0.0
        self._total_rounds: int = 0
    
    def record_round(self, num_clients: int) -> RoundCommStats:
        """
        记录一轮通信，返回统计结果
        
        Args:
            num_clients: 本轮参与的客户端数
            
        Returns:
            RoundCommStats: 本轮通信统计
        """
        # 上传 + 下载 = 2x
        round_comm = 2 * num_clients * self._comm_size_per_client
        self._cumulative_mb += round_comm
        self._total_rounds += 1
        
        return RoundCommStats(
            round_comm_mb=round_comm,
            cumulative_comm_mb=self._cumulative_mb,
            cumulative_comm_gb=self._cumulative_mb / 1024,
            num_clients=num_clients
        )
    
    def restore(self, cumulative_mb: float, total_rounds: int = 0) -> None:
        """
        恢复累计值（检查点恢复用）
        
        Args:
            cumulative_mb: 累计通信量
            total_rounds: 已完成轮次
        """
        self._cumulative_mb = cumulative_mb
        self._total_rounds = total_rounds
    
    @property
    def cumulative_mb(self) -> float:
        """累计通信量（MB）"""
        return self._cumulative_mb
    
    @property
    def cumulative_gb(self) -> float:
        """累计通信量（GB）"""
        return self._cumulative_mb / 1024
    
    @property
    def total_rounds(self) -> int:
        """已记录的轮次数"""
        return self._total_rounds
    
    @property
    def comm_size_per_client(self) -> float:
        """每个客户端单向通信量（MB）"""
        return self._comm_size_per_client
    
    def __repr__(self) -> str:
        return (
            f"CommTracker("
            f"comm_size_per_client={self._comm_size_per_client:.4f}MB, "
            f"cumulative={self._cumulative_mb:.2f}MB, "
            f"rounds={self._total_rounds})"
        )
