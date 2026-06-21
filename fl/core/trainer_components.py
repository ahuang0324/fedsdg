# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练器组件工厂

Centralized factory for trainer component construction.

Usage:
    factory = TrainerComponentsFactory(args, hydra_cfg, hydra_run_dir)
    metrics_manager = factory.create_metrics_manager()
    private_state_manager = factory.create_private_state_manager()
    checkpoint_manager = factory.create_checkpoint_manager(experiment_name)
"""

from typing import Optional, Tuple, Any, TYPE_CHECKING
from omegaconf import DictConfig

if TYPE_CHECKING:
    from ..utils.metrics import MetricsManager
    from ..utils.csv_logger import CSVLogger
    from ..utils.csv_metrics_builder import CSVMetricsBuilder
    from ..utils.best_model_saver import BestModelSaver
    from ..utils.early_stopping import EarlyStopping
    from ..utils.checkpoint import CheckpointManager
    from ..utils.logger_factory import BaseLogger
    from .private_state_manager import PrivateStateManager


class TrainerComponentsFactory:
    """
    训练器组件工厂
    
    职责：
    1. 创建各种训练组件（logger, checkpoint_manager, csv_logger 等）
    2. 从配置中提取参数
    3. 提供组件创建的统一接口
    
    设计原则：
    - 延迟导入，避免循环依赖
    - 配置驱动，从 hydra_cfg 提取参数
    - 可选组件返回 None（如禁用的检查点管理器）
    
    Usage:
        factory = TrainerComponentsFactory(args, hydra_cfg, hydra_run_dir)
        logger = factory.create_logger(log_dir)
        checkpoint_manager = factory.create_checkpoint_manager(experiment_name)
    """
    
    def __init__(
        self, 
        args, 
        hydra_cfg: Optional[DictConfig] = None,
        hydra_run_dir: Optional[str] = None
    ):
        """
        初始化组件工厂
        
        Args:
            args: 命令行参数（扁平化后的）
            hydra_cfg: Hydra 配置对象
            hydra_run_dir: Hydra 运行目录
        """
        self.args = args
        self.hydra_cfg = hydra_cfg
        self.hydra_run_dir = hydra_run_dir
    
    def create_logger(self, log_dir: str) -> 'BaseLogger':
        """
        创建日志器
        
        Args:
            log_dir: 日志目录
            
        Returns:
            BaseLogger 实例（TensorBoard 或 WandB）
        """
        from ..utils.logger_factory import LoggerFactory
        
        logger_group = getattr(self.hydra_cfg, 'logger', None) if self.hydra_cfg else None
        logging_cfg = getattr(logger_group, 'logging', None) if logger_group else None
        backend = getattr(logging_cfg, 'backend', 'tensorboard') if logging_cfg else 'tensorboard'
        
        logger_dir = self.hydra_run_dir if backend == 'wandb' else log_dir
        return LoggerFactory.create_from_config(
            logging_cfg=logging_cfg,
            hydra_cfg=self.hydra_cfg,
            log_dir=logger_dir,
        )
    
    def create_checkpoint_manager(
        self, 
        experiment_name: str
    ) -> Optional['CheckpointManager']:
        """
        创建检查点管理器
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            CheckpointManager 实例（如果启用）或 None
        """
        from ..utils.checkpoint import create_checkpoint_manager
        from ..utils.paths import PROJECT_ROOT
        
        if not getattr(self.args, 'enable_checkpoint', False):
            return None
        
        return create_checkpoint_manager(self.args, PROJECT_ROOT, experiment_name)
    
    def create_csv_logger(
        self, 
        log_dir: str,
        comm_stats: dict
    ) -> Tuple['CSVLogger', 'CSVMetricsBuilder']:
        """
        创建 CSV 日志器和指标构建器
        
        Args:
            log_dir: 日志目录
            comm_stats: 通信统计字典
        
        Returns:
            (CSVLogger, CSVMetricsBuilder) 元组
        """
        from ..utils.csv_logger import CSVLogger
        from ..utils.csv_metrics_builder import CSVMetricsBuilder
        
        csv_logger = CSVLogger(log_dir=log_dir, filename="metrics.csv")
        csv_metrics_builder = CSVMetricsBuilder(
            comm_stats=comm_stats,
            algorithm=self.args.alg,
            num_users=self.args.num_users,
            frac=self.args.frac
        )
        
        return csv_logger, csv_metrics_builder
    
    def create_best_model_saver(self, log_dir: str) -> Optional['BestModelSaver']:
        """
        创建最佳模型保存器
        
        Args:
            log_dir: 保存目录
        
        Returns:
            BestModelSaver 实例（如果启用）或 None
        """
        from ..utils.best_model_saver import BestModelSaver
        
        if not getattr(self.args, 'save_best_model', True):
            return None
        
        return BestModelSaver(
            save_dir=log_dir,
            monitor_metric='val_acc_avg',
            mode='max'
        )
    
    def create_early_stopper(self) -> Optional['EarlyStopping']:
        """
        创建早停器
        
        Returns:
            EarlyStopping 实例（如果启用）或 None
        """
        from ..utils.early_stopping import EarlyStopping
        
        if not getattr(self.args, 'early_stopping_enabled', False):
            return None
        
        return EarlyStopping(
            patience=getattr(self.args, 'early_stopping_patience', 20),
            min_delta=getattr(self.args, 'early_stopping_min_delta', 0.001),
            mode='max'
        )
    
    def create_metrics_manager(self) -> 'MetricsManager':
        """
        创建指标管理器
        
        Returns:
            MetricsManager 实例
        """
        from ..utils.metrics import MetricsManager
        
        return MetricsManager.from_hydra_config(self.hydra_cfg, self.args.alg)
    
    def create_private_state_manager(self) -> 'PrivateStateManager':
        """
        创建私有状态管理器
        
        Returns:
            PrivateStateManager 实例
        """
        from .private_state_manager import PrivateStateManager
        
        return PrivateStateManager(algorithm=self.args.alg)
