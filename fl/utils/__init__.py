# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for federated learning."""

from .paths import (
    PROJECT_ROOT, DATA_DIR, LOGS_DIR, OUTPUTS_DIR,
    HydraRunContext, ensure_dir, generate_experiment_name,
    get_log_dir, get_checkpoint_dir, get_model_path, 
    get_result_path, get_summary_path, get_visualization_path
)
from .checkpoint import CheckpointManager, create_checkpoint_manager
from .communication import (
    get_communication_stats, print_communication_profile,
    compute_round_communication, compute_total_communication
)
from .evaluation import test_inference, local_test_inference, evaluate_local_personalization
from .early_stopping import EarlyStopping
from .logger import exp_details
from .console_logger import (
    cprint, cprint_section, init_console_logger, close_console_logger,
    set_console_logger_mode, get_console_logger_mode, is_console_logger_initialized,
    log_config, log_round, log_fedsdg_aggregation, log_fedsdg_diagnostics, log_final,
    exp_details as exp_details_new,
)
from .visualization import visualize_all_gates
from .logger_factory import LoggerFactory, BaseLogger, TensorBoardLogger, WandBLogger, NoneLogger, create_logger
from .validator import validate_args, validate_dataset, ValidationError
from .reporting import generate_summary_report, ExperimentMetrics, generate_brief_summary
from .metrics import MetricsManager, MetricsConfig
from .csv_logger import CSVLogger
from .results_formatter import (
    ResultsFormatter, 
    build_experiment_info, build_hyperparameters, 
    build_comm_stats, build_final_test_results,
    calculate_last_n_rounds_average
)
from .best_model_saver import BestModelSaver
# Communication tracker and CSV metrics builder
from .comm_tracker import CommTracker, RoundCommStats
from .csv_metrics_builder import CSVMetricsBuilder

__all__ = [
    # 路径管理
    'PROJECT_ROOT', 'DATA_DIR', 'LOGS_DIR', 'OUTPUTS_DIR',
    'HydraRunContext', 'ensure_dir', 'generate_experiment_name',
    'get_log_dir', 'get_checkpoint_dir', 'get_model_path',
    'get_result_path', 'get_summary_path', 'get_visualization_path',
    # 检查点
    'CheckpointManager', 'create_checkpoint_manager',
    # 通信统计
    'get_communication_stats', 'print_communication_profile',
    'compute_round_communication', 'compute_total_communication',
    # 评估
    'test_inference', 'local_test_inference', 'evaluate_local_personalization',
    # 早停
    'EarlyStopping',
    # 日志
    'exp_details',
    # Console Logger
    'cprint', 'cprint_section', 'init_console_logger', 'close_console_logger',
    'set_console_logger_mode', 'get_console_logger_mode', 'is_console_logger_initialized',
    'log_config', 'log_round', 'log_fedsdg_aggregation', 'log_fedsdg_diagnostics', 'log_final',
    # 可视化
    'visualize_all_gates',
    # Logger Factory
    'LoggerFactory', 'BaseLogger', 'TensorBoardLogger', 'WandBLogger', 'NoneLogger', 'create_logger',
    # 验证
    'validate_args', 'validate_dataset', 'ValidationError',
    # 报告
    'generate_summary_report', 'ExperimentMetrics', 'generate_brief_summary',
    # 指标管理
    'MetricsManager', 'MetricsConfig',
    # CSV Logger
    'CSVLogger',
    # Results Formatter
    'ResultsFormatter', 'build_experiment_info', 'build_hyperparameters',
    'build_comm_stats', 'build_final_test_results', 'calculate_last_n_rounds_average',
    # Best Model Saver
    'BestModelSaver',
    # Communication tracker and CSV metrics builder
    'CommTracker', 'RoundCommStats', 'CSVMetricsBuilder',
]

