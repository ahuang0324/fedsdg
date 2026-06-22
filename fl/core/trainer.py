# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习训练器

封装联邦学习的完整训练流程，支持 FedAvg、FedLoRA、FedSDG 算法。

Usage:
    from fl.core import FederatedTrainer
    
    trainer = FederatedTrainer(args, hydra_cfg=cfg, hydra_run_dir=run_dir)
    trainer.run()
    
    # 或使用回调
    trainer = FederatedTrainer(args)
    trainer.on_round_end = my_callback
    trainer.run()
"""

import os
import sys
import copy
import time
import math
import pickle
import random
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig

from ..data import get_dataset
from ..models import get_model
from ..algorithms import server_aggregate
from ..clients import LocalUpdate
from ..utils import (
    exp_details, get_communication_stats, print_communication_profile,
    test_inference, evaluate_local_personalization,
    CheckpointManager, create_checkpoint_manager,
    visualize_all_gates,
    PROJECT_ROOT, generate_experiment_name, HydraRunContext,
    get_log_dir, get_result_path, get_summary_path, get_model_path,
    LoggerFactory, BaseLogger,
    EarlyStopping,  # Early stopping
    # Console Logger
    cprint, cprint_section, init_console_logger, close_console_logger,
    set_console_logger_mode, log_config, log_round, log_fedsdg_aggregation,
    log_fedsdg_diagnostics, log_final,
)
from ..utils.validator import validate_args, validate_dataset
from ..utils.reporting import generate_summary_report, ExperimentMetrics
from ..utils.metrics import MetricsManager
from ..utils.csv_logger import CSVLogger
from ..utils.csv_metrics_builder import CSVMetricsBuilder
from ..utils.results_formatter import ResultsFormatter, calculate_last_n_rounds_average
from ..utils.best_model_saver import BestModelSaver

# 模型池（用于复用模型实例）
from .model_pool import ModelPool
# 私有状态管理器
from .private_state_manager import PrivateStateManager
# 组件工厂
from .trainer_components import TrainerComponentsFactory
# 算法策略
from .algorithm_strategy import AlgorithmStrategyFactory, AlgorithmStrategy


@dataclass
class TrainingState:
    """
    训练状态容器
    
    封装训练过程中的所有状态变量，便于管理和序列化。
    """
    # 训练指标
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    eval_rounds: List[int] = field(default_factory=list)  # 记录实际评估的轮次
    # 实际进行评估的轮次索引 标记哪些轮次进行了评估，便于对齐指标

    # 通信统计 每轮累加 2 * 参与客户端数 * 每轮通信量（第522-523行）
    cumulative_comm_mb: float = 0.0
    
    # 效率指标
    best_efficiency_score: float = 0.0
    # 效率得分 = 测试准确率 / 累计通信量(MB)
    best_efficiency_epoch: int = 0
    # 达到最佳效率得分的轮次 与 best_efficiency_score 同时更新
    efficiency_score: float = 0.0
    #效率得分 = 测试准确率 / 累计通信量(MB)
    efficiency_score_per_gb: float = 0.0
    # test_acc / (cumulative_comm_mb / 1024) 每GB通信量的效率得分

    # FedSDG 私有状态
    local_private_states: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    
    # FedSDG 门控历史（训练动态追踪）
    # 每轮记录: {round, layer_means, layer_stds, overall_mean, overall_std, ...}
    gate_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def reset(self):
        """重置状态"""
        self.train_loss.clear()
        self.train_accuracy.clear()
        self.eval_rounds.clear()
        self.cumulative_comm_mb = 0.0
        self.best_efficiency_score = 0.0
        self.best_efficiency_epoch = 0
        self.efficiency_score = 0.0
        self.efficiency_score_per_gb = 0.0
        self.local_private_states = None
        self.gate_history.clear()


class FederatedTrainer:
    """
    联邦学习训练器
    
    封装完整的联邦学习训练流程，包括：
    - 初始化（模型、数据、日志）
    - 训练循环（客户端选择、本地训练、聚合）
    - 评估（全局测试、本地个性化测试）
    - 检查点保存
    - 报告生成
    
    Attributes:
        args: 配置对象
        state: 训练状态
        global_model: 全局模型
        device: 计算设备
        
    Callbacks:
        on_round_start: 每轮开始时调用
        on_round_end: 每轮结束时调用
        on_evaluation: 评估时调用
    """
    
    def __init__(
        self, 
        args, 
        hydra_cfg: Optional[DictConfig] = None,
        hydra_run_dir: Optional[str] = None
    ):
        """
        初始化训练器
        
        Args:
            args: 配置对象（ConfigAdapter 或 argparse Namespace）
            hydra_cfg: 原始 Hydra DictConfig（用于保存配置）
            hydra_run_dir: Hydra 运行目录
        """
        self.args = args
        self.hydra_cfg = hydra_cfg
        self.hydra_run_dir = hydra_run_dir
        
        # 训练状态
        self.state = TrainingState()
        self.start_time: Optional[float] = None
        
        # 组件（延迟初始化）
        self.device: str = 'cpu'
        self.global_model: Optional[nn.Module] = None
        self.train_dataset = None
        self.test_dataset = None
        self.user_groups: Dict = {}
        self.user_groups_test: Dict = {}
        self.logger: Optional[BaseLogger] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.best_model_saver: Optional[BestModelSaver] = None
        self.early_stopper: Optional[EarlyStopping] = None
        self.run_ctx: Optional[HydraRunContext] = None
        
        # 通信统计
        self.comm_stats: Dict[str, Any] = {}
        
        # 路径
        self.experiment_name: str = ""
        self.log_dir: str = ""
        
        # 模型池（延迟初始化）
        self.model_pool: Optional[ModelPool] = None
        
        # 指标管理器（延迟初始化）
        self.metrics_manager: Optional[MetricsManager] = None
        
        # 私有状态管理器（延迟初始化）
        self.private_state_manager: Optional[PrivateStateManager] = None
        
        # CSV 指标构建器（延迟初始化）
        self.csv_metrics_builder: Optional[CSVMetricsBuilder] = None
        
        # 算法策略（延迟初始化）
        self.algorithm_strategy: Optional[AlgorithmStrategy] = None
        
        # FedSDG 门控历史追踪
        self._gate_layer_names: List[str] = []
        
        # 回调函数（可扩展）
        self.on_round_start: Optional[Callable[[int], None]] = None
        self.on_round_end: Optional[Callable[[int, Dict], None]] = None
        self.on_evaluation: Optional[Callable[[int, float, float], None]] = None
    
    def run(self) -> Dict[str, Any]:
        """
        执行完整的训练流程
        
        Returns:
            包含最终结果的字典
        """
        self.start_time = time.time()
        
        try:
            self._initialize()
            self._train()
            results = self._finalize()
            return results
        except Exception as e:
            cprint(f"\n[错误] 训练过程中发生异常: {e}")
            if self.logger:
                self.logger.close()
            raise
    
    # =========================================================================
    # 初始化阶段
    # =========================================================================
    
    def _initialize(self) -> None:
        """初始化所有组件"""
        self._validate_config()
        self._set_random_seed()
        self._setup_paths()
        self._setup_device()
        self._load_data()
        self._build_model()
        self._setup_model_pool()  # 模型池初始化
        self._warmup_gpu_memory()  # 预占显存，防止被其他进程抢占
        self._setup_communication_stats()
        
        # FedSDG / FedALT: 提取门控层名称（用于训练动态追踪）
        if self.args.alg in ('fedsdg', 'fedalt'):
            self._extract_gate_layer_names()
        
        # FedTP: Phase 1 初始化（冻结私有分支）
        if self.args.alg == 'fedtp':
            self._fedtp_init_phase1()
        
        # 使用工厂创建组件
        self._setup_components_via_factory()
        
        exp_details(self.args)
        
        # 记录配置信息到 console.log
        log_config(
            args=self.args,
            comm_stats=self.comm_stats,
            experiment_name=self.experiment_name,
            hydra_run_dir=self.hydra_run_dir,
            device=self.device
        )
    
    def _validate_config(self) -> None:
        """验证配置"""
        validate_args(self.args)
    
    def _set_random_seed(self) -> None:
        """设置随机种子"""
        seed = getattr(self.args, 'seed', 0)
        if seed > 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            cprint(f"\n[随机种子] 设置为 {seed}")
    
    def _setup_paths(self) -> None:
        """设置路径和日志器"""
        self.experiment_name = generate_experiment_name(self.args)
        
        if self.hydra_run_dir:
            self.run_ctx = HydraRunContext.initialize(
                hydra_run_dir=self.hydra_run_dir,
                algorithm=self.args.alg,
                experiment_name=self.experiment_name
            )
            cprint(f"\n[Hydra] 运行目录: {self.hydra_run_dir}")
            cprint(f"[Hydra] 实验名称: {self.experiment_name}")
        
        # 日志目录（绝对路径）
        self.log_dir = get_log_dir(self.args.alg, self.experiment_name)
        
        # 使用 LoggerFactory 创建日志器
        # 配置路径: hydra_cfg.logger.logging (由 conf/logger/*.yaml 定义)
        logger_group = getattr(self.hydra_cfg, 'logger', None) if self.hydra_cfg else None
        logging_cfg = getattr(logger_group, 'logging', None) if logger_group else None
        backend = getattr(logging_cfg, 'backend', 'tensorboard') if logging_cfg else 'tensorboard'
        
        # WandB 使用 hydra_run_dir，TensorBoard 使用 log_dir
        logger_dir = self.hydra_run_dir if backend == 'wandb' else self.log_dir
        self.logger = LoggerFactory.create_from_config(
            logging_cfg=logging_cfg,
            hydra_cfg=self.hydra_cfg,
            log_dir=logger_dir,
        )
        
        # TensorBoard 软链接（仅 TensorBoard 模式）
        if self.run_ctx and backend == 'tensorboard':
            symlink = self.run_ctx.create_tensorboard_symlink()
            if symlink:
                cprint(f"[输出] TensorBoard 软链接: {symlink}")
        
        # 初始化控制台日志（console.log）
        if self.hydra_run_dir:
            console_log_path = os.path.join(self.hydra_run_dir, 'console.log')
            init_console_logger(console_log_path)
            
            # 从配置读取日志模式
            training_cfg = getattr(self.hydra_cfg, 'training', None) if self.hydra_cfg else None
            console_log_cfg = getattr(training_cfg, 'console_log', None) if training_cfg else None
            if console_log_cfg:
                set_console_logger_mode(
                    to_console=getattr(console_log_cfg, 'to_console', True),
                    to_file=getattr(console_log_cfg, 'to_file', True)
                )
            
            cprint(f"[输出] 终端日志: {console_log_path}")
    
    def _setup_device(self) -> None:
        """设置计算设备"""
        gpu = getattr(self.args, 'gpu', None)
        use_cuda = (
            gpu is not None and 
            int(gpu) >= 0 and 
            torch.cuda.is_available()
        )
        if use_cuda:
            torch.cuda.set_device(int(gpu))
            cprint(f"[设备] 使用 GPU {gpu}: {torch.cuda.get_device_name()}")
        else:
            cprint("[设备] 使用 CPU")
        self.device = 'cuda' if use_cuda else 'cpu'
    
    def _load_data(self) -> None:
        """加载数据集（Train/Val/Test三划分）"""
        cprint("\n[数据] 加载数据集...")
        (self.train_dataset, self.val_dataset, self.test_dataset,
         self.user_groups_train, self.user_groups_val, self.user_groups_test) = get_dataset(self.args)
        
        # 兼容接口别名
        self.user_groups = self.user_groups_train
        
        validate_dataset(
            self.train_dataset, self.val_dataset, self.test_dataset,
            self.user_groups_train, self.user_groups_val, self.user_groups_test
        )
    
    def _build_model(self) -> None:
        """构建模型"""
        cprint("\n[模型] 构建模型...")
        self.global_model = get_model(self.args, self.train_dataset, self.device)
        cprint(str(self.global_model))
    
    def _setup_model_pool(self) -> None:
        """
        初始化模型池
        
        使用模型池减少每客户端 deepcopy 开销
        优化: O(m * model_size) 内存 -> O(pool_size * model_size)
        """
        pool_size = getattr(self.args, 'model_pool_size', 1)  # 默认串行，pool_size=1
        self.model_pool = ModelPool(
            global_model=self.global_model,
            device=self.device,
            pool_size=pool_size
        )
    
    def _warmup_gpu_memory(self) -> None:
        """
        预占 GPU 显存，防止被其他进程抢占
        
        The estimate is obtained by running one synthetic forward/backward pass
        with the pooled model and reserving memory near the observed peak.
        
        这比公式估算准确得多，因为激活值（ViT 的中间特征图）
        是显存的大头，无法通过参数量推算。
        """
        if self.device != 'cuda':
            return
        
        try:
            torch.cuda.reset_peak_memory_stats()
            baseline_mem = torch.cuda.memory_allocated()
            
            # Measure the peak memory of a representative training step.
            model = self.model_pool.get_model(0)
            model.train()
            
            # Use the same tensor shape as the configured training batch.
            bs = self.args.local_bs
            img_size = getattr(self.args, 'image_size', 224)
            in_channels = 3
            sample_input = torch.randn(bs, in_channels, img_size, img_size, device=self.device)
            
            num_classes = getattr(self.args, 'num_classes', 10)
            sample_labels = torch.randint(0, num_classes, (bs,), device=self.device)
            
            criterion = nn.CrossEntropyLoss()
            logits = model(sample_input)
            loss = criterion(logits, sample_labels)
            loss.backward()
            
            # 记录训练峰值
            train_peak = torch.cuda.max_memory_allocated()
            
            # Keep the CUDA allocator cache warm after removing temporary tensors.
            del sample_input, sample_labels, logits, loss
            for param in model.parameters():
                param.grad = None
            
            # 评估时会 deepcopy 一个模型，额外加一份模型参数量
            model_mem = sum(p.numel() * p.element_size() for p in self.global_model.parameters())
            estimated_peak = train_peak + model_mem
            
            # 限制预占量：不超过 GPU 可用显存的 80%
            free_mem = torch.cuda.mem_get_info()[0]
            warmup_size = min(estimated_peak, int(free_mem * 0.8))
            
            # 当前已占用的显存
            current_reserved = torch.cuda.memory_reserved()
            extra_needed = warmup_size - current_reserved
            
            if extra_needed > 0:
                # 分配额外张量，撑大 PyTorch 缓存池
                warmup_tensor = torch.empty(extra_needed // 4, dtype=torch.float32, device='cuda')
                del warmup_tensor  # 释放张量，但缓存池仍持有显存
                # Keep reserved memory in PyTorch's CUDA caching allocator.
            
            reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
            cprint(f"\n[显存预占] 实测训练峰值: {train_peak / 1024 / 1024:.0f}MB")
            cprint(f"[显存预占] 预估总峰值(含评估): {estimated_peak / 1024 / 1024:.0f}MB")
            cprint(f"[显存预占] 实际 reserved: {reserved_mb:.0f}MB")
        except Exception as e:
            cprint(f"\n[显存预占] 预热失败（不影响训练）: {e}")
    
    def _setup_communication_stats(self) -> None:
        """设置通信统计"""
        self.comm_stats = get_communication_stats(self.global_model, self.args.alg)
        print_communication_profile(self.comm_stats, self.args)
        
        # 记录模型信息（step=0）
        self.logger.log_metrics({
            'info/total_params': self.comm_stats['total_params'],
            'info/trainable_params': self.comm_stats['trainable_params'],
            'info/comm_params_per_round': self.comm_stats['comm_params'],
            'info/comm_size_per_round_MB': self.comm_stats['comm_size_mb'],
        }, step=0)
        
        # 记录超参数
        self._log_hyperparams()
    
    def _log_hyperparams(self) -> None:
        """记录超参数"""
        hyperparams = {
            'algorithm': self.args.alg,
            'model': self.args.model,
            'dataset': self.args.dataset,
            'epochs': self.args.epochs,
            'local_ep': self.args.local_ep,
            'local_bs': self.args.local_bs,
            'lr': self.args.lr,
            'num_users': self.args.num_users,
            'frac': self.args.frac,
            'dirichlet_alpha': self.args.dirichlet_alpha,
        }
        if self.args.alg in ('fedlora', 'fedsdg', 'fedalt', 'local_only'):
            hyperparams.update({
                'lora_r': self.args.lora_r,
                'lora_alpha': self.args.lora_alpha,
            })
        self.logger.log_hyperparams(hyperparams)
    
    def _setup_components_via_factory(self) -> None:
        """
        使用工厂创建各种组件
        
        统一组件初始化
        """
        factory = TrainerComponentsFactory(
            args=self.args,
            hydra_cfg=self.hydra_cfg,
            hydra_run_dir=self.hydra_run_dir
        )
        
        log_dir = self.hydra_run_dir if self.hydra_run_dir else self.log_dir
        
        # 指标管理器
        self.metrics_manager = factory.create_metrics_manager()
        cprint(f"\n[指标] 评估频率: 每 {self.metrics_manager.config.eval_freq} 轮")
        
        # 私有状态管理器
        self.private_state_manager = factory.create_private_state_manager()
        if self.private_state_manager.enabled:
            cprint(f"\n[{self.args.alg.upper()}] 客户端私有状态管理已初始化")
        
        # 检查点管理器
        self.checkpoint_manager = factory.create_checkpoint_manager(self.experiment_name)
        
        # CSV 日志器和指标构建器
        self.csv_logger, self.csv_metrics_builder = factory.create_csv_logger(
            log_dir=log_dir,
            comm_stats=self.comm_stats
        )
        cprint(f"\n[CSV Logger] 初始化完成: {self.csv_logger.csv_path}")
        
        # 最佳模型保存器
        self.best_model_saver = factory.create_best_model_saver(log_dir)
        if self.best_model_saver is None:
            cprint("[BestModelSaver] 已禁用（save_best_model=False）")
        
        # 早停器
        self.early_stopper = factory.create_early_stopper()
        if self.early_stopper:
            cprint(f"[EarlyStopping] 已启用 (patience={self.early_stopper.patience}, min_delta={self.early_stopper.min_delta})")
        
        # 算法策略
        self.algorithm_strategy = AlgorithmStrategyFactory.create(self.args.alg)
        cprint(f"\n[算法策略] {self.algorithm_strategy.name} (聚合: {self.algorithm_strategy.requires_aggregation})")
    
    # =========================================================================
    # 训练阶段
    # =========================================================================
    
    def _train(self) -> None:
        """执行训练循环"""
        cprint_section("[训练开始]")
        cprint(f"总轮次: {self.args.epochs}")
        if self.early_stopper:
            cprint(f"早停: patience={self.early_stopper.patience}, min_delta={self.early_stopper.min_delta}")
        cprint("")
        
        for epoch in tqdm(range(self.args.epochs), desc="全局训练轮次"):
            should_stop = self._train_one_round(epoch)
            if should_stop:
                cprint(f"\n[训练] 早停触发，训练提前结束于 epoch {epoch}")
                break
    
    def _train_one_round(self, epoch: int) -> bool:
        """
        执行一轮训练
        
        统一架构设计:
        - 阶段1: 客户端本地训练，收集权重列表
        - 阶段2: 服务端聚合（使用统一的 server_aggregate() 接口）
          - 聚合方式由配置 server_agg_method 决定：'alignment' 或 'fedavg'
        """
        start_epoch_time = time.time()
        
        # 回调
        if self.on_round_start:
            self.on_round_start(epoch)
        
        # FedTP: Phase 切换
        if self.args.alg == 'fedtp':
            phase1_epochs = getattr(self.args, 'phase1_epochs', 50)
            strategy = self.algorithm_strategy
            old_phase = strategy.current_phase if hasattr(strategy, 'current_phase') else 1
            strategy.set_phase(epoch, phase1_epochs)
            new_phase = strategy.current_phase
            
            # Phase 转换点：从 Phase 1 → Phase 2
            if old_phase == 1 and new_phase == 2:
                cprint(f"\n{'='*70}")
                cprint(f"[FedTP] Phase 1 \u2192 Phase 2 \u5207\u6362 (epoch={epoch})")
                cprint(f"[FedTP] \u51bb\u7ed3\u5168\u5c40 LoRA\uff0c\u5f00\u59cb\u79c1\u6709\u5206\u652f\u672c\u5730\u5fae\u8c03")
                cprint(f"{'='*70}")
                self._fedtp_switch_to_phase2()
        
        # Round progress is printed to the terminal only.
        if self.args.alg == 'fedtp':
            phase_str = f"Phase {self.algorithm_strategy.current_phase}"
            print(f'\n | \u5168\u5c40\u8bad\u7ec3\u8f6e\u6b21 : {epoch+1} ({phase_str}) |\n')
        else:
            print(f'\n | \u5168\u5c40\u8bad\u7ec3\u8f6e\u6b21 : {epoch+1} |\n')
        
        # 选择客户端
        m = max(int(self.args.frac * self.args.num_users), 1)
        selected_clients = np.random.choice(range(self.args.num_users), m, replace=False)
        
        # 缓存全局状态（避免多次调用 state_dict()）
        global_state_cache = self.global_model.state_dict()
        
        # 统一的训练和聚合流程
        local_losses, train_metrics_list, new_global_state, aggregation_info = self._train_round_unified(
            epoch, selected_clients, global_state_cache
        )
        
        # 加载新的全局状态
        self.global_model.load_state_dict(new_global_state)
        
        # 更新状态
        loss_avg = sum(local_losses) / len(local_losses)
        self.state.train_loss.append(loss_avg)
        
        # 从 train_metrics_list 收集训练准确率
        if train_metrics_list:
            train_accs = [m['train_acc'] for m in train_metrics_list if 'train_acc' in m]
            if train_accs:
                self.state.train_accuracy.append(float(np.mean(train_accs)))
            else:
                self.state.train_accuracy.append(0.0)
        else:
            self.state.train_accuracy.append(0.0)
        
        # 累计通信量（使用本轮实际参与客户端数）
        # FedTP Phase 2: 零通信（纯本地训练，不聚合）
        if self.args.alg == 'fedtp' and self.algorithm_strategy.current_phase == 2:
            round_comm_mb = 0.0
        else:
            round_comm_mb = 2 * len(selected_clients) * self.comm_stats['comm_size_mb']
        self.state.cumulative_comm_mb += round_comm_mb
        
        # 评估
        round_metrics = self._evaluate_round(epoch, selected_clients)
        
        # 保存最佳模型（如果验证准确率提升）
        if self.best_model_saver and round_metrics.get('local_val_acc') is not None:
            self.best_model_saver.save_if_best(
                epoch=epoch,
                model=self.global_model,
                current_score=round_metrics['local_val_acc'],
                local_private_states=self.private_state_manager.states
            )
        
        # 早停判断（仅在评估轮次，且验证准确率可用时）
        should_stop = False
        if self.early_stopper and round_metrics.get('local_val_acc') is not None:
            if self.early_stopper.should_stop(round_metrics['local_val_acc'], epoch):
                should_stop = True
        
        # 收集本轮所有指标（使用 MetricsManager）
        metrics = self._collect_round_metrics(
            epoch, local_losses, loss_avg, round_metrics, 
            train_metrics_list, aggregation_info
        )
        round_time = time.time() - start_epoch_time
        metrics['time/round'] = round_time
        self.logger.log_metrics(metrics, step=epoch)
        
        # 记录到 CSV（用于画图）
        csv_metrics = self._prepare_csv_metrics(epoch, loss_avg, round_metrics, round_time, metrics)
        self.csv_logger.log(csv_metrics)
        
        # 记录到 console.log（每轮汇总）
        log_round(
            epoch=epoch,
            total_epochs=self.args.epochs,
            loss_avg=loss_avg,
            round_time=round_time,
            cumulative_comm_mb=self.state.cumulative_comm_mb,
            round_metrics=round_metrics
        )
        
        self._save_checkpoint(epoch, local_losses, selected_clients, aggregation_info, round_metrics)
        
        # 打印训练统计（根据评估频率）
        eval_freq = getattr(self.args, 'eval_freq', 5)
        if (epoch + 1) % eval_freq == 0:
            cprint(f'\n训练统计（{epoch+1} 轮后）:')
            cprint(f'  平均训练损失: {np.mean(self.state.train_loss):.4f}')
            if round_metrics.get('local_test_acc') is not None:
                cprint(f'  local/test_acc_avg: {100*round_metrics["local_test_acc"]:.2f}%')
        
        # FedSDG / FedALT: 记录门控历史（训练动态追踪）
        if self.args.alg in ('fedsdg', 'fedalt') and train_metrics_list:
            self._record_gate_history(epoch, train_metrics_list)
        
        # 回调
        if self.on_round_end:
            self.on_round_end(epoch, round_metrics)
        
        # 返回是否应该停止训练（早停判断）
        return should_stop
    
    
    def _train_round_unified(
        self,
        epoch: int,
        selected_clients: np.ndarray,
        global_state_cache: Dict[str, torch.Tensor]
    ) -> Tuple[List[float], List[Dict], Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        统一的训练轮次
        
        使用算法策略模式简化训练分支逻辑
        
        阶段1: 客户端本地训练，收集权重列表
        阶段2: 服务端聚合（使用统一的 server_aggregate() 接口）
        
        Args:
            epoch: 当前轮次
            selected_clients: 选中的客户端索引
            global_state_cache: 缓存的全局状态
            
        Returns:
            (local_losses, train_metrics_list, new_global_state, aggregation_info)
        """
        local_losses = []
        local_weights = []
        train_metrics_list = []
        
        # 使用算法策略
        strategy = self.algorithm_strategy
        model_flags = strategy.get_model_pool_flags()
        is_fedsdg = model_flags['is_fedsdg']
        
        is_actual_fedsdg = (self.args.alg == 'fedsdg')
        
        # ========================================
        # 阶段1: 客户端本地训练
        # ========================================
        for client_idx, idx in enumerate(selected_clients):
            # 使用策略获取私有状态
            private_state = strategy.get_private_state_for_model(
                self.private_state_manager, idx
            )
            
            # 使用模型池准备本地模型
            local_model = self.model_pool.prepare_model(
                worker_id=0,
                global_state=global_state_cache,
                private_state=private_state,
                **model_flags
            )
            
            # Reset Dynamic Alignment statistics once per round.
            if client_idx == 0:
                from ..models.lora import reset_da_diagnostics
                reset_da_diagnostics(local_model)
            
            # 本地训练
            local_trainer = LocalUpdate(
                args=self.args,
                dataset=self.train_dataset,
                idxs=self.user_groups[idx],
                logger=self.logger
            )
            
            # 使用策略获取训练时的个性化状态（如 Ditto）
            personal_state = strategy.get_personal_state_for_training(
                self.private_state_manager, idx
            )
            
            # 执行本地训练
            if personal_state is not None:
                # Ditto: 需要传入个性化状态
                output = local_trainer.update_weights(
                    model=local_model, global_round=epoch, personal_state=personal_state
                )
            else:
                output = local_trainer.update_weights(
                    model=local_model, global_round=epoch
                )
            
            # 使用策略处理训练输出
            public_w, loss, train_metrics = strategy.process_training_output(
                output, idx, self.private_state_manager
            )
            
            # 收集公共权重用于聚合（如果有）
            if public_w is not None:
                local_weights.append(public_w)
            
            local_losses.append(loss)
            train_metrics_list.append(train_metrics)
            
            # Optional FedSDG diagnostics.
            if is_actual_fedsdg and epoch % 5 == 0 and client_idx == 0 and getattr(self.args, 'verbose', 0) >= 2:
                self._log_fedsdg_diagnostics(epoch, idx, local_model)
        
        # ========================================
        # 阶段2: 服务端聚合
        # ========================================
        # 使用策略判断是否需要聚合
        if not strategy.requires_aggregation:
            # 不进行聚合，直接返回原始状态
            new_global_state = global_state_cache
            aggregation_info = {
                'algorithm': strategy.name,
                'agg_method': 'none',
                'num_clients': len(selected_clients),
                'aggregated_keys': [],
                'excluded_keys': list(global_state_cache.keys()),
                'weights': [],
            }
            server_agg_method = 'none'
        else:
            # 从配置中获取聚合参数（带默认值）
            server_agg_method = getattr(self.args, 'server_agg_method', 'fedavg')
            alignment_strategy = getattr(self.args, 'alignment_strategy', 'loo_mean')
            weight_transform = getattr(self.args, 'weight_transform', 'relu_normalize')
            softmax_temperature = getattr(self.args, 'softmax_temperature', 1.0)
            lambda_smooth = getattr(self.args, 'lambda_smooth', 0.0)
            
            # 获取 head_mode 配置（仅 FedSDG 使用）
            head_mode = getattr(self.args, 'head_mode', 'global')
            
            # 使用统一的服务端聚合接口
            # LoRA-FAIR 残差修正参数
            residual_mu = getattr(self.args, 'residual_mu', 0.1)
            
            new_global_state, aggregation_info = server_aggregate(
                client_weights=local_weights,
                global_state_dict=global_state_cache,
                algorithm=self.args.alg,
                agg_method=server_agg_method,
                alignment_strategy=alignment_strategy,
                weight_transform=weight_transform,
                softmax_temperature=softmax_temperature,
                lambda_smooth=lambda_smooth,
                head_mode=head_mode,
                residual_mu=residual_mu,
            )
        
        # 记录聚合信息到 console.log
        # 仅当使用 alignment 聚合时输出详细日志
        if server_agg_method == 'alignment':
            log_fedsdg_aggregation(epoch, aggregation_info)
        
        # 收集 drift 和 update 指标
        round_extra_metrics = {}
        if self.metrics_manager.config.drift and strategy.requires_aggregation:
            # Local-Only 不聚合，不需要收集 drift 指标
            # 客户端漂移指标
            drift_metrics = self.metrics_manager.drift.collect_client_drift(
                client_weights=local_weights,
                global_state=global_state_cache,
                comm_keys=self.comm_stats.get('comm_keys', [])
            )
            round_extra_metrics.update(drift_metrics)
            
            # 全局更新指标
            update_metrics = self.metrics_manager.drift.collect_global_update(
                new_state=new_global_state,
                old_state=global_state_cache,
                comm_keys=self.comm_stats.get('comm_keys', [])
            )
            round_extra_metrics.update(update_metrics)
        
        # 将 drift 指标添加到 aggregation_info 中传递
        aggregation_info['drift_metrics'] = round_extra_metrics
        
        return local_losses, train_metrics_list, new_global_state, aggregation_info
    
    def _collect_round_metrics(
        self, 
        epoch: int, 
        local_losses: List, 
        loss_avg: float,
        round_metrics: Dict[str, Any],
        train_metrics_list: Optional[List[Dict]] = None,
        aggregation_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        收集本轮所有指标（用于批量记录）
        
        使用 MetricsManager 收集指标
        """
        metrics = {
            # 训练指标
            'global/train_loss_avg': loss_avg,
            'lr': self.args.lr,
            # 通信指标（不在 TensorBoard 展示，但保留记录）
            'info/cumulative_comm_volume_MB': self.state.cumulative_comm_mb,
            'info/cumulative_comm_volume_GB': self.state.cumulative_comm_mb / 1024,
        }
        
        # 使用 MetricsManager 收集客户端训练指标
        if train_metrics_list:
            client_metrics = self.metrics_manager.client.collect(train_metrics_list)
            metrics.update(client_metrics)
            
            # FedSDG 特有指标
            if self.metrics_manager.fedsdg is not None:
                fedsdg_metrics = self.metrics_manager.fedsdg.collect(train_metrics_list)
                metrics.update(fedsdg_metrics)
                
                # 按层类型（Attention vs MLP/FFN）计算门控均值
                if self._gate_layer_names:
                    all_lambda = [m['lambda_values'] for m in train_metrics_list
                                  if 'lambda_values' in m and m['lambda_values']]
                    if all_lambda:
                        arr = np.array(all_lambda)  # (clients, layers)
                        layer_means = arr.mean(axis=0)  # per-layer mean across clients
                        attn_idx = [i for i, n in enumerate(self._gate_layer_names)
                                    if 'attn' in n or 'out_proj' in n]
                        mlp_idx = [i for i, n in enumerate(self._gate_layer_names)
                                   if 'mlp' in n or 'linear2' in n or 'fc2' in n]
                        if attn_idx:
                            metrics['gate/attn_mean'] = float(np.mean([layer_means[i] for i in attn_idx]))
                        if mlp_idx:
                            metrics['gate/mlp_mean'] = float(np.mean([layer_means[i] for i in mlp_idx]))
        
        # 添加 drift 指标（从 aggregation_info 中获取）
        if aggregation_info and 'drift_metrics' in aggregation_info:
            metrics.update(aggregation_info['drift_metrics'])
        
        # 评估指标（如果本轮有评估）
        # 注意：FedRep 和 Local-Only 没有全局模型，所以 val_acc/test_acc 为 None
        # 但 local_val_acc/local_test_acc 仍然有值，需要分别判断
        
        # Val 集指标（用于过程监控和早停）
        # Global Val 指标
        if round_metrics.get('val_acc') is not None:
            metrics.update({
                'global/val_acc': round_metrics['val_acc'],
                'global/val_loss': round_metrics['val_loss'],
            })
        
        # Local Val 指标（独立判断，支持 FedRep/Local-Only）
        if round_metrics.get('local_val_acc') is not None:
            metrics.update({
                'local/val_acc_avg': round_metrics['local_val_acc'],
                'local/val_loss_avg': round_metrics['local_val_loss'],
            })
            
            # 使用 MetricsManager 收集 Val 集客户端统计量
            if round_metrics.get('val_client_results'):
                local_val_metrics = self.metrics_manager.local.collect(round_metrics['val_client_results'])
                # 重命名以区分 val 和 test
                val_metrics_renamed = {}
                for k, v in local_val_metrics.items():
                    if k.startswith('local/test_'):
                        new_k = k.replace('local/test_', 'local/val_')
                        val_metrics_renamed[new_k] = v
                    else:
                        val_metrics_renamed[k] = v
                metrics.update(val_metrics_renamed)
        
        # Test 集指标（用于论文报告）
        # Global Test 指标
        if round_metrics.get('test_acc') is not None:
            metrics.update({
                'global/test_acc': round_metrics['test_acc'],
                'global/test_loss': round_metrics['test_loss'],
            })
        
        # Local Test 指标（独立判断，支持 FedRep/Local-Only）
        if round_metrics.get('local_test_acc') is not None:
            metrics.update({
                'local/test_acc_avg': round_metrics['local_test_acc'],
                'local/test_loss_avg': round_metrics['local_test_loss'],
                'Efficiency/accuracy_per_MB': self.state.efficiency_score,
                'Efficiency/accuracy_per_GB': self.state.efficiency_score_per_gb,
            })
            
            # 只有当两者都存在时才计算差异
            if round_metrics.get('test_acc') is not None:
                metrics['personalization/gap_local_vs_global'] = round_metrics['local_test_acc'] - round_metrics['test_acc']
            
            # 使用 MetricsManager 收集 Test 集客户端统计量
            if round_metrics.get('test_client_results'):
                local_test_metrics = self.metrics_manager.local.collect(round_metrics['test_client_results'])
                metrics.update(local_test_metrics)
        
        return metrics
    
    def _prepare_csv_metrics(
        self, 
        epoch: int, 
        loss_avg: float, 
        round_metrics: Dict[str, Any],
        round_time: float,
        metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        准备 CSV 记录的指标
        
        使用 CSVMetricsBuilder 构建 CSV 指标
        """
        train_acc = self.state.train_accuracy[-1] if self.state.train_accuracy else None
        return self.csv_metrics_builder.build(
            epoch=epoch,
            loss_avg=loss_avg,
            train_acc=train_acc,
            round_metrics=round_metrics,
            round_time=round_time,
            cumulative_comm_mb=self.state.cumulative_comm_mb,
            metrics=metrics
        )
    
    def _evaluate_round(self, epoch: int, selected_clients: np.ndarray) -> Dict[str, Any]:
        """
        轮次评估（返回指标，不直接记录日志）
        
        修改: 同时评估 Val 集和 Test 集
        - Val 集: 用于过程监控和早停
        - Test 集: 用于计算最后N轮平均值（论文报告）
        """
        round_metrics = {
            # Val 集评估结果
            'val_acc': None,
            'val_loss': None,
            'local_val_acc': None,
            'local_val_loss': None,
            'val_client_results': None,
            
            # Test 集评估结果
            'test_acc': None,
            'test_loss': None,
            'local_test_acc': None,
            'local_test_loss': None,
            'test_client_results': None,
        }
        
        # 使用 MetricsManager 判断是否评估
        if not self.metrics_manager.should_evaluate(epoch, self.args.epochs):
            return round_metrics
        
        # 记录评估轮次
        self.state.eval_rounds.append(epoch)
        
        self.global_model.eval()
        test_frac = getattr(self.args, 'test_frac', 0.3)
        num_test_clients = max(1, int(test_frac * self.args.num_users))
        test_client_idxs = np.random.choice(range(self.args.num_users), num_test_clients, replace=False)
        
        # 判断是否跳过全局模型评估
        # FedRep 和 Local-Only 没有完整的全局模型（FedRep 没有 Head，Local-Only 没有 LoRA）
        skip_global_eval = self.args.alg in ('local_only', 'fedrep')
        
        # ========== 1. 评估 Val 集 ==========
        # 全局模型在 Val 集上评估
        if skip_global_eval:
            val_acc, val_loss = None, None
        else:
            val_acc, val_loss = test_inference(self.args, self.global_model, self.val_dataset)
        
        # 个性化模型在 Val 集上评估
        # 使用 private_state_manager.states
        local_val_acc, local_val_loss, val_client_results = evaluate_local_personalization(
            args=self.args,
            global_model=self.global_model,
            test_dataset=self.val_dataset,
            user_groups_test=self.user_groups_val,
            local_private_states=self.private_state_manager.states,
            sample_clients=test_client_idxs
        )
        
        round_metrics.update({
            'val_acc': val_acc,
            'val_loss': val_loss,
            'local_val_acc': local_val_acc,
            'local_val_loss': local_val_loss,
            'val_client_results': val_client_results,
        })
        
        # ========== 2. 评估 Test 集 ==========
        # 全局模型在 Test 集上评估
        if skip_global_eval:
            test_acc, test_loss = None, None
        else:
            test_acc, test_loss = test_inference(self.args, self.global_model, self.test_dataset)
        
        # 个性化模型在 Test 集上评估
        # 使用 private_state_manager.states
        local_test_acc, local_test_loss, test_client_results = evaluate_local_personalization(
            args=self.args,
            global_model=self.global_model,
            test_dataset=self.test_dataset,
            user_groups_test=self.user_groups_test,
            local_private_states=self.private_state_manager.states,
            sample_clients=test_client_idxs
        )
        
        round_metrics.update({
            'test_acc': test_acc,
            'test_loss': test_loss,
            'local_test_acc': local_test_acc,
            'local_test_loss': local_test_loss,
            'test_client_results': test_client_results,
        })
        
        # 更新效率指标（基于 test_acc，如果是 FedRep/Local-Only 则使用 local_test_acc）
        effective_test_acc = test_acc if test_acc is not None else local_test_acc
        if self.state.cumulative_comm_mb > 0 and effective_test_acc is not None:
            self.state.efficiency_score = effective_test_acc / self.state.cumulative_comm_mb
            self.state.efficiency_score_per_gb = effective_test_acc / (self.state.cumulative_comm_mb / 1024)
            
            if self.state.efficiency_score > self.state.best_efficiency_score:
                self.state.best_efficiency_score = self.state.efficiency_score
                self.state.best_efficiency_epoch = epoch
        
        # 回调（保持向后兼容，使用 test 指标）
        if self.on_evaluation:
            self.on_evaluation(epoch, test_acc, local_test_acc)
        
        return round_metrics
    
    def _save_checkpoint(
        self, 
        epoch: int, 
        local_losses: List,
        selected_clients: np.ndarray,
        aggregation_info: Optional[Dict],
        round_metrics: Dict[str, Any]
    ) -> None:
        """
        保存检查点
        
        The checkpoint stores the global state, metrics, selected clients,
        and optional client-private states.
        """
        if self.checkpoint_manager is None:
            return
        
        self.checkpoint_manager.save_round_checkpoint(
            round_idx=epoch,
            global_model=self.global_model,
            local_weights=None,
            local_losses=local_losses,
            selected_clients=list(selected_clients),
            aggregation_info=aggregation_info,
            local_private_states=self.private_state_manager.states,
            train_loss=self.state.train_loss[-1],
            train_acc=self.state.train_accuracy[-1] if self.state.train_accuracy else None,
            test_acc=round_metrics.get('test_acc'),
            test_loss=round_metrics.get('test_loss'),
            local_test_acc=round_metrics.get('local_test_acc'),
            local_test_loss=round_metrics.get('local_test_loss'),
            comm_volume_mb=self.state.cumulative_comm_mb,
        )
    
    def _fedtp_init_phase1(self) -> None:
        """FedTP: Phase 1 初始化 — 冻结私有分支，门控设为 0
        
        注意：必须同时修改 global_model 和 model_pool 的 requires_grad，
        因为 model_pool 是在此方法之前通过 deepcopy 创建的，
        而 load_state_dict 只复制参数值，不复制 requires_grad 标志。
        """
        # 收集需要修改的模型列表（global_model + model_pool）
        models_to_update = [self.global_model]
        if hasattr(self, 'model_pool') and self.model_pool is not None:
            models_to_update.append(self.model_pool.get_model(0))
        
        frozen_count = 0
        for model in models_to_update:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if '_private' in name:
                        # 冻结私有参数（Phase 1 不使用私有分支）
                        param.requires_grad = False
                        if model is self.global_model:
                            frozen_count += 1
                    elif 'lambda_k' in name:
                        # 门控固定为 0（禁用私有分支输出）
                        # logit = -100 → sigmoid(-100) ≈ 0
                        param.data.fill_(-100.0)
                        param.requires_grad = False
                        if model is self.global_model:
                            frozen_count += 1
        
        phase1_epochs = getattr(self.args, 'phase1_epochs', 50)
        cprint(f"\n[FedTP] Phase 1 初始化完成:")
        cprint(f"  - 冻结 {frozen_count} 个私有/门控参数（已同步到 ModelPool）")
        cprint(f"  - Phase 1 轮次: 0 ~ {phase1_epochs - 1} (共 {phase1_epochs} 轮)")
        cprint(f"  - Phase 2 轮次: {phase1_epochs} ~ {self.args.epochs - 1} (共 {self.args.epochs - phase1_epochs} 轮)")
    
    def _fedtp_switch_to_phase2(self) -> None:
        """FedTP: 从 Phase 1 切换到 Phase 2 — 冻结全局 LoRA，解冻私有 LoRA
        
        注意：必须同时修改 global_model 和 model_pool 的 requires_grad，
        因为 load_state_dict 只复制参数值，不复制 requires_grad 标志。
        如果不同步 model_pool，Phase 2 的优化器会训练错误的参数：
        - global LoRA 仍然 requires_grad=True（应该冻结）
        - private LoRA 仍然 requires_grad=False（应该训练）
        导致 Phase 2 完全失效。
        """
        # 收集需要修改的模型列表（global_model + model_pool）
        models_to_update = [self.global_model]
        if hasattr(self, 'model_pool') and self.model_pool is not None:
            models_to_update.append(self.model_pool.get_model(0))
        
        global_frozen = 0
        private_unfrozen = 0
        gate_set = 0
        
        for model in models_to_update:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if '_private' in name:
                        # 解冻私有参数（Phase 2 训练私有分支）
                        param.requires_grad = True
                        if model is self.global_model:
                            private_unfrozen += 1
                    elif 'lambda_k' in name:
                        # 门控固定为 1.0（等比例相加：shared + private）
                        # logit = 100 → sigmoid(100) ≈ 1.0
                        param.data.fill_(100.0)
                        param.requires_grad = False
                        if model is self.global_model:
                            gate_set += 1
                    elif 'lora_' in name:
                        # 冻结全局 LoRA（Phase 1 已收敛）
                        param.requires_grad = False
                        if model is self.global_model:
                            global_frozen += 1
                    # Head 保持可训练（继续随私有分支微调）
        
        cprint(f"[FedTP] Phase 2 参数切换（已同步到 ModelPool）:")
        cprint(f"  - 冻结 {global_frozen} 个全局 LoRA 参数")
        cprint(f"  - 解冻 {private_unfrozen} 个私有 LoRA 参数")
        cprint(f"  - 设置 {gate_set} 个门控为 1.0（等比例相加）")
        
        # 打印可训练参数统计
        model = self.global_model
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        cprint(f"  - 可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def _extract_gate_layer_names(self) -> None:
        """从全局模型中提取门控参数的层名称（用于训练动态追踪）"""
        gate_granularity = getattr(self.args, 'gate_granularity', 'fine')
        self._gate_layer_names = []
        for name, param in self.global_model.named_parameters():
            if gate_granularity == 'fine' and 'lambda_k_logit' in name:
                layer_name = name.replace('.lambda_k_logit', '')
                self._gate_layer_names.append(layer_name)
            elif gate_granularity == 'coarse' and name == 'fedsdg_global_gate.lambda_k_global':
                self._gate_layer_names.append('global_gate')
                break
        if self._gate_layer_names:
            cprint(f"\n[FedSDG] 门控动态追踪: {len(self._gate_layer_names)} 个门控层")
    
    def _record_gate_history(self, epoch: int, train_metrics_list: List[Dict]) -> None:
        """记录本轮各客户端的门控值（用于训练动态追踪）"""
        all_lambda = [m['lambda_values'] for m in train_metrics_list
                      if 'lambda_values' in m and m['lambda_values']]
        if not all_lambda:
            return
        
        arr = np.array(all_lambda)  # (num_clients, num_layers)
        record = {
            'round': epoch,
            'layer_means': arr.mean(axis=0).tolist(),
            'layer_stds': arr.std(axis=0).tolist(),
            'layer_mins': arr.min(axis=0).tolist(),
            'layer_maxs': arr.max(axis=0).tolist(),
            'overall_mean': float(arr.mean()),
            'overall_std': float(arr.std()),
            'overall_min': float(arr.min()),
            'overall_max': float(arr.max()),
            'per_client_means': arr.mean(axis=1).tolist(),
            'num_clients': len(all_lambda),
        }
        self.state.gate_history.append(record)
    
    def _save_gate_history(self) -> None:
        """保存门控历史到 JSON 文件"""
        if not self.state.gate_history:
            return
        import json
        log_dir = self.hydra_run_dir if self.hydra_run_dir else self.log_dir
        path = os.path.join(log_dir, 'gate_history.json')
        data = {
            'gate_layer_names': self._gate_layer_names,
            'gate_init_value': getattr(self.args, 'gate_init_value', 0.0),
            'lambda1': getattr(self.args, 'lambda1', 0.0),
            'gate_penalty_type': getattr(self.args, 'gate_penalty_type', 'unilateral'),
            'history': self.state.gate_history,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        cprint(f"[输出] 门控历史已保存: {path}")
    
    def _visualize_gate_dynamics(self) -> None:
        """生成门控训练动态可视化图表"""
        if not self.state.gate_history:
            return
        try:
            from ..utils.visualization import visualize_gate_dynamics_all
            log_dir = self.hydra_run_dir if self.hydra_run_dir else self.log_dir
            visualize_gate_dynamics_all(
                gate_history=self.state.gate_history,
                layer_names=self._gate_layer_names,
                save_dir=log_dir,
                gate_init_value=getattr(self.args, 'gate_init_value', 0.0),
            )
        except Exception as e:
            cprint(f"[警告] 门控动态可视化失败: {e}")
    
    def _log_fedsdg_diagnostics(self, epoch: int, idx: int, model: nn.Module) -> None:
        """记录 FedSDG 诊断信息（使用 console_logger）"""
        # 提取 lambda_k 值
        lambda_k_values = []
        for name, param in model.named_parameters():
            if 'lambda_k_logit' in name:
                lambda_k_values.append(torch.sigmoid(param).item())
        
        # 使用 console_logger 输出
        log_fedsdg_diagnostics(epoch, idx, lambda_k_values)
    
    # =========================================================================
    # 结束阶段
    # =========================================================================
    
    def _finalize(self) -> Dict[str, Any]:
        """完成训练并保存结果"""
        cprint_section("[最终评估]")
        
        # 加载最佳模型（如果存在）
        if self.best_model_saver:
            best_checkpoint = self.best_model_saver.load_best()
            if best_checkpoint:
                cprint(f"\n[BestModelSaver] 加载最佳模型进行最终评估")
                cprint(f"  最佳轮次: epoch {best_checkpoint['epoch']}")
                cprint(f"  最佳 val_acc_avg: {best_checkpoint['best_score']:.4f}")
                
                # 加载模型权重
                self.global_model.load_state_dict(best_checkpoint['model_state_dict'])
                
                # 使用 private_state_manager 恢复私有状态
                if best_checkpoint.get('local_private_states'):
                    self.private_state_manager.restore_all(best_checkpoint['local_private_states'])
                    if self.private_state_manager.enabled:
                        cprint(f"[BestModelSaver] 已恢复 {self.args.alg.upper()} 私有状态")
            else:
                cprint("[BestModelSaver] 未找到最佳模型 checkpoint，使用当前模型")
        
        # 最终评估
        test_acc, test_loss = test_inference(self.args, self.global_model, self.test_dataset)
        # 使用 private_state_manager.states
        local_acc, local_loss, client_results = evaluate_local_personalization(
            args=self.args,
            global_model=self.global_model,
            test_dataset=self.test_dataset,
            user_groups_test=self.user_groups_test,
            local_private_states=self.private_state_manager.states,
            sample_clients=None
        )
        
        cprint(f'\n {self.args.epochs} 轮全局训练后的结果:')
        cprint(f"|---- local/test_acc_avg: {100*local_acc:.2f}%")
        cprint(f"|---- 全局测试准确率: {100*test_acc:.2f}%")
        cprint(f"|---- 准确率差异 (Local - Global): {100*(local_acc - test_acc):+.2f}%")
        
        total_time = time.time() - self.start_time
        cprint(f'\n 总运行时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)')
        
        # 保存结果
        results = self._save_results(test_acc, test_loss, local_acc, local_loss, client_results)
        
        # 生成报告
        metrics = ExperimentMetrics(
            train_accuracy=self.state.train_accuracy,
            train_loss=self.state.train_loss,
            test_acc=test_acc,
            test_loss=test_loss,
            local_acc=local_acc,
            local_loss=local_loss,
            comm_stats=self.comm_stats,
            cumulative_comm_mb=self.state.cumulative_comm_mb,
            total_time=total_time,
            efficiency_score=self.state.efficiency_score,
            efficiency_score_per_gb=self.state.efficiency_score_per_gb,
            best_efficiency_epoch=self.state.best_efficiency_epoch,
            best_efficiency_score=self.state.best_efficiency_score,
            client_results=client_results,
        )
        
        summary_text = generate_summary_report(self.args, metrics)
        self._save_summary(summary_text)
        
        # 生成 final_results.json
        self._save_final_results_json(test_acc, test_loss, local_acc, local_loss, total_time, client_results)
        
        # 记录最终汇总指标（WandB Summary）
        self.logger.log_summary({
            'final/test_acc': test_acc,
            'final/test_loss': test_loss,
            'final/local_acc': local_acc,
            'final/local_loss': local_loss,
            'final/acc_gap': local_acc - test_acc,
            'final/total_comm_mb': self.state.cumulative_comm_mb,
            'final/total_time_min': total_time / 60,
            'final/efficiency_per_gb': self.state.efficiency_score_per_gb,
        })
        
        # 上传模型 Artifact（WandB）- 使用 checkpoint_best.pt（如果存在）
        if hasattr(self.logger, 'log_artifact') and self.best_model_saver:
            checkpoint_path = self.best_model_saver.checkpoint_path
            if os.path.exists(checkpoint_path):
                self.logger.log_artifact(
                    name=f"{self.args.alg}_{self.args.dataset}_model",
                    artifact_type="model",
                    path=checkpoint_path
                )
        
        # 最终检查点
        if self.checkpoint_manager:
            self._save_final_checkpoint(test_acc, test_loss, local_acc, local_loss, client_results)
        
        # FedSDG / FedALT 可视化
        if self.args.alg in ('fedsdg', 'fedalt'):
            self._save_gate_history()
            self._visualize_gate_dynamics()
            self._visualize_gates()
        
        self.logger.close()
        
        # 记录最终结果到 console.log（使用 console_logger）
        log_final(
            total_epochs=self.args.epochs,
            test_acc=test_acc,
            local_acc=local_acc,
            cumulative_comm_mb=self.state.cumulative_comm_mb,
            efficiency_score_per_gb=self.state.efficiency_score_per_gb,
            total_time=total_time,
            hydra_run_dir=self.hydra_run_dir
        )
        
        # 关闭控制台日志
        close_console_logger()
        
        self._print_completion_info()
        
        return results
    
    def _save_results(self, test_acc, test_loss, local_acc, local_loss, client_results) -> Dict:
        """保存结果文件"""
        result_path = get_result_path(self.args.alg, self.experiment_name)
        
        # 处理 args 序列化
        try:
            args_dict = vars(self.args) if hasattr(self.args, '__dict__') else self.args.to_dict()
        except Exception:
            args_dict = {}
        
        results = {
            'train_loss': self.state.train_loss,
            'train_accuracy': self.state.train_accuracy,
            'eval_rounds': self.state.eval_rounds,
            'comm_stats': self.comm_stats,
            'total_comm_volume_mb': self.state.cumulative_comm_mb,
            'global_test_acc': test_acc,
            'global_test_loss': test_loss,
            'local_avg_acc': local_acc,
            'local_avg_loss': local_loss,
            'client_results': client_results,
            'args': args_dict,
        }
        
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)
        cprint(f"\n[输出] 结果已保存: {result_path}")
        
        return results
    
    def _save_summary(self, summary_text: str) -> None:
        """保存总结报告"""
        self.logger.add_text('Experiment_Summary', summary_text, 0)
        summary_path = get_summary_path(self.args.alg, self.experiment_name)
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        cprint(f"[输出] 摘要已保存: {summary_path}")
    
    def _save_final_results_json(
        self, 
        test_acc: float, 
        test_loss: float, 
        local_acc: float, 
        local_loss: float,
        total_time: float,
        client_results: list
    ) -> None:
        """
        保存 final_results.json
        
        使用 results_formatter 中的辅助函数构建结果
        """
        from ..utils.results_formatter import (
            build_experiment_info, build_hyperparameters, build_comm_stats,
            calculate_last_n_rounds_average
        )
        
        log_dir = self.hydra_run_dir if self.hydra_run_dir else self.log_dir
        formatter = ResultsFormatter(output_dir=log_dir)
        
        # 使用辅助函数构建数据
        total_rounds = len(self.state.train_loss)
        stopped_by = 'early_stopping' if (self.early_stopper and self.early_stopper.stopped) else 'max_epochs'
        early_stop_epoch = (self.early_stopper.stop_epoch + 1) if (self.early_stopper and self.early_stopper.stopped) else None
        best_epoch = self.best_model_saver.get_best_epoch() if self.best_model_saver else None
        
        experiment_info = build_experiment_info(
            experiment_name=self.experiment_name,
            args=self.args,
            total_rounds=total_rounds,
            stopped_by=stopped_by,
            early_stop_epoch=early_stop_epoch,
            best_validation_epoch=best_epoch,
            total_time_seconds=total_time,
        )
        # 补充 model 字段（build_experiment_info 没有）
        experiment_info['model'] = self.args.model
        
        hyperparameters = build_hyperparameters(self.args)
        
        comm_stats = build_comm_stats(
            comm_stats=self.comm_stats,
            total_rounds=total_rounds,
            cumulative_comm_mb=self.state.cumulative_comm_mb,
        )
        
        # 构建最终测试结果（提取客户端准确率）
        final_test_results = self._build_final_test_results(
            test_acc, test_loss, local_acc, local_loss, client_results
        )
        
        # 构建最佳验证结果
        best_validation_results = self._build_best_validation_results(local_acc, client_results)
        
        # 构建训练轨迹
        training_trajectory = {
            'train_loss': [float(x) for x in self.state.train_loss],
            'train_acc': [float(x) for x in self.state.train_accuracy],
            'eval_rounds': self.state.eval_rounds,
        }
        
        # 效率指标
        efficiency_metrics = {
            'accuracy_per_gb': self.state.efficiency_score_per_gb,
            'best_efficiency_epoch': self.state.best_efficiency_epoch,
            'best_efficiency_score': self.state.best_efficiency_score,
        }
        
        # 计算最后N轮平均值（用于论文报告）
        last_n_rounds_metrics = calculate_last_n_rounds_average(
            metrics_csv_path=self.csv_logger.csv_path,
            n_rounds=10,
        )
        
        # 保存
        formatter.save_final_results(
            experiment_info=experiment_info,
            hyperparameters=hyperparameters,
            comm_stats=comm_stats,
            final_test_results=final_test_results,
            best_validation_results=best_validation_results,
            training_trajectory=training_trajectory,
            efficiency_metrics=efficiency_metrics,
            last_n_rounds_metrics=last_n_rounds_metrics,
        )
    
    def _build_final_test_results(
        self,
        test_acc: float,
        test_loss: float,
        local_acc: float,
        local_loss: float,
        client_results: list
    ) -> Dict[str, Any]:
        """
        构建最终测试结果字典
        
        构建 final_results.json 的辅助方法
        """
        # 提取客户端准确率
        if client_results:
            if isinstance(client_results, dict):
                client_accs = [v[0] if isinstance(v, (tuple, list)) else v.get('acc', v) for v in client_results.values()]
            else:
                client_accs = [r[0] if isinstance(r, (tuple, list)) else r.get('acc', r) for r in client_results]
        else:
            client_accs = [local_acc]
        
        # 计算统计量
        client_accs_array = np.array(client_accs)
        local_acc_std = float(np.std(client_accs_array)) if len(client_accs) > 1 else 0.0
        
        return {
            'global_model': {
                'test_acc': float(test_acc),
                'test_loss': float(test_loss),
            },
            'personalized_models': {
                'test_acc_avg': float(local_acc),
                'test_acc_std': local_acc_std,
                'test_acc_min': float(np.min(client_accs_array)) if len(client_accs) > 0 else 0.0,
                'test_acc_max': float(np.max(client_accs_array)) if len(client_accs) > 0 else 0.0,
                'test_acc_p10': float(np.percentile(client_accs_array, 10)) if len(client_accs) > 0 else 0.0,
                'test_acc_p50': float(np.percentile(client_accs_array, 50)) if len(client_accs) > 0 else 0.0,
                'test_acc_p90': float(np.percentile(client_accs_array, 90)) if len(client_accs) > 0 else 0.0,
                'gap_vs_global': float(local_acc - test_acc),
                'num_clients_evaluated': len(client_results) if client_results else self.args.num_users,
            },
        }
    
    def _build_best_validation_results(
        self,
        local_acc: float,
        client_results: list
    ) -> Dict[str, Any]:
        """
        构建最佳验证结果字典
        
        构建 final_results.json 的辅助方法
        """
        # 计算客户端准确率标准差
        if client_results:
            if isinstance(client_results, dict):
                client_accs = [v[0] if isinstance(v, (tuple, list)) else v.get('acc', v) for v in client_results.values()]
            else:
                client_accs = [r[0] if isinstance(r, (tuple, list)) else r.get('acc', r) for r in client_results]
            local_acc_std = float(np.std(client_accs)) if len(client_accs) > 1 else 0.0
        else:
            local_acc_std = 0.0
        
        # 获取最佳轮次和验证准确率
        if self.best_model_saver and self.best_model_saver.get_best_epoch() is not None:
            best_epoch = self.best_model_saver.get_best_epoch() + 1  # +1 因为显示从1开始
            best_val_acc = self.best_model_saver.get_best_score()
        else:
            best_epoch = self.args.epochs
            best_val_acc = float(local_acc)
        
        return {
            'best_epoch': best_epoch,
            'val_acc_avg': float(best_val_acc),
            'val_acc_std': local_acc_std,
        }
    
    def _save_final_checkpoint(self, test_acc, test_loss, local_acc, local_loss, client_results) -> None:
        """保存最终检查点"""
        final_results = {
            'train_loss': self.state.train_loss,
            'train_accuracy': self.state.train_accuracy,
            'comm_stats': self.comm_stats,
            'total_comm_volume_mb': self.state.cumulative_comm_mb,
            'global_test_acc': test_acc,
            'global_test_loss': test_loss,
            'local_avg_acc': local_acc,
            'local_avg_loss': local_loss,
            'client_results': client_results,
        }
        
        self.checkpoint_manager.save_final(
            global_model=self.global_model,
            local_private_states=self.private_state_manager.states,
            final_results=final_results,
            args=self.args,
        )
    
    def _visualize_gates(self) -> None:
        """FedSDG 门控可视化"""
        try:
            visualize_all_gates(
                model=self.global_model,
                local_private_states=self.private_state_manager.states,
                algorithm=self.args.alg,
                experiment_name=self.experiment_name,
                prefix=f'{self.args.dataset}_{self.args.alg}_E{self.args.epochs}'
            )
        except Exception as e:
            cprint(f"[警告] 门控可视化失败: {e}")
    
    def _print_completion_info(self) -> None:
        """打印完成信息"""
        cprint_section("[训练完成]")
        cprint(f"实验名称: {self.experiment_name}")
        cprint(f"输出位置:")
        if self.run_ctx:
            cprint(f"  - Hydra 运行目录: {self.hydra_run_dir}")
            cprint(f"  - 配置快照: {self.hydra_run_dir}/.hydra/")
        cprint(f"  - TensorBoard 日志: {self.log_dir}")
        cprint("="*70 + "\n")


# =============================================================================
# 便捷函数（向后兼容）
# =============================================================================

def run_training(args, hydra_cfg: Optional[DictConfig] = None, hydra_run_dir: Optional[str] = None) -> Dict:
    """
    执行联邦学习训练（向后兼容的函数式接口）
    
    Args:
        args: 配置对象
        hydra_cfg: Hydra DictConfig
        hydra_run_dir: Hydra 运行目录
        
    Returns:
        训练结果字典
    """
    trainer = FederatedTrainer(args, hydra_cfg=hydra_cfg, hydra_run_dir=hydra_run_dir)
    return trainer.run()
