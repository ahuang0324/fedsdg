# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置适配器 - 将 Hydra DictConfig 转换为与原 args 兼容的对象

目的：确保 fl/ 下的核心代码无需修改，可以像使用 argparse args 一样使用配置。

使用方式:
    from fl.config import ConfigAdapter
    
    @hydra.main(config_path="conf", config_name="config")
    def main(cfg: DictConfig):
        args = ConfigAdapter(cfg)
        
        # 以下访问方式与原 argparse 完全兼容
        args.epochs      # -> cfg.training.epochs
        args.alg         # -> cfg.algorithm.name
        args.dataset     # -> cfg.dataset.name
        args.lora_r      # -> cfg.algorithm.lora_r
"""

from typing import Any, Dict, Optional, List, Set
from omegaconf import DictConfig, OmegaConf


class ConfigValidationError(ValueError):
    """配置验证错误"""
    pass


class ConfigAdapter:
    """
    配置适配器类
    
    将 Hydra 的嵌套配置结构展平为与原 argparse args 兼容的属性访问接口。
    
    属性映射规则:
        - algorithm.name -> alg
        - dataset.name -> dataset
        - model.name -> model
        - model.variant -> model_variant
        - training.* -> 直接展平
        - federated.* -> 直接展平
        - system.* -> 直接展平
        - algorithm.lora_* -> lora_*
        - algorithm.lambda* -> lambda*
        - checkpoint.enable -> enable_checkpoint
    """
    
    # 属性映射表: 原 args 属性名 -> Hydra cfg 路径
    _ATTR_MAPPING = {
        # Algorithm 
        'alg': 'algorithm.name',  # 从算法配置的 name 字段获取
        'mu': 'algorithm.mu',  # FedProx proximal term 系数
        'server_agg_method': 'algorithm.server_agg_method',
        'alignment_strategy': 'algorithm.alignment_strategy',
        'weight_transform': 'algorithm.weight_transform',
        'softmax_temperature': 'algorithm.softmax_temperature',
        'lambda_smooth': 'algorithm.lambda_smooth',
        'lambda1': 'algorithm.lambda1',
        'lambda2': 'algorithm.lambda2',
        'gate_penalty_type': 'algorithm.gate_penalty_type',
        'lr_gate': 'algorithm.lr_gate',
        'grad_clip': 'algorithm.grad_clip',
        'fix_gate': 'algorithm.fix_gate',
        'fixed_gate_value': 'algorithm.fixed_gate_value',
        'gate_init_value': 'algorithm.gate_init_value',
        'gate_warmup_rounds': 'algorithm.gate_warmup_rounds',  # 门控 Warmup 轮数
        'gate_granularity': 'algorithm.gate_granularity',  # 门控粒度配置
        'use_dynamic_alignment': 'algorithm.use_dynamic_alignment',  # Dynamic Alignment 模式
        'da_floor_gamma': 'algorithm.da_floor_gamma',  # Relative Floor 系数
        'da_target_mode': 'algorithm.da_target_mode',  # target_rms 计算模式：floor | base | geomean
        'use_da_scale': 'algorithm.use_da_scale',  # DA Scale 逐层可学习幅度标量
        'lr_da_scale': 'algorithm.lr_da_scale',  # DA Scale 学习率
        'da_detach_private_rms': 'algorithm.da_detach_private_rms',  # DA p_rms detach 消融开关
        
        # FedRep 参数
        'fedrep_rep_epochs': 'algorithm.fedrep_rep_epochs',
        'fedrep_head_epochs': 'algorithm.fedrep_head_epochs',
        'lr_head': 'algorithm.lr_head',
        
        # FedDPA 参数
        'train_mix_ratio': 'algorithm.train_mix_ratio',
        'inference_scale_factor': 'algorithm.inference_scale_factor',
        'anchor_count': 'algorithm.anchor_count',
        'lr_private': 'algorithm.lr_private',
        
        # FedSDG Head 模式参数
        'head_mode': 'algorithm.head_mode',
        
        # FedTP 参数
        'phase1_epochs': 'algorithm.phase1_epochs',
        
        # PF2LoRA 参数
        'enable_rank_pruning': 'algorithm.enable_rank_pruning',
        'target_rank_ratio': 'algorithm.target_rank_ratio',
        'pruning_start_round': 'algorithm.pruning_start_round',
        'pruning_interval': 'algorithm.pruning_interval',
        
        # Ditto 参数
        'lambda_ditto': 'algorithm.lambda_ditto',
        'ditto_reg_target': 'algorithm.ditto_reg_target',
        
        # LoRA-FAIR 参数
        'residual_mu': 'algorithm.residual_mu',
        
        # LoRA
        'lora_r': 'algorithm.lora_r',
        'lora_alpha': 'algorithm.lora_alpha',
        'lora_r_private': 'algorithm.lora_r_private',
        'lora_train_mlp_head': 'algorithm.lora_train_mlp_head',
        
        # Dataset
        'dataset': 'dataset.name',
        'num_classes': 'dataset.num_classes',
        'num_channels': 'dataset.num_channels',
        'image_size': 'dataset.image_size',
        'pathmnist_size': 'dataset.pathmnist_size',  # PathMNIST 原始尺寸选择
        'use_offline_data': 'dataset.use_offline',
        'offline_data_root': 'dataset.offline_data_root',
        'use_lmdb': 'dataset.use_lmdb',
        'cache_in_memory': 'dataset.cache_in_memory',
        'partition_mode': 'dataset.partition_mode',
        'clients_per_domain': 'dataset.clients_per_domain',
        'domains': 'dataset.domains',
        
        # Model
        'model': 'model.name',
        'model_variant': 'model.variant',
        'num_filters': 'model.num_filters',
        'kernel_num': 'model.kernel_num',
        'kernel_sizes': 'model.kernel_sizes',
        'norm': 'model.norm',
        'max_pool': 'model.max_pool',
        'vit_type': 'model.vit_type',
        
        # Training
        'epochs': 'training.epochs',
        'local_ep': 'training.local_ep',
        'local_bs': 'training.local_bs',
        'lr': 'training.lr',
        'optimizer': 'training.optimizer',
        'momentum': 'training.momentum',
        'experiment_note': 'training.experiment_note',
        'num_workers': 'training.num_workers',
        'prefetch_factor': 'training.prefetch_factor',
        
        # Data split
        'val_ratio': 'training.data.val_ratio',
        
        # Early stopping
        'early_stopping_enabled': 'training.early_stopping.enabled',
        'early_stopping_patience': 'training.early_stopping.patience',
        'early_stopping_min_delta': 'training.early_stopping.min_delta',
        'early_stopping_monitor': 'training.early_stopping.monitor',
        'early_stopping_mode': 'training.early_stopping.mode',
        
        # Evaluation
        'eval_freq': 'training.evaluation.eval_freq',
        'eval_at_last': 'training.evaluation.eval_at_last',
        
        # Federated
        'num_users': 'federated.num_users',
        'frac': 'federated.frac',
        'dirichlet_alpha': 'federated.dirichlet_alpha',
        'test_frac': 'federated.test_frac',
        'unequal': 'federated.unequal',
        
        # System
        'gpu': 'system.gpu',
        'seed': 'system.seed',
        'verbose': 'system.verbose',
        'stopping_rounds': 'system.stopping_rounds',
        
        # Checkpoint
        'enable_checkpoint': 'checkpoint.enable',
        'save_frequency': 'checkpoint.save_frequency',
        'save_client_weights': 'checkpoint.save_client_weights',
        'max_checkpoints': 'checkpoint.max_checkpoints',
        # Best model saver
        'save_best_model': 'training.save_best_model',
        
        # Logging
        'log_backend': 'logging.backend',
    }
    
    # 必备字段（训练必须有值）
    _REQUIRED_FIELDS: Set[str] = {
        'alg', 'dataset', 'model', 'epochs', 'num_users', 'frac',
        'local_ep', 'local_bs', 'lr', 'num_classes', 'dirichlet_alpha',
    }
    
    # 默认值（当配置中缺失时使用）
    _DEFAULTS: Dict[str, Any] = {
        'optimizer': 'adam',
        'momentum': 0.9,
        'verbose': 1,
        'seed': 42,
        'gpu': 0,
        'test_frac': 0.3,
        'unequal': False,
        'enable_checkpoint': False,
        'save_frequency': 5,
        'save_client_weights': True,
        'max_checkpoints': -1,
        'save_best_model': True,
        'log_backend': 'tensorboard',
        'num_workers': 2,
        'prefetch_factor': 2,
        'model_variant': 'scratch',
        'vit_type': 'tiny',
        'stopping_rounds': 10,
        'experiment_note': '',  # 实验备注，默认为空
        'use_lmdb': False,  # LMDB 加速模式，默认关闭
        'cache_in_memory': False,  # 内存缓存模式，默认关闭
        'partition_mode': None,  # 数据分区模式，None 表示使用 datasets.yaml 中的默认值
        'clients_per_domain': None,  # 每域客户端数，None 表示自动均分
        'domains': None,  # 域列表，None 表示使用全部域
        # LoRA 默认值
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_train_mlp_head': True,
        # FedSDG 默认值
        'lambda1': 0.01,
        'lambda2': 0.001,
        'gate_penalty_type': 'bilateral',
        'lr_gate': 0.01,
        'grad_clip': 1.0,
        'fix_gate': False,
        'fixed_gate_value': 0.5,
        'gate_init_value': 0.0,
        'gate_granularity': 'fine',  # 门控粒度默认值
        'use_dynamic_alignment': False,  # Dynamic Alignment 默认关闭
        'da_floor_gamma': 0.1,  # Relative Floor 系数默认 10%
        'use_da_scale': False,  # DA Scale 默认关闭
        'lr_da_scale': 0.1,  # DA Scale 学习率默认 0.1
        'da_detach_private_rms': True,  # DA p_rms detach 默认开启
        # FedRep 默认值
        'fedrep_rep_epochs': 1,
        'fedrep_head_epochs': 5,
        'lr_head': None,  # 如果为 None，使用 lr
        'server_agg_method': 'fedavg',
        # FedDPA 默认值
        'train_mix_ratio': 0.5,
        'inference_scale_factor': 0.5,
        'anchor_count': 5,
        'lr_private': None,  # 如果为 None，使用 lr
        # FedSDG Head 模式默认值
        'head_mode': 'global',  # global: head 参与聚合, private: head 不聚合
        # FedTP 默认值
        'phase1_epochs': 50,
        # PF2LoRA 默认值
        'enable_rank_pruning': True,
        'target_rank_ratio': 0.5,
        'pruning_start_round': 10,
        'pruning_interval': 5,
        'lora_r_private': None,  # 私有分支独立秩（None 则等于 lora_r）
        # Ditto 默认值
        'lambda_ditto': 0.1,
        'ditto_reg_target': 'server',
        'alignment_strategy': 'loo_mean',
        'weight_transform': 'relu_normalize',
        'softmax_temperature': 1.0,
        'lambda_smooth': 0.0,
        # FedProx 默认值
        'mu': 0.0,  # 默认不使用 proximal term
        # Data split 默认值
        'val_ratio': 0.2,
        # Early stopping 默认值
        'early_stopping_enabled': True,
        'early_stopping_patience': 20,
        'early_stopping_min_delta': 0.001,
        'early_stopping_monitor': 'val_acc_avg',
        'early_stopping_mode': 'max',
        # Evaluation 默认值
        'eval_freq': 5,
        'eval_at_last': True,
    }
    
    def __init__(self, cfg: DictConfig, validate: bool = True, verbose: bool = False):
        """
        初始化适配器
        
        Args:
            cfg: Hydra DictConfig 对象
            validate: 是否验证必备字段（默认 True）
            verbose: 是否打印配置映射信息（默认 False）
        """
        self._cfg = cfg
        self._cache: Dict[str, Any] = {}
        self._missing_fields: List[str] = []
        
        # 预处理: 将所有映射的属性缓存起来
        self._build_cache()
        
        # 验证必备字段
        if validate:
            self._validate_required_fields()
        
        # 打印配置信息（诊断用）
        if verbose:
            self._print_config_summary()
    
    def _build_cache(self) -> None:
        """构建属性缓存，应用默认值"""
        for attr_name, cfg_path in self._ATTR_MAPPING.items():
            try:
                value = OmegaConf.select(self._cfg, cfg_path)
                if value is not None:
                    self._cache[attr_name] = value
                elif attr_name in self._DEFAULTS:
                    # 应用默认值
                    self._cache[attr_name] = self._DEFAULTS[attr_name]
                else:
                    self._missing_fields.append(attr_name)
            except Exception:
                if attr_name in self._DEFAULTS:
                    self._cache[attr_name] = self._DEFAULTS[attr_name]
                else:
                    self._missing_fields.append(attr_name)
        
        # 特殊处理: 算法级别的优化器覆盖
        # 如果 algorithm.optimizer 存在且非 None，则覆盖 training.optimizer
        algo_optimizer = OmegaConf.select(self._cfg, 'algorithm.optimizer')
        if algo_optimizer is not None:
            self._cache['optimizer'] = algo_optimizer
    
    def _validate_required_fields(self) -> None:
        """验证必备字段是否存在"""
        missing_required = self._REQUIRED_FIELDS - set(self._cache.keys())
        if missing_required:
            raise ConfigValidationError(
                f"配置缺少必备字段: {sorted(missing_required)}\n"
                f"请检查 Hydra 配置文件是否正确设置了这些字段。"
            )
    
    def _print_config_summary(self) -> None:
        """打印配置摘要（诊断用）"""
        print("\n" + "="*70)
        print("[ConfigAdapter] 配置映射摘要")
        print("="*70)
        
        # 分组打印
        groups = {
            'Algorithm': ['alg', 'server_agg_method', 'alignment_strategy', 'weight_transform',
                         'softmax_temperature', 'lambda_smooth', 'lambda1', 'lambda2', 
                         'gate_penalty_type', 'lr_gate', 'grad_clip', 'fix_gate', 'fixed_gate_value', 'gate_init_value', 'gate_granularity'],
            'LoRA': ['lora_r', 'lora_alpha', 'lora_train_mlp_head'],
            'Dataset': ['dataset', 'num_classes', 'num_channels', 'image_size'],
            'Model': ['model', 'model_variant'],
            'Training': ['epochs', 'local_ep', 'local_bs', 'lr', 'optimizer', 'momentum'],
            'Federated': ['num_users', 'frac', 'dirichlet_alpha', 'test_frac'],
            'System': ['gpu', 'seed', 'verbose'],
        }
        
        for group_name, fields in groups.items():
            print(f"\n[{group_name}]")
            for field in fields:
                if field in self._cache:
                    value = self._cache[field]
                    source = "config" if field not in self._DEFAULTS or \
                             self._cache.get(field) != self._DEFAULTS.get(field) else "default"
                    print(f"  {field}: {value} ({source})")
        
        if self._missing_fields:
            print(f"\n[Missing (non-critical)]: {self._missing_fields[:10]}...")
        
        print("="*70 + "\n")
    
    def __getattr__(self, name: str) -> Any:
        """
        属性访问  Python 的特殊方法，当访问不存在的属性时会被调用。
            它实现了三层查找策略，将 Hydra 的嵌套配置映射为扁平属性访问。
        优先级:
        1. 缓存的映射属性
        2. 原始配置中的嵌套访问
        3. 抛出 AttributeError
        """
        # 避免递归
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # 1. 检查缓存
        if name in self._cache:
            return self._cache[name]
        
        # 2. 检查映射表
        if name in self._ATTR_MAPPING:
            cfg_path = self._ATTR_MAPPING[name]
            value = OmegaConf.select(self._cfg, cfg_path)
            if value is not None:
                self._cache[name] = value
                return value
        
        # 3. 尝试直接从 cfg 获取（支持 cfg.training.epochs 这样的访问）
        try:
            value = OmegaConf.select(self._cfg, name)
            if value is not None:
                return value
        except Exception:
            pass
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """属性设置"""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._cache[name] = value
    
    def __contains__(self, name: str) -> bool:
        """检查属性是否存在"""
        return name in self._cache or name in self._ATTR_MAPPING
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"ConfigAdapter({OmegaConf.to_yaml(self._cfg)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        返回与原 vars(args) 兼容的字典。
        """
        result = {}
        for attr_name in self._ATTR_MAPPING:
            try:
                result[attr_name] = getattr(self, attr_name)
            except AttributeError:
                pass
        return result
    
    def get_raw_config(self) -> DictConfig:
        """获取原始 DictConfig 对象"""
        return self._cfg
    
    @classmethod
    def get_mapping(cls) -> Dict[str, str]:
        """获取属性映射表（用于文档和配置检查）"""
        return cls._ATTR_MAPPING.copy()


def convert_config_to_args(cfg: DictConfig) -> ConfigAdapter:
    """
    便捷函数: 将 DictConfig 转换为 args 兼容对象
    
    Args:
        cfg: Hydra DictConfig
        
    Returns:
        ConfigAdapter 实例
    """
    return ConfigAdapter(cfg)
