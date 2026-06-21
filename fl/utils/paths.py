# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
路径管理工具

提供项目中所有目录路径的统一管理，支持两种模式：
1. Hydra 模式（默认）：所有输出集中到 Hydra 运行目录
2. 独立模式：按类型分散存储（兼容接口）

目录结构（Hydra 模式）：
    project_root/
    ├── datasets/                    # 数据目录
    ├── conf/                        # Hydra 配置
    ├── logs/                        # TensorBoard 软链接
    │   └── {algorithm}/
    │       └── {experiment_name}/   -> 软链接到 outputs/.../tensorboard/
    └── outputs/
        └── YYYY-MM-DD/
            └── HH-MM-SS/            # 一次完整实验
                ├── .hydra/          # Hydra 配置快照
                ├── main.log         # 运行日志
                ├── tensorboard/     # TensorBoard 日志
                ├── model_final.pth  # 最终模型
                ├── results.pkl      # 结果文件
                ├── summary.txt      # 摘要报告
                └── visualizations/  # 可视化图表
"""

import os
import time
from typing import Optional
from pathlib import Path


# =============================================================================
# 项目根目录（固定，不受 Hydra chdir 影响）
# =============================================================================
# 通过 __file__ 定位到项目根目录
_CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = str(_CURRENT_FILE.parent.parent.parent)

# =============================================================================
# 数据目录（固定路径）
# =============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets')
PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'preprocessed')

# =============================================================================
# 配置目录（固定路径）
# =============================================================================
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')

# =============================================================================
# 日志目录（TensorBoard 软链接目录）
# =============================================================================
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# =============================================================================
# 输出目录（Hydra 管理）
# =============================================================================
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# 兼容接口的目录常量（保留兼容，仅用于清理）
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, 'checkpoints')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUTS_DIR, 'results')
SUMMARIES_DIR = os.path.join(OUTPUTS_DIR, 'summaries')
VISUALIZATIONS_DIR = os.path.join(OUTPUTS_DIR, 'visualizations')


# =============================================================================
# 运行时上下文管理
# =============================================================================

class HydraRunContext:
    """
    Hydra 运行时上下文管理器
    
    用于在 Hydra 运行期间管理输出路径，确保所有输出集中到运行目录。
    
    使用方式:
        # 在 main.py 中初始化
        ctx = HydraRunContext.initialize(hydra_run_dir, args)
        
        # 获取各种路径
        tb_dir = ctx.get_tensorboard_dir()
        model_path = ctx.get_model_path()
    """
    
    _instance: Optional['HydraRunContext'] = None
    
    def __init__(
        self, 
        hydra_run_dir: str, 
        algorithm: str, 
        experiment_name: str
    ):
        """
        初始化运行上下文
        
        Args:
            hydra_run_dir: Hydra 运行目录（如 outputs/2026-01-13/23-30-13）
            algorithm: 算法名称（fedavg, fedlora, fedsdg）
            experiment_name: 实验名称（包含参数和时间戳）
        """
        self.hydra_run_dir = hydra_run_dir
        self.algorithm = algorithm
        self.experiment_name = experiment_name
        
        # 确保运行目录存在
        ensure_dir(self.hydra_run_dir)
    
    @classmethod
    def initialize(
        cls, 
        hydra_run_dir: str, 
        algorithm: str, 
        experiment_name: str
    ) -> 'HydraRunContext':
        """
        初始化全局运行上下文（单例模式）
        
        Args:
            hydra_run_dir: Hydra 运行目录
            algorithm: 算法名称
            experiment_name: 实验名称
            
        Returns:
            HydraRunContext 实例
        """
        cls._instance = cls(hydra_run_dir, algorithm, experiment_name)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> Optional['HydraRunContext']:
        """获取当前运行上下文实例"""
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """重置运行上下文（用于测试）"""
        cls._instance = None
    
    # -------------------------------------------------------------------------
    # 路径获取方法
    # -------------------------------------------------------------------------
    
    def get_tensorboard_dir(self) -> str:
        """获取 TensorBoard 日志目录"""
        tb_dir = os.path.join(self.hydra_run_dir, 'tensorboard')
        return ensure_dir(tb_dir)
    
    def get_model_path(self, suffix: str = 'final') -> str:
        """获取模型保存路径"""
        return os.path.join(self.hydra_run_dir, f'model_{suffix}.pth')
    
    def get_result_path(self) -> str:
        """获取结果文件路径"""
        return os.path.join(self.hydra_run_dir, 'results.pkl')
    
    def get_summary_path(self) -> str:
        """获取摘要文件路径"""
        return os.path.join(self.hydra_run_dir, 'summary.txt')
    
    def get_visualization_dir(self) -> str:
        """获取可视化目录"""
        viz_dir = os.path.join(self.hydra_run_dir, 'visualizations')
        return ensure_dir(viz_dir)
    
    def get_visualization_path(self, filename: str) -> str:
        """获取可视化文件路径"""
        return os.path.join(self.get_visualization_dir(), filename)
    
    def get_checkpoint_dir(self) -> str:
        """获取检查点目录"""
        ckpt_dir = os.path.join(self.hydra_run_dir, 'checkpoints')
        return ensure_dir(ckpt_dir)
    
    # -------------------------------------------------------------------------
    # 软链接管理
    # -------------------------------------------------------------------------
    
    def create_tensorboard_symlink(self) -> Optional[str]:
        """
        在 logs/ 目录创建 TensorBoard 软链接
        
        结构: logs/{algorithm}/{experiment_name}/ -> {hydra_run_dir}/tensorboard/
        
        Returns:
            软链接路径，如果创建失败返回 None
        """
        # 目标目录（TensorBoard 日志实际位置）
        target_dir = self.get_tensorboard_dir()
        
        # 软链接位置
        link_parent = os.path.join(LOGS_DIR, self.algorithm)
        ensure_dir(link_parent)
        link_path = os.path.join(link_parent, self.experiment_name)
        
        # 如果已存在同名链接或目录，先删除
        if os.path.islink(link_path):
            os.unlink(link_path)
        elif os.path.exists(link_path):
            # 如果是真实目录（旧数据），不删除，改用带后缀的名称
            link_path = f"{link_path}_hydra"
            if os.path.islink(link_path):
                os.unlink(link_path)
        
        try:
            # 创建相对路径的软链接（更可移植）
            rel_target = os.path.relpath(target_dir, link_parent)
            os.symlink(rel_target, link_path)
            return link_path
        except OSError as e:
            # Windows 或权限问题，回退到不创建软链接
            print(f"[警告] 无法创建软链接: {e}")
            print(f"[警告] TensorBoard 日志位于: {target_dir}")
            return None


# =============================================================================
# 工具函数
# =============================================================================

def ensure_dir(path: str) -> str:
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)
    return path


def get_data_dir(dataset_name: str) -> str:
    """获取指定数据集的目录路径"""
    return os.path.join(DATA_DIR, dataset_name)


def sanitize_experiment_note(note: str) -> str:
    """
    清理实验备注字符串，确保文件名安全
    
    规则:
    - 移除或替换不安全的文件名字符 (/, \, :, *, ?, ", <, >, |)
    - 限制长度为 50 字符
    - 如果非空且不以 _ 或 - 开头，自动添加 _ 前缀
    - 空字符串返回空字符串
    
    Args:
        note: 原始备注字符串
        
    Returns:
        清理后的备注字符串
    
    示例:
        >>> sanitize_experiment_note("loo_mean")
        "_loo_mean"
        >>> sanitize_experiment_note("_test/run")
        "_test_run"
        >>> sanitize_experiment_note("")
        ""
    """
    if not note or not note.strip():
        return ""
    
    # 移除首尾空格
    note = note.strip()
    
    # 替换不安全字符为下划线
    unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    for char in unsafe_chars:
        note = note.replace(char, '_')
    
    # 限制长度
    if len(note) > 50:
        note = note[:50]
    
    # 确保以 _ 或 - 开头（便于识别）
    if note and not note.startswith(('_', '-')):
        note = '_' + note
    
    return note


def generate_experiment_name(args) -> str:
    """
    根据参数生成规范化的实验名称
    
    格式: {dataset}_{model}_{variant}_E{epochs}_N{num_users}_C{frac}_alpha{dirichlet_alpha}_..._{timestamp}
    
    时间戳格式: YYYYMMDD_HHMMSS (确保每次运行唯一)
    
    示例:
        - cifar100_vit_pretrained_E50_N100_C0.1_alpha0.5_le5_bs10_lr0.001_20260110_143025
        - cifar100_vit_pretrained_E50_N100_C0.1_alpha0.5_r8_la16_le5_bs10_lr0.001_20260110_143025
    """
    # 基础部分
    model_variant = getattr(args, 'model_variant', 'scratch')
    parts = [
        args.dataset,
        args.model,
        model_variant,
        f'E{args.epochs}',
        f'N{args.num_users}',
        f'C{args.frac}',
        f'alpha{args.dirichlet_alpha}',
    ]
    
    # LoRA 参数
    if args.alg in ('fedlora', 'fedsdg', 'feddpa', 'fedsalora', 'local_only'):
        parts.extend([
            f'r{args.lora_r}',
            f'la{args.lora_alpha}',
        ])
    
    # FedSDG 参数
    if args.alg == 'fedsdg':
        parts.extend([
            f'l1_{args.lambda1}',
            f'l2_{args.lambda2}',
        ])
    
    # 训练参数
    parts.extend([
        f'le{args.local_ep}',
        f'bs{args.local_bs}',
        f'lr{args.lr}',
    ])
    
    # 添加时间戳（确保每次运行唯一）
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    parts.append(timestamp)
    
    # 添加实验备注（如果有）
    experiment_note = getattr(args, 'experiment_note', '')
    if experiment_note:
        # 清理备注字符串
        clean_note = sanitize_experiment_note(experiment_note)
        if clean_note:
            # 如果 clean_note 以下划线开头，移除开头的下划线
            # 因为 '_'.join(parts) 会在各部分之间自动添加下划线
            if clean_note.startswith('_'):
                clean_note = clean_note[1:]
            parts.append(clean_note)
    
    return '_'.join(parts)


# =============================================================================
# 兼容接口的路径函数（保留兼容，使用 HydraRunContext 替代）
# =============================================================================

def get_log_dir(algorithm: str, experiment_name: str) -> str:
    """
    获取日志目录路径（兼容接口）
    
    注意: 此函数保留兼容，新代码应使用 HydraRunContext.get_tensorboard_dir()
    
    如果存在 HydraRunContext，返回 Hydra 运行目录下的 tensorboard 目录
    否则回退到旧的 logs/{algorithm}/{experiment_name}/ 结构
    """
    ctx = HydraRunContext.get_instance()
    if ctx is not None:
        return ctx.get_tensorboard_dir()
    
    # 回退到旧结构
    log_path = os.path.join(LOGS_DIR, algorithm, experiment_name)
    return ensure_dir(log_path)


def get_checkpoint_dir(algorithm: str, experiment_name: str) -> str:
    """
    获取检查点目录路径（兼容接口）
    
    注意: 此函数保留兼容，新代码应使用 HydraRunContext.get_checkpoint_dir()
    """
    ctx = HydraRunContext.get_instance()
    if ctx is not None:
        return ctx.get_checkpoint_dir()
    
    # 回退到旧结构
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, algorithm, experiment_name)
    return ensure_dir(checkpoint_path)


def get_model_path(algorithm: str, experiment_name: str, suffix: str = 'final') -> str:
    """
    获取模型保存路径（兼容接口）
    
    注意: 此函数保留兼容，新代码应使用 HydraRunContext.get_model_path()
    """
    ctx = HydraRunContext.get_instance()
    if ctx is not None:
        return ctx.get_model_path(suffix)
    
    # 回退到旧结构
    ensure_dir(os.path.join(MODELS_DIR, algorithm))
    return os.path.join(MODELS_DIR, algorithm, f'{experiment_name}_{suffix}.pth')


def get_result_path(algorithm: str, experiment_name: str) -> str:
    """
    获取结果文件路径（兼容接口）
    
    注意: 此函数保留兼容，新代码应使用 HydraRunContext.get_result_path()
    """
    ctx = HydraRunContext.get_instance()
    if ctx is not None:
        return ctx.get_result_path()
    
    # 回退到旧结构
    ensure_dir(os.path.join(RESULTS_DIR, algorithm))
    return os.path.join(RESULTS_DIR, algorithm, f'{experiment_name}.pkl')


def get_summary_path(algorithm: str, experiment_name: str) -> str:
    """
    获取摘要文件路径（兼容接口）
    
    注意: 此函数保留兼容，新代码应使用 HydraRunContext.get_summary_path()
    """
    ctx = HydraRunContext.get_instance()
    if ctx is not None:
        return ctx.get_summary_path()
    
    # 回退到旧结构
    ensure_dir(os.path.join(SUMMARIES_DIR, algorithm))
    return os.path.join(SUMMARIES_DIR, algorithm, f'{experiment_name}.txt')


def get_visualization_path(algorithm: str, experiment_name: str, filename: str) -> str:
    """
    获取可视化文件路径（兼容接口）
    
    注意: 此函数保留兼容，新代码应使用 HydraRunContext.get_visualization_path()
    """
    ctx = HydraRunContext.get_instance()
    if ctx is not None:
        return ctx.get_visualization_path(filename)
    
    # 回退到旧结构
    viz_dir = os.path.join(VISUALIZATIONS_DIR, algorithm, experiment_name)
    ensure_dir(viz_dir)
    return os.path.join(viz_dir, filename)
