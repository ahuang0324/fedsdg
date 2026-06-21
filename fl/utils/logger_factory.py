# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一日志工厂 - 支持多后端的实验日志记录

支持后端:
- TensorBoard: 默认后端
- WandB: Weights & Biases 实验跟踪
- None: 禁用日志

使用方式:
    from fl.utils.logger_factory import LoggerFactory
    
    # 从 Hydra 配置创建
    logger = LoggerFactory.create_from_config(logging_cfg, hydra_cfg, log_dir)
    
    # 批量记录指标（推荐）
    logger.log_metrics({'train/loss': 0.5, 'test/acc': 0.85}, step=10)
    
    # 关闭
    logger.close()
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from fl.utils.console_logger import cprint


# =============================================================================
# Base Logger Interface
# =============================================================================

class BaseLogger(ABC):
    """日志记录器基类"""
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """批量记录指标（每轮调用一次）"""
        pass
    
    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """记录超参数"""
        pass
    
    @abstractmethod
    def log_summary(self, metrics: Dict[str, Any]) -> None:
        """记录最终汇总指标"""
        pass
    
    @abstractmethod
    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        """记录文本"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭日志器"""
        pass
    
    # 兼容接口
    def add_scalar(self, tag: str, value: float, step: int = None, global_step: int = None) -> None:
        """单个标量（兼容接口，建议使用 log_metrics）"""
        # 兼容 global_step 参数（TensorBoard 风格）
        actual_step = step if step is not None else global_step
        if actual_step is None:
            actual_step = 0
        self.log_metrics({tag: value}, actual_step)


# =============================================================================
# TensorBoard Logger
# =============================================================================

class TensorBoardLogger(BaseLogger):
    """TensorBoard 日志后端"""
    
    def __init__(self, log_dir: str, **kwargs):
        from tensorboardX import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir, **kwargs)
        self._log_dir = log_dir
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for tag, value in metrics.items():
            self._writer.add_scalar(tag, value, step)
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        import json
        params_str = json.dumps(params, indent=2, default=str)
        self._writer.add_text('hyperparameters', f'```json\n{params_str}\n```', 0)
    
    def log_summary(self, metrics: Dict[str, Any]) -> None:
        # TensorBoard 没有专门的 summary，记录为最后一步的标量
        pass
    
    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        self._writer.add_text(tag, text, step)
    
    def close(self) -> None:
        self._writer.close()
    
    @property
    def log_dir(self) -> str:
        return self._log_dir
    

# =============================================================================
# WandB Logger
# =============================================================================

class WandBLogger(BaseLogger):
    """
    WandB 日志后端
    
    特点：
    - 支持离线模式 (wandb_mode=offline)
    - DDP 环境仅 rank0 初始化
    - 每轮一次 wandb.log() 调用
    """
    
    def __init__(
        self, 
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        job_type: str = "train",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
        resume: Optional[str] = None,
        run_id: Optional[str] = None,
        dir: Optional[str] = None,
        **kwargs
    ):
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError("WandB 未安装。请运行: pip install wandb")
        
        self._enabled = True
        self._offline = (mode == "offline")
        self._dir = dir
        
        # DDP: 仅 rank0 初始化
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank != 0:
            self._enabled = False
            return
        
        # 初始化 run
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            job_type=job_type,
            config=config,
            tags=tags,
            notes=notes,
            mode=mode,
            resume=resume,
            id=run_id,
            dir=dir,
            reinit=True,
            **kwargs
        )
        
        mode_str = "离线" if self._offline else "在线"
        cprint(f"[WandB] {mode_str}模式已初始化: {self._run.url or '(offline)'}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if not self._enabled:
            return
        self._wandb.log(metrics, step=step)
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        self._wandb.config.update(params, allow_val_change=True)
    
    def log_summary(self, metrics: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        for key, value in metrics.items():
            self._wandb.run.summary[key] = value
    
    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        if not self._enabled:
            return
        self._wandb.log({tag: self._wandb.Html(f"<pre>{text[:2000]}</pre>")}, step=step)
    
    def log_artifact(self, name: str, artifact_type: str, path: str) -> None:
        """上传 Artifact（模型文件等）"""
        if not self._enabled:
            return
        artifact = self._wandb.Artifact(name, type=artifact_type)
        artifact.add_file(path)
        self._wandb.log_artifact(artifact)
    
    def close(self) -> None:
        if not self._enabled:
            return
        self._wandb.finish()
        
        # 离线模式提示
        if self._offline and self._dir:
            wandb_dir = os.path.join(self._dir, "wandb")
            if os.path.exists(wandb_dir):
                cprint(f"\n[WandB] 离线日志已保存，手动同步命令:")
                cprint(f"  wandb sync {wandb_dir}/offline-run-*")


# =============================================================================
# None Logger
# =============================================================================

class NoneLogger(BaseLogger):
    """空日志后端（禁用日志）"""
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        pass
    
    def log_summary(self, metrics: Dict[str, Any]) -> None:
        pass
    
    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        pass
    
    def close(self) -> None:
        pass


# =============================================================================
# Logger Factory
# =============================================================================

class LoggerFactory:
    """日志工厂"""
    
    @staticmethod
    def create_from_config(
        logging_cfg,
        hydra_cfg=None,
        log_dir: Optional[str] = None,
    ) -> BaseLogger:
        """
        从配置创建日志器
        
        Args:
            logging_cfg: logging 配置节点
            hydra_cfg: 完整 Hydra 配置（用于记录超参数）
            log_dir: 日志目录（绝对路径）
        """
        backend = getattr(logging_cfg, 'backend', 'tensorboard').lower()
        
        if backend == 'none':
            cprint("[日志] 后端: none (禁用)")
            return NoneLogger()
        
        elif backend == 'tensorboard':
            if not log_dir:
                raise ValueError("TensorBoard 需要指定 log_dir")
            cprint(f"[日志] 后端: tensorboard -> {log_dir}")
            return TensorBoardLogger(log_dir=log_dir)
        
        elif backend == 'wandb':
            from omegaconf import OmegaConf
            
            # 提取配置字典
            config_dict = None
            if hydra_cfg is not None:
                config_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
            
            # 处理 tags
            tags = None
            if logging_cfg.wandb_tags:
                tags = list(logging_cfg.wandb_tags)
            
            return WandBLogger(
                project=logging_cfg.wandb_project,
                entity=logging_cfg.wandb_entity,
                name=logging_cfg.wandb_run_name,
                group=logging_cfg.wandb_group,
                job_type=logging_cfg.wandb_job_type,
                config=config_dict,
                tags=tags,
                notes=logging_cfg.wandb_notes,
                mode=logging_cfg.wandb_mode,
                resume=logging_cfg.wandb_resume,
                run_id=logging_cfg.wandb_run_id,
                dir=log_dir,
            )
        
        else:
            raise ValueError(f"不支持的日志后端: {backend}")
    

# =============================================================================
# 便捷函数
# =============================================================================

def create_logger(
    backend: str = 'tensorboard',
    log_dir: Optional[str] = None,
    **kwargs
) -> BaseLogger:
    """创建日志器（简化接口）"""
    if backend == 'none':
        return NoneLogger()
    elif backend == 'tensorboard':
        return TensorBoardLogger(log_dir=log_dir, **kwargs)
    elif backend == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f"不支持的日志后端: {backend}")
