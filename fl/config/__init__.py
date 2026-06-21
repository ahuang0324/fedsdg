# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hydra 配置模块

提供：
- Structured Configs (Dataclasses): 类型安全的配置定义
- ConfigAdapter: 将 Hydra DictConfig 转换为兼容 args 的对象

使用示例:
    from fl.config import Config, ConfigAdapter
    
    # Hydra main 函数中
    @hydra.main(config_path="conf", config_name="config")
    def main(cfg: DictConfig):
        args = ConfigAdapter(cfg)  # 兼容原有代码
        # args.epochs, args.alg 等属性可正常访问
"""

from fl.config.schemas import (
    Config,
    AlgorithmConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    FederatedConfig,
    SystemConfig,
    CheckpointConfig,
    LoggingConfig,
    register_configs,
)
# 这个就是对外暴露接口，外部文件想访问上面这些文件的类
# 一般情况下需要from fl.config.adapter import ConfigAdapter
# 但是现在只需要 from fl.config import Config就行了
# 就相当于这个文件，把当前目录下的各个类 都拿到手里来了。

from fl.config.adapter import ConfigAdapter, ConfigValidationError

# 控制 Import* 的哪些内容被暴露出去
__all__ = [
    # Structured Configs
    'Config',
    'AlgorithmConfig',
    'DatasetConfig',
    'ModelConfig',
    'TrainingConfig',
    'FederatedConfig',
    'SystemConfig',
    'CheckpointConfig',
    'LoggingConfig',
    'register_configs',
    # Adapter
    'ConfigAdapter',
    'ConfigValidationError',
]
