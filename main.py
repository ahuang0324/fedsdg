# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习实验主入口 (Hydra 配置系统)

支持的算法：
- FedAvg: 联邦平均 (传输完整模型)
- FedLoRA: 联邦低秩适应 (仅传输 LoRA 参数)
- FedSDG: 联邦结构解耦门控 (双路架构: 全局 + 私有分支)

使用方法 (Hydra):
    python main.py                                    # 默认配置
    python main.py algorithm=fedavg                   # 切换算法
    python main.py algorithm=fedsdg dataset=cifar100  # 组合配置
    python main.py training.epochs=50                 # 覆盖单个参数
    python main.py --cfg job                          # 打印完整配置
    
旧命令行方式 (兼容模式，设置环境变量 USE_LEGACY_CONFIG=1):
    USE_LEGACY_CONFIG=1 python main.py --alg fedavg --model cnn --dataset cifar --epochs 100
"""

import os
import sys

# Hydra 配置系统
import hydra
from omegaconf import DictConfig

# 配置模块
from fl.config import ConfigAdapter, register_configs

# 注册 Structured Configs
register_configs()

# 训练器
from fl.core import FederatedTrainer


# =============================================================================
# Hydra 入口
# =============================================================================

# 这里conf用的是相对路径 所以 main的位置不能变
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra 主入口函数
    
    Args:
        cfg: Hydra DictConfig 配置对象
    """
    # 获取 Hydra 运行目录（使用 HydraConfig 获取，不依赖 chdir 设置）
    from hydra.core.hydra_config import HydraConfig
    hydra_run_dir = HydraConfig.get().runtime.output_dir
    
    # 打印配置信息
    print("\n" + "="*70)
    print("[Hydra] 配置系统已加载")
    print(f"[Hydra] 运行目录: {hydra_run_dir}")
    print("="*70)
    
    # 使用 ConfigAdapter 转换为与原 args 兼容的对象
    args = ConfigAdapter(cfg)
    
    # 创建训练器并运行
    trainer = FederatedTrainer(args, hydra_cfg=cfg, hydra_run_dir=hydra_run_dir)
    trainer.run()


def main_legacy():
    """
    旧配置系统入口 (兼容模式)
    
    通过环境变量 USE_LEGACY_CONFIG=1 启用
    """
    from options import args_parser
    
    args = args_parser()
    trainer = FederatedTrainer(args)
    trainer.run()


if __name__ == '__main__':
    # 检查是否使用旧配置系统
    if os.environ.get('USE_LEGACY_CONFIG', '').lower() in ('1', 'true', 'yes'):
        print("[兼容模式] 使用旧配置系统 (argparse)")
        main_legacy()
    else:
        main()
