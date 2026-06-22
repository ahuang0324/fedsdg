# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Federated learning experiment entry point.

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
    
Argparse interface:
    USE_LEGACY_CONFIG=1 python main.py --alg fedavg --model cnn --dataset cifar --epochs 100
"""

import os
import sys

import hydra
from omegaconf import DictConfig

from fl.config import ConfigAdapter, register_configs

register_configs()

from fl.core import FederatedTrainer


# =============================================================================
# Hydra 入口
# =============================================================================

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra 主入口函数
    
    Args:
        cfg: Hydra DictConfig 配置对象
    """
    from hydra.core.hydra_config import HydraConfig
    hydra_run_dir = HydraConfig.get().runtime.output_dir
    
    print("\n" + "="*70)
    print("[Hydra] Configuration loaded")
    print(f"[Hydra] Run directory: {hydra_run_dir}")
    print("="*70)
    
    args = ConfigAdapter(cfg)
    
    trainer = FederatedTrainer(args, hydra_cfg=cfg, hydra_run_dir=hydra_run_dir)
    trainer.run()


def main_legacy():
    """
    Argparse entry point enabled by USE_LEGACY_CONFIG=1.
    """
    from options import args_parser
    
    args = args_parser()
    trainer = FederatedTrainer(args)
    trainer.run()


if __name__ == '__main__':
    if os.environ.get('USE_LEGACY_CONFIG', '').lower() in ('1', 'true', 'yes'):
        print("[Config] Using argparse interface")
        main_legacy()
    else:
        main()
