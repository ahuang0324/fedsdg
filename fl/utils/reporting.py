# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验报告生成模块

生成 Markdown 格式的实验总结报告，将报告生成逻辑从 main.py 中解耦。

Usage:
    from fl.utils import generate_summary_report, ExperimentMetrics
    
    metrics = ExperimentMetrics(
        train_accuracy=train_accuracy,
        train_loss=train_loss,
        test_acc=test_acc,
        ...
    )
    
    report = generate_summary_report(args, metrics)
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ExperimentMetrics:
    """
    实验指标数据类
    
    封装实验过程中收集的所有指标，用于生成报告。
    """
    # 训练过程指标
    train_accuracy: List[float]
    train_loss: List[float]
    
    # 最终评估指标
    test_acc: float
    test_loss: float
    local_acc: float
    local_loss: float
    
    # 通信统计
    comm_stats: Dict[str, Any]
    cumulative_comm_mb: float
    
    # 时间统计
    total_time: float
    
    # 效率指标
    efficiency_score: float = 0.0
    efficiency_score_per_gb: float = 0.0
    best_efficiency_epoch: int = 0
    best_efficiency_score: float = 0.0
    
    # 客户端详细结果（可选）
    client_results: Optional[Dict[int, tuple]] = None


def generate_summary_report(args, metrics: ExperimentMetrics) -> str:
    """
    生成实验总结报告
    
    Args:
        args: 配置对象
        metrics: 实验指标
        
    Returns:
        Markdown 格式的报告文本
    """
    # 计算派生指标
    full_model_size_mb = metrics.comm_stats['total_params'] * 4 / (1024 * 1024)
    savings_ratio = (1 - metrics.comm_stats['comm_size_mb'] / full_model_size_mb) * 100
    
    # 考虑客户端数量的 FedAvg 估算
    m = max(int(args.frac * args.num_users), 1)
    fedavg_estimated_mb = 2 * m * full_model_size_mb * args.epochs
    saved_mb = fedavg_estimated_mb - metrics.cumulative_comm_mb
    savings_multiplier = fedavg_estimated_mb / metrics.cumulative_comm_mb if metrics.cumulative_comm_mb > 0 else 1
    
    # 构建报告各部分
    sections = [
        _section_header(),
        _section_config(args),
        _section_performance(args, metrics),
        _section_communication(
            args, metrics, full_model_size_mb, savings_ratio,
            fedavg_estimated_mb, saved_mb, savings_multiplier, m
        ),
    ]
    
    # 算法特定部分
    if args.alg in ('fedlora', 'fedsdg', 'feddpa', 'fedsalora', 'local_only'):
        sections.append(_section_lora(args))
    
    if args.alg == 'fedsdg':
        sections.append(_section_fedsdg(args))
    
    # 结论
    sections.append(_section_conclusion(
        args, metrics, savings_ratio, saved_mb, savings_multiplier
    ))
    
    # 页脚
    sections.append(_section_footer())
    
    return "\n".join(sections)


# =============================================================================
# 报告模板函数
# =============================================================================

def _section_header() -> str:
    """报告标题"""
    return "# 联邦学习实验总结报告\n"


def _section_config(args) -> str:
    """基本配置部分"""
    model_variant = getattr(args, 'model_variant', 'scratch')
    return f"""## 基本配置
- **算法**: {args.alg.upper()}
- **模型**: {args.model.upper()} ({model_variant})
- **数据集**: {args.dataset.upper()} ({args.num_classes} 类)
- **训练轮次**: {args.epochs}
- **客户端数量**: {args.num_users}
- **参与率**: {args.frac * 100:.1f}%
- **本地训练轮次**: {args.local_ep}
- **本地批次大小**: {args.local_bs}
- **学习率**: {args.lr}
- **优化器**: {args.optimizer}
- **Dirichlet Alpha**: {args.dirichlet_alpha}
"""


def _section_performance(args, metrics: ExperimentMetrics) -> str:
    """性能指标部分"""
    return f"""## 性能指标
### 全局模型性能
- **最终训练准确率**: {metrics.train_accuracy[-1] * 100:.2f}%
- **全局测试准确率**: {metrics.test_acc * 100:.2f}%
- **最终训练损失**: {metrics.train_loss[-1]:.4f}
- **全局测试损失**: {metrics.test_loss:.4f}

### 本地个性化性能（双重评估）
- **本地平均测试准确率**: {metrics.local_acc * 100:.2f}%
- **本地平均测试损失**: {metrics.local_loss:.4f}
- **准确率差异 (Local - Global)**: {(metrics.local_acc - metrics.test_acc) * 100:+.2f}%

### 训练时间
- **总训练时间**: {metrics.total_time / 60:.2f} 分钟 ({metrics.total_time:.2f} 秒)
- **平均每轮时间**: {metrics.total_time / args.epochs:.2f} 秒
"""


def _section_communication(
    args, 
    metrics: ExperimentMetrics,
    full_model_mb: float,
    savings_ratio: float,
    fedavg_mb: float,
    saved_mb: float,
    multiplier: float,
    clients_per_round: int
) -> str:
    """通信效率分析部分"""
    return f"""## 通信效率分析
### 模型参数统计
- **总参数量**: {metrics.comm_stats['total_params']:,} ({full_model_mb:.2f} MB)
- **可训练参数**: {metrics.comm_stats['trainable_params']:,}
- **每客户端通信参数**: {metrics.comm_stats['comm_params']:,} ({metrics.comm_stats['comm_size_mb']:.2f} MB)
- **压缩率**: {metrics.comm_stats['compression_ratio']:.2f}%

### 通信量统计
- **每轮参与客户端数**: {clients_per_round}
- **单轮通信量**: {2 * clients_per_round * metrics.comm_stats['comm_size_mb']:.2f} MB
- **总通信量**: {metrics.cumulative_comm_mb:.2f} MB ({metrics.cumulative_comm_mb / 1024:.2f} GB)
- **通信节省率**: {savings_ratio:.2f}%

### 相比 FedAvg 的优势 (传输完整模型)
- **FedAvg 预估通信量**: {fedavg_mb:.2f} MB ({fedavg_mb / 1024:.2f} GB)
- **节省的通信量**: {saved_mb:.2f} MB ({saved_mb / 1024:.2f} GB)
- **节省倍数**: {multiplier:.2f}x
- **通信效率提升**: {(multiplier - 1) * 100:.1f}%

### 效率评分
- **准确率/MB**: {metrics.efficiency_score:.6f}
- **准确率/GB**: {metrics.efficiency_score_per_gb:.4f}
- **最佳效率轮次**: 第 {metrics.best_efficiency_epoch + 1} 轮
- **最佳效率得分**: {metrics.best_efficiency_score:.6f}
"""


def _section_lora(args) -> str:
    """LoRA 配置部分"""
    lora_r = getattr(args, 'lora_r', 8)
    lora_alpha = getattr(args, 'lora_alpha', 16)
    train_head = getattr(args, 'lora_train_mlp_head', True)
    
    return f"""## LoRA 配置 (FedLoRA/FedSDG)
- **LoRA 秩 (r)**: {lora_r}
- **LoRA Alpha**: {lora_alpha}
- **训练分类头**: {'是' if train_head else '否'}
"""


def _section_fedsdg(args) -> str:
    """FedSDG 特定配置部分"""
    agg_method_desc = {
        'fedavg': 'FedAvg 均匀加权聚合',
        'alignment': '基于对齐度加权的 FedSDG 聚合算法'
    }
    
    gate_penalty_type = getattr(args, 'gate_penalty_type', 'bilateral')
    lambda1 = getattr(args, 'lambda1', 0.01)
    lambda2 = getattr(args, 'lambda2', 0.001)
    lr_gate = getattr(args, 'lr_gate', 0.01)
    server_agg_method = getattr(args, 'server_agg_method', 'fedavg')
    
    return f"""## FedSDG 配置
- **双路架构**: 全局分支 + 私有分支
- **门控惩罚类型**: {gate_penalty_type}
- **Lambda1 (门控惩罚)**: {lambda1}
- **Lambda2 (私有 L2)**: {lambda2}
- **门控学习率**: {lr_gate}
- **服务端聚合算法**: {server_agg_method} ({agg_method_desc.get(server_agg_method, 'unknown')})
"""


def _section_conclusion(
    args, 
    metrics: ExperimentMetrics,
    savings_ratio: float,
    saved_mb: float,
    multiplier: float
) -> str:
    """结论部分"""
    saved_gb = saved_mb / 1024
    
    if args.alg == 'fedlora':
        return f"""## 结论
本次实验使用 **FedLoRA** 算法，成功将通信开销降低至原来的 **{100 - savings_ratio:.2f}%**。
相比传统 FedAvg，节省了 **{saved_gb:.2f} GB** 的通信流量，相当于减少了 **{multiplier:.2f}** 倍的通信成本。
同时保持了 **{metrics.test_acc * 100:.2f}%** 的测试准确率，展现了参数高效联邦学习（PEFT）的强大优势。

**投入产出比**: 每传输 1 MB 数据，获得 {metrics.efficiency_score:.6f} 的准确率提升，
即每 GB 流量可换取 {metrics.efficiency_score_per_gb:.4f} 的准确率收益。
"""
    
    elif args.alg == 'fedsdg':
        return f"""## 结论
本次实验使用 **FedSDG** 算法，通过双路架构（全局分支 + 私有分支）对抗 Non-IID 数据分布。
通信开销与 FedLoRA 保持一致，降低至原来的 **{100 - savings_ratio:.2f}%**。
相比传统 FedAvg，节省了 **{saved_gb:.2f} GB** 的通信流量，相当于减少了 **{multiplier:.2f}** 倍的通信成本。

**FedSDG 特点**:
- 私有参数（lora_A_private, lora_B_private, lambda_k）仅在客户端本地更新
- 全局参数（lora_A, lora_B）参与服务器聚合
- 通过门控机制自动学习全局/私有分支的最优权重
- 最终测试准确率: **{metrics.test_acc * 100:.2f}%**
- 本地个性化准确率: **{metrics.local_acc * 100:.2f}%**

**投入产出比**: 每传输 1 MB 数据，获得 {metrics.efficiency_score:.6f} 的准确率提升，
即每 GB 流量可换取 {metrics.efficiency_score_per_gb:.4f} 的准确率收益。
"""
    
    else:  # fedavg
        return f"""## 结论
本次实验使用 **FedAvg** 算法，传输完整模型参数进行联邦学习。
总通信量为 **{metrics.cumulative_comm_mb / 1024:.2f} GB**，最终测试准确率达到 **{metrics.test_acc * 100:.2f}%**。

**投入产出比**: 每传输 1 MB 数据，获得 {metrics.efficiency_score:.6f} 的准确率提升，
即每 GB 流量可换取 {metrics.efficiency_score_per_gb:.4f} 的准确率收益。
"""


def _section_footer() -> str:
    """页脚"""
    return f"""---
*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""


# =============================================================================
# 辅助函数
# =============================================================================

def format_number(n: float, precision: int = 2) -> str:
    """格式化数字，大数使用千位分隔符"""
    if n >= 1e6:
        return f"{n/1e6:.{precision}f}M"
    elif n >= 1e3:
        return f"{n/1e3:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"


def generate_brief_summary(args, metrics: ExperimentMetrics) -> str:
    """
    生成简短的实验摘要（一行）
    
    适用于日志输出或文件名。
    """
    return (
        f"{args.alg.upper()} | "
        f"{args.dataset} | "
        f"Acc: {metrics.test_acc*100:.1f}% | "
        f"Local: {metrics.local_acc*100:.1f}% | "
        f"Comm: {metrics.cumulative_comm_mb/1024:.2f}GB | "
        f"Time: {metrics.total_time/60:.1f}min"
    )
