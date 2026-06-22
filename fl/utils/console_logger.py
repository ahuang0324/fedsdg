# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一控制台日志模块

提供 cprint() 函数，同时输出到终端和 console.log 文件。

功能：
- 所有 print 信息自动写入 console.log
- 通过 to_console/to_file 参数控制输出目标
- 支持全局模式配置（可在 Hydra 配置文件中设置）

Usage:
    from fl.utils.console_logger import cprint, init_console_logger, set_console_logger_mode
    
    # 在 trainer.py 初始化（设置日志文件路径）
    init_console_logger('/path/to/console.log')
    
    # 设置全局模式（从配置文件读取）
    set_console_logger_mode(to_console=True, to_file=True)
    
    # 在任何模块中使用
    cprint("信息")                      # 使用全局模式
    cprint("信息", to_console=False)    # 仅写入文件
    cprint("信息", to_file=False)       # 仅输出到终端
"""

import sys
import time
import io
import math
from typing import Optional, Dict, Any, List
import numpy as np


# =============================================================================
# 全局状态
# =============================================================================

_log_file = None
_log_path: Optional[str] = None

# 全局默认模式
_default_to_console: bool = True
_default_to_file: bool = True


# =============================================================================
# 初始化与配置
# =============================================================================

def init_console_logger(log_path: str) -> None:
    """
    初始化控制台日志器
    
    在 trainer.py 中调用，设置 console.log 文件路径。
    
    Args:
        log_path: console.log 文件的绝对路径
    """
    global _log_file, _log_path
    
    if _log_file is not None:
        _log_file.close()
    
    _log_path = log_path
    _log_file = open(log_path, 'w', encoding='utf-8')
    _log_file.write(f"# 训练日志 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log_file.write("=" * 70 + "\n\n")
    _log_file.flush()


def close_console_logger() -> None:
    """关闭日志文件"""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def set_console_logger_mode(to_console: bool = True, to_file: bool = True) -> None:
    """
    设置全局日志模式
    
    在 trainer.py 初始化时从配置文件读取后调用。
    
    Args:
        to_console: 是否输出到终端（默认 True）
        to_file: 是否写入 console.log（默认 True）
    """
    global _default_to_console, _default_to_file
    _default_to_console = to_console
    _default_to_file = to_file


def get_console_logger_mode() -> Dict[str, bool]:
    """获取当前全局日志模式"""
    return {
        'to_console': _default_to_console,
        'to_file': _default_to_file,
    }


def is_console_logger_initialized() -> bool:
    """检查日志器是否已初始化"""
    return _log_file is not None


# =============================================================================
# 核心打印函数
# =============================================================================

def cprint(
    *args, 
    to_console: Optional[bool] = None, 
    to_file: Optional[bool] = None,
    **kwargs
) -> None:
    """
    统一打印函数
    
    同时输出到终端和 console.log 文件。
    
    Args:
        *args: 要打印的内容（与 print() 相同）
        to_console: 是否输出到终端（None 表示使用全局默认值）
        to_file: 是否写入 console.log（None 表示使用全局默认值）
        **kwargs: 传递给 print 的其他参数（如 end, sep）
    """
    # 确定实际的输出模式
    _to_console = to_console if to_console is not None else _default_to_console
    _to_file = to_file if to_file is not None else _default_to_file
    
    # 构建消息字符串
    buffer = io.StringIO()
    print(*args, file=buffer, **kwargs)
    message = buffer.getvalue()
    
    # 输出到终端
    if _to_console:
        sys.stdout.write(message)
        sys.stdout.flush()
    
    # 写入文件
    if _to_file and _log_file is not None:
        _log_file.write(message)
        _log_file.flush()


def cprint_section(title: str, to_console: Optional[bool] = None, to_file: Optional[bool] = None) -> None:
    """
    打印分节标题
    
    Args:
        title: 标题文本
        to_console: 是否输出到终端
        to_file: 是否写入文件
    """
    cprint("", to_console=to_console, to_file=to_file)
    cprint("=" * 70, to_console=to_console, to_file=to_file)
    cprint(title, to_console=to_console, to_file=to_file)
    cprint("=" * 70, to_console=to_console, to_file=to_file)


# =============================================================================
# 格式化日志函数（从 trainer.py 迁移）
# =============================================================================

def log_config(
    args,
    comm_stats: Dict[str, Any],
    experiment_name: str,
    hydra_run_dir: Optional[str],
    device: str
) -> None:
    """
    记录配置信息
    
    Args:
        args: 配置对象
        comm_stats: 通信统计
        experiment_name: 实验名称
        hydra_run_dir: Hydra 运行目录
        device: 计算设备
    """
    cprint_section("实验配置")
    cprint(f"实验名称: {experiment_name}")
    cprint(f"运行目录: {hydra_run_dir}")
    
    # 显示实验备注（如果有）
    experiment_note = getattr(args, 'experiment_note', '')
    if experiment_note:
        cprint(f"实验备注: {experiment_note}")
    
    cprint("")
    cprint("基本配置:")
    cprint(f"  算法: {args.alg}")
    cprint(f"  模型: {args.model}")
    # 数据集信息 (包含 LMDB 模式)
    use_lmdb = getattr(args, 'use_lmdb', False)
    if use_lmdb:
        cprint(f"  数据集: {args.dataset} (LMDB 加速模式)")
    else:
        cprint(f"  数据集: {args.dataset}")
    cprint(f"  训练轮次: {args.epochs}")
    cprint(f"  本地轮次: {args.local_ep}")
    cprint(f"  批次大小: {args.local_bs}")
    cprint(f"  学习率: {args.lr}")
    cprint(f"  优化器: {args.optimizer}")
    cprint("")
    cprint("联邦学习配置:")
    cprint(f"  客户端数量: {args.num_users}")
    cprint(f"  参与率: {args.frac}")
    cprint(f"  Dirichlet Alpha: {args.dirichlet_alpha}")
    
    if args.alg in ('fedlora', 'fedsdg', 'feddpa', 'fedsalora', 'local_only'):
        cprint("")
        cprint("LoRA 配置:")
        cprint(f"  LoRA 秩 (r): {args.lora_r}")
        cprint(f"  LoRA Alpha: {args.lora_alpha}")
    
    if args.alg == 'fedsdg':
        cprint("")
        cprint("FedSDG 配置:")
        cprint(f"  Lambda1 (门控惩罚): {args.lambda1}")
        cprint(f"  Lambda2 (私有L2): {args.lambda2}")
        cprint(f"  门控惩罚类型: {args.gate_penalty_type}")
        
        # ：门控粒度信息
        gate_granularity = getattr(args, 'gate_granularity', 'fine')
        cprint(f"  门控粒度: {gate_granularity}")
        
        if gate_granularity == 'coarse':
            cprint(f"  全局门控参数: 1个 (所有层共享)")
            cprint(f"  参数效率: 高 (相比细粒度的24个参数)")
        else:
            cprint(f"  细粒度门控参数: 24个 (每层独立)")
            cprint(f"  参数效率: 标准")
        
        # 固定门控配置信息
        fix_gate = getattr(args, 'fix_gate', False)
        fixed_gate_value = getattr(args, 'fixed_gate_value', 0.5)
        if fix_gate:
            cprint(f"  门控模式: 固定 (值={fixed_gate_value:.4f})")
        else:
            gate_init_value = getattr(args, 'gate_init_value', 0.0)
            m_k_init = 1.0 / (1.0 + math.exp(-gate_init_value))  # sigmoid(gate_init_value)
            cprint(f"  门控模式: 可学习")
            cprint(f"  门控学习率: {args.lr_gate}")
            cprint(f"  门控初始值: {gate_init_value:.4f} (logit) → m_k = {m_k_init:.4f}")
        
        cprint(f"  服务端聚合: {args.server_agg_method}")
        
        # 添加 alignment 相关配置（仅当使用 alignment 聚合时）
        if args.server_agg_method == 'alignment':
            cprint(f"  对齐度计算策略: {args.alignment_strategy}")
            cprint(f"  权重转换方式: {args.weight_transform}")
            if args.weight_transform == 'softmax':
                cprint(f"  Softmax 温度: {args.softmax_temperature}")
            if getattr(args, 'lambda_smooth', 0) > 0:
                cprint(f"  权重平滑系数: {args.lambda_smooth}")
    
    cprint("")
    cprint("通信统计:")
    cprint(f"  总参数量: {comm_stats['total_params']:,}")
    cprint(f"  可训练参数: {comm_stats['trainable_params']:,}")
    cprint(f"  每轮通信参数: {comm_stats['comm_params']:,}")
    cprint(f"  每轮通信量: {comm_stats['comm_size_mb']:.2f} MB")
    cprint(f"  压缩率: {comm_stats.get('compression_ratio', 100):.2f}%")
    cprint("")
    cprint(f"设备: {device}")
    cprint(f"随机种子: {getattr(args, 'seed', 'N/A')}")


def log_round(
    epoch: int,
    total_epochs: int,
    loss_avg: float,
    round_time: float,
    cumulative_comm_mb: float,
    round_metrics: Dict[str, Any]
) -> None:
    """
    记录每轮汇总
    
    Args:
        epoch: 当前轮次
        total_epochs: 总轮次
        loss_avg: 平均损失
        round_time: 轮次耗时
        cumulative_comm_mb: 累计通信量
        round_metrics: 轮次指标
    """
    line = (f"[Round {epoch+1:3d}/{total_epochs}] "
            f"loss={loss_avg:.4f}, "
            f"time={round_time:.1f}s, "
            f"comm={cumulative_comm_mb:.1f}MB")
    
    # 如果有评估结果，追加
    if round_metrics.get('test_acc') is not None:
        test_acc = round_metrics.get('test_acc', 0)
        local_test_acc = round_metrics.get('local_test_acc', 0)
        line += (f"  | test_acc={100*test_acc:.2f}%, "
                 f"local_test_acc={100*local_test_acc:.2f}%")
    
    cprint(line)


def log_fedsdg_aggregation(epoch: int, aggregation_info: Dict[str, Any]) -> None:
    """
    记录 FedSDG 聚合详细信息（含完整诊断指标）
    
    Args:
        epoch: 当前轮次
        aggregation_info: 聚合信息字典
    """
    # 每轮开始前添加空行（提高可读性）
    if epoch > 0:
        cprint("")
    
    agg_method = aggregation_info.get('agg_method', 'fedavg')
    num_aggregated = len(aggregation_info.get('aggregated_keys', []))
    num_align_keys = len(aggregation_info.get('align_keys', []))
    num_excluded = len(aggregation_info.get('excluded_keys', []))
    
    # 基本信息
    cprint(f"  [FedSDG-{agg_method.capitalize()}] Aggregated {num_aggregated} params "
           f"(align_keys={num_align_keys}), excluded {num_excluded} private/gate parameters")
    
    # Alignment 权重统计（如果有）
    if agg_method == 'alignment' and 'weight_stats' in aggregation_info:
        weight_stats = aggregation_info['weight_stats']
        cprint(f"  [FedSDG-Alignment] Weight stats: mean={weight_stats['mean']:.4f}, "
               f"std={weight_stats['std']:.4f}, range=[{weight_stats['min']:.4f}, {weight_stats['max']:.4f}]")
        
        # 有效客户端数 N_eff + entropy + diff_to_uniform + KL
        if 'n_eff' in aggregation_info:
            max_weight = aggregation_info.get('max_weight', 0)
            n_eff = aggregation_info['n_eff']
            entropy = aggregation_info.get('entropy', 0)
            diff_to_uniform = aggregation_info.get('diff_to_uniform', 0)
            kl_from_uniform = aggregation_info.get('kl_from_uniform', 0)
            cprint(f"  [FedSDG-Alignment] max_weight={max_weight:.4f}, N_eff={n_eff:.2f}, entropy={entropy:.4f}")
            cprint(f"  [FedSDG-Alignment] diff_to_uniform={diff_to_uniform:.6f}, kl_from_uniform={kl_from_uniform:.6f}")
        
        # 扩展诊断指标：norm_mean 和 norm_delta
        if 'norm_mean' in weight_stats:
            cprint(f"  [FedSDG-Alignment] norm_mean={weight_stats['norm_mean']:.6e}, "
                   f"norm_delta: mean={weight_stats['norm_delta_mean']:.4e}, "
                   f"min={weight_stats['norm_delta_min']:.4e}, max={weight_stats['norm_delta_max']:.4e}")
        
        # 扩展诊断指标：cos_raw（ReLU前的原始cos值）
        if 'cos_raw' in weight_stats:
            cos_raw = weight_stats['cos_raw']
            cprint(f"  [FedSDG-Alignment] cos(delta_k, delta_mean) [before ReLU]: "
                   f"mean={np.mean(cos_raw):.4f}, std={np.std(cos_raw):.4f}, "
                   f"min={min(cos_raw):.4f}, max={max(cos_raw):.4f}")
        
        # 扩展诊断指标：alpha_raw（ReLU后的alpha值）
        if 'alpha_raw' in weight_stats and weight_stats['alpha_raw'] is not None:
            alpha_raw = weight_stats['alpha_raw']
            cprint(f"  [FedSDG-Alignment] alpha [after ReLU]: "
                   f"mean={np.mean(alpha_raw):.4f}, std={np.std(alpha_raw):.4f}, "
                   f"min={min(alpha_raw):.4f}, max={max(alpha_raw):.4f}")
        
        # 警告：fallback 到 uniform
        if weight_stats.get('fallback_uniform', False):
            cprint(f"  [FedSDG-Alignment] ⚠️  FALLBACK to uniform weights triggered!")
        
        # 警告：norm_mean 极小
        norm_mean = weight_stats.get('norm_mean', 1.0)
        if norm_mean < 1e-8:
            cprint(f"  [FedSDG-Alignment] ⚠️  WARNING: norm_mean={norm_mean:.2e} < 1e-8, "
                   f"mean update direction near zero, alignment signal extremely weak!")
        elif norm_mean < 1e-4:
            cprint(f"  [FedSDG-Alignment] ⚠️  NOTICE: norm_mean={norm_mean:.2e} is small, "
                   f"alignment signal may be weak")
        
        # 警告：alpha_std 极小
        if 'alpha_raw' in weight_stats and weight_stats['alpha_raw'] is not None:
            alpha_std = float(np.std(weight_stats['alpha_raw']))
            sum_alpha_raw = weight_stats.get('sum_alpha_raw', 1.0)
            if alpha_std < 1e-3 and sum_alpha_raw > 1e-6:
                cprint(f"  [FedSDG-Alignment] ⚠️  WARNING: alpha_std={alpha_std:.4e} < 1e-3, "
                       f"all clients have similar alignment scores, weights will be nearly uniform")
        
        # 警告：cos 值全部很高
        if 'cos_raw' in weight_stats:
            cos_raw = weight_stats['cos_raw']
            cos_mean_val = float(np.mean(cos_raw))
            cos_std_val = float(np.std(cos_raw))
            if cos_mean_val > 0.95 and cos_std_val < 0.05:
                cprint(f"  [FedSDG-Alignment] ⚠️  NOTICE: cos_mean={cos_mean_val:.4f}, cos_std={cos_std_val:.4f}, "
                       f"all clients updating in highly similar directions, alignment provides little differentiation")
            
            # 警告：存在负 cos 值
            M = len(cos_raw)
            num_negative = sum(1 for c in cos_raw if c < 0)
            if num_negative > 0:
                cprint(f"  [FedSDG-Alignment] ⚠️  {num_negative}/{M} clients have negative cos (divergent updates), "
                       f"these will get zero weight after ReLU")


def log_fedsdg_diagnostics(epoch: int, idx: int, lambda_k_values: List[float]) -> None:
    """
    记录 FedSDG 诊断信息
    
    Args:
        epoch: 当前轮次
        idx: 客户端索引
        lambda_k_values: lambda_k 值列表
    """
    msg1 = f"[FedSDG Diagnostics - 轮次 {epoch+1}, 客户端 {idx}]"
    cprint(f"\n{msg1}")
    
    if lambda_k_values:
        msg2 = (f"  Lambda_k 均值: {np.mean(lambda_k_values):.4f} "
                f"(范围: {min(lambda_k_values):.4f} - {max(lambda_k_values):.4f})")
        cprint(msg2)
    
    msg3 = f"[FedSDG Diagnostics 结束]"
    cprint(f"{msg3}\n")


def log_final(
    total_epochs: int,
    test_acc: float,
    local_acc: float,
    cumulative_comm_mb: float,
    efficiency_score_per_gb: float,
    total_time: float,
    hydra_run_dir: Optional[str]
) -> None:
    """
    记录最终结果
    
    Args:
        total_epochs: 总轮次
        test_acc: 全局测试准确率
        local_acc: 本地个性化准确率
        cumulative_comm_mb: 累计通信量
        efficiency_score_per_gb: 效率得分
        total_time: 总运行时间
        hydra_run_dir: Hydra 运行目录
    """
    cprint_section("训练完成")
    cprint("")
    cprint(f"最终结果 ({total_epochs} 轮后):")
    cprint(f"  全局测试准确率: {100*test_acc:.2f}%")
    cprint(f"  本地个性化准确率: {100*local_acc:.2f}%")
    cprint(f"  准确率差异 (Local - Global): {100*(local_acc - test_acc):+.2f}%")
    cprint("")
    cprint(f"通信统计:")
    cprint(f"  总通信量: {cumulative_comm_mb:.2f} MB ({cumulative_comm_mb/1024:.3f} GB)")
    cprint(f"  效率得分: {efficiency_score_per_gb:.4f} (准确率/GB)")
    cprint("")
    cprint(f"运行时间: {total_time:.1f}s ({total_time/60:.1f} min)")
    cprint("")
    cprint(f"输出文件:")
    cprint(f"  - {hydra_run_dir}/summary.txt")
    cprint(f"  - {hydra_run_dir}/results.pkl")
    cprint(f"  - {hydra_run_dir}/checkpoint_best.pt")


# =============================================================================
# 从 logger.py 迁移的函数
# =============================================================================

def exp_details(args) -> None:
    """
    打印实验详情和配置
    
    注意：此函数保持向后兼容，同时输出到终端和文件。
    """
    cprint('\nExperimental details:')
    cprint(f'    Model     : {args.model}')
    cprint(f'    Algorithm : {args.alg}')
    cprint(f'    Optimizer : {args.optimizer}')
    cprint(f'    Learning  : {args.lr}')
    cprint(f'    Global Rounds   : {args.epochs}\n')

    cprint('    Federated parameters:')
    cprint(f'    Dirichlet Alpha : {args.dirichlet_alpha}')
    cprint(f'    Fraction of users  : {args.frac}')
    cprint(f'    Local Batch size   : {args.local_bs}')
    cprint(f'    Local Epochs       : {args.local_ep}')
    
    # LoRA parameters
    if args.alg in ('fedlora', 'fedsdg', 'feddpa', 'fedsalora', 'local_only'):
        cprint(f'\n    LoRA parameters:')
        cprint(f'    LoRA rank (r)      : {args.lora_r}')
        cprint(f'    LoRA alpha         : {args.lora_alpha}')
        cprint(f'    Train mlp_head     : {bool(args.lora_train_mlp_head)}')
        if args.alg == 'fedsdg':
            cprint(f'\n    FedSDG specific:')
            cprint(f'    Dual-path mode     : Enabled (Global + Private branches)')
            cprint(f'    Private params     : Not communicated (client-local only)')
            agg_method_desc = {
                'fedavg': 'FedAvg uniform-weighted aggregation',
                'alignment': 'Alignment-based weighted FedSDG aggregation'
            }
            cprint(f'    Server Aggregation : {args.server_agg_method} ({agg_method_desc.get(args.server_agg_method, "unknown")})')
        if args.alg == 'local_only':
            cprint(f'\n    Local-Only specific:')
            cprint(f'    Training mode      : Each client trains independently')
            cprint(f'    Aggregation        : None (no server aggregation)')
    cprint()
