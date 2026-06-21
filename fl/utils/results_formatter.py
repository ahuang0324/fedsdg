# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Results Formatter - Generate final_results.json and summary.txt

Provides standardized output formats for paper-ready results.

Usage:
    from fl.utils.results_formatter import ResultsFormatter
    
    formatter = ResultsFormatter(output_dir='outputs/xxx')
    
    # At the end of training
    formatter.save_final_results(
        experiment_info=...,
        hyperparameters=...,
        comm_stats=...,
        final_test_results=...,
        ...
    )
"""

import json
import os
import time
from typing import Dict, Any, Optional, List

import numpy as np

from .console_logger import cprint


class ResultsFormatter:
    """
    结果格式化器
    
    生成:
    1. final_results.json - 完整的JSON格式结果（用于程序读取）
    2. summary.txt - 人类可读的文本摘要
    """
    
    def __init__(self, output_dir: str):
        """
        初始化结果格式化器
        
        Args:
            output_dir: 输出目录路径
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.json_path = os.path.join(output_dir, 'final_results.json')
        self.summary_path = os.path.join(output_dir, 'summary.txt')
    
    def save_final_results(
        self,
        experiment_info: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        comm_stats: Dict[str, Any],
        final_test_results: Dict[str, Any],
        best_validation_results: Dict[str, Any],
        training_trajectory: Dict[str, Any],
        fedsdg_specific: Optional[Dict[str, Any]] = None,
        efficiency_metrics: Optional[Dict[str, Any]] = None,
        last_n_rounds_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        保存最终结果
        
        Args:
            experiment_info: 实验基本信息
            hyperparameters: 超参数
            comm_stats: 通信统计
            final_test_results: 最终测试结果
            best_validation_results: 最佳验证结果
            training_trajectory: 训练轨迹
            fedsdg_specific: FedSDG特有信息（可选）
            efficiency_metrics: 效率指标（可选）
            last_n_rounds_metrics: 最后N轮平均指标（可选，用于论文报告）
        
        Returns:
            完整的结果字典
        """
        # 构建完整结果
        results = {
            'experiment_info': experiment_info,
            'hyperparameters': hyperparameters,
            'communication_stats': comm_stats,
            'final_test_results': final_test_results,
            'best_validation_results': best_validation_results,
            'training_trajectory': training_trajectory,
        }
        
        if fedsdg_specific:
            results['fedsdg_specific'] = fedsdg_specific
        
        if efficiency_metrics:
            results['efficiency_metrics'] = efficiency_metrics
        
        if last_n_rounds_metrics:
            results['last_n_rounds_metrics'] = last_n_rounds_metrics
        
        # 保存 JSON
        self._save_json(results)
        
        # 生成并保存 summary.txt
        summary = self._generate_summary(results)
        self._save_summary(summary)
        
        return results
    
    def _save_json(self, results: Dict[str, Any]) -> None:
        """保存 final_results.json"""
        # 转换 numpy 类型为 Python 原生类型
        results = self._convert_to_serializable(results)
        
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        cprint(f"[ResultsFormatter] Saved: {self.json_path}")
    
    def _save_summary(self, summary: str) -> None:
        """保存 summary.txt"""
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        cprint(f"[ResultsFormatter] Saved: {self.summary_path}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """生成人类可读的 summary.txt"""
        exp = results['experiment_info']
        hyper = results['hyperparameters']
        comm = results['communication_stats']
        test = results['final_test_results']
        best_val = results['best_validation_results']
        
        lines = [
            "=" * 70,
            "实验结果摘要 (Summary)",
            "=" * 70,
            "",
            "【实验信息】",
            f"  实验名称: {exp.get('experiment_name', 'N/A')}",
            f"  算法: {exp.get('algorithm', 'N/A')}",
            f"  数据集: {exp.get('dataset', 'N/A')}",
            f"  Dirichlet Alpha: {exp.get('dirichlet_alpha', 'N/A')}",
            f"  随机种子: {exp.get('seed', 'N/A')}",
            f"  总轮次: {exp.get('total_rounds', 'N/A')}",
            f"  停止原因: {exp.get('stopped_by', 'N/A')}",
            f"  最佳验证轮次: {best_val.get('best_epoch', 'N/A')}",
            f"  总时间: {exp.get('total_time_minutes', 0):.1f} 分钟",
            f"  日期: {exp.get('date', 'N/A')}",
            "",
            "【超参数】",
            f"  客户端数量: {hyper.get('num_users', 'N/A')}",
            f"  参与率: {hyper.get('participation_rate', hyper.get('frac', 'N/A'))}",
            f"  本地轮次: {hyper.get('local_epochs', hyper.get('local_ep', 'N/A'))}",
            f"  本地批次: {hyper.get('local_batch_size', hyper.get('local_bs', 'N/A'))}",
            f"  学习率: {hyper.get('learning_rate', hyper.get('lr', 'N/A'))}",
            f"  优化器: {hyper.get('optimizer', 'N/A')}",
            "",
            "【通信统计】",
            f"  总参数量: {comm.get('total_params', 0):,}",
            f"  每轮通信量: {comm.get('comm_size_per_round_mb', comm.get('comm_size_mb', 0)):.2f} MB",
            f"  总通信量: {comm.get('total_comm_volume_mb', 0):.2f} MB ({comm.get('total_comm_volume_gb', 0):.2f} GB)",
            "",
            "【最终测试结果】(在完整 Test Set 上评估)",
        ]
        
        # 全局模型结果
        if 'global_model' in test:
            gm = test['global_model']
            lines.extend([
                f"  全局模型:",
                f"    准确率: {gm.get('test_acc', 0)*100:.2f}%",
                f"    损失: {gm.get('test_loss', 0):.4f}",
            ])
        
        # 个性化模型结果
        if 'personalized_models' in test:
            pm = test['personalized_models']
            lines.extend([
                f"  个性化模型:",
                f"    准确率: {pm.get('test_acc_avg', 0)*100:.2f}% (±{pm.get('test_acc_std', 0)*100:.2f}%)",
                f"    范围: [{pm.get('test_acc_min', 0)*100:.2f}%, {pm.get('test_acc_max', 0)*100:.2f}%]",
                f"    P10/P50/P90: {pm.get('test_acc_p10', 0)*100:.2f}% / {pm.get('test_acc_p50', 0)*100:.2f}% / {pm.get('test_acc_p90', 0)*100:.2f}%",
                f"    vs 全局模型: {pm.get('gap_vs_global', 0)*100:+.2f}%",
                f"    评估客户端数: {pm.get('num_clients_evaluated', 'N/A')}",
            ])
        
        # 最佳验证结果
        lines.extend([
            "",
            "【最佳验证结果】",
            f"  最佳轮次: {best_val.get('best_epoch', 'N/A')}",
            f"  验证准确率: {best_val.get('val_acc_avg', 0)*100:.2f}% (±{best_val.get('val_acc_std', 0)*100:.2f}%)",
        ])
        
        # FedSDG 特有信息
        if 'fedsdg_specific' in results:
            fedsdg = results['fedsdg_specific']
            if 'final_gate_statistics' in fedsdg:
                gate = fedsdg['final_gate_statistics']
                lines.extend([
                    "",
                    "【FedSDG 门控统计】",
                    f"  门控值均值: {gate.get('mean', 0):.4f} (±{gate.get('std', 0):.4f})",
                    f"  范围: [{gate.get('min', 0):.4f}, {gate.get('max', 0):.4f}]",
                    f"  P10/P50/P90: {gate.get('p10', 0):.4f} / {gate.get('p50', 0):.4f} / {gate.get('p90', 0):.4f}",
                ])
        
        # 最后N轮平均值（论文报告关键指标）
        if 'last_n_rounds_metrics' in results:
            lnr = results['last_n_rounds_metrics']
            # 从键名中提取轮次数（例如 'last_10_rounds/test_acc_avg' -> 10）
            n_rounds = 10  # 默认值
            for key in lnr.keys():
                if key.startswith('last_') and '_rounds/' in key:
                    try:
                        n_rounds = int(key.split('_')[1])
                        break
                    except (IndexError, ValueError):
                        pass
            
            lines.extend([
                "",
                f"【最后{n_rounds}轮平均值】(论文报告指标)",
            ])
            
            # 个性化模型测试指标
            test_acc_key = f'last_{n_rounds}_rounds/test_acc_avg'
            test_acc_std_key = f'last_{n_rounds}_rounds/test_acc_avg_std'
            if test_acc_key in lnr:
                std_str = f" (±{lnr[test_acc_std_key]*100:.2f}%)" if test_acc_std_key in lnr else ""
                lines.append(f"  个性化模型测试准确率: {lnr[test_acc_key]*100:.2f}%{std_str}")
            
            # 全局模型测试指标
            global_test_key = f'last_{n_rounds}_rounds/global_test_acc'
            global_test_std_key = f'last_{n_rounds}_rounds/global_test_acc_std'
            if global_test_key in lnr:
                std_str = f" (±{lnr[global_test_std_key]*100:.2f}%)" if global_test_std_key in lnr else ""
                lines.append(f"  全局模型测试准确率: {lnr[global_test_key]*100:.2f}%{std_str}")
            
            # 验证指标
            val_acc_key = f'last_{n_rounds}_rounds/val_acc_avg'
            val_acc_std_key = f'last_{n_rounds}_rounds/val_acc_avg_std'
            if val_acc_key in lnr:
                std_str = f" (±{lnr[val_acc_std_key]*100:.2f}%)" if val_acc_std_key in lnr else ""
                lines.append(f"  个性化模型验证准确率: {lnr[val_acc_key]*100:.2f}%{std_str}")
            
            global_val_key = f'last_{n_rounds}_rounds/global_val_acc'
            global_val_std_key = f'last_{n_rounds}_rounds/global_val_acc_std'
            if global_val_key in lnr:
                std_str = f" (±{lnr[global_val_std_key]*100:.2f}%)" if global_val_std_key in lnr else ""
                lines.append(f"  全局模型验证准确率: {lnr[global_val_key]*100:.2f}%{std_str}")
        
        # 效率指标
        if 'efficiency_metrics' in results:
            eff = results['efficiency_metrics']
            lines.extend([
                "",
                "【效率指标】",
                f"  准确率/GB: {eff.get('accuracy_per_gb', 0):.4f}",
                f"  最佳效率轮次: {eff.get('best_efficiency_epoch', 'N/A')}",
            ])
        
        lines.extend([
            "",
            "=" * 70,
            f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """将 numpy 类型转换为 Python 原生类型（用于 JSON 序列化）"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def get_json_path(self) -> str:
        """获取 JSON 文件路径"""
        return self.json_path
    
    def get_summary_path(self) -> str:
        """获取 summary.txt 文件路径"""
        return self.summary_path


def build_experiment_info(
    experiment_name: str,
    args,
    total_rounds: int,
    stopped_by: str,
    early_stop_epoch: Optional[int],
    best_validation_epoch: Optional[int],
    total_time_seconds: float,
) -> Dict[str, Any]:
    """
    构建实验信息字典
    
    Args:
        experiment_name: 实验名称
        args: 配置对象
        total_rounds: 实际训练轮次
        stopped_by: 停止原因 ('max_epochs' | 'early_stopping')
        early_stop_epoch: 早停轮次（如果触发）
        best_validation_epoch: 最佳验证轮次
        total_time_seconds: 总训练时间（秒）
    
    Returns:
        实验信息字典
    """
    return {
        'experiment_name': experiment_name,
        'algorithm': getattr(args, 'alg', 'unknown'),
        'dataset': getattr(args, 'dataset', 'unknown'),
        'dirichlet_alpha': getattr(args, 'dirichlet_alpha', None),
        'seed': getattr(args, 'seed', None),
        'total_rounds': total_rounds,
        'stopped_by': stopped_by,
        'early_stop_epoch': early_stop_epoch,
        'best_validation_epoch': best_validation_epoch,
        'total_time_minutes': total_time_seconds / 60,
        'date': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }


def build_hyperparameters(args) -> Dict[str, Any]:
    """
    构建超参数字典
    
    Args:
        args: 配置对象
    
    Returns:
        超参数字典
    """
    hyper = {
        'num_users': getattr(args, 'num_users', None),
        'participation_rate': getattr(args, 'frac', None),
        'local_epochs': getattr(args, 'local_ep', None),
        'local_batch_size': getattr(args, 'local_bs', None),
        'learning_rate': getattr(args, 'lr', None),
        'optimizer': getattr(args, 'optimizer', None),
    }
    
    # LoRA 参数（FedLoRA, FedSDG, Local-Only, FedRep 都使用 LoRA）
    if getattr(args, 'alg', '') in ('fedlora', 'fedsdg', 'local_only', 'fedrep'):
        hyper.update({
            'lora_r': getattr(args, 'lora_r', None),
            'lora_alpha': getattr(args, 'lora_alpha', None),
        })
    
    # FedRep 参数
    if getattr(args, 'alg', '') == 'fedrep':
        hyper.update({
            'fedrep_rep_epochs': getattr(args, 'fedrep_rep_epochs', None),
            'fedrep_head_epochs': getattr(args, 'fedrep_head_epochs', None),
            'lr_head': getattr(args, 'lr_head', None),
        })
    
    # FedSDG 参数
    if getattr(args, 'alg', '') == 'fedsdg':
        hyper.update({
            'lambda1': getattr(args, 'lambda1', None),
            'lambda2': getattr(args, 'lambda2', None),
            'lr_gate': getattr(args, 'lr_gate', None),
            'gate_penalty_type': getattr(args, 'gate_penalty_type', None),
        })
    
    return hyper


def build_comm_stats(
    comm_stats: Dict[str, Any],
    total_rounds: int,
    cumulative_comm_mb: float,
) -> Dict[str, Any]:
    """
    构建通信统计字典
    
    Args:
        comm_stats: 原始通信统计
        total_rounds: 总轮次
        cumulative_comm_mb: 累计通信量 (MB)
    
    Returns:
        通信统计字典
    """
    return {
        'total_params': comm_stats.get('total_params', 0),
        'trainable_params': comm_stats.get('trainable_params', 0),
        'comm_params_per_round': comm_stats.get('comm_params', 0),
        'comm_size_per_round_mb': comm_stats.get('comm_size_mb', 0),
        'total_rounds': total_rounds,
        'total_comm_volume_mb': cumulative_comm_mb,
        'total_comm_volume_gb': cumulative_comm_mb / 1024,
        'compression_ratio': comm_stats.get('compression_ratio', 100),
    }


def build_final_test_results(
    global_acc: float,
    global_loss: float,
    local_acc_avg: float,
    local_acc_std: float,
    local_loss_avg: float,
    local_loss_std: float,
    client_results: List[Dict[str, Any]],
    num_test_samples: int,
) -> Dict[str, Any]:
    """
    构建最终测试结果字典
    
    Args:
        global_acc: 全局模型准确率
        global_loss: 全局模型损失
        local_acc_avg: 个性化模型平均准确率
        local_acc_std: 个性化模型准确率标准差
        local_loss_avg: 个性化模型平均损失
        local_loss_std: 个性化模型损失标准差
        client_results: 每个客户端的详细结果
        num_test_samples: 测试集样本数
    
    Returns:
        最终测试结果字典
    """
    # 计算统计量
    client_accs = [c.get('test_acc', c.get('acc', 0)) for c in client_results]
    
    return {
        'description': 'Evaluated on COMPLETE test set ONCE at the end',
        'global_model': {
            'test_acc': float(global_acc),
            'test_loss': float(global_loss),
            'num_samples': num_test_samples,
        },
        'personalized_models': {
            'test_acc_avg': float(local_acc_avg),
            'test_acc_std': float(local_acc_std),
            'test_acc_min': float(np.min(client_accs)) if client_accs else 0,
            'test_acc_max': float(np.max(client_accs)) if client_accs else 0,
            'test_acc_p10': float(np.percentile(client_accs, 10)) if client_accs else 0,
            'test_acc_p50': float(np.percentile(client_accs, 50)) if client_accs else 0,
            'test_acc_p90': float(np.percentile(client_accs, 90)) if client_accs else 0,
            'test_loss_avg': float(local_loss_avg),
            'test_loss_std': float(local_loss_std),
            'gap_vs_global': float(local_acc_avg - global_acc),
            'num_clients_evaluated': len(client_results),
        },
        'per_client_results': client_results,
    }


def calculate_last_n_rounds_average(
    metrics_csv_path: str,
    n_rounds: int = 10,
    metric_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    计算最后 N 轮的平均值（用于论文报告）
    
    如果总轮次不足 N 轮，则计算所有轮次的平均值。
    
    Args:
        metrics_csv_path: metrics.csv 文件路径
        n_rounds: 要计算平均值的轮次数（默认10）
        metric_names: 要计算的指标名称列表，如果为 None 则使用默认列表
    
    Returns:
        字典，键为 'last_{n}_rounds/{metric_name}'，值为平均值
        
    Example:
        {
            'last_10_rounds/test_acc_avg': 0.7523,
            'last_10_rounds/global_test_acc': 0.7412,
            'last_10_rounds/local_test_acc_avg': 0.7689,
        }
    """
    import pandas as pd
    
    if metric_names is None:
        metric_names = [
            'test_acc_avg',
            'global_test_acc',
            'test_loss_avg',
            'global_test_loss',
            'val_acc_avg',
            'global_val_acc',
            'val_loss_avg',
            'global_val_loss',
        ]
    
    try:
        df = pd.read_csv(metrics_csv_path)
    except FileNotFoundError:
        cprint(f"[警告] 未找到 metrics.csv: {metrics_csv_path}")
        return {}
    except Exception as e:
        cprint(f"[错误] 读取 metrics.csv 失败: {e}")
        return {}
    
    if len(df) == 0:
        cprint("[警告] metrics.csv 为空")
        return {}
    
    # 确定实际计算的轮次数（如果总轮次不足 N 轮，则用所有轮次）
    actual_n_rounds = min(n_rounds, len(df))
    last_n_df = df.tail(actual_n_rounds)
    
    results = {}
    for metric_name in metric_names:
        if metric_name in last_n_df.columns:
            # 过滤掉缺失值（NaN）
            valid_values = last_n_df[metric_name].dropna()
            if len(valid_values) > 0:
                avg_value = float(valid_values.mean())
                std_value = float(valid_values.std())
                results[f'last_{actual_n_rounds}_rounds/{metric_name}'] = avg_value
                results[f'last_{actual_n_rounds}_rounds/{metric_name}_std'] = std_value
    
    if results:
        cprint(f"[ResultsFormatter] 计算最后 {actual_n_rounds} 轮平均值: {len(results)} 个指标")
    
    return results
