# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Structured Configs - 使用 Dataclasses 定义配置结构

config.yaml 是填好的“表格”，那么这个文件就是“表格的格式定义”——它规定了有哪些填空题，每个空应该填数字还是文字，以及默认填什么。

相当于argeparse 中创建一个对象，然后不断的add

提供类型检查和 IDE 补全支持。
所有配置类都注册到 Hydra ConfigStore。

配置层级:
    Config (根配置)
    ├── algorithm: AlgorithmConfig      # 算法配置 (fedavg/fedlora/fedsdg)
    ├── dataset: DatasetConfig          # 数据集配置
    ├── model: ModelConfig              # 模型配置
    ├── training: TrainingConfig        # 训练配置
    ├── federated: FederatedConfig      # 联邦学习配置
    ├── system: SystemConfig            # 系统配置
    ├── checkpoint: CheckpointConfig    # 检查点配置
    └── logging: LoggingConfig          # 日志配置 (WandB 预留)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
from omegaconf import MISSING


# =============================================================================
# Algorithm Configs
# =============================================================================

# @dataclass 是 Python 3.7+ 的装饰器，用于自动生成常用方法，简化数据类的定义。
@dataclass
class AlgorithmConfig:
    """算法配置基类"""
    name: str = "fedavg"  # fedavg | fedlora | fedsdg | fedprox_avg | fedprox_lora
    
    # FedProx 参数
    mu: float = 0.0  # proximal term 系数 (仅 fedprox_* 使用)
    
    # LoRA 参数 (FedLoRA/FedSDG 共享)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_train_mlp_head: bool = True
    
    server_agg_method: str = "fedavg"  # fedavg | alignment
    alignment_strategy: str = "loo_mean"  # loo_mean | standard_mean
    weight_transform: str = "relu_normalize"  # relu_normalize | softmax
    softmax_temperature: float = 1.0   # softmax 温度
    lambda_smooth: float = 0.0         # 权重平滑系数
    lambda1: float = 0.01              # 门控稀疏惩罚 (L1)
    lambda2: float = 0.001             # 私有参数正则化 (L2)
    gate_penalty_type: str = "unilateral"  # unilateral | bilateral
    lr_gate: float = 0.01              # 门控参数学习率
    grad_clip: float = 1.0             # 梯度裁剪 (0 禁用)
    
    # 固定门控配置（用于对比实验）
    fix_gate: bool = False              # 是否固定门控系数（不学习）
    fixed_gate_value: float = 0.5      # 固定门控值（仅在 fix_gate=true 时生效，范围 [0, 1]）
    
    # 可学习门控初始化配置
    gate_init_value: float = 0.0        # 门控参数初始值（logit形式，仅在 fix_gate=false 时生效）
                                       # 默认 0.0 → m_k = sigmoid(0.0) = 0.5
                                       # 可设置为其他值，如 -1.0 → m_k ≈ 0.27, 1.0 → m_k ≈ 0.73
    
    # 优化器配置 (可选，用于算法级别覆盖)
    optimizer: Optional[str] = None    # None | sgd | adam (None 表示使用 training.optimizer)


@dataclass
class FedAvgConfig(AlgorithmConfig):
    """FedAvg 算法配置"""
    name: str = "fedavg"
    lr_head: Optional[float] = None  # Head 学习率（None 表示使用主学习率 lr）


@dataclass
class FedLoRAConfig(AlgorithmConfig):
    """FedLoRA 算法配置"""
    name: str = "fedlora"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_train_mlp_head: bool = True
    optimizer: str = "adam"  # FedLoRA 推荐使用 Adam
    lr_head: Optional[float] = None  # Head 学习率（None 表示使用主学习率 lr）


@dataclass
class FedSDGConfig(AlgorithmConfig):
    """FedSDG 算法配置"""
    name: str = "fedsdg"
    
    # LoRA 参数
    lora_r: int = 8
    lora_alpha: int = 16
    lora_r_private: Optional[int] = None  # 私有分支独立秩（None 则等于 lora_r）
    lora_train_mlp_head: bool = True
    
    # 门控粒度配置（）
    gate_granularity: str = 'fine'  # fine | coarse
    
    # 聚合参数
    server_agg_method: str = "fedavg"
    alignment_strategy: str = "loo_mean"
    weight_transform: str = "relu_normalize"
    softmax_temperature: float = 1.0
    lambda_smooth: float = 0.0
    
    # 正则化配置
    lambda1: float = 0.01
    lambda2: float = 0.001
    gate_penalty_type: str = "unilateral"
    lr_gate: float = 0.01
    grad_clip: float = 1.0
    
    # 固定门控配置
    fix_gate: bool = False
    fixed_gate_value: float = 0.5
    gate_init_value: float = 0.0
    
    # 门控 Warmup 配置
    gate_warmup_rounds: int = 0          # 前 N 轮不施加门控惩罚(λ₁=0)，让门控自由学习
    
    # Dynamic Alignment 配置
    use_dynamic_alignment: bool = False  # 启用后 m_k 语义变为"相对于全局分支的私有信号比例"
    da_floor_gamma: float = 0.1          # Relative Floor 系数：target_rms ≥ γ × RMS(base_output)
    da_target_mode: str = 'floor'        # target_rms 计算模式：floor | base | geomean
    use_da_scale: bool = False           # 启用全局可学习放大系数 α = exp(s)，自动适应数据集异构特性
    lr_da_scale: float = 0.01            # DA Scale 学习率（独立参数组，全局参数需较小 lr）
    da_detach_private_rms: bool = True   # 是否 detach p_scaled 用于计算 p_rms（消融实验用）
    
    # Head 参数模式配置
    head_mode: str = "global"           # global: Head 参与聚合 | private: Head 不参与聚合（类似 FedRep）
    lr_head: Optional[float] = None     # Head 学习率（None 表示使用主学习率 lr）
    optimizer: str = "adam"  # FedSDG 必须使用 Adam
    
    def __post_init__(self):
        # 基础验证
        if self.gate_granularity not in ['fine', 'coarse']:
            raise ValueError(f"gate_granularity must be 'fine' or 'coarse', got {self.gate_granularity}")
        
        # 高级验证
        self._validate_compatibility()
    
    def _validate_compatibility(self):
        """验证配置兼容性"""
        if self.gate_granularity == 'coarse':
            if hasattr(self, 'fix_gate') and self.fix_gate:
                if not hasattr(self, 'fixed_gate_value') or not (0.0 <= self.fixed_gate_value <= 1.0):
                    raise ValueError("fixed_gate_value must be in [0.0, 1.0] when fix_gate=True")


@dataclass
class FedProxAvgConfig(AlgorithmConfig):
    """FedProx (基于 FedAvg) 算法配置"""
    name: str = "fedprox_avg"
    mu: float = 0.01  # proximal term 系数
    optimizer: Optional[str] = None  # None 表示使用 training.optimizer


@dataclass
class FedProxLoRAConfig(AlgorithmConfig):
    """FedProx + LoRA 算法配置"""
    name: str = "fedprox_lora"
    mu: float = 0.01  # proximal term 系数
    lora_r: int = 8
    lora_alpha: int = 16
    lora_train_mlp_head: bool = True
    lr_head: Optional[float] = None  # Head 学习率，None 表示使用 training.lr
    optimizer: Optional[str] = None  # None 表示使用 training.optimizer


@dataclass
class FedDPAConfig(AlgorithmConfig):
    """
    FedDPA (Federated Dual-Path Adapter) 算法配置
    
    FedDPA 是 Baseline 方法，用于与 FedSDG 进行对比:
    - 双路 LoRA 架构: Global 分支 (聚合) + Private 分支 (本地持久化)
    - 训练时固定混合比例 λ
    - 推理时 Instance-wise Dynamic Weighting
    
    前向传播公式:
      训练: output = Wx + scaling * [(1-λ) * global + λ * private]
      推理: output = Wx + scaling * [(1-α_t) * global + α_t * private]
    """
    name: str = "feddpa"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_train_mlp_head: bool = True
    
    # FedDPA 特有参数
    train_mix_ratio: float = 0.5       # 训练时混合比例 λ（固定值）
    inference_scale_factor: float = 0.5  # 推理时缩放因子
    anchor_count: int = 5              # Anchor 样本数量
    
    # 服务端聚合配置（FedDPA 支持 fedavg 和 alignment）
    server_agg_method: str = "fedavg"
    alignment_strategy: str = "loo_mean"
    weight_transform: str = "softmax"
    softmax_temperature: float = 0.05
    lambda_smooth: float = 0.0
    
    # 优化器配置
    optimizer: str = "adam"
    grad_clip: float = 1.0
    
    # Head 参数模式配置
    head_mode: str = "global"           # global: Head 参与聚合 | private: Head 不参与聚合（个性化分类头）
    
    # 学习率配置（支持不同参数组使用不同学习率）
    lr_head: Optional[float] = None     # Head 学习率（None 表示使用主学习率 lr）
    lr_private: Optional[float] = None  # Private 参数学习率（None 表示使用主学习率 lr）
    
    # Dynamic Alignment 配置（实验性，与 FedSDG DA 一致，不含 DA-Scale）
    use_dynamic_alignment: bool = False
    da_floor_gamma: float = 0.1
    da_target_mode: str = "global"
    da_detach_private_rms: bool = True


@dataclass
class LocalOnlyConfig(AlgorithmConfig):
    """
    Local-Only (LoRA) 算法配置
    
    每个客户端独立训练，只使用私有 LoRA，不进行聚合。
    相当于 FedSDG 禁用 global_LoRA，每个客户端维护自己的私有 LoRA。
    """
    name: str = "local_only"
    
    # LoRA 参数
    lora_r: int = 8
    lora_alpha: int = 16
    lora_train_mlp_head: bool = True
    
    # Head 学习率
    lr_head: Optional[float] = None  # Head 学习率（None 表示使用主学习率 lr）
    
    # 优化器配置
    optimizer: str = "adam"


@dataclass
class FedRepConfig(AlgorithmConfig):
    """
    FedRep (联邦表示学习) 算法配置
    
    将模型分为两部分:
    - Backbone (表示层): 参与联邦聚合，学习共享表示
    - Head (分类头): 本地维护，学习个性化分类器
    
    训练流程（每轮本地训练）:
      Step 1: 冻结 Head，训练 Backbone (fedrep_rep_epochs 个 epoch)
      Step 2: 冻结 Backbone，训练 Head (fedrep_head_epochs 个 epoch)
    """
    name: str = "fedrep"
    
    # FedRep 特有参数
    fedrep_rep_epochs: int = 1      # Step 1: 训练 Backbone 的 epoch 数
    fedrep_head_epochs: int = 4     # Step 2: 训练 Head 的 epoch 数
    
    # LoRA 参数
    lora_r: int = 8
    lora_alpha: int = 16
    lora_train_mlp_head: bool = True
    
    # Head 学习率
    lr_head: Optional[float] = 0.001
    
    # 服务端聚合配置
    server_agg_method: str = "fedavg"
    
    # 优化器配置
    optimizer: str = "adam"


@dataclass
class DittoConfig(AlgorithmConfig):
    """
    Ditto 算法配置
    
    核心思想：
    - 每个客户端维护两个模型：全局模型 w 和个性化模型 v
    - 个性化模型通过正则化向全局模型靠拢
    
    损失函数：
      L(v) = L_task(v) + (λ/2) ||v - w||²
    
    训练流程（每轮本地训练）:
      Step 1: 在全局模型上训练（用于聚合）
      Step 2: 在个性化模型上训练（带正则化）
    """
    name: str = "ditto"
    
    # Ditto 特有参数
    lambda_ditto: float = 0.1           # 正则化系数 λ
    ditto_reg_target: str = "server"    # 正则化参考点 'server' | 'local'
    
    # LoRA 参数
    lora_r: int = 8
    lora_alpha: int = 16
    lora_train_mlp_head: bool = True
    
    # 服务端聚合配置（聚合全局模型，与 FedLoRA 相同）
    server_agg_method: str = "fedavg"
    alignment_strategy: str = "loo_mean"
    weight_transform: str = "softmax"
    softmax_temperature: float = 0.3
    lambda_smooth: float = 0.0
    
    # 优化器配置
    optimizer: str = "adam"  # Ditto 推荐使用 Adam
    
    # Head 学习率配置
    lr_head: Optional[float] = None  # Head 学习率（None 表示使用主学习率 lr）


@dataclass
class FedSALoRAConfig(AlgorithmConfig):
    """
    FedSA-LoRA (Federated Share-A Low-Rank Adaptation) 算法配置
    
    核心思想：
    - 使用标准单路 LoRA（A 和 B 矩阵都参与训练）
    - 聚合时仅上传 A 矩阵（学习通用知识）到服务器进行 FedAvg
    - B 矩阵（捕获客户端特有知识）保留在本地，不参与聚合
    
    参考论文:
      Guo et al., "Selective Aggregation for Low-Rank Adaptation
      in Federated Learning", ICLR 2025
    """
    name: str = "fedsalora"
    
    # LoRA 参数
    lora_r: int = 8
    lora_alpha: int = 16
    lora_train_mlp_head: bool = True
    
    # 服务端聚合配置
    server_agg_method: str = "fedavg"
    alignment_strategy: str = "loo_mean"
    weight_transform: str = "softmax"
    softmax_temperature: float = 0.3
    lambda_smooth: float = 0.0
    
    # 优化器配置
    optimizer: str = "adam"
    
    # Head 学习率配置
    lr_head: Optional[float] = None


@dataclass
class PF2LoRAConfig(AlgorithmConfig):
    """
    PF2LoRA (Personalized Federated Fine-tuning via Two-Level LoRA) 算法配置
    
    核心思想：
    - 双路 LoRA 架构 (Shared + Private)，纯加性组合（无门控）
    - Shared LoRA (A_s, B_s): 参与服务器 FedAvg 聚合，学习通用知识
    - Private LoRA (A_p, B_p): 本地保留，捕获客户端特有知识
    - 自动秩学习: 基于重要性分数对 Private LoRA 进行秩剪枝
    
    参考论文:
      Hao et al., "Personalized Federated Fine-tuning for Heterogeneous Data:
      An Automatic Rank Learning Approach via Two-Level LoRA", ICLR 2026
    """
    name: str = "pf2lora"
    
    # LoRA 参数（双路架构）
    lora_r: int = 8                # Shared LoRA 秩（与其他 baseline 一致）
    lora_alpha: int = 16           # 缩放因子 (2*lora_r)
    lora_r_private: Optional[int] = 16  # Private LoRA 秩（r_max，剪枝前的初始秩）
    lora_train_mlp_head: bool = True
    
    # PF2LoRA 特有参数 —— 自动秩学习
    enable_rank_pruning: bool = True
    target_rank_ratio: float = 0.5  # target_rank = r_max * ratio
    pruning_start_round: int = 10
    pruning_interval: int = 5
    
    # 服务端聚合配置
    server_agg_method: str = "fedavg"
    
    # FedSDG 兼容参数（PF2LoRA 中 gate 固定为 1.0）
    lambda1: float = 0.0
    lambda2: float = 0.0
    gate_penalty_type: str = "unilateral"
    lr_gate: float = 0.0
    grad_clip: float = 0.0
    fix_gate: bool = True
    fixed_gate_value: float = 1.0
    gate_init_value: float = 100.0
    gate_granularity: str = "layer"
    
    # Dynamic Alignment（PF2LoRA 不使用）
    use_dynamic_alignment: bool = False
    da_floor_gamma: float = 0.1
    da_target_mode: str = "floor"
    use_da_scale: bool = False
    
    # Head 参数模式
    head_mode: str = "global"
    lr_head: Optional[float] = None
    
    # 优化器
    optimizer: str = "adam"


@dataclass
class LoRAFAIRConfig(AlgorithmConfig):
    """
    LoRA-FAIR (Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement) 算法配置
    
    核心思想：
    - 标准单路 LoRA 训练（与 FedLoRA 完全相同的客户端训练逻辑）
    - 服务端聚合后引入残差修正项 ΔB，修正聚合偏差
    - 客户端接收修正后的 B'=B̄+ΔB 和 Ā 作为下一轮初始化
    
    参考论文:
      Bian et al., "LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation
      and Initialization Refinement", ICCV 2025
    """
    name: str = "lorafair"
    
    # LoRA 参数
    lora_r: int = 8
    lora_alpha: int = 16
    lora_train_mlp_head: bool = True
    
    # LoRA-FAIR 特有参数
    residual_mu: float = 0.1          # 残差修正正则化系数 μ
    
    # 服务端聚合配置
    server_agg_method: str = "fedavg"
    alignment_strategy: str = "loo_mean"
    weight_transform: str = "softmax"
    softmax_temperature: float = 0.3
    lambda_smooth: float = 0.0
    
    # 优化器配置
    optimizer: str = "adam"
    
    # Head 学习率配置
    lr_head: Optional[float] = None


@dataclass
class FedALTConfig(AlgorithmConfig):
    """
    FedALT-adapted (Federated Alternating Local Training) 算法配置
    
    核心思想（适配版本）：
    - 复用 FedSDG 双路 LoRA 架构，但反转训练和聚合角色
    - Public 分支 (lora_A/B): Global LoRA，训练时冻结，接收服务端聚合结果
    - Private 分支 (lora_A/B_private): Individual LoRA，本地训练 + 上传聚合
    - Gate 作为 Mixer，动态混合 Global 和 Individual 分支
    - Individual LoRA 跨轮次持续演化，不被聚合重置
    
    与完整 FedALT 的简化：
    - RoW LoRA 简化为全局平均（所有客户端收到相同模型）
    - MoE Mixer 简化为 FedSDG 的 sigmoid gate
    
    参考论文:
      Chen et al., "FedALT: Federated Fine-Tuning through Adaptive Local
      Training with Rest-of-World LoRA", 2025
    """
    name: str = "fedalt"
    
    # LoRA 参数（双路架构）
    lora_r: int = 8
    lora_alpha: int = 16
    lora_train_mlp_head: bool = True
    
    # 门控/Mixer 配置
    gate_granularity: str = "fine"
    gate_init_value: float = 0.0
    lr_gate: float = 0.1
    fix_gate: bool = False
    fixed_gate_value: float = 0.5
    
    # 正则化配置
    lambda1: float = 0.001          # 门控稀疏惩罚
    lambda2: float = 0.0005         # Individual LoRA L2 正则化
    gate_penalty_type: str = "unilateral"
    grad_clip: float = 1.0
    
    # 服务端聚合配置
    server_agg_method: str = "fedavg"
    alignment_strategy: str = "loo_mean"
    weight_transform: str = "softmax"
    softmax_temperature: float = 0.3
    lambda_smooth: float = 0.0
    
    # Head 参数模式
    head_mode: str = "global"
    lr_head: Optional[float] = None
    
    # 优化器
    optimizer: str = "adam"


@dataclass
class FedTPConfig(AlgorithmConfig):
    """
    FedTP (Federated Two-Phase LoRA) 算法配置
    
    核心思想：
    - 彻底去除门控机制，将训练解耦为两个连续阶段
    - Phase 1: 仅激活共享 LoRA 分支，标准 FedAvg 联邦聚合捕获通用先验
    - Phase 2: 冻结共享 LoRA，仅训练私有 LoRA 分支，零通信
    - 推理时: 共享分支 + 私有分支等比例直接相加（无门控权重）
    
    对标价值：直接测试 FedSDG 动态对齐与门控机制是否具有不可替代的性能优势
    """
    name: str = "fedtp"
    
    # LoRA 参数（双路架构）
    lora_r: int = 8
    lora_alpha: int = 16
    lora_train_mlp_head: bool = True
    
    # Two-Phase 核心参数
    phase1_epochs: int = 50            # Phase 1 全局训练轮次
    
    # 服务端聚合配置（Phase 1 使用，Phase 2 不聚合）
    server_agg_method: str = "fedavg"
    alignment_strategy: str = "loo_mean"
    weight_transform: str = "softmax"
    softmax_temperature: float = 0.3
    lambda_smooth: float = 0.0
    
    # FedSDG 兼容参数（FedTP 中不使用）
    lambda1: float = 0.0
    lambda2: float = 0.0
    gate_penalty_type: str = "unilateral"
    lr_gate: float = 0.0
    grad_clip: float = 1.0
    fix_gate: bool = True
    fixed_gate_value: float = 0.0
    gate_init_value: float = -100.0
    gate_granularity: str = "fine"
    gate_warmup_rounds: int = 0
    
    # Head 参数模式（与 FedSDG 保持一致）
    head_mode: str = "global"
    lr_head: Optional[float] = None
    
    # Dynamic Alignment（FedTP 不使用）
    use_dynamic_alignment: bool = False
    da_floor_gamma: float = 0.1
    da_target_mode: str = "floor"
    use_da_scale: bool = False
    lr_da_scale: float = 0.01
    da_detach_private_rms: bool = True
    
    # 优化器
    optimizer: str = "adam"


# =============================================================================
# Dataset Config
# =============================================================================

@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str = "cifar100"             # mnist | fmnist | cifar | cifar100
    num_classes: int = 100
    num_channels: int = 3
    image_size: int = 224
    
    # 离线数据配置
    use_offline: bool = False
    offline_data_root: str = "./datasets/preprocessed"


# =============================================================================
# Model Config
# =============================================================================

@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "vit"                  # mlp | cnn | vit
    variant: str = "pretrained"        # scratch | pretrained
    vit_type: str = "tiny"             # tiny | base (vit_tiny_patch16_224 或 vit_base_patch16_224)
    
    # CNN/MLP 参数
    num_filters: int = 32
    kernel_num: int = 9
    kernel_sizes: str = "3,4,5"
    norm: str = "batch_norm"
    max_pool: bool = True


# =============================================================================
# Training Config
# =============================================================================

@dataclass
class ConsoleLogConfig:
    """控制台日志配置"""
    to_console: bool = True            # 是否输出到终端
    to_file: bool = True               # 是否写入 console.log


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100                  # 全局通信轮次
    local_ep: int = 5                  # 本地训练轮次
    local_bs: int = 128                # 本地批次大小
    lr: float = 0.001                  # 学习率
    optimizer: str = "sgd"             # sgd | adam
    momentum: float = 0.5              # SGD 动量
    num_workers: int = 2               # DataLoader 子进程数，联邦学习场景不宜过高
    prefetch_factor: int = 2           # 每 worker 预取 batch 数（仅 num_workers>0 时生效）
    console_log: ConsoleLogConfig = field(default_factory=ConsoleLogConfig)


# =============================================================================
# Federated Learning Config
# =============================================================================

@dataclass
class FederatedConfig:
    """联邦学习配置"""
    num_users: int = 100               # 客户端总数
    frac: float = 0.1                  # 每轮参与率
    dirichlet_alpha: float = 0.1       # Non-IID 程度 (越小越异构)
    test_frac: float = 0.3             # 测试时采样客户端比例
    unequal: bool = False              # 不均等数据划分


# =============================================================================
# System Config
# =============================================================================

@dataclass
class SystemConfig:
    """系统配置"""
    gpu: int = 0                       # GPU ID (-1 表示 CPU)
    seed: int = 42                     # 随机种子
    verbose: int = 1                   # 日志详细程度
    stopping_rounds: int = 10          # 早停轮次


# =============================================================================
# Checkpoint Config
# =============================================================================

@dataclass
class CheckpointConfig:
    """检查点配置"""
    enable: bool = False               # 是否启用检查点
    save_frequency: int = 10           # 保存频率
    save_client_weights: bool = True   # 保存客户端权重
    max_checkpoints: int = -1          # 最大保存数量 (-1 无限制)


# =============================================================================
# Logging Config (WandB 预留)
# =============================================================================

@dataclass
class LoggingConfig:
    """日志配置 (支持 TensorBoard / WandB / None)"""
    backend: str = "tensorboard"       # tensorboard | wandb | none
    
    # WandB 配置
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_job_type: str = "train"
    wandb_tags: Optional[List[str]] = None
    wandb_notes: Optional[str] = None
    wandb_mode: str = "online"         # online | offline | disabled
    wandb_resume: Optional[str] = None # allow | must | never | auto
    wandb_run_id: Optional[str] = None # 用于 resume
    
    # 日志选项（预留字段，当前未被 LoggerFactory 使用）
    log_every_n_rounds: int = 1
    save_config: bool = True


# =============================================================================
# Root Config
# =============================================================================

@dataclass
class Config:
    """
    根配置类
    
    Hydra 会从 conf/config.yaml 加载并合并所有配置组。
    使用 ConfigAdapter 可将此配置转换为与原 args 兼容的对象。
    """
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# =============================================================================
# Register Structured Configs with Hydra ConfigStore
# =============================================================================

def register_configs() -> None:
    """
    将所有 Structured Configs 注册到 Hydra ConfigStore
    
    在 main.py 中调用，确保 Hydra 能够使用类型检查。
    """
    from hydra.core.config_store import ConfigStore
    # ConfigStore 是 Hydra 的配置注册中心

    cs = ConfigStore.instance()
    # instance() 获取单例实例
    # 注册根配置
    cs.store(name="config_schema", node=Config)
    
    # 注册算法配置组 name 是配置的标识符 node 是要注册的配置类
    cs.store(group="algorithm", name="fedavg", node=FedAvgConfig)
    cs.store(group="algorithm", name="fedlora", node=FedLoRAConfig)
    cs.store(group="algorithm", name="fedsdg", node=FedSDGConfig)
    cs.store(group="algorithm", name="feddpa", node=FedDPAConfig)
    cs.store(group="algorithm", name="ditto", node=DittoConfig)
    cs.store(group="algorithm", name="fedprox_avg", node=FedProxAvgConfig)
    cs.store(group="algorithm", name="fedprox_lora", node=FedProxLoRAConfig)
    cs.store(group="algorithm", name="local_only", node=LocalOnlyConfig)
    cs.store(group="algorithm", name="fedrep", node=FedRepConfig)
    cs.store(group="algorithm", name="fedsalora", node=FedSALoRAConfig)
    cs.store(group="algorithm", name="pf2lora", node=PF2LoRAConfig)
    cs.store(group="algorithm", name="fedtp", node=FedTPConfig)
    cs.store(group="algorithm", name="lorafair", node=LoRAFAIRConfig)
    cs.store(group="algorithm", name="fedalt", node=FedALTConfig)
    
    # 注册其他配置（作为默认 schema）
    cs.store(group="dataset", name="base_dataset", node=DatasetConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="system", name="base_system", node=SystemConfig)
    cs.store(group="checkpoint", name="base_checkpoint", node=CheckpointConfig)
    cs.store(group="logging", name="base_logging", node=LoggingConfig)


