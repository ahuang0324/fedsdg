# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA (Low-Rank Adaptation) implementation for federated learning.

Supports:
- Standard LoRA for FedLoRA
- Dual-path LoRA for FedSDG (global + private branches)

LoRALayer 类：核心数学原理与实现。
inject_lora 函数：针对手写 ViT 的模型改造。
inject_lora_timm 函数：针对 timm 库 ViT 的模型改造。
get_lora_state_dict 函数：联邦学习专用的参数提取。
- 分别返回共有参数和私有参数
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from .vit import ViT
from fl.utils.console_logger import cprint


class GlobalGateModule(nn.Module):
    """
    全局门控参数管理模块 - 仅用于参数存储
    
    设计原则：
    - 不参与前向传播，避免不必要的计算开销
    - 专门用于PyTorch参数管理，符合模块规范
    - 线程安全的参数访问
    """
    def __init__(self, gate_init_value=0.0):
        super().__init__()
        self.lambda_k_global = nn.Parameter(torch.tensor([gate_init_value]))
    
    # 完全不定义forward方法，明确标识不参与计算


class GlobalDAScaleModule(nn.Module):
    """
    全局 DA-Scale 参数管理模块
    
    持有唯一的可学习放大系数 α = exp(da_scale_logit)。
    所有 LoRA 层通过引用访问此参数，确保全局共享。
    参与全局聚合，收敛到数据集级别最优值。
    """
    def __init__(self, init_value=0.0):
        super().__init__()
        # exp(0.0) = 1.0，初始 α=1.0（向后兼容）
        self.da_scale_logit = nn.Parameter(torch.tensor([init_value]))
    
    # 不定义 forward 方法，此模块仅用于参数管理


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层实现
    支持两种模式：
    1. 标准 LoRA (is_fedsdg=False): h = Wx + (alpha/r) * B * A * x
    2. FedSDG 双路架构 (is_fedsdg=True): 
       h = Wx + scaling * [(Global_Path)  + (Private_Path) * lambda_k]
       
    其中:
    - W: 原始冻结权重 (in_features × out_features)
    - A/B: 低秩矩阵（全局分支）
    - A_private/B_private: 低秩矩阵（私有分支，仅 FedSDG）
    - lambda_k: 可学习的门控参数（仅 FedSDG）
    - r: LoRA 秩
    - alpha: 缩放因子
    """
    # DA 诊断：类级别累积器，汇总所有层的统计后输出一行摘要
    _da_round_stats = []       # 每层的 (g_rms, p_rms, base_rms, target_rms, floor_active, m_k)
    _da_total_layers = 0       # DA 层总数
    _da_first_round = True     # 首轮标记（首轮输出全量，后续输出摘要）
    
    def __init__(self, original_layer, r=8, lora_alpha=16, lora_dropout=0.0, is_fedsdg=False, fix_gate=False, fixed_gate_value=0.5, gate_init_value=0.0, gate_granularity='fine', use_dynamic_alignment=False, da_floor_gamma=0.1, da_target_mode='floor', use_da_scale=False, da_detach_private_rms=True, r_private=None):
        super().__init__()
        # 保存原始线性层（冻结）
        self.original_layer = original_layer
        # 冻结原始权重
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA 参数
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r  # 缩放因子 alpha/r
        self.r_private = r_private if r_private is not None else r
        self.scaling_private = lora_alpha / self.r_private  # 私有分支缩放因子
        self.is_fedsdg = is_fedsdg  # FedSDG 模式标志
        self.gate_granularity = gate_granularity  # 门控粒度模式
        self.global_gate_ref = None  # 全局门控参数引用
        self.use_dynamic_alignment = use_dynamic_alignment  # Dynamic Alignment 模式
        self.da_floor_gamma = da_floor_gamma  # Relative Floor 系数
        self.da_target_mode = da_target_mode  # target_rms 计算模式：floor | base | geomean
        self.use_da_scale = use_da_scale  # DA Scale 全局可学习幅度标量
        self.global_da_scale_ref = None  # 全局 DA-Scale 参数引用（由 builder 注入）
        self.da_detach_private_rms = da_detach_private_rms  # p_rms detach 消融开关
        
        # DA 层数量由 reset_da_diagnostics() 动态计算，不在 __init__ 中累加
        # 避免联邦学习中多次重建模型导致类变量 _da_total_layers 不断累积
        
        # DA 诊断：每轮首次前向传播时报告，由 reset_da_diagnostics() 重置
        self._da_reported_this_round = False
        
        # ========== LoRA Dropout (标准 LoRA 实现) ==========
        # 在 LoRA 分支的输入之前应用 Dropout，遵循 HuggingFace PEFT 的设计
        # 数据流: input x -> Dropout(p) -> A -> B -> output
        # 用于防止过拟合，提高泛化能力
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        
        # ========== 全局分支（Global Path）==========
        # 低秩分解矩阵 A 和 B（在 FedSDG 中作为全局分支）
        # A: (in_features, r) - 使用正态分布初始化（遵循 LoRA 论文）
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        # B: (r, out_features) - 初始化为 0，确保初始时 LoRA 不影响输出
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        # 初始化 A 矩阵：使用正态分布，标准差为 1/sqrt(r) 高斯初始化
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0/r**0.5)
        # B 矩阵保持为 0（已经是 0）
        
        # ========== FedSDG 专用：私有分支（Private Path）==========
        if self.is_fedsdg:
            # ==================== 私有参数初始化策略 ====================
            # 
            # 设计原则:
            # 1. 私有分支需要能够接收梯度，让门控参数 lambda_k 能够学习
            # 2. 初始时私有分支贡献应该较小，但不能为零
            # 
            # 关键问题：如果 B_private = 0，则：
            # - private_output = x @ A_private @ B_private = 0
            # - ∂Loss/∂lambda_k = ∂Loss/∂output * private_output = 0
            # - 门控参数永远不会更新！
            # 
            # 解决方案：给 B_private 一个小的非零初始化
            # - A_private: 正态分布 N(0, 1/sqrt(r))
            # - B_private: 小的正态分布 N(0, 0.01)
            # 
            # 这确保:
            # - 初始时 private_output 很小但非零
            # - 门控参数可以从一开始就接收梯度
            # - 私有分支可以正常学习
            # ==========================================================
            
            # 私有低秩矩阵（不参与服务器聚合，支持独立秩 r_private）
            self.lora_A_private = nn.Parameter(torch.zeros(in_features, self.r_private))
            self.lora_B_private = nn.Parameter(torch.zeros(self.r_private, out_features))
            
            # 初始化 A_private：使用正态分布
            nn.init.normal_(self.lora_A_private, mean=0.0, std=1.0/self.r_private**0.5)
            # 初始化 B_private
            if use_dynamic_alignment:
                # DA 模式：Kaiming 初始化，RMS 归一化消除幅度影响，
                # 更大的初始值提供更稳定的 RMS 计算
                nn.init.kaiming_uniform_(self.lora_B_private, a=math.sqrt(5))
            elif fix_gate:
                # 固定门控模式（PF2LoRA 等）：标准 LoRA B=0 初始化
                # 无可学习门控 → 不需要非零 B_private 来传导门控梯度
                pass  # 保持零初始化
            else:
                # 可学习门控模式（FedSDG）：小的非零初始化，确保门控参数能接收梯度
                nn.init.normal_(self.lora_B_private, mean=0.0, std=0.01)
            
            # 秩剪枝掩码（PF2LoRA 自动秩学习用，FedSDG 默认全 1 无影响）
            self.register_buffer('rank_mask', torch.ones(self.r_private))
            
            # ========== DA Scale：全局可学习幅度标量 ==========
            # 不再创建逐层 da_scale_logit，改用 GlobalDAScaleModule 全局共享
            # 由 builder.py 注入全局引用到 self.global_da_scale_ref
            
            # ==================== 门控参数初始化 ====================
            # 根据门控粒度创建不同的门控参数
            # ==========================================================================
            if gate_granularity == 'coarse':
                # 粗粒度：不创建层门控参数，使用全局门控
                self.lambda_k_logit = None
            else:
                # 细粒度：创建层门控参数（原有逻辑）
                if fix_gate:
                    # 固定门控模式：计算对应的 logit 值
                    if fixed_gate_value <= 0:
                        fixed_logit = -100.0  # m_k ≈ 0
                    elif fixed_gate_value >= 1:
                        fixed_logit = 100.0   # m_k ≈ 1
                    else:
                        # logit = log(p / (1-p))
                        fixed_logit = math.log(fixed_gate_value / (1 - fixed_gate_value))
                    self.lambda_k_logit = nn.Parameter(
                        torch.tensor([fixed_logit]), 
                        requires_grad=False  # 不参与梯度更新
                    )
                else:
                    # 可学习模式：使用配置的初始值（logit形式）
                    # 默认 gate_init_value=0.0 → m_k = sigmoid(0.0) = 0.5，可学习
                    self.lambda_k_logit = nn.Parameter(torch.tensor([gate_init_value]))
        
        # 保存 in_features 和 out_features 供外部访问
        self.in_features = in_features
        self.out_features = out_features
    
    @property
    def weight(self):
        """代理属性：返回原始层的 weight，用于兼容 PyTorch 内部调用"""
        return self.original_layer.weight
    
    @property
    def bias(self):
        """代理属性：返回原始层的 bias，用于兼容 PyTorch 内部调用"""
        return self.original_layer.bias
    
    def set_global_gate(self, global_gate_param):
        """设置全局门控参数引用 - 避免lambda函数"""
        self.global_gate_ref = global_gate_param
    
    def set_global_da_scale(self, da_scale_param):
        """设置全局 DA-Scale 参数引用（不注册为本层参数）
        
        使用 object.__setattr__ 绕过 nn.Module.__setattr__，
        避免将共享参数注册为每个 LoRA 层的参数。
        这确保参数仅在 GlobalDAScaleModule 中注册一次，
        named_parameters() 中只出现 fedsdg_global_da_scale.da_scale_logit。
        """
        object.__setattr__(self, 'global_da_scale_ref', da_scale_param)
        
    def forward(self, x):
        """
        前向传播
        
        FedSDG 模式实现 Equation 4 (FedSDG_Design.md):
        θ̃_{k,l} = θ_{g,l} + m_{k,l} · θ_{p,k,l}
        
        这是**加性残差形式**，将个性化建模为共享结构的残差扰动：
        - θ_{g,l}: 共享适应参数（全局 LoRA）
        - θ_{p,k,l}: 客户端特定适应参数（私有 LoRA）
        - m_{k,l}: 门控权重，调节偏差幅度
        
        极端情况：
        - m_{k,l} = 0: 仅使用共享适应（完全全局）
        - m_{k,l} = 1: 共享 + 完整私有残差（完全个性化）
        - 0 < m_{k,l} < 1: 从共享模型部分偏离（混合模式）
        
        LoRA Dropout (遵循 HuggingFace PEFT 标准实现):
        - 数据流: input x -> Dropout(p) -> A -> B -> output
        - 仅在训练时生效，推理时自动禁用
        """
        # 原始输出: Wx (+ b)
        original_output = self.original_layer(x)
        
        # ========== LoRA Dropout: 在进入 LoRA 分支前应用 ==========
        # 遵循标准 LoRA 实现: input -> Dropout -> A -> B
        x_lora = self.lora_dropout(x)
        
        if self.is_fedsdg:
            # ==================== FedSDG 模式：残差分解适应 (Equation 4) ====================
            # 计算门控参数 m_{k,l} = σ(a_{k,l}) ∈ [0, 1]
            if self.gate_granularity == 'coarse' and self.global_gate_ref is not None:
                # 粗粒度：使用全局门控参数，保持梯度追踪
                m_k = torch.sigmoid(self.global_gate_ref)
            elif self.lambda_k_logit is not None:
                # 细粒度：使用层门控参数
                m_k = torch.sigmoid(self.lambda_k_logit)
            else:
                raise RuntimeError("Gate parameter not properly initialized")
            
            if self.use_dynamic_alignment:
                # ========== Dynamic Alignment 模式 ==========
                # 将私有分支输出对齐到全局分支的量级空间，
                # 使 m_k 成为纯粹的"私有信号相对于全局信号的混合比例"
                # 详见 FedSDG_DynamicAlignment_Design.md
                
                # 约束5：先乘 scaling，在实际输出空间中计算 RMS
                g_scaled = (x_lora @ self.lora_A @ self.lora_B) * self.scaling
                # 私有分支应用 rank_mask 和独立 scaling
                A_p_masked = self.lora_A_private * self.rank_mask.unsqueeze(0)
                B_p_masked = self.lora_B_private * self.rank_mask.unsqueeze(1)
                p_scaled = (x_lora @ A_p_masked @ B_p_masked) * self.scaling_private
                
                # 约束2+4+6：per-image RMS，detach，eps 在 sqrt 内
                g_rms = torch.sqrt(g_scaled.detach().pow(2).mean(dim=(-2, -1), keepdim=True) + 1e-6)
                p_for_rms = p_scaled.detach() if self.da_detach_private_rms else p_scaled
                p_rms = torch.sqrt(p_for_rms.pow(2).mean(dim=(-2, -1), keepdim=True) + 1e-6)
                
                # 约束3：target_rms 计算（多模式支持）
                base_rms = torch.sqrt(original_output.detach().pow(2).mean(dim=(-2, -1), keepdim=True) + 1e-6)
                if self.da_target_mode == 'base':
                    # base 模式：始终以 backbone 输出为参考，完全消除 γ 依赖
                    target_rms = base_rms
                elif self.da_target_mode == 'geomean':
                    # 几何均值模式：target = sqrt(g_rms * base_rms)，无超参数
                    target_rms = torch.sqrt(g_rms * base_rms)
                else:
                    # floor 模式（原始）：target = max(g_rms, γ × base_rms)
                    target_rms = torch.max(g_rms, self.da_floor_gamma * base_rms)
                
                # 提前计算 da_scale，诊断和实际对齐共享同一结果，避免重复 clamp+exp
                if self.use_da_scale and self.global_da_scale_ref is not None:
                    da_scale = torch.exp(self.global_da_scale_ref.clamp(-2.0, 3.0))
                else:
                    da_scale = None
                
                # [DA 诊断] 首轮：每层输出一行；后续轮次：汇总后输出一行摘要
                # _da_reported_this_round 在 __init__ 或 reset_da_diagnostics 中初始化
                if not self._da_reported_this_round:
                    self._da_reported_this_round = True
                    # 将所有 .item() 调用集中在 no_grad 块中，减少计算图开销
                    # 注意：.item() 仍会触发 GPU→CPU 同步，但每轮每层仅执行一次
                    with torch.no_grad():
                        g_rms_val = g_rms.mean().item()
                        p_rms_val = p_rms.mean().item()
                        base_rms_val = base_rms.mean().item()
                        target_rms_val = target_rms.mean().item()
                        floor_active = (g_rms < self.da_floor_gamma * base_rms).float().mean().item() if self.da_target_mode == 'floor' else 0.0
                        m_k_val = m_k.item()
                        gamma_eff_val = target_rms_val / (base_rms_val + 1e-8)
                        effective_val = m_k_val * gamma_eff_val
                        
                        da_scale_val = da_scale.item() if da_scale is not None else 1.0
                    
                    if LoRALayer._da_first_round:
                        # 首轮：每层输出完整诊断
                        cprint(f"  [DA-{self.da_target_mode}] "
                               f"g_rms={g_rms_val:.6f} | "
                               f"p_rms={p_rms_val:.6f} | "
                               f"base_rms={base_rms_val:.4f} | "
                               f"target_rms={target_rms_val:.6f} | "
                               f"γ_eff={gamma_eff_val:.4f} | "
                               f"m_k={m_k_val:.4f} | "
                               f"eff={effective_val:.4f} | "
                               f"da_scale={da_scale_val:.4f}")
                    
                    # 累积统计
                    LoRALayer._da_round_stats.append({
                        'g_rms': g_rms_val, 'p_rms': p_rms_val,
                        'base_rms': base_rms_val, 'target_rms': target_rms_val,
                        'floor_active': floor_active, 'm_k': m_k_val,
                        'da_scale': da_scale_val,
                        'gamma_eff': gamma_eff_val, 'effective': effective_val,
                    })
                    # 所有 DA 层都报告后，输出摘要（非首轮）
                    if not LoRALayer._da_first_round and len(LoRALayer._da_round_stats) == LoRALayer._da_total_layers:
                        stats = LoRALayer._da_round_stats
                        avg_g = sum(s['g_rms'] for s in stats) / len(stats)
                        avg_t = sum(s['target_rms'] for s in stats) / len(stats)
                        avg_fa = sum(s['floor_active'] for s in stats) / len(stats)
                        mk_vals = [s['m_k'] for s in stats]
                        scale_vals = [s['da_scale'] for s in stats]
                        geff_vals = [s['gamma_eff'] for s in stats]
                        eff_vals = [s['effective'] for s in stats]
                        geff_median = sorted(geff_vals)[len(geff_vals)//2]
                        summary_parts = [
                            f"g_rms_avg={avg_g:.6f}",
                            f"target_rms_avg={avg_t:.6f}",
                        ]
                        if self.da_target_mode == 'floor':
                            summary_parts.append(f"floor_active={avg_fa*100:.0f}%")
                        summary_parts.extend([
                            f"γ_eff: {min(geff_vals):.4f}~{max(geff_vals):.4f} (med={geff_median:.4f})",
                            f"m_k: {min(mk_vals):.4f}~{max(mk_vals):.4f} (avg={sum(mk_vals)/len(mk_vals):.4f})",
                            f"eff: {min(eff_vals):.4f}~{max(eff_vals):.4f} (avg={sum(eff_vals)/len(eff_vals):.4f})",
                            f"da_scale: {min(scale_vals):.4f}~{max(scale_vals):.4f} (avg={sum(scale_vals)/len(scale_vals):.4f})",
                        ])
                        cprint(f"  [DA Summary] " + " | ".join(summary_parts))
                
                # 对齐：私有分支归一化到全局分支的尺度
                if da_scale is not None:
                    p_aligned = p_scaled / p_rms * target_rms * da_scale
                else:
                    p_aligned = p_scaled / p_rms * target_rms
                
                # 最终输出（scaling 已在前面应用）
                lora_output = g_scaled + m_k * p_aligned
            else:
                # ========== 原始 FedSDG 模式（Equation 4）==========
                # 全局分支输出: x @ θ_{g,l}（使用 dropout 后的输入）
                global_output = x_lora @ self.lora_A @ self.lora_B
                
                # 私有分支输出: x @ θ_{p,k,l}（应用 rank_mask 和独立 scaling）
                A_p_masked = self.lora_A_private * self.rank_mask.unsqueeze(0)
                B_p_masked = self.lora_B_private * self.rank_mask.unsqueeze(1)
                private_output = x_lora @ A_p_masked @ B_p_masked
                
                # 有效 LoRA 输出 = 全局输出 * scaling + 门控权重 * 私有输出 * scaling_private
                lora_output = global_output * self.scaling + m_k * private_output * self.scaling_private
            # =========================================================================
        else:
            # ========== 标准 LoRA 模式：单路计算 ==========
            # LoRA 输出: (alpha/r) * Dropout(x) @ A @ B
            # lora_A 和 lora_B 是 nn.Parameter，会自动跟随模型移动到正确设备
            lora_output = (x_lora @ self.lora_A @ self.lora_B) * self.scaling
        
        # 组合输出
        return original_output + lora_output


def reset_da_diagnostics(model):
    """重置 DA 诊断标记，使下一轮首次前向时重新输出诊断信息"""
    LoRALayer._da_round_stats = []
    LoRALayer._da_first_round = False
    # 动态计算当前模型中实际的 DA 层数，避免类变量累积
    da_layer_count = 0
    for module in model.modules():
        if isinstance(module, LoRALayer) and module.use_dynamic_alignment:
            module._da_reported_this_round = False
            da_layer_count += 1
    LoRALayer._da_total_layers = da_layer_count


def inject_lora(model, r=8, lora_alpha=16, lora_dropout=0.0, train_mlp_head=True, is_fedsdg=False, fix_gate=False, fixed_gate_value=0.5, gate_init_value=0.0, gate_granularity='fine', use_dynamic_alignment=False, da_floor_gamma=0.1, da_target_mode='floor', use_da_scale=False, da_detach_private_rms=True, r_private=None):
    """
    将 LoRA 层注入到 ViT 模型中
    
    手术位置：
    - 针对 ViT 的 6 层 TransformerEncoderLayer
    - 替换每层中的 self_attn.out_proj (注意力输出投影)
    - 替换每层中的 linear2 (FFN 的第二个线性层)
    
    参数:
        model: ViT 模型实例
        r: LoRA 秩，默认 8
        lora_alpha: LoRA 缩放参数，默认 16
        lora_dropout: LoRA Dropout 概率，默认 0.0（不使用）
            - 遵循 HuggingFace PEFT 标准实现
            - 常用值: 0.05 或 0.1
        train_mlp_head: 是否训练分类头，默认 True
        is_fedsdg: 是否启用 FedSDG 双路架构，默认 False
    
    返回:
        注入 LoRA 后的模型
    """
    if not isinstance(model, ViT):
        raise ValueError("inject_lora 目前仅支持 ViT 模型")
    
    # 0. 获取模型所在设备（用于确保 LoRA 参数在正确设备上）
    device = next(model.parameters()).device
    
    # 1. 冻结整个模型的所有参数
    model.requires_grad_(False)
    
    # 2. 遍历 TransformerEncoder 的所有层，注入 LoRA
    for layer_idx, encoder_layer in enumerate(model.transformer.layers):
        # 2.1 替换 self_attn.out_proj (注意力输出投影层)
        original_out_proj = encoder_layer.self_attn.out_proj
        lora_out_proj = LoRALayer(original_out_proj, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, is_fedsdg=is_fedsdg, fix_gate=fix_gate, fixed_gate_value=fixed_gate_value, gate_init_value=gate_init_value, gate_granularity=gate_granularity, use_dynamic_alignment=use_dynamic_alignment, da_floor_gamma=da_floor_gamma, da_target_mode=da_target_mode, use_da_scale=use_da_scale, da_detach_private_rms=da_detach_private_rms, r_private=r_private)
        lora_out_proj.to(device)  # 将 LoRA 层移动到模型所在设备
        encoder_layer.self_attn.out_proj = lora_out_proj
        
        # 2.2 替换 linear2 (FFN 的第二个线性层)
        original_linear2 = encoder_layer.linear2
        lora_linear2 = LoRALayer(original_linear2, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, is_fedsdg=is_fedsdg, fix_gate=fix_gate, fixed_gate_value=fixed_gate_value, gate_init_value=gate_init_value, gate_granularity=gate_granularity, use_dynamic_alignment=use_dynamic_alignment, da_floor_gamma=da_floor_gamma, da_target_mode=da_target_mode, use_da_scale=use_da_scale, da_detach_private_rms=da_detach_private_rms, r_private=r_private)
        lora_linear2.to(device)  # 将 LoRA 层移动到模型所在设备
        encoder_layer.linear2 = lora_linear2
        
        mode_str = "FedSDG" if is_fedsdg else "LoRA"
        cprint(f"  [{mode_str}] 已注入第 {layer_idx} 层: out_proj 和 linear2")
    
    # 3. 可选：开放分类头的梯度更新
    if train_mlp_head:
        for param in model.mlp_head.parameters():
            param.requires_grad = True
        cprint(f"  [LoRA] mlp_head 参数已解冻用于训练")
    
    # 4. 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cprint(f"  [LoRA] 总参数: {total_params:,} | 可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 5. 验证参数冻结：确保只有 LoRA 参数、mlp_head 和 FedSDG 门控参数可训练
    trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    for name in trainable_param_names:
        if not ('lora_' in name or 'mlp_head' in name or 'lambda_k' in name or 'da_scale' in name):
            raise RuntimeError(
                f"参数冻结验证失败：发现非 LoRA/FedSDG 参数 '{name}' 是可训练的！"
                f"这违背了 LoRA 参数高效微调的原则。"
            )
    cprint(f"  [LoRA] 参数冻结验证通过：仅 LoRA 参数和 mlp_head 可训练")
    
    return model


def inject_lora_timm(model, r=8, lora_alpha=16, lora_dropout=0.0, train_head=True, is_fedsdg=False, fix_gate=False, fixed_gate_value=0.5, gate_init_value=0.0, gate_granularity='fine', use_dynamic_alignment=False, da_floor_gamma=0.1, da_target_mode='floor', use_da_scale=False, da_detach_private_rms=True, r_private=None):
    """
    为 timm 预训练 ViT 模型注入 LoRA 层
    
    与手写 ViT 的主要区别:
        - timm 使用 'blocks' 而非 'layers'
        - 注意力层命名: blocks[i].attn.proj (而非 self_attn.out_proj)
        - FFN 命名: blocks[i].mlp.fc2 (而非 linear2)
        - 分类头命名: 'head' (而非 'mlp_head')
    
    参数:
        model: timm 创建的 ViT 模型
        r: LoRA 秩
        lora_alpha: LoRA 缩放因子
        lora_dropout: LoRA Dropout 概率，默认 0.0（不使用）
            - 遵循 HuggingFace PEFT 标准实现
            - 常用值: 0.05 或 0.1
        train_head: 是否训练分类头
        is_fedsdg: 是否启用 FedSDG 双路架构，默认 False
    
    返回:
        注入 LoRA 后的模型
    """
    mode_str = "FedSDG" if is_fedsdg else "LoRA"
    print("\n" + "="*60)
    print(f"[{mode_str} Injection - timm ViT] 开始注入 {mode_str}...")
    print("="*60)
    
    # 0. 获取模型所在设备
    device = next(model.parameters()).device
    
    # 1. 冻结整个模型
    model.requires_grad_(False)
    print("  [LoRA] 已冻结所有参数")
    
    # 2. 遍历 Transformer blocks，注入 LoRA
    # timm ViT 结构: model.blocks[i].attn.proj 和 model.blocks[i].mlp.fc2
    num_blocks = len(model.blocks)
    
    for block_idx, block in enumerate(model.blocks):
        # 2.1 替换注意力输出投影层 (attn.proj)
        if hasattr(block.attn, 'proj'):
            original_proj = block.attn.proj
            lora_proj = LoRALayer(original_proj, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, is_fedsdg=is_fedsdg, fix_gate=fix_gate, fixed_gate_value=fixed_gate_value, gate_init_value=gate_init_value, gate_granularity=gate_granularity, use_dynamic_alignment=use_dynamic_alignment, da_floor_gamma=da_floor_gamma, da_target_mode=da_target_mode, use_da_scale=use_da_scale, da_detach_private_rms=da_detach_private_rms, r_private=r_private)
            lora_proj.to(device)
            block.attn.proj = lora_proj
            cprint(f"  [{mode_str}] Block {block_idx}: 已注入 attn.proj")
        
        # 2.2 替换 FFN 第二层 (mlp.fc2)
        if hasattr(block.mlp, 'fc2'):
            original_fc2 = block.mlp.fc2
            lora_fc2 = LoRALayer(original_fc2, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, is_fedsdg=is_fedsdg, fix_gate=fix_gate, fixed_gate_value=fixed_gate_value, gate_init_value=gate_init_value, gate_granularity=gate_granularity, use_dynamic_alignment=use_dynamic_alignment, da_floor_gamma=da_floor_gamma, da_target_mode=da_target_mode, use_da_scale=use_da_scale, da_detach_private_rms=da_detach_private_rms, r_private=r_private)
            lora_fc2.to(device)
            block.mlp.fc2 = lora_fc2
            cprint(f"  [{mode_str}] Block {block_idx}: 已注入 mlp.fc2")
    
    # 3. 可选：开放分类头的梯度更新
    if train_head and hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
        cprint(f"  [{mode_str}] 分类头 'head' 参数已解冻用于训练")
    
    # 4. 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cprint(f"  [{mode_str}] 总参数: {total_params:,} | 可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 5. 验证参数冻结
    trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    for name in trainable_param_names:
        if not ('lora_' in name or 'head' in name or 'lambda_k' in name or 'da_scale' in name):
            raise RuntimeError(
                f"参数冻结验证失败：发现非 LoRA/FedSDG 参数 '{name}' 是可训练的！"
            )
    cprint(f"  [{mode_str}] 参数冻结验证通过：仅 {mode_str} 参数和 head 可训练")
    cprint("="*60 + "\n")
    
    return model


def _is_head_param(name: str) -> bool:
    """
    判断参数是否属于分类头（Head）
    
    精确匹配规则，避免误伤注意力相关参数：
    - mlp_head: 手写 ViT 的分类头
    - head: timm ViT 的分类头（但排除 lora_ 前缀和注意力相关）
    
    排除的误匹配模式：
    - attention.head, attn.head: 注意力头
    - head_dim: 注意力头维度
    - multihead, multi_head: 多头注意力
    - num_heads: 注意力头数量
    """
    # 精确匹配 mlp_head（手写 ViT）
    if 'mlp_head' in name:
        return True
    
    # 排除 LoRA 参数（lora_A, lora_B 等）
    if 'lora_' in name:
        return False
    
    # 排除注意力相关的 head
    attention_patterns = [
        'attention.head', 'attn.head',  # 注意力头
        'head_dim',                      # 注意力头维度
        'multihead', 'multi_head',       # 多头注意力
        'num_heads', 'n_heads',          # 注意力头数量
    ]
    name_lower = name.lower()
    for pattern in attention_patterns:
        if pattern in name_lower:
            return False
    
    # timm ViT: 分类头通常是 'head.weight', 'head.bias'
    # 需要确保是顶层的 head，而不是嵌套在其他模块中的
    if '.head.' in name or name.startswith('head.'):
        return True
    
    return False


def get_lora_state_dict(model, include_private: bool = False, head_mode: str = 'global'):
    """
    提取模型中所有 LoRA 相关的参数（包含 'lora_' 关键词的参数）
    用于 FedLoRA 和 FedSDG 的选择性聚合
    
    FedSDG 特殊处理：
    - 排除私有参数（包含 '_private' 的参数）
    - 排除门控参数（包含 'lambda_k' 的参数）
    - 仅上传全局分支参数（lora_A, lora_B）
    
    Args:
        model: 注入了 LoRA 的模型
        include_private: 是否同时返回私有参数（FedSDG 专用）
            - False: 仅返回全局参数 dict（向后兼容）
            - True: 返回 (public_dict, private_dict) 元组
        head_mode: Head 参数模式（FedSDG 专用）
            - 'global': Head 归入 public_dict，参与聚合（默认）
            - 'private': Head 归入 private_dict，不参与聚合
    
    Returns:
        include_private=False: 仅包含 LoRA 全局参数的 state_dict
        include_private=True: (public_dict, private_dict) 元组
            - public_dict: 全局参数，用于服务器聚合
            - private_dict: 私有参数 + 门控参数 [+ Head（若 head_mode='private'）]
    
    Notes:
        include_private controls whether private parameters are returned.
        head_mode controls whether the classifier head is public or private.
        _is_head_param() is used to match classifier-head parameters precisely.
    """
    public_dict = {}
    private_dict = {}
    
    for name, param in model.named_parameters():
        # 检查是否为 LoRA 相关参数、分类头或门控参数
        is_lora = 'lora_' in name
        is_head = _is_head_param(name)
        is_gate = 'lambda_k' in name
        is_global_da_scale = 'fedsdg_global_da_scale' in name
        is_da_scale = 'da_scale_logit' in name
        
        if is_lora or is_head or is_gate or is_da_scale:
            # 使用 detach().clone() 确保独立副本，不共享存储
            tensor_copy = param.data.detach().clone()
            
            # 判断是否为私有参数
            # 全局 DA-Scale → public（参与聚合）
            # 门控、私有 LoRA → private（不参与聚合）
            is_private_param = '_private' in name or is_gate
            if is_da_scale and not is_global_da_scale:
                # 逐层 da_scale（旧版兼容）→ private
                is_private_param = True
            
            # head_mode='private' 时，Head 也归入私有
            if head_mode == 'private' and is_head:
                is_private_param = True
            
            if is_private_param:
                # 私有参数：不参与服务器聚合，仅客户端本地保存
                private_dict[name] = tensor_copy
            else:
                # 全局参数：参与服务器聚合
                public_dict[name] = tensor_copy
    
    if include_private:
        return public_dict, private_dict
    else:
        return public_dict


def get_fedalt_state_dict(model, head_mode: str = 'global'):
    """
    FedALT-adapted 专用的参数提取函数
    
    FedALT 反转 FedSDG 的训练/聚合角色：
    - Individual LoRA (lora_A_private/lora_B_private) → public_dict（上传聚合）
      * 键名去掉 '_private' 后缀，映射为 lora_A/lora_B（与全局状态键名对应）
    - Gate (lambda_k) + Individual LoRA (原始键名) → private_dict（本地保存）
    - Head 参数 → public_dict（参与聚合）
    
    Args:
        model: 注入了双路 LoRA 的模型
        head_mode: Head 参数模式
            - 'global': Head 归入 public_dict（参与聚合）
            - 'private': Head 归入 private_dict（不参与聚合）
    
    Returns:
        (public_dict, private_dict) 元组
        - public_dict: Individual LoRA (renamed) + Head → 用于服务器聚合
        - private_dict: Individual LoRA (original keys) + gate → 客户端本地保存
    """
    public_dict = {}
    private_dict = {}
    
    for name, param in model.named_parameters():
        is_lora = 'lora_' in name
        is_head = _is_head_param(name)
        is_gate = 'lambda_k' in name
        
        if not (is_lora or is_head or is_gate):
            continue
        
        tensor_copy = param.data.detach().clone()
        
        if '_private' in name and is_lora:
            # Individual LoRA → 聚合（键名去 _private，映射到 public 键）
            public_key = name.replace('_private', '')
            public_dict[public_key] = tensor_copy
            # 同时保存到私有状态（训练后的版本，用于个性化评估）
            private_dict[name] = tensor_copy
        elif is_gate:
            # Gate/Mixer → 仅本地保存
            private_dict[name] = tensor_copy
        elif is_head:
            if head_mode == 'private':
                private_dict[name] = tensor_copy
            else:
                public_dict[name] = tensor_copy
        # 注意: public 分支的 lora_A/lora_B 不提取（冻结，不上传）
    
    return public_dict, private_dict


def get_fedsalora_state_dict(model):
    """
    FedSA-LoRA 专用的参数提取函数
    
    FedSA-LoRA 的选择性聚合策略：
    - lora_A 矩阵 → public_dict（上传服务器聚合，学习通用知识）
    - lora_B 矩阵 → private_dict（本地保留，捕获客户端特有知识）
    - Head 参数 → public_dict（参与聚合）
    
    Args:
        model: 注入了 LoRA 的模型
    
    Returns:
        (public_dict, private_dict) 元组
        - public_dict: lora_A + Head 参数，用于服务器聚合
        - private_dict: lora_B 参数，客户端本地保存
    """
    public_dict = {}
    private_dict = {}
    
    for name, param in model.named_parameters():
        is_lora = 'lora_' in name
        is_head = _is_head_param(name)
        
        if not (is_lora or is_head):
            continue
        
        # 使用 detach().clone() 确保独立副本，不共享存储
        tensor_copy = param.data.detach().clone()
        
        if 'lora_B' in name:
            # B 矩阵：本地保留（捕获客户端特有知识）
            private_dict[name] = tensor_copy
        else:
            # A 矩阵 + Head：参与聚合（学习通用知识）
            public_dict[name] = tensor_copy
    
    return public_dict, private_dict


def get_fedtp_state_dict(model, phase: int):
    """
    FedTP (Two-Phase LoRA) 专用的参数提取函数
    
    根据当前训练阶段提取不同的参数：
    - Phase 1 (全局训练): 返回全局 LoRA + Head 用于聚合
    - Phase 2 (本地微调): 返回私有 LoRA + Head 用于本地保存
    
    Args:
        model: 注入了双路 LoRA 的模型
        phase: 当前训练阶段 (1 或 2)
    
    Returns:
        (public_dict, private_dict) 元组
        - Phase 1: public_dict=全局LoRA+Head, private_dict=None
        - Phase 2: public_dict=None, private_dict=私有LoRA+Head
    """
    if phase == 1:
        # Phase 1: 提取全局 LoRA + Head 用于服务器聚合
        public_dict = {}
        for name, param in model.named_parameters():
            is_lora = 'lora_' in name
            is_head = _is_head_param(name)
            is_private = '_private' in name
            is_gate = 'lambda_k' in name
            is_da = 'da_scale' in name
            
            if (is_lora or is_head) and not is_private and not is_gate and not is_da:
                public_dict[name] = param.data.detach().clone()
        return public_dict, None
    else:
        # Phase 2: 提取私有 LoRA + Head 用于本地保存
        # Head 也需要保存，因为 Phase 2 中 Head 随私有分支本地微调
        private_dict = {}
        for name, param in model.named_parameters():
            is_private_lora = '_private' in name and 'lora_' in name
            is_head = _is_head_param(name)
            
            if is_private_lora or is_head:
                private_dict[name] = param.data.detach().clone()
        return None, private_dict


def get_pf2lora_state_dict(model):
    """
    PF2LoRA 专用的参数提取函数
    
    PF2LoRA 的选择性聚合策略（双路加性 LoRA）：
    - Shared LoRA (lora_A, lora_B) → public_dict（上传服务器聚合，学习通用知识）
    - Private LoRA (lora_A_private, lora_B_private) → private_dict（本地保留，捕获客户端特有知识）
    - Head 参数 → public_dict（参与聚合）
    - Gate 参数 (lambda_k) → 排除（固定为 1.0，不需要保存/传输）
    
    Args:
        model: 注入了双路 LoRA 的模型
    
    Returns:
        (public_dict, private_dict) 元组
        - public_dict: Shared LoRA (A+B) + Head 参数，用于服务器聚合
        - private_dict: Private LoRA (A_private+B_private) 参数，客户端本地保存
    """
    public_dict = {}
    private_dict = {}
    
    for name, param in model.named_parameters():
        is_lora = 'lora_' in name
        is_head = _is_head_param(name)
        is_private = '_private' in name
        is_gate = 'lambda_k' in name
        is_da = 'da_scale' in name
        
        if not (is_lora or is_head):
            continue
        
        # 排除 gate 和 DA 参数（PF2LoRA 不使用）
        if is_gate or is_da:
            continue
        
        tensor_copy = param.data.detach().clone()
        
        if is_private:
            # Private LoRA：本地保留
            private_dict[name] = tensor_copy
        else:
            # Shared LoRA (A+B) + Head：参与聚合
            public_dict[name] = tensor_copy
    
    # 保存 rank_mask buffer（剪枝持久化掩码）到 private_dict
    for name, buf in model.named_buffers():
        if 'rank_mask' in name:
            private_dict[name] = buf.data.detach().clone()
    
    return public_dict, private_dict


def prune_private_lora_ranks(model, target_rank):
    """
    PF2LoRA 的自动秩学习：基于重要性分数对 Private LoRA 进行秩剪枝
    
    重要性分数计算：
        s_i = ||A_private[:, i]||₂ · ||B_private[i, :]||₂
    
    保留 top-k 重要维度，将不重要维度的参数置零并更新 rank_mask。
    rank_mask 确保被剪枝维度在 forward 中永久被屏蔽，防止优化器“复活”已剪枝维度。
    
    Args:
        model: 注入了双路 LoRA 的模型
        target_rank: 目标秩（保留的维度数）
    
    Returns:
        pruning_info: 包含剪枝统计信息的字典
    """
    total_pruned = 0
    total_dims = 0
    
    for module in model.modules():
        if isinstance(module, LoRALayer) and module.is_fedsdg:
            r_p = module.r_private
            if target_rank >= r_p:
                continue
            
            # 计算重要性分数: s_i = ||A_private[:, i]|| * ||B_private[i, :]||
            # 仅对当前未被剪枝的维度计算（已剪枝维度 importance=0）
            A_norms = torch.norm(module.lora_A_private.data, dim=0)  # [r_private]
            B_norms = torch.norm(module.lora_B_private.data, dim=1)  # [r_private]
            importance = A_norms * B_norms  # [r_private]
            
            # 找到不重要的维度并置零
            num_to_prune = r_p - target_rank
            _, indices_to_prune = torch.topk(importance, num_to_prune, largest=False)
            
            with torch.no_grad():
                module.lora_A_private.data[:, indices_to_prune] = 0.0
                module.lora_B_private.data[indices_to_prune, :] = 0.0
                # 更新持久化掩码，确保被剪枝维度在 forward 中不参与计算
                module.rank_mask[indices_to_prune] = 0.0
            
            total_pruned += num_to_prune
            total_dims += r_p
    
    pruning_info = {
        'target_rank': target_rank,
        'total_pruned_dims': total_pruned,
        'total_dims': total_dims,
        'pruning_ratio': total_pruned / total_dims if total_dims > 0 else 0.0,
    }
    
    return pruning_info
