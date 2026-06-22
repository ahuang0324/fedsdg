# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedDPA Dual-Path LoRA 层实现

FedDPA (Federated Dual-Path Adapter) 的核心 LoRA 层实现。

与 FedSDG 的关键区别:
1. 混合方式: (1-λ)*global + λ*private (加权插值) vs global + m_k*private (加性残差)
2. 门控参数: 训练时固定 λ，推理时动态计算 vs 可学习 m_k
3. 命名规范: 使用 _private 后缀（与 FedSDG 一致，便于复用基础设施）

Usage:
    from fl.models.lora_dpa import DualPathLoRALayer, inject_lora_dpa, get_dpa_state_dict
"""

import math

import torch
from torch import nn
from .vit import ViT


class DualPathLoRALayer(nn.Module):
    """
    FedDPA 双路 LoRA 层
    
    前向传播公式:
        训练: output = Wx + scaling * [(1-λ) * global + λ * private]
        推理: output = Wx + scaling * [(1-α_t) * global + α_t * private]
    
    与 FedSDG LoRALayer 的区别:
    - FedSDG: output = Wx + scaling * [global + m_k * private] (加性残差)
    - FedDPA: output = Wx + scaling * [(1-λ) * global + λ * private] (加权插值)
    
    Args:
        original_layer: 原始线性层（冻结）
        r: LoRA 秩
        lora_alpha: LoRA 缩放因子
        lora_dropout: Dropout 概率
        train_mix_ratio: 训练时的固定混合比例 λ
    """
    
    # DA 诊断：类级别累积器（与 FedSDG LoRALayer 同模式）
    _da_round_stats = []
    _da_total_layers = 0
    _da_first_round = True
    
    def __init__(self, original_layer, r=8, lora_alpha=16, 
                 lora_dropout=0.0, train_mix_ratio=0.5,
                 use_dynamic_alignment=False, da_floor_gamma=0.1,
                 da_target_mode='floor', da_detach_private_rms=True):
        super().__init__()
        
        # 保存原始层（冻结）
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA 参数
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.train_mix_ratio = train_mix_ratio
        
        # Dynamic Alignment 参数
        self.use_dynamic_alignment = use_dynamic_alignment
        self.da_floor_gamma = da_floor_gamma
        self.da_target_mode = da_target_mode
        self.da_detach_private_rms = da_detach_private_rms
        self._da_reported_this_round = False
        
        # Dynamic mixing ratio used during inference.
        self._dynamic_mix_ratio = None
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        
        # ========== Global LoRA (参与联邦聚合) ==========
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        # A: 正态分布初始化
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0/r**0.5)
        # B: 初始化为 0（标准 LoRA）
        
        # ========== Private LoRA (不参与聚合，本地持久化) ==========
        # 命名使用 _private 后缀，与 FedSDG 一致
        self.lora_A_private = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B_private = nn.Parameter(torch.zeros(r, out_features))
        # A_private: 正态分布初始化
        nn.init.normal_(self.lora_A_private, mean=0.0, std=1.0/r**0.5)
        # B_private: DA 模式用 Kaiming 初始化（更稳定的 RMS），否则小非零值
        if use_dynamic_alignment:
            nn.init.kaiming_uniform_(self.lora_B_private, a=math.sqrt(5))
        else:
            nn.init.normal_(self.lora_B_private, mean=0.0, std=0.01)
        
        # 保存维度信息
        self.in_features = in_features
        self.out_features = out_features
    
    @property
    def weight(self):
        """代理属性：返回原始层的 weight"""
        return self.original_layer.weight
    
    @property
    def bias(self):
        """代理属性：返回原始层的 bias"""
        return self.original_layer.bias
    
    def set_dynamic_mix_ratio(self, ratio: float):
        """设置动态混合比例（推理时使用）"""
        self._dynamic_mix_ratio = ratio
    
    def clear_dynamic_mix_ratio(self):
        """清除动态混合比例，恢复使用 train_mix_ratio"""
        self._dynamic_mix_ratio = None
    
    @classmethod
    def reset_da_diagnostics(cls):
        """重置 DA 诊断状态（每轮训练开始前调用）"""
        cls._da_round_stats = []
        # 动态计算 DA 层总数（避免联邦学习中多次重建模型导致累积）
        cls._da_total_layers = 0
    
    def forward(self, x):
        """
        前向传播
        
        混合比例优先级:
        1. _dynamic_mix_ratio（推理时动态设置）
        2. train_mix_ratio（训练时固定值）
        
        DA 模式:
        将私有分支输出对齐到全局分支的量级空间，
        使 λ 成为纯粹的"私有信号相对于全局信号的混合比例"。
        
        Returns:
            输出张量
        """
        # 原始输出
        base_out = self.original_layer(x)
        
        # Dropout
        x_lora = self.lora_dropout(x)
        
        # 确定混合比例
        lambda_val = self._dynamic_mix_ratio if self._dynamic_mix_ratio is not None else self.train_mix_ratio
        
        if self.use_dynamic_alignment:
            # ========== Dynamic Alignment 模式 ==========
            # 先乘 scaling，在实际输出空间中计算 RMS
            g_scaled = (x_lora @ self.lora_A @ self.lora_B) * self.scaling
            p_scaled = (x_lora @ self.lora_A_private @ self.lora_B_private) * self.scaling
            
            # per-image RMS，detach，eps 在 sqrt 内
            g_rms = torch.sqrt(g_scaled.detach().pow(2).mean(dim=(-2, -1), keepdim=True) + 1e-6)
            p_for_rms = p_scaled.detach() if self.da_detach_private_rms else p_scaled
            p_rms = torch.sqrt(p_for_rms.pow(2).mean(dim=(-2, -1), keepdim=True) + 1e-6)
            
            # target_rms 计算（多模式支持）
            base_rms = torch.sqrt(base_out.detach().pow(2).mean(dim=(-2, -1), keepdim=True) + 1e-6)
            if self.da_target_mode == 'global':
                # global 模式：对齐到全局分支 RMS（FedDPA 推荐）
                # 使 λ 成为纯粹的同量级信号混合比例
                target_rms = g_rms
            elif self.da_target_mode == 'base':
                target_rms = base_rms
            elif self.da_target_mode == 'geomean':
                target_rms = torch.sqrt(g_rms * base_rms)
            else:
                # floor 模式：target = max(g_rms, γ × base_rms)
                target_rms = torch.max(g_rms, self.da_floor_gamma * base_rms)
            
            # DA 诊断日志
            if not self._da_reported_this_round:
                self._da_reported_this_round = True
                with torch.no_grad():
                    g_rms_val = g_rms.mean().item()
                    p_rms_val = p_rms.mean().item()
                    base_rms_val = base_rms.mean().item()
                    target_rms_val = target_rms.mean().item()
                    floor_active = (g_rms < self.da_floor_gamma * base_rms).float().mean().item() if self.da_target_mode == 'floor' else 0.0
                    gamma_eff_val = target_rms_val / (base_rms_val + 1e-8)
                    effective_val = lambda_val * gamma_eff_val
                
                if DualPathLoRALayer._da_first_round:
                    print(f"  [FedDPA-DA-{self.da_target_mode}] "
                          f"g_rms={g_rms_val:.6f} | "
                          f"p_rms={p_rms_val:.6f} | "
                          f"base_rms={base_rms_val:.4f} | "
                          f"target_rms={target_rms_val:.6f} | "
                          f"gamma_eff={gamma_eff_val:.4f} | "
                          f"lambda={lambda_val:.4f} | "
                          f"eff={effective_val:.4f}")
                
                DualPathLoRALayer._da_round_stats.append({
                    'g_rms': g_rms_val, 'p_rms': p_rms_val,
                    'base_rms': base_rms_val, 'target_rms': target_rms_val,
                    'floor_active': floor_active, 'lambda': lambda_val,
                    'gamma_eff': gamma_eff_val, 'effective': effective_val,
                })
                # 所有 DA 层都报告后，输出摘要（非首轮）
                if (not DualPathLoRALayer._da_first_round
                        and len(DualPathLoRALayer._da_round_stats) == DualPathLoRALayer._da_total_layers):
                    stats = DualPathLoRALayer._da_round_stats
                    avg_g = sum(s['g_rms'] for s in stats) / len(stats)
                    avg_t = sum(s['target_rms'] for s in stats) / len(stats)
                    avg_fa = sum(s['floor_active'] for s in stats) / len(stats)
                    geff_vals = [s['gamma_eff'] for s in stats]
                    eff_vals = [s['effective'] for s in stats]
                    geff_median = sorted(geff_vals)[len(geff_vals) // 2]
                    summary_parts = [
                        f"g_rms_avg={avg_g:.6f}",
                        f"target_rms_avg={avg_t:.6f}",
                    ]
                    if self.da_target_mode == 'floor':
                        summary_parts.append(f"floor_active={avg_fa * 100:.0f}%")
                    summary_parts.extend([
                        f"gamma_eff: {min(geff_vals):.4f}~{max(geff_vals):.4f} (med={geff_median:.4f})",
                        f"lambda={lambda_val:.4f}",
                        f"eff: {min(eff_vals):.4f}~{max(eff_vals):.4f} (avg={sum(eff_vals) / len(eff_vals):.4f})",
                    ])
                    print(f"  [FedDPA-DA Summary] " + " | ".join(summary_parts))
            
            # 对齐：私有分支归一化到目标量级
            p_aligned = p_scaled / p_rms * target_rms
            
            # 最终输出（scaling 已在前面应用）
            lora_output = (1 - lambda_val) * g_scaled + lambda_val * p_aligned
            return base_out + lora_output
        else:
            # ========== 原始 FedDPA 模式 ==========
            # Global LoRA 输出
            global_out = x_lora @ self.lora_A @ self.lora_B
            
            # Private LoRA 输出
            private_out = x_lora @ self.lora_A_private @ self.lora_B_private
            
            # 加权混合 + 统一缩放
            # 公式: (1-λ) * global + λ * private
            delta = (1 - lambda_val) * global_out + lambda_val * private_out
            
            return base_out + delta * self.scaling


def inject_lora_dpa(model, r=8, lora_alpha=16, lora_dropout=0.0, 
                    train_mlp_head=True, train_mix_ratio=0.5,
                    use_dynamic_alignment=False, da_floor_gamma=0.1,
                    da_target_mode='floor', da_detach_private_rms=True):
    """
    将 FedDPA 双路 LoRA 层注入到手写 ViT 模型中
    
    注入位置（与 FedSDG 一致）:
    - encoder_layer.self_attn.out_proj (注意力输出投影)
    - encoder_layer.linear2 (FFN 的第二个线性层)
    
    Args:
        model: ViT 模型实例
        r: LoRA 秩
        lora_alpha: LoRA 缩放参数
        lora_dropout: Dropout 概率
        train_mlp_head: 是否训练分类头
        train_mix_ratio: 训练时的固定混合比例
        use_dynamic_alignment: 是否启用 Dynamic Alignment
        da_floor_gamma: DA Relative Floor 系数
        da_target_mode: DA target_rms 计算模式
        da_detach_private_rms: 是否 detach p_rms
    
    Returns:
        注入 LoRA 后的模型
    """
    if not isinstance(model, ViT):
        raise ValueError("inject_lora_dpa 目前仅支持 ViT 模型")
    
    device = next(model.parameters()).device
    
    # DA 参数字典，避免重复书写
    da_kwargs = dict(
        use_dynamic_alignment=use_dynamic_alignment,
        da_floor_gamma=da_floor_gamma,
        da_target_mode=da_target_mode,
        da_detach_private_rms=da_detach_private_rms,
    )
    
    # 1. 冻结整个模型
    model.requires_grad_(False)
    
    # 2. 遍历 TransformerEncoder 的所有层，注入 LoRA
    for layer_idx, encoder_layer in enumerate(model.transformer.layers):
        # 2.1 替换 self_attn.out_proj
        original_out_proj = encoder_layer.self_attn.out_proj
        lora_out_proj = DualPathLoRALayer(
            original_out_proj, r=r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, train_mix_ratio=train_mix_ratio,
            **da_kwargs
        )
        lora_out_proj.to(device)
        encoder_layer.self_attn.out_proj = lora_out_proj
        
        # 2.2 替换 linear2
        original_linear2 = encoder_layer.linear2
        lora_linear2 = DualPathLoRALayer(
            original_linear2, r=r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, train_mix_ratio=train_mix_ratio,
            **da_kwargs
        )
        lora_linear2.to(device)
        encoder_layer.linear2 = lora_linear2
        
        print(f"  [FedDPA] 已注入第 {layer_idx} 层: out_proj 和 linear2")
    
    # 3. 可选：开放分类头的梯度更新
    if train_mlp_head:
        for param in model.mlp_head.parameters():
            param.requires_grad = True
        print(f"  [FedDPA] mlp_head 参数已解冻用于训练")
    
    # 4. 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [FedDPA] 总参数: {total_params:,} | 可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 5. 验证参数冻结
    trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    for name in trainable_param_names:
        if not ('lora_' in name or 'mlp_head' in name):
            raise RuntimeError(
                f"参数冻结验证失败：发现非 LoRA/FedDPA 参数 '{name}' 是可训练的！"
            )
    print(f"  [FedDPA] 参数冻结验证通过：仅 LoRA 参数和 mlp_head 可训练")
    
    # 6. DA 层计数（用于诊断摘要）
    if use_dynamic_alignment:
        da_count = sum(1 for m in model.modules() if isinstance(m, DualPathLoRALayer) and m.use_dynamic_alignment)
        DualPathLoRALayer._da_total_layers = da_count
        DualPathLoRALayer._da_first_round = True
        print(f"  [FedDPA] Dynamic Alignment: {da_count} 层已启用 (mode={da_target_mode}, gamma={da_floor_gamma})")
    
    return model


def inject_lora_dpa_timm(model, r=8, lora_alpha=16, lora_dropout=0.0, 
                         train_head=True, train_mix_ratio=0.5,
                         use_dynamic_alignment=False, da_floor_gamma=0.1,
                         da_target_mode='floor', da_detach_private_rms=True):
    """
    为 timm 预训练 ViT 模型注入 FedDPA 双路 LoRA 层
    
    注入位置（与 FedSDG 一致）:
    - blocks[i].attn.proj (注意力输出投影)
    - blocks[i].mlp.fc2 (FFN 第二层)
    
    Args:
        model: timm 创建的 ViT 模型
        r: LoRA 秩
        lora_alpha: LoRA 缩放因子
        lora_dropout: Dropout 概率
        train_head: 是否训练分类头
        train_mix_ratio: 训练时的固定混合比例
        use_dynamic_alignment: 是否启用 Dynamic Alignment
        da_floor_gamma: DA Relative Floor 系数
        da_target_mode: DA target_rms 计算模式
        da_detach_private_rms: 是否 detach p_rms
    
    Returns:
        注入 LoRA 后的模型
    """
    print("\n" + "="*60)
    print(f"[FedDPA Injection - timm ViT] 开始注入 FedDPA...")
    if use_dynamic_alignment:
        print(f"  [FedDPA] Dynamic Alignment: ON (mode={da_target_mode}, gamma={da_floor_gamma})")
    print("="*60)
    
    device = next(model.parameters()).device
    
    # DA 参数字典
    da_kwargs = dict(
        use_dynamic_alignment=use_dynamic_alignment,
        da_floor_gamma=da_floor_gamma,
        da_target_mode=da_target_mode,
        da_detach_private_rms=da_detach_private_rms,
    )
    
    # 1. 冻结整个模型
    model.requires_grad_(False)
    print("  [FedDPA] 已冻结所有参数")
    
    # 2. 遍历 Transformer blocks，注入 LoRA
    for block_idx, block in enumerate(model.blocks):
        # 2.1 替换注意力输出投影层 (attn.proj)
        if hasattr(block.attn, 'proj'):
            original_proj = block.attn.proj
            lora_proj = DualPathLoRALayer(
                original_proj, r=r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, train_mix_ratio=train_mix_ratio,
                **da_kwargs
            )
            lora_proj.to(device)
            block.attn.proj = lora_proj
            print(f"  [FedDPA] Block {block_idx}: 已注入 attn.proj")
        
        # 2.2 替换 FFN 第二层 (mlp.fc2)
        if hasattr(block.mlp, 'fc2'):
            original_fc2 = block.mlp.fc2
            lora_fc2 = DualPathLoRALayer(
                original_fc2, r=r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, train_mix_ratio=train_mix_ratio,
                **da_kwargs
            )
            lora_fc2.to(device)
            block.mlp.fc2 = lora_fc2
            print(f"  [FedDPA] Block {block_idx}: 已注入 mlp.fc2")
    
    # 3. 可选：开放分类头的梯度更新
    if train_head and hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
        print(f"  [FedDPA] 分类头 'head' 参数已解冻用于训练")
    
    # 4. 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [FedDPA] 总参数: {total_params:,} | 可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 5. 验证参数冻结
    trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    for name in trainable_param_names:
        if not ('lora_' in name or 'head' in name):
            raise RuntimeError(
                f"参数冻结验证失败：发现非 LoRA/FedDPA 参数 '{name}' 是可训练的！"
            )
    print(f"  [FedDPA] 参数冻结验证通过：仅 LoRA 参数和 head 可训练")
    
    # 6. DA 层计数（用于诊断摘要）
    if use_dynamic_alignment:
        da_count = sum(1 for m in model.modules() if isinstance(m, DualPathLoRALayer) and m.use_dynamic_alignment)
        DualPathLoRALayer._da_total_layers = da_count
        DualPathLoRALayer._da_first_round = True
        print(f"  [FedDPA] Dynamic Alignment: {da_count} 层已启用 (mode={da_target_mode}, gamma={da_floor_gamma})")
    print("="*60 + "\n")
    
    return model


def get_dpa_state_dict(model, include_private: bool = False, head_mode: str = 'global'):
    """
    提取 FedDPA 模型参数
    
    与 FedSDG 的 get_lora_state_dict 逻辑一致，便于复用。
    
    Args:
        model: 注入了 DualPathLoRA 的模型
        include_private: 是否同时返回 Private 参数
        head_mode: Head 参数模式（FedDPA 专用）
            - 'global': Head 归入 global_dict，参与聚合（默认）
            - 'private': Head 归入 private_dict，不参与聚合
    
    Returns:
        include_private=False: 仅 Global 参数（用于聚合）
        include_private=True: (global_dict, private_dict) 元组
    """
    from .lora import _is_head_param
    
    global_dict = {}
    private_dict = {}
    
    for name, param in model.named_parameters():
        # 检查是否为 LoRA 相关参数或分类头
        is_lora = 'lora_' in name
        is_head = _is_head_param(name)
        
        if is_lora or is_head:
            tensor_copy = param.data.detach().clone()
            
            # 判断是否为私有参数
            is_private_param = '_private' in name
            
            # head_mode='private' 时，Head 也归入私有
            if head_mode == 'private' and is_head:
                is_private_param = True
            
            if is_private_param:
                # Private 参数：不参与聚合
                private_dict[name] = tensor_copy
            else:
                # Global 参数：参与聚合
                global_dict[name] = tensor_copy
    
    if include_private:
        return global_dict, private_dict
    else:
        return global_dict


def set_model_mix_ratio(model, ratio: float):
    """
    设置模型中所有 DualPathLoRALayer 的动态混合比例
    
    用于推理时动态调整混合比例。
    
    Args:
        model: 注入了 DualPathLoRA 的模型
        ratio: 混合比例
    """
    for module in model.modules():
        if isinstance(module, DualPathLoRALayer):
            module.set_dynamic_mix_ratio(ratio)


def clear_model_mix_ratio(model):
    """
    清除模型中所有 DualPathLoRALayer 的动态混合比例
    
    恢复使用 train_mix_ratio。
    
    Args:
        model: 注入了 DualPathLoRA 的模型
    """
    for module in model.modules():
        if isinstance(module, DualPathLoRALayer):
            module.clear_dynamic_mix_ratio()
