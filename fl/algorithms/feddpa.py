# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedDPA: Federated Dual-Path Adapter

FedDPA 算法的核心组件：
- FedDPAClientState: 客户端状态管理（Private LoRA + Anchor Cache）
- FedDPAInference: 推理时 Instance-wise Dynamic Weighting

与 FedSDG 的关键区别:
1. 混合方式: (1-λ)*global + λ*private (加权插值) vs global + m_k*private (加性残差)
2. 门控参数: 训练时固定 λ，推理时动态计算 vs 可学习 m_k
3. 推理阶段: Instance-wise Dynamic Weighting vs 直接使用学习到的 m_k

Usage:
    from fl.algorithms.feddpa import FedDPAClientState, FedDPAInference
    
    # 客户端状态管理
    state_manager = FedDPAClientState()
    state_manager.save_private_state(client_id, private_state)
    
    # 推理时动态权重
    inference = FedDPAInference(model, device, scale_factor=0.5)
    inference.cache_anchors(train_loader, num_samples=5)
    logits = inference.inference_with_dynamic_weight(images)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class FedDPAClientState:
    """
    FedDPA 客户端状态管理器
    
    管理:
    - Private LoRA 参数（跨轮持久化，不参与聚合）
    - Anchor Embeddings（推理时缓存）
    
    与 FedSDG 的 PrivateStateManager 类似，但不管理门控参数。
    """
    
    def __init__(self):
        self.private_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.anchor_cache: Dict[int, torch.Tensor] = {}
    
    def save_private_state(self, client_id: int, private_state: Dict[str, torch.Tensor]) -> None:
        """
        保存客户端的 Private LoRA 参数
        
        Args:
            client_id: 客户端 ID
            private_state: Private 参数字典（包含 _private 后缀的参数）
        """
        self.private_states[client_id] = {
            k: v.clone().cpu() for k, v in private_state.items()
        }
    
    def load_private_state(self, client_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        加载客户端的 Private LoRA 参数
        
        Args:
            client_id: 客户端 ID
            
        Returns:
            Private 参数字典，如果不存在则返回 None
        """
        return self.private_states.get(client_id, None)
    
    def has_state(self, client_id: int) -> bool:
        """检查客户端是否有保存的状态"""
        return client_id in self.private_states
    
    def cache_anchors(self, client_id: int, embeddings: torch.Tensor) -> None:
        """
        缓存 Anchor Embeddings（推理前调用）
        
        Args:
            client_id: 客户端 ID
            embeddings: [N, D] Anchor Embeddings
        """
        self.anchor_cache[client_id] = embeddings.clone().cpu()
    
    def get_anchors(self, client_id: int) -> Optional[torch.Tensor]:
        """
        获取缓存的 Anchor Embeddings
        
        Args:
            client_id: 客户端 ID
            
        Returns:
            Anchor Embeddings，如果不存在则返回 None
        """
        return self.anchor_cache.get(client_id, None)
    
    def get_all_states(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """获取所有客户端的 Private 状态"""
        return self.private_states
    
    def clear_anchor_cache(self):
        """清除所有 Anchor 缓存"""
        self.anchor_cache.clear()


class FedDPAInference:
    """
    FedDPA 推理时 Instance-wise Dynamic Weighting
    
    实现论文中的推理阶段动态权重计算：
    1. Anchor Caching: 从训练集采样，提取 Global-only 模式下的 Embedding
    2. 动态权重: α_t = scale_factor * mean(cosine_sim(test, anchors))
    3. 混合推理: 使用动态 α_t 进行前向传播
    
    Args:
        model: 注入了 DualPathLoRA 的模型
        device: 计算设备
        scale_factor: 推理时的缩放因子（通常等于 train_mix_ratio）
        anchor_count: Anchor 样本数量
    """
    
    def __init__(self, model: nn.Module, device: str, 
                 scale_factor: float = 0.5, anchor_count: int = 5):
        self.model = model
        self.device = device
        self.scale_factor = scale_factor
        self.anchor_count = anchor_count
        self.anchor_embeddings: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._last_hidden_state: Optional[torch.Tensor] = None
    
    def _get_last_transformer_block(self) -> nn.Module:
        """获取最后一个 Transformer Block"""
        if hasattr(self.model, 'blocks'):
            # timm ViT
            return self.model.blocks[-1]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            # 手写 ViT
            return self.model.transformer.layers[-1]
        else:
            raise ValueError("Unsupported model architecture for FedDPA inference")
    
    def _register_hook(self):
        """注册 Forward Hook 提取最后一层 Hidden States"""
        last_block = self._get_last_transformer_block()
        
        def hook_fn(module, input, output):
            # 取最后一个 token 的 hidden state
            if isinstance(output, tuple):
                output = output[0]
            # output: [B, seq_len, hidden_dim]
            # 对于 ViT，通常取 CLS token（第一个）或最后一个 token
            # 这里取最后一个 token，与论文一致
            self._last_hidden_state = output[:, -1, :].clone()  # [B, hidden_dim]
        
        self._hook_handle = last_block.register_forward_hook(hook_fn)
    
    def _remove_hook(self):
        """移除 Forward Hook"""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
    
    def _set_global_only_mode(self):
        """
        设置模型为 Global-only 模式（λ=0）
        
        用于提取 Anchor Embeddings 和测试样本 Embedding。
        """
        from ..models.lora_dpa import DualPathLoRALayer
        for module in self.model.modules():
            if isinstance(module, DualPathLoRALayer):
                module.set_dynamic_mix_ratio(0.0)
    
    def _restore_mode(self):
        """恢复模型混合比例"""
        from ..models.lora_dpa import DualPathLoRALayer
        for module in self.model.modules():
            if isinstance(module, DualPathLoRALayer):
                module.clear_dynamic_mix_ratio()
    
    def _set_dynamic_mix_ratio(self, ratio: float):
        """设置动态混合比例"""
        from ..models.lora_dpa import DualPathLoRALayer
        for module in self.model.modules():
            if isinstance(module, DualPathLoRALayer):
                module.set_dynamic_mix_ratio(ratio)
    
    def cache_anchors(self, train_loader, num_samples: Optional[int] = None, 
                      gpu_transform=None) -> None:
        """
        从训练集采样并缓存 Anchor Embeddings
        
        使用 Global-only 模式提取 Embeddings（论文设定）。
        
        Args:
            train_loader: 训练数据加载器
            num_samples: 采样数量（默认使用 self.anchor_count）
            gpu_transform: GPU 上的数据变换（如 FEMNIST 的 resize）
        """
        if num_samples is None:
            num_samples = self.anchor_count
        
        self.model.eval()
        self._register_hook()
        self._set_global_only_mode()
        
        embeddings = []
        samples_collected = 0
        
        with torch.no_grad():
            for images, _ in train_loader:
                if samples_collected >= num_samples:
                    break
                
                images = images.to(self.device)
                
                # 应用 GPU Transform（如果有）
                if gpu_transform is not None:
                    images = gpu_transform(images)
                
                batch_size = images.size(0)
                needed = num_samples - samples_collected
                
                if batch_size > needed:
                    images = images[:needed]
                
                # Forward pass（触发 hook）
                _ = self.model(images)
                embeddings.append(self._last_hidden_state.clone())
                samples_collected += images.size(0)
        
        self._restore_mode()
        self._remove_hook()
        
        if embeddings:
            self.anchor_embeddings = torch.cat(embeddings, dim=0)[:num_samples]
            # [num_samples, hidden_dim]
        else:
            self.anchor_embeddings = None
    
    def compute_dynamic_weight(self, test_embedding: torch.Tensor) -> float:
        """
        计算单个测试样本的动态权重
        
        公式: α_t = scale_factor * mean(cosine_sim(test, anchors))
        
        Args:
            test_embedding: [hidden_dim] 测试样本的 Embedding
        
        Returns:
            动态权重 α_t（范围 [0, scale_factor]）
        """
        if self.anchor_embeddings is None:
            # 没有 Anchor，返回默认值
            return self.scale_factor
        
        anchors = self.anchor_embeddings.to(test_embedding.device)
        
        # 余弦相似度
        test_norm = F.normalize(test_embedding.unsqueeze(0), dim=1)  # [1, D]
        anchor_norm = F.normalize(anchors, dim=1)  # [N, D]
        
        cos_sim = torch.mm(test_norm, anchor_norm.t())  # [1, N]
        
        # 平均相似度（ReLU 截断避免负值，论文建议）
        mean_sim = torch.clamp(cos_sim.mean(), min=0.0).item()
        
        # 动态权重
        alpha_t = self.scale_factor * mean_sim
        
        return alpha_t
    
    def compute_batch_dynamic_weights(self, test_embeddings: torch.Tensor) -> torch.Tensor:
        """
        批量计算动态权重
        
        Args:
            test_embeddings: [B, hidden_dim] 测试样本的 Embeddings
        
        Returns:
            [B] 动态权重
        """
        if self.anchor_embeddings is None:
            return torch.full((test_embeddings.size(0),), self.scale_factor, 
                            device=test_embeddings.device)
        
        anchors = self.anchor_embeddings.to(test_embeddings.device)
        
        # 批量余弦相似度
        test_norm = F.normalize(test_embeddings, dim=1)  # [B, D]
        anchor_norm = F.normalize(anchors, dim=1)  # [N, D]
        
        cos_sim = torch.mm(test_norm, anchor_norm.t())  # [B, N]
        
        # 每个样本的平均相似度
        mean_sim = torch.clamp(cos_sim.mean(dim=1), min=0.0)  # [B]
        
        # 动态权重
        alpha_t = self.scale_factor * mean_sim
        
        return alpha_t
    
    def inference_with_dynamic_weight(self, images: torch.Tensor, 
                                       gpu_transform=None) -> torch.Tensor:
        """
        使用动态权重进行推理
        
        对每个测试样本：
        1. Global-only 模式提取 Embedding
        2. 计算与 Anchors 的相似度，得到动态权重 α_t
        3. 使用 α_t 进行混合推理
        
        Args:
            images: [B, C, H, W] 输入图像
            gpu_transform: GPU 上的数据变换
        
        Returns:
            logits: [B, num_classes]
        """
        self.model.eval()
        
        # 应用 GPU Transform
        if gpu_transform is not None:
            images = gpu_transform(images)
        
        batch_size = images.size(0)
        
        # 优化：批量处理
        # Step 1: Global-only 模式，获取所有样本的 Embedding
        self._register_hook()
        self._set_global_only_mode()
        
        with torch.no_grad():
            _ = self.model(images)
            test_embeddings = self._last_hidden_state.clone()  # [B, hidden_dim]
        
        self._restore_mode()
        self._remove_hook()
        
        # Step 2: 计算每个样本的动态权重
        alpha_t_batch = self.compute_batch_dynamic_weights(test_embeddings)  # [B]
        
        # Step 3: 逐样本使用动态权重进行推理
        # 注意：由于每个样本的 α_t 不同，需要逐样本处理
        all_logits = []
        
        with torch.no_grad():
            for i in range(batch_size):
                img = images[i:i+1]
                alpha_t = alpha_t_batch[i].item()
                
                # 设置动态混合比例
                self._set_dynamic_mix_ratio(alpha_t)
                logits = self.model(img)
                all_logits.append(logits)
        
        self._restore_mode()
        
        return torch.cat(all_logits, dim=0)
    
    def inference_with_fixed_ratio(self, images: torch.Tensor, 
                                    ratio: float,
                                    gpu_transform=None) -> torch.Tensor:
        """
        使用固定混合比例进行推理（用于消融实验）
        
        Args:
            images: [B, C, H, W] 输入图像
            ratio: 固定的混合比例
            gpu_transform: GPU 上的数据变换
        
        Returns:
            logits: [B, num_classes]
        """
        self.model.eval()
        
        # 应用 GPU Transform
        if gpu_transform is not None:
            images = gpu_transform(images)
        
        self._set_dynamic_mix_ratio(ratio)
        
        with torch.no_grad():
            logits = self.model(images)
        
        self._restore_mode()
        
        return logits
