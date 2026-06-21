# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vision Transformer (ViT) model definitions."""

import torch
from torch import nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARNING] timm library not found. Pretrained models will not be available.")
    print("[WARNING] Install with: pip install timm")


class ViT(nn.Module):
    """
    Vision Transformer implementation from scratch.
    
    A simple ViT for small-scale datasets like CIFAR-10/100.
    """
    
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim=128,
        depth=6,
        heads=8,
        mlp_dim=256,
        channels=3,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError('image_size must be divisible by patch_size')
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        patch_dim = channels * patch_size * patch_size

        self.patch_size = patch_size
        self.dim = dim

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size, bias=True),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mlp_head.weight, std=0.02)
        if self.mlp_head.bias is not None:
            nn.init.zeros_(self.mlp_head.bias)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.emb_dropout(x)

        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.mlp_head(x)
        return x



# ViT 模型注册表：vit_type -> (timm 模型名, 简要描述, 近似参数量)
VIT_REGISTRY = {
    'tiny': ('vit_tiny_patch16_224', 'ViT-Tiny (5.7M params, 适合联邦学习)', '5.7M'),
    'base': ('vit_base_patch16_224', 'ViT-Base (86M params, 更强表达力)', '86M'),
}


def get_pretrained_vit(num_classes=10, image_size=224, vit_type='tiny', use_mirror=True):
    """
    创建预训练的 ViT 模型（基于 timm 库）
    
    参数:
        num_classes: 分类数量，默认 10（CIFAR-10）
        image_size: 输入图像尺寸，默认 224
        vit_type: ViT 架构类型，默认 'tiny'
            - 'tiny': vit_tiny_patch16_224 (~5.7M params，轻量级)
            - 'base': vit_base_patch16_224 (~86M params，更强表达力)
        use_mirror: 是否使用国内镜像（hf-mirror.com），默认 True
    
    返回:
        预训练的 ViT 模型实例
    
    注意:
        - timm 会自动缓存权重到 ~/.cache/huggingface/hub/，无需手动管理
        - 分类头会被自动替换为适配 num_classes 的新头
        - ViT-Base 显存需求较大，建议配合降低 batch size 使用（如 local_bs=32）
    """
    if not TIMM_AVAILABLE:
        raise ImportError(
            "timm library is required for pretrained models. "
            "Install with: pip install timm"
        )
    
    # 查找模型
    if vit_type not in VIT_REGISTRY:
        raise ValueError(
            f"不支持的 vit_type: '{vit_type}'，有效选项: {list(VIT_REGISTRY.keys())}"
        )
    
    timm_model_name, description, approx_params = VIT_REGISTRY[vit_type]
    
    print(f"\n{'='*60}")
    print(f"[Pretrained ViT] 正在加载预训练模型...")
    print(f"  架构: {timm_model_name} ({description})")
    print(f"{'='*60}")
    
    # 设置 HuggingFace 镜像（国内用户）
    if use_mirror:
        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print(f"  使用 HuggingFace 镜像: https://hf-mirror.com")
    
    print(f"  从 timm 加载预训练权重（ImageNet-1k）")
    print(f"  权重缓存位置: ~/.cache/huggingface/hub/")
    
    # 创建预训练模型
    # timm 会自动处理缓存，首次下载后会从本地加载
    try:
        model = timm.create_model(timm_model_name, pretrained=True, num_classes=num_classes)
        print(f"  ✓ 预训练权重加载成功")
    except Exception as e:
        print(f"\n  ✗ 加载失败: {str(e)}")
        print(f"  → 降级到无预训练模式（从零初始化）")
        print(f"  → 注意：性能会显著下降")
        model = timm.create_model(timm_model_name, pretrained=False, num_classes=num_classes)
        print(f"  ✓ 已创建无预训练权重的 {timm_model_name} 模型")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型: {timm_model_name}")
    print(f"  总参数量: {total_params:,}")
    print(f"  分类数: {num_classes}")
    print(f"  输入尺寸: {image_size}x{image_size}")
    print(f"{'='*60}\n")
    
    return model


