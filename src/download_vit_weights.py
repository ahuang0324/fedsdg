#!/usr/bin/env python3
"""
手动下载 ViT 预训练权重
由于网络限制，提供多种下载方式
"""

import os
import sys

print("="*60)
print("ViT 预训练权重下载指南")
print("="*60)
print()

# 权重信息
model_name = "vit_tiny_patch16_224"
weight_url = "https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/main/model.safetensors"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

print(f"模型名称: {model_name}")
print(f"权重 URL: {weight_url}")
print(f"缓存目录: {cache_dir}")
print()

print("由于网络限制，请选择以下方式之一：")
print()

print("【方案 1】使用 wget 下载（如果有外网访问）")
print("-" * 60)
print(f"mkdir -p {cache_dir}")
print(f"cd {cache_dir}")
print(f"wget {weight_url} -O vit_tiny_patch16_224.safetensors")
print()

print("【方案 2】使用镜像站下载")
print("-" * 60)
print("# 国内镜像（如果可用）")
print("# 1. HF Mirror: https://hf-mirror.com")
mirror_url = weight_url.replace("huggingface.co", "hf-mirror.com")
print(f"wget {mirror_url} -O {cache_dir}/vit_tiny_patch16_224.safetensors")
print()

print("【方案 3】从其他机器传输")
print("-" * 60)
print("# 如果你有其他可以访问外网的机器：")
print("# 1. 在外网机器上运行：")
print("python3 -c \"import timm; timm.create_model('vit_tiny_patch16_224', pretrained=True)\"")
print("# 2. 找到下载的权重文件（通常在 ~/.cache/huggingface/hub）")
print("# 3. 使用 scp 传输到当前机器")
print()

print("【方案 4】使用不需要预训练的模式")
print("-" * 60)
print("# 修改代码，先不使用预训练权重，直接训练")
print("# 在 models.py 中设置 pretrained=False")
print()

print("="*60)
print("权重文件大小约: 22 MB")
print("="*60)

# 尝试使用 timm 的离线模式
print("\n尝试检查是否已有缓存的权重...")
try:
    import timm
    from timm.models._hub import get_cache_dir
    cache = get_cache_dir()
    print(f"timm 缓存目录: {cache}")
    
    # 列出已缓存的模型
    if os.path.exists(cache):
        files = os.listdir(cache)
        if files:
            print(f"已缓存的文件: {len(files)} 个")
            for f in files[:5]:
                print(f"  - {f}")
        else:
            print("缓存目录为空")
except Exception as e:
    print(f"检查失败: {e}")
