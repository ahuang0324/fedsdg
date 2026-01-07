#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 LoRA 层的逻辑正确性"""

import torch
import torch.nn as nn
from models import LoRALayer

def test_lora_initialization():
    """测试 LoRA 初始化是否正确（初始时应该不影响输出）"""
    print("=" * 60)
    print("测试 1: LoRA 初始化（应该不影响原始输出）")
    print("=" * 60)
    
    # 创建一个简单的线性层
    in_features, out_features = 128, 128
    original_layer = nn.Linear(in_features, out_features)
    
    # 创建 LoRA 层
    lora_layer = LoRALayer(original_layer, r=8, lora_alpha=16)
    
    # 测试输入
    x = torch.randn(2, 10, in_features)  # (batch, seq, features)
    
    # 原始输出
    with torch.no_grad():
        original_output = original_layer(x)
        lora_output = lora_layer(x)
    
    # 计算差异
    diff = torch.abs(lora_output - original_output).max().item()
    print(f"初始化后的最大输出差异: {diff:.6f}")
    
    # 检查 lora_B 是否为 0
    print(f"lora_B 的最大绝对值: {lora_layer.lora_B.abs().max().item():.6f}")
    print(f"lora_A 的最大绝对值: {lora_layer.lora_A.abs().max().item():.6f}")
    
    if diff < 1e-5:
        print("✓ 初始化测试通过：LoRA 初始时不影响输出")
    else:
        print("✗ 初始化测试失败：LoRA 初始时改变了输出！")
    print()

def test_lora_gradient_flow():
    """测试梯度是否正确流向 LoRA 参数"""
    print("=" * 60)
    print("测试 2: LoRA 梯度流动")
    print("=" * 60)
    
    in_features, out_features = 128, 128
    original_layer = nn.Linear(in_features, out_features)
    lora_layer = LoRALayer(original_layer, r=8, lora_alpha=16)
    
    # 测试输入和目标
    x = torch.randn(2, 10, in_features)
    target = torch.randn(2, 10, out_features)
    
    # 前向传播
    output = lora_layer(x)
    loss = nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"原始层权重梯度: {original_layer.weight.grad}")
    print(f"lora_A 梯度是否存在: {lora_layer.lora_A.grad is not None}")
    print(f"lora_B 梯度是否存在: {lora_layer.lora_B.grad is not None}")
    
    if lora_layer.lora_A.grad is not None and lora_layer.lora_B.grad is not None:
        print(f"lora_A 梯度范数: {lora_layer.lora_A.grad.norm().item():.6f}")
        print(f"lora_B 梯度范数: {lora_layer.lora_B.grad.norm().item():.6f}")
        print("✓ 梯度流动测试通过")
    else:
        print("✗ 梯度流动测试失败：LoRA 参数没有梯度！")
    
    if original_layer.weight.grad is None:
        print("✓ 原始层权重正确冻结（无梯度）")
    else:
        print("✗ 原始层权重未冻结（有梯度）！")
    print()

def test_lora_parameter_update():
    """测试 LoRA 参数更新后是否影响输出"""
    print("=" * 60)
    print("测试 3: LoRA 参数更新效果")
    print("=" * 60)
    
    in_features, out_features = 128, 128
    original_layer = nn.Linear(in_features, out_features)
    lora_layer = LoRALayer(original_layer, r=8, lora_alpha=16)
    
    x = torch.randn(2, 10, in_features)
    
    # 初始输出
    with torch.no_grad():
        output_before = lora_layer(x)
    
    # 手动更新 LoRA 参数
    with torch.no_grad():
        lora_layer.lora_A.data += 0.1
        lora_layer.lora_B.data += 0.1
    
    # 更新后输出
    with torch.no_grad():
        output_after = lora_layer(x)
    
    diff = torch.abs(output_after - output_before).max().item()
    print(f"参数更新后的输出变化: {diff:.6f}")
    
    if diff > 1e-5:
        print("✓ 参数更新测试通过：LoRA 参数更新影响输出")
    else:
        print("✗ 参数更新测试失败：LoRA 参数更新不影响输出！")
    print()

def test_lora_scaling():
    """测试 LoRA 缩放因子"""
    print("=" * 60)
    print("测试 4: LoRA 缩放因子")
    print("=" * 60)
    
    in_features, out_features = 128, 128
    original_layer = nn.Linear(in_features, out_features)
    
    # 测试不同的 alpha 和 r
    configs = [(8, 8), (8, 16), (16, 16)]
    
    for r, alpha in configs:
        lora_layer = LoRALayer(original_layer, r=r, lora_alpha=alpha)
        expected_scaling = alpha / r
        actual_scaling = lora_layer.scaling
        print(f"r={r}, alpha={alpha}: 期望缩放={expected_scaling:.2f}, 实际缩放={actual_scaling:.2f}")
    print()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("LoRA 层逻辑测试")
    print("=" * 60 + "\n")
    
    test_lora_initialization()
    test_lora_gradient_flow()
    test_lora_parameter_update()
    test_lora_scaling()
    
    print("=" * 60)
    print("所有测试完成")
    print("=" * 60)
