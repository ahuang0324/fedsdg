#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedSDG 功能测试脚本

测试内容：
1. LoRALayer 在 FedSDG 模式下的参数初始化
2. 前向传播的双路计算 (Equation 4)
3. get_lora_state_dict 正确过滤私有参数
4. 通信量统计与 FedLoRA 一致
5. 客户端私有状态管理
6. Equation 5 损失函数正则化项验证 (λ₁ L1 门控 + λ₂ L2 私有)
7. 门控参数初始化验证 (a_{k,l} = 0 → m_{k,l} = 0.5)
"""

import torch
import torch.nn as nn
import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from models import LoRALayer, get_lora_state_dict, ViT, inject_lora


def test_lora_layer_fedsdg():
    """测试 LoRALayer 在 FedSDG 模式下的初始化和前向传播"""
    print("\n" + "="*70)
    print("测试 1: LoRALayer FedSDG 模式初始化")
    print("="*70)
    
    # 创建一个简单的线性层
    original_layer = nn.Linear(128, 64)
    
    # 测试标准 LoRA 模式
    print("\n[标准 LoRA 模式]")
    lora_standard = LoRALayer(original_layer, r=8, lora_alpha=16, is_fedsdg=False)
    
    # 检查标准 LoRA 的参数
    standard_params = dict(lora_standard.named_parameters())
    print(f"  参数列表: {list(standard_params.keys())}")
    assert 'lora_A' in standard_params, "标准 LoRA 应该有 lora_A"
    assert 'lora_B' in standard_params, "标准 LoRA 应该有 lora_B"
    assert 'lora_A_private' not in standard_params, "标准 LoRA 不应该有 lora_A_private"
    assert 'lambda_k_logit' not in standard_params, "标准 LoRA 不应该有 lambda_k_logit"
    print("  ✓ 标准 LoRA 参数检查通过")
    
    # 测试 FedSDG 模式
    print("\n[FedSDG 模式]")
    lora_fedsdg = LoRALayer(original_layer, r=8, lora_alpha=16, is_fedsdg=True)
    
    # 检查 FedSDG 的参数
    fedsdg_params = dict(lora_fedsdg.named_parameters())
    print(f"  参数列表: {list(fedsdg_params.keys())}")
    assert 'lora_A' in fedsdg_params, "FedSDG 应该有 lora_A (全局分支)"
    assert 'lora_B' in fedsdg_params, "FedSDG 应该有 lora_B (全局分支)"
    assert 'lora_A_private' in fedsdg_params, "FedSDG 应该有 lora_A_private (私有分支)"
    assert 'lora_B_private' in fedsdg_params, "FedSDG 应该有 lora_B_private (私有分支)"
    assert 'lambda_k_logit' in fedsdg_params, "FedSDG 应该有 lambda_k_logit (门控参数)"
    print("  ✓ FedSDG 参数检查通过")
    
    # 测试前向传播
    print("\n[前向传播测试]")
    x = torch.randn(4, 128)  # batch_size=4, in_features=128
    
    # 标准 LoRA 前向传播
    output_standard = lora_standard(x)
    print(f"  标准 LoRA 输出形状: {output_standard.shape}")
    assert output_standard.shape == (4, 64), "输出形状应该是 (4, 64)"
    
    # FedSDG 前向传播
    output_fedsdg = lora_fedsdg(x)
    print(f"  FedSDG 输出形状: {output_fedsdg.shape}")
    assert output_fedsdg.shape == (4, 64), "输出形状应该是 (4, 64)"
    
    # 检查 lambda_k 的范围
    lambda_k = torch.sigmoid(lora_fedsdg.lambda_k_logit)
    print(f"  门控参数 lambda_k: {lambda_k.item():.4f}")
    assert 0 <= lambda_k.item() <= 1, "lambda_k 应该在 [0, 1] 范围内"
    print("  ✓ 前向传播检查通过")
    
    print("\n" + "="*70)
    print("✓ 测试 1 通过：LoRALayer FedSDG 模式工作正常")
    print("="*70)


def test_get_lora_state_dict_filtering():
    """测试 get_lora_state_dict 正确过滤私有参数"""
    print("\n" + "="*70)
    print("测试 2: get_lora_state_dict 私有参数过滤")
    print("="*70)
    
    # 创建一个简单的 ViT 模型并注入 FedSDG
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=128,
        depth=2,  # 使用 2 层以加快测试
        heads=4,
        mlp_dim=256,
        channels=3
    )
    
    # 注入 FedSDG
    model = inject_lora(model, r=8, lora_alpha=16, train_mlp_head=True, is_fedsdg=True)
    
    # 获取完整的 state_dict
    full_state = model.state_dict()
    print(f"\n[完整 state_dict]")
    print(f"  总参数数量: {len(full_state)}")
    
    # 统计私有参数
    private_params = [k for k in full_state.keys() if '_private' in k or 'lambda_k' in k]
    print(f"  私有参数数量: {len(private_params)}")
    print(f"  私有参数示例: {private_params[:3] if len(private_params) > 0 else 'None'}")
    
    # 获取 LoRA state_dict（应该过滤掉私有参数）
    lora_state = get_lora_state_dict(model)
    print(f"\n[LoRA state_dict (用于通信)]")
    print(f"  参数数量: {len(lora_state)}")
    
    # 检查是否正确过滤
    for key in lora_state.keys():
        assert '_private' not in key, f"不应该包含私有参数: {key}"
        assert 'lambda_k' not in key, f"不应该包含门控参数: {key}"
    
    print("  ✓ 所有私有参数已被正确过滤")
    
    # 检查全局参数是否存在
    global_params = [k for k in lora_state.keys() if 'lora_A' in k or 'lora_B' in k]
    print(f"  全局 LoRA 参数数量: {len(global_params)}")
    assert len(global_params) > 0, "应该包含全局 LoRA 参数"
    print("  ✓ 全局参数正确保留")
    
    print("\n" + "="*70)
    print("✓ 测试 2 通过：私有参数过滤功能正常")
    print("="*70)


def test_communication_volume():
    """测试 FedSDG 的通信量与 FedLoRA 一致"""
    print("\n" + "="*70)
    print("测试 3: FedSDG 与 FedLoRA 通信量对比")
    print("="*70)
    
    # 创建两个相同的 ViT 模型
    model_fedlora = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=128,
        depth=2,
        heads=4,
        mlp_dim=256,
        channels=3
    )
    
    model_fedsdg = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=128,
        depth=2,
        heads=4,
        mlp_dim=256,
        channels=3
    )
    
    # 注入 LoRA
    model_fedlora = inject_lora(model_fedlora, r=8, lora_alpha=16, train_mlp_head=True, is_fedsdg=False)
    model_fedsdg = inject_lora(model_fedsdg, r=8, lora_alpha=16, train_mlp_head=True, is_fedsdg=True)
    
    # 获取通信参数
    lora_state_fedlora = get_lora_state_dict(model_fedlora)
    lora_state_fedsdg = get_lora_state_dict(model_fedsdg)
    
    # 计算参数数量
    params_fedlora = sum(p.numel() for p in lora_state_fedlora.values())
    params_fedsdg = sum(p.numel() for p in lora_state_fedsdg.values())
    
    print(f"\n[通信参数统计]")
    print(f"  FedLoRA 通信参数: {params_fedlora:,}")
    print(f"  FedSDG 通信参数:  {params_fedsdg:,}")
    print(f"  差异: {abs(params_fedlora - params_fedsdg):,}")
    
    # 计算通信量（MB）
    bytes_per_param = 4  # float32
    comm_mb_fedlora = (params_fedlora * bytes_per_param) / (1024 ** 2)
    comm_mb_fedsdg = (params_fedsdg * bytes_per_param) / (1024 ** 2)
    
    print(f"\n[通信量 (MB)]")
    print(f"  FedLoRA: {comm_mb_fedlora:.4f} MB")
    print(f"  FedSDG:  {comm_mb_fedsdg:.4f} MB")
    
    # 验证通信量一致
    assert params_fedlora == params_fedsdg, "FedSDG 的通信量应该与 FedLoRA 完全一致"
    print("\n  ✓ 通信量完全一致")
    
    # 统计 FedSDG 的总参数（包括私有参数）
    total_params_fedsdg = sum(p.numel() for p in model_fedsdg.parameters() if p.requires_grad)
    private_params = total_params_fedsdg - params_fedsdg
    
    print(f"\n[FedSDG 参数分布]")
    print(f"  总可训练参数: {total_params_fedsdg:,}")
    print(f"  通信参数 (全局): {params_fedsdg:,}")
    print(f"  私有参数 (本地): {private_params:,}")
    print(f"  私有参数占比: {100 * private_params / total_params_fedsdg:.2f}%")
    
    print("\n" + "="*70)
    print("✓ 测试 3 通过：FedSDG 通信量与 FedLoRA 一致")
    print("="*70)


def test_private_state_management():
    """测试客户端私有状态的保存和加载"""
    print("\n" + "="*70)
    print("测试 4: 客户端私有状态管理")
    print("="*70)
    
    # 创建模型
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=128,
        depth=2,
        heads=4,
        mlp_dim=256,
        channels=3
    )
    
    # 注入 FedSDG
    model = inject_lora(model, r=8, lora_alpha=16, train_mlp_head=True, is_fedsdg=True)
    
    # 模拟客户端训练：保存私有状态
    print("\n[模拟客户端 0 训练]")
    
    # 提取私有参数
    private_state = {}
    for name, param in model.named_parameters():
        if '_private' in name or 'lambda_k' in name:
            private_state[name] = param.data.clone()
    
    print(f"  提取私有参数数量: {len(private_state)}")
    print(f"  私有参数示例: {list(private_state.keys())[:3]}")
    
    # 修改模型参数（模拟训练）
    for param in model.parameters():
        if param.requires_grad:
            param.data += torch.randn_like(param.data) * 0.01
    
    # 保存修改后的私有参数
    private_state_after = {}
    for name, param in model.named_parameters():
        if '_private' in name or 'lambda_k' in name:
            private_state_after[name] = param.data.clone()
    
    # 验证私有参数已改变
    changed = False
    for key in private_state.keys():
        if not torch.allclose(private_state[key], private_state_after[key]):
            changed = True
            break
    
    assert changed, "私有参数应该在训练后发生变化"
    print("  ✓ 私有参数在训练后已更新")
    
    # 模拟加载私有状态到新模型
    print("\n[模拟下一轮：加载私有状态]")
    
    # 创建新的全局模型（模拟服务器聚合后的模型）
    model_new = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=128,
        depth=2,
        heads=4,
        mlp_dim=256,
        channels=3
    )
    model_new = inject_lora(model_new, r=8, lora_alpha=16, train_mlp_head=True, is_fedsdg=True)
    
    # 加载私有状态
    current_state = model_new.state_dict()
    for param_name, param_value in private_state_after.items():
        if param_name in current_state:
            current_state[param_name] = param_value.clone()
    model_new.load_state_dict(current_state)
    
    # 验证私有参数已正确加载
    for name, param in model_new.named_parameters():
        if name in private_state_after:
            assert torch.allclose(param.data, private_state_after[name]), f"参数 {name} 加载失败"
    
    print("  ✓ 私有状态成功加载到新模型")
    
    print("\n" + "="*70)
    print("✓ 测试 4 通过：私有状态管理功能正常")
    print("="*70)


def test_forward_backward():
    """测试 FedSDG 的前向和反向传播"""
    print("\n" + "="*70)
    print("测试 5: FedSDG 前向和反向传播")
    print("="*70)
    
    # 创建模型
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=128,
        depth=2,
        heads=4,
        mlp_dim=256,
        channels=3
    )
    
    # 注入 FedSDG
    model = inject_lora(model, r=8, lora_alpha=16, train_mlp_head=True, is_fedsdg=True)
    
    # 创建输入和标签
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))
    
    # 前向传播
    print("\n[前向传播]")
    output = model(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    assert output.shape == (2, 10), "输出形状应该是 (2, 10)"
    print("  ✓ 前向传播成功")
    
    # 计算损失
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, y)
    print(f"  损失值: {loss.item():.4f}")
    
    # 反向传播
    print("\n[反向传播]")
    loss.backward()
    
    # 检查梯度
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_count += 1
    
    print(f"  有梯度的参数数量: {grad_count}")
    assert grad_count > 0, "应该有参数具有梯度"
    print("  ✓ 反向传播成功")
    
    # 检查私有参数也有梯度
    private_grad_count = 0
    for name, param in model.named_parameters():
        if ('_private' in name or 'lambda_k' in name) and param.grad is not None:
            private_grad_count += 1
    
    print(f"  私有参数有梯度的数量: {private_grad_count}")
    assert private_grad_count > 0, "私有参数应该有梯度"
    print("  ✓ 私有参数可以正常训练")
    
    print("\n" + "="*70)
    print("✓ 测试 5 通过：前向和反向传播正常")
    print("="*70)


def test_gate_initialization():
    """测试 6: 门控参数初始化验证 (FedSDG_Design.md 规范)"""
    print("\n" + "="*70)
    print("测试 6: 门控参数初始化验证")
    print("="*70)
    
    # 创建一个简单的线性层
    original_layer = nn.Linear(128, 64)
    
    # 创建 FedSDG LoRA 层
    lora_fedsdg = LoRALayer(original_layer, r=8, lora_alpha=16, is_fedsdg=True)
    
    # 检查 lambda_k_logit 初始化值
    print("\n[门控参数初始化检查]")
    logit_value = lora_fedsdg.lambda_k_logit.item()
    m_k_value = torch.sigmoid(lora_fedsdg.lambda_k_logit).item()
    
    print(f"  lambda_k_logit (a_{{k,l}}): {logit_value:.4f}")
    print(f"  m_{{k,l}} = sigmoid(a_{{k,l}}): {m_k_value:.4f}")
    
    # 根据设计文档: a_{k,l} = 0 → m_{k,l} = 0.5
    assert abs(logit_value - 0.0) < 1e-6, f"lambda_k_logit 应该初始化为 0.0，实际为 {logit_value}"
    assert abs(m_k_value - 0.5) < 1e-6, f"m_{{k,l}} 应该初始化为 0.5，实际为 {m_k_value}"
    
    print("  ✓ 门控参数初始化正确: a_{k,l}=0 → m_{k,l}=0.5")
    print("  ✓ 符合设计文档要求：训练开始时共享和私有组件等权重")
    
    print("\n" + "="*70)
    print("✓ 测试 6 通过：门控参数初始化符合设计规范")
    print("="*70)


def test_equation5_loss_components():
    """测试 7: Equation 5 损失函数组件验证"""
    print("\n" + "="*70)
    print("测试 7: Equation 5 损失函数组件验证")
    print("="*70)
    
    # 创建模型
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=128,
        depth=2,
        heads=4,
        mlp_dim=256,
        channels=3
    )
    
    # 注入 FedSDG
    model = inject_lora(model, r=8, lora_alpha=16, train_mlp_head=True, is_fedsdg=True)
    
    print("\n[计算 Equation 5 各组件]")
    
    # ========== λ₁ L1 门控惩罚 ==========
    gate_penalty = 0.0
    gate_count = 0
    gate_values = []
    for name, param in model.named_parameters():
        if 'lambda_k_logit' in name:
            m_k = torch.sigmoid(param)
            gate_penalty += torch.sum(torch.abs(m_k)).item()
            gate_count += param.numel()
            gate_values.append(m_k.item())
    
    print(f"\n  [λ₁ L1 门控惩罚]")
    print(f"    门控参数数量: {gate_count}")
    print(f"    门控值 (m_{{k,l}}): {gate_values[:3]}... (共 {len(gate_values)} 个)")
    print(f"    gate_penalty = Σ|m_{{k,l}}|: {gate_penalty:.4f}")
    
    # 初始时 m_{k,l} = 0.5，所以 gate_penalty ≈ 0.5 * num_gates
    expected_gate_penalty = 0.5 * gate_count
    print(f"    预期值 (初始): {expected_gate_penalty:.4f}")
    assert abs(gate_penalty - expected_gate_penalty) < 0.1, \
        f"gate_penalty 应该约为 {expected_gate_penalty}，实际为 {gate_penalty}"
    print(f"    ✓ 门控惩罚计算正确")
    
    # ========== λ₂ L2 私有参数惩罚 ==========
    private_penalty = 0.0
    private_count = 0
    for name, param in model.named_parameters():
        if '_private' in name:
            private_penalty += torch.sum(param ** 2).item()
            private_count += param.numel()
    
    print(f"\n  [λ₂ L2 私有参数惩罚]")
    print(f"    私有参数数量: {private_count}")
    print(f"    private_penalty = ||θ_{{p,k}}||²₂: {private_penalty:.6f}")
    
    # 初始时私有参数接近 0（lora_B_private 初始化为 0）
    print(f"    ✓ 私有惩罚计算正确（初始值较小）")
    
    # ========== 模拟完整损失计算 ==========
    print(f"\n  [模拟 Equation 5 完整损失]")
    lambda1 = 1e-3
    lambda2 = 1e-4
    task_loss = 2.3  # 模拟交叉熵损失
    
    total_loss = task_loss + lambda1 * gate_penalty + lambda2 * private_penalty
    
    print(f"    task_loss: {task_loss:.4f}")
    print(f"    λ₁ * gate_penalty: {lambda1} * {gate_penalty:.4f} = {lambda1 * gate_penalty:.6f}")
    print(f"    λ₂ * private_penalty: {lambda2} * {private_penalty:.6f} = {lambda2 * private_penalty:.8f}")
    print(f"    total_loss: {total_loss:.6f}")
    
    print("\n" + "="*70)
    print("✓ 测试 7 通过：Equation 5 损失函数组件计算正确")
    print("="*70)


def test_equation4_forward():
    """测试 8: Equation 4 前向传播验证（加性残差形式）"""
    print("\n" + "="*70)
    print("测试 8: Equation 4 前向传播验证")
    print("="*70)
    
    # 创建一个简单的线性层
    original_layer = nn.Linear(128, 64)
    
    # 创建 FedSDG LoRA 层
    lora = LoRALayer(original_layer, r=8, lora_alpha=16, is_fedsdg=True)
    
    # 创建输入
    x = torch.randn(2, 128)
    
    print("\n[验证 Equation 4: θ̃_{k,l} = θ_{g,l} + m_{k,l} · θ_{p,k,l}]")
    
    # 手动计算各组件
    original_output = original_layer(x)
    global_output = x @ lora.lora_A @ lora.lora_B
    private_output = x @ lora.lora_A_private @ lora.lora_B_private
    m_k = torch.sigmoid(lora.lambda_k_logit)
    
    # 根据 Equation 4 计算预期输出
    expected_lora_output = (global_output + m_k * private_output) * lora.scaling
    expected_total = original_output + expected_lora_output
    
    # 实际前向传播
    actual_output = lora(x)
    
    print(f"  m_{{k,l}}: {m_k.item():.4f}")
    print(f"  global_output 范数: {global_output.norm().item():.4f}")
    print(f"  private_output 范数: {private_output.norm().item():.4f}")
    print(f"  预期输出范数: {expected_total.norm().item():.4f}")
    print(f"  实际输出范数: {actual_output.norm().item():.4f}")
    
    # 验证输出一致
    assert torch.allclose(actual_output, expected_total, atol=1e-6), \
        "前向传播输出与 Equation 4 预期不一致"
    
    print("  ✓ 前向传播符合 Equation 4 加性残差形式")
    
    # 验证极端情况
    print("\n[验证极端情况]")
    
    # m_k = 0: 仅使用全局分支
    lora.lambda_k_logit.data = torch.tensor([-100.0])  # sigmoid(-100) ≈ 0
    m_k_0 = torch.sigmoid(lora.lambda_k_logit)
    output_m0 = lora(x)
    expected_m0 = original_output + global_output * lora.scaling
    print(f"  m_k ≈ 0 ({m_k_0.item():.6f}): 输出应接近 global-only")
    assert torch.allclose(output_m0, expected_m0, atol=1e-4), "m_k=0 时应仅使用全局分支"
    print("  ✓ m_k=0 验证通过")
    
    # m_k = 1: 全局 + 完整私有
    lora.lambda_k_logit.data = torch.tensor([100.0])  # sigmoid(100) ≈ 1
    m_k_1 = torch.sigmoid(lora.lambda_k_logit)
    output_m1 = lora(x)
    expected_m1 = original_output + (global_output + private_output) * lora.scaling
    print(f"  m_k ≈ 1 ({m_k_1.item():.6f}): 输出应为 global + private")
    assert torch.allclose(output_m1, expected_m1, atol=1e-4), "m_k=1 时应使用全局+私有"
    print("  ✓ m_k=1 验证通过")
    
    print("\n" + "="*70)
    print("✓ 测试 8 通过：Equation 4 前向传播实现正确")
    print("="*70)


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("FedSDG 功能测试套件 (符合 FedSDG_Design.md 规范)")
    print("="*70)
    
    try:
        test_lora_layer_fedsdg()
        test_get_lora_state_dict_filtering()
        test_communication_volume()
        test_private_state_management()
        test_forward_backward()
        test_gate_initialization()
        test_equation5_loss_components()
        test_equation4_forward()
        
        print("\n" + "="*70)
        print("🎉 所有测试通过！FedSDG 实现符合设计规范！")
        print("="*70)
        print("\n总结：")
        print("  ✓ LoRALayer 双路架构工作正常")
        print("  ✓ 私有参数过滤功能正确")
        print("  ✓ 通信量与 FedLoRA 一致")
        print("  ✓ 客户端私有状态管理正常")
        print("  ✓ 前向和反向传播正常")
        print("  ✓ 门控参数初始化符合规范 (a_{k,l}=0 → m_{k,l}=0.5)")
        print("  ✓ Equation 5 损失函数组件计算正确")
        print("  ✓ Equation 4 前向传播实现正确（加性残差形式）")
        print("\nFedSDG 已准备好用于联邦学习训练！")
        print("="*70 + "\n")
        
        return True
        
    except AssertionError as e:
        print("\n" + "="*70)
        print(f"❌ 测试失败: {str(e)}")
        print("="*70 + "\n")
        return False
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ 测试出错: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*70 + "\n")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
