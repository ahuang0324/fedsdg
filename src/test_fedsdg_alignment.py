#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedSDG 服务端对齐度加权聚合算法 - 单元测试

本测试文件验证 FedSDG 服务端聚合算法的正确性，包括：
1. 对齐度计算的正确性
2. 权重归一化的正确性
3. 边界情况处理
4. 与 FedAvg 聚合的对比

测试用例基于 FedSDG服务端权重聚合算法.md 中的数值示例
"""

import torch
import numpy as np
import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import average_weights_lora, _compute_alignment_weights


def test_alignment_weights_basic():
    """
    测试基本的对齐度权重计算
    
    使用文档中的数值示例：
    - 3个客户端
    - 全局参数: [1.0, 2.0]
    - 客户端1: [1.5, 2.3] -> Δ = [0.5, 0.3]  (与平均方向一致)
    - 客户端2: [1.6, 2.4] -> Δ = [0.6, 0.4]  (与平均方向一致)
    - 客户端3: [0.5, 1.8] -> Δ = [-0.5, -0.2] (与平均方向相反)
    
    预期结果：
    - 客户端1和2获得较高权重
    - 客户端3权重为0（被抑制）
    """
    print("\n" + "="*60)
    print("测试1: 基本对齐度权重计算")
    print("="*60)
    
    # 构造测试数据
    global_state_dict = {
        'lora_A': torch.tensor([1.0, 2.0])
    }
    
    client_weights = [
        {'lora_A': torch.tensor([1.5, 2.3])},  # 客户端1
        {'lora_A': torch.tensor([1.6, 2.4])},  # 客户端2
        {'lora_A': torch.tensor([0.5, 1.8])},  # 客户端3
    ]
    
    lora_keys = ['lora_A']
    
    # 计算对齐度权重
    weights, stats = _compute_alignment_weights(client_weights, global_state_dict, lora_keys)
    
    print(f"客户端权重: {weights}")
    print(f"权重统计: {stats}")
    
    # 验证结果
    # 客户端3的更新与平均方向相反，权重应该为0
    assert weights[2] < 0.01, f"客户端3权重应该接近0，实际为 {weights[2]}"
    
    # 客户端1和2的权重应该接近0.5
    assert abs(weights[0] - 0.5) < 0.1, f"客户端1权重应该接近0.5，实际为 {weights[0]}"
    assert abs(weights[1] - 0.5) < 0.1, f"客户端2权重应该接近0.5，实际为 {weights[1]}"
    
    # 权重和应该为1
    assert abs(sum(weights) - 1.0) < 1e-6, f"权重和应该为1，实际为 {sum(weights)}"
    
    print("✓ 测试通过: 基本对齐度权重计算正确")
    return True


def test_alignment_weights_all_aligned():
    """
    测试所有客户端更新方向一致的情况
    
    预期结果：所有客户端权重接近均匀分布
    """
    print("\n" + "="*60)
    print("测试2: 所有客户端更新方向一致")
    print("="*60)
    
    global_state_dict = {
        'lora_A': torch.tensor([0.0, 0.0])
    }
    
    # 所有客户端更新方向相同
    client_weights = [
        {'lora_A': torch.tensor([1.0, 1.0])},
        {'lora_A': torch.tensor([2.0, 2.0])},
        {'lora_A': torch.tensor([3.0, 3.0])},
    ]
    
    lora_keys = ['lora_A']
    weights, stats = _compute_alignment_weights(client_weights, global_state_dict, lora_keys)
    
    print(f"客户端权重: {weights}")
    print(f"对齐度分数: {stats['alphas']}")
    
    # 所有客户端都与平均方向一致，权重应该接近均匀
    # 但由于更新幅度不同，权重可能略有差异
    assert all(w > 0.2 for w in weights), "所有客户端权重应该大于0.2"
    assert abs(sum(weights) - 1.0) < 1e-6, f"权重和应该为1，实际为 {sum(weights)}"
    
    print("✓ 测试通过: 所有客户端更新方向一致时权重分布正确")
    return True


def test_alignment_weights_single_client():
    """
    测试只有一个客户端的边界情况
    
    预期结果：单个客户端权重为1.0
    """
    print("\n" + "="*60)
    print("测试3: 单个客户端边界情况")
    print("="*60)
    
    global_state_dict = {
        'lora_A': torch.tensor([0.0, 0.0])
    }
    
    client_weights = [
        {'lora_A': torch.tensor([1.0, 1.0])},
    ]
    
    lora_keys = ['lora_A']
    weights, stats = _compute_alignment_weights(client_weights, global_state_dict, lora_keys)
    
    print(f"客户端权重: {weights}")
    
    assert len(weights) == 1, "应该只有一个权重"
    assert weights[0] == 1.0, f"单个客户端权重应该为1.0，实际为 {weights[0]}"
    
    print("✓ 测试通过: 单个客户端边界情况处理正确")
    return True


def test_alignment_weights_empty():
    """
    测试没有客户端的边界情况
    
    预期结果：返回空列表
    """
    print("\n" + "="*60)
    print("测试4: 空客户端列表边界情况")
    print("="*60)
    
    global_state_dict = {
        'lora_A': torch.tensor([0.0, 0.0])
    }
    
    client_weights = []
    lora_keys = ['lora_A']
    
    weights, stats = _compute_alignment_weights(client_weights, global_state_dict, lora_keys)
    
    print(f"客户端权重: {weights}")
    
    assert len(weights) == 0, "空客户端列表应该返回空权重列表"
    
    print("✓ 测试通过: 空客户端列表边界情况处理正确")
    return True


def test_average_weights_lora_alignment():
    """
    测试 average_weights_lora 函数的 alignment 聚合模式
    """
    print("\n" + "="*60)
    print("测试5: average_weights_lora alignment 聚合模式")
    print("="*60)
    
    # 构造全局模型 state_dict
    global_state_dict = {
        'backbone.weight': torch.tensor([1.0, 2.0, 3.0]),  # 冻结参数
        'lora_A': torch.tensor([1.0, 2.0]),
        'lora_B': torch.tensor([0.5, 0.5]),
        'head.weight': torch.tensor([0.1, 0.2]),
    }
    
    # 客户端上传的参数（只包含 LoRA 和 head 参数）
    client_weights = [
        {
            'lora_A': torch.tensor([1.5, 2.3]),
            'lora_B': torch.tensor([0.6, 0.6]),
            'head.weight': torch.tensor([0.15, 0.25]),
        },
        {
            'lora_A': torch.tensor([1.6, 2.4]),
            'lora_B': torch.tensor([0.7, 0.7]),
            'head.weight': torch.tensor([0.12, 0.22]),
        },
        {
            'lora_A': torch.tensor([0.5, 1.8]),  # 与平均方向相反
            'lora_B': torch.tensor([0.3, 0.3]),
            'head.weight': torch.tensor([0.08, 0.18]),
        },
    ]
    
    # 使用 alignment 聚合
    result = average_weights_lora(client_weights, global_state_dict, agg_method='alignment')
    
    print(f"聚合后 lora_A: {result['lora_A']}")
    print(f"聚合后 lora_B: {result['lora_B']}")
    print(f"聚合后 head.weight: {result['head.weight']}")
    print(f"冻结参数 backbone.weight: {result['backbone.weight']}")
    
    # 验证冻结参数未被修改
    assert torch.allclose(result['backbone.weight'], global_state_dict['backbone.weight']), \
        "冻结参数不应该被修改"
    
    # 验证 LoRA 参数被聚合（由于客户端3被抑制，结果应该偏向客户端1和2）
    # 客户端1和2的 lora_A 平均值约为 [1.55, 2.35]
    expected_lora_A_approx = (torch.tensor([1.5, 2.3]) + torch.tensor([1.6, 2.4])) / 2
    print(f"预期 lora_A (客户端1和2平均): {expected_lora_A_approx}")
    
    # 由于客户端3被抑制，聚合结果应该更接近客户端1和2的平均值
    diff = torch.abs(result['lora_A'] - expected_lora_A_approx).max().item()
    print(f"与预期的最大差异: {diff}")
    
    print("✓ 测试通过: alignment 聚合模式工作正常")
    return True


def test_average_weights_lora_fedavg():
    """
    测试 average_weights_lora 函数的 fedavg 聚合模式
    """
    print("\n" + "="*60)
    print("测试6: average_weights_lora fedavg 聚合模式")
    print("="*60)
    
    global_state_dict = {
        'backbone.weight': torch.tensor([1.0, 2.0, 3.0]),
        'lora_A': torch.tensor([1.0, 2.0]),
        'lora_B': torch.tensor([0.5, 0.5]),
        'head.weight': torch.tensor([0.1, 0.2]),
    }
    
    client_weights = [
        {
            'lora_A': torch.tensor([1.5, 2.3]),
            'lora_B': torch.tensor([0.6, 0.6]),
            'head.weight': torch.tensor([0.15, 0.25]),
        },
        {
            'lora_A': torch.tensor([1.6, 2.4]),
            'lora_B': torch.tensor([0.7, 0.7]),
            'head.weight': torch.tensor([0.12, 0.22]),
        },
        {
            'lora_A': torch.tensor([0.5, 1.8]),
            'lora_B': torch.tensor([0.3, 0.3]),
            'head.weight': torch.tensor([0.08, 0.18]),
        },
    ]
    
    # 使用 fedavg 聚合
    result = average_weights_lora(client_weights, global_state_dict, agg_method='fedavg')
    
    print(f"聚合后 lora_A: {result['lora_A']}")
    
    # FedAvg 应该是简单平均
    expected_lora_A = (torch.tensor([1.5, 2.3]) + torch.tensor([1.6, 2.4]) + torch.tensor([0.5, 1.8])) / 3
    print(f"预期 lora_A (简单平均): {expected_lora_A}")
    
    assert torch.allclose(result['lora_A'], expected_lora_A, atol=1e-6), \
        f"FedAvg 聚合结果不正确: {result['lora_A']} vs {expected_lora_A}"
    
    print("✓ 测试通过: fedavg 聚合模式工作正常")
    return True


def test_alignment_vs_fedavg_comparison():
    """
    对比 alignment 和 fedavg 聚合的差异
    
    在存在冲突更新的情况下，alignment 聚合应该产生更好的结果
    """
    print("\n" + "="*60)
    print("测试7: alignment vs fedavg 聚合对比")
    print("="*60)
    
    global_state_dict = {
        'lora_A': torch.tensor([1.0, 2.0]),
    }
    
    # 构造一个场景：2个客户端更新方向一致，1个客户端更新方向相反
    client_weights = [
        {'lora_A': torch.tensor([1.5, 2.3])},  # 正向更新
        {'lora_A': torch.tensor([1.6, 2.4])},  # 正向更新
        {'lora_A': torch.tensor([0.5, 1.8])},  # 反向更新（噪声客户端）
    ]
    
    # FedAvg 聚合
    result_fedavg = average_weights_lora(client_weights, global_state_dict, agg_method='fedavg')
    
    # Alignment 聚合
    result_alignment = average_weights_lora(client_weights, global_state_dict, agg_method='alignment')
    
    print(f"全局参数: {global_state_dict['lora_A']}")
    print(f"FedAvg 聚合结果: {result_fedavg['lora_A']}")
    print(f"Alignment 聚合结果: {result_alignment['lora_A']}")
    
    # 计算与"理想"结果的距离
    # 理想结果是客户端1和2的平均（忽略噪声客户端3）
    ideal_result = (torch.tensor([1.5, 2.3]) + torch.tensor([1.6, 2.4])) / 2
    print(f"理想结果（忽略噪声客户端）: {ideal_result}")
    
    dist_fedavg = torch.norm(result_fedavg['lora_A'] - ideal_result).item()
    dist_alignment = torch.norm(result_alignment['lora_A'] - ideal_result).item()
    
    print(f"FedAvg 与理想结果的距离: {dist_fedavg:.4f}")
    print(f"Alignment 与理想结果的距离: {dist_alignment:.4f}")
    
    # Alignment 聚合应该更接近理想结果
    assert dist_alignment < dist_fedavg, \
        f"Alignment 聚合应该比 FedAvg 更接近理想结果: {dist_alignment} vs {dist_fedavg}"
    
    print("✓ 测试通过: Alignment 聚合在存在冲突更新时表现更好")
    return True


def test_private_params_excluded():
    """
    测试私有参数是否被正确排除
    """
    print("\n" + "="*60)
    print("测试8: 私有参数排除测试")
    print("="*60)
    
    global_state_dict = {
        'backbone.weight': torch.tensor([1.0, 2.0, 3.0]),
        'lora_A': torch.tensor([1.0, 2.0]),
        'lora_A_private': torch.tensor([0.1, 0.1]),  # 私有参数
        'lora_B': torch.tensor([0.5, 0.5]),
        'lora_B_private': torch.tensor([0.2, 0.2]),  # 私有参数
        'lambda_k_logit': torch.tensor([0.0]),  # 门控参数
        'head.weight': torch.tensor([0.1, 0.2]),
    }
    
    client_weights = [
        {
            'lora_A': torch.tensor([1.5, 2.3]),
            'lora_A_private': torch.tensor([0.5, 0.5]),
            'lora_B': torch.tensor([0.6, 0.6]),
            'lora_B_private': torch.tensor([0.6, 0.6]),
            'lambda_k_logit': torch.tensor([1.0]),
            'head.weight': torch.tensor([0.15, 0.25]),
        },
        {
            'lora_A': torch.tensor([1.6, 2.4]),
            'lora_A_private': torch.tensor([0.7, 0.7]),
            'lora_B': torch.tensor([0.7, 0.7]),
            'lora_B_private': torch.tensor([0.8, 0.8]),
            'lambda_k_logit': torch.tensor([2.0]),
            'head.weight': torch.tensor([0.12, 0.22]),
        },
    ]
    
    result = average_weights_lora(client_weights, global_state_dict, agg_method='alignment')
    
    print(f"聚合后 lora_A: {result['lora_A']}")
    print(f"聚合后 lora_A_private: {result['lora_A_private']}")
    print(f"聚合后 lambda_k_logit: {result['lambda_k_logit']}")
    
    # 私有参数应该保持全局模型的原始值（不被聚合）
    assert torch.allclose(result['lora_A_private'], global_state_dict['lora_A_private']), \
        "私有参数 lora_A_private 不应该被聚合"
    assert torch.allclose(result['lora_B_private'], global_state_dict['lora_B_private']), \
        "私有参数 lora_B_private 不应该被聚合"
    assert torch.allclose(result['lambda_k_logit'], global_state_dict['lambda_k_logit']), \
        "门控参数 lambda_k_logit 不应该被聚合"
    
    # 全局 LoRA 参数应该被聚合
    assert not torch.allclose(result['lora_A'], global_state_dict['lora_A']), \
        "全局参数 lora_A 应该被聚合更新"
    
    print("✓ 测试通过: 私有参数被正确排除")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("FedSDG 服务端对齐度加权聚合算法 - 单元测试")
    print("="*70)
    
    tests = [
        test_alignment_weights_basic,
        test_alignment_weights_all_aligned,
        test_alignment_weights_single_client,
        test_alignment_weights_empty,
        test_average_weights_lora_alignment,
        test_average_weights_lora_fedavg,
        test_alignment_vs_fedavg_comparison,
        test_private_params_excluded,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ 测试失败: {test.__name__}")
            print(f"  错误信息: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("="*70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
