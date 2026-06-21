# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation utilities for federated learning.

Provides global and local evaluation functions for dual evaluation mechanism.

GPU Transform Optimization:
- For FEMNIST, uses GPU-accelerated resize (28×28 → 224×224)
- Reduces CPU→GPU bandwidth by ~200x
- Expected speedup: 30-50% for evaluation
"""

import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

from fl.data.gpu_transform import get_gpu_transform, needs_gpu_transform


def test_inference(args, model, test_dataset):
    """
    Global test inference function.
    
    For FedSDG: Disable private branches during global testing by setting
    gate parameters to a very negative value (m_k ≈ 0).
    
    GPU Transform Optimization:
    - For FEMNIST, uses GPU-accelerated resize (28×28 → 224×224)
    - Reduces evaluation time by ~30-50%
    
    Args:
        args: Command line arguments
        model: Global model
        test_dataset: Test dataset
    
    Returns:
        accuracy: Test accuracy
        loss: Test loss
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    nw = getattr(args, 'num_workers', 2)
    # 智能pin_memory设置：PathMNIST禁用，其他数据集启用
    dataset_name = getattr(args, 'dataset', '').lower()
    use_pin_memory = not ('pathmnist' in dataset_name)
    
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False, num_workers=nw, pin_memory=use_pin_memory)
    
    # GPU Transform for FEMNIST (28×28 → 224×224 on GPU)
    dataset_name = getattr(args, 'dataset', '')
    use_gpu_transform = getattr(args, 'use_gpu_transform', True)
    if use_gpu_transform and needs_gpu_transform(dataset_name):
        image_size = getattr(args, 'image_size', 224)
        gpu_transform = get_gpu_transform(size=image_size, device=device)
    else:
        gpu_transform = None

    # FedSDG: Disable private branch during global testing
    # 支持两种门控粒度：fine (lambda_k_logit) 和 coarse (lambda_k_global)
    original_gate_values = {}
    if args.alg == 'fedsdg':
        gate_granularity = getattr(args, 'gate_granularity', 'fine')
        with torch.no_grad():
            for name, param in model.named_parameters():
                if gate_granularity == 'fine' and 'lambda_k_logit' in name:
                    original_gate_values[name] = param.data.clone()
                    param.data.fill_(-100.0)
                elif gate_granularity == 'coarse' and 'lambda_k_global' in name:
                    original_gate_values[name] = param.data.clone()
                    param.data.fill_(-100.0)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # GPU Transform: resize + channel expand + normalize (for FEMNIST)
            if gpu_transform is not None:
                images = gpu_transform(images)

            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    # Restore gate values
    if args.alg == 'fedsdg':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_gate_values:
                    param.data.copy_(original_gate_values[name])

    accuracy = correct/total
    return accuracy, loss


def local_test_inference(args, model, test_dataset, idxs, private_state=None, gpu_transform=None):
    """
    Local personalization test inference function.
    
    Evaluates client performance on their local test set.
    
    Args:
        args: Command line arguments
        model: Model (global or with private parameters loaded)
        test_dataset: Test dataset
        idxs: Local test set indices for this client
        private_state: FedSDG private parameters {param_name: tensor}
        gpu_transform: Optional GPUTransform instance for FEMNIST
    
    Returns:
        accuracy: Local test accuracy
        loss: Local test loss
    """
    # Lazy import to avoid circular import (fl.data.datasets imports fl.utils.paths)
    from fl.data import DatasetSplit
    
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    
    use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    
    nw = getattr(args, 'num_workers', 2)
    
    # 智能pin_memory设置：PathMNIST禁用，其他数据集启用
    dataset_name = getattr(args, 'dataset', '').lower()
    use_pin_memory = not ('pathmnist' in dataset_name)
    
    local_test_loader = DataLoader(
        DatasetSplit(test_dataset, idxs),
        batch_size=128,
        shuffle=False,
        num_workers=min(8, nw),
        pin_memory=use_pin_memory
    )
    
    # FedSDG: Load client private parameters
    # 注意：这里使用 load_state_dict 来正确加载私有参数
    # 因为 model.state_dict() 返回的是副本，直接修改不会生效
    original_state = None
    if args.alg == 'fedsdg' and private_state is not None:
        original_state = {}
        current_state = model.state_dict()
        
        with torch.no_grad():
            for param_name, param_value in private_state.items():
                if param_name in current_state:
                    original_state[param_name] = current_state[param_name].clone()
                    current_state[param_name] = param_value.to(device)
            # 使用 load_state_dict 正确加载修改后的状态
            model.load_state_dict(current_state, strict=False)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(local_test_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # GPU Transform: resize + channel expand + normalize (for FEMNIST)
            if gpu_transform is not None:
                images = gpu_transform(images)
            
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    
    # Restore original state
    if args.alg == 'fedsdg' and original_state is not None:
        with torch.no_grad():
            current_state = model.state_dict()
            for param_name, param_value in original_state.items():
                current_state[param_name] = param_value
            model.load_state_dict(current_state, strict=False)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, loss


def evaluate_local_personalization(args, global_model, test_dataset, user_groups_test, 
                                    local_private_states=None, sample_clients=None):
    """
    Evaluate local personalization performance for all (or sampled) clients.
    
    This is Step B of dual evaluation mechanism:
    - Iterate through clients
    - Load their personalized model state (Global + Private for FedSDG)
    - Test on their local test set
    - Compute average local accuracy and loss
    
    GPU Transform Optimization:
    - For FEMNIST, uses GPU-accelerated resize (28×28 → 224×224)
    - Reduces evaluation time by ~30-50%
    
    Args:
        args: Command line arguments
        global_model: Global model
        test_dataset: Test dataset
        user_groups_test: {client_id: test_indices}
        local_private_states: FedSDG private states {client_id: {param_name: tensor}}
        sample_clients: List of clients to evaluate, None for all
    
    Returns:
        avg_acc: Average local test accuracy
        avg_loss: Average local test loss
        client_results: {client_id: (acc, loss)} detailed results
    """
    global_model.eval()
    
    if sample_clients is None:
        clients_to_eval = list(user_groups_test.keys())
    else:
        clients_to_eval = sample_clients
    
    client_results = {}
    total_acc, total_loss = 0.0, 0.0
    
    use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    
    # GPU Transform for FEMNIST (28×28 → 224×224 on GPU)
    dataset_name = getattr(args, 'dataset', '')
    use_gpu_transform = getattr(args, 'use_gpu_transform', True)
    if use_gpu_transform and needs_gpu_transform(dataset_name):
        image_size = getattr(args, 'image_size', 224)
        gpu_transform = get_gpu_transform(size=image_size, device=device)
    else:
        gpu_transform = None
    
    # 优化：复用单个模型实例，避免每次 deepcopy（性能提升 5-10x）
    # 仅在需要加载私有状态时创建评估模型
    eval_model = None
    if (args.alg in ('fedsdg', 'feddpa', 'local_only', 'fedrep', 'ditto', 'fedsalora', 'pf2lora', 'fedtp', 'fedalt') and 
        local_private_states is not None and 
        len(local_private_states) > 0):
        # 创建一次评估模型，后续复用
        eval_model = copy.deepcopy(global_model)
        eval_model.eval()
        eval_model.to(device)
    
    # 缓存全局状态，避免每次调用 state_dict()
    global_state_cache = global_model.state_dict() if eval_model is not None else None
    
    for client_id in clients_to_eval:
        test_idxs = user_groups_test[client_id]
        
        if len(test_idxs) == 0:
            continue
        
        # Prepare model based on algorithm
        if args.alg == 'fedsdg' and local_private_states is not None and client_id in local_private_states:
            # FedSDG: 复用评估模型，加载私有参数
            private_state = local_private_states[client_id]
            
            with torch.no_grad():
                # 先恢复全局状态，再加载私有参数
                eval_model.load_state_dict(global_state_cache, strict=False)
                current_state = eval_model.state_dict()
                for param_name, param_value in private_state.items():
                    if param_name in current_state:
                        current_state[param_name] = param_value.to(device)
                eval_model.load_state_dict(current_state, strict=False)
            
            acc, loss = local_test_inference(args, eval_model, test_dataset, test_idxs, gpu_transform=gpu_transform)
            
        elif args.alg == 'local_only' and local_private_states is not None and client_id in local_private_states:
            # Local-Only: 复用评估模型，加载 LoRA 参数
            lora_state = local_private_states[client_id]
            
            with torch.no_grad():
                # 先恢复全局状态，再加载 LoRA 参数
                eval_model.load_state_dict(global_state_cache, strict=False)
                current_state = eval_model.state_dict()
                for param_name, param_value in lora_state.items():
                    if param_name in current_state:
                        current_state[param_name] = param_value.to(device)
                eval_model.load_state_dict(current_state, strict=False)
            
            acc, loss = local_test_inference(args, eval_model, test_dataset, test_idxs, gpu_transform=gpu_transform)
        
        elif args.alg == 'fedrep' and local_private_states is not None and client_id in local_private_states:
            # FedRep: 复用评估模型，加载 Head 参数
            # FedRep 的私有状态是 Head 参数（mlp_head, head）
            head_state = local_private_states[client_id]
            
            with torch.no_grad():
                # 先恢复全局状态（Backbone），再加载 Head 参数
                eval_model.load_state_dict(global_state_cache, strict=False)
                current_state = eval_model.state_dict()
                for param_name, param_value in head_state.items():
                    if param_name in current_state:
                        current_state[param_name] = param_value.to(device)
                eval_model.load_state_dict(current_state, strict=False)
            
            acc, loss = local_test_inference(args, eval_model, test_dataset, test_idxs, gpu_transform=gpu_transform)
        
        elif args.alg == 'fedrep' and (local_private_states is None or client_id not in local_private_states):
            # FedRep: 客户端没有 Head 状态，使用全局模型评估
            # 这样可以保证与其他算法（如 FedAvg）的公平比较
            acc, loss = local_test_inference(args, global_model, test_dataset, test_idxs, gpu_transform=gpu_transform)
        
        elif args.alg == 'ditto' and local_private_states is not None and client_id in local_private_states:
            # Ditto: 复用评估模型，加载个性化模型参数
            # Ditto 的私有状态是完整的个性化模型（LoRA + Head）
            personal_state = local_private_states[client_id]
            
            with torch.no_grad():
                # 先恢复全局状态，再加载个性化参数
                eval_model.load_state_dict(global_state_cache, strict=False)
                current_state = eval_model.state_dict()
                for param_name, param_value in personal_state.items():
                    if param_name in current_state:
                        current_state[param_name] = param_value.to(device)
                eval_model.load_state_dict(current_state, strict=False)
            
            acc, loss = local_test_inference(args, eval_model, test_dataset, test_idxs, gpu_transform=gpu_transform)
        
        elif args.alg == 'fedalt' and local_private_states is not None and client_id in local_private_states:
            # FedALT-adapted: 私有状态包含 trained Individual LoRA + gate
            # 评估时加载全局状态 + 私有状态，体现本地适应的个性化
            private_state = local_private_states[client_id]
            
            with torch.no_grad():
                eval_model.load_state_dict(global_state_cache, strict=False)
                current_state = eval_model.state_dict()
                for param_name, param_value in private_state.items():
                    if param_name in current_state:
                        current_state[param_name] = param_value.to(device)
                eval_model.load_state_dict(current_state, strict=False)
            
            acc, loss = local_test_inference(args, eval_model, test_dataset, test_idxs, gpu_transform=gpu_transform)
        
        elif args.alg == 'feddpa' and local_private_states is not None and client_id in local_private_states:
            # FedDPA: 复用评估模型，加载 Private LoRA 参数
            # FedDPA 的私有状态是 Private LoRA 参数（lora_A_private, lora_B_private）
            private_state = local_private_states[client_id]
            
            with torch.no_grad():
                # 先恢复全局状态，再加载 Private 参数
                eval_model.load_state_dict(global_state_cache, strict=False)
                current_state = eval_model.state_dict()
                for param_name, param_value in private_state.items():
                    if param_name in current_state:
                        current_state[param_name] = param_value.to(device)
                eval_model.load_state_dict(current_state, strict=False)
            
            acc, loss = local_test_inference(args, eval_model, test_dataset, test_idxs, gpu_transform=gpu_transform)
        
        elif args.alg in ('fedsalora', 'pf2lora') and local_private_states is not None and client_id in local_private_states:
            # FedSA-LoRA / PF2LoRA: 复用评估模型，加载私有参数
            # FedSA-LoRA: lora_B 参数; PF2LoRA: Private LoRA (A_p, B_p) 参数
            private_state = local_private_states[client_id]
            
            with torch.no_grad():
                # 先恢复全局状态，再加载私有参数
                eval_model.load_state_dict(global_state_cache, strict=False)
                current_state = eval_model.state_dict()
                for param_name, param_value in private_state.items():
                    if param_name in current_state:
                        current_state[param_name] = param_value.to(device)
                eval_model.load_state_dict(current_state, strict=False)
            
            acc, loss = local_test_inference(args, eval_model, test_dataset, test_idxs, gpu_transform=gpu_transform)
        
        elif args.alg == 'fedtp' and local_private_states is not None and client_id in local_private_states:
            # FedTP Phase 2: 复用评估模型，加载私有 LoRA + Head 参数
            # FedTP 的私有状态是 private LoRA + Head 参数（Phase 2 本地微调结果）
            private_state = local_private_states[client_id]
            
            with torch.no_grad():
                # 先恢复全局状态（含 Phase 1 收敛的全局 LoRA），再加载私有参数
                eval_model.load_state_dict(global_state_cache, strict=False)
                current_state = eval_model.state_dict()
                for param_name, param_value in private_state.items():
                    if param_name in current_state:
                        current_state[param_name] = param_value.to(device)
                eval_model.load_state_dict(current_state, strict=False)
            
            acc, loss = local_test_inference(args, eval_model, test_dataset, test_idxs, gpu_transform=gpu_transform)
            
        else:
            # FedAvg / FedLoRA / Ditto(无个性化状态): Use global model directly
            acc, loss = local_test_inference(args, global_model, test_dataset, test_idxs, gpu_transform=gpu_transform)
        
        client_results[client_id] = (acc, loss)
        total_acc += acc
        total_loss += loss
    
    # 清理评估模型：释放到 PyTorch 缓存池（不归还给 CUDA runtime）
    # 注意：不调用 empty_cache()，避免显存被其他进程抢占
    if eval_model is not None:
        del eval_model
    
    num_clients = len(client_results)
    avg_acc = total_acc / num_clients if num_clients > 0 else 0.0
    avg_loss = total_loss / num_clients if num_clients > 0 else 0.0
    
    return avg_acc, avg_loss, client_results


