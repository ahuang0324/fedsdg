# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local training module for federated learning clients.

Implements local training logic for:
- FedAvg: Standard SGD training
- FedLoRA: LoRA parameter training
- FedSDG: Dual-path training with gate regularization
- FedProx (fedprox_avg): FedAvg + proximal term
- FedProx+LoRA (fedprox_lora): FedLoRA + proximal term
- FedRep: Two-stage training (backbone + head)

GPU Transform Optimization:
- For FEMNIST, uses GPU-accelerated resize (28×28 → 224×224)
- Reduces CPU→GPU bandwidth by ~200x
- Expected speedup: 30-50% per training round
"""

import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from fl.data import DatasetSplit
from fl.data.gpu_transform import get_gpu_transform, needs_gpu_transform
from ..models.lora import get_lora_state_dict, _is_head_param
from ..models.lora_dpa import get_dpa_state_dict
from ..algorithms.fedrep import fedrep_update_weights
from ..algorithms.ditto import ditto_update_weights
from ..utils.console_logger import cprint


class LocalUpdate(object):
    """
    Local training handler for federated learning clients.
    
    Handles local training, validation, and inference for each client.
    Supports FedAvg, FedLoRA, FedSDG, FedRep algorithms.
    """
    
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        
        # CUDA error-checking configuration
        if use_cuda:
            # CUDA_LAUNCH_BLOCKING can help diagnose device-side errors, but may reduce performance.
            if os.environ.get('CUDA_LAUNCH_BLOCKING', '0') == '1':
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            # 注意：不调用 torch.cuda.empty_cache()，避免将缓存显存归还给 CUDA runtime
            # 多任务共享 GPU 时，empty_cache() 会导致显存被其他进程抢占
            pass
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        # GPU Transform for FEMNIST (28×28 → 224×224 on GPU)
        dataset_name = getattr(args, 'dataset', '')
        use_gpu_transform = getattr(args, 'use_gpu_transform', True)
        if use_gpu_transform and needs_gpu_transform(dataset_name):
            image_size = getattr(args, 'image_size', 224)
            self.gpu_transform = get_gpu_transform(size=image_size, device=self.device)
        else:
            self.gpu_transform = None

    def train_val_test(self, dataset, idxs):
        """
        Split client data into train and validation sets.
        
        Split: 90% train, 10% val (for future extension)
        Note: testloader is kept as None for backward compatibility
        """
        idxs = list(idxs)
        np.random.shuffle(idxs)
        # Split: 90% train, 10% val
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_val = idxs[int(0.9*len(idxs)):]

        # num_workers / prefetch_factor：由 training.num_workers、training.prefetch_factor 配置
        # 联邦学习中频繁创建 DataLoader，过多 worker 可能引发多进程段错误，可按需调整
        num_workers = getattr(self.args, 'num_workers', 2)
        prefetch = getattr(self.args, 'prefetch_factor', 2)
        # cache_in_memory 模式: 数据已在内存中，多进程 worker 会触发大 Tensor 的
        # copy-on-write，反而更慢。强制 num_workers=0 使用主进程加载。
        cache_in_memory = getattr(self.args, 'cache_in_memory', False)
        if cache_in_memory:
            num_workers = 0
        # persistent_workers: 避免每个 epoch 结束后重新 fork workers
        # LMDB 数据集需禁用，避免多进程文件锁冲突
        use_lmdb = getattr(self.args, 'use_lmdb', False)
        use_persistent = (num_workers > 0) and (not use_lmdb)

        # 智能pin_memory设置：PathMNIST禁用，其他数据集启用
        dataset_name = getattr(self.args, 'dataset', '').lower()
        use_pin_memory = not ('pathmnist' in dataset_name)
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True,
                                 num_workers=num_workers, pin_memory=use_pin_memory,
                                 prefetch_factor=prefetch if num_workers > 0 else None,
                                 persistent_workers=use_persistent)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=max(1, int(len(idxs_val)/10)), shuffle=False,
                                 num_workers=min(8, num_workers), pin_memory=use_pin_memory,
                                 persistent_workers=False)
        # testloader is None, kept for future extension
        testloader = None
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, personal_state=None):
        """
        Client local training function.
        
        FedSDG Loss (Equation 5 from FedSDG_Design.md):
        Loss = (1/|B|) Σ ℓ(f(x), y) + λ₁ Σ|m_{k,l}| + λ₂ ||θ_{p,k}||²₂
        
        FedProx Loss:
        Loss = L_task + (μ/2) * ||w - w_global||²
        
        Args:
            model: Model to train
            global_round: Current global round number
            personal_state: Personal model state dict (for Ditto only)
        
        Returns:
            For FedSDG: (public_w, private_w, avg_loss, train_metrics)
            For FedRep: (backbone_w, head_w, avg_loss, train_metrics)
            For Ditto: (global_w, personal_w, avg_loss, train_metrics)
            For others: (state_dict, avg_loss, train_metrics)
            
            train_metrics: dict containing:
                - train_acc: Training accuracy (last epoch)
                - task_loss: Task loss (last epoch avg)
                - reg_loss: Regularization loss (FedSDG/Ditto)
                - lambda_values: Gate values list (FedSDG only)
        """
        # FedRep: 调用独立的两阶段训练函数
        if self.args.alg == 'fedrep':
            return fedrep_update_weights(
                args=self.args,
                model=model,
                trainloader=self.trainloader,
                device=self.device,
                global_round=global_round,
                gpu_transform=self.gpu_transform,
            )
        
        # Ditto: 调用独立的两阶段训练函数
        if self.args.alg == 'ditto':
            return ditto_update_weights(
                args=self.args,
                model=model,
                trainloader=self.trainloader,
                device=self.device,
                global_round=global_round,
                personal_state=personal_state,
                gpu_transform=self.gpu_transform,
                verbose=self.args.verbose,
            )
        
        model.train()
        epoch_loss = []
        epoch_task_loss = []  # : 任务损失
        epoch_reg_loss = []   # : 正则化损失 (FedSDG)
        epoch_gate_penalty = []  # : 门控惩罚 (FedSDG，未乘以 lambda1)
        epoch_private_penalty = []  # : 私有参数惩罚 (FedSDG，未乘以 lambda2)
        
        # 训练准确率统计 (最后一个 epoch)
        last_epoch_correct = 0
        last_epoch_total = 0

        # FedProx: 保存全局模型参数副本（用于计算 proximal term）
        global_params_fedprox = None
        if self.args.alg == 'fedprox_avg':
            # FedProx (基于 FedAvg): 保存所有可训练参数
            global_params_fedprox = {name: param.clone().detach() 
                           for name, param in model.named_parameters() 
                           if param.requires_grad}
        elif self.args.alg == 'fedprox_lora':
            # FedProx+LoRA: 只保存会被聚合的参数（LoRA + head）
            # 与聚合逻辑保持一致：只对聚合参数计算 proximal term
            global_params_fedprox = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 只保存 LoRA 参数和分类头（与 get_lora_state_dict 逻辑一致）
                    if 'lora_' in name or 'mlp_head' in name or 'head' in name:
                        # 排除私有参数和门控参数（fedprox_lora 不使用，但为了安全）
                        if '_private' not in name and 'lambda_k' not in name:
                            global_params_fedprox[name] = param.clone().detach()

        # Optimizer configuration
        if self.args.alg == 'fedsdg':
            # FedSDG 配置：一次性获取所有需要的配置
            fix_gate = getattr(self.args, 'fix_gate', False)
            head_mode = getattr(self.args, 'head_mode', 'global')
            lr_head = getattr(self.args, 'lr_head', None) or self.args.lr
            gate_granularity = getattr(self.args, 'gate_granularity', 'fine')
            
            gate_params = []
            da_scale_params = []
            private_params = []
            head_params = []
            global_lora_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 门控参数识别（支持两种粒度）
                    is_gate_param = False
                    if gate_granularity == 'fine' and 'lambda_k_logit' in name:
                        is_gate_param = True
                    elif gate_granularity == 'coarse' and name == 'fedsdg_global_gate.lambda_k_global':
                        is_gate_param = True
                    
                    if is_gate_param:
                        if not fix_gate:
                            # 仅在可学习模式下添加到优化器
                            gate_params.append(param)
                        # 固定模式下，门控参数不参与优化
                    elif 'da_scale_logit' in name:
                        # Global DA Scale：独立参数组，使用独立学习率
                        # 参数名形如: fedsdg_global_da_scale.da_scale_logit
                        da_scale_params.append(param)
                    elif '_private' in name:
                        private_params.append(param)
                    elif _is_head_param(name):
                        # 分类头参数：支持独立学习率
                        head_params.append(param)
                    else:
                        global_lora_params.append(param)
            
            # 构建参数组
            # 公有参数 (global_lora, head) 使用 weight_decay=1e-4
            # 私有参数和门控参数保持 weight_decay=0（由自定义正则化 λ₁·gate + λ₂·private 控制）
            param_groups = [
                {'params': global_lora_params, 'lr': self.args.lr, 'weight_decay': 1e-4},
                {'params': private_params, 'lr': self.args.lr, 'weight_decay': 0},
                {'params': head_params, 'lr': lr_head, 'weight_decay': 1e-4},
            ]
            
            # 仅在可学习模式下添加门控参数组
            if not fix_gate and len(gate_params) > 0:
                param_groups.append({'params': gate_params, 'lr': self.args.lr_gate, 'weight_decay': 0})
            
            # DA Scale 独立参数组
            if len(da_scale_params) > 0:
                lr_da_scale = getattr(self.args, 'lr_da_scale', 0.01)
                param_groups.append({'params': da_scale_params, 'lr': lr_da_scale, 'weight_decay': 0})
            
            # 这里的优化器采取了硬编码，强制了fedsdg必须使用adam
            optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))
            
            # 优化：缓存门控参数和私有参数的引用，避免每个batch都遍历named_parameters()
            # 复用优化器配置时已经收集的参数列表
            gate_params_refs = gate_params if not fix_gate else []
            private_params_refs = private_params
            head_params_refs = head_params  # 缓存 head 参数引用，用于验收检查
            
            # 缓存正则化相关配置，避免内层 batch 循环中重复 getattr
            gate_penalty_type = getattr(self.args, 'gate_penalty_type', 'unilateral')
            gate_warmup_rounds = getattr(self.args, 'gate_warmup_rounds', 0)
            in_warmup = global_round < gate_warmup_rounds
            lambda1 = 0.0 if (fix_gate or in_warmup) else self.args.lambda1
            lambda2 = self.args.lambda2
        
        elif self.args.alg == 'fedalt':
            # FedALT-adapted: 反转 FedSDG 的训练角色
            # - Public 分支 (lora_A/B): Global LoRA, 冻结（聚合后的 RoW 参考）
            # - Private 分支 (lora_A/B_private): Individual LoRA, 训练 + 上传聚合
            # - Gate (lambda_k): Mixer, 训练, 本地保留
            #
            # 关键: Individual LoRA 每轮从聚合结果初始化（不跨轮保持本地状态）
            # 这确保了联邦知识传递，类似 FedAvg 每轮从全局模型出发
            fix_gate = getattr(self.args, 'fix_gate', False)
            head_mode = getattr(self.args, 'head_mode', 'global')
            lr_head = getattr(self.args, 'lr_head', None) or self.args.lr
            gate_granularity = getattr(self.args, 'gate_granularity', 'fine')
            
            # Step 0: 从聚合的 Public LoRA 初始化 Individual LoRA
            # lora_A/lora_B (aggregated) → lora_A_private/lora_B_private
            # 确保 Individual 每轮从联邦共识出发，而非延续旧的本地版本
            with torch.no_grad():
                param_dict = dict(model.named_parameters())
                for name, param in param_dict.items():
                    if '_private' in name and 'lora_' in name:
                        public_name = name.replace('_private', '')
                        if public_name in param_dict:
                            param.copy_(param_dict[public_name].data)
            
            gate_params = []
            private_params = []  # Individual LoRA
            head_params = []
            frozen_public_params = []  # Public LoRA (to freeze)
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # 门控参数识别
                is_gate_param = False
                if gate_granularity == 'fine' and 'lambda_k_logit' in name:
                    is_gate_param = True
                elif gate_granularity == 'coarse' and name == 'fedsdg_global_gate.lambda_k_global':
                    is_gate_param = True
                
                if is_gate_param:
                    if not fix_gate:
                        gate_params.append(param)
                elif '_private' in name:
                    # Individual LoRA → 训练
                    private_params.append(param)
                elif _is_head_param(name):
                    head_params.append(param)
                elif 'lora_' in name:
                    # Public LoRA (lora_A/lora_B without _private) → 冻结
                    param.requires_grad = False
                    frozen_public_params.append(name)
            
            # 构建参数组
            param_groups = [
                {'params': private_params, 'lr': self.args.lr, 'weight_decay': 0},
                {'params': head_params, 'lr': lr_head, 'weight_decay': 1e-4},
            ]
            if not fix_gate and len(gate_params) > 0:
                param_groups.append({'params': gate_params, 'lr': self.args.lr_gate, 'weight_decay': 0})
            
            optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))
            
            # 缓存参数引用
            gate_params_refs = gate_params if not fix_gate else []
            private_params_refs = private_params
            head_params_refs = head_params
            
            # 缓存正则化配置
            gate_penalty_type = getattr(self.args, 'gate_penalty_type', 'unilateral')
            gate_warmup_rounds = getattr(self.args, 'gate_warmup_rounds', 0)
            in_warmup = global_round < gate_warmup_rounds
            lambda1 = 0.0 if (fix_gate or in_warmup) else self.args.lambda1
            lambda2 = self.args.lambda2
        
        elif self.args.alg in ('feddpa', 'pf2lora'):
            # FedDPA / PF2LoRA: Global + Private + Head 三组参数，支持不同学习率
            # 获取各参数组的学习率
            lr_head = getattr(self.args, 'lr_head', None) or self.args.lr
            lr_private = getattr(self.args, 'lr_private', None) or self.args.lr
            
            global_params = []
            private_params = []
            head_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if '_private' in name:
                        private_params.append(param)
                    elif _is_head_param(name):
                        head_params.append(param)
                    else:
                        global_params.append(param)
            
            # 构建参数组，支持不同学习率
            # 公有参数 (global, head) 使用 weight_decay=1e-4，私有参数保持 0
            param_groups = [
                {'params': global_params, 'lr': self.args.lr, 'weight_decay': 1e-4},
                {'params': private_params, 'lr': lr_private, 'weight_decay': 0},
                {'params': head_params, 'lr': lr_head, 'weight_decay': 1e-4},
            ]
            
            # FedDPA / PF2LoRA 使用 Adam 优化器
            optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))
            
            # FedDPA / PF2LoRA 不需要门控参数引用
            gate_params_refs = []
            private_params_refs = private_params
        
        elif self.args.alg in ('fedlora', 'fedprox_lora', 'local_only', 'fedsalora', 'fedtp', 'lorafair'):
            # FedLoRA / FedSA-LoRA: 支持head和backbone不同学习率
            # 获取 head 学习率（如果未设置，使用主学习率）
            lr_head = getattr(self.args, 'lr_head', None) or self.args.lr
            
            # 参数分组：LoRA参数 + Head参数
            lora_params = []
            head_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if _is_head_param(name):
                        head_params.append(param)
                    else:
                        # FedLoRA主要训练LoRA参数
                        lora_params.append(param)
            
            # 构建参数组，支持不同学习率
            param_groups = [
                {'params': lora_params, 'lr': self.args.lr},
                {'params': head_params, 'lr': lr_head},
            ]
            
            # 根据配置选择优化器类型，兼容SGD和Adam
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(param_groups, lr=self.args.lr,
                                            momentum=self.args.momentum)
            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
            else:
                raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
            
            # FedLoRA 不需要缓存参数引用
            gate_params_refs = []
            private_params_refs = []
            
        else:  # fedavg / fedprox_avg 等：支持 head 独立学习率
            # 非 FedSDG/FedDPA/FedLoRA 算法：不需要缓存参数引用
            gate_params_refs = []
            private_params_refs = []
            # 获取 head 学习率（如果未设置，使用主学习率）
            lr_head = getattr(self.args, 'lr_head', None) or self.args.lr
            
            # 参数分组：backbone + head
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if _is_head_param(name):
                        head_params.append(param)
                    else:
                        backbone_params.append(param)
            
            param_groups = [
                {'params': backbone_params, 'lr': self.args.lr},
                {'params': head_params, 'lr': lr_head},
            ]
            
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(param_groups, lr=self.args.lr,
                                            momentum=self.args.momentum)
            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)

        is_last_epoch = False
        for iter in range(self.args.local_ep):  # 本地训练的轮数 当前是串行训练
            is_last_epoch = (iter == self.args.local_ep - 1)
            batch_loss = []
            batch_task_loss = []
            batch_reg_loss = []
            batch_gate_penalty = []  # : 门控惩罚 (未乘以 lambda1)
            batch_private_penalty = []  # : 私有参数惩罚 (未乘以 lambda2)
            
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                # PathMNIST 兼容性修复：使用阻塞式数据传输避免 CUDA 错误
                dataset_name = getattr(self.args, 'dataset', '').lower()
                use_blocking = 'pathmnist' in dataset_name
                
                # 数据验证 - 检查数据格式和标签范围
                if batch_idx == 0 and iter == 0:  # 只在第一个batch第一次迭代时检查
                    assert images.dtype in (torch.float16, torch.float32, torch.bfloat16), f"Invalid image dtype: {images.dtype}"
                    assert labels.dtype == torch.long, f"Invalid label dtype: {labels.dtype}, expected torch.long"
                    assert labels.ndim == 1, f"Invalid label ndim: {labels.ndim}, expected 1"
                    assert 0 <= labels.min().item(), f"Invalid min label: {labels.min().item()}"
                    num_classes = getattr(self.args, 'num_classes', 1000)
                    assert labels.max().item() < num_classes, f"Invalid max label: {labels.max().item()}, expected < {num_classes} for {dataset_name}"
                    
                    # 检查图像是否有限值
                    if not torch.isfinite(images).all():
                        raise RuntimeError(f"Images contain NaN/Inf values")
                    if getattr(self.args, 'verbose', 0) >= 2:
                        print(f"[Data Check] {dataset_name}: images={tuple(images.shape)}, labels={tuple(labels.shape)}, label_range=({labels.min().item()}, {labels.max().item()})")
                
                if use_blocking:
                    images, labels = images.to(self.device), labels.to(self.device)
                else:
                    images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                # GPU Transform: resize + channel expand + normalize (for FEMNIST)
                if self.gpu_transform is not None:
                    images = self.gpu_transform(images)
                    # 检查 transform 后的图像是否有效
                    if not torch.isfinite(images).all():
                        print(f"  [ERROR] Invalid images after GPU transform at Round {global_round}, Epoch {iter}, Batch {batch_idx}")
                        continue

                model.zero_grad()
                
                try:
                    logits = model(images)
                except RuntimeError as e:
                    if 'CUDA' in str(e) or 'cuda' in str(e):
                        print(f"  [ERROR] CUDA error during forward pass at Round {global_round}, Epoch {iter}, Batch {batch_idx}: {e}")
                        # 跳过这个 batch（不调用 empty_cache()，避免显存被其他进程抢占）
                        continue
                    else:
                        raise  # 重新抛出非 CUDA 错误
                
                # 检查 logits 是否有效
                if not torch.isfinite(logits).all():
                    print(f"  [ERROR] Invalid logits detected at Round {global_round}, Epoch {iter}, Batch {batch_idx}")
                    continue
                
                # Task loss
                task_loss = self.criterion(logits, labels)
                
                # 检查损失是否有效
                if not torch.isfinite(task_loss).all():
                    print(f"  [ERROR] Invalid task_loss detected at Round {global_round}, Epoch {iter}, Batch {batch_idx}")
                    print(f"    task_loss: {task_loss.item()}")
                    print(f"    logits range: {logits.min().item():.4f} - {logits.max().item():.4f}")
                    print(f"    labels range: {labels.min().item()} - {labels.max().item()}")
                    print(f"    logits finite: {torch.isfinite(logits).all().item()}")
                    print(f"    labels finite: {torch.isfinite(labels).all().item()}")
                    continue
                
                reg_loss = torch.tensor(0.0, device=self.device)
                
                if self.args.alg in ('fedsdg', 'fedalt'):
                    # Gate sparsity penalty (λ₁) - 支持两种粒度
                    # 使用缓存的 gate_params_refs（fine 和 coarse 模式均在优化器配置阶段收集）
                    # fix_gate、gate_granularity 等已在外层缓存，不再重复 getattr
                    gate_penalty = torch.tensor(0.0, device=self.device)
                    if not fix_gate and gate_params_refs:
                        for param in gate_params_refs:
                            m_k = torch.sigmoid(param)
                            if gate_penalty_type == 'bilateral':
                                gate_penalty = gate_penalty + torch.sum(torch.min(m_k, 1 - m_k))
                            else:
                                gate_penalty = gate_penalty + torch.sum(torch.abs(m_k))
                    
                    # Private parameter L2 regularization (λ₂)
                    # 优化：使用缓存的参数引用，避免每次batch都遍历named_parameters()
                    private_penalty = torch.tensor(0.0, device=self.device)
                    if private_params_refs:
                        for param in private_params_refs:
                            private_penalty = private_penalty + torch.sum(param ** 2)
                    
                    # Combined loss（lambda1/lambda2/in_warmup 已在外层缓存）
                    reg_loss = lambda1 * gate_penalty + lambda2 * private_penalty
                    loss = task_loss + reg_loss
                elif self.args.alg in ('feddpa', 'pf2lora'):
                    # FedDPA / PF2LoRA: 仅 task loss，无额外正则化
                    # 与 FedSDG 的区别：无门控惩罚，无私有参数正则化
                    loss = task_loss
                elif self.args.alg == 'fedprox_avg':
                    # FedProx (基于 FedAvg): 对所有可训练参数计算 proximal term
                    # Loss = L_task + (μ/2) * ||w - w_global||²
                    proximal_term = torch.tensor(0.0, device=self.device)
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in global_params_fedprox:
                            proximal_term += ((param - global_params_fedprox[name]) ** 2).sum()
                    
                    reg_loss = (self.args.mu / 2) * proximal_term
                    loss = task_loss + reg_loss
                elif self.args.alg == 'fedprox_lora':
                    # FedProx+LoRA: 只对会被聚合的参数（LoRA + head）计算 proximal term
                    # Loss = L_task + (μ/2) * ||w_lora - w_lora_global||²
                    # 与聚合逻辑保持一致：只约束会被聚合的参数
                    proximal_term = torch.tensor(0.0, device=self.device)
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in global_params_fedprox:
                            # global_params_fedprox 已经只包含 LoRA + head
                            proximal_term += ((param - global_params_fedprox[name]) ** 2).sum()
                    
                    reg_loss = (self.args.mu / 2) * proximal_term
                    loss = task_loss + reg_loss
                else:
                    loss = task_loss
                
                # 检查 loss 是否为 NaN/Inf
                if not torch.isfinite(loss):
                    print(f"  [ERROR] Invalid loss detected: {loss.item()} at Round {global_round}, Epoch {iter}, Batch {batch_idx}")
                    print(f"    Task loss: {task_loss.item()}, Reg loss: {reg_loss.item()}")
                    # 跳过这个 batch
                    continue
                
                loss.backward()
                
                # FedSDG/FedDPA/PF2LoRA: Gradient clipping
                if self.args.alg in ('fedsdg', 'feddpa', 'pf2lora', 'fedalt') and getattr(self.args, 'grad_clip', 0) > 0:
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
                        # 检查裁剪后的梯度范数
                        if not torch.isfinite(grad_norm):
                            print(f"  [ERROR] Invalid grad_norm after clipping: {grad_norm} at Round {global_round}, Epoch {iter}, Batch {batch_idx}")
                            model.zero_grad()
                            continue
                    except RuntimeError as e:
                        print(f"  [ERROR] Gradient clipping failed: {e} at Round {global_round}, Epoch {iter}, Batch {batch_idx}")
                        model.zero_grad()
                        continue
                
                # Gate Warmup 日志（每轮首个 batch 输出一次）
                if self.args.alg == 'fedsdg' and batch_idx == 0 and iter == 0 and in_warmup:
                    print(f"  [Gate Warmup] Round {global_round}/{gate_warmup_rounds}: λ₁=0 (warmup active)")
                
                # Optional gradient diagnostics for gate and da_scale parameters.
                if self.args.alg == 'fedsdg' and batch_idx == 0 and iter == 0 and getattr(self.args, 'verbose', 0) >= 2:
                    _found_gate, _found_scale = False, False
                    for name, param in model.named_parameters():
                        if not _found_gate and 'lambda_k_logit' in name:
                            grad_val = param.grad.item() if param.grad is not None else 0.0
                            print(f"  [Gradient Diagnostics] {name}: grad={grad_val:.6f}, value={param.data.item():.4f}")
                            _found_gate = True
                        elif not _found_scale and 'da_scale_logit' in name:
                            grad_val = param.grad.item() if param.grad is not None else 0.0
                            alpha_val = torch.exp(param.data.clamp(-2.0, 3.0)).item()
                            print(f"  [Gradient Diagnostics] {name}: grad={grad_val:.6f}, logit={param.data.item():.4f}, α={alpha_val:.4f}")
                            _found_scale = True
                        if _found_gate and _found_scale:
                            break
                
                optimizer.step()
                
                # 最后一个 epoch 时收集训练准确率（训练模式下）
                if is_last_epoch:
                    with torch.no_grad():
                        _, pred = torch.max(logits, 1)
                        last_epoch_correct += (pred == labels).sum().item()
                        last_epoch_total += labels.size(0)

                if self.args.verbose and batch_idx == 0:
                    print('| Global Round : {} | Local Epoch : {} | Starting training...'.format(
                        global_round, iter))
                batch_loss.append(loss.item())
                batch_task_loss.append(task_loss.item())
                batch_reg_loss.append(reg_loss.item())
                
                # FedSDG: 记录分别的惩罚值（未乘以 lambda）
                if self.args.alg == 'fedsdg':
                    batch_gate_penalty.append(gate_penalty.item())
                    batch_private_penalty.append(private_penalty.item())
                else:
                    batch_gate_penalty.append(0.0)
                    batch_private_penalty.append(0.0)
                
            # 每 epoch 末尾检查参数是否有 NaN/Inf（低频检查，避免每 batch 同步开销）
            for name, param in model.named_parameters():
                if not torch.isfinite(param.data).all():
                    print(f"  [CRITICAL] Invalid parameter detected in {name} after epoch {iter}, Round {global_round}")
                    print(f"  [CRITICAL] Model parameters contain NaN/Inf. Training may be unstable.")
                    break
            
            # 防止除以零：当所有批次都因错误（如 CUDA OOM）被跳过时
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                epoch_task_loss.append(sum(batch_task_loss)/len(batch_task_loss))
                epoch_reg_loss.append(sum(batch_reg_loss)/len(batch_reg_loss))
                epoch_gate_penalty.append(sum(batch_gate_penalty)/len(batch_gate_penalty))
                epoch_private_penalty.append(sum(batch_private_penalty)/len(batch_private_penalty))
            else:
                print(f"  [WARNING] No valid batches in epoch {iter} (all batches skipped due to errors)")

        # 计算训练指标（防止除以零）
        if len(epoch_loss) == 0:
            raise RuntimeError(
                f"[Client {self.client_idx}] All epochs failed - no valid batches were processed. "
                f"This is likely due to CUDA OOM or other critical errors. "
                f"Try reducing batch size (current: {self.args.local_bs}) or model complexity."
            )
        
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        avg_task_loss = sum(epoch_task_loss) / len(epoch_task_loss)
        avg_reg_loss = sum(epoch_reg_loss) / len(epoch_reg_loss)
        train_acc = last_epoch_correct / last_epoch_total if last_epoch_total > 0 else 0.0
        
        # 构建训练指标字典
        train_metrics = {
            'train_acc': train_acc,
            'task_loss': avg_task_loss,
        }
        
        # FedSDG / FedALT 特有指标
        if self.args.alg in ('fedsdg', 'fedalt'):
            # 计算平均惩罚值
            avg_gate_penalty = sum(epoch_gate_penalty) / len(epoch_gate_penalty) if epoch_gate_penalty else 0.0
            avg_private_penalty = sum(epoch_private_penalty) / len(epoch_private_penalty) if epoch_private_penalty else 0.0
            
            # 获取 lambda 值
            fix_gate = getattr(self.args, 'fix_gate', False)
            lambda1 = self.args.lambda1 if not fix_gate else 0.0
            lambda2 = self.args.lambda2
            
            # 记录正则化损失（保持向后兼容）
            train_metrics['reg_loss'] = avg_reg_loss
            # 分别记录门控和私有参数的正则化损失（乘以 lambda 后的最终值）
            train_metrics['reg_loss_gate'] = lambda1 * avg_gate_penalty
            train_metrics['reg_loss_private'] = lambda2 * avg_private_penalty
            
            # 收集门控值 - 支持两种粒度
            lambda_values = []
            gate_granularity = getattr(self.args, 'gate_granularity', 'fine')
            
            if gate_granularity == 'fine':
                # 细粒度：收集所有层的门控值
                for name, param in model.named_parameters():
                    if 'lambda_k_logit' in name:
                        lambda_values.append(torch.sigmoid(param).item())
            else:
                # 粗粒度：收集单个全局门控值
                for name, param in model.named_parameters():
                    if name == 'fedsdg_global_gate.lambda_k_global':
                        lambda_values.append(torch.sigmoid(param).item())
                        break
            
            train_metrics['lambda_values'] = lambda_values
        
        # Return appropriate state_dict with train_metrics
        if self.args.alg == 'fedsdg':
            # FedSDG: 返回 (public_w, private_w, loss, train_metrics)
            # public_w 用于服务器聚合，private_w 用于客户端本地保存
            head_mode = getattr(self.args, 'head_mode', 'global')
            public_w, private_w = get_lora_state_dict(model, include_private=True, head_mode=head_mode)
            
            # ========================================
            # 验收检查（仅在第一轮输出一次，使用类变量控制）
            # ========================================
            if global_round == 0 and not getattr(LocalUpdate, '_head_mode_verified', False):
                LocalUpdate._head_mode_verified = True
                self._verify_head_mode_params(public_w, private_w, head_mode, head_params_refs)
            
            return public_w, private_w, avg_loss, train_metrics
        elif self.args.alg == 'fedalt':
            # FedALT-adapted: 返回 (public_w, private_w, loss, train_metrics)
            # public_w: Individual LoRA (renamed to public keys) + Head → 聚合
            # private_w: Individual LoRA (original keys) + gate → 本地保存
            from ..models.lora import get_fedalt_state_dict
            head_mode = getattr(self.args, 'head_mode', 'global')
            public_w, private_w = get_fedalt_state_dict(model, head_mode=head_mode)
            
            # 恢复 public LoRA 的 requires_grad（避免影响下次训练初始化）
            for name, param in model.named_parameters():
                if 'lora_' in name and '_private' not in name and 'lambda_k' not in name:
                    param.requires_grad = True
            
            return public_w, private_w, avg_loss, train_metrics
        elif self.args.alg == 'feddpa':
            # FedDPA: 返回 (global_w, private_w, loss, train_metrics)
            # global_w 用于服务器聚合，private_w 用于客户端本地保存
            head_mode = getattr(self.args, 'head_mode', 'global')
            global_w, private_w = get_dpa_state_dict(model, include_private=True, head_mode=head_mode)
            return global_w, private_w, avg_loss, train_metrics
        elif self.args.alg == 'fedsalora':
            # FedSA-LoRA: 返回 (public_w, private_w, loss, train_metrics)
            # public_w (lora_A + Head) 用于服务器聚合，private_w (lora_B) 用于客户端本地保存
            from ..models.lora import get_fedsalora_state_dict
            public_w, private_w = get_fedsalora_state_dict(model)
            return public_w, private_w, avg_loss, train_metrics
        elif self.args.alg == 'pf2lora':
            # PF2LoRA: 返回 (public_w, private_w, loss, train_metrics)
            # public_w (Shared LoRA A+B + Head) 用于服务器聚合
            # private_w (Private LoRA A_p+B_p) 用于客户端本地保存
            from ..models.lora import get_pf2lora_state_dict, prune_private_lora_ranks
            # 自动秩学习：基于重要性分数对 Private LoRA 进行秩剪枝
            enable_pruning = getattr(self.args, 'enable_rank_pruning', True)
            if enable_pruning:
                pruning_start = getattr(self.args, 'pruning_start_round', 10)
                pruning_interval = getattr(self.args, 'pruning_interval', 5)
                if global_round >= pruning_start and (global_round - pruning_start) % pruning_interval == 0:
                    r_private = getattr(self.args, 'lora_r_private', None) or self.args.lora_r
                    target_rank = max(1, int(r_private * getattr(self.args, 'target_rank_ratio', 0.5)))
                    prune_private_lora_ranks(model, target_rank)
            public_w, private_w = get_pf2lora_state_dict(model)
            return public_w, private_w, avg_loss, train_metrics
        elif self.args.alg == 'fedtp':
            # FedTP: 根据当前 Phase 返回不同的参数
            # Phase 1: (public_w, None, loss, train_metrics) — 全局 LoRA + Head 用于聚合
            # Phase 2: (None, private_w, loss, train_metrics) — 私有 LoRA + Head 用于本地保存
            from ..models.lora import get_fedtp_state_dict
            phase1_epochs = getattr(self.args, 'phase1_epochs', 50)
            phase = 1 if global_round < phase1_epochs else 2
            public_w, private_w = get_fedtp_state_dict(model, phase)
            return public_w, private_w, avg_loss, train_metrics
        elif self.args.alg in ('fedlora', 'fedprox_lora', 'local_only', 'lorafair'):
            # FedLoRA / FedProx+LoRA / Local-Only: 返回 LoRA 参数 + MLP head
            # Local-Only: 每个客户端独立维护自己的 LoRA 参数，不进行聚合
            return get_lora_state_dict(model), avg_loss, train_metrics
        else:
            # FedAvg / FedProx: 返回完整 state_dict 的深拷贝
            # 重要: model.state_dict() 返回的是引用，不是副本！
            # 使用模型池时，模型参数会被后续客户端训练前的 load_state_dict 覆盖，
            # 导致之前收集的 state_dict 也被修改。必须使用 clone() 创建独立副本。
            return {k: v.clone() for k, v in model.state_dict().items()}, avg_loss, train_metrics

    def _verify_head_mode_params(
        self, 
        public_w: dict, 
        private_w: dict, 
        head_mode: str,
        head_params_refs: list
    ) -> None:
        """
        验收检查：验证 head_mode 配置是否正确生效
        
        检查项：
        1. 上传参数检查：head_mode=private 时，public_w 不应包含 head 参数
        2. 私有参数检查：head_mode=private 时，private_w 应包含 head 参数
        3. 训练更新检查：head 参数应被优化器更新（梯度非零或参数变化）
        
        Args:
            public_w: 公共参数字典（用于聚合）
            private_w: 私有参数字典（本地保存）
            head_mode: Head 参数模式 ('global' | 'private')
            head_params_refs: Head 参数引用列表
        """
        # 检查 public_w 中的 head 参数
        public_head_keys = [k for k in public_w.keys() if _is_head_param(k)]
        # 检查 private_w 中的 head 参数
        private_head_keys = [k for k in private_w.keys() if _is_head_param(k)]
        
        cprint(f"  [FedSDG Head Mode] head_mode='{head_mode}'")
        
        if head_mode == 'private':
            # 验证 1: public_w 不应包含 head 参数
            if public_head_keys:
                cprint(f"  [WARNING] head_mode='private' 但 public_w 包含 head 参数: {public_head_keys}")
            else:
                cprint(f"  [OK] head_mode='private': public_w 不包含 head 参数 (不参与聚合)")
            
            # 验证 2: private_w 应包含 head 参数
            if private_head_keys:
                cprint(f"  [OK] head_mode='private': private_w 包含 head 参数: {private_head_keys}")
            else:
                cprint(f"  [WARNING] head_mode='private' 但 private_w 不包含 head 参数")
        else:
            # head_mode='global' 时，head 应在 public_w 中
            if public_head_keys:
                cprint(f"  [OK] head_mode='global': public_w 包含 head 参数: {public_head_keys}")
            else:
                cprint(f"  [WARNING] head_mode='global' 但 public_w 不包含 head 参数")
        
        # 验证 3: 检查 head 参数是否被更新（梯度非零）
        if head_params_refs:
            head_with_grad = sum(1 for p in head_params_refs if p.grad is not None and p.grad.abs().sum() > 0)
            total_head = len(head_params_refs)
            if head_with_grad > 0:
                cprint(f"  [OK] Head 参数更新检查: {head_with_grad}/{total_head} 个参数有非零梯度")
            else:
                cprint(f"  [WARNING] Head 参数更新检查: 所有 {total_head} 个参数梯度为零或 None")

    def inference(self, model, loader='train'):
        """Client inference evaluation."""
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        if loader == 'train':
            dataloader = self.trainloader
        elif loader == 'val':
            dataloader = self.validloader
        elif loader == 'test':
            # testloader is deprecated, fallback to validloader
            dataloader = self.validloader
        else:
            raise ValueError(f"Unknown loader: {loader}")
        
        if dataloader is None:
            return 0.0, 0.0

        # FedSDG: Disable private branch during global model evaluation
        # 单次遍历收集门控参数引用，保存原始值并设为 -100（m_k ≈ 0）
        gate_param_backups = []  # [(param, original_data), ...]
        if self.args.alg == 'fedsdg':
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'lambda_k_logit' in name:
                        gate_param_backups.append((param, param.data.clone()))
                        param.data.fill_(-100.0)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                # GPU Transform: resize + channel expand + normalize (for FEMNIST)
                if self.gpu_transform is not None:
                    images = self.gpu_transform(images)

                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        # Restore gate values（使用缓存引用，无需再次遍历 named_parameters）
        if gate_param_backups:
            with torch.no_grad():
                for param, original_data in gate_param_backups:
                    param.data.copy_(original_data)

        accuracy = correct/total
        return accuracy, loss
    
    def inference_feddpa(self, model, loader='train', use_dynamic_weight=True):
        """
        FedDPA 专用推理方法
        
        实现 Instance-wise Dynamic Weighting:
        1. 从训练集采样 Anchors，提取 Global-only 模式下的 Embedding
        2. 对每个测试样本，计算与 Anchors 的相似度，得到动态权重 α_t
        3. 使用动态 α_t 进行混合推理
        
        Args:
            model: 模型
            loader: 数据加载器类型 ('train' | 'val' | 'test')
            use_dynamic_weight: 是否使用动态权重（False 则使用固定 train_mix_ratio）
        
        Returns:
            (accuracy, loss)
        """
        from ..algorithms.feddpa import FedDPAInference
        
        model.eval()
        
        if loader == 'train':
            dataloader = self.trainloader
        elif loader == 'val':
            dataloader = self.validloader
        elif loader == 'test':
            dataloader = self.validloader
        else:
            raise ValueError(f"Unknown loader: {loader}")
        
        if dataloader is None:
            return 0.0, 0.0
        
        # 获取推理配置
        scale_factor = getattr(self.args, 'inference_scale_factor', 0.5)
        anchor_count = getattr(self.args, 'anchor_count', 5)
        
        if use_dynamic_weight:
            # 创建推理器并缓存 Anchors
            inference = FedDPAInference(
                model=model,
                device=self.device,
                scale_factor=scale_factor,
                anchor_count=anchor_count
            )
            inference.cache_anchors(self.trainloader, num_samples=anchor_count, 
                                   gpu_transform=self.gpu_transform)
        
        loss, total, correct = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # GPU Transform
                if self.gpu_transform is not None:
                    images = self.gpu_transform(images)
                
                if use_dynamic_weight:
                    # 使用动态权重推理
                    outputs = inference.inference_with_dynamic_weight(
                        images, gpu_transform=None  # 已经在上面应用过
                    )
                else:
                    # 使用固定 train_mix_ratio
                    outputs = model(images)
                
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()
                
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, loss
