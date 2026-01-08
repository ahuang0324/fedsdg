#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models import get_lora_state_dict


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs = list(idxs)
        np.random.shuffle(idxs)
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True,
                                 num_workers=4, pin_memory=True, prefetch_factor=2)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=max(1, int(len(idxs_val)/10)), shuffle=False,
                                 num_workers=2, pin_memory=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=max(1, int(len(idxs_test)/10)), shuffle=False,
                                num_workers=2, pin_memory=True)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        """
        客户端本地训练函数
        
        FedSDG 算法核心实现 (Equation 5 from FedSDG_Design.md):
        Loss = (1/|B|) Σ ℓ(f(x), y) + λ₁ Σ|m_{k,l}| + λ₂ ||θ_{p,k}||²₂
               └────────────────┘   └────────┘   └──────────┘
               Task Loss          L1 Gate      L2 Private
        
        其中:
        - m_{k,l} = sigmoid(lambda_k_logit): 门控权重，范围 [0, 1]
        - θ_{p,k}: 私有 LoRA 参数（名称包含 '_private'）
        - λ₁: L1 门控稀疏性惩罚系数，鼓励门控参数接近 0 或 1
        - λ₂: L2 私有参数正则化系数，限制私有参数容量
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # ==================== 优化器配置 ====================
        # FedLoRA/FedSDG: 仅优化可训练参数（LoRA 参数 + mlp_head）
        # FedAvg: 优化所有参数
        # 
        # FedSDG 特殊配置（根据 proposal 设计）:
        # - 三组参数使用不同学习率：ηg (共享), ηp (私有), ηm (门控)
        # - 不使用 weight_decay，因为我们手动实现 λ₂ 正则化
        # - 使用梯度裁剪防止训练不稳定
        
        if self.args.alg == 'fedsdg':
            # FedSDG: 三组参数分别设置学习率
            # ηg: 共享参数 (lora_A, lora_B, head)
            # ηp: 私有参数 (_private)
            # ηm: 门控参数 (lambda_k_logit)
            global_params = []  # 共享 LoRA + head
            private_params = []  # 私有 LoRA
            gate_params = []  # 门控参数
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'lambda_k_logit' in name:
                        gate_params.append(param)
                    elif '_private' in name:
                        private_params.append(param)
                    else:
                        global_params.append(param)
            
            # 根据 proposal: lr_global = lr_private = args.lr, lr_gate = args.lr_gate
            param_groups = [
                {'params': global_params, 'lr': self.args.lr},      # ηg: 共享参数
                {'params': private_params, 'lr': self.args.lr},     # ηp: 私有参数
                {'params': gate_params, 'lr': self.args.lr_gate}    # ηm: 门控参数
            ]
            optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), weight_decay=0)
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(trainable_params, lr=self.args.lr,
                                            momentum=0.5)
            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(trainable_params, lr=self.args.lr,
                                             weight_decay=1e-4)
        # =========================================================

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                logits = model(images)
                
                # ==================== FedSDG 损失函数计算 (Equation 5) ====================
                # 基础任务损失: (1/|B|) Σ ℓ(f(x), y)
                task_loss = self.criterion(logits, labels)
                
                if self.args.alg == 'fedsdg':
                    # ========== λ₁ L1 门控稀疏性惩罚 ==========
                    # 计算: λ₁ Σ_{l=1}^L |m_{k,l}|
                    # 其中 m_{k,l} = sigmoid(lambda_k_logit)
                    # 目的: 鼓励门控参数接近 0 或 1，实现层级选择性个性化
                    gate_penalty = torch.tensor(0.0, device=self.device)
                    gate_count = 0
                    for name, param in model.named_parameters():
                        if 'lambda_k_logit' in name:
                            # m_{k,l} = sigmoid(a_{k,l})
                            m_k = torch.sigmoid(param)
                            # L1 惩罚: |m_{k,l}|
                            gate_penalty = gate_penalty + torch.sum(torch.abs(m_k))
                            gate_count += param.numel()
                    
                    # ========== λ₂ L2 私有参数正则化 ==========
                    # 计算: λ₂ ||θ_{p,k}||²₂
                    # 目的: 限制私有参数容量，防止过拟合
                    private_penalty = torch.tensor(0.0, device=self.device)
                    private_count = 0
                    for name, param in model.named_parameters():
                        if '_private' in name:
                            # L2 惩罚: Σ param²
                            private_penalty = private_penalty + torch.sum(param ** 2)
                            private_count += param.numel()
                    
                    # ========== 组合总损失 (Equation 5) ==========
                    # Loss = TaskLoss + λ₁ * gate_penalty + λ₂ * private_penalty
                    lambda1 = self.args.lambda1  # L1 门控稀疏性惩罚系数
                    lambda2 = self.args.lambda2  # L2 私有参数正则化系数
                    loss = task_loss + lambda1 * gate_penalty + lambda2 * private_penalty
                    
                    # ========== 记录 FedSDG 指标到 TensorBoard ==========
                    # 每个 epoch 的最后一个 batch 记录一次（减少日志量）
                    if batch_idx == len(self.trainloader) - 1:
                        global_step_epoch = global_round * self.args.local_ep + iter
                        # 损失分解
                        self.logger.add_scalar('FedSDG/task_loss', task_loss.item(), global_step_epoch)
                        self.logger.add_scalar('FedSDG/gate_penalty_raw', gate_penalty.item(), global_step_epoch)
                        self.logger.add_scalar('FedSDG/gate_penalty_weighted', lambda1 * gate_penalty.item(), global_step_epoch)
                        self.logger.add_scalar('FedSDG/private_penalty_raw', private_penalty.item(), global_step_epoch)
                        self.logger.add_scalar('FedSDG/private_penalty_weighted', lambda2 * private_penalty.item(), global_step_epoch)
                        
                        # 门控参数统计
                        gate_values = []
                        for name, param in model.named_parameters():
                            if 'lambda_k_logit' in name:
                                gate_values.append(torch.sigmoid(param).item())
                        if gate_values:
                            self.logger.add_scalar('FedSDG/gate_mean', sum(gate_values) / len(gate_values), global_step_epoch)
                            self.logger.add_scalar('FedSDG/gate_min', min(gate_values), global_step_epoch)
                            self.logger.add_scalar('FedSDG/gate_max', max(gate_values), global_step_epoch)
                else:
                    # FedAvg / FedLoRA: 仅使用任务损失
                    loss = task_loss
                # ================================================================
                
                loss.backward()
                
                # ========== FedSDG: 梯度裁剪（根据 proposal 设计）==========
                if self.args.alg == 'fedsdg' and self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
                # =========================================================
                
                # ========== 调试：检查门控参数梯度 ==========
                if self.args.alg == 'fedsdg' and batch_idx == 0 and iter == 0:
                    for name, param in model.named_parameters():
                        if 'lambda_k_logit' in name:
                            grad_val = param.grad.item() if param.grad is not None else 0.0
                            print(f"  [Gradient Debug] {name}: grad={grad_val:.6f}, value={param.data.item():.4f}")
                            break  # 只打印第一个
                # =============================================
                
                optimizer.step()

                # 每个 epoch 只在开始时打印一次（减少日志输出）
                if self.args.verbose and batch_idx == 0:
                    print('| Global Round : {} | Local Epoch : {} | Starting training...'.format(
                        global_round, iter))
                global_step = (global_round * self.args.local_ep * len(self.trainloader)
                               + iter * len(self.trainloader) + batch_idx)
                self.logger.add_scalar('loss', loss.item(), global_step=global_step)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # FedLoRA/FedSDG: 仅返回 LoRA 参数（过滤私有参数和冻结权重）
        # FedAvg: 返回完整 state_dict
        if self.args.alg in ('fedlora', 'fedsdg'):
            return get_lora_state_dict(model), sum(epoch_loss) / len(epoch_loss)
        else:
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, loader='train'):
        """
        客户端推理评估函数
        
        对于 FedSDG 算法的特殊处理:
        - 当使用 global_model 进行评估时，应该仅使用全局 LoRA 分支
        - 因为 global_model 中的私有参数是初始化值，不是该客户端训练的
        - 因此在评估时需要禁用私有分支（设置门控 m_k = 0）
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        if loader == 'train':
            dataloader = self.trainloader
        elif loader == 'val':
            dataloader = self.validloader
        elif loader == 'test':
            dataloader = self.testloader
        else:
            raise ValueError(f"Unknown loader: {loader}")

        # ==================== FedSDG: 评估时禁用私有分支 ====================
        # 保存原始门控参数值，评估完成后恢复
        original_gate_values = {}
        if self.args.alg == 'fedsdg':
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'lambda_k_logit' in name:
                        # 保存原始值
                        original_gate_values[name] = param.data.clone()
                        # 设置为 -100，使得 m_k = sigmoid(-100) ≈ 0
                        param.data.fill_(-100.0)
        # ===================================================================

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        # ==================== FedSDG: 恢复原始门控参数值 ====================
        if self.args.alg == 'fedsdg':
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in original_gate_values:
                        param.data.copy_(original_gate_values[name])
        # ===================================================================

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """
    全局测试推理函数
    
    对于 FedSDG 算法的特殊处理:
    - 全局测试应该评估聚合后的全局模型的泛化能力
    - 私有参数是客户端特定的，不应该在全局测试中使用
    - 因此在测试时需要禁用私有分支（设置门控 m_k = 0）
    
    实现方式:
    - 临时将所有 lambda_k_logit 设置为一个很小的负数（如 -100）
    - 这使得 m_k = sigmoid(-100) ≈ 0，仅使用全局 LoRA 分支
    - 测试完成后恢复原始值
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ==================== FedSDG: 全局测试时禁用私有分支 ====================
    # 保存原始门控参数值，测试完成后恢复
    original_gate_values = {}
    if args.alg == 'fedsdg':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'lambda_k_logit' in name:
                    # 保存原始值
                    original_gate_values[name] = param.data.clone()
                    # 设置为 -100，使得 m_k = sigmoid(-100) ≈ 0
                    # 这样全局测试时仅使用全局 LoRA 分支
                    param.data.fill_(-100.0)
    # =========================================================================

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    # ==================== FedSDG: 恢复原始门控参数值 ====================
    if args.alg == 'fedsdg':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_gate_values:
                    param.data.copy_(original_gate_values[name])
    # ===================================================================

    accuracy = correct/total
    return accuracy, loss
