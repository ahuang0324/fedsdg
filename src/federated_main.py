#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ViT, inject_lora, get_pretrained_vit, inject_lora_timm
from utils import get_dataset, average_weights, average_weights_lora, exp_details, get_communication_stats, print_communication_profile


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    args = args_parser()
    exp_details(args)

    logs_root = os.path.join(path_project, 'logs')
    log_subdir = os.path.normpath(args.log_subdir).lstrip(os.sep)
    if log_subdir.startswith('..') or os.path.isabs(args.log_subdir):
        raise ValueError("--log_subdir must be a relative path under ../logs")
    log_dir = os.path.join(logs_root, log_subdir)
    logger = SummaryWriter(log_dir)

    use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if use_cuda else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'vit':
        # Vision Transformer
        # 根据 model_variant 选择从零训练或预训练模型
        if hasattr(args, 'model_variant') and args.model_variant == 'pretrained':
            # 使用预训练 ViT（timm）
            global_model = get_pretrained_vit(
                num_classes=args.num_classes,
                image_size=args.image_size if hasattr(args, 'image_size') else 224,
                pretrained_path=args.pretrained_path if hasattr(args, 'pretrained_path') else None
            )
        else:
            # 从零训练手写 ViT
            img_size = train_dataset[0][0].shape[-1]  # 32 for CIFAR, 28 for MNIST
            channels = train_dataset[0][0].shape[0]   # 3 for CIFAR, 1 for MNIST
            global_model = ViT(
                image_size=img_size,
                patch_size=4,
                num_classes=args.num_classes,
                dim=128,
                depth=6,
                heads=8,
                mlp_dim=256,
                channels=channels,
            )

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)
    
    # FedLoRA 和 FedSDG: 如果算法是 fedlora 或 fedsdg，则注入 LoRA 到 ViT 模型
    if args.alg in ('fedlora', 'fedsdg'):
        # 模型兼容性检查：确保 FedLoRA/FedSDG 仅用于 ViT 模型
        if args.model != 'vit':
            raise ValueError(
                f"{args.alg.upper()} currently only supports ViT model, but got model='{args.model}'. "
                f"Please use --model vit or switch to --alg fedavg for other models."
            )
        
        # 判断是否为 FedSDG 模式
        is_fedsdg = (args.alg == 'fedsdg')
        
        # 根据模型类型选择对应的 LoRA 注入函数
        if hasattr(args, 'model_variant') and args.model_variant == 'pretrained':
            # 预训练模型：使用 timm 专用的 LoRA 注入函数
            global_model = inject_lora_timm(
                global_model, 
                r=args.lora_r, 
                lora_alpha=args.lora_alpha,
                train_head=bool(args.lora_train_mlp_head),
                is_fedsdg=is_fedsdg
            )
        else:
            # 从零训练模型：使用手写 ViT 的 LoRA 注入函数
            print("\n" + "="*60)
            alg_name = "FedSDG" if is_fedsdg else "FedLoRA"
            print(f"[{alg_name}] 开始注入 {alg_name} 到手写 ViT 模型...")
            print("="*60)
            global_model = inject_lora(
                global_model, 
                r=args.lora_r, 
                lora_alpha=args.lora_alpha,
                train_mlp_head=bool(args.lora_train_mlp_head),
                is_fedsdg=is_fedsdg
            )
            print("="*60 + "\n")

    # copy weights
    global_weights = global_model.state_dict()

    # ==================== 通信量统计 ====================
    # 计算并显示通信量统计信息
    comm_stats = get_communication_stats(global_model, args.alg)
    print_communication_profile(comm_stats, args)
    
    # 记录基础通信量信息到 TensorBoard（一次性记录）
    logger.add_scalar('info/total_params', comm_stats['total_params'], 0)
    logger.add_scalar('info/trainable_params', comm_stats['trainable_params'], 0)
    logger.add_scalar('info/comm_params_per_round', comm_stats['comm_params'], 0)
    logger.add_scalar('info/comm_size_per_round_MB', comm_stats['comm_size_mb'], 0)
    logger.add_scalar('info/compression_ratio_percent', comm_stats['compression_ratio'], 0)
    
    # 计算总通信量（双向：上传 + 下载）
    comm_per_round_2way_mb = comm_stats['comm_size_mb'] * 2
    total_comm_volume_mb = comm_per_round_2way_mb * args.epochs
    logger.add_scalar('info/estimated_total_volume_MB', total_comm_volume_mb, 0)
    logger.add_scalar('info/estimated_total_volume_GB', total_comm_volume_mb / 1024, 0)
    
    # ==================== 增强的效率指标 ====================
    # 1. 计算完整模型大小（用于节省率计算）
    full_model_size_mb = comm_stats['total_params'] * 4 / (1024 * 1024)  # float32 = 4 bytes
    
    # 2. 计算通信节省率 (Savings Ratio)
    # 公式: (1 - actual_comm / full_model) * 100%
    savings_ratio_percent = (1 - comm_stats['comm_size_mb'] / full_model_size_mb) * 100
    logger.add_scalar('Efficiency/communication_savings_percent', savings_ratio_percent, 0)
    
    # 3. 记录对数刻度的通信量（用于 TensorBoard 可视化）
    import math
    log10_comm_size = math.log10(comm_stats['comm_size_mb']) if comm_stats['comm_size_mb'] > 0 else 0
    logger.add_scalar('Efficiency/log10_comm_size_MB', log10_comm_size, 0)
    
    print(f"\n{'='*70}")
    print("[Efficiency Metrics] 效率指标")
    print(f"{'='*70}")
    print(f"  完整模型大小: {full_model_size_mb:.2f} MB")
    print(f"  实际通信大小: {comm_stats['comm_size_mb']:.2f} MB")
    print(f"  通信节省率: {savings_ratio_percent:.2f}%")
    print(f"  对数刻度通信量: 10^{log10_comm_size:.2f} MB")
    print(f"{'='*70}\n")
    # ======================================================

    # ==================== FedSDG 专用：客户端私有状态管理 ====================
    # FedSDG 需要为每个客户端维护私有参数（不参与服务器聚合）
    # local_private_states: {user_id: {param_name: tensor}}
    local_private_states = {} if args.alg == 'fedsdg' else None
    
    if args.alg == 'fedsdg':
        print("\n" + "="*70)
        print("[FedSDG] 客户端私有状态管理已初始化")
        print("="*70)
        print(f"  每个客户端将维护独立的私有参数（lora_A_private, lora_B_private, lambda_k）")
        print(f"  私有参数不参与服务器聚合，仅在本地更新")
        print(f"  全局参数（lora_A, lora_B）参与服务器聚合，保持通信量与 FedLoRA 一致")
        print("="*70 + "\n")
    # ========================================================================
    
    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    
    # 初始化累计通信量（用于每轮记录）
    cumulative_comm_volume_mb = 0
    
    # 初始化效率追踪变量
    best_efficiency_score = 0  # 最佳效率得分
    best_efficiency_epoch = 0  # 最佳效率得分对应的轮次

    for epoch in tqdm(range(args.epochs)):
        start_epoch_time = time.time()
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            # ========== FedSDG：加载客户端私有状态 ==========
            if args.alg == 'fedsdg':
                # 深拷贝全局模型
                local_model_copy = copy.deepcopy(global_model)
                
                # 如果该客户端有私有状态，则加载
                if idx in local_private_states:
                    # 获取当前模型的 state_dict
                    current_state = local_model_copy.state_dict()
                    # 更新私有参数
                    for param_name, param_value in local_private_states[idx].items():
                        if param_name in current_state:
                            current_state[param_name] = param_value.clone()
                    # 加载更新后的 state_dict
                    local_model_copy.load_state_dict(current_state)
                # 如果是首次训练该客户端，私有参数使用模型初始化的值（已在 LoRALayer 中初始化）
            else:
                # FedAvg 和 FedLoRA：直接深拷贝全局模型
                local_model_copy = copy.deepcopy(global_model)
            # ================================================
            
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=local_model_copy, global_round=epoch)
            
            # ========== FedSDG：保存客户端私有状态 ==========
            if args.alg == 'fedsdg':
                # 提取并保存私有参数（_private 和 lambda_k）
                private_state = {}
                for name, param in local_model_copy.named_parameters():
                    if '_private' in name or 'lambda_k' in name:
                        private_state[name] = param.data.clone().cpu()  # 保存到 CPU 以节省 GPU 内存
                local_private_states[idx] = private_state
            # ================================================
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            
            # ========== FedSDG Debug: 每5轮打印调试信息 ==========
            if args.alg == 'fedsdg' and epoch % 5 == 0 and idx == idxs_users[0]:
                # 打印第一个客户端的调试信息
                print(f"\n[FedSDG Debug - Round {epoch+1}, Client {idx}]")
                
                # 1. 打印 lambda_k 的均值（门控参数）
                lambda_k_values = []
                for name, param in local_model_copy.named_parameters():
                    if 'lambda_k_logit' in name:
                        lambda_k = torch.sigmoid(param).item()
                        lambda_k_values.append(lambda_k)
                if lambda_k_values:
                    print(f"  Lambda_k 均值: {np.mean(lambda_k_values):.4f} (范围: {min(lambda_k_values):.4f} - {max(lambda_k_values):.4f})")
                    print(f"  解释: lambda_k={np.mean(lambda_k_values):.4f} 表示 {np.mean(lambda_k_values)*100:.1f}% 私有分支 + {(1-np.mean(lambda_k_values))*100:.1f}% 全局分支")
                
                # 2. 打印上传的 state_dict 键名（前5个）
                print(f"  上传参数键名（前5个）: {list(w.keys())[:5]}")
                print(f"  上传参数总数: {len(w)} 个键")
                
                # 3. 打印分类头权重范数（确认其正在更新）
                head_norm = 0.0
                for name, param in local_model_copy.named_parameters():
                    if 'head' in name and 'weight' in name:
                        head_norm = param.data.norm().item()
                        print(f"  分类头权重范数: {head_norm:.4f}")
                        break
                
                # 4. 打印全局 LoRA 参数的范数（确认其正在更新）
                global_lora_norms = []
                for name, param in local_model_copy.named_parameters():
                    if 'lora_A' in name and '_private' not in name:
                        global_lora_norms.append(param.data.norm().item())
                if global_lora_norms:
                    print(f"  全局 LoRA_A 平均范数: {np.mean(global_lora_norms):.4f}")
                
                print(f"[FedSDG Debug End]\n")
            # =====================================================

        # update global weights
        # FedLoRA 和 FedSDG: 使用选择性聚合（仅聚合 LoRA 全局参数）
        # FedAvg: 使用全量聚合
        if args.alg in ('fedlora', 'fedsdg'):
            global_weights = average_weights_lora(local_weights, global_model.state_dict())
            
            # ========== FedSDG Debug: 每5轮打印聚合后的信息 ==========
            if args.alg == 'fedsdg' and epoch % 5 == 0:
                print(f"\n[FedSDG Aggregation Debug - Round {epoch+1}]")
                print(f"  聚合的参数键数量: {len([k for k in global_weights.keys() if 'lora_' in k or 'head' in k])}")
                print(f"  全局模型总键数: {len(global_weights)}")
                
                # 检查是否包含私有参数（不应该包含）
                private_keys = [k for k in global_weights.keys() if '_private' in k or 'lambda_k' in k]
                if private_keys:
                    print(f"  ⚠️ 警告: 全局模型包含私有参数（不应该出现）: {private_keys[:3]}")
                else:
                    print(f"  ✓ 验证通过: 全局模型不包含私有参数")
                print(f"[FedSDG Aggregation Debug End]\n")
            # ========================================================
        else:
            global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        logger.add_scalar('global/train_loss_avg', loss_avg, epoch)
        logger.add_scalar('client/train_loss_mean', np.mean(local_losses), epoch)
        logger.add_scalar('client/train_loss_var', np.var(local_losses), epoch)
        logger.add_scalar('lr', args.lr, epoch)

        # Calculate avg training accuracy on participating clients (not all users)
        # 性能优化：仅评估本轮参与训练的客户端，而非所有客户端
        # 这大幅减少了评估时间，尤其是在客户端数量较多时
        list_acc, list_loss = [], []
        global_model.eval()
        for idx in idxs_users:  # 仅评估参与训练的客户端
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model, loader='train')
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        
        logger.add_scalar('global/train_acc_avg', train_accuracy[-1], epoch)
        logger.add_scalar('global/train_loss_eval', sum(list_loss)/len(list_loss), epoch)
        
        # 记录累计通信量（每轮双向通信）
        cumulative_comm_volume_mb += comm_per_round_2way_mb
        logger.add_scalar('info/cumulative_comm_volume_MB', cumulative_comm_volume_mb, epoch)
        logger.add_scalar('info/cumulative_comm_volume_GB', cumulative_comm_volume_mb / 1024, epoch)

        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        logger.add_scalar('global/test_acc', test_acc, epoch)
        logger.add_scalar('global/test_loss', test_loss, epoch)
        logger.add_scalar('time/round', time.time() - start_epoch_time, epoch)
        
        # ==================== 每轮效率指标 ====================
        # 1. 通信节省率（每轮记录，应该是稳定的）
        logger.add_scalar('Efficiency/communication_savings_percent', savings_ratio_percent, epoch)
        
        # 2. 对数刻度累计通信量
        log10_cumulative_comm = math.log10(cumulative_comm_volume_mb) if cumulative_comm_volume_mb > 0 else 0
        logger.add_scalar('Efficiency/log10_cumulative_comm_MB', log10_cumulative_comm, epoch)
        
        # 3. 通信资源效率评分 (Accuracy per MB)
        # 公式: current_test_accuracy / cumulative_MB_transferred
        # 避免除零：第一轮使用单轮通信量
        comm_for_efficiency = cumulative_comm_volume_mb if cumulative_comm_volume_mb > 0 else comm_per_round_2way_mb
        efficiency_score = test_acc / comm_for_efficiency if comm_for_efficiency > 0 else 0
        logger.add_scalar('Efficiency/accuracy_per_MB', efficiency_score, epoch)
        
        # 追踪最佳效率
        if efficiency_score > best_efficiency_score:
            best_efficiency_score = efficiency_score
            best_efficiency_epoch = epoch
        
        # 4. 额外的效率指标：每 GB 准确率
        efficiency_score_per_gb = test_acc / (cumulative_comm_volume_mb / 1024) if cumulative_comm_volume_mb > 0 else 0
        logger.add_scalar('Efficiency/accuracy_per_GB', efficiency_score_per_gb, epoch)
        # ====================================================

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    objects_dir = os.path.join(path_project, 'save', 'objects')
    os.makedirs(objects_dir, exist_ok=True)
    
    # 文件名包含算法类型和 LoRA 参数（如果使用 FedLoRA）
    if args.alg == 'fedlora':
        file_name = os.path.join(objects_dir, '{}_{}_{}_E[{}]_C[{}]_alpha[{}]_LE[{}]_B[{}]_r[{}]_lalpha[{}].pkl'.
                                 format(args.dataset, args.model, args.alg, args.epochs, args.frac, 
                                        args.dirichlet_alpha, args.local_ep, args.local_bs, 
                                        args.lora_r, args.lora_alpha))
    else:
        file_name = os.path.join(objects_dir, '{}_{}_{}_C[{}]_alpha[{}]_E[{}]_B[{}].pkl'.
                                 format(args.dataset, args.model, args.epochs, args.frac, args.dirichlet_alpha,
                                        args.local_ep, args.local_bs))

    # 保存训练结果，包含通信量统计
    results = {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'comm_stats': comm_stats,  # 通信量统计
        'total_comm_volume_mb': cumulative_comm_volume_mb,  # 总通信量
        'args': vars(args)  # 保存所有参数配置
    }
    
    with open(file_name, 'wb') as f:
        pickle.dump(results, f)

    total_time = time.time() - start_time
    print('\n Total Run Time: {0:0.4f}'.format(total_time))
    
    # ==================== 生成文本摘要报告 ====================
    # 计算相比 FedAvg 的节省量（假设 FedAvg 传输完整模型）
    fedavg_estimated_comm_mb = full_model_size_mb * 2 * args.epochs  # 双向通信
    saved_comm_mb = fedavg_estimated_comm_mb - cumulative_comm_volume_mb
    saved_comm_gb = saved_comm_mb / 1024
    savings_multiplier = fedavg_estimated_comm_mb / cumulative_comm_volume_mb if cumulative_comm_volume_mb > 0 else 1
    
    # 构建格式化的实验总结报告
    summary_text = f"""
# 联邦学习实验总结报告

## 基本配置
- **算法**: {args.alg.upper()}
- **模型**: {args.model.upper()} ({args.model_variant if hasattr(args, 'model_variant') else 'scratch'})
- **数据集**: {args.dataset.upper()}
- **训练轮次**: {args.epochs}
- **客户端数量**: {args.num_users}
- **参与率**: {args.frac * 100:.1f}%

## 性能指标
- **最终训练准确率**: {train_accuracy[-1] * 100:.2f}%
- **最终测试准确率**: {test_acc * 100:.2f}%
- **最终训练损失**: {train_loss[-1]:.4f}
- **最终测试损失**: {test_loss:.4f}
- **总训练时间**: {total_time / 60:.2f} 分钟 ({total_time:.2f} 秒)
- **平均每轮时间**: {total_time / args.epochs:.2f} 秒

## 通信效率分析
### 模型参数统计
- **总参数量**: {comm_stats['total_params']:,} ({full_model_size_mb:.2f} MB)
- **可训练参数**: {comm_stats['trainable_params']:,}
- **每轮通信参数**: {comm_stats['comm_params']:,} ({comm_stats['comm_size_mb']:.2f} MB)
- **压缩率**: {comm_stats['compression_ratio']:.2f}%

### 通信量统计
- **单轮通信量（双向）**: {comm_per_round_2way_mb:.2f} MB
- **总通信量**: {cumulative_comm_volume_mb:.2f} MB ({cumulative_comm_volume_mb / 1024:.2f} GB)
- **通信节省率**: {savings_ratio_percent:.2f}%

### 相比 FedAvg 的优势 (假设 FedAvg 传输完整模型)
- **FedAvg 预估通信量**: {fedavg_estimated_comm_mb:.2f} MB ({fedavg_estimated_comm_mb / 1024:.2f} GB)
- **节省的通信量**: {saved_comm_mb:.2f} MB ({saved_comm_gb:.2f} GB)
- **节省倍数**: {savings_multiplier:.2f}x
- **通信效率提升**: {(savings_multiplier - 1) * 100:.1f}%

### 效率评分
- **准确率/MB**: {efficiency_score:.6f}
- **准确率/GB**: {efficiency_score_per_gb:.4f}
- **最佳效率轮次**: 第 {best_efficiency_epoch + 1} 轮
- **最佳效率得分**: {best_efficiency_score:.6f}

## LoRA 配置 (FedLoRA/FedSDG)
"""
    
    if args.alg in ('fedlora', 'fedsdg'):
        summary_text += f"""
- **LoRA 秩 (r)**: {args.lora_r}
- **LoRA Alpha**: {args.lora_alpha}
- **训练分类头**: {'是' if args.lora_train_mlp_head else '否'}
"""
        if args.alg == 'fedsdg':
            summary_text += f"""
- **FedSDG 双路架构**: 全局分支 + 私有分支
- **私有参数**: 不参与服务器聚合（仅本地更新）
- **门控机制**: 可学习的 λ_k 参数动态平衡全局/私有权重
"""
    
    summary_text += f"""

## 结论
"""
    
    if args.alg == 'fedlora':
        summary_text += f"""
本次实验使用 **FedLoRA** 算法，成功将通信开销降低至原来的 **{100 - savings_ratio_percent:.2f}%**。
相比传统 FedAvg，节省了 **{saved_comm_gb:.2f} GB** 的通信流量，相当于减少了 **{savings_multiplier:.2f}** 倍的通信成本。
同时保持了 **{test_acc * 100:.2f}%** 的测试准确率，展现了参数高效联邦学习（PEFT）的强大优势。

**投入产出比**: 每传输 1 MB 数据，获得 {efficiency_score:.6f} 的准确率提升，
即每 GB 流量可换取 {efficiency_score_per_gb:.4f} 的准确率收益。
"""
    elif args.alg == 'fedsdg':
        summary_text += f"""
本次实验使用 **FedSDG** 算法，通过双路架构（全局分支 + 私有分支）对抗 Non-IID 数据分布。
通信开销与 FedLoRA 保持一致，降低至原来的 **{100 - savings_ratio_percent:.2f}%**。
相比传统 FedAvg，节省了 **{saved_comm_gb:.2f} GB** 的通信流量，相当于减少了 **{savings_multiplier:.2f}** 倍的通信成本。

**FedSDG 特点**:
- 私有参数（lora_A_private, lora_B_private, lambda_k）仅在客户端本地更新
- 全局参数（lora_A, lora_B）参与服务器聚合
- 通过门控机制自动学习全局/私有分支的最优权重
- 最终测试准确率: **{test_acc * 100:.2f}%**

**投入产出比**: 每传输 1 MB 数据，获得 {efficiency_score:.6f} 的准确率提升，
即每 GB 流量可换取 {efficiency_score_per_gb:.4f} 的准确率收益。
"""
    else:
        summary_text += f"""
本次实验使用 **FedAvg** 算法，传输完整模型参数进行联邦学习。
总通信量为 **{cumulative_comm_volume_mb / 1024:.2f} GB**，最终测试准确率达到 **{test_acc * 100:.2f}%**。

**投入产出比**: 每传输 1 MB 数据，获得 {efficiency_score:.6f} 的准确率提升，
即每 GB 流量可换取 {efficiency_score_per_gb:.4f} 的准确率收益。
"""
    
    summary_text += f"""

---
*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 将摘要写入 TensorBoard
    logger.add_text('Experiment_Summary', summary_text, 0)
    
    # 同时保存为文本文件
    summary_dir = os.path.join(path_project, 'save', 'summaries')
    os.makedirs(summary_dir, exist_ok=True)
    
    if args.alg == 'fedlora':
        summary_filename = f'{args.dataset}_{args.model}_{args.alg}_E{args.epochs}_r{args.lora_r}_summary.txt'
    else:
        summary_filename = f'{args.dataset}_{args.model}_{args.alg}_E{args.epochs}_summary.txt'
    
    summary_path = os.path.join(summary_dir, summary_filename)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"\n{'='*70}")
    print(f"实验总结报告已保存到:")
    print(f"  - TensorBoard: 'Experiment_Summary' 标签页")
    print(f"  - 文本文件: {summary_path}")
    print(f"{'='*70}\n")
    # ==========================================================
    
    # 保存最终模型（可选）
    model_dir = os.path.join(path_project, 'save', 'models')
    os.makedirs(model_dir, exist_ok=True)
    if args.alg == 'fedlora':
        model_path = os.path.join(model_dir, '{}_{}_{}_final_r[{}]_lalpha[{}].pth'.
                                  format(args.dataset, args.model, args.alg, args.lora_r, args.lora_alpha))
    else:
        model_path = os.path.join(model_dir, '{}_{}_final.pth'.format(args.dataset, args.model))
    torch.save(global_model.state_dict(), model_path)
    print(f'Final model saved to: {model_path}')

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_alpha[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.dirichlet_alpha, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_alpha[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.dirichlet_alpha, args.local_ep, args.local_bs))
    
    logger.close()
