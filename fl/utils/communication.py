# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Communication statistics utilities for federated learning.

Provides functions to compute and display communication overhead.

通信量计算说明:
- comm_size_mb: 单个客户端一次传输的模型更新大小
- 每轮总通信量 = 2 * m * comm_size_mb
  - 下行: 服务器向 m 个客户端各发送一份模型 (m * comm_size_mb)
  - 上行: m 个客户端各上传一份更新 (m * comm_size_mb)
- 注意: 某些论文假设广播模式（下行只算一份），请根据实际网络模型选择
"""


def get_communication_stats(model, alg):
    """
    Calculate communication statistics for federated learning.
    
    Args:
        model: PyTorch model
        alg: Algorithm type ('fedavg', 'fedprox_avg', 'fedlora', 'fedprox_lora', 'fedsdg', 'fedrep', or 'local_only')
    
    Returns:
        dict: Communication statistics
            - total_params: Total model parameters
            - trainable_params: Trainable parameters
            - comm_params: Parameters communicated per round
            - comm_keys: List of parameter names that are communicated
            - total_size_mb: Total model size (MB)
            - trainable_size_mb: Trainable parameters size (MB)
            - comm_size_mb: Communication size per round (MB)
            - compression_ratio: Compression ratio (%)
    """
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Communication parameters and keys
    comm_params = 0
    comm_keys = []
    
    if alg in ('fedavg', 'fedprox_avg'):
        # FedAvg / FedProx: Communicate all parameters
        comm_params = total_params
        comm_keys = [name for name, _ in model.named_parameters()]
    elif alg in ('fedlora', 'fedprox_lora', 'fedsdg', 'feddpa', 'pf2lora', 'lorafair', 'fedalt'):
        # FedLoRA / FedProx+LoRA / FedSDG / FedDPA / PF2LoRA: Only communicate global LoRA parameters
        # Exclude private parameters (_private) and gate parameters (lambda_k)
        for name, p in model.named_parameters():
            if p.requires_grad:
                if '_private' not in name and 'lambda_k' not in name:
                    comm_params += p.numel()
                    comm_keys.append(name)
    elif alg == 'fedsalora':
        # FedSA-LoRA: Only communicate lora_A + Head (lora_B stays local)
        for name, p in model.named_parameters():
            if p.requires_grad:
                if 'lora_B' not in name:
                    comm_params += p.numel()
                    comm_keys.append(name)
    elif alg == 'fedtp':
        # FedTP: Phase 1 communicates global LoRA + Head (same as FedLoRA)
        # Phase 2 has zero communication (handled in trainer.py)
        # Here we report Phase 1 per-round communication volume
        for name, p in model.named_parameters():
            if p.requires_grad:
                if '_private' not in name and 'lambda_k' not in name:
                    comm_params += p.numel()
                    comm_keys.append(name)
    elif alg == 'fedrep':
        # FedRep: Only communicate LoRA parameters (Backbone)
        # Head (mlp_head, head) is kept local and NOT communicated
        for name, p in model.named_parameters():
            if p.requires_grad:
                # Only include LoRA parameters, exclude head
                if 'lora_' in name and '_private' not in name and 'lambda_k' not in name:
                    comm_params += p.numel()
                    comm_keys.append(name)
    elif alg == 'ditto':
        # Ditto: Communicate all trainable parameters (LoRA + Head)
        # Same as FedLoRA, personal model stays local
        for name, p in model.named_parameters():
            if p.requires_grad:
                if '_private' not in name and 'lambda_k' not in name:
                    comm_params += p.numel()
                    comm_keys.append(name)
    elif alg == 'local_only':
        # Local-Only: No communication (each client maintains its own LoRA parameters)
        comm_params = 0
        comm_keys = []
    else:
        comm_params = total_params
        comm_keys = [name for name, _ in model.named_parameters()]
    
    # Convert to MB (assuming float32, 4 bytes per parameter)
    bytes_per_param = 4
    total_size_mb = (total_params * bytes_per_param) / (1024 ** 2)
    trainable_size_mb = (trainable_params * bytes_per_param) / (1024 ** 2)
    comm_size_mb = (comm_params * bytes_per_param) / (1024 ** 2)
    
    # Compression ratio
    compression_ratio = (comm_params / total_params) * 100 if total_params > 0 else 100
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'comm_params': comm_params,
        'comm_keys': comm_keys,  # : 通信参数的 key 列表
        'total_size_mb': total_size_mb,
        'trainable_size_mb': trainable_size_mb,
        'comm_size_mb': comm_size_mb,  # 单个客户端一次传输大小
        'compression_ratio': compression_ratio
    }


def compute_round_communication(comm_stats: dict, num_selected_clients: int) -> dict:
    """
    计算单轮的总通信量
    
    Args:
        comm_stats: get_communication_stats() 返回的统计信息
        num_selected_clients: 本轮参与的客户端数量 (m)
    
    Returns:
        dict: 包含详细通信量信息
            - per_client_mb: 单个客户端传输量
            - downlink_mb: 下行总量 (服务器 -> 客户端)
            - uplink_mb: 上行总量 (客户端 -> 服务器)
            - total_mb: 本轮总通信量
    """
    per_client_mb = comm_stats['comm_size_mb']
    
    # 下行: 服务器向每个参与客户端发送模型
    downlink_mb = num_selected_clients * per_client_mb
    
    # 上行: 每个参与客户端上传更新
    uplink_mb = num_selected_clients * per_client_mb
    
    return {
        'per_client_mb': per_client_mb,
        'downlink_mb': downlink_mb,
        'uplink_mb': uplink_mb,
        'total_mb': downlink_mb + uplink_mb,
        'num_clients': num_selected_clients,
    }


def compute_total_communication(
    comm_stats: dict, 
    epochs: int, 
    num_users: int, 
    frac: float
) -> dict:
    """
    估算整个训练过程的总通信量
    
    Args:
        comm_stats: get_communication_stats() 返回的统计信息
        epochs: 总训练轮次
        num_users: 客户端总数
        frac: 每轮参与率
    
    Returns:
        dict: 包含总通信量估算
    """
    m = max(int(frac * num_users), 1)  # 每轮参与客户端数
    per_round = compute_round_communication(comm_stats, m)
    
    total_mb = per_round['total_mb'] * epochs
    
    # 对比 FedAvg（传输完整模型）
    full_model_mb = comm_stats['total_params'] * 4 / (1024 ** 2)
    fedavg_per_round_mb = 2 * m * full_model_mb
    fedavg_total_mb = fedavg_per_round_mb * epochs
    
    savings_mb = fedavg_total_mb - total_mb
    savings_ratio = savings_mb / fedavg_total_mb if fedavg_total_mb > 0 else 0
    
    return {
        'per_round_mb': per_round['total_mb'],
        'total_mb': total_mb,
        'total_gb': total_mb / 1024,
        'fedavg_total_mb': fedavg_total_mb,
        'fedavg_total_gb': fedavg_total_mb / 1024,
        'savings_mb': savings_mb,
        'savings_gb': savings_mb / 1024,
        'savings_ratio': savings_ratio,
        'savings_multiplier': fedavg_total_mb / total_mb if total_mb > 0 else 1,
        'clients_per_round': m,
    }


def print_communication_profile(comm_stats, args):
    """
    Print formatted communication statistics.
    
    Args:
        comm_stats: Statistics from get_communication_stats()
        args: Command line arguments
    """
    # 计算每轮参与客户端数
    m = max(int(args.frac * args.num_users), 1)
    
    # 使用正确的通信量计算
    total_comm = compute_total_communication(
        comm_stats, args.epochs, args.num_users, args.frac
    )
    
    print("\n" + "="*70)
    print("COMMUNICATION PROFILE".center(70))
    print("="*70)
    
    print(f"\n{'Metric':<40} {'Value':>18} {'Unit':>8}")
    print("-"*70)
    
    # Parameter statistics
    print(f"{'Total Parameters':<40} {comm_stats['total_params']:>18,} {'params':>8}")
    print(f"{'Trainable Parameters':<40} {comm_stats['trainable_params']:>18,} {'params':>8}")
    print(f"{'Communication Parameters':<40} {comm_stats['comm_params']:>18,} {'params':>8}")
    
    print("-"*70)
    
    # Size statistics
    print(f"{'Total Model Size':<40} {comm_stats['total_size_mb']:>18.2f} {'MB':>8}")
    print(f"{'Trainable Size':<40} {comm_stats['trainable_size_mb']:>18.2f} {'MB':>8}")
    print(f"{'Per-Client Transfer (1-way)':<40} {comm_stats['comm_size_mb']:>18.2f} {'MB':>8}")
    
    print("-"*70)
    
    # Per-round communication (考虑客户端数量)
    print(f"{'Clients per Round (m)':<40} {m:>18} {'clients':>8}")
    print(f"{'Downlink per Round (m × size)':<40} {total_comm['per_round_mb']/2:>18.2f} {'MB':>8}")
    print(f"{'Uplink per Round (m × size)':<40} {total_comm['per_round_mb']/2:>18.2f} {'MB':>8}")
    print(f"{'Total per Round (2 × m × size)':<40} {total_comm['per_round_mb']:>18.2f} {'MB':>8}")
    
    print("-"*70)
    
    # Total communication estimate
    print(f"{'Total Rounds':<40} {args.epochs:>18} {'rounds':>8}")
    print(f"{'Estimated Total Volume':<40} {total_comm['total_mb']:>18.2f} {'MB':>8}")
    print(f"{'Estimated Total Volume':<40} {total_comm['total_gb']:>18.2f} {'GB':>8}")
    
    print("-"*70)
    
    # Comparison with FedAvg
    print(f"{'FedAvg Estimated Total':<40} {total_comm['fedavg_total_gb']:>18.2f} {'GB':>8}")
    print(f"{'Communication Savings':<40} {total_comm['savings_gb']:>18.2f} {'GB':>8}")
    print(f"{'Savings Ratio':<40} {total_comm['savings_ratio']*100:>18.2f} {'%':>8}")
    print(f"{'Savings Multiplier':<40} {total_comm['savings_multiplier']:>18.2f} {'x':>8}")
    
    print("-"*70)
    
    # Efficiency metrics
    print(f"{'Compression Ratio':<40} {comm_stats['compression_ratio']:>18.2f} {'%':>8}")
    
    print("="*70)
    
    # Algorithm-specific notes
    if args.alg == 'fedavg':
        print("\n[FedAvg] Communicating ALL model parameters each round")
    elif args.alg == 'fedprox_avg':
        print("\n[FedProx] Communicating ALL model parameters each round (with proximal term)")
    elif args.alg == 'fedlora':
        print("\n[FedLoRA] Communicating ONLY LoRA parameters + classification head")
        print(f"[FedLoRA] Parameter Efficiency: {comm_stats['compression_ratio']:.2f}% of full model")
    elif args.alg == 'fedprox_lora':
        print("\n[FedProx+LoRA] Communicating ONLY LoRA parameters + classification head (with proximal term)")
        print(f"[FedProx+LoRA] Parameter Efficiency: {comm_stats['compression_ratio']:.2f}% of full model")
    elif args.alg == 'fedsdg':
        print("\n[FedSDG] Communicating ONLY Global LoRA parameters (lora_A, lora_B) + classification head")
        print(f"[FedSDG] Private parameters (lora_A_private, lora_B_private, lambda_k) stay local")
        print(f"[FedSDG] Communication Efficiency: {comm_stats['compression_ratio']:.2f}% of full model")
    elif args.alg == 'feddpa':
        print("\n[FedDPA] Communicating ONLY Global LoRA parameters (lora_A, lora_B) + classification head")
        print(f"[FedDPA] Private parameters (lora_A_private, lora_B_private) stay local")
        print(f"[FedDPA] Communication Efficiency: {comm_stats['compression_ratio']:.2f}% of full model")
    elif args.alg == 'fedsalora':
        print("\n[FedSA-LoRA] Communicating ONLY lora_A parameters + classification head")
        print(f"[FedSA-LoRA] lora_B parameters stay local (client-specific knowledge)")
        print(f"[FedSA-LoRA] Communication Efficiency: {comm_stats['compression_ratio']:.2f}% of full model")
    elif args.alg == 'pf2lora':
        print("\n[PF2LoRA] Communicating Shared LoRA (lora_A, lora_B) + classification head")
        print(f"[PF2LoRA] Private LoRA (lora_A_private, lora_B_private) stays local")
        print(f"[PF2LoRA] Automatic rank pruning on Private LoRA (importance-based)")
        print(f"[PF2LoRA] Communication Efficiency: {comm_stats['compression_ratio']:.2f}% of full model")
    elif args.alg == 'fedtp':
        phase1_epochs = getattr(args, 'phase1_epochs', 50)
        print(f"\n[FedTP] Two-Phase LoRA: Phase 1 (epoch 0~{phase1_epochs-1}) communicates global LoRA + Head")
        print(f"[FedTP] Phase 2 (epoch {phase1_epochs}~end): zero communication (local-only fine-tuning)")
        print(f"[FedTP] Phase 1 per-round Communication: {comm_stats['compression_ratio']:.2f}% of full model")
    elif args.alg == 'fedrep':
        print("\n[FedRep] Communicating ONLY LoRA parameters (Backbone)")
        print(f"[FedRep] Head (mlp_head/head) stays local for personalization")
        print(f"[FedRep] Communication Efficiency: {comm_stats['compression_ratio']:.2f}% of full model")
    elif args.alg == 'ditto':
        print("\n[Ditto] Communicating Global model (LoRA + Head)")
        print(f"[Ditto] Personal model stays local (not communicated)")
        print(f"[Ditto] Communication Efficiency: {comm_stats['compression_ratio']:.2f}% of full model")
    
    print("\n[Note] Communication = 2 × m × per_client_size (m = selected clients per round)")
    print("="*70 + "\n")


