# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型池 (Model Pool) 实现

用于复用模型实例，避免每个客户端都 deepcopy 全局模型。

设计原则：
- pool_size=1 时串行复用单个模型实例
- 未来可扩展到 pool_size>1 支持多 worker 并行
- 每次使用前必须完全重置状态，避免污染

Optimization:
- 内存: O(m * model_size) → O(pool_size * model_size)
- 速度: deepcopy → load_state_dict (快 5-10x)
"""

import copy
from typing import Dict, Optional, List

import torch
import torch.nn as nn

from ..utils.console_logger import cprint


class ModelPool:
    """
    模型池：管理可复用的本地模型实例
    
    优化点：
    - 避免每客户端 deepcopy 全局模型（O(m) → O(1) 内存）
    - 通过 load_state_dict 快速重置模型状态
    - 支持 FedSDG 私有状态注入
    
    Attributes:
        pool_size: 池中模型数量（当前固定为 1，未来可扩展）
        device: 模型运行设备
        _models: 模型实例列表
        
    Usage:
        pool = ModelPool(global_model, device='cuda', pool_size=1)
        
        for client_idx in selected_clients:
            local_model = pool.prepare_model(
                worker_id=0,
                global_state=global_state_cache,
                private_state=private_states.get(client_idx),
                is_fedsdg=True
            )
            w, loss = local_trainer.update_weights(model=local_model)
    """
    
    def __init__(
        self, 
        global_model: nn.Module, 
        device: str = 'cuda',
        pool_size: int = 1
    ):
        """
        初始化模型池
        
        Args:
            global_model: 全局模型（用于创建池模型的模板）
            device: 计算设备
            pool_size: 池大小（默认 1，串行训练）
        """
        self.pool_size = pool_size
        self.device = device
        self._models: List[nn.Module] = []
        
        # 创建池中的模型实例（只在初始化时 deepcopy）
        cprint(f"\n[ModelPool] 初始化模型池 (pool_size={pool_size}, device={device})")
        for i in range(pool_size):
            model_copy = copy.deepcopy(global_model)
            model_copy.to(device)
            self._models.append(model_copy)
        
        cprint(f"[ModelPool] 模型池初始化完成，共 {pool_size} 个实例\n")
    
    def get_model(self, worker_id: int = 0) -> nn.Module:
        """
        获取池中的模型实例（不做任何状态重置）
        
        Args:
            worker_id: Worker ID（串行时固定为 0）
            
        Returns:
            池中对应的模型实例
        """
        if worker_id >= self.pool_size:
            raise ValueError(f"worker_id {worker_id} 超出池大小 {self.pool_size}")
        return self._models[worker_id]
    
    def prepare_model(
        self,
        worker_id: int,
        global_state: Dict[str, torch.Tensor],
        private_state: Optional[Dict[str, torch.Tensor]] = None,
        is_fedsdg: bool = False,
        is_feddpa: bool = False,
        is_local_only: bool = False,
        is_fedrep: bool = False,
        is_ditto: bool = False,
        is_fedsalora: bool = False
    ) -> nn.Module:
        """
        准备本地模型用于客户端训练
        
        执行完整的状态重置（避免状态污染）：
        1. 加载全局模型状态（覆盖所有参数）
        2. 注入私有状态（如果有）
        3. 设置训练模式
        4. 清理梯度（set_to_none=True 风格）
        
        Args:
            worker_id: Worker ID
            global_state: 全局模型的 state_dict（应在轮开始时缓存）
            private_state: 客户端私有状态（可选）
            is_fedsdg: 是否为 FedSDG 算法
            is_local_only: 是否为 Local-Only 算法
            is_fedrep: 是否为 FedRep 算法
            is_ditto: 是否为 Ditto 算法
            is_fedsalora: 是否为 FedSA-LoRA 算法
            
        Returns:
            准备好的模型实例（可直接用于训练）
            
        Note:
            - global_state 应该是完整的 state_dict，包含所有参数
            - private_state: 
              * FedSDG: 只包含私有参数键（_private, lambda_k）
              * FedDPA: 只包含私有参数键（_private）
              * Local-Only: 包含 LoRA 参数键（lora_A, lora_B, mlp_head）
              * FedRep: 包含 Head 参数键（mlp_head, head）
              * Ditto: 不在这里注入（在 update_weights 中处理）
              * FedSA-LoRA: 包含 lora_B 参数键
            - optimizer 在 LocalUpdate 中创建，这里只清理模型梯度
        """
        model = self.get_model(worker_id)
        
        # 优化：合并全局状态和私有状态，只调用一次 load_state_dict
        if (is_fedsdg or is_feddpa or is_local_only or is_fedrep or is_fedsalora) and private_state is not None and len(private_state) > 0:
            # 合并：先复制全局状态，再覆盖私有参数
            # 注意：不修改原 global_state，创建浅拷贝
            merged_state = dict(global_state)  # 浅拷贝，张量不复制
            for name, value in private_state.items():
                if name in merged_state:
                    # 确保设备和 dtype 一致
                    merged_state[name] = value.to(
                        device=self.device, 
                        dtype=merged_state[name].dtype
                    )
            
            # 安全加载状态字典，处理 CUDA 错误
            try:
                # 注意：不调用 torch.cuda.empty_cache()，避免将缓存显存归还给 CUDA runtime
                # 多任务共享 GPU 时，empty_cache() 会导致显存被其他进程抢占
                
                # 同步 CUDA 操作以确保错误准确定位
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                model.load_state_dict(merged_state, strict=True)
                
                # 再次同步确保加载完成
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
            except RuntimeError as e:
                if "CUDA error" in str(e):
                    print(f"[ModelPool] CUDA error during load_state_dict: {e}")
                    print(f"[ModelPool] Attempting safe parameter-by-parameter loading...")
                    
                    # 安全的逐参数加载
                    current_state = model.state_dict()
                    for name, param in merged_state.items():
                        try:
                            if name in current_state:
                                # 检查张量有效性
                                if not torch.isnan(param).any() and not torch.isinf(param).any():
                                    current_state[name].copy_(param)
                                else:
                                    print(f"[ModelPool] Warning: Skipping invalid parameter {name}")
                        except RuntimeError as param_error:
                            print(f"[ModelPool] Error copying parameter {name}: {param_error}")
                            continue
                    
                    # 使用修改后的状态字典
                    model.load_state_dict(current_state, strict=False)
                else:
                    raise e
        else:
            # 非 FedSDG 或无私有状态：直接加载全局状态
            try:
                # 注意：不调用 torch.cuda.empty_cache()，避免显存被其他进程抢占
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                model.load_state_dict(global_state, strict=True)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
            except RuntimeError as e:
                if "CUDA error" in str(e):
                    print(f"[ModelPool] CUDA error during global state loading: {e}")
                    print(f"[ModelPool] Attempting parameter-by-parameter loading...")
                    
                    current_state = model.state_dict()
                    for name, param in global_state.items():
                        try:
                            if name in current_state:
                                if not torch.isnan(param).any() and not torch.isinf(param).any():
                                    current_state[name].copy_(param)
                        except RuntimeError as param_error:
                            print(f"[ModelPool] Error copying parameter {name}: {param_error}")
                            continue
                    
                    model.load_state_dict(current_state, strict=False)
                else:
                    raise e
        
        # Step 3: 设置训练模式
        model.train()
        
        # Step 4: 清理梯度（等效于 zero_grad(set_to_none=True)）
        # 这比 zero_grad() 更高效，避免分配零张量
        for param in model.parameters():
            param.grad = None
        
        return model
    
    def __len__(self) -> int:
        """返回池大小"""
        return self.pool_size
    
    def __repr__(self) -> str:
        return f"ModelPool(pool_size={self.pool_size}, device={self.device})"
