# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model definitions for federated learning."""

from .mlp import MLP
from .cnn import CNNMnist, CNNFashion_Mnist, CNNCifar, modelC
from .vit import ViT, get_pretrained_vit
from .lora import LoRALayer, inject_lora, inject_lora_timm, get_lora_state_dict
from .lora_dpa import (
    DualPathLoRALayer, inject_lora_dpa, inject_lora_dpa_timm, 
    get_dpa_state_dict, set_model_mix_ratio, clear_model_mix_ratio
)
from .builder import (
    build_model, inject_lora_to_model, get_model, 
    get_model_info, ModelBuildError
)

__all__ = [
    # 模型类
    'MLP',
    'CNNMnist', 'CNNFashion_Mnist', 'CNNCifar', 'modelC',
    'ViT', 'get_pretrained_vit',
    # LoRA
    'LoRALayer', 'inject_lora', 'inject_lora_timm', 'get_lora_state_dict',
    # FedDPA LoRA
    'DualPathLoRALayer', 'inject_lora_dpa', 'inject_lora_dpa_timm',
    'get_dpa_state_dict', 'set_model_mix_ratio', 'clear_model_mix_ratio',
    # 工厂函数
    'build_model', 'inject_lora_to_model', 'get_model', 
    'get_model_info', 'ModelBuildError',
]

