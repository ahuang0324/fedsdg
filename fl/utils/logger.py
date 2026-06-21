# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logging utilities for federated learning experiments.
"""


def exp_details(args):
    """Print experiment details and configuration."""
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Algorithm : {args.alg}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    Dirichlet Alpha : {args.dirichlet_alpha}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}')
    
    # LoRA parameters
    if args.alg in ('fedlora', 'fedsdg', 'feddpa', 'fedsalora', 'local_only'):
        print(f'\n    LoRA parameters:')
        print(f'    LoRA rank (r)      : {args.lora_r}')
        print(f'    LoRA alpha         : {args.lora_alpha}')
        print(f'    Train mlp_head     : {bool(args.lora_train_mlp_head)}')
        if args.alg == 'fedsdg':
            print(f'\n    FedSDG specific:')
            print(f'    Dual-path mode     : Enabled (Global + Private branches)')
            print(f'    Private params     : Not communicated (client-local only)')
            agg_method_desc = {
                'fedavg': 'FedAvg uniform-weighted aggregation',
                'alignment': 'Alignment-based weighted FedSDG aggregation'
            }
            print(f'    Server Aggregation : {args.server_agg_method} ({agg_method_desc.get(args.server_agg_method, "unknown")})')
    print()
    return


