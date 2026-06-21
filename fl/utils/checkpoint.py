# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Checkpoint management for federated learning training.

Features:
1. Per-round global model saving
2. LoRA parameter evolution tracking
3. Client local weights saving
4. Aggregation weights and alignment scores saving
5. FedSDG private parameter evolution saving
6. Gate coefficient history tracking
"""

import os
import copy
import pickle
import json
import time
from typing import Dict, List, Optional, Any
import numpy as np
import torch

from .paths import CHECKPOINTS_DIR, ensure_dir, generate_experiment_name


class CheckpointManager:
    """
    Checkpoint manager for federated learning.
    
    Manages training checkpoints, history, and parameter evolution analysis.
    """
    
    def __init__(
        self,
        save_dir: str,
        experiment_name: str,
        alg: str = 'fedavg',
        save_frequency: int = 5,
        save_client_weights: bool = True,
        save_global_model: bool = True,
        save_aggregation_info: bool = True,
        max_checkpoints: int = -1,
        device: str = 'cpu'
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Root save directory
            experiment_name: Experiment name (creates subdirectory)
            alg: FL algorithm type ('fedavg', 'fedlora', 'fedsdg')
            save_frequency: Save frequency (every N rounds)
            save_client_weights: Whether to save client local weights
            save_global_model: Whether to save global model parameters
            save_aggregation_info: Whether to save aggregation info
            max_checkpoints: Maximum checkpoints to keep (-1 for unlimited)
            device: Device for saving ('cpu' recommended)
        """
        self.save_dir = os.path.join(save_dir, experiment_name)
        self.experiment_name = experiment_name
        self.alg = alg
        self.save_frequency = save_frequency
        self.save_client_weights = save_client_weights
        self.save_global_model = save_global_model
        self.save_aggregation_info = save_aggregation_info
        self.max_checkpoints = max_checkpoints
        self.device = device
        
        self._create_directories()
        
        # Lightweight training history
        self.training_history = {
            'rounds': [],
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'test_loss': [],
            'local_test_acc': [],
            'local_test_loss': [],
            'comm_volume_mb': [],
            'timestamps': [],
            'lora_param_norms': [],
            'aggregation_weights': [],
            'alignment_scores': [],
            'gate_values_mean': [],
            'gate_values_std': [],
            'gate_values_min': [],
            'gate_values_max': [],
        }
        
        self.param_evolution = {
            'rounds': [],
            'param_deltas': [],
            'update_magnitudes': [],
            'update_directions': [],
        }
        
        self.checkpoint_list = []
        
        print(f"\n{'='*70}")
        print("[CheckpointManager] Initialized")
        print(f"{'='*70}")
        print(f"  Save directory: {self.save_dir}")
        print(f"  Algorithm: {self.alg}")
        print(f"  Save frequency: every {self.save_frequency} rounds")
        print(f"{'='*70}\n")
    
    def _create_directories(self):
        """Create save directory structure."""
        dirs = [
            self.save_dir,
            os.path.join(self.save_dir, 'checkpoints'),
            os.path.join(self.save_dir, 'global_models'),
            os.path.join(self.save_dir, 'client_weights'),
            os.path.join(self.save_dir, 'aggregation'),
            os.path.join(self.save_dir, 'private_states'),
            os.path.join(self.save_dir, 'param_evolution'),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def _to_cpu(self, state_dict: Dict) -> Dict:
        """Move state_dict to CPU."""
        return {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v 
                for k, v in state_dict.items()}
    
    def _get_param_norm(self, state_dict: Dict, key_filter: str = None) -> float:
        """Compute parameter norm."""
        total_norm = 0.0
        for key, param in state_dict.items():
            if key_filter is None or key_filter in key:
                if isinstance(param, torch.Tensor):
                    total_norm += param.float().norm().item() ** 2
        return np.sqrt(total_norm)
    
    def _compute_param_delta(self, old_state: Dict, new_state: Dict, keys: List[str] = None) -> Dict[str, float]:
        """Compute parameter update magnitude."""
        deltas = {}
        keys_to_check = keys or new_state.keys()
        
        for key in keys_to_check:
            if key in old_state and key in new_state:
                old_param = old_state[key].float()
                new_param = new_state[key].float()
                delta = (new_param - old_param).norm().item()
                deltas[key] = delta
        
        return deltas
    
    def save_round_checkpoint(
        self,
        round_idx: int,
        global_model: torch.nn.Module,
        local_weights: Optional[List[Dict]] = None,
        local_losses: Optional[List[float]] = None,
        selected_clients: Optional[List[int]] = None,
        aggregation_info: Optional[Dict[Any, Any]] = None,
        local_private_states: Optional[Dict[Any, Any]] = None,
        train_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        test_acc: Optional[float] = None,
        test_loss: Optional[float] = None,
        local_test_acc: Optional[float] = None,
        local_test_loss: Optional[float] = None,
        comm_volume_mb: Optional[float] = None,
        previous_global_state: Optional[Dict[str, Any]] = None,
    ):
        """Save single round checkpoint."""
        timestamp = time.time()
        is_detailed_checkpoint = (round_idx % self.save_frequency == 0) or (round_idx == 0)
        
        # Update lightweight history
        self.training_history['rounds'].append(round_idx)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_acc'].append(train_acc)
        self.training_history['test_acc'].append(test_acc)
        self.training_history['test_loss'].append(test_loss)
        self.training_history['local_test_acc'].append(local_test_acc)
        self.training_history['local_test_loss'].append(local_test_loss)
        self.training_history['comm_volume_mb'].append(comm_volume_mb)
        self.training_history['timestamps'].append(timestamp)
        
        # LoRA parameter norm
        if self.alg in ('fedlora', 'fedsdg'):
            global_state = global_model.state_dict()
            lora_norm = self._get_param_norm(global_state, 'lora_')
            self.training_history['lora_param_norms'].append(lora_norm)
        
        # Aggregation info
        if aggregation_info is not None:
            if 'weights' in aggregation_info:
                self.training_history['aggregation_weights'].append(aggregation_info['weights'])
            if 'alignment_scores' in aggregation_info:
                self.training_history['alignment_scores'].append(aggregation_info['alignment_scores'])
        
        # Gate values (FedSDG)
        if self.alg == 'fedsdg':
            gate_values = self._extract_gate_values(global_model, local_private_states)
            if gate_values:
                values = list(gate_values.values())
                self.training_history['gate_values_mean'].append(np.mean(values))
                self.training_history['gate_values_std'].append(np.std(values))
                self.training_history['gate_values_min'].append(np.min(values))
                self.training_history['gate_values_max'].append(np.max(values))
        
        # Parameter evolution
        if previous_global_state is not None:
            current_state = self._to_cpu(global_model.state_dict())
            param_deltas = self._compute_param_delta(previous_global_state, current_state)
            total_magnitude = np.sqrt(sum(d**2 for d in param_deltas.values()))
            
            self.param_evolution['rounds'].append(round_idx)
            self.param_evolution['update_magnitudes'].append(total_magnitude)
            self.param_evolution['param_deltas'].append(param_deltas)
        
        # Save detailed checkpoint
        if is_detailed_checkpoint:
            checkpoint_path = os.path.join(
                self.save_dir, 'checkpoints', f'round_{round_idx:04d}.pkl'
            )
            
            checkpoint_data = {
                'round_idx': round_idx,
                'timestamp': timestamp,
                'alg': self.alg,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'local_test_acc': local_test_acc,
                'local_test_loss': local_test_loss,
                'comm_volume_mb': comm_volume_mb,
            }
            
            if self.save_global_model:
                checkpoint_data['global_state'] = self._to_cpu(global_model.state_dict())
            
            if self.save_client_weights and local_weights is not None:
                checkpoint_data['local_weights'] = [self._to_cpu(w) for w in local_weights]
                checkpoint_data['local_losses'] = local_losses
                checkpoint_data['selected_clients'] = selected_clients
            
            if self.save_aggregation_info and aggregation_info is not None:
                checkpoint_data['aggregation_info'] = aggregation_info
            
            if self.alg == 'fedsdg' and local_private_states is not None:
                private_states_cpu = {}
                for client_id, state in local_private_states.items():
                    private_states_cpu[client_id] = self._to_cpu(state)
                checkpoint_data['private_states'] = private_states_cpu
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.checkpoint_list.append(checkpoint_path)
            
            if self.max_checkpoints > 0 and len(self.checkpoint_list) > self.max_checkpoints:
                old_checkpoint = self.checkpoint_list.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
            
            print(f"  [Checkpoint] Round {round_idx} saved: {checkpoint_path}")
    
    def _extract_gate_values(self, model: torch.nn.Module, local_private_states: Dict = None) -> Dict[str, float]:
        """Extract gate parameter values."""
        gate_values = {}
        
        if local_private_states:
            all_gates = {}
            for client_id, state in local_private_states.items():
                for name, param in state.items():
                    if 'lambda_k_logit' in name:
                        m_k = torch.sigmoid(param).item()
                        layer_name = name.replace('.lambda_k_logit', '')
                        if layer_name not in all_gates:
                            all_gates[layer_name] = []
                        all_gates[layer_name].append(m_k)
            
            for layer_name, values in all_gates.items():
                gate_values[layer_name] = np.mean(values)
        else:
            for name, param in model.named_parameters():
                if 'lambda_k_logit' in name:
                    m_k = torch.sigmoid(param).item()
                    layer_name = name.replace('.lambda_k_logit', '')
                    gate_values[layer_name] = m_k
        
        return gate_values
    
    def save_final(
        self,
        global_model: torch.nn.Module,
        local_private_states: Optional[Dict[Any, Any]] = None,
        final_results: Optional[Dict[Any, Any]] = None,
        args: Any = None,
    ):
        """Save final training results."""
        print(f"\n{'='*70}")
        print("[CheckpointManager] Saving final results")
        print(f"{'='*70}")
        
        # Final model
        final_model_path = os.path.join(self.save_dir, 'final_model.pth')
        torch.save(global_model.state_dict(), final_model_path)
        print(f"  ✓ Final model saved: {final_model_path}")
        
        # Training history
        history_path = os.path.join(self.save_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        print(f"  ✓ Training history saved: {history_path}")
        
        # Parameter evolution
        evolution_path = os.path.join(self.save_dir, 'param_evolution.pkl')
        with open(evolution_path, 'wb') as f:
            pickle.dump(self.param_evolution, f)
        print(f"  ✓ Parameter evolution saved: {evolution_path}")
        
        # FedSDG private states
        if self.alg == 'fedsdg' and local_private_states is not None:
            private_states_cpu = {}
            for client_id, state in local_private_states.items():
                private_states_cpu[client_id] = self._to_cpu(state)
            
            private_path = os.path.join(self.save_dir, 'final_private_states.pkl')
            with open(private_path, 'wb') as f:
                pickle.dump(private_states_cpu, f)
            print(f"  ✓ FedSDG private states saved: {private_path}")
        
        # Summary
        summary = {
            'experiment_name': self.experiment_name,
            'alg': self.alg,
            'training_history': self.training_history,
            'final_results': final_results,
            'args': vars(args) if args else None,
            'checkpoint_list': self.checkpoint_list,
        }
        
        summary_path = os.path.join(self.save_dir, 'experiment_summary.pkl')
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        print(f"  ✓ Experiment summary saved: {summary_path}")
        
        # JSON summary
        json_summary = {
            'experiment_name': self.experiment_name,
            'alg': self.alg,
            'total_rounds': len(self.training_history['rounds']),
            'final_train_acc': self.training_history['train_acc'][-1] if self.training_history['train_acc'] else None,
            'final_test_acc': self.training_history['test_acc'][-1] if self.training_history['test_acc'] else None,
            'final_local_acc': self.training_history['local_test_acc'][-1] if self.training_history['local_test_acc'] else None,
            'checkpoint_count': len(self.checkpoint_list),
        }
        
        if final_results:
            json_summary.update({
                'global_test_acc': final_results.get('global_test_acc'),
                'local_avg_acc': final_results.get('local_avg_acc'),
            })
        
        json_path = os.path.join(self.save_dir, 'summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False)
        print(f"  ✓ JSON summary saved: {json_path}")
        
        print(f"\n  All files saved to: {self.save_dir}")
        print(f"{'='*70}\n")
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict:
        """Load checkpoint from file."""
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_training_history(save_dir: str) -> Dict:
        """Load training history."""
        history_path = os.path.join(save_dir, 'training_history.pkl')
        with open(history_path, 'rb') as f:
            return pickle.load(f)


def create_checkpoint_manager(args, path_project: str, experiment_name: str = None) -> CheckpointManager:
    """
    Create CheckpointManager instance from args.
    
    Args:
        args: Command line arguments
        path_project: Project root directory (unused, kept for compatibility)
        experiment_name: Optional experiment name (if None, will generate from args)
        
    Returns:
        CheckpointManager instance
    """
    # Use provided experiment name or generate one
    if experiment_name is None:
        experiment_name = generate_experiment_name(args)
    
    # Add timestamp to make checkpoint directories unique
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    checkpoint_exp_name = f"{experiment_name}_{timestamp}"
    
    # Use new path structure: outputs/checkpoints/{algorithm}/
    save_dir = os.path.join(CHECKPOINTS_DIR, args.alg)
    ensure_dir(save_dir)
    
    save_frequency = getattr(args, 'save_frequency', 5)
    save_client_weights = getattr(args, 'save_client_weights', True)
    
    return CheckpointManager(
        save_dir=save_dir,
        experiment_name=checkpoint_exp_name,
        alg=args.alg,
        save_frequency=save_frequency,
        save_client_weights=save_client_weights,
        save_global_model=True,
        save_aggregation_info=True,
    )

