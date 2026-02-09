"""
Improved Split Layer Selection for ATLAS

Advances over current implementation:
1. Layer-wise importance scoring (gradient magnitudes, activations)
2. Communication cost modeling (activation sizes, compression)
3. Task-specific optimization
4. Bandwidth-aware selection

References:
- SplitLoRA: Optimal split based on memory-communication tradeoff
- HSplitLoRA: Heterogeneous device-aware splitting
- VFLAIR-LLM: Vertical FL with adaptive splitting

Date: February 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoModel, AutoConfig
import warnings


class ImprovedSplitSelector:
    """
    Adaptive split point selection considering:
    - Memory constraints (device-specific)
    - Communication costs (bandwidth, activation sizes)
    - Layer importance (gradient magnitudes, task-relevance)
    - Task heterogeneity
    """
    
    def __init__(
        self,
        model_name: str,
        device_profiles: Dict[int, Dict],
        task_assignments: Dict[int, str],
        bandwidth_mbps: float = 10.0,  # Average network bandwidth
        compression_ratio: float = 1.0  # Activation compression (1.0 = no compression)
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            device_profiles: Per-client device capabilities {client_id: profile}
            task_assignments: Per-client task {client_id: task_name}
            bandwidth_mbps: Network bandwidth in Mbps
            compression_ratio: Activation compression ratio (< 1.0 = compressed)
        """
        self.model_name = model_name
        self.device_profiles = device_profiles
        self.task_assignments = task_assignments
        self.bandwidth_mbps = bandwidth_mbps
        self.compression_ratio = compression_ratio
        
        # Load model config
        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.n_layers = getattr(
                self.config, 
                'n_layer', 
                getattr(self.config, 'num_hidden_layers', 12)
            )
            self.hidden_size = getattr(self.config, 'hidden_size', 768)
        except:
            warnings.warn(f"Could not load config for {model_name}, using defaults")
            self.n_layers = 12
            self.hidden_size = 768
        
        # Cache for layer importance scores
        self.layer_importance = None
    
    def compute_optimal_split(
        self,
        client_id: int,
        fingerprint_gradients: Optional[np.ndarray] = None
    ) -> int:
        """
        Compute optimal split point for a specific client.
        
        Args:
            client_id: Client identifier
            fingerprint_gradients: Gradient fingerprint for importance estimation
            
        Returns:
            Optimal split layer index
        """
        device_profile = self.device_profiles.get(client_id, {})
        memory_mb = device_profile.get('memory_mb', 2048)
        
        # Candidate split points (middle third of network)
        min_split = max(2, self.n_layers // 3)
        max_split = min(self.n_layers - 2, 2 * self.n_layers // 3)
        
        best_split = min_split
        best_score = -np.inf
        
        scores = []
        
        for split in range(min_split, max_split + 1):
            # Component scores
            memory_score = self._score_memory_constraint(split, memory_mb)
            comm_score = self._score_communication_cost(split)
            importance_score = self._score_layer_importance(split, fingerprint_gradients)
            balance_score = self._score_workload_balance(split)
            
            # Weighted combination
            total_score = (
                0.35 * memory_score +      # Must fit in memory
                0.30 * comm_score +         # Minimize communication
                0.25 * importance_score +   # Keep important layers on client
                0.10 * balance_score        # Balance client/server workload
            )
            
            scores.append({
                'split': split,
                'total': total_score,
                'memory': memory_score,
                'comm': comm_score,
                'importance': importance_score,
                'balance': balance_score
            })
            
            if total_score > best_score:
                best_score = total_score
                best_split = split
        
        # Log decision
        print(f"[SPLIT] Client {client_id}: Optimal split = Layer {best_split}/{self.n_layers}")
        print(f"        Score: {best_score:.3f} (mem={scores[best_split-min_split]['memory']:.2f}, "
              f"comm={scores[best_split-min_split]['comm']:.2f}, "
              f"imp={scores[best_split-min_split]['importance']:.2f})")
        
        return best_split
    
    def _score_memory_constraint(self, split: int, memory_mb: float) -> float:
        """
        Score based on memory constraint satisfaction.
        
        Returns:
            Score in [0, 1] where 1 = fits comfortably, 0 = exceeds budget
        """
        # Estimate memory requirement for split
        # Base model memory (proportional to layers)
        base_memory_per_layer = 50  # MB per transformer layer (rough estimate)
        base_memory = split * base_memory_per_layer
        
        # LoRA adapter memory (2 * d * r * 4 bytes per layer)
        # Assume average rank of 16
        lora_memory_per_layer = 2 * self.hidden_size * 16 * 4 / (1024**2)  # MB
        lora_memory = split * lora_memory_per_layer
        
        # Activation memory (batch_size * seq_len * hidden_size * 4 bytes)
        # Assume batch_size=16, seq_len=128
        activation_memory = 16 * 128 * self.hidden_size * 4 / (1024**2)  # MB
        
        # Total estimated memory
        total_memory = base_memory + lora_memory + activation_memory
        
        # Score: sigmoid-based soft constraint
        memory_ratio = total_memory / (memory_mb * 0.8)  # 80% safety margin
        
        if memory_ratio <= 1.0:
            score = 1.0  # Fits comfortably
        else:
            score = np.exp(-(memory_ratio - 1.0) * 3)  # Exponential penalty
        
        return score
    
    def _score_communication_cost(self, split: int) -> float:
        """
        Score based on communication efficiency.
        
        Lower split = more layers on server = larger activations to communicate
        Higher split = fewer layers on server = smaller activations
        
        Returns:
            Score in [0, 1] where 1 = minimal communication
        """
        # Activation size to communicate (proportional to remaining layers)
        # Each layer produces activations of size: batch_size * seq_len * hidden_size
        # Assume batch_size=16, seq_len=128
        activation_size_mb = 16 * 128 * self.hidden_size * 4 / (1024**2)
        activation_size_mb *= self.compression_ratio  # Apply compression
        
        # Communication cost: time to transfer (upload + download)
        # Upload: client → server (activations)
        # Download: server → client (gradients, similar size)
        transfer_time = (2 * activation_size_mb * 8) / self.bandwidth_mbps  # seconds
        
        # Score: prefer splits that minimize transfer time
        # Normalize by maximum possible time (all layers on server)
        max_transfer_time = (2 * activation_size_mb * self.n_layers * 8) / self.bandwidth_mbps
        
        if max_transfer_time > 0:
            score = 1.0 - (transfer_time / max_transfer_time)
        else:
            score = 1.0  # No communication
        
        # Additional factor: prefer higher splits (more on client)
        # This captures the intuition that client-side is cheaper
        split_ratio = split / self.n_layers
        score = 0.7 * score + 0.3 * split_ratio
        
        return np.clip(score, 0, 1)
    
    def _score_layer_importance(
        self, 
        split: int, 
        fingerprint_gradients: Optional[np.ndarray]
    ) -> float:
        """
        Score based on keeping important layers on client.
        
        Important layers (high gradient magnitudes, task-specific features)
        should stay on client for better personalization.
        
        Returns:
            Score in [0, 1] where 1 = important layers on client
        """
        if fingerprint_gradients is None:
            # Heuristic: middle layers are most important for task features
            # Score based on how close split is to middle
            middle = self.n_layers / 2
            distance_from_middle = abs(split - middle)
            score = 1.0 - (distance_from_middle / (self.n_layers / 2))
            return np.clip(score, 0, 1)
        
        # Compute layer-wise gradient magnitudes
        if self.layer_importance is None:
            self.layer_importance = self._compute_layer_importance(fingerprint_gradients)
        
        # Score: proportion of important layers kept on client
        if len(self.layer_importance) >= split:
            client_importance = np.sum(self.layer_importance[:split])
            total_importance = np.sum(self.layer_importance)
            
            if total_importance > 0:
                score = client_importance / total_importance
            else:
                score = 0.5
        else:
            score = 0.5
        
        return np.clip(score, 0, 1)
    
    def _score_workload_balance(self, split: int) -> float:
        """
        Score based on balanced workload between client and server.
        
        Balanced splits (near 50/50) are preferred for load distribution.
        
        Returns:
            Score in [0, 1] where 1 = perfectly balanced
        """
        split_ratio = split / self.n_layers
        ideal_ratio = 0.5
        
        # Distance from ideal 50/50 split
        distance = abs(split_ratio - ideal_ratio)
        
        # Score: closer to 50/50 is better
        score = 1.0 - (distance / 0.5)
        
        return np.clip(score, 0, 1)
    
    def _compute_layer_importance(self, fingerprint_gradients: np.ndarray) -> np.ndarray:
        """
        Compute layer-wise importance from gradient fingerprints.
        
        Args:
            fingerprint_gradients: Gradient fingerprint of shape (n_params,)
            
        Returns:
            Array of shape (n_layers,) with importance scores
        """
        # Assume gradients are ordered by layers
        # Split into n_layers chunks and compute magnitude per layer
        grad_per_layer = np.array_split(fingerprint_gradients, self.n_layers)
        
        importance = np.array([
            np.linalg.norm(grad_chunk) if len(grad_chunk) > 0 else 0.0
            for grad_chunk in grad_per_layer
        ])
        
        # Normalize
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance
    
    def compute_task_specific_splits(
        self,
        task_fingerprints: Dict[str, np.ndarray]
    ) -> Dict[str, int]:
        """
        Compute optimal split points per task (for task-heterogeneous FL).
        
        Args:
            task_fingerprints: Per-task gradient fingerprints {task_name: gradients}
            
        Returns:
            Optimal splits per task {task_name: split_layer}
        """
        task_splits = {}
        
        for task_name, fingerprint in task_fingerprints.items():
            # Compute importance for this task
            self.layer_importance = self._compute_layer_importance(fingerprint)
            
            # Find clients working on this task
            task_clients = [
                cid for cid, t in self.task_assignments.items() 
                if t == task_name
            ]
            
            if not task_clients:
                continue
            
            # Use average device profile for this task
            avg_memory = np.mean([
                self.device_profiles.get(cid, {}).get('memory_mb', 2048)
                for cid in task_clients
            ])
            
            # Compute optimal split for average device
            min_split = max(2, self.n_layers // 3)
            max_split = min(self.n_layers - 2, 2 * self.n_layers // 3)
            
            best_split = min_split
            best_score = -np.inf
            
            for split in range(min_split, max_split + 1):
                memory_score = self._score_memory_constraint(split, avg_memory)
                comm_score = self._score_communication_cost(split)
                importance_score = self._score_layer_importance(split, fingerprint)
                balance_score = self._score_workload_balance(split)
                
                total_score = (
                    0.35 * memory_score +
                    0.30 * comm_score +
                    0.25 * importance_score +
                    0.10 * balance_score
                )
                
                if total_score > best_score:
                    best_score = total_score
                    best_split = split
            
            task_splits[task_name] = best_split
            print(f"[TASK SPLIT] {task_name}: Layer {best_split}/{self.n_layers} (score={best_score:.3f})")
        
        return task_splits


def adaptive_split_with_bandwidth(
    model_name: str,
    device_profiles: Dict[int, Dict],
    bandwidth_mbps: float = 10.0
) -> Dict[int, int]:
    """
    Convenience function: Compute per-client adaptive splits.
    
    Args:
        model_name: Model identifier
        device_profiles: Device capabilities per client
        bandwidth_mbps: Network bandwidth
        
    Returns:
        Optimal splits per client {client_id: split_layer}
    """
    # Dummy task assignments (all same task for simple case)
    task_assignments = {cid: 'task_0' for cid in device_profiles.keys()}
    
    selector = ImprovedSplitSelector(
        model_name=model_name,
        device_profiles=device_profiles,
        task_assignments=task_assignments,
        bandwidth_mbps=bandwidth_mbps
    )
    
    client_splits = {}
    for client_id in device_profiles.keys():
        split = selector.compute_optimal_split(client_id)
        client_splits[client_id] = split
    
    return client_splits


# Example usage
if __name__ == '__main__':
    # Test with sample device profiles
    device_profiles = {
        0: {'memory_mb': 2048, 'device_type': 'cpu_2gb'},
        1: {'memory_mb': 4096, 'device_type': 'tablet_4gb'},
        2: {'memory_mb': 8192, 'device_type': 'laptop_8gb'},
        3: {'memory_mb': 16384, 'device_type': 'gpu_16gb'},
    }
    
    task_assignments = {
        0: 'sst2',
        1: 'sst2',
        2: 'mrpc',
        3: 'mrpc'
    }
    
    selector = ImprovedSplitSelector(
        model_name='distilbert-base-uncased',
        device_profiles=device_profiles,
        task_assignments=task_assignments,
        bandwidth_mbps=10.0
    )
    
    print("Per-client optimal splits:")
    for client_id in device_profiles.keys():
        split = selector.compute_optimal_split(client_id)
        print(f"  Client {client_id}: Split at layer {split}")
