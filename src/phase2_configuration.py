"""
Phase 2: Heterogeneous Configuration Components
- DeviceProfiler: Profile device capabilities
- WeightImportanceScorer: Compute parameter importance
- RankAllocator: Allocate heterogeneous LoRA ranks
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
import logging
logger = logging.getLogger(__name__)


class DeviceProfiler:
    """
    Profile device capabilities for heterogeneous rank allocation.
    
    Profiles include memory capacity and compute power for different device types.
    """
    
    PROFILES = {
        'cpu': {
            'memory_mb': 2048,
            'compute_ratio': 1.0,
            'suggested_ranks': [4, 8]
        },
        'edge_gpu': {
            'memory_mb': 4096,
            'compute_ratio': 5.0,
            'suggested_ranks': [8, 16]
        },
        'gpu': {
            'memory_mb': 8192,
            'compute_ratio': 10.0,
            'suggested_ranks': [16, 32, 64]
        },
        'smartphone': {
            'memory_mb': 2048,
            'compute_ratio': 0.5,
            'suggested_ranks': [2, 4]
        },
        'tablet': {
            'memory_mb': 3072,
            'compute_ratio': 1.5,
            'suggested_ranks': [4, 8]
        },
        'laptop_cpu': {
            'memory_mb': 4096,
            'compute_ratio': 2.0,
            'suggested_ranks': [8, 16]
        },
        'laptop_gpu': {
            'memory_mb': 8192,
            'compute_ratio': 8.0,
            'suggested_ranks': [16, 32]
        },
        'workstation': {
            'memory_mb': 32768,
            'compute_ratio': 15.0,
            'suggested_ranks': [32, 64, 128]
        }
            ,
            # Explicit device profiles used by experiments
            'cpu_2gb': {
                'memory_mb': 2048,
                'compute_ratio': 1.0,
                'suggested_ranks': [4, 8]
            },
            'tablet_4gb': {
                'memory_mb': 4096,
                'compute_ratio': 1.5,
                'suggested_ranks': [8, 16]
            },
            'laptop_8gb': {
                'memory_mb': 8192,
                'compute_ratio': 4.0,
                'suggested_ranks': [16, 32]
            },
            'gpu_16gb': {
                'memory_mb': 16384,
                'compute_ratio': 12.0,
                'suggested_ranks': [32, 64]
            }
    }
    
    def __init__(self):
        """Initialize DeviceProfiler."""
        pass
    
    def profile_device(self, device_type: str) -> Dict:
        """
        Get device profile.
        
        Args:
            device_type: Type of device ('cpu', 'edge_gpu', 'gpu')
            
        Returns:
            Dict with memory_mb, compute_ratio, suggested_ranks
        """
        if device_type not in self.PROFILES:
            # Downgrade unknown device-type message to debug to avoid noisy warnings
            logger.debug(f"Unknown device type '{device_type}'. Using 'cpu' profile.")
            device_type = 'cpu'
        
        return self.PROFILES[device_type].copy()
    
    def estimate_rank(self, device_type: str, model_dim: int, 
                     target_layers: int) -> int:
        """
        Estimate maximum LoRA rank for device.
        
        Args:
            device_type: Type of device
            model_dim: Model hidden dimension
            target_layers: Number of layers to apply LoRA
            
        Returns:
            Estimated maximum rank
        """
        profile = self.profile_device(device_type)
        
        # Memory per rank: 2 matrices (A and B) * model_dim * rank * 4 bytes (float32)
        # A: (model_dim, rank), B: (rank, model_dim)
        memory_per_rank = 2 * model_dim * 4  # bytes per rank per layer
        total_memory_bytes = profile['memory_mb'] * 1024 * 1024
        
        # Reserve 50% for base model and other overhead
        available_memory = total_memory_bytes * 0.5
        
        # Calculate max rank
        rank_max = available_memory / (memory_per_rank * target_layers)
        
        # Cap at 64 and round down to multiple of 4
        rank = min(int(rank_max), 64)
        rank = (rank // 4) * 4
        
        return max(rank, 4)  # Minimum rank of 4
    
    def get_available_devices(self) -> List[str]:
        """Get list of available device types."""
        return list(self.PROFILES.keys())
    
    def compare_devices(self) -> Dict:
        """
        Compare all device profiles.
        
        Returns:
            Dict with comparison data
        """
        comparison = {}
        for device_type, profile in self.PROFILES.items():
            comparison[device_type] = {
                'memory_gb': profile['memory_mb'] / 1024,
                'compute_ratio': profile['compute_ratio'],
                'suggested_ranks': profile['suggested_ranks']
            }
        return comparison


class WeightImportanceScorer:
    """
    Compute parameter importance for each layer.
    
    Uses gradient-based sensitivity analysis to identify critical parameters.
    """
    
    def __init__(self, model: Optional[torch.nn.Module] = None):
        """
        Initialize WeightImportanceScorer.
        
        Args:
            model: PyTorch model (optional)
        """
        self.model = model
        self.importance_cache = {}
    
    def compute_importance(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute parameter importance via gradient norms.
        
        Args:
            gradients: Dict of {layer_name: gradient_tensor}
            
        Returns:
            Dict of {layer_name: importance_score}
        """
        importance = {}
        total_norm = 0.0
        
        # Compute gradient norm for each layer
        for layer_name, grad_tensor in gradients.items():
            if grad_tensor is None:
                continue
            
            # Move to CPU if on GPU
            if grad_tensor.is_cuda:
                grad_tensor = grad_tensor.cpu()
            
            # Compute L2 norm
            grad_norm = torch.norm(grad_tensor).item()
            importance[layer_name] = grad_norm
            total_norm += grad_norm
        
        # Normalize to sum to 1.0
        if total_norm > 1e-8:
            for layer_name in importance:
                importance[layer_name] /= total_norm
        
        # Cache results
        self.importance_cache = importance
        
        return importance
    
    def compute_importance_from_model(self, batch_data: Dict, 
                                     loss_fn: Optional[callable] = None) -> Dict[str, float]:
        """
        Compute importance directly from model using dummy forward/backward pass.
        
        Args:
            batch_data: Input batch (dict with 'input' and optionally 'labels')
            loss_fn: Loss function (default: cross entropy)
            
        Returns:
            Dict of {layer_name: importance_score}
        """
        if self.model is None:
            raise ValueError("Model not set. Use compute_importance() instead.")
        
        # Zero gradients
        self.model.zero_grad()
        
        # Forward pass
        if isinstance(batch_data, dict):
            inputs = batch_data.get('input', batch_data.get('inputs'))
            labels = batch_data.get('labels', batch_data.get('targets'))
        else:
            inputs = batch_data
            labels = None
        
        outputs = self.model(inputs)
        
        # Compute loss
        if loss_fn is not None:
            loss = loss_fn(outputs, labels)
        elif labels is not None:
            loss = F.cross_entropy(outputs, labels)
        else:
            # Dummy loss if no labels
            loss = outputs.mean()
        
        # Backward pass
        loss.backward()
        
        # Extract gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        # Compute importance
        return self.compute_importance(gradients)
    
    def get_layer_importance(self, layer_name: str, 
                            importance_dict: Optional[Dict] = None) -> float:
        """
        Get importance for specific layer.
        
        Args:
            layer_name: Name of layer
            importance_dict: Importance dict (uses cached if None)
            
        Returns:
            Importance score for layer
        """
        if importance_dict is None:
            importance_dict = self.importance_cache
        
        # Sum importance for all parameters in this layer
        total_importance = 0.0
        for param_name, importance in importance_dict.items():
            if param_name.startswith(layer_name):
                total_importance += importance
        
        return total_importance
    
    def get_layer_ranking(self, importance_dict: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """
        Get layers ranked by importance.
        
        Args:
            importance_dict: Importance dict (uses cached if None)
            
        Returns:
            List of (layer_name, importance) sorted by importance (descending)
        """
        if importance_dict is None:
            importance_dict = self.importance_cache
        
        # Group by layer prefix
        layer_importance = {}
        for param_name, importance in importance_dict.items():
            # Extract layer name (e.g., 'layer_0' from 'layer_0.weight')
            layer_name = param_name.split('.')[0]
            if layer_name not in layer_importance:
                layer_importance[layer_name] = 0.0
            layer_importance[layer_name] += importance
        
        # Sort by importance
        ranked = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
        return ranked


class RankAllocator:
    """
    Allocate heterogeneous LoRA ranks based on HSplitLoRA formulation.
    
    Follows the constraint: Σ(2·d·r_ℓ·b) ≤ C_mem
    where C_mem = M_device · (1 - α_base - α_act - α_opt)
    """
    
    def __init__(self, model_dim: int = 768, bytes_per_param: int = 4):
        """
        Initialize RankAllocator.
        
        Args:
            model_dim: Model hidden dimension (e.g., 768 for GPT-2)
            bytes_per_param: Bytes per parameter (4 for fp32, 2 for fp16)
        """
        self.model_dim = model_dim
        self.bytes_per_param = bytes_per_param
        self.rank_candidates = [4, 8, 16, 32, 64]
        self.profiler = DeviceProfiler()
        
        # Memory fraction reservations (HSplitLoRA style)
        self.alpha_base = 0.35      # Base model (frozen)
        self.alpha_act = 0.25       # Activations + gradients
        self.alpha_opt = 0.15       # Optimizer states (Adam: 2x params)
        # Remaining: 1 - 0.35 - 0.25 - 0.15 = 0.25 (25% for LoRA adapters)
    
    def compute_adapter_memory_budget(self, device_memory_mb: float) -> float:
        """
        Compute available memory for LoRA adapters following HSplitLoRA.
        
        C_mem = M_device · (1 - α_base - α_act - α_opt)
        
        Args:
            device_memory_mb: Total device memory in MB
            
        Returns:
            Available adapter memory in bytes
        """
        M_device = device_memory_mb * 1024 * 1024  # Convert to bytes
        C_mem = M_device * (1.0 - self.alpha_base - self.alpha_act - self.alpha_opt)
        return C_mem
    
    def compute_rank_memory(self, rank: int, model_dim: Optional[int] = None) -> float:
        """
        Compute memory for single LoRA adapter per layer.
        
        M_param(W, r) = 2·d·r·b
        
        Args:
            rank: LoRA rank
            model_dim: Model dimension (uses self.model_dim if None)
            
        Returns:
            Memory in bytes
        """
        d = model_dim if model_dim is not None else self.model_dim
        # A: (d, r), B: (r, d) → total params = d·r + r·d = 2·d·r
        return 2 * d * rank * self.bytes_per_param
    
    def allocate_ranks(self, device_profile: Dict, 
                      importance_scores: Dict[str, float],
                      n_layers: int,
                      split_point: Optional[int] = None) -> List[int]:
        """
        Allocate heterogeneous ranks per layer using HSplitLoRA formulation.
        
        Constraint: Σ(2·d·r_ℓ·b) ≤ C_mem
        
        Uses greedy importance-based allocation:
        1. Sort layers by importance (descending)
        2. For each layer, assign highest rank that keeps sum ≤ C_mem
        3. Respects device suggested_ranks as capability ceiling
        
        Args:
            device_profile: Device profile from DeviceProfiler
            importance_scores: Importance score per layer (normalized to sum to 1)
            n_layers: Total number of layers in model
            split_point: If specified, only allocate for client-side layers (0 to split_point-1)
            
        Returns:
            List of ranks per layer
        """
        # Compute adapter memory budget
        C_mem = self.compute_adapter_memory_budget(device_profile['memory_mb'])
        
        # Get device capability ceiling (max suggested rank)
        max_device_rank = max(device_profile.get('suggested_ranks', [64]))
        
        # Determine which layers to allocate (client-side only if split_point given)
        if split_point is not None:
            layers_to_allocate = list(range(split_point))
        else:
            layers_to_allocate = list(range(n_layers))
        
        # Extract importance per layer
        layer_importance = []
        for layer_idx in layers_to_allocate:
            importance = importance_scores.get(f'layer_{layer_idx}', 1.0 / len(layers_to_allocate))
            layer_importance.append((layer_idx, importance))
        
        # Normalize importance to sum to 1
        total_importance = sum(imp for _, imp in layer_importance)
        if total_importance > 1e-8:
            layer_importance = [(idx, imp / total_importance) for idx, imp in layer_importance]
        
        # **FIXED ALGORITHM**: Budget-proportional allocation
        # Old greedy algorithm failed because incrementally upgrading each layer
        # allowed all layers to reach same rank (budget was permissive enough)
        
        # 1. Find maximum uniform rank that fits in budget
        n_alloc = len(layers_to_allocate)
        if n_alloc == 0:
            return [self.rank_candidates[0]] * n_layers
        
        min_rank = self.rank_candidates[0]
        max_rank = min(self.rank_candidates[-1], max_device_rank)
        
        # Find best uniform rank (baseline)
        best_uniform_rank = min_rank
        for r in self.rank_candidates:
            if r > max_rank:
                break
            memory = n_alloc * self.compute_rank_memory(r)
            if memory <= C_mem:
                best_uniform_rank = r
        
        # 2. Define total rank budget (ensures same memory as uniform allocation)
        total_rank_budget = n_alloc * best_uniform_rank
        
        # 3. Distribute budget proportionally to importance
        ranks = [min_rank] * n_layers
        for layer_idx, importance in layer_importance:
            # Allocate rank proportional to importance
            target_rank = importance * total_rank_budget
            
            # Clamp to device capability
            target_rank = min(target_rank, max_device_rank)
            
            # Round to nearest valid candidate
            ranks[layer_idx] = self._select_rank(target_rank)
        
        # 4. Validate and adjust if over budget
        current_memory = sum(
            self.compute_rank_memory(ranks[layer_idx]) 
            for layer_idx in layers_to_allocate
        )
        
        # If over budget, reduce ranks starting from least important
        layer_importance_sorted = sorted(layer_importance, key=lambda x: x[1])  # ascending
        attempt = 0
        max_attempts = n_alloc * len(self.rank_candidates)
        while current_memory > C_mem and attempt < max_attempts:
            attempt += 1
            downgraded = False
            # Find least important layer with rank > min
            for layer_idx, _ in layer_importance_sorted:
                if ranks[layer_idx] > min_rank:
                    # Downgrade to next lower rank
                    old_rank = ranks[layer_idx]
                    try:
                        rank_idx = self.rank_candidates.index(old_rank)
                        if rank_idx > 0:
                            new_rank = self.rank_candidates[rank_idx - 1]
                            old_memory = self.compute_rank_memory(old_rank)
                            new_memory = self.compute_rank_memory(new_rank)
                            ranks[layer_idx] = new_rank
                            current_memory += (new_memory - old_memory)
                            downgraded = True
                            
                            if current_memory <= C_mem:
                                break
                    except ValueError:
                        continue
            
            if not downgraded:
                break  # No more downgrades possible
        
        # Log allocation summary
        allocation_log = []
        for layer_idx in layers_to_allocate:
            importance = next((imp for idx, imp in layer_importance if idx == layer_idx), 0.0)
            allocation_log.append(f"L{layer_idx}:r{ranks[layer_idx]}(imp={importance:.3f})")
        
        # Log allocation summary for debugging (only if varies)
        if len(set(ranks)) > 1:
            logger.info(f"Heterogeneous allocation: {allocation_log}")
        
        return ranks
    
    def _select_rank(self, target_rank: float) -> int:
        """
        Select closest valid rank from candidates.
        
        Args:
            target_rank: Target rank (may be fractional)
            
        Returns:
            Valid rank from candidates
        """
        # Find closest rank that doesn't exceed target
        valid_ranks = [r for r in self.rank_candidates if r <= target_rank]
        if valid_ranks:
            return valid_ranks[-1]  # Return largest valid rank
        else:
            return self.rank_candidates[0]  # Return minimum rank
    
    def allocate_uniform_ranks(self, device_type: str, n_layers: int,
                              split_point: Optional[int] = None) -> List[int]:
        """
        Allocate uniform ranks for all layers (simpler baseline).
        
        Uses HSplitLoRA constraint to find maximum uniform rank:
        L_client · 2·d·r·b ≤ C_mem
        → r ≤ C_mem / (L_client · 2·d·b)
        
        Args:
            device_type: Type of device
            n_layers: Total number of layers
            split_point: If specified, only allocate for client-side layers
            
        Returns:
            List of uniform ranks
        """
        profile = self.profiler.profile_device(device_type)
        C_mem = self.compute_adapter_memory_budget(profile['memory_mb'])
        
        # Determine number of client-side layers
        if split_point is not None:
            L_client = split_point
        else:
            L_client = n_layers
        
        # Compute maximum uniform rank under constraint
        # L_client · 2·d·r·b ≤ C_mem
        # r ≤ C_mem / (L_client · 2·d·b)
        max_uniform_rank = C_mem / (L_client * 2 * self.model_dim * self.bytes_per_param)
        
        # Select largest candidate rank that fits
        uniform_rank = self._select_rank(max_uniform_rank)
        
        # Fall back to suggested ranks if computed rank seems unreasonable
        suggested_ranks = profile['suggested_ranks']
        if uniform_rank > suggested_ranks[-1]:
            uniform_rank = suggested_ranks[-1]  # Cap at max suggested
        elif uniform_rank < suggested_ranks[0]:
            uniform_rank = suggested_ranks[0]  # Floor at min suggested
        
        return [uniform_rank] * n_layers
    
    def get_rank_for_device(self, device_id: int, device_type: str,
                           task_group_importance: Dict,
                           n_layers: int = 12,
                           split_point: Optional[int] = None) -> List[int]:
        """
        Get ranks for specific device following HSplitLoRA formulation.
        
        Args:
            device_id: Device identifier
            device_type: Type of device
            task_group_importance: Importance scores for device's task group
            n_layers: Total number of model layers
            split_point: Split point (allocates for client-side layers only)
            
        Returns:
            List of ranks per layer
        """
        profile = self.profiler.profile_device(device_type)
        importance = task_group_importance.get(device_id, {})
        
        return self.allocate_ranks(profile, importance, n_layers, split_point)
    
    def compute_optimal_split_point(self, device_profile: Dict,
                                   importance_scores: Dict[str, float],
                                   n_layers: int,
                                   candidate_splits: Optional[List[int]] = None) -> Tuple[int, List[int], float]:
        """
        Find optimal split point following HSplitLoRA.
        
        For each candidate split S, computes client-side LoRA allocation
        and selects split that maximizes importance-weighted ranks under budget.
        
        Objective: max_S Σ(I_ℓ · r_ℓ) for ℓ in client-side layers
        Subject to: Σ(2·d·r_ℓ·b) ≤ C_mem
        
        Args:
            device_profile: Device profile
            importance_scores: Importance scores per layer
            n_layers: Total layers
            candidate_splits: List of candidate split points (default: [4, 6, 8])
            
        Returns:
            Tuple of (best_split, best_ranks, best_utility)
        """
        if candidate_splits is None:
            # Default: try splitting at 1/3, 1/2, 2/3 of model
            candidate_splits = [n_layers // 3, n_layers // 2, 2 * n_layers // 3]
        
        best_split = candidate_splits[0]
        best_ranks = None
        best_utility = -float('inf')
        
        for split_point in candidate_splits:
            # Allocate ranks for this split
            ranks = self.allocate_ranks(device_profile, importance_scores, 
                                       n_layers, split_point)
            
            # Compute utility: Σ(I_ℓ · r_ℓ) for client-side layers
            utility = 0.0
            for layer_idx in range(split_point):
                importance = importance_scores.get(f'layer_{layer_idx}', 0.0)
                utility += importance * ranks[layer_idx]
            
            # Check if valid
            is_valid, _, _ = self.validate_memory_constraint(ranks, device_profile, split_point)
            
            if is_valid and utility > best_utility:
                best_split = split_point
                best_ranks = ranks
                best_utility = utility
        
        return best_split, best_ranks, best_utility
    
    def validate_memory_constraint(self, ranks: List[int], 
                                   device_profile: Dict,
                                   split_point: Optional[int] = None) -> Tuple[bool, float]:
        """
        Validate that rank allocation satisfies HSplitLoRA constraint.
        
        Constraint: Σ(2·d·r_ℓ·b) ≤ C_mem
        
        Args:
            ranks: List of ranks per layer
            device_profile: Device profile
            split_point: If specified, only validate client-side layers
            
        Returns:
            Tuple of (is_valid, memory_usage_mb)
        """
        # Determine which layers to validate
        if split_point is not None:
            layers_to_check = list(range(split_point))
        else:
            layers_to_check = list(range(len(ranks)))
        
        # Calculate total adapter memory: Σ(2·d·r_ℓ·b)
        total_adapter_memory = 0.0
        for layer_idx in layers_to_check:
            rank = ranks[layer_idx]
            memory = self.compute_rank_memory(rank)
            total_adapter_memory += memory
        
        # Compute budget
        C_mem = self.compute_adapter_memory_budget(device_profile['memory_mb'])
        
        # Check constraint
        is_valid = total_adapter_memory <= C_mem
        
        # Convert to MB for readability
        adapter_mb = total_adapter_memory / (1024 * 1024)
        budget_mb = C_mem / (1024 * 1024)
        
        # Return validation result and memory usage
        return is_valid, adapter_mb

    def get_memory_breakdown(self, ranks: List[int], 
                            device_profile: Dict,
                            split_point: Optional[int] = None) -> Dict:
        """
        Get detailed memory breakdown.
        
        Args:
            ranks: List of ranks per layer
            device_profile: Device profile
            split_point: If specified, only validate client-side layers
            
        Returns:
            Dictionary with detailed breakdown
        """
        # Determine which layers to validate
        if split_point is not None:
            layers_to_check = list(range(split_point))
        else:
            layers_to_check = list(range(len(ranks)))
        
        # Calculate total adapter memory
        total_adapter_memory = 0.0
        for layer_idx in layers_to_check:
            rank = ranks[layer_idx]
            memory = self.compute_rank_memory(rank)
            total_adapter_memory += memory
        
        C_mem = self.compute_adapter_memory_budget(device_profile['memory_mb'])
        adapter_mb = total_adapter_memory / (1024 * 1024)
        budget_mb = C_mem / (1024 * 1024)
        
        M_device = device_profile['memory_mb']
        breakdown = {
            'total_device_memory_mb': M_device,
            'base_model_reserved_mb': M_device * self.alpha_base,
            'activations_reserved_mb': M_device * self.alpha_act,
            'optimizer_reserved_mb': M_device * self.alpha_opt,
            'adapter_budget_mb': budget_mb,
            'adapter_used_mb': adapter_mb,
            'utilization_percent': (adapter_mb / budget_mb * 100) if budget_mb > 0 else 0,
        }
        
        return breakdown


def visualize_rank_allocation(ranks: List[int], importance_scores: Dict,
                              device_type: str, save_path: Optional[str] = None):
    """
    Visualize rank allocation per layer.
    
    Args:
        ranks: List of ranks per layer
        importance_scores: Importance scores
        device_type: Device type
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        return
    
    n_layers = len(ranks)
    layer_indices = list(range(n_layers))
    
    # Extract importance per layer
    importance = []
    for i in range(n_layers):
        imp = importance_scores.get(f'layer_{i}', 0.0)
        importance.append(imp)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot ranks
    ax1.bar(layer_indices, ranks, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('LoRA Rank', fontsize=12)
    ax1.set_title(f'Rank Allocation per Layer ({device_type.upper()})', 
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot importance
    ax2.bar(layer_indices, importance, color='coral', alpha=0.7)
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Importance Score', fontsize=12)
    ax2.set_title('Parameter Importance per Layer', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Quick test
    print("Phase 2 Configuration Module Loaded Successfully!")
    print("=" * 60)
    print("Available classes:")
    print("  - DeviceProfiler")
    print("  - WeightImportanceScorer")
    print("  - RankAllocator")
    print("Available functions:")
    print("  - visualize_rank_allocation()")
    print("=" * 60)
