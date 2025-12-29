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
            warnings.warn(
                f"Unknown device type '{device_type}'. Using 'cpu' profile."
            )
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
    Allocate heterogeneous LoRA ranks based on device capabilities and importance.
    """
    
    def __init__(self, model_dim: int = 768):
        """
        Initialize RankAllocator.
        
        Args:
            model_dim: Model hidden dimension (e.g., 768 for GPT-2)
        """
        self.model_dim = model_dim
        self.rank_candidates = [4, 8, 16, 32, 64]
        self.profiler = DeviceProfiler()
    
    def allocate_ranks(self, device_profile: Dict, 
                      importance_scores: Dict[str, float],
                      n_layers: int) -> List[int]:
        """
        Allocate heterogeneous ranks per layer.
        
        Args:
            device_profile: Device profile from DeviceProfiler
            importance_scores: Importance score per layer
            n_layers: Number of layers
            
        Returns:
            List of ranks per layer
        """
        memory_budget = device_profile['memory_mb'] * 1024 * 1024  # Convert to bytes
        
        # Reserve 50% for base model
        available_memory = memory_budget * 0.5
        
        # Compute total importance if scores provided per layer
        if any(k.startswith('layer_') for k in importance_scores.keys()):
            # Aggregate importance per layer
            layer_importance = []
            for layer_idx in range(n_layers):
                importance = importance_scores.get(f'layer_{layer_idx}', 1.0 / n_layers)
                layer_importance.append(importance)
        else:
            # Use uniform importance
            layer_importance = [1.0 / n_layers] * n_layers
        
        # Allocate ranks proportional to importance
        ranks_per_layer = []
        
        for layer_idx in range(n_layers):
            importance = layer_importance[layer_idx]
            
            # Compute max rank based on memory
            memory_per_rank = 2 * self.model_dim * 4  # bytes per rank (A and B matrices)
            max_rank_memory = available_memory / (n_layers * memory_per_rank)
            
            # Scale by importance (more important layers get higher ranks)
            scaled_rank = max_rank_memory * (1 + importance)
            
            # Select from candidates
            rank = self._select_rank(scaled_rank)
            ranks_per_layer.append(rank)
        
        return ranks_per_layer
    
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
    
    def allocate_uniform_ranks(self, device_type: str, n_layers: int) -> List[int]:
        """
        Allocate uniform ranks for all layers (simpler baseline).
        
        Args:
            device_type: Type of device
            n_layers: Number of layers
            
        Returns:
            List of uniform ranks
        """
        profile = self.profiler.profile_device(device_type)
        suggested_ranks = profile['suggested_ranks']
        
        # Use the median suggested rank
        uniform_rank = suggested_ranks[len(suggested_ranks) // 2]
        
        return [uniform_rank] * n_layers
    
    def get_rank_for_device(self, device_id: int, device_type: str,
                           task_group_importance: Dict,
                           n_layers: int = 12) -> List[int]:
        """
        Get ranks for specific device.
        
        Args:
            device_id: Device identifier
            device_type: Type of device
            task_group_importance: Importance scores for device's task group
            n_layers: Number of model layers
            
        Returns:
            List of ranks per layer
        """
        profile = self.profiler.profile_device(device_type)
        importance = task_group_importance.get(device_id, {})
        
        return self.allocate_ranks(profile, importance, n_layers)
    
    def validate_memory_constraint(self, ranks: List[int], 
                                   device_profile: Dict) -> Tuple[bool, float]:
        """
        Validate that rank allocation fits in memory.
        
        Args:
            ranks: List of ranks per layer
            device_profile: Device profile
            
        Returns:
            (is_valid, memory_usage_mb)
        """
        # Calculate memory usage
        total_memory = 0
        for rank in ranks:
            # A: (model_dim, rank), B: (rank, model_dim)
            memory_per_layer = 2 * self.model_dim * rank * 4  # float32
            total_memory += memory_per_layer
        
        memory_mb = total_memory / (1024 * 1024)
        available_mb = device_profile['memory_mb'] * 0.5  # 50% reserved for base
        
        is_valid = memory_mb <= available_mb
        
        return is_valid, memory_mb


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
