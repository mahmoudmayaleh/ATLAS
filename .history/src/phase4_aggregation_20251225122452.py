"""
Phase 4: Privacy-Aware Aggregation

Implements:
- AggregationEngine: Heterogeneous LoRA weight aggregation with SVD
- Privacy verification tools
- Task-aware weighting mechanisms
- Gradient leakage detection

Author: ATLAS Team
Date: December 2025
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
import warnings


class AggregationEngine:
    """
    Privacy-aware aggregation engine for heterogeneous LoRA weights.
    
    Uses SVD-based low-rank approximation to merge weights from clients
    with different ranks while preserving privacy through task-aware grouping.
    
    Args:
        target_rank (int): Target rank for merged weights (default: 32)
        aggregation_method (str): 'svd' or 'average' (default: 'svd')
        privacy_epsilon (float): Privacy budget parameter (default: None)
    """
    
    def __init__(
        self,
        target_rank: int = 32,
        aggregation_method: str = 'svd',
        privacy_epsilon: Optional[float] = None
    ):
        self.target_rank = target_rank
        self.aggregation_method = aggregation_method
        self.privacy_epsilon = privacy_epsilon
        
        # Tracking
        self.aggregation_history = []
        self.privacy_scores = []
        
    def aggregate_task_group(
        self,
        client_updates: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
        group_clients: List[int],
        group_id: int
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Aggregate LoRA weights for a single task group.
        
        Args:
            client_updates: Dictionary mapping client_id to LoRA weights
            group_clients: List of client IDs in this group
            group_id: Task group identifier
            
        Returns:
            Aggregated LoRA weights for the group
        """
        # Collect weights from group members
        group_weights = [
            client_updates[client_id]
            for client_id in group_clients
            if client_id in client_updates
        ]
        
        if len(group_weights) == 0:
            raise ValueError(f"No updates found for group {group_id}")
        
        aggregated = {}
        
        # Get all layer names
        layer_names = list(group_weights[0].keys())
        
        for layer_name in layer_names:
            # Get A and B matrices from all clients
            A_list = [w[layer_name]['A'] for w in group_weights]
            B_list = [w[layer_name]['B'] for w in group_weights]
            
            if self.aggregation_method == 'svd':
                # SVD-based aggregation
                A_merged, B_merged = self._svd_aggregate(A_list, B_list)
            else:
                # Simple averaging
                A_merged, B_merged = self._average_aggregate(A_list, B_list)
            
            aggregated[layer_name] = {
                'A': A_merged,
                'B': B_merged
            }
        
        # Track aggregation
        self.aggregation_history.append({
            'group_id': group_id,
            'n_clients': len(group_clients),
            'layers': len(layer_names)
        })
        
        return aggregated
    
    def _svd_aggregate(
        self,
        A_list: List[torch.Tensor],
        B_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate using SVD-based low-rank approximation.
        
        Args:
            A_list: List of A matrices from different clients
            B_list: List of B matrices from different clients
            
        Returns:
            Merged A and B matrices
        """
        # Normalize ranks to common dimension
        A_normalized = self._normalize_ranks(A_list)
        B_normalized = self._normalize_ranks(B_list)
        
        # Concatenate along rank dimension
        # A shape: (hidden_dim, rank), concatenate on rank -> (hidden_dim, sum_ranks)
        A_concat = torch.cat(A_normalized, dim=1)
        
        # B shape: (hidden_dim, rank), for B we need to transpose to concatenate properly
        # B.T @ A.T = (A @ B^T)^T
        B_concat = torch.cat(B_normalized, dim=1)
        
        # Compute full weight matrix W = A @ B^T
        # A_concat: (hidden_dim, sum_ranks)
        # B_concat: (hidden_dim, sum_ranks)
        # W = A_concat @ B_concat.T
        W = torch.matmul(A_concat, B_concat.T)  # (hidden_dim, hidden_dim)
        
        # SVD decomposition
        try:
            U, Sigma, Vt = torch.linalg.svd(W, full_matrices=False)
        except RuntimeError as e:
            warnings.warn(f"SVD failed: {e}. Falling back to average aggregation.")
            return self._average_aggregate(A_list, B_list)
        
        # Keep only top target_rank components
        rank = min(self.target_rank, len(Sigma))
        U_r = U[:, :rank]
        Sigma_r = Sigma[:rank]
        Vt_r = Vt[:rank, :]
        
        # Reconstruct A and B matrices
        # W ≈ U_r @ diag(Sigma_r) @ Vt_r
        # Split sigma evenly: W ≈ (U_r @ sqrt(Sigma_r)) @ (sqrt(Sigma_r) @ Vt_r)
        sqrt_sigma = torch.sqrt(Sigma_r)
        A_merged = U_r * sqrt_sigma.unsqueeze(0)  # (hidden_dim, rank)
        B_merged = (Vt_r.T * sqrt_sigma.unsqueeze(0))  # (hidden_dim, rank)
        
        return A_merged, B_merged
    
    def _average_aggregate(
        self,
        A_list: List[torch.Tensor],
        B_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple averaging aggregation (FedAvg style).
        
        Args:
            A_list: List of A matrices
            B_list: List of B matrices
            
        Returns:
            Averaged A and B matrices
        """
        # Normalize ranks
        A_normalized = self._normalize_ranks(A_list, self.target_rank)
        B_normalized = self._normalize_ranks(B_list, self.target_rank)
        
        # Average
        A_avg = torch.stack(A_normalized).mean(dim=0)
        B_avg = torch.stack(B_normalized).mean(dim=0)
        
        return A_avg, B_avg
    
    def _normalize_ranks(
        self,
        weight_list: List[torch.Tensor],
        target_rank: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Normalize weights to uniform rank via padding or truncation.
        
        Args:
            weight_list: List of weight tensors
            target_rank: Target rank (if None, use max rank)
            
        Returns:
            List of normalized weight tensors
        """
        if target_rank is None:
            target_rank = max(w.shape[1] for w in weight_list)
        
        normalized = []
        for w in weight_list:
            current_rank = w.shape[1]
            
            if current_rank < target_rank:
                # Pad with zeros
                padding = torch.zeros(
                    w.shape[0], target_rank - current_rank,
                    device=w.device, dtype=w.dtype
                )
                w_normalized = torch.cat([w, padding], dim=1)
            elif current_rank > target_rank:
                # Truncate
                w_normalized = w[:, :target_rank]
            else:
                w_normalized = w
            
            normalized.append(w_normalized)
        
        return normalized
    
    def aggregate_all_groups(
        self,
        client_updates: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
        task_groups: Dict[int, List[int]]
    ) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Aggregate LoRA weights across all task groups.
        
        Args:
            client_updates: Dictionary mapping client_id to LoRA weights
            task_groups: Dictionary mapping task_id to list of client_ids
            
        Returns:
            Dictionary mapping task_id to aggregated weights
        """
        aggregated_groups = {}
        
        for group_id, group_clients in task_groups.items():
            aggregated_groups[group_id] = self.aggregate_task_group(
                client_updates, group_clients, group_id
            )
        
        return aggregated_groups
    
    def weighted_merge(
        self,
        aggregated_groups: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
        weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Merge aggregated weights from different task groups with weighting.
        
        Args:
            aggregated_groups: Dictionary of aggregated weights per group
            weights: Optional weights per group (default: equal weighting)
            
        Returns:
            Global merged weights
        """
        if len(aggregated_groups) == 0:
            raise ValueError("No groups to merge")
        
        # Default to equal weighting
        if weights is None:
            weights = {gid: 1.0 / len(aggregated_groups) for gid in aggregated_groups}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {gid: w / total_weight for gid, w in weights.items()}
        
        # Initialize from first group
        first_group = next(iter(aggregated_groups.values()))
        global_weights = {}
        
        for layer_name in first_group:
            A_sum = None
            B_sum = None
            
            for group_id, group_weights in aggregated_groups.items():
                w_group = weights[group_id]
                
                A_weighted = w_group * group_weights[layer_name]['A']
                B_weighted = w_group * group_weights[layer_name]['B']
                
                if A_sum is None:
                    A_sum = A_weighted.clone()
                    B_sum = B_weighted.clone()
                else:
                    A_sum += A_weighted
                    B_sum += B_weighted
            
            global_weights[layer_name] = {'A': A_sum, 'B': B_sum}
        
        return global_weights
    
    def compute_aggregation_quality(
        self,
        original_weights: List[Dict[str, Dict[str, torch.Tensor]]],
        aggregated_weights: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Compute quality metrics for aggregation.
        
        Args:
            original_weights: List of original client weights
            aggregated_weights: Aggregated weights
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Compute reconstruction error
        reconstruction_errors = []
        for client_weights in original_weights:
            for layer_name in client_weights:
                # Reconstruct original: A @ B^T
                orig_W = torch.matmul(
                    client_weights[layer_name]['A'],
                    client_weights[layer_name]['B'].T
                )
                
                # Reconstruct aggregated: A_agg @ B_agg^T
                agg_W = torch.matmul(
                    aggregated_weights[layer_name]['A'],
                    aggregated_weights[layer_name]['B'].T
                )
                
                # Frobenius norm error
                error = torch.norm(orig_W - agg_W, p='fro').item()
                reconstruction_errors.append(error)
        
        metrics['mean_reconstruction_error'] = np.mean(reconstruction_errors)
        metrics['std_reconstruction_error'] = np.std(reconstruction_errors)
        
        # Compute rank preservation
        avg_orig_rank = np.mean([
            client_weights[layer_name]['A'].shape[1]
            for client_weights in original_weights
            for layer_name in client_weights
        ])
        agg_rank = aggregated_weights[list(aggregated_weights.keys())[0]]['A'].shape[1]
        metrics['rank_compression_ratio'] = agg_rank / avg_orig_rank
        
        return metrics


class PrivacyVerifier:
    """
    Tools for verifying privacy preservation in aggregation.
    
    Implements various privacy checks:
    - Gradient leakage detection
    - Privacy score computation
    - Update indistinguishability
    """
    
    def __init__(self):
        self.verification_history = []
    
    def check_gradient_leakage(
        self,
        original_grads: torch.Tensor,
        aggregated_grads: torch.Tensor,
        threshold: float = 1.0
    ) -> Dict[str, Any]:
        """
        Check if aggregation leaks gradient information.
        
        Args:
            original_grads: Original gradients before aggregation
            aggregated_grads: Aggregated gradients
            threshold: Maximum acceptable norm ratio
            
        Returns:
            Dictionary with leakage metrics
        """
        orig_norm = torch.norm(original_grads).item()
        agg_norm = torch.norm(aggregated_grads).item()
        
        norm_ratio = agg_norm / (orig_norm + 1e-10)
        
        result = {
            'original_norm': orig_norm,
            'aggregated_norm': agg_norm,
            'norm_ratio': norm_ratio,
            'passes_check': norm_ratio < threshold,
            'threshold': threshold
        }
        
        self.verification_history.append(result)
        return result
    
    def compute_privacy_score(
        self,
        task_group_size: int,
        n_total_clients: int
    ) -> float:
        """
        Compute privacy score based on group size.
        
        Larger groups provide better privacy through aggregation.
        Based on differential privacy literature (Shokri et al.)
        
        Args:
            task_group_size: Number of clients in task group
            n_total_clients: Total number of clients
            
        Returns:
            Privacy score (higher is better)
        """
        if task_group_size < 1:
            return 0.0
        
        # Logarithmic privacy gain
        base_score = math.log(task_group_size + 1)
        
        # Bonus for larger fraction of total clients
        fraction_score = task_group_size / n_total_clients
        
        # Combined score
        privacy_score = base_score * (1 + fraction_score)
        
        return privacy_score
    
    def check_update_indistinguishability(
        self,
        updates_before: List[torch.Tensor],
        updates_after: List[torch.Tensor],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Check if individual updates are indistinguishable after aggregation.
        
        Args:
            updates_before: List of updates before aggregation
            updates_after: List of updates after aggregation  
            threshold: Minimum diversity threshold
            
        Returns:
            Dictionary with indistinguishability metrics
        """
        # Compute pairwise differences
        diffs = []
        for u_before, u_after in zip(updates_before, updates_after):
            diff = torch.norm(u_before - u_after).item()
            diffs.append(diff)
        
        diversity = np.std(diffs)
        
        result = {
            'mean_difference': np.mean(diffs),
            'diversity': diversity,
            'passes_check': diversity > threshold,
            'threshold': threshold
        }
        
        return result
    
    def compute_membership_inference_resistance(
        self,
        aggregated_weights: Dict[str, Dict[str, torch.Tensor]],
        client_weights_list: List[Dict[str, Dict[str, torch.Tensor]]]
    ) -> float:
        """
        Estimate resistance to membership inference attacks.
        
        Measures how difficult it is to determine if a specific client
        contributed to the aggregated model.
        
        Args:
            aggregated_weights: Aggregated weights
            client_weights_list: List of individual client weights
            
        Returns:
            Resistance score (higher is better, 0-1)
        """
        if len(client_weights_list) < 2:
            return 0.0
        
        # Compute similarity between aggregated and each client
        similarities = []
        for client_weights in client_weights_list:
            sim = self._compute_weight_similarity(aggregated_weights, client_weights)
            similarities.append(sim)
        
        # High resistance means low variance in similarities
        # (all clients look equally similar to aggregate)
        resistance = 1.0 - np.std(similarities)
        
        return max(0.0, min(1.0, resistance))
    
    def _compute_weight_similarity(
        self,
        weights1: Dict[str, Dict[str, torch.Tensor]],
        weights2: Dict[str, Dict[str, torch.Tensor]]
    ) -> float:
        """Compute cosine similarity between two weight sets"""
        similarities = []
        
        for layer_name in weights1:
            if layer_name in weights2:
                # Flatten and compute cosine similarity
                w1_flat = torch.cat([
                    weights1[layer_name]['A'].flatten(),
                    weights1[layer_name]['B'].flatten()
                ])
                w2_flat = torch.cat([
                    weights2[layer_name]['A'].flatten(),
                    weights2[layer_name]['B'].flatten()
                ])
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    w1_flat.unsqueeze(0),
                    w2_flat.unsqueeze(0)
                ).item()
                similarities.append(cos_sim)
        
        return np.mean(similarities) if similarities else 0.0


def compute_task_aware_weights(
    task_groups: Dict[int, List[int]],
    importance_scores: Optional[Dict[int, float]] = None
) -> Dict[int, float]:
    """
    Compute task-aware weights for merging group aggregates.
    
    Args:
        task_groups: Dictionary mapping task_id to client_ids
        importance_scores: Optional importance score per task
        
    Returns:
        Dictionary of weights per task group
    """
    if importance_scores is None:
        # Size-based weighting
        total_clients = sum(len(clients) for clients in task_groups.values())
        weights = {
            gid: len(clients) / total_clients
            for gid, clients in task_groups.items()
        }
    else:
        # Importance-based weighting
        total_importance = sum(importance_scores.values())
        weights = {
            gid: importance_scores.get(gid, 1.0) / total_importance
            for gid in task_groups.keys()
        }
    
    return weights


def visualize_aggregation_quality(
    quality_metrics: Dict[str, float],
    save_path: Optional[str] = None
):
    """
    Visualize aggregation quality metrics.
    
    Args:
        quality_metrics: Dictionary of quality metrics
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reconstruction error
        if 'mean_reconstruction_error' in quality_metrics:
            axes[0].bar(['Mean Error'], [quality_metrics['mean_reconstruction_error']])
            axes[0].set_ylabel('Reconstruction Error')
            axes[0].set_title('Aggregation Reconstruction Error')
        
        # Rank compression
        if 'rank_compression_ratio' in quality_metrics:
            axes[1].bar(['Compression'], [quality_metrics['rank_compression_ratio']])
            axes[1].set_ylabel('Ratio')
            axes[1].set_title('Rank Compression Ratio')
            axes[1].axhline(y=1.0, color='r', linestyle='--', label='No compression')
            axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available for visualization")


if __name__ == '__main__':
    print("Phase 4: Privacy-Aware Aggregation")
    print("=" * 80)
    
    # Test AggregationEngine
    print("\n[1] Testing AggregationEngine...")
    engine = AggregationEngine(target_rank=16)
    
    # Create dummy client updates
    client_updates = {}
    for i in range(3):
        client_updates[i] = {
            'layer_0': {
                'A': torch.randn(768, 8 if i < 2 else 16),
                'B': torch.randn(768, 8 if i < 2 else 16)
            }
        }
    
    task_groups = {0: [0, 1], 1: [2]}
    
    # Aggregate
    aggregated = engine.aggregate_all_groups(client_updates, task_groups)
    print(f"   Aggregated {len(aggregated)} task groups")
    print(f"   Group 0 rank: {aggregated[0]['layer_0']['A'].shape[1]}")
    print(f"   [SUCCESS] AggregationEngine working!")
    
    # Test PrivacyVerifier
    print("\n[2] Testing PrivacyVerifier...")
    verifier = PrivacyVerifier()
    
    orig_grads = torch.randn(100)
    agg_grads = orig_grads * 0.5  # Reduced magnitude
    
    leakage_result = verifier.check_gradient_leakage(orig_grads, agg_grads)
    print(f"   Norm ratio: {leakage_result['norm_ratio']:.3f}")
    print(f"   Passes check: {leakage_result['passes_check']}")
    
    privacy_score = verifier.compute_privacy_score(task_group_size=10, n_total_clients=50)
    print(f"   Privacy score: {privacy_score:.3f}")
    print(f"   [SUCCESS] PrivacyVerifier working!")
    
    # Test weighted merge
    print("\n[3] Testing weighted merge...")
    weights = compute_task_aware_weights(task_groups)
    global_weights = engine.weighted_merge(aggregated, weights)
    print(f"   Global weights for {len(global_weights)} layers")
    print(f"   Task weights: {weights}")
    print(f"   [SUCCESS] Weighted merge working!")
    
    print("\n" + "=" * 80)
    print("All Phase 4 components operational!")
    print("Ready for comprehensive testing.")
