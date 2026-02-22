"""
Quick test to verify that varying `eta` (lambda) changes Laplacian updates.
Run with: python experiments/quick_eta_test.py
"""

from src.phase4_laplacian import apply_mira_laplacian
import torch
import numpy as np
import os
import sys

# Ensure project root is on sys.path so `src` imports work when running
# from the `experiments` directory or other CWDs.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def make_toy_models():
    n_clients = 4
    hidden_dim = 32
    client_models = {}
    for cid in range(n_clients):
        # different ranks to exercise alignment code
        rank = 4 + (cid % 3) * 4
        client_models[cid] = {
            'layer_0': {
                'A': torch.randn(rank, hidden_dim) * 0.01,
                'B': torch.randn(hidden_dim, rank) * 0.01
            },
            'layer_1': {
                'A': torch.randn(rank, hidden_dim) * 0.01,
                'B': torch.randn(hidden_dim, rank) * 0.01
            }
        }
    return client_models


def main():
    client_models = make_toy_models()
    # Two clusters: clients [0,1] and [2,3]
    task_clusters = {0: [0, 1], 1: [2, 3]}

    # Synthetic similarity matrix (higher within clusters)
    gradient_similarities = np.array([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.8],
        [0.1, 0.1, 0.8, 1.0]
    ])

    etas = [0.0, 0.01, 0.1, 0.5, 1.0]

    print("Testing different eta values (avg update norm and diversity):")
    for eta in etas:
        updated, meta = apply_mira_laplacian(
            client_lora_models=client_models,
            task_clusters=task_clusters,
            gradient_similarities=gradient_similarities,
            eta=eta,
            adjacency_method='similarity',
            rank_alignment_mode='truncate',
            log_diversity=True
        )

        avg_update = meta.get('update_stats', {}).get('avg_update_norm', None)
        div_before = meta.get('diversity_before', {}).get('mean_diversity')
        div_after = meta.get('diversity_after', {}).get('mean_diversity')

        print(f"  eta={eta:>4}: avg_update_norm={avg_update:.6f}, diversity_before={div_before:.6f}, diversity_after={div_after:.6f}")


if __name__ == '__main__':
    main()
