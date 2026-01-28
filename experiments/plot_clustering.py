"""
Plot ATLAS Phase 1 clustering results.

Usage:
  python experiments/plot_clustering.py --results results/atlas_integrated_quick.json

This script adapts to the project's results JSON structure and writes
`results/atlas_clustering.png` (PCA 2D scatter) and `results/atlas_clustering_simple.png`
(simple cluster-id vs client index visualization similar to the provided snippet).
"""
import json
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def make_color_map(n):
    base = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3', '#a65628', '#f781bf']
    if n <= len(base):
        return base[:n]
    # repeat palette if needed
    colors = []
    for i in range(n):
        colors.append(base[i % len(base)])
    return colors


def plot_pca_scatter(fingerprints, cluster_labels, out_path):
    # fingerprints: dict client_id -> list/array
    client_ids = list(cluster_labels.keys())
    X = np.vstack([fingerprints[str(cid)] if str(cid) in fingerprints else fingerprints.get(cid) for cid in client_ids])
    X = np.asarray(X, dtype=float)

    # PCA to 2D
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    labels = [cluster_labels[cid] for cid in client_ids]
    unique_labels = sorted(list(set(labels)))
    colors = make_color_map(len(unique_labels))

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, lab in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == lab]
        ax.scatter(Z[idx, 0], Z[idx, 1], c=colors[i], label=f'Cluster {lab}', s=120, edgecolor='k', alpha=0.8)

    for i, cid in enumerate(client_ids):
        ax.text(Z[i, 0], Z[i, 1], str(cid), fontsize=8, ha='center', va='center')

    ax.set_title('ATLAS Phase 1: Fingerprint PCA (2D) by Cluster')
    ax.legend()
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_simple_clusters(cluster_labels, out_path):
    # Create simple visualization similar to user's snippet: cluster id vs index
    client_ids = list(cluster_labels.keys())
    cluster_to_clients = {}
    for cid, cluster_id in cluster_labels.items():
        cluster_to_clients.setdefault(cluster_id, []).append(cid)

    n_clusters = len(cluster_to_clients)
    colors = make_color_map(n_clusters)

    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster_id in sorted(cluster_to_clients.keys()):
        clients = cluster_to_clients[cluster_id]
        for i, cid in enumerate(clients):
            c = colors[sorted(cluster_to_clients.keys()).index(cluster_id)]
            ax.scatter(i, cluster_id, s=300, c=c, alpha=0.6, edgecolors='black', linewidth=1.5)
            ax.text(i, cluster_id, str(cid), ha='center', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Client Index within Cluster', fontsize=12)
    ax.set_ylabel('Cluster ID', fontsize=12)
    ax.set_title('ATLAS Phase 1: Task-Based Client Clustering', fontsize=14, fontweight='bold')
    ax.set_yticks(sorted(cluster_to_clients.keys()))
    ax.grid(True, alpha=0.3, axis='y')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True, help='Path to atlas results JSON')
    parser.add_argument('--outdir', type=str, default='results', help='Output directory for plots')
    args = parser.parse_args()

    results = load_results(args.results)

    # adapt to different key names: prefer 'cluster_labels' then legacy 'clustering.assignments'
    if 'cluster_labels' in results:
        cluster_labels = {int(k): int(v) for k, v in results['cluster_labels'].items()}
    elif 'clustering' in results and isinstance(results['clustering'], dict) and 'assignments' in results['clustering']:
        cluster_labels = {int(k): int(v) for k, v in results['clustering']['assignments'].items()}
    else:
        raise ValueError('No cluster labels found in results JSON')

    # fingerprints may be saved under results['fingerprints'] as dict of lists
    fingerprints = results.get('fingerprints', {})

    outdir = Path(args.outdir)
    pca_out = outdir / 'atlas_clustering_pca.png'
    simple_out = outdir / 'atlas_clustering.png'

    # If fingerprints exist, create PCA scatter; otherwise skip
    if fingerprints:
        # Ensure keys are strings for consistent lookup
        fingerprints = {str(k): np.asarray(v) for k, v in fingerprints.items()}
        plot_pca_scatter(fingerprints, cluster_labels, pca_out)
    else:
        print('No fingerprints found in results; skipping PCA scatter')

    plot_simple_clusters(cluster_labels, simple_out)
    print(f"✅ Saved plots to {outdir}")


if __name__ == '__main__':
    main()
