"""
IEEE-Quality Publication Plots for ATLAS Framework
Generates all figures needed for the paper including ablation studies, 
model comparisons, communication efficiency, and clustering analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


# Color scheme for different methods - fixed to ensure unique colors
COLORS = {
    'ATLAS': '#1f77b4',
    'FedAvg (Clustered)': '#2ca02c',
    'Standard FL': '#9467bd',
    'distilbert': '#1f77b4',
    'gpt2': '#8c564b',
    'qwen': '#e377c2',
}

LINE_STYLES = {
    'ATLAS': '-',
    'FedAvg (Clustered)': '-.',
    'Standard FL': ':'
}

MARKERS = {
    'ATLAS': 'o',
    'FedAvg (Clustered)': 'D',
    'Standard FL': 'v'
}


class PublicationPlotter:
    """Generate IEEE-quality publication plots for ATLAS experiments."""
    
    def __init__(self, results_dir: str = 'results', output_dir: str = 'figures'):
        """Initialize plotter with results and output directories."""
        # Resolve results directory: try provided path, then common alternatives
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            alt1 = Path(__file__).resolve().parent.parent / 'results'
            alt2 = Path.cwd() / 'results'
            alt3 = Path(__file__).resolve().parent / 'results'
            resolved = None
            for p in (alt1, alt2, alt3):
                if p.exists():
                    resolved = p
                    break
            if resolved is not None:
                print(f"  Info: using results directory {resolved}")
                self.results_dir = resolved
            else:
                print(f"  Warning: results directory '{results_dir}' not found; looked in: {alt1}, {alt2}, {alt3}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        sns.set_style('whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 10

    def _canonical_method(self, label: str) -> str:
        """Normalize various label variants to canonical method keys used in COLORS.

        This handles variants like 'ATLAS (Full)', 'ATLAS (No Laplacian)' etc.
        """
        if label is None:
            return label
        low = label.lower()
        if 'no lap' in low or 'no lapl' in low:
            return 'ATLAS'
        if 'atlas' in low and 'no lap' not in low and 'full' in low:
            return 'ATLAS'
        if low.strip() == 'atlas':
            return 'ATLAS'
        if 'standard' in low or 'standard fl' in low or low.strip() == 'fl':
            return 'Standard FL'
        if 'fedavg' in low or 'cluster' in low:
            return 'FedAvg (Clustered)'
        if 'distilbert' in low:
            return 'distilbert'
        if 'gpt' in low:
            return 'gpt2'
        if 'qwen' in low:
            return 'qwen'
        return label

    def _color(self, label: str) -> str:
        return COLORS.get(self._canonical_method(label), '#000000')

    def _linestyle(self, label: str) -> str:
        return LINE_STYLES.get(self._canonical_method(label), '-')

    def _marker(self, label: str) -> str:
        return MARKERS.get(self._canonical_method(label), 'o')
    
    def load_result(self, filename: str):
        """Load a result JSON file."""
        tried = []
        candidates = [
            self.results_dir / filename,
            Path(__file__).resolve().parent.parent / 'results' / filename,
            Path.cwd() / 'results' / filename,
            Path.cwd() / filename,
        ]
        for fp in candidates:
            tried.append(str(fp))
            if fp.exists():
                try:
                    with open(fp, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"  Error loading {fp}: {e}")
                    return None

        print(f"  Warning: {filename} not found. Tried: {', '.join(tried)}")
        return None
    
    def extract_metrics_per_round(self, result):
        """Extract per-round metrics (rounds, accuracies, f1s, times)."""
        if result is None or 'round_metrics' not in result:
            return None, None, None, None
        
        rounds = []
        accuracies = []
        f1_scores = []
        times = []
        
        for round_data in result['round_metrics']:
            rounds.append(round_data.get('round', len(rounds) + 1))
            accuracies.append(round_data.get('avg_accuracy', 0.0))
            
            # Extract F1 scores
            test_f1 = round_data.get('test_f1', {})
            if test_f1:
                avg_f1 = np.mean(list(test_f1.values()))
                f1_scores.append(avg_f1)
            else:
                f1_scores.append(0.0)
            
            times.append(round_data.get('time_seconds', 0.0))
        
        return (np.array(rounds), np.array(accuracies), 
                np.array(f1_scores), np.array(times))
    
    def extract_communication_stats(self, result):
        """Extract communication statistics from result."""
        if result is None or 'round_metrics' not in result:
            return {'total_comm_mb': 0.0}
        
        total_comm = 0
        for round_data in result['round_metrics']:
            upload = sum(round_data.get('comm_upload_bytes', {}).values())
            download = sum(round_data.get('comm_download_bytes', {}).values())
            total_comm += (upload + download)
        
        return {'total_comm_mb': total_comm / (1024 * 1024)}
    
    def get_method_key(self, label):
        """Get the correct color/style key for a method label."""
        # Use the label directly for method configurations 
        return label
    
    def plot_ablation_study(self):
        """Generate Figure 1: Ablation Study."""
        print("Generating Figure 1: Ablation Study...")
        
        configs = {
            'ATLAS': 'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
            'FedAvg (Clustered)': 'atlas_distilbert-base-uncased_fedavg_cluster_seed42_r10.json',
            'Standard FL': 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
        }
        
        # Accuracy convergence
        fig, ax = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
        for label, filename in configs.items():
            result = self.load_result(filename)
            rounds, accs, f1s, times = self.extract_metrics_per_round(result)
            if rounds is None:
                continue
            ax.plot(rounds, accs * 100,
                    label=label,
                    color=self._color(label),
                    linestyle=self._linestyle(label),
                    marker=self._marker(label),
                    markevery=1,
                    linewidth=1.5,
                    markersize=5)
        
        ax.set_xlabel('Communication Round', fontweight='bold')
        ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
        ax.set_title('Ablation: Accuracy Convergence', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlim(left=1)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
        
        figfile = self.output_dir / 'fig1_ablation_accuracy.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
        
        # F1 score convergence
        fig, ax = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
        for label, filename in configs.items():
            result = self.load_result(filename)
            rounds, accs, f1s, times = self.extract_metrics_per_round(result)
            if rounds is None:
                continue
            ax.plot(rounds, f1s * 100,
                    label=label,
                    color=self._color(label),
                    linestyle=self._linestyle(label),
                    marker=self._marker(label),
                    markevery=1,
                    linewidth=1.5,
                    markersize=5)
        
        ax.set_xlabel('Communication Round', fontweight='bold')
        ax.set_ylabel('Average F1 Score (%)', fontweight='bold')
        ax.set_title('Ablation: F1 Score Convergence', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlim(left=1)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
        
        figfile = self.output_dir / 'fig1_ablation_f1.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
    
    def plot_model_comparison(self):
        """Generate Figure 2: Cross-Model Performance."""
        print("Generating Figure 2: Model Comparison...")
        
        models_data = {
            'DistilBERT': ['atlas_distilbert-base-uncased_atlas_seed42_r10.json',
                            'atlas_distilbert-base-uncased_atlas_seed123_r10.json',
                            'atlas_distilbert-base-uncased_atlas_seed456_r10.json'],
            'GPT-2': [
                'atlas_gpt2_atlas_seed42_r10.json',
                'atlas_gpt2_atlas_seed123_r10.json',
                'atlas_gpt2_atlas_seed456_r10.json',
            ],
            'Qwen-0.5B': [
                'atlas_Qwen_Qwen2.5-0.5B_atlas_seed42_r10.json',
                'atlas_Qwen_Qwen2.5-0.5B_atlas_seed123_r10.json',
                'atlas_Qwen_Qwen2.5-0.5B_atlas_seed456_r10.json',
            ],
        }
        
        # Accuracy across models
        fig, ax = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
        for model_name, filenames in models_data.items():
            all_accs = []
            rounds_arr = None
            
            for filename in filenames:
                result = self.load_result(filename)
                rounds, accs, f1s, times = self.extract_metrics_per_round(result)
                if rounds is not None:
                    all_accs.append(accs)
                    rounds_arr = rounds
            
            if all_accs:
                mean_accs = np.mean(all_accs, axis=0)
                std_accs = np.std(all_accs, axis=0) if len(all_accs) > 1 else np.zeros_like(mean_accs)
                
                key = model_name.lower().replace('-', '').replace('.', '')
                if 'distilbert' in key:
                    key = 'distilbert'
                elif 'gpt' in key:
                    key = 'gpt2'
                elif 'qwen' in key:
                    key = 'qwen'
                
                ax.plot(rounds_arr, mean_accs * 100,
                    label=model_name,
                    color=self._color(key),
                    linestyle=self._linestyle(key),
                    marker=self._marker(key),
                    markevery=1,
                    linewidth=1.5,
                    markersize=5)
                
                if len(all_accs) > 1:
                    ax.fill_between(rounds_arr, 
                                    (mean_accs - std_accs) * 100,
                                    (mean_accs + std_accs) * 100,
                                    color=self._color(key),
                                    alpha=0.25)
        
        ax.set_xlabel('Communication Round', fontweight='bold')
        ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
        ax.set_title('Model Architecture Comparison: Accuracy', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', framealpha=0.9, fontsize=8)
        ax.set_xlim(left=1)
        
        figfile = self.output_dir / 'fig2_model_accuracy.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
        
        # Baseline comparison (plot mean +/- std across available seeds)
        fig, ax = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
        baseline_comparisons = {
            'DistilBERT': {
                'ATLAS': [
                    'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
                    'atlas_distilbert-base-uncased_atlas_seed123_r10.json',
                    'atlas_distilbert-base-uncased_atlas_seed456_r10.json',
                ],
                'Standard FL': [
                    'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
                    'atlas_distilbert-base-uncased_standard_fl_seed123_r10.json',
                    'atlas_distilbert-base-uncased_standard_fl_seed456_r10.json',
                ],
            },
            'GPT-2': {
                'ATLAS': [
                    'atlas_gpt2_atlas_seed42_r10.json',
                    'atlas_gpt2_atlas_seed123_r10.json',
                    'atlas_gpt2_atlas_seed456_r10.json',
                ],
                'Standard FL': [
                    'atlas_gpt2_standard_fl_seed42_r10.json',
                    'atlas_gpt2_standard_fl_seed123_r10.json',
                    'atlas_gpt2_standard_fl_seed456_r10.json',
                ],
            },
            'Qwen-0.5B': {
                'ATLAS': [
                    'atlas_Qwen_Qwen2.5-0.5B_atlas_seed42_r10.json',
                    'atlas_Qwen_Qwen2.5-0.5B_atlas_seed123_r10.json',
                    'atlas_Qwen_Qwen2.5-0.5B_atlas_seed456_r10.json',
                ],
                'Standard FL': [
                    'atlas_Qwen_Qwen2.5-0.5B_standard_fl_seed42_r10.json',
                    'atlas_Qwen_Qwen2.5-0.5B_standard_fl_seed123_r10.json',
                    'atlas_Qwen_Qwen2.5-0.5B_standard_fl_seed456_r10.json',
                ],
            },
        }

        for model_name, configs in baseline_comparisons.items():
            for method, filenames in configs.items():
                all_accs = []
                rounds_arr = None
                for filename in filenames:
                    result = self.load_result(filename)
                    rounds, accs, f1s, times = self.extract_metrics_per_round(result)
                    if rounds is not None:
                        all_accs.append(accs)
                        rounds_arr = rounds

                if not all_accs:
                    continue

                mean_accs = np.mean(all_accs, axis=0)
                std_accs = np.std(all_accs, axis=0) if len(all_accs) > 1 else np.zeros_like(mean_accs)

                linestyle = '-' if method == 'ATLAS' else '--'
                label = f"{model_name} ({method})"
                # Use model-specific color so each model's traces are visually
                # distinct; use linestyle to indicate the method (ATLAS vs Standard FL)
                ax.plot(rounds_arr, mean_accs * 100,
                    label=label,
                    color=self._color(model_name),
                    linestyle=linestyle,
                    linewidth=1.5,
                    alpha=0.9)

                if len(all_accs) > 1:
                    ax.fill_between(rounds_arr,
                                    (mean_accs - std_accs) * 100,
                                    (mean_accs + std_accs) * 100,
                                    color=self._color(model_name),
                                    alpha=0.25)
        
        ax.set_xlabel('Communication Round', fontweight='bold')
        ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
        ax.set_title('ATLAS vs. Standard FL', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', framealpha=0.9, fontsize=8)
        ax.set_xlim(left=1)
        
        figfile = self.output_dir / 'fig2_model_baseline.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)

    def _extract_accuracy_vs_cumulative_comm(self, result):
        """Return (rounds, cumulative_comm_mb, avg_accuracy_pct)."""
        if result is None or 'round_metrics' not in result:
            return None, None, None

        rounds = []
        cumulative_comm_mb = []
        acc_pct = []
        total_bytes = 0

        for i, round_data in enumerate(result['round_metrics']):
            rounds.append(round_data.get('round', i + 1))
            upload = sum(round_data.get('comm_upload_bytes', {}).values())
            download = sum(round_data.get('comm_download_bytes', {}).values())
            total_bytes += (upload + download)
            cumulative_comm_mb.append(total_bytes / (1024 * 1024))
            acc_pct.append(round_data.get('avg_accuracy', 0.0) * 100.0)

        return np.asarray(rounds), np.asarray(cumulative_comm_mb), np.asarray(acc_pct)

    def plot_accuracy_vs_communication_tradeoff(self):
        """Plot accuracy as a function of cumulative communication.

        This makes it easy to see where ATLAS is better: ATLAS exceeds a baseline
        when its curve is above the baseline curve for the same communication budget.
        """
        print("Generating Figure 3: Accuracy vs Communication... ")

        configs = {
            'ATLAS': 'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
            'FedAvg (Clustered)': 'atlas_distilbert-base-uncased_fedavg_cluster_seed42_r10.json',
            'Standard FL': 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
        }

        fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)

        series = {}
        for label, filename in configs.items():
            result = self.load_result(filename)
            rounds, cum_mb, acc_pct = self._extract_accuracy_vs_cumulative_comm(result)
            if rounds is None:
                continue
            series[label] = (rounds, cum_mb, acc_pct)

        if not series:
            print("  Warning: no results available for tradeoff plot")
            plt.close(fig)
            return

        for label, (_rounds, cum_mb, acc_pct) in series.items():
            ax.plot(
                cum_mb,
                acc_pct,
                label=label,
                color=self._color(label),
                linestyle=self._linestyle(label),
                marker=self._marker(label),
                markevery=1,
                linewidth=1.5,
                markersize=5,
            )

        # Optional: Highlight where ATLAS exceeds Standard FL (disabled for cleaner plot)
        # if 'ATLAS' in series and 'Standard FL' in series:
        #     pass

        ax.set_xlabel('Cumulative Communication (MB)', fontweight='bold')
        ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
        ax.set_title('Accuracy vs. Communication Budget', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', framealpha=0.9, fontsize=8)

        figfile = self.output_dir / 'fig3_accuracy_vs_communication.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)


    def plot_fingerprint_map(self):
        """Generate a 2D fingerprint embedding map colored by cluster.

        Uses PCA on stored gradient fingerprints (from Phase 1). The intent is
        purely visual: clients that are close in fingerprint space should appear
        close on the map, and cluster-colored points should form groups.
        """
        print("Generating Figure 9: Fingerprint Map...")

        result = self.load_result('atlas_gpt2_atlas_seed123_r10.json')
        if result is None:
            print("  Warning: Could not load result for fingerprint map")
            return

        fingerprints = result.get('fingerprints', {})
        cluster_labels = result.get('cluster_labels', {})
        if not fingerprints or not cluster_labels:
            print("  Warning: Missing fingerprints and/or cluster_labels in result")
            return

        # Normalize dict keys
        fingerprints = {str(k): np.asarray(v, dtype=float) for k, v in fingerprints.items()}
        cluster_labels = {str(k): int(v) for k, v in cluster_labels.items()}
        clients = sorted([int(k) for k in cluster_labels.keys()])
        clusters = [cluster_labels.get(str(c), 0) for c in clients]
        unique_clusters = sorted(list(set(clusters)))

        # Build (n_clients x d) fingerprint matrix
        X = []
        kept_clients = []
        for c in clients:
            fp = fingerprints.get(str(c), None)
            if fp is None:
                continue
            X.append(fp.reshape(-1))
            kept_clients.append(c)
        if not X:
            print("  Warning: No fingerprints found for clustered clients")
            return
        X = np.vstack(X)

        # Use t-SNE for better cluster separation (fallback to PCA)
        try:
            from sklearn.manifold import TSNE
            Z = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1), n_iter=1000).fit_transform(X)
        except Exception:
            try:
                from sklearn.decomposition import PCA
                Z = PCA(n_components=2, random_state=42).fit_transform(X)
            except Exception:
                # numpy SVD fallback
                Xc = X - X.mean(axis=0, keepdims=True)
                _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
                Z = Xc @ vt.T[:, :2]

        # Colors
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, max(1, len(unique_clusters))))
        cluster_to_color = {c: cluster_colors[i % len(cluster_colors)] for i, c in enumerate(unique_clusters)}
        colors = [cluster_to_color.get(cluster_labels.get(str(c), 0), (0.2, 0.2, 0.2, 1.0)) for c in kept_clients]

        fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
        ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=90, edgecolor='black', linewidth=0.8, alpha=0.9)

        # Annotate client ids (small, but helps mapping)
        for i, c in enumerate(kept_clients):
            ax.text(Z[i, 0], Z[i, 1], str(c), fontsize=8, ha='center', va='center')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cluster_to_color[c], edgecolor='black', label=f'Cluster {c}')
            for c in unique_clusters
        ]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='best', framealpha=0.9, fontsize=8)

        ax.set_xlabel('Dimension 1', fontweight='bold')
        ax.set_ylabel('Dimension 2', fontweight='bold')
        ax.set_title('Client Fingerprint Clustering', fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

        figfile = self.output_dir / 'fig9_fingerprint_map.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)

    def plot_comm_needed_for_targets(self, targets=(79.0, 80.0)):
        """Compute MB needed to reach target accuracies and save a small comparison figure.

        Targets are in percentage points (e.g., 79.0 for 79%). We use the DistilBERT
        results for the communication comparison (same files used elsewhere).
        """
        print("Generating: MB needed to reach target accuracies...")

        configs = {
            'ATLAS': 'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
            'Standard FL': 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
        }

        series = {}
        for label, filename in configs.items():
            result = self.load_result(filename)
            rounds, cum_mb, acc_pct = self._extract_accuracy_vs_cumulative_comm(result)
            if rounds is None:
                continue
            series[label] = (cum_mb, acc_pct)

        if not series:
            print("  Warning: no series available for comm-needed plot")
            return

        rows = []
        for target in targets:
            row = {'Target (%)': f"{target:.1f}%"}
            for label, (cum_mb, acc_pct) in series.items():
                # Find first cumulative MB where accuracy >= target
                idx = np.where(acc_pct >= target)[0]
                if idx.size == 0:
                    mb = np.nan
                else:
                    i = idx[0]
                    # linear interpolate for smoother number if possible
                    if i == 0:
                        mb = float(cum_mb[0])
                    else:
                        # interpolate between (i-1) and i
                        x0, x1 = cum_mb[i-1], cum_mb[i]
                        y0, y1 = acc_pct[i-1], acc_pct[i]
                        if x1 == x0:
                            mb = float(x1)
                        else:
                            t = (target - y0) / (y1 - y0) if (y1 - y0) != 0 else 0.0
                            mb = float(x0 + t * (x1 - x0))
                row[label] = mb
            rows.append(row)

        # Save a compact table-like figure
        fig, ax = plt.subplots(figsize=(5.5, 2.2), constrained_layout=True)
        ax.axis('off')
        table_data = []
        cols = ['Target (%)'] + list(series.keys())
        for r in rows:
            table_data.append([r.get(c, '') if not (isinstance(r.get(c,''), float) and np.isnan(r.get(c,''))) else 'N/A' for c in cols])

        # format MB values nicely
        for i in range(len(table_data)):
            for j in range(1, len(cols)):
                v = rows[i].get(cols[j])
                table_data[i][j] = f"{v:.1f}" if (isinstance(v, float) and not np.isnan(v)) else 'N/A'

        table = ax.table(cellText=table_data, colLabels=cols, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.3)
        ax.set_title('MB Needed to Reach Target Accuracy', fontweight='bold')

        figfile = self.output_dir / 'fig3_comm_to_reach_targets.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)

    def plot_pareto_frontier(self):
        """Plot Pareto frontier (accuracy vs cumulative MB) across rounds for methods.

        This shows per-round points and draws the non-dominated frontier (max acc
        for a given communication budget), which highlights low-budget advantages.
        """
        print("Generating Pareto frontier plot...")

        configs = {
            'ATLAS': 'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
            'FedAvg (Clustered)': 'atlas_distilbert-base-uncased_fedavg_cluster_seed42_r10.json',
            'Standard FL': 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
        }

        data = {}
        for label, filename in configs.items():
            result = self.load_result(filename)
            rounds, cum_mb, acc_pct = self._extract_accuracy_vs_cumulative_comm(result)
            if rounds is None:
                continue
            data[label] = (cum_mb, acc_pct)

        if not data:
            print('  Warning: no data for Pareto plot')
            return

        fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

        # plot scatter points
        for label, (cum_mb, acc_pct) in data.items():
            ax.scatter(cum_mb, acc_pct, label=label, color=self._color(label), alpha=0.7, s=40, edgecolor='k')

        # compute combined Pareto frontier from all points
        all_points = []
        for label, (cum_mb, acc_pct) in data.items():
            for x, y in zip(cum_mb, acc_pct):
                all_points.append((float(x), float(y)))
        all_points = sorted(all_points, key=lambda p: (p[0], -p[1]))

        # Non-dominated: for increasing comm, keep points with strictly increasing max acc
        pareto_x = []
        pareto_y = []
        max_acc = -np.inf
        for x, y in all_points:
            if y > max_acc:
                pareto_x.append(x)
                pareto_y.append(y)
                max_acc = y

        if pareto_x:
            ax.plot(pareto_x, pareto_y, color='black', linestyle='-', linewidth=1.5, label='Pareto frontier')
            ax.scatter(pareto_x, pareto_y, color='black', s=30)

        ax.set_xlabel('Cumulative Communication (MB)', fontweight='bold')
        ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
        ax.set_title('Pareto Frontier: Accuracy vs Communication', fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc='best', framealpha=0.9, fontsize=8)

        figfile = self.output_dir / 'fig_pareto_accuracy_vs_comm.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)

    
    def plot_parameter_efficiency(self):
        """
        Figure: Parameter Efficiency - Heterogeneous vs Homogeneous vs Full Fine-tuning
        
        Creates a multi-panel figure showing:
        1. Total trainable parameters comparison
        2. Per-device parameter allocation
        3. Layer-wise rank heatmap
        4. Memory efficiency metrics
        """
        print("Generating Figure: Parameter Efficiency Comparison...")
        
        # Load one result from each method for DistilBERT
        atlas_result = self.load_result('atlas_distilbert-base-uncased_atlas_seed42_r10.json')
        std_fl_result = self.load_result('atlas_distilbert-base-uncased_standard_fl_seed42_r10.json')
        
        if not atlas_result or not std_fl_result:
            print("  Warning: Could not load results for parameter efficiency plot")
            return
        
        # Get phase2 rank allocation data
        atlas_ranks = atlas_result.get('phase2_rank_allocation', [])
        std_fl_ranks = std_fl_result.get('phase2_rank_allocation', [])
        
        if not atlas_ranks or not std_fl_ranks:
            print("  Warning: No phase2_rank_allocation data found")
            return
        
        # Extract data
        atlas_params = [c['lora_params'] for c in atlas_ranks]
        std_fl_params = [c['lora_params'] for c in std_fl_ranks]
        
        atlas_total = sum(atlas_params) / len(atlas_params)  # Average per client
        std_fl_total = sum(std_fl_params) / len(std_fl_params)
        
        # DistilBERT base has ~66M parameters
        full_finetune_params = 66_000_000
        
        # Group by device type
        device_types = ['cpu_2gb', 'tablet_4gb', 'laptop_8gb', 'gpu_16gb']
        atlas_by_device = {dt: [] for dt in device_types}
        std_fl_by_device = {dt: [] for dt in device_types}
        
        for c in atlas_ranks:
            atlas_by_device[c['device']].append(c['lora_params'])
        for c in std_fl_ranks:
            std_fl_by_device[c['device']].append(c['lora_params'])
        
        # Average params per device type
        atlas_device_avg = [np.mean(atlas_by_device[dt]) if atlas_by_device[dt] else 0 for dt in device_types]
        std_fl_device_avg = [np.mean(std_fl_by_device[dt]) if std_fl_by_device[dt] else 0 for dt in device_types]
        
        # Extract rank matrices for heatmap
        atlas_rank_matrix = np.array([c['ranks'] for c in atlas_ranks])
        std_fl_rank_matrix = np.array([c['ranks'] for c in std_fl_ranks])
        
        # Create 2x2 figure
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # ============ Panel A: Total Parameters Comparison ============
        ax1 = fig.add_subplot(gs[0, 0])
        
        methods = ['ATLAS\n(Heterogeneous)', 'Standard FL\n(Homogeneous)', 'Full\nFine-tuning']
        params = [atlas_total, std_fl_total, full_finetune_params]
        colors_panel = ['#1f77b4', '#9467bd', '#d62728']
        
        bars = ax1.bar(methods, params, color=colors_panel, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, param in zip(bars, params):
            height = bar.get_height()
            if param > 1_000_000:
                label = f'{param/1_000_000:.1f}M'
            else:
                label = f'{param/1000:.0f}K'
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_ylabel('Trainable Parameters', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Total Trainable Parameters', fontsize=13, fontweight='bold', pad=10)
        ax1.set_yscale('log')
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', labelsize=10)
        
        # Add reduction annotations
        reduction_atlas = (1 - atlas_total/full_finetune_params) * 100
        reduction_std = (1 - std_fl_total/full_finetune_params) * 100
        ax1.text(0.5, 0.95, f'ATLAS: {reduction_atlas:.1f}% reduction\nStd FL: {reduction_std:.1f}% reduction',
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ============ Panel B: Per-Device Allocation ============
        ax2 = fig.add_subplot(gs[0, 1])
        
        x = np.arange(len(device_types))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, np.array(atlas_device_avg)/1000, width, 
                       label='ATLAS (Heterogeneous)', color='#1f77b4', alpha=0.8, edgecolor='black')
        bars2 = ax2.bar(x + width/2, np.array(std_fl_device_avg)/1000, width,
                       label='Standard FL (Homogeneous)', color='#9467bd', alpha=0.8, edgecolor='black')
        
        ax2.set_xlabel('Device Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('LoRA Parameters (×1000)', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Per-Device Parameter Allocation', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['CPU\n2GB', 'Tablet\n4GB', 'Laptop\n8GB', 'GPU\n16GB'], fontsize=10)
        ax2.legend(fontsize=9, loc='upper left')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.0f}K', ha='center', va='bottom', fontsize=8)
        
        # ============ Panel C: ATLAS Rank Heatmap ============
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Sort by device type for better visualization
        device_order = []
        for dt in device_types:
            device_order.extend([i for i, c in enumerate(atlas_ranks) if c['device'] == dt])
        
        atlas_sorted = atlas_rank_matrix[device_order]
        
        im1 = ax3.imshow(atlas_sorted, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax3.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Client ID', fontsize=12, fontweight='bold')
        ax3.set_title('(c) ATLAS: Heterogeneous Ranks', fontsize=13, fontweight='bold', pad=10)
        ax3.set_xticks(range(len(atlas_ranks[0]['ranks'])))
        ax3.set_xticklabels(range(len(atlas_ranks[0]['ranks'])))
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax3)
        cbar1.set_label('LoRA Rank', rotation=270, labelpad=15, fontsize=10)
        
        # Add device type annotations on y-axis
        y_pos = 0
        for dt in device_types:
            n_clients = len([c for c in atlas_ranks if c['device'] == dt])
            if n_clients > 0:
                ax3.axhline(y_pos, color='white', linewidth=2, linestyle='--', alpha=0.5)
                ax3.text(-0.5, y_pos + n_clients/2, dt.replace('_', '\n'), 
                        fontsize=8, ha='right', va='center')
                y_pos += n_clients
        
        # ============ Panel D: Standard FL Rank Heatmap ============
        ax4 = fig.add_subplot(gs[1, 1])
        
        im2 = ax4.imshow(std_fl_rank_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax4.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Client ID', fontsize=12, fontweight='bold')
        ax4.set_title('(d) Standard FL: Homogeneous Ranks', fontsize=13, fontweight='bold', pad=10)
        ax4.set_xticks(range(len(std_fl_ranks[0]['ranks'])))
        ax4.set_xticklabels(range(len(std_fl_ranks[0]['ranks'])))
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax4)
        cbar2.set_label('LoRA Rank', rotation=270, labelpad=15, fontsize=10)
        
        # Overall title
        fig.suptitle('Parameter Efficiency: Heterogeneous vs Homogeneous LoRA Allocation',
                    fontsize=15, fontweight='bold', y=0.995)
        
        # Save
        output_path = self.output_dir / 'fig_parameter_efficiency.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
        print(f"    ATLAS avg: {atlas_total/1000:.1f}K params")
        print(f"    Standard FL avg: {std_fl_total/1000:.1f}K params")
        print(f"    Full fine-tuning: {full_finetune_params/1_000_000:.1f}M params")

    def plot_device_parameter_allocation(self):
        """
        Standalone Figure: Per-Device Parameter Allocation
        Shows heterogeneous vs homogeneous LoRA parameter allocation across device types
        """
        print("Generating Figure: Per-Device Parameter Allocation...")
        
        # Load one result from each method for DistilBERT
        atlas_result = self.load_result('atlas_distilbert-base-uncased_atlas_seed42_r10.json')
        std_fl_result = self.load_result('atlas_distilbert-base-uncased_standard_fl_seed42_r10.json')
        
        if not atlas_result or not std_fl_result:
            print("  Warning: Could not load results for device allocation plot")
            return
        
        # Get phase2 rank allocation data
        atlas_ranks = atlas_result.get('phase2_rank_allocation', [])
        std_fl_ranks = std_fl_result.get('phase2_rank_allocation', [])
        
        if not atlas_ranks or not std_fl_ranks:
            print("  Warning: No phase2_rank_allocation data found")
            return
        
        # Group by device type
        device_types = ['cpu_2gb', 'tablet_4gb', 'laptop_8gb', 'gpu_16gb']
        device_labels = ['CPU\n2GB', 'Tablet\n4GB', 'Laptop\n8GB', 'GPU\n16GB']
        
        atlas_by_device = {dt: [] for dt in device_types}
        std_fl_by_device = {dt: [] for dt in device_types}
        
        for c in atlas_ranks:
            atlas_by_device[c['device']].append(c['lora_params'])
        for c in std_fl_ranks:
            std_fl_by_device[c['device']].append(c['lora_params'])
        
        # Average params per device type
        atlas_device_avg = [np.mean(atlas_by_device[dt])/1000 if atlas_by_device[dt] else 0 for dt in device_types]
        std_fl_device_avg = [np.mean(std_fl_by_device[dt])/1000 if std_fl_by_device[dt] else 0 for dt in device_types]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(device_types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, atlas_device_avg, width, 
                       label='ATLAS (Heterogeneous)', color='#1f77b4', alpha=0.85, 
                       edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, std_fl_device_avg, width,
                       label='Standard FL (Homogeneous)', color='#9467bd', alpha=0.85, 
                       edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Device Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('LoRA Parameters (×1000)', fontsize=14, fontweight='bold')
        ax.set_title('Per-Device Parameter Allocation: Heterogeneous vs Homogeneous', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(device_labels, fontsize=12)
        ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.0f}K', ha='center', va='bottom', 
                            fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'fig_device_allocation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")

    def plot_atlas_vs_qwen_params(self):
        """
        Figure: ATLAS vs Qwen2.5-0.5B Parameter Comparison
        Simple bar chart comparing ATLAS LoRA parameters vs full Qwen2.5-0.5B fine-tuning
        """
        print("Generating Figure: ATLAS vs Qwen2.5-0.5B Parameter Comparison...")
        
        # Load ATLAS result for Qwen
        atlas_result = self.load_result('atlas_Qwen_Qwen2.5-0.5B_atlas_seed42_r10.json')
        
        if not atlas_result:
            print("  Warning: Could not load Qwen ATLAS results")
            return
        
        # Get phase2 rank allocation data
        atlas_ranks = atlas_result.get('phase2_rank_allocation', [])
        
        if not atlas_ranks:
            print("  Warning: No phase2_rank_allocation data found")
            return
        
        # Extract ATLAS parameters
        atlas_params = [c['lora_params'] for c in atlas_ranks]
        atlas_total = sum(atlas_params) / len(atlas_params)  # Average per client
        
        # Qwen2.5-0.5B has 500M parameters
        qwen_full_params = 500_000_000
        
        # Create figure - similar style to the provided image
        fig, ax = plt.subplots(figsize=(8, 6))
        
        methods = ['ATLAS\n(Heterogeneous)', 'Full\nFine-tuning']
        params = [atlas_total, qwen_full_params]
        colors_panel = ['#1f77b4', '#d62728']
        
        bars = ax.bar(methods, params, color=colors_panel, alpha=0.8, 
                     edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, param in zip(bars, params):
            height = bar.get_height()
            if param > 1_000_000:
                label = f'{param/1_000_000:.1f}M'
            else:
                label = f'{param/1000:.0f}K'
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                   label, ha='center', va='center', fontsize=14, 
                   fontweight='bold', color='white')
        
        ax.set_ylabel('Trainable Parameters', fontsize=14, fontweight='bold')
        ax.set_title('Total Trainable Parameters\nATLAS vs Qwen2.5-0.5B Full Fine-tuning', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', labelsize=12)
        ax.set_ylim(bottom=10000)  # Start from 10K to show the difference better
        
        # Add reduction annotation with background box
        reduction_atlas = (1 - atlas_total/qwen_full_params) * 100
        textstr = f'ATLAS: {reduction_atlas:.2f}% reduction'
        props = dict(boxstyle='round', facecolor='#ffebcd', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.text(0.5, 0.95, textstr,
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', horizontalalignment='center',
                bbox=props)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'fig_atlas_vs_qwen_params.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
        print(f"    ATLAS avg: {atlas_total/1000:.1f}K params")
        print(f"    Qwen2.5-0.5B full: {qwen_full_params/1_000_000:.1f}M params")
        print(f"    Reduction: {reduction_atlas:.2f}%")

    def generate_all_plots(self):
        """Generate all publication plots."""
        print("\n" + "="*60)
        print("Generating IEEE-Quality Publication Plots")
        print("="*60 + "\n")

        # Clean old PNGs so only the remaining plots are kept.
        for fp in self.output_dir.glob('*.png'):
            try:
                fp.unlink()
            except Exception:
                pass
        
        self.plot_ablation_study()
        self.plot_model_comparison()
        self.plot_accuracy_vs_communication_tradeoff()
        # Communication-efficiency focused figures
        self.plot_comm_needed_for_targets(targets=(79.0, 80.0))
        self.plot_pareto_frontier()
        self.plot_fingerprint_map()
        # Parameter efficiency
        self.plot_parameter_efficiency()
        self.plot_device_parameter_allocation()
        self.plot_atlas_vs_qwen_params()
        
        print("\n" + "="*60)
        print("All plots generated successfully!")
        print(f"Output directory: {self.output_dir.absolute()}")
        print("="*60 + "\n")


def main():
    """Main function to generate all plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate IEEE-quality publication plots for ATLAS')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing results JSON files')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Directory to save output figures')
    
    args = parser.parse_args()
    
    plotter = PublicationPlotter(results_dir=args.results_dir, output_dir=args.output_dir)
    plotter.generate_all_plots()


if __name__ == '__main__':
    main()
