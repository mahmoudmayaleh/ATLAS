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
    'ATLAS (No Lap)': '#ff7f0e',
    'FedAvg (Clustered)': '#2ca02c',
    'Standard FL': '#9467bd',
    'distilbert': '#1f77b4',
    'gpt2': '#8c564b',
    'qwen': '#e377c2',
}

LINE_STYLES = {
    'ATLAS': '-',
    'ATLAS (No Lap)': '--',
    'FedAvg (Clustered)': '-.',
    'Standard FL': ':'
}

MARKERS = {
    'ATLAS': 'o',
    'ATLAS (No Lap)': 's',
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
            return 'ATLAS (No Lap)'
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
            'ATLAS (No Lap)': 'atlas_distilbert-base-uncased_atlas_no_laplacian_seed42_r10.json',
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
            'DistilBERT': ['atlas_distilbert-base-uncased_atlas_seed42_r10.json'],
            'GPT-2': [
                'atlas_gpt2_atlas_seed42_r10.json',
                'atlas_gpt2_atlas_seed123_r10.json',
                'atlas_gpt2_atlas_seed456_r10.json',
            ],
            'Qwen-0.5B': [
                'atlas_Qwen_Qwen2.5-0.5B_atlas_seed42_r10.json',
                'atlas_Qwen_Qwen2.5-0.5B_atlas_seed123_r10.json',
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
                                    alpha=0.15)
        
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
        
        # Baseline comparison
        fig, ax = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
        baseline_comparisons = {
            'DistilBERT': {
                'ATLAS': 'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
                'Standard FL': 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
            },
            'GPT-2': {
                'ATLAS': 'atlas_gpt2_atlas_seed42_r10.json',
                'Standard FL': 'atlas_gpt2_standard_fl_seed42_r10.json',
            },
            'Qwen-0.5B': {
                'ATLAS': 'atlas_Qwen_Qwen2.5-0.5B_atlas_seed42_r10.json',
                'Standard FL': 'atlas_Qwen_Qwen2.5-0.5B_standard_fl_seed42_r10.json',
            },
        }
        
        for model_name, configs in baseline_comparisons.items():
            for method, filename in configs.items():
                result = self.load_result(filename)
                rounds, accs, f1s, times = self.extract_metrics_per_round(result)

                if rounds is not None:
                    linestyle = '-' if method == 'ATLAS' else '--'
                    # Plot using method-specific color (normalize method label)
                    label = f"{model_name} ({method})"
                    ax.plot(rounds, accs * 100,
                            label=label,
                            color=self._color(method),
                            linestyle=linestyle,
                            linewidth=1.5,
                            alpha=0.8)
        
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
    
    def plot_communication_efficiency(self):
        """Generate Figure 3: Communication Efficiency Analysis."""
        print("Generating Figure 3: Communication Efficiency...")
        
        configs = {
            'ATLAS': 'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
            'ATLAS (No Lap)': 'atlas_distilbert-base-uncased_atlas_no_laplacian_seed42_r10.json',
            'FedAvg (Clustered)': 'atlas_distilbert-base-uncased_fedavg_cluster_seed42_r10.json',
            'Standard FL': 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
        }
        
        # Cumulative communication
        fig, ax = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
        for label, filename in configs.items():
            result = self.load_result(filename)
            if result is None:
                continue
            
            rounds = []
            cumulative_comm = []
            total_comm = 0
            
            for round_data in result['round_metrics']:
                rounds.append(round_data.get('round', len(rounds) + 1))
                upload = sum(round_data.get('comm_upload_bytes', {}).values())
                download = sum(round_data.get('comm_download_bytes', {}).values())
                total_comm += (upload + download)
                cumulative_comm.append(total_comm / (1024 * 1024))
            ax.plot(rounds, cumulative_comm,
                    label=label,
                    color=self._color(label),
                    linestyle=self._linestyle(label),
                    marker=self._marker(label),
                    markevery=2,
                    linewidth=1.5,
                    markersize=4)
        
        ax.set_xlabel('Communication Round', fontweight='bold')
        ax.set_ylabel('Cumulative Data (MB)', fontweight='bold')
        ax.set_title('Communication Overhead', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
        ax.set_xlim(left=1)
        
        figfile = self.output_dir / 'fig3_cumulative_communication.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
        
        # Communication efficiency
        fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
        methods = []
        efficiencies = []
        colors_list = []
        
        for label, filename in configs.items():
            result = self.load_result(filename)
            if result is None:
                continue
            
            comm_stats = self.extract_communication_stats(result)
            final_acc = result['round_metrics'][-1].get('avg_accuracy', 0.0)
            efficiency = (final_acc * 100) / comm_stats['total_comm_mb'] if comm_stats['total_comm_mb'] > 0 else 0
            
            methods.append(label.replace(' (No Lap)', '\n(No Lap)').replace(' (Clustered)', '\n(Clustered)'))
            efficiencies.append(efficiency)
            colors_list.append(self._color(label))
        
        bars = ax.bar(range(len(methods)), efficiencies, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=0, ha='center', fontsize=8)
        ax.set_ylabel('Accuracy / MB (%·MB⁻¹)', fontweight='bold')
        ax.set_title('Communication Efficiency', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{eff:.2f}',
                    ha='center', va='bottom', fontsize=7)
        
        figfile = self.output_dir / 'fig3_communication_efficiency.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
        
        # Total communication
        fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
        methods_comm = []
        total_comms = []
        colors_list_comm = []
        
        for label, filename in configs.items():
            result = self.load_result(filename)
            if result is None:
                continue
            
            comm_stats = self.extract_communication_stats(result)
            methods_comm.append(label.replace(' (No Lap)', '\n(No Lap)').replace(' (Clustered)', '\n(Clustered)'))
            total_comms.append(comm_stats['total_comm_mb'])
            colors_list_comm.append(self._color(label))
        
        bars = ax.bar(range(len(methods_comm)), total_comms, color=colors_list_comm, alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_xticks(range(len(methods_comm)))
        ax.set_xticklabels(methods_comm, rotation=0, ha='center', fontsize=8)
        ax.set_ylabel('Total Communication (MB)', fontweight='bold')
        ax.set_title('Total Data Transferred', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        for bar, comm in zip(bars, total_comms):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{comm:.0f}',
                    ha='center', va='bottom', fontsize=7)
        
        figfile = self.output_dir / 'fig3_total_communication.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
    
    def plot_clustering_analysis(self):
        """Generate Figure 4: Clustering and Heterogeneity Analysis."""
        print("Generating Figure 4: Clustering Analysis...")
        
        result = self.load_result('atlas_distilbert-base-uncased_atlas_seed42_r10.json')
        if result is None:
            print("  Warning: Could not load result for clustering analysis")
            return
        
        # Per-client accuracy evolution
        num_clients = len(result['round_metrics'][0].get('test_accuracies', {}))
        fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
        client_colors = plt.cm.tab20(np.linspace(0, 1, max(2, num_clients)))
        
        for client_id in range(num_clients):
            rounds = []
            accs = []
            for round_data in result['round_metrics']:
                rounds.append(round_data.get('round', len(rounds) + 1))
                acc = round_data.get('test_accuracies', {}).get(str(client_id), 0)
                accs.append(acc * 100)
            
            ax.plot(rounds, accs, color=client_colors[client_id % len(client_colors)], 
                    alpha=0.6, linewidth=1.0)
        
        ax.set_xlabel('Communication Round', fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
        ax.set_title('Per-Client Accuracy Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlim(left=1)
        
        figfile = self.output_dir / 'fig4_per_client_accuracy.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
        
        # Final accuracy distribution
        configs = {
            'ATLAS': 'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
            'Standard FL': 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
        }
        
        final_accs = {}
        for label, filename in configs.items():
            res = self.load_result(filename)
            if res is None:
                continue
            final_round = res['round_metrics'][-1]
            client_accs = [v * 100 for v in final_round.get('test_accuracies', {}).values()]
            final_accs[label] = client_accs
        
        fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
        data = [final_accs.get('ATLAS', []), final_accs.get('Standard FL', [])]
        
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, labels=['ATLAS', 'Standard FL'],
                        showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
        
        colors_box = [COLORS['ATLAS'], COLORS['Standard FL']]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Final Test Accuracy (%)', fontweight='bold')
        ax.set_title('Client Accuracy Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        if all(len(lst) for lst in data):
            atlas_mean = np.mean(data[0])
            fl_mean = np.mean(data[1])
            atlas_std = np.std(data[0])
            fl_std = np.std(data[1])
            y_max = max(max(data[0]) if data[0] else 0, max(data[1]) if data[1] else 0)
            
            ax.text(1, y_max + 2, f'μ={atlas_mean:.1f}%\nσ={atlas_std:.1f}',
                    ha='center', fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            ax.text(2, y_max + 2, f'μ={fl_mean:.1f}%\nσ={fl_std:.1f}',
                    ha='center', fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        figfile = self.output_dir / 'fig4_client_distribution.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
    
    
    def plot_convergence_speed(self):
        """Generate Figure 6: Training Convergence Speed."""
        print("Generating Figure 6: Convergence Speed Analysis...")
        
        configs = {
            'ATLAS': 'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
            'ATLAS (No Lap)': 'atlas_distilbert-base-uncased_atlas_no_laplacian_seed42_r10.json',
            'FedAvg (Clustered)': 'atlas_distilbert-base-uncased_fedavg_cluster_seed42_r10.json',
            'Standard FL': 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
        }
        
        # Accuracy vs cumulative training time
        fig, ax = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
        
        for label, filename in configs.items():
            result = self.load_result(filename)
            if result is None:
                continue
            
            rounds = []
            cumulative_time = []
            accuracies = []
            total_time = 0
            
            for round_data in result['round_metrics']:
                rounds.append(round_data.get('round', len(rounds) + 1))
                total_time += round_data.get('time_seconds', 0.0)
                cumulative_time.append(total_time / 60)
                accuracies.append(round_data.get('avg_accuracy', 0.0) * 100)
            ax.plot(cumulative_time, accuracies, label=label,
                    color=self._color(label),
                    linestyle=self._linestyle(label),
                    marker=self._marker(label),
                    markevery=2, linewidth=1.5, markersize=4)
        
        ax.set_xlabel('Cumulative Time (minutes)', fontweight='bold')
        ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
        ax.set_title('Accuracy vs. Training Time', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
        
        figfile = self.output_dir / 'fig6_accuracy_vs_time.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
        
        # Average time per round
        methods = []
        avg_times = []
        colors_list = []
        
        for label, filename in configs.items():
            result = self.load_result(filename)
            if result is None:
                continue
            
            times = [rd.get('time_seconds', 0.0) for rd in result['round_metrics']]
            avg_time = np.mean(times) / 60 if len(times) else 0.0
            
            methods.append(label.replace(' (No Lap)', '\n(No Lap)').replace(' (Clustered)', '\n(Clustered)'))
            avg_times.append(avg_time)
            colors_list.append(self._color(label))
        
        fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
        bars = ax.bar(range(len(methods)), avg_times, color=colors_list,
                      alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=0, ha='center', fontsize=8)
        ax.set_ylabel('Avg. Time per Round (min)', fontweight='bold')
        ax.set_title('Training Time per Round', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        for bar, time_val in zip(bars, avg_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{time_val:.1f}',
                    ha='center', va='bottom', fontsize=7)
        
        figfile = self.output_dir / 'fig6_time_per_round.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
    
    def plot_clustering_visualization(self):
        """Generate Figure 7: Clustering Visualization."""
        print("Generating Figure 7: Clustering Visualization...")
        
        result = self.load_result('atlas_distilbert-base-uncased_atlas_seed42_r10.json')
        if result is None:
            print("  Warning: Could not load result for clustering visualization")
            return
        
        if 'cluster_labels' not in result or 'phase1_clustering' not in result:
            print("  Warning: No clustering data found in result")
            return
        
        cluster_labels = result['cluster_labels']
        clustering_info = result['phase1_clustering']
        num_clusters = clustering_info.get('num_clusters', 4)
        
        # Create cluster mapping
        clients = sorted([int(k) for k in cluster_labels.keys()])
        clusters = [cluster_labels[str(c)] for c in clients]
        
        # Cluster visualization as bar chart
        fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
        
        # Color map for clusters
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, num_clusters))
        colors = [cluster_colors[clusters[i]] for i in range(len(clients))]
        
        bars = ax.bar(clients, [1] * len(clients), color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Client ID', fontweight='bold')
        ax.set_ylabel('Cluster Assignment', fontweight='bold')
        ax.set_title('Client Cluster Assignments', fontweight='bold')
        ax.set_xticks(clients)
        ax.set_yticks([])
        ax.set_ylim(0, 1.2)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cluster_colors[i], edgecolor='black',
                                 label=f'Cluster {i}') for i in range(num_clusters)]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=8)
        
        # Add cluster labels on bars
        for i, (client, cluster) in enumerate(zip(clients, clusters)):
            ax.text(client, 0.5, f'C{cluster}', ha='center', va='center',
                   fontweight='bold', fontsize=9)
        
        figfile = self.output_dir / 'fig7_clustering_visualization.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
        
        # Also create a cluster size distribution
        fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
        
        cluster_counts = {}
        for c in clusters:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
        
        cluster_ids = sorted(cluster_counts.keys())
        counts = [cluster_counts[cid] for cid in cluster_ids]
        colors_bar = [cluster_colors[cid] for cid in cluster_ids]
        
        bars = ax.bar(cluster_ids, counts, color=colors_bar, edgecolor='black',
                      linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Cluster ID', fontweight='bold')
        ax.set_ylabel('Number of Clients', fontweight='bold')
        ax.set_title('Cluster Size Distribution', fontweight='bold')
        ax.set_xticks(cluster_ids)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        figfile = self.output_dir / 'fig7_cluster_distribution.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)

    def plot_clustering_map(self):
        """Generate a 2D embedding map of clients colored by cluster assignment.

        Embedding is computed from per-client accuracy time series using PCA
        (or t-SNE if available). This provides a spatial view of cluster groups.
        """
        print("Generating Figure 9: Clustering Map...")

        result = self.load_result('atlas_distilbert-base-uncased_atlas_seed42_r10.json')
        if result is None:
            print("  Warning: Could not load result for clustering map")
            return

        # need cluster labels
        if 'cluster_labels' not in result:
            print("  Warning: No cluster_labels found in result")
            return

        cluster_labels = result['cluster_labels']
        clients = sorted([int(k) for k in cluster_labels.keys()])
        clusters = [cluster_labels[str(c)] for c in clients]
        num_clusters = max(clusters) + 1 if clusters else 0

        # Build per-client accuracy time-series matrix (clients x rounds)
        rounds = [rd.get('round', i + 1) for i, rd in enumerate(result['round_metrics'])]
        T = len(rounds)
        X = []
        for c in clients:
            series = []
            for rd in result['round_metrics']:
                series.append(rd.get('test_accuracies', {}).get(str(c), 0.0))
            X.append(series)
        X = np.array(X)

        # If no temporal info (T==1) try to augment with final accuracy and importance if available
        if X.shape[1] <= 1:
            alt = []
            for i, c in enumerate(clients):
                final_acc = X[i, -1] if X.shape[1] else 0.0
                alt.append([final_acc])
            X = np.array(alt)

        # Standardize
        X_mean = X.mean(axis=0)
        Xc = X - X_mean

        # Try t-SNE if available, else PCA via numpy SVD
        try:
            from sklearn.manifold import TSNE
            embed = TSNE(n_components=2, random_state=42, init='pca')
            Y = embed.fit_transform(Xc)
        except Exception:
            # PCA fallback using SVD
            try:
                # compute principal components via SVD on centered data
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                # scores = U * S, but U @ np.diag(S) gives same as Xc @ Vt.T
                Y = Xc @ Vt.T[:, :2]
            except Exception:
                print('  Warning: embedding failed; cannot compute clustering map')
                return

        # Plot embedding colored by cluster
        fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, max(1, num_clusters)))
        colors = [cluster_colors[clusters[i] % len(cluster_colors)] for i in range(len(clients))]

        sc = ax.scatter(Y[:, 0], Y[:, 1], c=colors, s=80, edgecolor='k', linewidth=0.7)

        # Annotate client ids if not too many
        if len(clients) <= 40:
            for i, c in enumerate(clients):
                ax.text(Y[i, 0], Y[i, 1], str(c), fontsize=7, va='center', ha='center')

        # Legend for clusters
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor=cluster_colors[i % len(cluster_colors)], edgecolor='k', label=f'Cluster {i}')
                        for i in range(num_clusters)]
        if legend_elems:
            ax.legend(handles=legend_elems, loc='best', framealpha=0.9, fontsize=8)

        ax.set_xlabel('Embedding 1', fontweight='bold')
        ax.set_ylabel('Embedding 2', fontweight='bold')
        ax.set_title('Client Embedding Colored by Cluster', fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--')

        figfile = self.output_dir / 'fig9_cluster_map.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
    
    def plot_importance_scores(self):
        """Generate Figure 8: Layer Importance Scores."""
        print("Generating Figure 8: Layer Importance Scores...")
        
        result = self.load_result('atlas_distilbert-base-uncased_atlas_seed42_r10.json')
        if result is None:
            print("  Warning: Could not load result for importance scores")
            return
        
        # Extract importance scores from phase2_rank_allocation
        if 'phase2_rank_allocation' not in result:
            print("  Warning: No importance scores found in result")
            return
        
        # Get importance scores from first allocation (cluster 0)
        allocations = result['phase2_rank_allocation']
        if not allocations or 'importance_scores' not in allocations[0]:
            print("  Warning: No importance scores in phase2_rank_allocation")
            return
        
        importance_scores = allocations[0]['importance_scores']
        
        # Prepare data: normalize to relative importance (percent)
        layers = list(importance_scores.keys())
        raw_scores = np.array([importance_scores[layer] for layer in layers], dtype=float)
        total = raw_scores.sum() if raw_scores.size else 0.0
        if total <= 0:
            print("  Warning: importance scores sum to zero; skipping importance plots")
            return

        rel = raw_scores / total

        # Create visualization (relative importance in percent) - main view
        fig, ax = plt.subplots(figsize=(8.0, 3.5), constrained_layout=True)
        colors_gradient = plt.cm.viridis(np.linspace(0.25, 0.85, len(layers)))

        rel_pct = rel * 100.0
        bars = ax.bar(range(len(layers)), rel_pct, color=colors_gradient,
                      edgecolor='black', linewidth=1.0, alpha=0.95)

        ax.set_xlabel('Layer', fontweight='bold')
        ax.set_ylabel('Relative Importance (%)', fontweight='bold')
        ax.set_title('Layer-wise Importance Scores', fontweight='bold')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

        # Annotate bars with percent values and cap y-axis at 100%
        for i, (bar, pct) in enumerate(zip(bars, rel_pct)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

        ax.set_ylim(0, max(100.0, rel_pct.max() * 1.05))

        figfile = self.output_dir / 'fig8_importance_scores.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)

        # Complementary log-scale view of raw scores so small contributors are visible
        try:
            fig, ax = plt.subplots(figsize=(8.0, 3.5), constrained_layout=True)
            eps = 1e-8
            ax.bar(range(len(layers)), np.log10(raw_scores + eps), color=colors_gradient,
                   edgecolor='black', linewidth=1.0, alpha=0.95)
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('log10(Raw importance + eps)', fontweight='bold')
            ax.set_title('Layer-wise Importance (log scale)', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            figfile = self.output_dir / 'fig8_importance_scores_log.png'
            fig.savefig(figfile, dpi=300)
            print(f"  Saved to {figfile}")
            plt.close(fig)
        except Exception:
            pass

        # Also create a pie chart for relative importance (show top contributors)
        fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)

        threshold = 0.05  # show layers >=5% individually
        major_layers = [layers[i] for i in range(len(layers)) if rel[i] >= threshold]
        major_scores = [rel[i] for i in range(len(layers)) if rel[i] >= threshold]
        other_score = rel[rel < threshold].sum()
        if other_score > 0:
            major_layers.append('Others')
            major_scores.append(other_score)

        colors_pie = plt.cm.Set3(np.linspace(0, 1, max(1, len(major_layers))))
        wedges, texts, autotexts = ax.pie(major_scores, labels=major_layers,
                                           autopct='%1.1f%%', startangle=90,
                                           colors=colors_pie,
                                           textprops={'fontsize': 10, 'fontweight': 'bold'})

        ax.set_title('Relative Layer Importance', fontweight='bold', fontsize=12)

        figfile = self.output_dir / 'fig8_importance_distribution.png'
        fig.savefig(figfile, dpi=300)
        print(f"  Saved to {figfile}")
        plt.close(fig)
    
    def generate_all_plots(self):
        """Generate all publication plots."""
        print("\n" + "="*60)
        print("Generating IEEE-Quality Publication Plots")
        print("="*60 + "\n")
        
        self.plot_ablation_study()
        self.plot_model_comparison()
        self.plot_communication_efficiency()
        self.plot_clustering_analysis()
        self.plot_convergence_speed()
        self.plot_clustering_visualization()
        self.plot_importance_scores()
        
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
