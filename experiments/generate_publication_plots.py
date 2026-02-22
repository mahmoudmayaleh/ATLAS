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
from matplotlib import rcParams

# Set IEEE-quality plot parameters
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 12
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# Color palette (IEEE-friendly)
COLORS = {
    'atlas': '#1f77b4',           # Blue
    'no_laplacian': '#ff7f0e',    # Orange
    'fedavg_cluster': '#2ca02c',  # Green
    'standard_fl': '#d62728',     # Red
    'gpt2': '#9467bd',            # Purple
    'qwen': '#8c564b',            # Brown
    'distilbert': '#1f77b4',      # Blue
}

# Line styles
LINE_STYLES = {
    'atlas': '-',
    'no_laplacian': '--',
    'fedavg_cluster': '-.',
    'standard_fl': ':',
    'gpt2': '-',
    'qwen': '--',
    'distilbert': '-',
}

MARKERS = {
    'atlas': 'o',
    'no_laplacian': 's',
    'fedavg_cluster': '^',
    'standard_fl': 'v',
    'gpt2': 'D',
    'qwen': 'p',
    'distilbert': 'o',
}


class PublicationPlotter:
    """Generate publication-quality plots for ATLAS experiments."""
    
    def __init__(self, results_dir: str = "results", output_dir: str = "figures"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store loaded results
        self.results = {}
        
    def load_result(self, filename: str) -> Dict:
        """Load a single results JSON file."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found")
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def extract_metrics_per_round(self, result: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract accuracy, F1, and time metrics per round."""
        if result is None:
            return None, None, None
        
        rounds = []
        accuracies = []
        f1_scores = []
        times = []
        
        for round_data in result['round_metrics']:
            rounds.append(round_data['round'])
            accuracies.append(round_data['avg_accuracy'])
            
            # Calculate average F1 score
            f1_dict = round_data.get('test_f1', {})
            if f1_dict:
                avg_f1 = np.mean([v for v in f1_dict.values()])
                f1_scores.append(avg_f1)
            else:
                f1_scores.append(0)
            
            times.append(round_data['time_seconds'])
        
        return np.array(rounds), np.array(accuracies), np.array(f1_scores), np.array(times)
    
    def extract_communication_stats(self, result: Dict) -> Dict:
        """Extract communication statistics."""
        if result is None:
            return None
        
        total_upload = 0
        total_download = 0
        
        for round_data in result['round_metrics']:
            upload_bytes = round_data.get('comm_upload_bytes', {})
            download_bytes = round_data.get('comm_download_bytes', {})
            
            total_upload += sum(upload_bytes.values())
            total_download += sum(download_bytes.values())
        
        return {
            'total_upload_mb': total_upload / (1024 * 1024),
            'total_download_mb': total_download / (1024 * 1024),
            'total_comm_mb': (total_upload + total_download) / (1024 * 1024),
        }
    
    def plot_ablation_study(self):
        """
        Figure 1: Ablation Study
        Compare ATLAS variants to show the impact of each component.
        """
        print("Generating Figure 1: Ablation Study...")
        
        # Load DistilBERT results for fair comparison
        configs = {
            'ATLAS (Full)': 'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
            'ATLAS (No Laplacian)': 'atlas_distilbert-base-uncased_atlas_no_laplacian_seed42_r10.json',
            'FedAvg (Clustered)': 'atlas_distilbert-base-uncased_fedavg_cluster_seed42_r10.json',
            'Standard FL': 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        
        # Plot accuracy over rounds
        ax1 = axes[0]
        for label, filename in configs.items():
            result = self.load_result(filename)
            rounds, accs, f1s, times = self.extract_metrics_per_round(result)
            
            if rounds is not None:
                key = label.lower().replace(' ', '_').replace('(', '').replace(')', '').split('_')[0]
                if key == 'fedavg':
                    key = 'fedavg_cluster'
                
                ax1.plot(rounds, accs * 100, 
                        label=label,
                        color=COLORS.get(key, '#000000'),
                        linestyle=LINE_STYLES.get(key, '-'),
                        marker=MARKERS.get(key, 'o'),
                        markevery=1,
                        linewidth=1.5,
                        markersize=5)
        
        ax1.set_xlabel('Communication Round', fontweight='bold')
        ax1.set_ylabel('Average Accuracy (%)', fontweight='bold')
        ax1.set_title('(a) Accuracy Convergence', fontweight='bold', loc='left')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.legend(loc='lower right', framealpha=0.95)
        ax1.set_xlim(left=1)
        
        # Plot F1 scores
        ax2 = axes[1]
        for label, filename in configs.items():
            result = self.load_result(filename)
            rounds, accs, f1s, times = self.extract_metrics_per_round(result)
            
            if rounds is not None:
                key = label.lower().replace(' ', '_').replace('(', '').replace(')', '').split('_')[0]
                if key == 'fedavg':
                    key = 'fedavg_cluster'
                
                ax2.plot(rounds, f1s * 100,
                        label=label,
                        color=COLORS.get(key, '#000000'),
                        linestyle=LINE_STYLES.get(key, '-'),
                        marker=MARKERS.get(key, 'o'),
                        markevery=1,
                        linewidth=1.5,
                        markersize=5)
        
        ax2.set_xlabel('Communication Round', fontweight='bold')
        ax2.set_ylabel('Average F1 Score (%)', fontweight='bold')
        ax2.set_title('(b) F1 Score Convergence', fontweight='bold', loc='left')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.legend(loc='lower right', framealpha=0.95)
        ax2.set_xlim(left=1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_ablation_study.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig1_ablation_study.png', dpi=300, bbox_inches='tight')
        print(f"  Saved to {self.output_dir / 'fig1_ablation_study.pdf'}")
        plt.close()
    
    def plot_model_comparison(self):
        """
        Figure 2: Cross-Model Performance
        Compare ATLAS performance across different model architectures.
        """
        print("Generating Figure 2: Model Comparison...")
        
        # Load results for different models (using seeds for variance)
        models_data = {
            'DistilBERT': [
                'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
            ],
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
        
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        
        # Plot accuracy for different models
        ax1 = axes[0]
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
                
                ax1.plot(rounds_arr, mean_accs * 100,
                        label=model_name,
                        color=COLORS.get(key, '#000000'),
                        linestyle=LINE_STYLES.get(key, '-'),
                        marker=MARKERS.get(key, 'o'),
                        markevery=1,
                        linewidth=1.5,
                        markersize=5)
                
                if len(all_accs) > 1:
                    ax1.fill_between(rounds_arr, 
                                    (mean_accs - std_accs) * 100,
                                    (mean_accs + std_accs) * 100,
                                    color=COLORS.get(key, '#000000'),
                                    alpha=0.15)
        
        ax1.set_xlabel('Communication Round', fontweight='bold')
        ax1.set_ylabel('Average Accuracy (%)', fontweight='bold')
        ax1.set_title('(a) Model Architecture Comparison', fontweight='bold', loc='left')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.legend(loc='best', framealpha=0.95)
        ax1.set_xlim(left=1)
        
        # Plot baseline comparison for each model
        ax2 = axes[1]
        
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
                    key = model_name.lower().replace('-', '').replace('.', '')
                    if 'distilbert' in key:
                        key = 'distilbert'
                    elif 'gpt' in key:
                        key = 'gpt2'
                    elif 'qwen' in key:
                        key = 'qwen'
                    
                    label = f"{model_name} ({method})"
                    ax2.plot(rounds, accs * 100,
                            label=label,
                            color=COLORS.get(key, '#000000'),
                            linestyle=linestyle,
                            linewidth=1.5,
                            alpha=0.8)
        
        ax2.set_xlabel('Communication Round', fontweight='bold')
        ax2.set_ylabel('Average Accuracy (%)', fontweight='bold')
        ax2.set_title('(b) ATLAS vs. Standard FL', fontweight='bold', loc='left')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.legend(loc='best', framealpha=0.95, fontsize=7)
        ax2.set_xlim(left=1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_model_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig2_model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  Saved to {self.output_dir / 'fig2_model_comparison.pdf'}")
        plt.close()
    
    def plot_communication_efficiency(self):
        """
        Figure 3: Communication Efficiency Analysis
        Show communication costs and efficiency gains.
        """
        print("Generating Figure 3: Communication Efficiency...")
        
        configs = {
            'ATLAS': 'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
            'ATLAS (No Lap)': 'atlas_distilbert-base-uncased_atlas_no_laplacian_seed42_r10.json',
            'FedAvg (Clustered)': 'atlas_distilbert-base-uncased_fedavg_cluster_seed42_r10.json',
            'Standard FL': 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
        
        # (a) Cumulative communication over rounds
        ax1 = axes[0]
        for label, filename in configs.items():
            result = self.load_result(filename)
            if result is None:
                continue
            
            rounds = []
            cumulative_comm = []
            total_comm = 0
            
            for round_data in result['round_metrics']:
                rounds.append(round_data['round'])
                upload = sum(round_data.get('comm_upload_bytes', {}).values())
                download = sum(round_data.get('comm_download_bytes', {}).values())
                total_comm += (upload + download)
                cumulative_comm.append(total_comm / (1024 * 1024))  # Convert to MB
            
            key = label.lower().replace(' ', '_').replace('(', '').replace(')', '').split('_')[0]
            if key == 'fedavg':
                key = 'fedavg_cluster'
            
            ax1.plot(rounds, cumulative_comm,
                    label=label,
                    color=COLORS.get(key, '#000000'),
                    linestyle=LINE_STYLES.get(key, '-'),
                    marker=MARKERS.get(key, 'o'),
                    markevery=2,
                    linewidth=1.5,
                    markersize=4)
        
        ax1.set_xlabel('Communication Round', fontweight='bold')
        ax1.set_ylabel('Cumulative Data (MB)', fontweight='bold')
        ax1.set_title('(a) Communication Overhead', fontweight='bold', loc='left')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.legend(loc='upper left', framealpha=0.95, fontsize=7)
        ax1.set_xlim(left=1)
        
        # (b) Communication efficiency (accuracy per MB)
        ax2 = axes[1]
        methods = []
        efficiencies = []
        colors_list = []
        
        for label, filename in configs.items():
            result = self.load_result(filename)
            if result is None:
                continue
            
            comm_stats = self.extract_communication_stats(result)
            final_acc = result['round_metrics'][-1]['avg_accuracy']
            
            # Efficiency: accuracy per MB communicated
            efficiency = (final_acc * 100) / comm_stats['total_comm_mb']
            
            methods.append(label.replace(' (No Lap)', '\n(No Lap)').replace(' (Clustered)', '\n(Clustered)'))
            efficiencies.append(efficiency)
            
            key = label.lower().replace(' ', '_').replace('(', '').replace(')', '').split('_')[0]
            if key == 'fedavg':
                key = 'fedavg_cluster'
            colors_list.append(COLORS.get(key, '#000000'))
        
        bars = ax2.bar(range(len(methods)), efficiencies, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1)
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=0, ha='center', fontsize=8)
        ax2.set_ylabel('Accuracy / MB (%·MB⁻¹)', fontweight='bold')
        ax2.set_title('(b) Communication Efficiency', fontweight='bold', loc='left')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        # Add value labels on bars
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff:.2f}',
                    ha='center', va='bottom', fontsize=7)
        
        # (c) Total communication comparison
        ax3 = axes[2]
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
            
            key = label.lower().replace(' ', '_').replace('(', '').replace(')', '').split('_')[0]
            if key == 'fedavg':
                key = 'fedavg_cluster'
            colors_list_comm.append(COLORS.get(key, '#000000'))
        
        bars = ax3.bar(range(len(methods_comm)), total_comms, color=colors_list_comm, alpha=0.7, edgecolor='black', linewidth=1)
        ax3.set_xticks(range(len(methods_comm)))
        ax3.set_xticklabels(methods_comm, rotation=0, ha='center', fontsize=8)
        ax3.set_ylabel('Total Communication (MB)', fontweight='bold')
        ax3.set_title('(c) Total Data Transferred', fontweight='bold', loc='left')
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        # Add value labels on bars
        for bar, comm in zip(bars, total_comms):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{comm:.0f}',
                    ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_communication_efficiency.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig3_communication_efficiency.png', dpi=300, bbox_inches='tight')
        print(f"  Saved to {self.output_dir / 'fig3_communication_efficiency.pdf'}")
        plt.close()
    
    def plot_clustering_analysis(self):
        """
        Figure 4: Clustering and Heterogeneity Analysis
        Show how ATLAS handles client heterogeneity through clustering.
        """
        print("Generating Figure 4: Clustering Analysis...")
        
        # Load DistilBERT ATLAS result for detailed analysis
        result = self.load_result('atlas_distilbert-base-uncased_atlas_seed42_r10.json')
        
        if result is None:
            print("  Warning: Could not load result for clustering analysis")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        
        # (a) Per-client accuracy over time
        ax1 = axes[0]
        
        # Extract per-client accuracies
        num_clients = len(result['round_metrics'][0]['test_accuracies'])
        client_colors = plt.cm.tab20(np.linspace(0, 1, num_clients))
        
        for client_id in range(num_clients):
            rounds = []
            accs = []
            
            for round_data in result['round_metrics']:
                rounds.append(round_data['round'])
                acc = round_data['test_accuracies'].get(str(client_id), 0)
                accs.append(acc * 100)
            
            ax1.plot(rounds, accs,
                    label=f'Client {client_id}',
                    color=client_colors[client_id],
                    alpha=0.6,
                    linewidth=1.2)
        
        ax1.set_xlabel('Communication Round', fontweight='bold')
        ax1.set_ylabel('Test Accuracy (%)', fontweight='bold')
        ax1.set_title('(a) Per-Client Accuracy Evolution', fontweight='bold', loc='left')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.legend(loc='right', bbox_to_anchor=(1.15, 0.5), framealpha=0.95, fontsize=6, ncol=1)
        ax1.set_xlim(left=1)
        
        # (b) Final accuracy distribution and clustering benefit
        ax2 = axes[1]
        
        # Compare final accuracies between ATLAS and Standard FL
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
            client_accs = [v * 100 for v in final_round['test_accuracies'].values()]
            final_accs[label] = client_accs
        
        # Create box plot
        positions = [1, 2]
        bp = ax2.boxplot([final_accs.get('ATLAS', []), final_accs.get('Standard FL', [])],
                         positions=positions,
                         widths=0.6,
                         patch_artist=True,
                         labels=['ATLAS', 'Standard FL'],
                         showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
        
        # Color the boxes
        colors_box = [COLORS['atlas'], COLORS['standard_fl']]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax2.set_ylabel('Final Test Accuracy (%)', fontweight='bold')
        ax2.set_title('(b) Client Accuracy Distribution', fontweight='bold', loc='left')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        # Add statistical annotations
        if 'ATLAS' in final_accs and 'Standard FL' in final_accs:
            atlas_mean = np.mean(final_accs['ATLAS'])
            fl_mean = np.mean(final_accs['Standard FL'])
            atlas_std = np.std(final_accs['ATLAS'])
            fl_std = np.std(final_accs['Standard FL'])
            
            # Add mean value annotations
            y_max = max(max(final_accs['ATLAS']), max(final_accs['Standard FL']))
            ax2.text(1, y_max + 2, f'μ={atlas_mean:.1f}%\nσ={atlas_std:.1f}', 
                    ha='center', fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            ax2.text(2, y_max + 2, f'μ={fl_mean:.1f}%\nσ={fl_std:.1f}', 
                    ha='center', fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_clustering_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig4_clustering_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  Saved to {self.output_dir / 'fig4_clustering_analysis.pdf'}")
        plt.close()
    
    def plot_eta_parameter_study(self):
        """
        Figure 5: Eta Parameter Sensitivity Analysis
        Show the impact of eta parameter on performance.
        """
        print("Generating Figure 5: Eta Parameter Study...")
        
        eta_configs = {
            'η = 0.0': 'atlas_integrated_full_atlas_00_eta_seed42.json',
            'η = 0.1': 'atlas_integrated_full_atlas_01_eta_seed42.json',
            'η = 0.5': 'atlas_integrated_full_atlas_05_eta_seed42.json',
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        
        # (a) Accuracy convergence for different eta values
        ax1 = axes[0]
        colors_eta = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for (label, filename), color in zip(eta_configs.items(), colors_eta):
            result = self.load_result(filename)
            rounds, accs, f1s, times = self.extract_metrics_per_round(result)
            
            if rounds is not None:
                ax1.plot(rounds, accs * 100,
                        label=label,
                        color=color,
                        marker='o',
                        markevery=1,
                        linewidth=1.5,
                        markersize=5)
        
        ax1.set_xlabel('Communication Round', fontweight='bold')
        ax1.set_ylabel('Average Accuracy (%)', fontweight='bold')
        ax1.set_title('(a) Impact of Eta Parameter', fontweight='bold', loc='left')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.legend(loc='best', framealpha=0.95)
        ax1.set_xlim(left=1)
        
        # (b) Summary comparison bar chart
        ax2 = axes[1]
        
        eta_values = []
        final_accs = []
        final_f1s = []
        
        for label, filename in eta_configs.items():
            result = self.load_result(filename)
            if result is None:
                continue
            
            eta_val = label.split('=')[1].strip()
            eta_values.append(eta_val)
            
            final_round = result['round_metrics'][-1]
            final_accs.append(final_round['avg_accuracy'] * 100)
            
            # Calculate average F1
            f1_dict = final_round.get('test_f1', {})
            avg_f1 = np.mean([v for v in f1_dict.values()]) * 100
            final_f1s.append(avg_f1)
        
        x = np.arange(len(eta_values))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, final_accs, width, label='Accuracy', 
                       color=COLORS['atlas'], alpha=0.7, edgecolor='black')
        bars2 = ax2.bar(x + width/2, final_f1s, width, label='F1 Score',
                       color=COLORS['no_laplacian'], alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Eta Value (η)', fontweight='bold')
        ax2.set_ylabel('Performance (%)', fontweight='bold')
        ax2.set_title('(b) Final Performance Comparison', fontweight='bold', loc='left')
        ax2.set_xticks(x)
        ax2.set_xticklabels(eta_values)
        ax2.legend(loc='best', framealpha=0.95)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=7)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig5_eta_parameter_study.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig5_eta_parameter_study.png', dpi=300, bbox_inches='tight')
        print(f"  Saved to {self.output_dir / 'fig5_eta_parameter_study.pdf'}")
        plt.close()
    
    def plot_convergence_speed(self):
        """
        Figure 6: Training Convergence Speed
        Compare time to reach target accuracy thresholds.
        """
        print("Generating Figure 6: Convergence Speed Analysis...")
        
        configs = {
            'ATLAS': 'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
            'ATLAS (No Lap)': 'atlas_distilbert-base-uncased_atlas_no_laplacian_seed42_r10.json',
            'FedAvg (Clustered)': 'atlas_distilbert-base-uncased_fedavg_cluster_seed42_r10.json',
            'Standard FL': 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        
        # (a) Cumulative time to accuracy
        ax1 = axes[0]
        
        for label, filename in configs.items():
            result = self.load_result(filename)
            if result is None:
                continue
            
            rounds = []
            cumulative_time = []
            accuracies = []
            total_time = 0
            
            for round_data in result['round_metrics']:
                rounds.append(round_data['round'])
                total_time += round_data['time_seconds']
                cumulative_time.append(total_time / 60)  # Convert to minutes
                accuracies.append(round_data['avg_accuracy'] * 100)
            
            key = label.lower().replace(' ', '_').replace('(', '').replace(')', '').split('_')[0]
            if key == 'fedavg':
                key = 'fedavg_cluster'
            
            ax1.plot(cumulative_time, accuracies,
                    label=label,
                    color=COLORS.get(key, '#000000'),
                    linestyle=LINE_STYLES.get(key, '-'),
                    marker=MARKERS.get(key, 'o'),
                    markevery=2,
                    linewidth=1.5,
                    markersize=4)
        
        ax1.set_xlabel('Cumulative Time (minutes)', fontweight='bold')
        ax1.set_ylabel('Average Accuracy (%)', fontweight='bold')
        ax1.set_title('(a) Accuracy vs. Training Time', fontweight='bold', loc='left')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.legend(loc='lower right', framealpha=0.95, fontsize=8)
        
        # (b) Time per round comparison
        ax2 = axes[1]
        
        methods = []
        avg_times = []
        colors_list = []
        
        for label, filename in configs.items():
            result = self.load_result(filename)
            if result is None:
                continue
            
            times = [rd['time_seconds'] for rd in result['round_metrics']]
            avg_time = np.mean(times) / 60  # minutes
            
            methods.append(label.replace(' (No Lap)', '\n(No Lap)').replace(' (Clustered)', '\n(Clustered)'))
            avg_times.append(avg_time)
            
            key = label.lower().replace(' ', '_').replace('(', '').replace(')', '').split('_')[0]
            if key == 'fedavg':
                key = 'fedavg_cluster'
            colors_list.append(COLORS.get(key, '#000000'))
        
        bars = ax2.bar(range(len(methods)), avg_times, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1)
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=0, ha='center', fontsize=8)
        ax2.set_ylabel('Avg. Time per Round (min)', fontweight='bold')
        ax2.set_title('(b) Training Time per Round', fontweight='bold', loc='left')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        # Add value labels
        for bar, time_val in zip(bars, avg_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.1f}',
                    ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig6_convergence_speed.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig6_convergence_speed.png', dpi=300, bbox_inches='tight')
        print(f"  Saved to {self.output_dir / 'fig6_convergence_speed.pdf'}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all publication plots."""
        print("\n" + "="*60)
        print("Generating IEEE-Quality Publication Plots")
        print("="*60 + "\n")
        
        self.plot_ablation_study()
        self.plot_model_comparison()
        self.plot_communication_efficiency()
        self.plot_clustering_analysis()
        self.plot_eta_parameter_study()
        self.plot_convergence_speed()
        
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
