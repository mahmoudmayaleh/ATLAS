"""
Visualization Utilities for ATLAS Experiments

Provides plotting functions for results comparison, convergence curves,
memory-accuracy trade-offs, and other experiment visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


class ExperimentVisualizer:
    """Main visualization class for ATLAS experiments"""
    
    def __init__(self, results_dir: str = "./results", save_dir: str = "./figures"):
        self.results_dir = Path(results_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments = {}
        self.round_data = {}
        
    def load_experiment(self, experiment_name: str):
        """Load experiment results"""
        # Load results file (contains both summary and round data)
        results_path = self.results_dir / f"{experiment_name}_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                data = json.load(f)
                self.experiments[experiment_name] = data
                # Extract round metrics if available
                if 'round_metrics' in data:
                    self.round_data[experiment_name] = data['round_metrics']
            print(f"[OK] Loaded: {experiment_name}")
        else:
            print(f"[WARNING] Not found: {experiment_name}")
    
    def load_all_experiments(self):
        """Load all experiments in results directory"""
        for results_path in self.results_dir.glob("*_results.json"):
            exp_name = results_path.stem.replace("_results", "")
            self.load_experiment(exp_name)
        
        print(f"[OK] Loaded {len(self.experiments)} experiments")
    
    def plot_convergence_curves(self, experiment_names: Optional[List[str]] = None,
                                metric: str = "accuracy",
                                save_name: str = "convergence.png"):
        """Plot training convergence curves"""
        if experiment_names is None:
            experiment_names = list(self.round_data.keys())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set ylabel based on metric
        if metric == "accuracy":
            ylabel = "Accuracy"
        elif metric == "loss":
            ylabel = "Loss"
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        for exp_name in self.experiments.keys():
            if exp_name not in self.round_data:
                continue
            
            rounds_data = self.round_data[exp_name]
            rounds = [r['round'] for r in rounds_data]
            
            if metric == "accuracy":
                values = [r['accuracy'] for r in rounds_data]
            elif metric == "loss":
                values = [r['loss'] for r in rounds_data]
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            ax.plot(rounds, values, marker='o', markersize=4, 
                   linewidth=2, label=exp_name, alpha=0.8)
        
        ax.set_xlabel("Round", fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(f"Training Convergence: {ylabel}", fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_comparison_bar(self, metric: str = "final_accuracy",
                           save_name: str = "comparison_bar.png"):
        """Plot bar chart comparing experiments on a metric"""
        if not self.experiments:
            print("❌ No experiments loaded!")
            return
        
        exp_names = list(self.experiments.keys())
        
        # Extract values
        if metric == "final_accuracy":
            values = [self.experiments[name]['final_accuracy'] for name in exp_names]
            ylabel = "Final Accuracy"
            title = "Final Accuracy Comparison"
        elif metric == "peak_memory_mb":
            values = [self.experiments[name]['peak_memory_mb'] for name in exp_names]
            ylabel = "Peak Memory (MB)"
            title = "Memory Usage Comparison"
        elif metric == "total_comm_mb":
            values = [self.experiments[name]['total_comm_mb'] for name in exp_names]
            ylabel = "Total Communication (MB)"
            title = "Communication Cost Comparison"
        elif metric == "convergence_round":
            values = [self.experiments[name]['convergence_round'] for name in exp_names]
            ylabel = "Convergence Round"
            title = "Convergence Speed Comparison"
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = sns.color_palette("husl", len(exp_names))
        bars = ax.bar(range(len(exp_names)), values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}' if metric != "convergence_round" else f'{int(value)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_memory_accuracy_tradeoff(self, save_name: str = "memory_accuracy_tradeoff.png"):
        """Plot memory vs accuracy trade-off"""
        if not self.experiments:
            print("❌ No experiments loaded!")
            return
        
        exp_names = list(self.experiments.keys())
        accuracies = [self.experiments[name]['final_accuracy'] for name in exp_names]
        memories = [self.experiments[name]['peak_memory_mb'] for name in exp_names]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        colors = sns.color_palette("husl", len(exp_names))
        
        for i, (name, acc, mem) in enumerate(zip(exp_names, accuracies, memories)):
            ax.scatter(mem, acc, s=300, color=colors[i], 
                      alpha=0.7, edgecolor='black', linewidth=2, label=name)
            ax.text(mem, acc, name, fontsize=9, ha='center', va='center', fontweight='bold')
        
        ax.set_xlabel("Peak Memory (MB)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Final Accuracy", fontsize=13, fontweight='bold')
        ax.set_title("Memory-Accuracy Trade-off", fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Pareto frontier (optional)
        # Sort by memory
        sorted_indices = np.argsort(memories)
        pareto_mem = []
        pareto_acc = []
        max_acc = -1
        
        for idx in sorted_indices:
            if accuracies[idx] > max_acc:
                pareto_mem.append(memories[idx])
                pareto_acc.append(accuracies[idx])
                max_acc = accuracies[idx]
        
        if len(pareto_mem) > 1:
            ax.plot(pareto_mem, pareto_acc, 'r--', linewidth=2, 
                   alpha=0.5, label='Pareto Frontier')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_communication_comparison(self, save_name: str = "communication_comparison.png"):
        """Plot communication cost breakdown"""
        if not self.experiments:
            print("[ERROR] No experiments loaded!")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract communication data from experiments
        exp_names = list(self.experiments.keys())
        comm_per_round = [self.experiments[name].get('comm_mb_per_round', 0) for name in exp_names]
        total_comm = [self.experiments[name].get('total_comm_mb', 0) for name in exp_names]
        
        x = np.arange(len(exp_names))
        width = 0.35
        
        ax.bar(x - width/2, comm_per_round, width, label='Per Round', alpha=0.8)
        ax.bar(x + width/2, total_comm, width, label='Total', alpha=0.8)
        
        ax.set_xlabel("Experiment", fontsize=13, fontweight='bold')
        ax.set_ylabel("Communication (MB)", fontsize=13, fontweight='bold')
        ax.set_title("Communication Costs Comparison", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
        plt.close()
    
    def plot_comprehensive_dashboard(self, save_name: str = "dashboard.png"):
        """Create comprehensive dashboard with multiple metrics"""
        if not self.experiments or not self.round_data:
            print("❌ No data loaded!")
            return
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Convergence curves
        ax1 = fig.add_subplot(gs[0, :2])
        for exp_name, rounds_data in self.round_data.items():
            rounds = [r['round'] for r in rounds_data]
            accs = [r['accuracy'] for r in rounds_data]
            ax1.plot(rounds, accs, marker='o', markersize=3, linewidth=2, label=exp_name)
        ax1.set_xlabel("Round", fontweight='bold')
        ax1.set_ylabel("Accuracy", fontweight='bold')
        ax1.set_title("Training Convergence", fontweight='bold', fontsize=13)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Final accuracy comparison
        ax2 = fig.add_subplot(gs[0, 2])
        exp_names = list(self.experiments.keys())
        accs = [self.experiments[name]['final_accuracy'] for name in exp_names]
        colors = sns.color_palette("husl", len(exp_names))
        ax2.barh(range(len(exp_names)), accs, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(exp_names)))
        ax2.set_yticklabels(exp_names, fontsize=9)
        ax2.set_xlabel("Accuracy", fontweight='bold')
        ax2.set_title("Final Accuracy", fontweight='bold', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Memory usage
        ax3 = fig.add_subplot(gs[1, 0])
        memories = [self.experiments[name]['peak_memory_mb'] for name in exp_names]
        ax3.bar(range(len(exp_names)), memories, color=colors, alpha=0.8)
        ax3.set_xticks(range(len(exp_names)))
        ax3.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel("Memory (MB)", fontweight='bold')
        ax3.set_title("Peak Memory", fontweight='bold', fontsize=13)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Communication cost
        ax4 = fig.add_subplot(gs[1, 1])
        comms = [self.experiments[name]['total_comm_mb'] for name in exp_names]
        ax4.bar(range(len(exp_names)), comms, color=colors, alpha=0.8)
        ax4.set_xticks(range(len(exp_names)))
        ax4.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel("Communication (MB)", fontweight='bold')
        ax4.set_title("Total Communication", fontweight='bold', fontsize=13)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Convergence speed
        ax5 = fig.add_subplot(gs[1, 2])
        conv_rounds = [self.experiments[name]['convergence_round'] for name in exp_names]
        ax5.bar(range(len(exp_names)), conv_rounds, color=colors, alpha=0.8)
        ax5.set_xticks(range(len(exp_names)))
        ax5.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
        ax5.set_ylabel("Rounds", fontweight='bold')
        ax5.set_title("Convergence Speed", fontweight='bold', fontsize=13)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Memory-Accuracy trade-off
        ax6 = fig.add_subplot(gs[2, :2])
        for i, name in enumerate(exp_names):
            ax6.scatter(memories[i], accs[i], s=300, color=colors[i], 
                       alpha=0.7, edgecolor='black', linewidth=2)
            ax6.text(memories[i], accs[i], name, fontsize=8, 
                    ha='center', va='center', fontweight='bold')
        ax6.set_xlabel("Peak Memory (MB)", fontweight='bold')
        ax6.set_ylabel("Final Accuracy", fontweight='bold')
        ax6.set_title("Memory-Accuracy Trade-off", fontweight='bold', fontsize=13)
        ax6.grid(True, alpha=0.3)
        
        # 7. Summary table
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        table_data = []
        for name in exp_names:
            exp = self.experiments[name]
            table_data.append([
                name[:15],  # Truncate long names
                f"{exp['final_accuracy']:.3f}",
                f"{exp['peak_memory_mb']:.0f}",
                f"{exp['convergence_round']}"
            ])
        
        table = ax7.table(cellText=table_data,
                         colLabels=['Experiment', 'Acc', 'Mem(MB)', 'Conv'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax7.set_title("Summary", fontweight='bold', fontsize=13, pad=20)
        
        fig.suptitle("ATLAS Experiment Dashboard", fontsize=18, fontweight='bold', y=0.98)
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def create_all_plots(self):
        """Generate all standard plots"""
        print("\n" + "=" * 70)
        print("[PLOTS] Generating Visualizations")
        print("=" * 70 + "\n")
        
        self.plot_convergence_curves(metric="accuracy", save_name="convergence_accuracy.png")
        self.plot_convergence_curves(metric="loss", save_name="convergence_loss.png")
        self.plot_comparison_bar(metric="final_accuracy", save_name="comparison_accuracy.png")
        self.plot_comparison_bar(metric="peak_memory_mb", save_name="comparison_memory.png")
        self.plot_comparison_bar(metric="total_comm_mb", save_name="comparison_communication.png")
        self.plot_memory_accuracy_tradeoff()
        self.plot_communication_comparison()
        self.plot_comprehensive_dashboard()
        
        print("\n[OK] All visualizations created!")
        print(f"[SAVE] Saved to: {self.save_dir}")


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("VISUALIZATION DEMO")
    print("=" * 70)
    
    # Create visualizer
    viz = ExperimentVisualizer(results_dir="./results", save_dir="./figures")
    
    # Load experiments
    viz.load_all_experiments()
    
    # Generate plots
    if viz.experiments:
        viz.create_all_plots()
    else:
        print("⚠️  No experiments found. Run experiments first!")
    
    print("\n✅ Demo complete!")

