"""
Multi-Seed Statistical Experiments for ATLAS

Runs multiple seeds per configuration for:
- Mean ± std dev computation
- Statistical significance tests (t-test, Wilcoxon)
- Publication-quality results

Features:
- 3-5 seeds per config
- Paired statistical tests
- JSON output with per-round/task metrics
- Automatic result aggregation

Usage:
    python experiments/run_statistical_experiments.py --seeds 3 --configs atlas fedavg_cluster local_only
"""

import json
import os
import sys
import argparse
import warnings
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')


class StatisticalExperimentRunner:
    """Run multiple seeds and compute statistical metrics"""
    
    def __init__(
        self,
        seeds: List[int],
        configs: List[str],
        base_config: Dict[str, Any],
        output_dir: str = 'results/statistical'
    ):
        """
        Args:
            seeds: List of random seeds to run
            configs: List of configurations ['atlas', 'fedavg_cluster', 'local_only']
            base_config: Base experimental parameters
            output_dir: Directory to save results
        """
        self.seeds = seeds
        self.configs = configs
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for all results
        self.all_results = {config: [] for config in configs}
        
    def run_all_experiments(self):
        """Run all configuration-seed combinations"""
        total_runs = len(self.configs) * len(self.seeds)
        current_run = 0
        
        print("=" * 80)
        print(f"STATISTICAL EXPERIMENTS: {len(self.configs)} configs × {len(self.seeds)} seeds = {total_runs} runs")
        print("=" * 80)
        
        for config in self.configs:
            print(f"\n[CONFIG] {config.upper()}")
            print("-" * 80)
            
            for seed in self.seeds:
                current_run += 1
                print(f"\n[RUN {current_run}/{total_runs}] Config={config}, Seed={seed}")
                
                # Run single experiment
                result = self._run_single_experiment(config, seed)
                
                if result is not None:
                    self.all_results[config].append(result)
                    print(f"[OK] Completed: Avg Acc={result['final_avg_accuracy']:.4f}")
                else:
                    print(f"[ERROR] Failed to run experiment")
        
        # Aggregate and analyze
        self._compute_statistics()
        self._perform_statistical_tests()
        
        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETE")
        print("=" * 80)
    
    def _run_single_experiment(self, config: str, seed: int) -> Dict[str, Any]:
        """
        Run single experiment with specific config and seed
        
        Returns:
            Dictionary with experiment results
        """
        # Build command
        cmd = [
            sys.executable,
            'experiments/atlas_integrated.py',
            '--mode', self.base_config['mode'],
            '--rounds', str(self.base_config['rounds']),
            '--ablation', config,
            '--model', self.base_config['model'],
            '--tasks', *self.base_config['tasks'],
            '--clients-per-task', str(self.base_config['clients_per_task']),
            '--samples', str(self.base_config['samples']),
            '--local-epochs', str(self.base_config['local_epochs']),
            '--seed', str(seed)  # Pass seed to experiment
        ]
        
        # Add eta if specified
        if 'eta' in self.base_config and config == 'atlas':
            cmd.extend(['--eta', str(self.base_config['eta'])])
        
        # Result file path (include seed to avoid overwriting other runs)
        result_file = f"results/atlas_integrated_{self.base_config['mode']}_{config}_seed{seed}.json"
        
        try:
            # Run experiment
            print(f"  Running: {' '.join(cmd[-10:])}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode != 0:
                print(f"  [ERROR] Process failed: {result.stderr[:200]}")
                return None
            
            # Load results
            if not Path(result_file).exists():
                print(f"  [ERROR] Result file not found: {result_file}")
                return None
            
            with open(result_file, 'r') as f:
                experiment_result = json.load(f)
            
            # Add metadata
            experiment_result['config'] = config
            experiment_result['seed'] = seed
            
            # Compute summary metrics
            experiment_result['final_avg_accuracy'] = np.mean(
                list(experiment_result.get('final_accuracies', {}).values())
            )
            experiment_result['final_std_accuracy'] = np.std(
                list(experiment_result.get('final_accuracies', {}).values())
            )
            
            # Total communication
            total_comm = 0
            for rm in experiment_result.get('round_metrics', []):
                up = rm.get('comm_upload_bytes', {})
                down = rm.get('comm_download_bytes', {})
                if isinstance(up, dict):
                    total_comm += sum(up.values()) + sum(down.values())
                else:
                    total_comm += up + down
            experiment_result['total_communication_mb'] = total_comm / (1024**2)
            
            # Save individual run
            individual_file = self.output_dir / f"{config}_seed{seed}.json"
            with open(individual_file, 'w') as f:
                json.dump(experiment_result, f, indent=2)
            
            return experiment_result
            
        except subprocess.TimeoutExpired:
            print(f"  [ERROR] Timeout after 2 hours")
            return None
        except Exception as e:
            print(f"  [ERROR] Exception: {str(e)}")
            return None
    
    def _compute_statistics(self):
        """Compute mean ± std for each configuration"""
        print("\n" + "=" * 80)
        print("STATISTICAL SUMMARY")
        print("=" * 80)
        
        summary = []
        
        for config in self.configs:
            results = self.all_results[config]
            
            if not results:
                print(f"\n[{config.upper()}] No results available")
                continue
            
            # Extract metrics
            final_accs = [r['final_avg_accuracy'] for r in results]
            final_stds = [r['final_std_accuracy'] for r in results]
            comm_costs = [r['total_communication_mb'] for r in results]
            
            # Compute statistics
            stats_dict = {
                'config': config,
                'n_seeds': len(results),
                'final_acc_mean': np.mean(final_accs),
                'final_acc_std': np.std(final_accs),
                'final_acc_min': np.min(final_accs),
                'final_acc_max': np.max(final_accs),
                'personalization_mean': np.mean(final_stds),
                'personalization_std': np.std(final_stds),
                'comm_mean_mb': np.mean(comm_costs),
                'comm_std_mb': np.std(comm_costs),
            }
            
            summary.append(stats_dict)
            
            # Print summary
            print(f"\n[{config.upper()}] ({len(results)} seeds)")
            print(f"  Final Accuracy:    {stats_dict['final_acc_mean']:.4f} ± {stats_dict['final_acc_std']:.4f}")
            print(f"  Accuracy Range:    [{stats_dict['final_acc_min']:.4f}, {stats_dict['final_acc_max']:.4f}]")
            print(f"  Personalization:   {stats_dict['personalization_mean']:.4f} ± {stats_dict['personalization_std']:.4f}")
            print(f"  Communication:     {stats_dict['comm_mean_mb']:.1f} ± {stats_dict['comm_std_mb']:.1f} MB")
        
        # Save summary
        summary_df = pd.DataFrame(summary)
        summary_file = self.output_dir / 'statistical_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n[SAVED] Summary: {summary_file}")
        
        # Also save as formatted table
        print("\n" + "=" * 80)
        print("LATEX TABLE (Copy-paste ready)")
        print("=" * 80)
        self._print_latex_table(summary_df)
    
    def _perform_statistical_tests(self):
        """Perform paired statistical tests between configurations"""
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 80)
        
        # Prepare test results
        test_results = []
        
        # Compare all pairs
        for i, config1 in enumerate(self.configs):
            for config2 in self.configs[i+1:]:
                results1 = self.all_results[config1]
                results2 = self.all_results[config2]
                
                if not results1 or not results2:
                    continue
                
                # Extract metrics
                acc1 = np.array([r['final_avg_accuracy'] for r in results1])
                acc2 = np.array([r['final_avg_accuracy'] for r in results2])
                comm1 = np.array([r['total_communication_mb'] for r in results1])
                comm2 = np.array([r['total_communication_mb'] for r in results2])
                
                # Paired t-test (parametric)
                if len(acc1) == len(acc2):
                    t_stat_acc, p_value_t_acc = stats.ttest_rel(acc1, acc2)
                    t_stat_comm, p_value_t_comm = stats.ttest_rel(comm1, comm2)
                else:
                    t_stat_acc, p_value_t_acc = stats.ttest_ind(acc1, acc2)
                    t_stat_comm, p_value_t_comm = stats.ttest_ind(comm1, comm2)
                
                # Wilcoxon signed-rank test (non-parametric)
                if len(acc1) == len(acc2) and len(acc1) > 1:
                    try:
                        w_stat_acc, p_value_w_acc = stats.wilcoxon(acc1, acc2)
                        w_stat_comm, p_value_w_comm = stats.wilcoxon(comm1, comm2)
                    except:
                        w_stat_acc, p_value_w_acc = np.nan, np.nan
                        w_stat_comm, p_value_w_comm = np.nan, np.nan
                else:
                    # Use Mann-Whitney U test for unpaired
                    w_stat_acc, p_value_w_acc = stats.mannwhitneyu(acc1, acc2)
                    w_stat_comm, p_value_w_comm = stats.mannwhitneyu(comm1, comm2)
                
                # Effect size (Cohen's d)
                cohen_d_acc = (np.mean(acc1) - np.mean(acc2)) / np.sqrt(
                    (np.std(acc1)**2 + np.std(acc2)**2) / 2
                )
                cohen_d_comm = (np.mean(comm1) - np.mean(comm2)) / np.sqrt(
                    (np.std(comm1)**2 + np.std(comm2)**2) / 2
                )
                
                test_result = {
                    'comparison': f"{config1} vs {config2}",
                    'metric': 'Final Accuracy',
                    't_statistic': t_stat_acc,
                    'p_value_ttest': p_value_t_acc,
                    'p_value_wilcoxon': p_value_w_acc,
                    'cohen_d': cohen_d_acc,
                    'significant': p_value_t_acc < 0.05,
                    'mean_diff': np.mean(acc1) - np.mean(acc2)
                }
                test_results.append(test_result)
                
                print(f"\n[{config1.upper()} vs {config2.upper()}]")
                print(f"  Accuracy:")
                print(f"    Δμ = {test_result['mean_diff']:+.4f}")
                print(f"    t-test:    p={p_value_t_acc:.4f} {'✓ SIGNIFICANT' if p_value_t_acc < 0.05 else '✗ n.s.'}")
                print(f"    Wilcoxon:  p={p_value_w_acc:.4f} {'✓ SIGNIFICANT' if p_value_w_acc < 0.05 else '✗ n.s.'}")
                print(f"    Cohen's d: {cohen_d_acc:.3f} ({self._interpret_effect_size(cohen_d_acc)})")
                
                print(f"  Communication:")
                print(f"    Δμ = {np.mean(comm1) - np.mean(comm2):+.1f} MB")
                print(f"    t-test:    p={p_value_t_comm:.4f} {'✓ SIGNIFICANT' if p_value_t_comm < 0.05 else '✗ n.s.'}")
                print(f"    Wilcoxon:  p={p_value_w_comm:.4f} {'✓ SIGNIFICANT' if p_value_w_comm < 0.05 else '✗ n.s.'}")
                print(f"    Cohen's d: {cohen_d_comm:.3f} ({self._interpret_effect_size(cohen_d_comm)})")
        
        # Save test results
        test_df = pd.DataFrame(test_results)
        test_file = self.output_dir / 'statistical_tests.csv'
        test_df.to_csv(test_file, index=False)
        
        print(f"\n[SAVED] Statistical tests: {test_file}")
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _print_latex_table(self, df: pd.DataFrame):
        """Print LaTeX-formatted table"""
        print("\\begin{table}[t]")
        print("\\centering")
        print("\\caption{Statistical Comparison of ATLAS Variants}")
        print("\\label{tab:statistical_results}")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("Method & Final Acc & Personalization & Comm (MB) & Seeds \\\\")
        print("\\midrule")
        
        for _, row in df.iterrows():
            config = row['config'].replace('_', ' ').title()
            print(f"{config} & "
                  f"${row['final_acc_mean']:.3f} \\pm {row['final_acc_std']:.3f}$ & "
                  f"${row['personalization_mean']:.3f} \\pm {row['personalization_std']:.3f}$ & "
                  f"${row['comm_mean_mb']:.1f} \\pm {row['comm_std_mb']:.1f}$ & "
                  f"{row['n_seeds']} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed statistical experiments")
    
    # Experiment configuration
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds (default: 3)')
    parser.add_argument('--configs', nargs='+', default=['atlas', 'fedavg_cluster', 'local_only'],
                       choices=['atlas', 'fedavg_cluster', 'local_only'],
                       help='Configurations to test')
    parser.add_argument('--mode', choices=['quick', 'full'], default='full',
                       help='Experiment mode (quick=validation, full=publication)')
    
    # Model & task configuration
    parser.add_argument('--model', type=str, default='distilbert-base-uncased',
                       help='Model to use')
    parser.add_argument('--tasks', nargs='+', default=['sst2', 'mrpc', 'cola'],
                       help='Tasks to include')
    parser.add_argument('--clients-per-task', type=int, default=3,
                       help='Clients per task')
    
    # Training configuration
    parser.add_argument('--rounds', type=int, default=15,
                       help='Number of training rounds')
    parser.add_argument('--samples', type=int, default=3000,
                       help='Max samples per client')
    parser.add_argument('--local-epochs', type=int, default=3,
                       help='Local epochs per round')
    parser.add_argument('--eta', type=float, default=0.1,
                       help='Laplacian regularization strength (for ATLAS)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results/statistical',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Generate seeds
    if args.seeds <= 0:
        print("[ERROR] Number of seeds must be positive")
        return
    
    seeds = [42, 123, 456, 789, 1024][:args.seeds]  # Use predetermined seeds for reproducibility
    
    print(f"[CONFIG] Seeds: {seeds}")
    print(f"[CONFIG] Configurations: {args.configs}")
    print(f"[CONFIG] Model: {args.model}")
    print(f"[CONFIG] Tasks: {args.tasks}")
    print(f"[CONFIG] Rounds: {args.rounds}")
    
    # Base configuration
    base_config = {
        'mode': args.mode,
        'rounds': args.rounds,
        'model': args.model,
        'tasks': args.tasks,
        'clients_per_task': args.clients_per_task,
        'samples': args.samples,
        'local_epochs': args.local_epochs,
        'eta': args.eta
    }
    
    # Run experiments
    runner = StatisticalExperimentRunner(
        seeds=seeds,
        configs=args.configs,
        base_config=base_config,
        output_dir=args.output_dir
    )
    
    runner.run_all_experiments()
    
    print("\n" + "=" * 80)
    print("✓ ALL STATISTICAL EXPERIMENTS COMPLETE")
    print(f"✓ Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
