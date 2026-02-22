"""
IEEE-Quality Results Tables for ATLAS Framework
Generates LaTeX and CSV tables for the paper.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from statistics import mean, stdev


class ResultsTableGenerator:
    """Generate publication-quality results tables."""
    
    def __init__(self, results_dir: str = "results", output_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_result(self, filename: str) -> Dict:
        """Load a single results JSON file."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found")
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def extract_final_metrics(self, result: Dict) -> Dict:
        """Extract final round metrics from a result."""
        if result is None:
            return None
        
        final_round = result['round_metrics'][-1]
        
        # Calculate statistics
        accuracies = list(final_round['test_accuracies'].values())
        f1_scores = list(final_round['test_f1'].values())
        
        # Communication stats
        total_upload = 0
        total_download = 0
        total_time = 0
        
        for round_data in result['round_metrics']:
            upload = sum(round_data.get('comm_upload_bytes', {}).values())
            download = sum(round_data.get('comm_download_bytes', {}).values())
            total_upload += upload
            total_download += download
            total_time += round_data['time_seconds']
        
        return {
            'avg_accuracy': final_round['avg_accuracy'] * 100,
            'std_accuracy': np.std(accuracies) * 100,
            'min_accuracy': min(accuracies) * 100,
            'max_accuracy': max(accuracies) * 100,
            'avg_f1': np.mean(f1_scores) * 100,
            'std_f1': np.std(f1_scores) * 100,
            'total_comm_mb': (total_upload + total_download) / (1024 * 1024),
            'total_time_min': total_time / 60,
            'num_rounds': len(result['round_metrics']),
        }
    
    def generate_main_results_table(self):
        """
        Table I: Main Performance Comparison
        Compare all methods on DistilBERT.
        """
        print("Generating Table I: Main Performance Comparison...")
        
        configs = [
            ('ATLAS (Full)', 'atlas_distilbert-base-uncased_atlas_seed42_r10.json'),
            ('ATLAS (No Laplacian)', 'atlas_distilbert-base-uncased_atlas_no_laplacian_seed42_r10.json'),
            ('FedAvg (Clustered)', 'atlas_distilbert-base-uncased_fedavg_cluster_seed42_r10.json'),
            ('Standard FL', 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json'),
        ]
        
        data = []
        
        for method, filename in configs:
            result = self.load_result(filename)
            metrics = self.extract_final_metrics(result)
            
            if metrics:
                data.append({
                    'Method': method,
                    'Avg. Accuracy (%)': f"{metrics['avg_accuracy']:.2f} ± {metrics['std_accuracy']:.2f}",
                    'Avg. F1 (%)': f"{metrics['avg_f1']:.2f} ± {metrics['std_f1']:.2f}",
                    'Min Acc. (%)': f"{metrics['min_accuracy']:.2f}",
                    'Max Acc. (%)': f"{metrics['max_accuracy']:.2f}",
                    'Comm. (MB)': f"{metrics['total_comm_mb']:.1f}",
                    'Time (min)': f"{metrics['total_time_min']:.1f}",
                })
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_file = self.output_dir / 'table1_main_results.csv'
        df.to_csv(csv_file, index=False)
        print(f"  Saved CSV: {csv_file}")
        
        # Generate LaTeX
        latex = self._generate_latex_table(
            df,
            caption="Performance Comparison of ATLAS Variants on DistilBERT",
            label="tab:main_results",
            column_format="l|cc|cc|cc"
        )
        
        latex_file = self.output_dir / 'table1_main_results.tex'
        with open(latex_file, 'w') as f:
            f.write(latex)
        print(f"  Saved LaTeX: {latex_file}")
        
        return df
    
    def generate_cross_model_table(self):
        """
        Table II: Cross-Model Performance Comparison
        Compare ATLAS performance across different models.
        """
        print("Generating Table II: Cross-Model Performance...")
        
        model_configs = [
            ('DistilBERT', [
                ('ATLAS', 'atlas_distilbert-base-uncased_atlas_seed42_r10.json'),
                ('Standard FL', 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json'),
            ]),
            ('GPT-2', [
                ('ATLAS', [
                    'atlas_gpt2_atlas_seed42_r10.json',
                    'atlas_gpt2_atlas_seed123_r10.json',
                    'atlas_gpt2_atlas_seed456_r10.json',
                ]),
                ('Standard FL', [
                    'atlas_gpt2_standard_fl_seed42_r10.json',
                    'atlas_gpt2_standard_fl_seed123_r10.json',
                    'atlas_gpt2_standard_fl_seed456_r10.json',
                ]),
            ]),
            ('Qwen-0.5B', [
                ('ATLAS', [
                    'atlas_Qwen_Qwen2.5-0.5B_atlas_seed42_r10.json',
                    'atlas_Qwen_Qwen2.5-0.5B_atlas_seed123_r10.json',
                ]),
                ('Standard FL', [
                    'atlas_Qwen_Qwen2.5-0.5B_standard_fl_seed42_r10.json',
                    'atlas_Qwen_Qwen2.5-0.5B_standard_fl_seed123_r10.json',
                ]),
            ]),
        ]
        
        data = []
        
        for model_name, methods in model_configs:
            for method_name, files in methods:
                # Handle multiple seeds
                if isinstance(files, list):
                    metrics_list = []
                    for filename in files:
                        result = self.load_result(filename)
                        metrics = self.extract_final_metrics(result)
                        if metrics:
                            metrics_list.append(metrics)
                    
                    if metrics_list:
                        avg_acc = mean([m['avg_accuracy'] for m in metrics_list])
                        std_acc = stdev([m['avg_accuracy'] for m in metrics_list]) if len(metrics_list) > 1 else 0
                        avg_f1 = mean([m['avg_f1'] for m in metrics_list])
                        std_f1 = stdev([m['avg_f1'] for m in metrics_list]) if len(metrics_list) > 1 else 0
                        avg_comm = mean([m['total_comm_mb'] for m in metrics_list])
                        avg_time = mean([m['total_time_min'] for m in metrics_list])
                        
                        data.append({
                            'Model': model_name,
                            'Method': method_name,
                            'Accuracy (%)': f"{avg_acc:.2f} ± {std_acc:.2f}",
                            'F1 Score (%)': f"{avg_f1:.2f} ± {std_f1:.2f}",
                            'Comm. (MB)': f"{avg_comm:.1f}",
                            'Time (min)': f"{avg_time:.1f}",
                        })
                else:
                    result = self.load_result(files)
                    metrics = self.extract_final_metrics(result)
                    
                    if metrics:
                        data.append({
                            'Model': model_name,
                            'Method': method_name,
                            'Accuracy (%)': f"{metrics['avg_accuracy']:.2f}",
                            'F1 Score (%)': f"{metrics['avg_f1']:.2f}",
                            'Comm. (MB)': f"{metrics['total_comm_mb']:.1f}",
                            'Time (min)': f"{metrics['total_time_min']:.1f}",
                        })
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_file = self.output_dir / 'table2_cross_model.csv'
        df.to_csv(csv_file, index=False)
        print(f"  Saved CSV: {csv_file}")
        
        # Generate LaTeX
        latex = self._generate_latex_table(
            df,
            caption="Cross-Model Performance Comparison: ATLAS vs. Standard FL",
            label="tab:cross_model",
            column_format="l|l|cc|cc"
        )
        
        latex_file = self.output_dir / 'table2_cross_model.tex'
        with open(latex_file, 'w') as f:
            f.write(latex)
        print(f"  Saved LaTeX: {latex_file}")
        
        return df
    
    def generate_ablation_table(self):
        """
        Table III: Ablation Study Results
        Detailed ablation analysis showing contribution of each component.
        """
        print("Generating Table III: Ablation Study...")
        
        configs = [
            ('ATLAS (Full)', 'atlas_distilbert-base-uncased_atlas_seed42_r10.json', 'Full system'),
            ('w/o Laplacian', 'atlas_distilbert-base-uncased_atlas_no_laplacian_seed42_r10.json', 'Without graph regularization'),
            ('w/o Split Training', 'atlas_distilbert-base-uncased_fedavg_cluster_seed42_r10.json', 'Without parameter splitting'),
            ('w/o Clustering', 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json', 'Standard federated learning'),
        ]
        
        data = []
        baseline_acc = None
        baseline_f1 = None
        
        for method, filename, description in configs:
            result = self.load_result(filename)
            metrics = self.extract_final_metrics(result)
            
            if metrics:
                if baseline_acc is None:  # First entry is ATLAS (Full)
                    baseline_acc = metrics['avg_accuracy']
                    baseline_f1 = metrics['avg_f1']
                    delta_acc = "—"
                    delta_f1 = "—"
                else:
                    delta_acc = f"{metrics['avg_accuracy'] - baseline_acc:+.2f}"
                    delta_f1 = f"{metrics['avg_f1'] - baseline_f1:+.2f}"
                
                data.append({
                    'Configuration': method,
                    'Description': description,
                    'Accuracy (%)': f"{metrics['avg_accuracy']:.2f}",
                    'Δ Acc.': delta_acc,
                    'F1 Score (%)': f"{metrics['avg_f1']:.2f}",
                    'Δ F1': delta_f1,
                    'Comm. Efficiency': f"{metrics['avg_accuracy'] / metrics['total_comm_mb']:.3f}",
                })
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_file = self.output_dir / 'table3_ablation.csv'
        df.to_csv(csv_file, index=False)
        print(f"  Saved CSV: {csv_file}")
        
        # Generate LaTeX
        latex = self._generate_latex_table(
            df,
            caption="Ablation Study: Impact of ATLAS Components on Performance",
            label="tab:ablation",
            column_format="l|p{4cm}|cc|cc|c",
            small_font=True
        )
        
        latex_file = self.output_dir / 'table3_ablation.tex'
        with open(latex_file, 'w') as f:
            f.write(latex)
        print(f"  Saved LaTeX: {latex_file}")
        
        return df
    
    def generate_communication_table(self):
        """
        Table IV: Communication Efficiency Comparison
        Detailed communication cost breakdown.
        """
        print("Generating Table IV: Communication Efficiency...")
        
        configs = [
            ('ATLAS', 'atlas_distilbert-base-uncased_atlas_seed42_r10.json'),
            ('ATLAS (No Lap)', 'atlas_distilbert-base-uncased_atlas_no_laplacian_seed42_r10.json'),
            ('FedAvg (Clustered)', 'atlas_distilbert-base-uncased_fedavg_cluster_seed42_r10.json'),
            ('Standard FL', 'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json'),
        ]
        
        data = []
        
        for method, filename in configs:
            result = self.load_result(filename)
            if result is None:
                continue
            
            metrics = self.extract_final_metrics(result)
            
            # Calculate per-round averages
            total_upload = 0
            total_download = 0
            
            for round_data in result['round_metrics']:
                upload = sum(round_data.get('comm_upload_bytes', {}).values())
                download = sum(round_data.get('comm_download_bytes', {}).values())
                total_upload += upload
                total_download += download
            
            num_rounds = len(result['round_metrics'])
            avg_per_round_mb = (total_upload + total_download) / num_rounds / (1024 * 1024)
            
            # Calculate efficiency metrics
            final_acc = metrics['avg_accuracy']
            comm_efficiency = final_acc / metrics['total_comm_mb']
            time_efficiency = final_acc / metrics['total_time_min']
            
            data.append({
                'Method': method,
                'Total Comm. (MB)': f"{metrics['total_comm_mb']:.1f}",
                'Per-Round (MB)': f"{avg_per_round_mb:.1f}",
                'Upload (MB)': f"{total_upload / (1024 * 1024):.1f}",
                'Download (MB)': f"{total_download / (1024 * 1024):.1f}",
                'Acc./MB': f"{comm_efficiency:.3f}",
                'Acc./min': f"{time_efficiency:.2f}",
            })
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_file = self.output_dir / 'table4_communication.csv'
        df.to_csv(csv_file, index=False)
        print(f"  Saved CSV: {csv_file}")
        
        # Generate LaTeX
        latex = self._generate_latex_table(
            df,
            caption="Communication Efficiency Analysis: Data Transfer and Computational Cost",
            label="tab:communication",
            column_format="l|cc|cc|cc"
        )
        
        latex_file = self.output_dir / 'table4_communication.tex'
        with open(latex_file, 'w') as f:
            f.write(latex)
        print(f"  Saved LaTeX: {latex_file}")
        
        return df

    
    def generate_statistical_summary_table(self):
        """
        Table VI: Statistical Summary of All Experiments
        High-level overview with statistical significance.
        """
        print("Generating Table VI: Statistical Summary...")
        
        # Aggregate results by category
        categories = {
            'ATLAS Methods': [
                'atlas_distilbert-base-uncased_atlas_seed42_r10.json',
                'atlas_gpt2_atlas_seed42_r10.json',
                'atlas_gpt2_atlas_seed123_r10.json',
                'atlas_gpt2_atlas_seed456_r10.json',
                'atlas_Qwen_Qwen2.5-0.5B_atlas_seed42_r10.json',
                'atlas_Qwen_Qwen2.5-0.5B_atlas_seed123_r10.json',
            ],
            'Standard FL': [
                'atlas_distilbert-base-uncased_standard_fl_seed42_r10.json',
                'atlas_gpt2_standard_fl_seed42_r10.json',
                'atlas_gpt2_standard_fl_seed123_r10.json',
                'atlas_gpt2_standard_fl_seed456_r10.json',
                'atlas_Qwen_Qwen2.5-0.5B_standard_fl_seed42_r10.json',
                'atlas_Qwen_Qwen2.5-0.5B_standard_fl_seed123_r10.json',
            ],
        }
        
        data = []
        
        for category, files in categories.items():
            metrics_list = []
            
            for filename in files:
                result = self.load_result(filename)
                metrics = self.extract_final_metrics(result)
                if metrics:
                    metrics_list.append(metrics)
            
            if metrics_list:
                accs = [m['avg_accuracy'] for m in metrics_list]
                f1s = [m['avg_f1'] for m in metrics_list]
                comms = [m['total_comm_mb'] for m in metrics_list]
                times = [m['total_time_min'] for m in metrics_list]
                
                data.append({
                    'Category': category,
                    'Experiments': len(metrics_list),
                    'Avg. Acc. (%)': f"{mean(accs):.2f} ± {stdev(accs):.2f}" if len(accs) > 1 else f"{accs[0]:.2f}",
                    'Avg. F1 (%)': f"{mean(f1s):.2f} ± {stdev(f1s):.2f}" if len(f1s) > 1 else f"{f1s[0]:.2f}",
                    'Avg. Comm. (MB)': f"{mean(comms):.1f}",
                    'Avg. Time (min)': f"{mean(times):.1f}",
                })
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_file = self.output_dir / 'table6_statistical_summary.csv'
        df.to_csv(csv_file, index=False)
        print(f"  Saved CSV: {csv_file}")
        
        # Generate LaTeX
        latex = self._generate_latex_table(
            df,
            caption="Statistical Summary Across All Experiments",
            label="tab:statistical_summary",
            column_format="l|c|cc|cc"
        )
        
        latex_file = self.output_dir / 'table6_statistical_summary.tex'
        with open(latex_file, 'w') as f:
            f.write(latex)
        print(f"  Saved LaTeX: {latex_file}")
        
        return df
    
    def _generate_latex_table(self, df: pd.DataFrame, caption: str, label: str, 
                              column_format: str, small_font: bool = False) -> str:
        """Generate IEEE-style LaTeX table."""
        
        # Start table
        latex = "\\begin{table}[!t]\n"
        latex += "\\renewcommand{\\arraystretch}{1.2}\n"
        latex += "\\caption{" + caption + "}\n"
        latex += "\\label{" + label + "}\n"
        latex += "\\centering\n"
        
        if small_font:
            latex += "\\small\n"
        
        # Begin tabular
        latex += "\\begin{tabular}{" + column_format + "}\n"
        latex += "\\hline\\hline\n"
        
        # Header
        header = " & ".join([col.replace('%', '\\%').replace('_', '\\_') for col in df.columns])
        latex += header + " \\\\\n"
        latex += "\\hline\n"
        
        # Data rows
        for _, row in df.iterrows():
            row_str = " & ".join([str(val).replace('%', '\\%').replace('_', '\\_') for val in row])
            latex += row_str + " \\\\\n"
        
        # End table
        latex += "\\hline\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def generate_all_tables(self):
        """Generate all results tables."""
        print("\n" + "="*60)
        print("Generating IEEE-Quality Results Tables")
        print("="*60 + "\n")
        
        self.generate_main_results_table()
        self.generate_cross_model_table()
        self.generate_ablation_table()
        self.generate_communication_table()
        self.generate_statistical_summary_table()
        
        print("\n" + "="*60)
        print("All tables generated successfully!")
        print(f"Output directory: {self.output_dir.absolute()}")
        print("="*60 + "\n")


def main():
    """Main function to generate all tables."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate IEEE-quality results tables for ATLAS')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing results JSON files')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save output tables')
    
    args = parser.parse_args()
    
    generator = ResultsTableGenerator(results_dir=args.results_dir, output_dir=args.output_dir)
    generator.generate_all_tables()


if __name__ == '__main__':
    main()
