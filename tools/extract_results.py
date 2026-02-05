"""Extract REAL metrics from ATLAS training - NO ESTIMATES."""

import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['savefig.dpi'] = 300


def parse_training_output(file_path: str):
    """Parse actual training output."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    results = {'rounds': [], 'per_client': [], 'phase2': {}}
    
    # REAL rank allocation
    rank_pattern = r'Client (\d+) \((\w+), cluster (\d+)\): ranks=\[([\d, ]+)\]'
    results['phase2']['rank_allocation'] = []
    for match in re.finditer(rank_pattern, content):
        client_id = int(match.group(1))
        device = match.group(2)
        ranks = [int(x.strip()) for x in match.group(4).split(',')]
        lora_params = sum(r * 768 * 2 for r in ranks)  # ACTUAL calculation
        
        results['phase2']['rank_allocation'].append({
            'client_id': client_id,
            'device': device,
            'ranks': ranks,
            'lora_params': lora_params,
            'params_per_round': lora_params * 2
        })
    
    # REAL round metrics
    round_pattern = r'ROUND (\d+)/\d+.*?Avg accuracy: ([\d.]+), Time: ([\d.]+)s'
    for match in re.finditer(round_pattern, content, re.DOTALL):
        round_num = int(match.group(1))
        avg_acc = float(match.group(2))
        time_s = float(match.group(3))
        
        round_block = match.group(0)
        eval_pattern = r'Client (\d+) \((\w+)\): acc=([\d.]+), f1=([\d.]+)'
        
        client_metrics = []
        for eval_match in re.finditer(eval_pattern, round_block):
            client_metrics.append({
                'round': round_num,
                'client_id': int(eval_match.group(1)),
                'task': eval_match.group(2),
                'accuracy': float(eval_match.group(3)),
                'f1': float(eval_match.group(4))
            })
        
        results['per_client'].extend(client_metrics)
        
        # Task averages
        task_metrics = {}
        for cm in client_metrics:
            task = cm['task']
            if task not in task_metrics:
                task_metrics[task] = []
            task_metrics[task].append(cm['accuracy'])
        
        task_avg = {task: sum(accs)/len(accs) for task, accs in task_metrics.items()}
        
        results['rounds'].append({
            'round': round_num,
            'avg_accuracy': avg_acc,
            'time_seconds': time_s,
            'task_metrics': task_avg
        })
    
    return results


def create_plots(results, output_dir="results"):
    """Generate plots from REAL data."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Learning curve
    rounds = [r['round'] for r in results['rounds']]
    accs = [r['avg_accuracy'] for r in results['rounds']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accs, 'b-o', linewidth=2)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('ATLAS Learning Curve (REAL DATA)', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curve.png')
    plt.close()
    print(f"✓ {output_dir}/learning_curve.png")
    
    # Per-task
    plt.figure(figsize=(12, 6))
    tasks = ['sst2', 'mrpc', 'cola']
    colors = ['blue', 'green', 'red']
    for task, color in zip(tasks, colors):
        task_accs = [r['task_metrics'].get(task, 0) for r in results['rounds']]
        plt.plot(rounds, task_accs, marker='o', label=task.upper(), color=color, linewidth=2)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Task Accuracy (REAL DATA)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_task.png')
    plt.close()
    print(f"✓ {output_dir}/per_task.png")
    
    # ACTUAL communication
    if 'rank_allocation' in results['phase2']:
        comm_df = pd.DataFrame(results['phase2']['rank_allocation'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        device_params = comm_df.groupby('device')['lora_params'].mean() / 1000
        axes[0].bar(device_params.index, device_params.values, color='steelblue')
        axes[0].set_ylabel('LoRA Params (×1000)', fontsize=12)
        axes[0].set_title('ACTUAL LoRA Parameters', fontsize=13, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=15)
        
        axes[1].bar(comm_df['client_id'], comm_df['params_per_round'] / 1e6, color='orange')
        axes[1].set_xlabel('Client ID', fontsize=12)
        axes[1].set_ylabel('Params/Round (M)', fontsize=12)
        axes[1].set_title('ACTUAL Communication per Client', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/actual_communication.png')
        plt.close()
        print(f"✓ {output_dir}/actual_communication.png")


def generate_tables(results, output_dir="results"):
    """Generate CSV tables."""
    # Round summary
    round_data = []
    for r in results['rounds']:
        row = {'round': r['round'], 'avg_acc': r['avg_accuracy'], 'time_s': r['time_seconds']}
        for task, acc in r['task_metrics'].items():
            row[f'{task}_acc'] = acc
        round_data.append(row)
    
    df_rounds = pd.DataFrame(round_data)
    df_rounds.to_csv(f'{output_dir}/rounds_real.csv', index=False)
    print(f"✓ {output_dir}/rounds_real.csv")
    
    # Communication (ACTUAL)
    if 'rank_allocation' in results['phase2']:
        df_comm = pd.DataFrame(results['phase2']['rank_allocation'])
        df_comm.to_csv(f'{output_dir}/communication_real.csv', index=False)
        print(f"✓ {output_dir}/communication_real.csv")
    
    # Final performance
    final_clients = [c for c in results['per_client'] if c['round'] == max(r['round'] for r in results['rounds'])]
    df_final = pd.DataFrame(final_clients)
    df_final.to_csv(f'{output_dir}/final_performance_real.csv', index=False)
    print(f"✓ {output_dir}/final_performance_real.csv")


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("ATLAS RESULTS SUMMARY (REAL DATA ONLY)")
    print("="*70)
    
    rounds_data = results['rounds']
    print(f"\nRounds: {len(rounds_data)}")
    print(f"Initial accuracy: {rounds_data[0]['avg_accuracy']:.4f}")
    print(f"Final accuracy: {rounds_data[-1]['avg_accuracy']:.4f}")
    print(f"Improvement: {(rounds_data[-1]['avg_accuracy'] - rounds_data[0]['avg_accuracy']):.4f}")
    print(f"Total time: {sum(r['time_seconds'] for r in rounds_data)/60:.1f} min")
    
    print(f"\nFinal per-task:")
    for task, acc in rounds_data[-1]['task_metrics'].items():
        print(f"  {task.upper()}: {acc:.4f}")
    
    if 'rank_allocation' in results['phase2']:
        comm_df = pd.DataFrame(results['phase2']['rank_allocation'])
        print(f"\nACTUAL Communication (15 rounds):")
        total_15rounds = comm_df['params_per_round'].sum() * 15
        print(f"  Total: {total_15rounds/1e6:.1f}M params")
        print(f"  Per client per round: {comm_df['params_per_round'].mean()/1e6:.2f}M params")
    
    print("\n" + "="*70)


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_results.py <training_output.txt>")
        return
    
    input_file = sys.argv[1]
    output_dir = "results"
    
    print(f"\n[Parsing] {input_file}")
    results = parse_training_output(input_file)
    
    print(f"\n[Creating plots] -> {output_dir}/")
    create_plots(results, output_dir)
    
    print(f"\n[Generating tables] -> {output_dir}/")
    generate_tables(results, output_dir)
    
    print_summary(results)
    
    # Save JSON
    json_path = f'{output_dir}/results_real.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ {json_path}")
    
    print(f"\n✅ All REAL data extracted to {output_dir}/")


if __name__ == "__main__":
    main()
