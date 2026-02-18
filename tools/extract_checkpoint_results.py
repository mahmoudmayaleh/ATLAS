#!/usr/bin/env python3
"""
Extract results from ATLAS checkpoint files and save as JSON.

Usage:
    # Single checkpoint
    python tools/extract_checkpoint_results.py checkpoints/atlas_atlas_seed42_round_10.pkl
    
    # All checkpoints with seed pattern
    python tools/extract_checkpoint_results.py checkpoints/atlas_*_seed*_round_10.pkl
    
    # Specify model name (default: gpt2-xl)
    python tools/extract_checkpoint_results.py checkpoints/*.pkl --model gpt2-xl

This will create JSON files in the results/ directory with model name and seed info.
"""

import sys
import pickle
import json
import numpy as np
import re
import glob
from pathlib import Path


def to_jsonable(obj):
    """Convert numpy/torch objects to JSON-serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def extract_info_from_filename(checkpoint_path: str):
    """Extract method, seed, and round from checkpoint filename"""
    filename = Path(checkpoint_path).stem
    
    # Pattern: atlas_<method>_seed<num>_round_<num>
    method_match = re.search(r'atlas_([a-z_]+)_seed', filename)
    seed_match = re.search(r'seed(\d+)', filename)
    round_match = re.search(r'round_(\d+)', filename)
    
    method = method_match.group(1) if method_match else 'atlas'
    seed = int(seed_match.group(1)) if seed_match else None
    round_num = int(round_match.group(1)) if round_match else None
    
    return method, seed, round_num


def extract_results_from_checkpoint(checkpoint_path: str, output_path: str = None, model_name: str = "gpt2-xl"):
    """Extract results from checkpoint and save as JSON"""
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"✓ Checkpoint loaded successfully")
    print(f"  Round: {checkpoint.get('round', 'unknown')}")
    
    # Extract method, seed, round from filename
    method, seed, round_num = extract_info_from_filename(checkpoint_path)
    
    # Extract results
    results = checkpoint.get('results', {})
    
    if not results:
        print("\n⚠️  Warning: No 'results' key found in checkpoint!")
        print("Available keys:", list(checkpoint.keys()))
        
        # Try to reconstruct basic results from checkpoint
        results = {
            'round': checkpoint.get('round'),
            'cluster_labels': checkpoint.get('cluster_labels'),
            'clustering_metrics': checkpoint.get('clustering_metrics'),
            'fingerprints': checkpoint.get('fingerprints'),
        }
        print("Reconstructed partial results from checkpoint metadata")
    
    # Add checkpoint metadata to results
    results_json = {
        'model_name': model_name,
        'method': method,
        'seed': seed,
        'checkpoint_round': checkpoint.get('round'),
        'round_metrics': to_jsonable(results.get('round_metrics', [])),
        'final_accuracies': to_jsonable(results.get('final_accuracies', {})),
        'cluster_labels': to_jsonable(results.get('cluster_labels', checkpoint.get('cluster_labels', {}))),
        'clustering_metrics': to_jsonable(results.get('clustering_metrics', checkpoint.get('clustering_metrics', {}))),
        'phase1_clustering': to_jsonable(results.get('phase1_clustering', {})),
        'phase2_rank_allocation': to_jsonable(results.get('phase2_rank_allocation', [])),
        'communication_costs': to_jsonable(results.get('communication_costs', {})),
        'time_metrics': to_jsonable(results.get('time_metrics', {})),
        'fingerprints': to_jsonable(results.get('fingerprints', checkpoint.get('fingerprints', {}))),
        'device_configs': to_jsonable(results.get('device_configs', {})),
        'layer_importances': to_jsonable(results.get('layer_importances', {})),
    }
    
    # Determine output path
    if output_path is None:
        # Create filename with model name and seed: atlas_integrated_quick_<method>_<model>_seed<num>.json
        model_safe = model_name.replace('-', '_').replace('/', '_')
        output_filename = f"atlas_integrated_quick_{method}_{model_safe}_seed{seed}.json"
        output_path = Path("./results") / output_filename
    else:
        output_path = Path(output_path)
    
    # Create results directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary
    num_rounds = len(results_json.get('round_metrics', []))
    final_acc = results_json.get('final_accuracies', {})
    
    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Method: {method}")
    print(f"  Seed: {seed}")
    print(f"  Total rounds completed: {num_rounds}")
    print(f"  Checkpoint round: {results_json['checkpoint_round']}")
    
    if final_acc:
        print(f"\n  Final accuracies:")
        for client_id, acc in sorted(final_acc.items()):
            print(f"    Client {client_id}: {acc:.4f}")
        print(f"    Average: {np.mean(list(final_acc.values())):.4f}")
    
    if results_json.get('round_metrics'):
        print(f"\n  Round-by-round progress:")
        for i, metric in enumerate(results_json['round_metrics'][-5:], start=max(0, num_rounds-5)+1):
            avg_acc = metric.get('avg_accuracy', 0)
            time_s = metric.get('time_seconds', 0)
            print(f"    Round {i}: avg_acc={avg_acc:.4f}, time={time_s:.1f}s")
    
    print(f"{'='*60}\n")
    
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/extract_checkpoint_results.py <checkpoint_path> [--model MODEL_NAME]")
        print("\nExamples:")
        print("  # Single checkpoint")
        print("  python tools/extract_checkpoint_results.py checkpoints/atlas_atlas_seed42_round_10.pkl")
        print("")
        print("  # With model name")
        print("  python tools/extract_checkpoint_results.py checkpoints/atlas_atlas_seed42_round_10.pkl --model gpt2-xl")
        print("")
        print("  # All checkpoints matching pattern (use quotes)")
        print('  python tools/extract_checkpoint_results.py "checkpoints/*_seed*_round_10.pkl" --model gpt2-xl')
        sys.exit(1)
    
    # Parse arguments
    checkpoint_pattern = sys.argv[1]
    model_name = "gpt2-xl"  # Default
    
    # Check for --model flag
    if "--model" in sys.argv:
        model_idx = sys.argv.index("--model")
        if model_idx + 1 < len(sys.argv):
            model_name = sys.argv[model_idx + 1]
    
    # Expand glob pattern
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"Error: No checkpoint files found matching: {checkpoint_pattern}")
        sys.exit(1)
    
    print(f"Found {len(checkpoint_files)} checkpoint file(s)")
    print(f"Model name: {model_name}")
    print("="*70)
    
    # Process each checkpoint
    success_count = 0
    error_count = 0
    
    for checkpoint_path in sorted(checkpoint_files):
        try:
            extract_results_from_checkpoint(checkpoint_path, output_path=None, model_name=model_name)
            success_count += 1
        except Exception as e:
            print(f"\n❌ Error extracting {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            print()
    
    # Final summary
    print("\n" + "="*70)
    print(f"Extraction complete!")
    print(f"  ✓ Success: {success_count}")
    if error_count > 0:
        print(f"  ✗ Failed: {error_count}")
    print("="*70)
