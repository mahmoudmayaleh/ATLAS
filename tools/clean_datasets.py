"""
Clean GLUE datasets: Remove duplicates and train/val overlaps
Saves cleaned datasets to tools/cleaned_data/ for use in experiments
"""
import hashlib
from pathlib import Path
from datasets import load_dataset, Dataset
import sys

def text_hash(example, text_col, text_col2=None):
    """Generate hash from text content"""
    if text_col2:
        s = (example[text_col] or "") + " ||| " + (example[text_col2] or "")
    else:
        s = example[text_col] or ""
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def clean_task(task_name, dataset_name, text_col, text_col2=None):
    """Remove duplicates and overlaps for one task"""
    print(f"\n{'='*60}")
    print(f"Cleaning {task_name}")
    print(f"{'='*60}")
    
    # Load datasets
    if task_name == 'sst2':
        train = load_dataset(dataset_name, split='train')
        val = load_dataset(dataset_name, split='validation')
    else:
        train = load_dataset(dataset_name, task_name, split='train')
        val = load_dataset(dataset_name, task_name, split='validation')
    
    print(f"Original sizes: train={len(train)}, val={len(val)}")
    
    # Build unique indices for train
    train_hash_to_idx = {}
    unique_train_idxs = []
    for i, ex in enumerate(train):
        h = text_hash(ex, text_col, text_col2)
        if h not in train_hash_to_idx:
            train_hash_to_idx[h] = i
            unique_train_idxs.append(i)
    
    # Build unique indices for validation
    val_hash_to_idx = {}
    unique_val_idxs = []
    for i, ex in enumerate(val):
        h = text_hash(ex, text_col, text_col2)
        if h not in val_hash_to_idx:
            val_hash_to_idx[h] = i
            unique_val_idxs.append(i)
    
    # Find overlaps
    overlap_hashes = set(train_hash_to_idx.keys()) & set(val_hash_to_idx.keys())
    
    # Remove overlapping examples from train
    if overlap_hashes:
        print(f"Found {len(overlap_hashes)} train↔val overlaps - removing from train")
        remove_idxs = {train_hash_to_idx[h] for h in overlap_hashes}
        unique_train_idxs = [i for i in unique_train_idxs if i not in remove_idxs]
    
    # Apply cleaning
    train_before = len(train)
    val_before = len(val)
    
    if len(unique_train_idxs) != train_before:
        train = train.select(unique_train_idxs)
    if len(unique_val_idxs) != val_before:
        val = val.select(unique_val_idxs)
    
    print(f"Cleaned sizes: train={len(train)}, val={len(val)}")
    print(f"  Removed from train: {train_before - len(train)} ({len(overlap_hashes)} overlaps + {train_before - len(train) - len(overlap_hashes)} duplicates)")
    print(f"  Removed from val: {val_before - len(val)} (duplicates)")
    
    # Save to disk
    output_dir = Path('tools/cleaned_data') / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train.save_to_disk(str(output_dir / 'train'))
    val.save_to_disk(str(output_dir / 'validation'))
    
    print(f"✓ Saved to {output_dir}")
    
    return len(train), len(val)

def main():
    print("\nGLUE Dataset Cleaner")
    print("Removes duplicates and train/val overlaps")
    
    tasks = [
        ('sst2', 'stanfordnlp/sst2', 'sentence', None),
        ('mrpc', 'nyu-mll/glue', 'sentence1', 'sentence2'),
        ('cola', 'nyu-mll/glue', 'sentence', None),
    ]
    
    results = {}
    for task_name, dataset_name, col1, col2 in tasks:
        try:
            train_size, val_size = clean_task(task_name, dataset_name, col1, col2)
            results[task_name] = {'train': train_size, 'val': val_size}
        except Exception as e:
            print(f"ERROR cleaning {task_name}: {e}")
            results[task_name] = {'error': str(e)}
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for task, sizes in results.items():
        if 'error' in sizes:
            print(f"{task}: ERROR - {sizes['error']}")
        else:
            print(f"{task}: train={sizes['train']}, val={sizes['val']}")
    
    print(f"\nCleaned datasets saved to: tools/cleaned_data/")
    print("\nNext steps:")
    print("1. Run QuickLeakCheck again to verify:")
    print("   python tools/quick_leak_check.py")
    print("2. Update atlas_integrated.py to load from cleaned_data/")
    print("3. Start fresh training")

if __name__ == '__main__':
    main()
