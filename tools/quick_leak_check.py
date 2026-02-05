"""
Quick Leak Check: Detect train/val overlaps and duplicates in GLUE datasets
Run from repository root: python tools/quick_leak_check.py
"""
import hashlib
from datasets import load_dataset

dataset_map = {
    'sst2': ('stanfordnlp/sst2', 'sentence', None),
    'mrpc': ('nyu-mll/glue', 'sentence1', 'sentence2'),
    'cola': ('nyu-mll/glue', 'sentence', None),
}

def text_hash(example, text_col, text_col2=None):
    if text_col2:
        s = (example[text_col] or "") + " ||| " + (example[text_col2] or "")
    else:
        s = example[text_col] or ""
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def check_task(task):
    dataset_name, c1, c2 = dataset_map[task]
    # loading same splits as atlas_integrated.py
    if task == 'sst2':
        train = load_dataset(dataset_name, split='train')
        val = load_dataset(dataset_name, split='validation')
    else:
        train = load_dataset(dataset_name, task, split='train')
        val = load_dataset(dataset_name, task, split='validation')

    print(f"\n== {task} ==")
    print(f"train size: {len(train)}, val size: {len(val)}")

    h_train = set(text_hash(ex, c1, c2) for ex in train)
    h_val = set(text_hash(ex, c1, c2) for ex in val)

    inter = h_train & h_val
    print(f"overlap train∩val (by text-hash): {len(inter)}")
    if inter:
        print(" sample overlapping hash -> show one example text:")
        h = next(iter(inter))
        # find and print example (first match)
        for ex in train:
            if text_hash(ex,c1,c2)==h:
                print(" train example:", (ex.get(c1), ex.get(c2)))
                break
        for ex in val:
            if text_hash(ex,c1,c2)==h:
                print(" val example:  ", (ex.get(c1), ex.get(c2)))
                break

    # duplicates within splits
    dup_train = len(train) - len(h_train)
    dup_val = len(val) - len(h_val)
    print(f"duplicate examples in train: {dup_train} in val: {dup_val}")

def main():
    print("QuickLeakCheck: Scanning GLUE datasets for overlaps and duplicates")
    print("="*60)
    
    for t in dataset_map:
        check_task(t)
    
    print("\n" + "="*60)
    print("✓ Check complete")
    print("\nIf overlaps or duplicates found, run: python tools/clean_datasets.py")

if __name__ == '__main__':
    main()
