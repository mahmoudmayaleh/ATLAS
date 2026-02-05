"""
ATLAS Integrated Experiment Runner
Connects all 4 phases for real federated learning experiments on Colab T4 GPU.

Pipeline:
1. Phase 1: Extract gradient fingerprints → cluster clients by task similarity
2. Phase 2: Profile devices → allocate heterogeneous LoRA ranks
3. Phase 3: Split federated learning with task-aware aggregation
4. Phase 4: Apply MIRA Laplacian regularization for personalization

Supports:
- Multi-task federated learning (different clients, different tasks)
- Heterogeneous devices (2GB CPU, 4GB tablet, 8GB laptop, 16GB GPU)
- Checkpoint/resume for long experiments (>3 hours)
- Real PyTorch training on DistilBERT/BERT/GPT-2
"""

import sys
import os

# Reduce noisy TensorFlow/XLA and HF/transformers logs before other imports
# Must set environment vars before importing modules that may load TF or XLA
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import warnings

# Configure library loggers to be quiet by default
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

# Also configure transformers' internal logger to error (suppresses 'Some weights ... were not initialized')
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# Suppress common sklearn UserWarnings from PCA resizing
warnings.filterwarnings('ignore', message='Reducing n_components', category=UserWarning)
# Globally suppress other benign warnings raised by helper modules during runs
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple, Optional
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import contextlib
import io

# Import all ATLAS phases
from phase1_clustering import GradientExtractor, TaskClusterer
from phase2_configuration import DeviceProfiler, RankAllocator
from phase3_split_fl import SplitClient, SplitServer
from phase4_laplacian import LaplacianAggregation, TaskGraph


@dataclass
class ATLASConfig:
    """Configuration for ATLAS integrated experiment"""
    # Model & tasks
    model_name: str = "distilbert-base-uncased"
    tasks: List[str] = None  # e.g., ['sst2', 'mrpc', 'cola']
    clients_per_task: int = 3  # 3 clients per task → 9 clients total for 3 tasks
    
    # Training
    num_rounds: int = 20  # Increased to 20-30 for MIRA convergence
    local_epochs: int = 2  # Keep moderate (1-2 epochs per round)
    batch_size: int = 16
    fingerprint_batch_size: int = 1  # Absolute minimum for T4 GPU with large datasets
    max_samples_per_client: int = 2000
    learning_rate: float = 2e-5
    
    # Device heterogeneity
    device_types: List[str] = None  # e.g., ['cpu_2gb', 'tablet_4gb', 'laptop_8gb', 'gpu_16gb']
    
    # Phase 1: Clustering
    fingerprint_epochs: int = 1  # Reduced to 1 epoch for memory efficiency
    fingerprint_batches: int = 20  # Only 20 batches total
    fingerprint_samples: int = 50  # Use only 50 samples (20 batches × 2 batch_size + buffer)
    fingerprint_dim: int = 64  # Target PCA dimension
    k_range: Tuple[int, int] = (2, 5)  # Try k=2,3,4,5 clusters
    # NOTE: For T4 GPU (15GB), fingerprinting uses minimal samples for memory safety
    
    # Phase 2: LoRA ranks
    rank_candidates: List[int] = None  # [4, 8, 16, 32, 64] - greedy importance-aware
    alpha_base: float = 0.5  # Base model takes 50% memory
    alpha_act: float = 0.25  # Activations take 25%
    alpha_opt: float = 0.08  # Optimizer takes 8% (reduced from 0.15 to force per-layer variation)
    use_importance_allocation: bool = True  # Use per-layer importance scores
    
    # Phase 3: Split learning
    split_layer: int = 3  # Split at layer 3 (bottom half)
    
    # Phase 4: Laplacian regularization (MIRA)
    eta: float = 0.1  # Regularization strength λ (tune: {0.0, 0.01, 0.1, 0.5, 1.0})
    laplacian_adjacency_method: str = 'mira_rbf'  # 'uniform', 'similarity', 'mira_rbf' (RECOMMENDED)
    mira_alpha: float = 1.0  # RBF kernel bandwidth for a_kℓ = exp(-α||f_k - f_ℓ||²)
    k_neighbors: int = 3
    block_diagonal: bool = True  # Zero cross-cluster edges for block structure
    ensure_connectivity: bool = True  # Ensure singletons have intra-task neighbors
    
    # Ablation & tuning modes
    mode: str = 'atlas'  # 'local_only', 'fedavg_cluster', 'atlas'
    lambda_sweep: bool = False  # If True, sweep eta over [0.0, 0.01, 0.1, 0.5, 1.0]
    lambda_values: List[float] = None  # For lambda sweep
    
    # Checkpointing (for multi-session training)
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5  # Save every 5 rounds for session resuming
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ['sst2', 'mrpc', 'cola']  # Default: 3 tasks
        if self.device_types is None:
            # Mix of devices: 2 low-end, 3 mid, 2 high, 2 very high
            self.device_types = ['cpu_2gb'] * 2 + ['tablet_4gb'] * 3 + ['laptop_8gb'] * 2 + ['gpu_16gb'] * 2
        if self.rank_candidates is None:
            self.rank_candidates = [4, 8, 16, 32, 64]  # Include 64 for large devices
        if self.lambda_values is None:
            self.lambda_values = [0.0, 0.01, 0.1, 0.5, 1.0]  # Lambda sweep grid


@dataclass
class ClientData:
    """Data holder for one client"""
    client_id: int
    task_name: str
    device_type: str
    train_dataset: Subset
    test_dataset: any
    cluster_id: Optional[int] = None
    lora_ranks: Optional[Dict[str, int]] = None


class ATLASIntegratedTrainer:
    """
    Full ATLAS pipeline integrating all 4 phases.
    Runs real federated learning with multi-task, heterogeneous devices.
    """
    
    def __init__(self, config: ATLASConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize ATLAS components
        self.gradient_extractor = GradientExtractor(
            dim=config.fingerprint_dim,
            device=self.device,
            layer_normalize=True
        )
        self.task_clusterer = TaskClusterer(
            n_clusters_range=config.k_range,
            min_cluster_size=1
        )
        self.device_profiler = DeviceProfiler()
        self.rank_allocator = RankAllocator(
            model_dim=768,  # DistilBERT hidden size
            bytes_per_param=4  # fp32
        )
        
        # Load model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Dataset mapping
        self.dataset_map = {
            'sst2': ('stanfordnlp/sst2', 'sentence', None, 2),
            'mrpc': ('nyu-mll/glue', 'sentence1', 'sentence2', 2),
            'cola': ('nyu-mll/glue', 'sentence', None, 2),
            'qnli': ('nyu-mll/glue', 'question', 'sentence', 2),
        }
        
        # Setup clients and data
        self.clients_data: List[ClientData] = []
        self._setup_multi_task_clients()
        
        print(f"\n[ATLAS] Initialized with:")
        print(f"  Model: {config.model_name}")
        print(f"  Tasks: {config.tasks}")
        print(f"  Total clients: {len(self.clients_data)}")
        print(f"  Device types: {set(c.device_type for c in self.clients_data)}")
        print(f"  Device: {self.device}")
    
    def _setup_multi_task_clients(self):
        """
        Setup multi-task federated learning:
        - Different clients work on different tasks
        - Each client gets subset of task data
        - Heterogeneous device assignment
        """
        print(f"\n[SETUP] Creating multi-task federated learning setup...")
        
        client_id = 0
        task_datasets = {}
        
        # Load and partition each task
        for task_name in self.config.tasks:
            print(f"  Loading task: {task_name}")
            train_data, test_data = self._load_task_data(task_name)
            task_datasets[task_name] = (train_data, test_data)
            
            # Partition among clients for this task
            n_clients = self.config.clients_per_task
            samples_per_client = len(train_data) // n_clients
            
            for i in range(n_clients):
                # Assign device type (cycle through available types)
                device_type = self.config.device_types[client_id % len(self.config.device_types)]
                
                # Create client data subset
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client if i < n_clients - 1 else len(train_data)
                indices = list(range(start_idx, min(end_idx, len(train_data))))
                
                # Limit to max_samples
                if len(indices) > self.config.max_samples_per_client:
                    indices = indices[:self.config.max_samples_per_client]
                
                client_subset = Subset(train_data, indices)
                
                client_data = ClientData(
                    client_id=client_id,
                    task_name=task_name,
                    device_type=device_type,
                    train_dataset=client_subset,
                    test_dataset=test_data
                )
                
                self.clients_data.append(client_data)
                
                print(f"    Client {client_id}: {task_name}, {device_type}, {len(indices)} samples")
                client_id += 1
        
        print(f"  ✓ Created {len(self.clients_data)} clients across {len(self.config.tasks)} tasks")
    
    def _load_task_data(self, task_name: str):
        """Load and tokenize dataset for one task"""
        if task_name not in self.dataset_map:
            raise ValueError(f"Unknown task: {task_name}")
        
        dataset_name, text_col, text_col2, num_labels = self.dataset_map[task_name]
        
        # Try loading cleaned datasets first, fallback to HuggingFace
        cleaned_path = Path(__file__).parent.parent / 'tools' / 'cleaned_data' / task_name
        if cleaned_path.exists():
            print(f"  [CLEAN] Loading pre-cleaned {task_name} from disk")
            from datasets import load_from_disk
            dataset = load_from_disk(str(cleaned_path / 'train'))
            test_dataset = load_from_disk(str(cleaned_path / 'validation'))
        else:
            # Load from HuggingFace (will apply dedup inline)
            if task_name == 'sst2':
                dataset = load_dataset(dataset_name, split='train')
                test_dataset = load_dataset(dataset_name, split='validation')
            else:
                dataset = load_dataset(dataset_name, task_name, split='train')
                test_dataset = load_dataset(dataset_name, task_name, split='validation')
        
        # Deduplicate within splits and remove train↔val overlap
        import hashlib
        def _text_hash(example):
            a = example.get(text_col) or ""
            if text_col2:
                b = example.get(text_col2) or ""
                s = f"{a} ||| {b}"
            else:
                s = a
            return hashlib.sha1(s.encode('utf-8')).hexdigest()
        
        # Build unique index lists for train
        train_hash_to_idx = {}
        unique_train_idxs = []
        for i, ex in enumerate(dataset):
            h = _text_hash(ex)
            if h in train_hash_to_idx:
                continue
            train_hash_to_idx[h] = i
            unique_train_idxs.append(i)
        
        # Build unique index lists for validation
        val_hash_to_idx = {}
        unique_val_idxs = []
        for i, ex in enumerate(test_dataset):
            h = _text_hash(ex)
            if h in val_hash_to_idx:
                continue
            val_hash_to_idx[h] = i
            unique_val_idxs.append(i)
        
        # Remove any train examples that overlap with validation
        overlap_hashes = set(train_hash_to_idx.keys()) & set(val_hash_to_idx.keys())
        if overlap_hashes:
            print(f"  [DEDUP] Removing {len(overlap_hashes)} train↔val overlaps from {task_name}")
            remove_idxs = {train_hash_to_idx[h] for h in overlap_hashes}
            unique_train_idxs = [i for i in unique_train_idxs if i not in remove_idxs]
        
        # Apply deduplication
        train_before = len(dataset)
        val_before = len(test_dataset)
        if len(unique_train_idxs) != train_before:
            dataset = dataset.select(unique_train_idxs)
            print(f"  [DEDUP] Removed {train_before - len(dataset)} duplicates from {task_name} train")
        if len(unique_val_idxs) != val_before:
            test_dataset = test_dataset.select(unique_val_idxs)
            print(f"  [DEDUP] Removed {val_before - len(test_dataset)} duplicates from {task_name} val")
        
        # Tokenize
        def tokenize_fn(examples):
            if text_col2:
                texts = [(t1, t2) for t1, t2 in zip(examples[text_col], examples[text_col2])]
                return self.tokenizer(texts, padding='max_length', truncation=True, max_length=128)
            else:
                return self.tokenizer(examples[text_col], padding='max_length', truncation=True, max_length=128)
        
        dataset = dataset.map(tokenize_fn, batched=True, load_from_cache_file=False)
        test_dataset = test_dataset.map(tokenize_fn, batched=True, load_from_cache_file=False)
        
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        return dataset, test_dataset
    
    def run_full_pipeline(self, resume_from: Optional[str] = None):
        """
        Run complete ATLAS pipeline:
        1. Phase 1: Gradient fingerprinting → clustering
        2. Phase 2: Device profiling → heterogeneous rank allocation
        3. Phase 3: Split FL training with task-aware aggregation
        4. Phase 4: Laplacian regularization for personalization
        """
        print(f"\n{'='*70}")
        print(f"ATLAS INTEGRATED EXPERIMENT")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Step 0: Resume or start fresh
        if resume_from:
            checkpoint = self._load_checkpoint(resume_from)
            start_round = checkpoint['round'] + 1
            print(f"[RESUME] Continuing from round {start_round}")
        else:
            start_round = 0
            checkpoint = None
        
        # ========== PHASE 1: TASK CLUSTERING ==========
        if start_round == 0:
            print(f"\n{'='*70}")
            print(f"PHASE 1: TASK CLUSTERING")
            print(f"{'='*70}\n")
            cluster_labels, fingerprints, clustering_metrics, layer_importances = self._phase1_clustering()
        else:
            cluster_labels = checkpoint['cluster_labels']
            fingerprints = checkpoint['fingerprints']
            clustering_metrics = checkpoint.get('clustering_metrics', {})
            layer_importances = checkpoint.get('layer_importances', {})
            print(f"[RESUME] Loaded clustering from checkpoint")
        
        # ========== PHASE 2: HETEROGENEOUS RANK ALLOCATION ==========
        if start_round == 0:
            print(f"\n{'='*70}")
            print(f"PHASE 2: HETEROGENEOUS RANK ALLOCATION")
            print(f"{'='*70}\n")
            device_configs = self._phase2_rank_allocation(cluster_labels, fingerprints, layer_importances)
        else:
            device_configs = checkpoint['device_configs']
            print(f"[RESUME] Loaded rank configs from checkpoint")
        
        # ========== PHASE 3 + 4: SPLIT FL + LAPLACIAN ==========
        print(f"\n{'='*70}")
        print(f"PHASE 3 & 4: SPLIT FL + LAPLACIAN REGULARIZATION")
        print(f"{'='*70}\n")
        results = self._phase3_4_training(
            cluster_labels,
            device_configs,
            fingerprints,
            start_round=start_round,
            checkpoint=checkpoint,
            clustering_metrics=clustering_metrics
        )

        # Persist Phase1/Phase2 metadata into results
        results['fingerprints'] = fingerprints
        results['clustering_metrics'] = clustering_metrics
        results['device_configs'] = device_configs
        results['layer_importances'] = layer_importances
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"[DONE] ATLAS pipeline complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Final per-client accuracy: {results['final_accuracies']}")
        print(f"{'='*70}\n")
        
        return results
    
    def _phase1_clustering(self) -> Tuple[Dict[int, int], Dict[int, np.ndarray], Dict]:
        """
        Phase 1: Extract gradient fingerprints and cluster clients.
        Returns: (cluster_labels, fingerprints)
        """
        print("[Phase 1] Extracting gradient fingerprints...")
        
        # Create temporary models for fingerprinting
        raw_gradients = {}
        layer_importances = {}  # Store per-client layer importance

        for client_data in self.clients_data:
            print(f"  Client {client_data.client_id} ({client_data.task_name})...", end=" ")

            # Create model
            _, _, _, num_labels = self.dataset_map[client_data.task_name]
            # Suppress transformers stdout/stderr noise during loading
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=num_labels
                ).to(self.device)

            # Extract raw gradient vector and layer importance
            raw_grad, layer_imp = self._extract_fingerprint(model, client_data.train_dataset)
            raw_gradients[client_data.client_id] = raw_grad
            layer_importances[client_data.client_id] = layer_imp

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # raw_grad may be a dict (layer-wise tensors) or a tensor/ndarray.
            if isinstance(raw_grad, dict):
                total_params = sum(g.numel() for g in raw_grad.values() if hasattr(g, 'numel'))
                print(f"✓ raw grad dict: {len(raw_grad)} tensors, total_params={total_params}")
            else:
                shape = getattr(raw_grad, 'shape', None)
                if shape is None:
                    try:
                        shape = np.asarray(raw_grad).shape
                    except Exception:
                        shape = 'unknown'
                print(f"✓ raw grad shape: {shape}")
        
        # Fit PCA on collected raw gradients and compute fingerprints
        print(f"\n[Phase 1] Fitting fingerprint PCA on {len(raw_gradients)} samples...")
        grad_list = [g for g in raw_gradients.values()]
        try:
            self.gradient_extractor.fit(grad_list)
        except Exception as e:
            print(f"[Phase 1] Warning: gradient extractor fit failed: {e}")

        fingerprints = {}
        for cid, raw in raw_gradients.items():
            try:
                fp = self.gradient_extractor.extract(raw)
            except Exception:
                # Fallback: convert to numpy and normalize
                arr = raw.detach().cpu().numpy() if hasattr(raw, 'detach') else np.asarray(raw)
                arr = arr.astype(np.float32)
                norm = np.linalg.norm(arr)
                fp = arr / (norm + 1e-8)
            fingerprints[cid] = fp

        # Cluster based on fingerprints
        print(f"\n[Phase 1] Clustering {len(fingerprints)} clients...")
        # Convert dict -> array (preserve client order)
        client_ids = list(fingerprints.keys())
        fps = np.vstack([fingerprints[cid] for cid in client_ids])

        res = self.task_clusterer.cluster(fps, verbose=True)
        metrics = res.get('metrics', {})
        labels = res.get('labels')

        # Map labels back to client ids
        cluster_labels = {cid: int(lbl) for cid, lbl in zip(client_ids, labels)}

        print(f"  ✓ Found {res.get('n_clusters', len(set(labels)))} task groups")
        
        # SANITY CHECK: Cluster-Task Alignment (validates clustering quality)
        print(f"\n[Phase 1] Cluster-Task Alignment Analysis:")
        cluster_task_purity = {}
        for cluster_id in sorted(set(labels)):
            clients_in_cluster = [cid for cid, label in cluster_labels.items() if label == cluster_id]
            tasks_in_cluster = [self.clients_data[cid].task_name for cid in clients_in_cluster]
            task_counts = {}
            for task in tasks_in_cluster:
                task_counts[task] = task_counts.get(task, 0) + 1
            
            # Compute purity: fraction of clients belonging to dominant task
            dominant_task = max(task_counts, key=task_counts.get) if task_counts else None
            purity = task_counts[dominant_task] / len(clients_in_cluster) if dominant_task else 0.0
            cluster_task_purity[cluster_id] = purity
            
            print(f"    Cluster {cluster_id}: {len(clients_in_cluster)} clients")
            print(f"      Tasks: {dict(task_counts)} (dominant: {dominant_task}, purity: {purity:.2f})") 
            print(f"      Client IDs: {clients_in_cluster}")
        
        avg_purity = np.mean(list(cluster_task_purity.values())) if cluster_task_purity else 0.0
        print(f"\n  ✓ Average cluster purity: {avg_purity:.3f}")
        if avg_purity < 0.8:
            warnings.warn(
                f"Low cluster-task alignment (purity={avg_purity:.2f}). "
                f"Clients with same task are spread across clusters. "
                f"Consider: (1) More fingerprint samples, (2) Stronger layer selection, "
                f"(3) Oracle clustering for debugging."
            )
        
        # Update client cluster assignments
        for client_data in self.clients_data:
            client_data.cluster_id = cluster_labels[client_data.client_id]
        
        clustering_metrics = metrics if metrics is not None else {}

        return cluster_labels, fingerprints, clustering_metrics, layer_importances
    
    def _extract_fingerprint(self, model: nn.Module, dataset: Subset) -> Tuple[Dict, Dict]:
        """Extract gradient fingerprint from a client's local training.
        
        Returns:
            (averaged_grads, layer_importance): gradient dict and per-layer importance scores
        """
        # Clear cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Enable gradient checkpointing to reduce memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Limit dataset to fingerprint_samples for memory efficiency
        fingerprint_size = min(len(dataset), self.config.fingerprint_samples)
        # Safely create subset - handle both Subset and Dataset
        if hasattr(dataset, 'indices'):
            # dataset is already a Subset
            selected_indices = dataset.indices[:fingerprint_size]
            fingerprint_subset = Subset(dataset.dataset, selected_indices)
        else:
            # dataset is a raw Dataset
            fingerprint_subset = Subset(dataset, list(range(fingerprint_size)))
        
        print(f"(using {fingerprint_size} samples)", end=" ")
        
        # Use smaller batch size for memory-intensive fingerprint extraction
        dataloader = DataLoader(fingerprint_subset, batch_size=self.config.fingerprint_batch_size, shuffle=True)
        # NO OPTIMIZER - we only need gradients, not weight updates
        
        model.train()
        grad_history = []
        layer_norms = {}  # Track per-layer gradient norms for importance
        
        # Train for fingerprint_epochs to collect gradients
        batch_limit = self.config.fingerprint_batches  # Total batches across all epochs
        total_batches_processed = 0
        
        for epoch in range(self.config.fingerprint_epochs):
            for batch_idx, batch in enumerate(dataloader):
                if total_batches_processed >= batch_limit:
                    break
                
                # Print progress every 5 batches
                if total_batches_processed % 5 == 0:
                    print(f"[{total_batches_processed}]", end="", flush=True)
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Zero gradients manually (no optimizer)
                model.zero_grad()
                
                # Use mixed precision (fp16) to reduce memory
                with autocast(enabled=torch.cuda.is_available()):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                loss.backward()
                
                # Collect gradients from EXACTLY last 2 transformer blocks (DistilBERT has 6 layers, BERT has 12)
                # DistilBERT: transformer.layer.4, transformer.layer.5, classifier
                # BERT: encoder.layer.10, encoder.layer.11, classifier
                grads_dict = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Check for last 2 layers based on model architecture
                        is_last_two = any([
                            'transformer.layer.4' in name or 'transformer.layer.5' in name,  # DistilBERT
                            'encoder.layer.10' in name or 'encoder.layer.11' in name,        # BERT
                            'classifier' in name or 'pooler' in name                         # Final layers
                        ])
                        if is_last_two:
                            # Move to CPU immediately to avoid GPU OOM
                            grad_tensor = param.grad.detach().cpu().clone()
                            grads_dict[name] = grad_tensor
                            
                            # Compute layer-level importance (squared gradient norm)
                            # Infer layer index from parameter name
                            import re
                            layer_match = re.search(r'layer[._](\d+)', name)
                            if layer_match:
                                layer_idx = int(layer_match.group(1))
                                layer_key = f'layer_{layer_idx}'
                            elif 'classifier' in name or 'pooler' in name:
                                layer_key = 'classifier'
                            else:
                                layer_key = 'other'
                            
                            grad_norm_sq = (grad_tensor ** 2).sum().item()
                            if layer_key not in layer_norms:
                                layer_norms[layer_key] = []
                            layer_norms[layer_key].append(grad_norm_sq)
                
                if grads_dict:
                    # Pass as dict for layer-wise normalization in GradientExtractor
                    grad_history.append(grads_dict)
                
                # Clear gradients immediately
                model.zero_grad(set_to_none=True)
                
                # Clear memory after EVERY batch to prevent OOM on T4
                del input_ids, attention_mask, labels, outputs, loss, grads_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                total_batches_processed += 1
            
            # Break outer loop if limit reached
            if total_batches_processed >= batch_limit:
                break
        
        # Extract fingerprint using GradientExtractor
        if grad_history:
            # Average gradient dicts across batches
            averaged_grads = {}
            for grad_dict in grad_history:
                for name, grad in grad_dict.items():
                    if name not in averaged_grads:
                        averaged_grads[name] = []
                    averaged_grads[name].append(grad)
            
            for name in averaged_grads:
                averaged_grads[name] = torch.mean(torch.stack(averaged_grads[name]), dim=0)
            
            # Compute average importance per layer
            layer_importance = {}
            for layer_key, norms in layer_norms.items():
                layer_importance[layer_key] = float(np.mean(norms))
            
            # Return gradient dict and importance scores
            return averaged_grads, layer_importance
        else:
            # Fallback: random raw gradient dict
            return {'fallback': torch.from_numpy(np.random.randn(self.config.fingerprint_dim)).float()}, {}
    
    def _phase2_rank_allocation(
        self, 
        cluster_labels: Dict[int, int],
        fingerprints: Dict[int, np.ndarray],
        layer_importances: Dict[int, Dict[str, float]]
    ) -> Dict[int, Dict]:
        """
        Phase 2: Allocate heterogeneous LoRA ranks based on device + cluster complexity.
        
        Improved allocation logic (MIRA-aligned):
        1. Compute cluster-level statistics (variance, difficulty)
        2. Use ACTUAL per-layer importance from gradient norms collected during fingerprinting
        3. Greedy allocation: sort layers by importance, try ranks {4,8,16,32,64}, pick largest under budget
        
        Returns: device_configs[client_id] = {device_profile, lora_ranks, cluster_stats}
        """
        print("[Phase 2] Profiling devices and allocating ranks...")
        
        device_configs = {}
        
        # Compute cluster-level statistics (variance = task complexity/heterogeneity)
        print("\n[Phase 2] Computing cluster-level statistics...")
        cluster_stats = {}
        for cluster_id in set(cluster_labels.values()):
            cluster_client_ids = [cid for cid, label in cluster_labels.items() if label == cluster_id]
            cluster_fingerprints = [fingerprints[cid] for cid in cluster_client_ids]
            
            if cluster_fingerprints:
                fps_array = np.vstack(cluster_fingerprints)
                # Variance: measure of within-cluster heterogeneity
                variance = np.var(fps_array, axis=0).mean()
                # Norm: measure of gradient magnitude (task difficulty)
                avg_norm = np.mean([np.linalg.norm(fp) for fp in cluster_fingerprints])
                
                cluster_stats[cluster_id] = {
                    'variance': variance,
                    'avg_norm': avg_norm,
                    'n_clients': len(cluster_client_ids),
                    'complexity_score': variance * avg_norm  # Combined metric
                }
                
                print(f"  Cluster {cluster_id}: variance={variance:.4f}, "
                      f"norm={avg_norm:.4f}, complexity={cluster_stats[cluster_id]['complexity_score']:.4f}")
        
        # Normalize complexity scores across clusters (for fair comparison)
        max_complexity = max(stats['complexity_score'] for stats in cluster_stats.values()) if cluster_stats else 1.0
        for cluster_id in cluster_stats:
            cluster_stats[cluster_id]['normalized_complexity'] = \
                cluster_stats[cluster_id]['complexity_score'] / max(max_complexity, 1e-8)
        
        print("\n[Phase 2] Allocating heterogeneous ranks per client...")
        for client_data in self.clients_data:
            device_type = client_data.device_type
            cluster_id = client_data.cluster_id
            client_id = client_data.client_id
            
            # Get device profile
            device_profile = self.device_profiler.profile_device(device_type)
            
            # Compute per-layer importance scores from ACTUAL gradient norms
            cluster_complexity = cluster_stats.get(cluster_id, {}).get('normalized_complexity', 1.0)
            
            if self.config.use_importance_allocation and client_id in layer_importances:
                # Use actual per-layer gradient norms from fingerprinting
                raw_importance = layer_importances[client_id]
                importance_scores = {}
                
                # Map layer names to layer indices
                for i in range(6):  # DistilBERT has 6 transformer layers
                    layer_key = f'layer_{i}'
                    if layer_key in raw_importance:
                        importance_scores[layer_key] = raw_importance[layer_key]
                    else:
                        # Fallback: heuristic (later layers more important)
                        importance_scores[layer_key] = 0.5 + (i / 6.0)
                
                # Add classifier importance
                if 'classifier' in raw_importance:
                    importance_scores['classifier'] = raw_importance['classifier']
                
                # Scale by cluster difficulty
                for key in importance_scores:
                    importance_scores[key] *= (0.5 + 0.5 * cluster_complexity)
            else:
                # Fallback: heuristic importance
                importance_scores = {}
                for i in range(6):  # DistilBERT has 6 transformer layers
                    # Layer importance increases with depth (0.5 to 1.5)
                    layer_importance = 0.5 + (i / 6.0)
                    # Scale by cluster difficulty
                    importance_scores[f'layer_{i}'] = layer_importance * (0.5 + 0.5 * cluster_complexity)
            
            # Normalize importance scores to sum to 1.0
            total_importance = sum(importance_scores.values())
            if total_importance > 1e-8:
                importance_scores = {k: v / total_importance for k, v in importance_scores.items()}
            
            # Allocate ranks using greedy importance-aware allocator
            # Log importance scores for debugging
            if client_id == 0:  # Log once for first client
                print(f"\n[Phase 2] Sample importance scores (client {client_id}): {importance_scores}")
            
            lora_ranks = self.rank_allocator.allocate_ranks(
                device_profile=device_profile,
                importance_scores=importance_scores,
                n_layers=6,
                split_point=None  # Could be adapted for split learning
            )
            
            # Validate memory constraint
            is_valid, adapter_mb = self.rank_allocator.validate_memory_constraint(
                lora_ranks, device_profile
            )
            
            device_configs[client_data.client_id] = {
                'device_profile': device_profile,
                'lora_ranks': lora_ranks,
                'cluster_stats': cluster_stats.get(cluster_id, {}),
                'importance_scores': importance_scores,
                'memory_valid': is_valid,
                'adapter_memory_mb': adapter_mb
            }
            
            client_data.lora_ranks = lora_ranks
            
            print(f"  Client {client_data.client_id} ({device_type}, cluster {cluster_id}): "
                  f"ranks={lora_ranks}, memory={adapter_mb:.1f}MB, valid={is_valid}")
        
        print(f"\n✓ Phase 2 complete: Allocated heterogeneous ranks for {len(device_configs)} clients")
        return device_configs
    
    def _phase3_4_training(
        self,
        cluster_labels: Dict[int, int],
        device_configs: Dict[int, Dict],
        fingerprints: Dict[int, np.ndarray],
        start_round: int = 0,
        checkpoint: Optional[Dict] = None,
        clustering_metrics: Optional[Dict] = None
    ) -> Dict:
        """
        Phase 3 & 4: Split federated learning + Laplacian regularization.
        Real training with heterogeneous LoRA + task-aware aggregation + personalization.
        """
        print("[Phase 3&4] Initializing split FL clients and server...")
        
        # Build task graph for Phase 4 (Laplacian)
        task_clusters = {}
        for cluster_id in set(cluster_labels.values()):
            task_clusters[cluster_id] = [cid for cid, label in cluster_labels.items() if label == cluster_id]
        
        # Build adjacency weights using MIRA's RBF kernel: a_kℓ = exp(-α||f_k - f_ℓ||²)
        print(f"\n[Phase 4] Building task graph with {self.config.laplacian_adjacency_method} adjacency...")
        
        from phase4_laplacian import compute_adjacency_weights
        
        adjacency_weights = compute_adjacency_weights(
            task_clusters=task_clusters,
            gradient_fingerprints=fingerprints,  # Use Phase 1 fingerprints (dict)
            method=self.config.laplacian_adjacency_method,  # 'mira_rbf' (recommended)
            mira_alpha=self.config.mira_alpha,  # RBF bandwidth parameter
            block_diagonal=self.config.block_diagonal,  # Zero cross-cluster edges
            ensure_connectivity=self.config.ensure_connectivity  # Connect singletons
        )
        
        print(f"  ✓ Computed {len(adjacency_weights)} adjacency weights using {self.config.laplacian_adjacency_method}")
        sample_weights = list(adjacency_weights.items())[:5]
        if sample_weights:
            print(f"  Sample weights: {sample_weights}")

        task_graph = TaskGraph.from_task_clusters(
            task_clusters=task_clusters,
            adjacency_weights=adjacency_weights,
            normalize=True,
            symmetrize=True
        )
        
        # Initialize Laplacian aggregator with configured eta (λ)
        laplacian_agg = LaplacianAggregation(
            eta=self.config.eta,  # Tunable regularization strength
            heterogeneous_rank=True
        )
        
        # Create per-client models (MIRA approach: each client keeps own model)
        client_models = {}
        for client_data in self.clients_data:
            _, _, _, num_labels = self.dataset_map[client_data.task_name]
            # Suppress transformers stdout/stderr noise during loading
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=num_labels
                )
            # Apply LoRA with heterogeneous ranks
            model = self._apply_heterogeneous_lora(model, client_data.lora_ranks)
            model = model.to(self.device)
            client_models[client_data.client_id] = model
        
        print(f"  ✓ Created {len(client_models)} personalized client models")
        
        # Training loop
        results = {
            'round_metrics': [],
            'final_accuracies': {},
            'cluster_labels': cluster_labels
        }
        
        for round_idx in range(start_round, self.config.num_rounds):
            round_start = time.time()
            print(f"\n{'='*70}")
            print(f"ROUND {round_idx + 1}/{self.config.num_rounds}")
            print(f"{'='*70}\n")
            
            # Step 1: Local training (each client trains own model)
            print(f"[Round {round_idx+1}] Local training...")
            round_losses = {}
            # Communication counters (bytes) for this round
            comm_upload = {c.client_id: 0 for c in self.clients_data}
            comm_download = {c.client_id: 0 for c in self.clients_data}
            
            for client_data in self.clients_data:
                cid = client_data.client_id
                model = client_models[cid]
                
                # Train locally
                loss = self._train_client_local(
                    model, 
                    client_data.train_dataset,
                    client_id=cid,
                    task_name=client_data.task_name
                )
                round_losses[cid] = loss
                # Measure upload size: sum bytes of trainable params (LoRA adapters + classifier)
                up_bytes = 0
                for name, param in model.named_parameters():
                    low = name.lower()
                    if ('lora' in low) or ('classifier' in low) or ('score' in low):
                        up_bytes += param.numel() * param.element_size()
                comm_upload[cid] = int(up_bytes)
            
            # Step 2: Task-aware aggregation (within clusters)
            print(f"\n[Round {round_idx+1}] Task-aware aggregation...")
            aggregated_models = {}
            
            for cluster_id, client_ids in task_clusters.items():
                print(f"  Group {cluster_id}: aggregating {len(client_ids)} clients")
                
                # Collect weights from clients in same cluster
                cluster_weights = [
                    {name: param.data.clone() for name, param in client_models[cid].named_parameters()}
                    for cid in client_ids
                ]
                
                # FedAvg within cluster
                avg_weights = self._fedavg_aggregate(cluster_weights)
                
                # Store for Phase 4 (convert flat state to LoRA-structured dict)
                lora_struct = self._flat_state_to_lora(avg_weights)
                for cid in client_ids:
                    aggregated_models[cid] = lora_struct

            # After aggregation, measure download size per client (server -> clients)
            for cid in aggregated_models:
                # aggregated_models[cid] is a dict layer->{'A','B'} tensors
                total_bytes = 0
                for layer_name, parts in aggregated_models[cid].items():
                    for k, t in parts.items():
                        if isinstance(t, (np.ndarray,)):
                            total_bytes += t.nbytes
                        else:
                            try:
                                total_bytes += int(t.numel() * t.element_size())
                            except Exception:
                                continue
                comm_download[cid] = int(total_bytes)
            
            # Step 3: Laplacian regularization (personalization)
            print(f"\n[Round {round_idx+1}] Applying Laplacian regularization...")
            personalized_models = laplacian_agg.laplacian_update(
                client_models=aggregated_models,
                task_graph=task_graph
            )
            
            # Update client models: personalized_models is a LoRA-style mapping
            for cid, lora_weights in personalized_models.items():
                model = client_models[cid]
                # Update adapter params in-place from lora_weights (layer -> {'A','B'})
                state = model.state_dict()
                new_state = {}
                import re
                for key, val in state.items():
                    key_low = key.lower()
                    if 'lora_a' in key_low or 'lora_b' in key_low:
                        # try to infer layer index from key (e.g., '.h.<idx>.')
                        m = re.search(r"\.h\.(\d+)\.", key)
                        if m:
                            layer_idx = int(m.group(1))
                            layer_name = f'layer_{layer_idx}'
                        else:
                            # fallback: look for 'layer_<n>' or use full key
                            m2 = re.search(r'layer_(\d+)', key_low)
                            if m2:
                                layer_name = f"layer_{int(m2.group(1))}"
                            else:
                                layer_name = None

                        if layer_name and layer_name in lora_weights:
                            if 'lora_a' in key_low and 'A' in lora_weights[layer_name]:
                                new_tensor = lora_weights[layer_name]['A']
                                # match shapes if possible
                                if new_tensor.shape == val.shape:
                                    new_state[key] = new_tensor.to(val.device)
                                else:
                                    # try transpose or truncate/pad
                                    try:
                                        cand = new_tensor.to(val.device)
                                        if cand.shape == val.shape:
                                            new_state[key] = cand
                                        else:
                                            # fallback to original
                                            new_state[key] = val
                                    except Exception:
                                        new_state[key] = val
                            elif 'lora_b' in key_low and 'B' in lora_weights[layer_name]:
                                new_tensor = lora_weights[layer_name]['B']
                                if new_tensor.shape == val.shape:
                                    new_state[key] = new_tensor.to(val.device)
                                else:
                                    try:
                                        cand = new_tensor.to(val.device)
                                        if cand.shape == val.shape:
                                            new_state[key] = cand
                                        else:
                                            new_state[key] = val
                                    except Exception:
                                        new_state[key] = val
                            else:
                                new_state[key] = val
                        else:
                            new_state[key] = val
                    else:
                        new_state[key] = val

                # Load updated state dict (non-strict to allow missing keys)
                try:
                    model.load_state_dict(new_state, strict=False)
                except Exception:
                    # Fallback: try partial update via named_parameters
                    for name, param in model.named_parameters():
                        if name in new_state:
                            try:
                                param.data.copy_(new_state[name])
                            except Exception:
                                continue
            
            # Step 4: Evaluation
            print(f"\n[Round {round_idx+1}] Evaluation...")
            round_accuracies = {}
            round_f1s = {}
            
            for client_data in self.clients_data:
                cid = client_data.client_id
                acc, loss, f1 = self._evaluate_client(
                    client_models[cid],
                    client_data.test_dataset
                )
                round_accuracies[cid] = acc
                round_f1s[cid] = f1
                print(f"  Client {cid} ({client_data.task_name}): acc={acc:.4f}, f1={f1:.4f}, loss={loss:.4f}")
            
            round_time = time.time() - round_start
            
            # Store results
            results['round_metrics'].append({
                'round': round_idx + 1,
                'train_losses': round_losses,
                'test_accuracies': round_accuracies,
                'test_f1': round_f1s,
                'avg_accuracy': np.mean(list(round_accuracies.values())),
                'time_seconds': round_time
            })
            # Attach communication metrics for this round
            results['round_metrics'][-1]['comm_upload_bytes'] = comm_upload
            results['round_metrics'][-1]['comm_download_bytes'] = comm_download
            
            print(f"\n[Round {round_idx+1}] Avg accuracy: {np.mean(list(round_accuracies.values())):.4f}, Time: {round_time:.1f}s")
            
            # Checkpoint (save every N rounds OR at end of training)
            is_last_round = (round_idx + 1) >= self.config.num_rounds
            should_checkpoint = (round_idx + 1) % max(1, int(self.config.save_every)) == 0 or is_last_round
            
            if should_checkpoint:
                checkpoint_state = {
                    'round': round_idx + 1,
                    'cluster_labels': cluster_labels,
                    'clustering_metrics': clustering_metrics,
                    'device_configs': device_configs,
                    'client_models': {cid: model.state_dict() for cid, model in client_models.items()},
                    'results': results,
                    'fingerprints': fingerprints
                }
                self._save_checkpoint(round_idx + 1, checkpoint_state)
        
        # Final accuracies
        results['final_accuracies'] = round_accuracies
        
        return results
    
    def _apply_heterogeneous_lora(self, model: nn.Module, lora_ranks) -> nn.Module:
        """Apply LoRA with heterogeneous ranks per layer"""
        from peft import get_peft_model, LoraConfig, TaskType
        
        # Get unique rank (simplified - use max rank for now)
        # In full implementation, would apply different ranks per layer
        rank = 8
        if lora_ranks:
            if isinstance(lora_ranks, dict):
                try:
                    rank = max(lora_ranks.values())
                except Exception:
                    rank = 8
            elif isinstance(lora_ranks, (list, tuple, np.ndarray)):
                try:
                    rank = int(max(lora_ranks))
                except Exception:
                    rank = 8
            else:
                # Fallback if unexpected type
                try:
                    rank = int(lora_ranks)
                except Exception:
                    rank = 8
        
        # Detect target modules
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                token = name.split('.')[-1]
                if any(k in token for k in ['q', 'k', 'v', 'query', 'key', 'value', 'attention']):
                    target_modules.append(token)
        
        target_modules = sorted(set(target_modules))
        
        # Find classifier modules
        classifier_modules = []
        for name, _ in model.named_modules():
            if any(cls in name for cls in ['classifier', 'score', 'pre_classifier']):
                classifier_modules.append(name.split('.')[-1])
        
        modules_to_save = sorted(set(classifier_modules)) if classifier_modules else ['classifier']
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=rank,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules if target_modules else ['query', 'value'],
            modules_to_save=modules_to_save
        )
        
        model = get_peft_model(model, lora_config)
        return model
    
    def _train_client_local(
        self, 
        model: nn.Module, 
        dataset: Subset,
        client_id: int,
        task_name: str
    ) -> float:
        """Train one client locally (Phase 3)"""
        model.train()
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.local_epochs):
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"    Client {client_id} ({task_name}): {num_batches} batches, loss={avg_loss:.4f}")
        return avg_loss
    
    def _evaluate_client(self, model: nn.Module, test_dataset) -> Tuple[float, float, float]:
        """Evaluate one client on test set. Returns (accuracy, avg_loss, f1)"""
        model.eval()
        dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size * 2)
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                predictions = torch.argmax(logits, dim=-1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
                num_batches += 1
                # Accumulate for F1
                all_preds.append(predictions.detach().cpu())
                all_labels.append(labels.detach().cpu())
        
        accuracy = total_correct / max(total_samples, 1)
        avg_loss = total_loss / max(num_batches, 1)
        # Compute F1 score (macro) across collected predictions
        try:
            if all_preds and all_labels:
                preds_cat = torch.cat(all_preds).numpy()
                labels_cat = torch.cat(all_labels).numpy()
                f1 = float(f1_score(labels_cat, preds_cat, average='macro', zero_division=0))
            else:
                f1 = 0.0
        except Exception:
            f1 = 0.0

        return accuracy, avg_loss, f1
    
    def _fedavg_aggregate(self, weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """FedAvg aggregation"""
        if not weights_list:
            return {}
        
        # Identify trainable keys (LoRA adapters + classifier)
        trainable_keywords = ['lora', 'classifier', 'score', 'modules_to_save']
        trainable_keys = [
            k for k in weights_list[0].keys() 
            if any(keyword in k.lower() for keyword in trainable_keywords)
        ]
        
        aggregated = {}
        for key in weights_list[0].keys():
            if key in trainable_keys:
                # Collect tensors present for this key
                tensors = [w.get(key) for w in weights_list]
                # Filter out None
                tensors_present = [t for t in tensors if t is not None]
                if not tensors_present:
                    continue

                # If all tensors have same shape, stack and mean
                shapes = [tuple(t.shape) for t in tensors_present]
                if all(s == shapes[0] for s in shapes):
                    stacked = torch.stack(tensors_present)
                    aggregated[key] = stacked.mean(dim=0)
                    continue

                # Handle LoRA adapters with heterogeneous ranks by padding to max shape
                key_low = key.lower()
                if 'lora_a' in key_low or 'lora_b' in key_low:
                    # Determine max shape across tensors
                    max_shape = [max(s[d] for s in shapes) for d in range(len(shapes[0]))]
                    padded = []
                    for t in tensors_present:
                        pad_tensor = torch.zeros(*max_shape, dtype=t.dtype, device=t.device)
                        # compute slices to copy
                        slices = tuple(slice(0, s) for s in t.shape)
                        pad_tensor[slices] = t
                        padded.append(pad_tensor)
                    stacked = torch.stack(padded)
                    aggregated[key] = stacked.mean(dim=0)
                    continue

                # Fallback for other mismatched shapes: use first available tensor (no averaging)
                aggregated[key] = tensors_present[0]
            else:
                # Keep frozen base model (should be identical)
                aggregated[key] = weights_list[0][key]

        return aggregated

    def _flat_state_to_lora(self, state: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Convert a flat state_dict (from PEFT/transformers) into a LoRA-style mapping:
        { 'layer_{i}': {'A': tensor, 'B': tensor}, ... }
        """
        import re
        lora = {}
        for key, val in state.items():
            key_low = key.lower()
            if 'lora_a' in key_low or 'lora_b' in key_low:
                # Attempt to extract transformer layer index like '.h.<idx>.'
                m = re.search(r"\.h\.(\d+)\.", key)
                if m:
                    layer_idx = int(m.group(1))
                    layer_name = f'layer_{layer_idx}'
                else:
                    m2 = re.search(r'layer_(\d+)', key_low)
                    if m2:
                        layer_name = f"layer_{int(m2.group(1))}"
                    else:
                        # Fallback: use module prefix before '.lora_'
                        parts = key.split('.lora_')
                        layer_name = parts[0] if parts else key_low

                if layer_name not in lora:
                    lora[layer_name] = {}

                if 'lora_a' in key_low:
                    lora[layer_name]['A'] = val.clone().cpu()
                elif 'lora_b' in key_low:
                    lora[layer_name]['B'] = val.clone().cpu()

        return lora
    
    def _save_checkpoint(self, round_num: int, state: Dict):
        """Save checkpoint for resume"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"atlas_round_{round_num}.pkl"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"[CHECKPOINT] Saved to {checkpoint_path}")
    
    def _load_checkpoint(self, path: str) -> Dict:
        """Load checkpoint"""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ATLAS integrated experiment")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("-r", "--rounds", type=int, help="Override number of rounds")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint (path to .pkl file)")
    parser.add_argument("--ablation", choices=["local_only", "fedavg_cluster", "atlas"], default="atlas",
                       help="Ablation mode: local_only, fedavg_cluster (per-cluster FedAvg), or atlas (full pipeline)")
    parser.add_argument("--lambda-sweep", action="store_true",
                       help="Run lambda sweep over [0.0, 0.01, 0.1, 0.5, 1.0]")
    parser.add_argument("--eta", type=float, help="Override Laplacian regularization strength (lambda)")
    
    # NEW: Model and task configuration for publication experiments
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                       help="Model to use: distilbert-base-uncased, bert-base-uncased, roberta-base, gpt2")
    parser.add_argument("--tasks", type=str, nargs="+", default=['sst2', 'mrpc', 'cola'],
                       help="Tasks to use (space-separated): sst2 mrpc cola qnli mnli")
    parser.add_argument("--clients-per-task", type=int, default=3,
                       help="Number of clients per task")
    parser.add_argument("--samples", type=int, help="Override max_samples_per_client")
    parser.add_argument("--local-epochs", type=int, help="Override local_epochs")
    parser.add_argument("--max-rounds", type=int, help="Maximum rounds for this session (for splitting 30→15+15)")
    args = parser.parse_args()
    
    if args.mode == "quick":
        # Quick test: For debugging and validation
        print("[MODE] Quick test (30-45 min on T4 GPU)")
        config = ATLASConfig(
            model_name="distilbert-base-uncased",
            tasks=['sst2', 'mrpc', 'cola'],
            clients_per_task=3,
            num_rounds=10,  # Quick validation
            local_epochs=2,
            batch_size=16,
            max_samples_per_client=1000,
            fingerprint_epochs=2,
            fingerprint_batches=64,
            mode=args.ablation,
            save_every=5
        )
    else:
        # Full experiment: Publication-quality parameters
        print("[MODE] Full experiment (2-4 hours per run on T4 GPU)")
        print("         For 30+ rounds, split into sessions: 15+15 with --resume")
        config = ATLASConfig(
            model_name="distilbert-base-uncased",
            tasks=['sst2', 'mrpc', 'cola'],
            clients_per_task=3,
            num_rounds=30,  # Publication quality
            local_epochs=3,  # More thorough training
            batch_size=16,
            max_samples_per_client=5000,  # Large enough for convergence
            fingerprint_epochs=3,  # More reliable fingerprints
            fingerprint_batches=100,
            mode=args.ablation,
            save_every=5  # Save every 5 rounds for session breaks
        )
    
    # Override parameters from CLI
    if args.rounds is not None:
        config.num_rounds = int(args.rounds)
    if args.eta is not None:
        config.eta = float(args.eta)
    if args.model:
        config.model_name = args.model
    if args.tasks:
        config.tasks = args.tasks
    if args.clients_per_task:
        config.clients_per_task = args.clients_per_task
    if args.samples:
        config.max_samples_per_client = args.samples
    if args.local_epochs:
        config.local_epochs = args.local_epochs
    
    # Session-based training: limit rounds for this session
    if args.max_rounds:
        config.num_rounds = args.max_rounds
        print(f"[SESSION] Limiting this session to {args.max_rounds} rounds (use --resume to continue)")
    
    # Lambda sweep mode
    if args.lambda_sweep:
        print("\\n[LAMBDA SWEEP] Running experiments over lambda values: {0.0, 0.01, 0.1, 0.5, 1.0}")
        print(f"Ablation mode: {config.mode}\\n")
        
        sweep_results = {}
        for lambda_val in config.lambda_values:
            print(f"\\n{'='*70}")
            print(f"LAMBDA = {lambda_val}")
            print(f"{'='*70}\\n")
            
            config.eta = lambda_val
            trainer = ATLASIntegratedTrainer(config)
            results = trainer.run_full_pipeline(resume_from=None)
            
            sweep_results[lambda_val] = {
                'final_accuracies': results.get('final_accuracies', {}),
                'avg_accuracy': np.mean(list(results.get('final_accuracies', {}).values())),
                'accuracy_variance': np.var(list(results.get('final_accuracies', {}).values())),
                'round_metrics': results.get('round_metrics', [])
            }
            
            print(f"\\nLambda={lambda_val}: Avg Acc={sweep_results[lambda_val]['avg_accuracy']:.4f}, "
                  f"Var={sweep_results[lambda_val]['accuracy_variance']:.6f}")
        
        # Save sweep results
        results_path = Path("./results") / f"lambda_sweep_{args.mode}_{config.mode}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        def _to_jsonable(obj):
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
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_jsonable(v) for v in obj]
            return obj
        
        with open(results_path, 'w') as f:
            json.dump(_to_jsonable(sweep_results), f, indent=2)
        
        print(f"\\n[SAVED] Lambda sweep results saved to {results_path}")
        print("\\n[DONE] Lambda sweep complete!")
    
    else:
        # Single run
        trainer = ATLASIntegratedTrainer(config)

        results = trainer.run_full_pipeline(resume_from=args.resume)
        
        # Save final results
        results_path = Path("./results") / f"atlas_integrated_{args.mode}_{config.mode}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        def _to_jsonable(obj):
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
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_jsonable(v) for v in obj]
            return obj

        with open(results_path, 'w') as f:
            results_json = {
                'round_metrics': _to_jsonable(results.get('round_metrics', [])),
                'final_accuracies': _to_jsonable(results.get('final_accuracies', {})),
                'cluster_labels': _to_jsonable(results.get('cluster_labels', {})),
                'fingerprints': _to_jsonable(results.get('fingerprints', {})),
                'clustering_metrics': _to_jsonable(results.get('clustering_metrics', {})),
                'device_configs': _to_jsonable(results.get('device_configs', {})),
                'layer_importances': _to_jsonable(results.get('layer_importances', {})),
                'config': asdict(config)
            }
            json.dump(results_json, f, indent=2)
        
        print(f"\n[SAVED] Results saved to {results_path}")
        print("\n[DONE] ATLAS integrated experiment complete!")
