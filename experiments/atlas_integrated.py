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

# Ensure the repository root is on sys.path so `src.*` imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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
from torch.utils.data import DataLoader, Subset, Dataset as TorchDataset
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
import numpy as np
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple, Optional, Any, Literal, cast
import time
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
import pickle
import contextlib
import io

# Import all ATLAS phases
from src.phase1_clustering import GradientExtractor, TaskClusterer
from src.phase2_configuration import DeviceProfiler, RankAllocator
from src.phase3_split_fl import SplitClient, SplitServer
from src.phase4_laplacian import LaplacianAggregation, TaskGraph


@dataclass
class ATLASConfig:
    """Configuration for ATLAS integrated experiment"""
    # Model & tasks
    model_name: str = "distilbert-base-uncased"
    tasks: List[str] = field(default_factory=lambda: ['sst2', 'mrpc', 'cola'])  # e.g., ['sst2', 'mrpc', 'cola']
    clients_per_task: int = 3  # 3 clients per task → 9 clients total for 3 tasks
    
    # Training
    num_rounds: int = 20  # Increased to 20-30 for MIRA convergence
    local_epochs: int = 2  # Keep moderate (1-2 epochs per round)
    batch_size: int = 16
    fingerprint_batch_size: int = 1  # Absolute minimum for T4 GPU with large datasets
    max_samples_per_client: int = 2000
    learning_rate: float = 5e-6
    gradient_clip_norm: float = 1.0  # Clip gradients to prevent explosion (critical for large models)
    
    # Device heterogeneity
    device_types: List[str] = field(default_factory=lambda: ['cpu_2gb'] * 2 + ['tablet_4gb'] * 3 + ['laptop_8gb'] * 2 + ['gpu_16gb'] * 2)
    
    # Phase 1: Clustering
    fingerprint_epochs: int = 1  # Reduced to 1 epoch for memory efficiency
    fingerprint_batches: int = 20  # Only 20 batches total
    fingerprint_samples: int = 50  # Use only 50 samples (20 batches × 2 batch_size + buffer)
    fingerprint_dim: int = 64  # Target PCA dimension
    k_range: Tuple[int, int] = (2, 5)  # Try k=2,3,4,5 clusters
    # NOTE: For T4 GPU (15GB), fingerprinting uses minimal samples for memory safety
    
    # Phase 2: LoRA ranks
    rank_candidates: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])  # [4, 8, 16, 32, 64] - greedy importance-aware
    alpha_base: float = 0.5  # Base model takes 50% memory
    alpha_act: float = 0.25  # Activations take 25%
    alpha_opt: float = 0.08  # Optimizer takes 8% (reduced from 0.15 to force per-layer variation)
    use_importance_allocation: bool = True  # Use per-layer importance scores
    
    # Phase 3: Split learning
    split_layer: int = 3  # Split at layer 3 (bottom half)
    
    # Phase 4: Laplacian regularization (MIRA)
    eta: float = 0.1  # Regularization strength λ (tune: {0.0, 0.01, 0.1, 0.5, 1.0})
    laplacian_adjacency_method: Literal['uniform', 'similarity', 'adaptive', 'mira_rbf'] = 'mira_rbf'  # 'mira_rbf' (RECOMMENDED)
    mira_alpha: float = 1.0  # RBF kernel bandwidth for a_kℓ = exp(-α||f_k - f_ℓ||²)
    k_neighbors: int = 3
    block_diagonal: bool = True  # Zero cross-cluster edges for block structure
    ensure_connectivity: bool = True  # Ensure singletons have intra-task neighbors
    
    # Ablation & tuning modes
    mode: str = 'atlas'  # 'local_only', 'fedavg_cluster', 'atlas'
    lambda_sweep: bool = False  # If True, sweep eta over [0.0, 0.01, 0.1, 0.5, 1.0]
    lambda_values: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.1, 0.5, 1.0])  # For lambda sweep
    
    # Checkpointing (for multi-session training)
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 999  # Only save final checkpoint (last round)
    seed: int = 42
    
    def __post_init__(self):
        # Backward-compatible safety: allow callers to pass None explicitly
        if self.tasks is None:
            self.tasks = ['sst2', 'mrpc', 'cola']
        if self.device_types is None:
            self.device_types = ['cpu_2gb'] * 2 + ['tablet_4gb'] * 3 + ['laptop_8gb'] * 2 + ['gpu_16gb'] * 2
        if self.rank_candidates is None:
            self.rank_candidates = [4, 8, 16, 32, 64]
        if self.lambda_values is None:
            self.lambda_values = [0.0, 0.01, 0.1, 0.5, 1.0]


@dataclass
class ClientData:
    """Data holder for one client"""
    client_id: int
    task_name: str
    device_type: str
    train_dataset: Subset
    test_dataset: Any
    cluster_id: Optional[int] = None
    lora_ranks: Any = None


class SplitServerWrapper(nn.Module):
    """
    Server-side model for genuine split federated learning.

    Receives intermediate activations from a client at `split_layer` and runs
    the remaining transformer layers plus the classification head, then returns
    the loss and activation-level gradients back to the client.

    Supports DistilBERT, GPT-2, and Qwen2/LLaMA architectures.
    """

    def __init__(self, model: nn.Module, split_layer: int, n_total_layers: int):
        super().__init__()
        self.split_layer = split_layer
        self.n_total_layers = n_total_layers
        self.arch = self._detect_arch(model)
        self._extract_top_components(model)

    @staticmethod
    def _detect_arch(model) -> str:
        if hasattr(model, 'distilbert'):
            return 'distilbert'
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return 'gpt2'
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            return 'llama_qwen'
        raise ValueError(f"SplitServerWrapper: unsupported architecture {type(model).__name__}")

    def _extract_top_components(self, model):
        s = self.split_layer
        if self.arch == 'distilbert':
            self.top_layers = nn.ModuleList(model.distilbert.transformer.layer[s:])
            self.pre_classifier = model.pre_classifier
            self.classifier = model.classifier
        elif self.arch == 'gpt2':
            self.top_layers = nn.ModuleList(model.transformer.h[s:])
            self.ln_f = model.transformer.ln_f
            self.score = model.score
        elif self.arch == 'llama_qwen':
            self.top_layers = nn.ModuleList(model.model.layers[s:])
            self.norm = model.model.norm
            self.score = model.score

    def forward(
        self,
        split_activations: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            split_activations: (batch, seq_len, hidden) – tensor with requires_grad=True
                               so that .grad is populated after loss.backward().
            attention_mask:    (batch, seq_len) int/bool mask (1 = keep, 0 = pad).
            labels:            (batch,) class indices for cross-entropy loss.
        Returns:
            logits: (batch, num_classes)
            loss:   scalar tensor if labels provided, else None.
        """
        import torch.nn.functional as F
        x = split_activations

        if self.arch == 'distilbert':
            # DistilBERT expects an *additive* mask: 0 for keep, -inf for pad.
            if attention_mask is not None:
                ext_mask = attention_mask[:, None, None, :].float()
                ext_mask = (1.0 - ext_mask) * -10000.0
            else:
                ext_mask = None
            for layer in self.top_layers:
                out = layer(x, attn_mask=ext_mask)
                x = out[-1] if isinstance(out, tuple) else out
            pooled = x[:, 0]                          # CLS token
            pooled = torch.relu(self.pre_classifier(pooled))
            logits = self.classifier(pooled)

        elif self.arch == 'gpt2':
            for block in self.top_layers:
                out = block(x)
                x = out[0] if isinstance(out, tuple) else out
            x = self.ln_f(x)
            # Classification uses the last *real* token.
            if attention_mask is not None:
                seq_lens = attention_mask.sum(dim=1).long() - 1
                seq_lens = seq_lens.clamp(min=0)
                batch_idx = torch.arange(x.size(0), device=x.device)
                pooled = x[batch_idx, seq_lens]
            else:
                pooled = x[:, -1]
            logits = self.score(pooled)

        elif self.arch == 'llama_qwen':
            for layer in self.top_layers:
                out = layer(x)
                x = out[0] if isinstance(out, tuple) else out
            x = self.norm(x)
            if attention_mask is not None:
                seq_lens = attention_mask.sum(dim=1).long() - 1
                seq_lens = seq_lens.clamp(min=0)
                batch_idx = torch.arange(x.size(0), device=x.device)
                pooled = x[batch_idx, seq_lens]
            else:
                pooled = x[:, -1]
            logits = self.score(pooled)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)  # type: ignore[possibly-unbound]

        return logits, loss  # type: ignore[possibly-unbound]


class ATLASIntegratedTrainer:
    """
    Full ATLAS pipeline integrating all 4 phases.
    Runs real federated learning with multi-task, heterogeneous devices.
    """
    
    def __init__(self, config: ATLASConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get model-specific configuration
        from config import get_model_hyperparameters
        try:
            model_hparams = get_model_hyperparameters(config.model_name)
            model_hidden_size = model_hparams['hidden_size']
        except:
            # Fallback to default
            model_hidden_size = 768
        
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
            model_dim=model_hidden_size,  # Use model-specific hidden size
            bytes_per_param=4  # fp32
        )
        
        # Load model & tokenizer (pass HF token if available)
        hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_HUB_TOKEN')
        try:
                        # Resolve common aliases (e.g., 'distilbert' -> 'distilbert-base-uncased')
            try:
                model_repo = get_model_config(config.model_name)['name']
            except Exception:
                model_repo = config.model_name

            self.tokenizer = AutoTokenizer.from_pretrained(model_repo, use_auth_token=hf_token)
        except Exception:
            # Fallback: try without token and with original name
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            except Exception:
                # Last resort: re-raise the original error for visibility
                raise
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
                
                client_subset = Subset(cast(TorchDataset, train_data), indices)
                
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
        
        # Normalize to HF Dataset objects (guard against load_from_disk returning DatasetDict)
        if isinstance(dataset, DatasetDict):
            dataset = dataset['train'] if 'train' in dataset else next(iter(dataset.values()))
        if isinstance(test_dataset, DatasetDict):
            test_dataset = test_dataset['validation'] if 'validation' in test_dataset else next(iter(test_dataset.values()))
        dataset = cast(HFDataset, dataset)
        test_dataset = cast(HFDataset, test_dataset)

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
            # checkpoint['round'] stores the number of completed rounds (1-based).
            # To resume, continue from the next 0-based round index = checkpoint['round']
            start_round = checkpoint['round']
            print(f"[RESUME] Continuing from round {start_round + 1}")
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
            assert checkpoint is not None
            cluster_labels = checkpoint['cluster_labels']
            fingerprints = checkpoint['fingerprints']
            clustering_metrics = checkpoint.get('clustering_metrics', {})
            layer_importances = checkpoint.get('layer_importances', {})
            print(f"[RESUME] Loaded clustering from checkpoint")

            # Restore per-client cluster ids (used by Phase 2+ and for result metadata)
            for client_data in self.clients_data:
                try:
                    client_data.cluster_id = int(cluster_labels[client_data.client_id])
                except Exception:
                    client_data.cluster_id = None
        
        # ========== PHASE 2: HETEROGENEOUS RANK ALLOCATION ==========
        if start_round == 0:
            print(f"\n{'='*70}")
            if self.config.mode in ['standard_fl', 'fedavg_cluster']:
                print(f"PHASE 2: HOMOGENEOUS RANK ALLOCATION")
            else:
                print(f"PHASE 2: HETEROGENEOUS RANK ALLOCATION")
            print(f"{'='*70}\n")
            
            if self.config.mode in ['standard_fl', 'fedavg_cluster']:
                # Homogeneous baseline: same rank for all clients
                device_configs = self._phase2_homogeneous_ranks(cluster_labels)
            else:
                # Heterogeneous: adaptive ranks per device/cluster (atlas, atlas_no_laplacian)
                device_configs = self._phase2_rank_allocation(cluster_labels, fingerprints, layer_importances)
        else:
            assert checkpoint is not None
            device_configs = checkpoint['device_configs']
            print(f"[RESUME] Loaded rank configs from checkpoint")

            # Restore per-client LoRA ranks (used for LoRA application and result metadata)
            for client_data in self.clients_data:
                cfg = device_configs.get(client_data.client_id) if isinstance(device_configs, dict) else None
                if isinstance(cfg, dict) and 'lora_ranks' in cfg:
                    client_data.lora_ranks = cfg.get('lora_ranks')
        
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
    
    def _phase1_clustering(self) -> Tuple[Dict[int, int], Dict[int, np.ndarray], Dict, Dict[int, Dict[str, float]]]:
        """
        Phase 1: Extract gradient fingerprints and cluster clients.
        Returns: (cluster_labels, fingerprints)
        """
        print("[Phase 1] Extracting gradient fingerprints...")
        
        # Create temporary models for fingerprinting
        raw_gradients = {}
        layer_importances = {}  # Store per-client layer importance

        # Collect both per-client averaged gradients and per-batch gradient samples
        grad_samples = []  # List to hold per-batch gradient samples
        grad_history = []  # List to hold per-batch gradient history
        for client_data in self.clients_data:
            print(f"  Client {client_data.client_id} ({client_data.task_name})...", end=" ")

            # Create model
            _, _, _, num_labels = self.dataset_map[client_data.task_name]
            # Suppress transformers stdout/stderr noise during loading
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=num_labels,
                    torch_dtype=torch.float32,  # Use FP32 for stability (GPT2-XL fits in 44GB)
                    ignore_mismatched_sizes=True
                ).to(self.device)
            
            # Set padding token for GPT2 (required for batch processing)
            if model.config.pad_token_id is None:
                model.config.pad_token_id = model.config.eos_token_id
            
            # Enable gradient checkpointing for large models (trade compute for memory)
            if hasattr(model, 'gradient_checkpointing_enable') and callable(getattr(model, 'gradient_checkpointing_enable', None)):
                try:
                    model.gradient_checkpointing_enable()
                    print("[Gradient checkpointing enabled]", end=" ")
                except Exception as e:
                    print(f"[Warning: gradient checkpointing failed: {e}]", end=" ")
            
            # Reinitialize classification head with small weights for stability
            if hasattr(model, 'classifier'):
                torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
                if model.classifier.bias is not None:
                    torch.nn.init.zeros_(model.classifier.bias)
            elif hasattr(model, 'score'):
                torch.nn.init.normal_(model.score.weight, mean=0.0, std=0.02)
                if model.score.bias is not None:
                    torch.nn.init.zeros_(model.score.bias)

            # Extract raw gradient vector, layer importance, and per-batch grads
            raw_grad, layer_imp, per_batch_grads = self._extract_fingerprint(model, client_data.train_dataset)
            raw_gradients[client_data.client_id] = raw_grad
            layer_importances[client_data.client_id] = layer_imp
            # Append per-batch gradient dicts (if any) so PCA has more samples
            if per_batch_grads:
                for pg in per_batch_grads:
                    grad_samples.append(pg)

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
        
        # Fit PCA on collected raw gradients (prefer per-batch samples if available)
        n_samples_msg = len(grad_samples) if grad_samples else len(raw_gradients)
        print(f"\n[Phase 1] Fitting fingerprint PCA on {n_samples_msg} samples...")
        grad_list = grad_samples if grad_samples else [g for g in raw_gradients.values()]
        try:
            self.gradient_extractor.fit(grad_list)
        except Exception as e:
            print(f"[Phase 1] Warning: gradient extractor fit failed: {e}")

        fingerprints = {}  # Dictionary to hold client fingerprints
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
        if labels is None:
            # Fallback: assign each client to its own cluster if clustering failed
            labels = list(range(len(client_ids)))
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
            dominant_task = max(task_counts, key=lambda t: task_counts[t]) if task_counts else None
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
    
    def _extract_fingerprint(self, model: nn.Module, dataset: Subset) -> Tuple[Dict, Dict, list]:
        """Extract gradient fingerprint from a client's local training.
        
        Returns:
            (averaged_grads, layer_importance, per_batch_grads): gradient dict, per-layer importance scores,
            and a list of per-batch gradient dicts collected during fingerprinting (may be empty).
        """
        # Clear cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Enable gradient checkpointing to reduce memory
        gradient_checkpointing_enable = getattr(model, 'gradient_checkpointing_enable', None)
        if callable(gradient_checkpointing_enable):
            try:
                gradient_checkpointing_enable()
            except Exception as e:
                print(f"[Warning: gradient checkpointing failed: {e}]", end=" ")
        
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
                model.zero_grad(set_to_none=True)
                
                # Forward pass (FP32)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Check for NaN before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ NaN loss in fingerprint, skipping batch", end="")
                    del input_ids, attention_mask, labels, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
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
            
            # Return gradient dict, importance scores, and per-batch gradients
            return averaged_grads, layer_importance, grad_history
        else:
            # Fallback: random raw gradient dict
            return {'fallback': torch.from_numpy(np.random.randn(self.config.fingerprint_dim)).float()}, {}, []
    
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
            
            if self.config.use_importance_allocation:
                # Prefer actual per-layer gradient norms from fingerprinting
                raw_importance = layer_importances.get(client_id)
                if raw_importance is None:
                    # Attempt to use cluster-average per-layer importance as a non-disruptive fallback
                    cluster_client_ids = [cid for cid, lbl in cluster_labels.items() if lbl == cluster_id]
                    # Gather available importances in the same cluster
                    gathered = [layer_importances[cid] for cid in cluster_client_ids if cid in layer_importances]
                    if gathered:
                        # Average per-layer values across gathered clients
                        avg_importance = {}
                        keys = set().union(*[set(d.keys()) for d in gathered])
                        for k in keys:
                            vals = [d.get(k, 0.0) for d in gathered]
                            avg_importance[k] = float(np.mean(vals)) if vals else 0.0
                        raw_importance = avg_importance
                        importance_source = 'cluster_average'
                    else:
                        raw_importance = None
                importance_scores = {}
                
                if raw_importance is not None:
                    # Map layer names to layer indices
                    for i in range(6):  # DistilBERT has 6 transformer layers
                        layer_key = f'layer_{i}'
                        if layer_key in raw_importance:
                            importance_scores[layer_key] = raw_importance[layer_key]
                        else:
                            # Fallback: heuristic (later layers more important)
                            importance_scores[layer_key] = 0.5 + (i / 6.0)
                else:
                    # No importance info available: fall back to deterministic heuristic
                    for i in range(6):
                        layer_key = f'layer_{i}'
                        importance_scores[layer_key] = 0.5 + (i / 6.0)
                
                # Add classifier importance if present in raw_importance
                if raw_importance is not None and 'classifier' in raw_importance:
                    # incorporate classifier importance as an extra key
                    importance_scores['classifier'] = float(raw_importance.get('classifier', 0.0))

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
    
    def _phase2_homogeneous_ranks(
        self, 
        cluster_labels: Dict[int, int]
    ) -> Dict[int, Dict]:
        """
        Phase 2 (Homogeneous variant): Allocate same LoRA rank for all clients.
        Used for standard_fl and homogeneous_atlas baselines.
        
        Returns: device_configs[client_id] = {device_profile, lora_ranks, cluster_stats}
        """
        print("[Phase 2] Allocating homogeneous ranks (same for all clients)...")

        from config import get_model_hyperparameters
        
        device_configs = {}
        homogeneous_rank = 16  # Fixed rank for all clients (adjustable)
        
        # Use model-specific LoRA ranks if available
        model_hparams = get_model_hyperparameters(self.config.model_name)
        if 'lora_ranks' in model_hparams and model_hparams['lora_ranks']:
            # Use median rank from model config
            homogeneous_rank = int(np.median(model_hparams['lora_ranks']))
        
        print(f"  Using homogeneous rank: {homogeneous_rank} for all clients")
        
        for client_data in self.clients_data:
            device_type = client_data.device_type
            cluster_id = client_data.cluster_id
            client_id = client_data.client_id
            
            # Get device profile
            device_profile = self.device_profiler.profile_device(device_type)
            
            # Assign same rank to all layers
            lora_ranks = [homogeneous_rank] * 6  # 6 layers for DistilBERT/GPT-2
            
            # Validate memory constraint
            is_valid, adapter_mb = self.rank_allocator.validate_memory_constraint(
                lora_ranks, device_profile
            )
            
            device_configs[client_id] = {
                'device_profile': device_profile,
                'lora_ranks': lora_ranks,
                'cluster_stats': {},  # No cluster complexity for homogeneous
                'importance_scores': {},
                'memory_valid': is_valid,
                'adapter_memory_mb': adapter_mb
            }
            
            client_data.lora_ranks = lora_ranks
            
            print(f"  Client {client_id} ({device_type}, cluster {cluster_id}): "
                  f"ranks={lora_ranks}, memory={adapter_mb:.1f}MB, valid={is_valid}")
        
        print(f"\n✓ Phase 2 complete: Allocated homogeneous ranks for {len(device_configs)} clients")
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
        print("[Phase 2] Profiling devices and allocating ranks...")  # Start of Phase 2

        mode = getattr(self.config, 'mode', 'atlas')
        
        # Build task graph for Phase 4 (Laplacian)
        task_clusters = {}
        for cluster_id in set(cluster_labels.values()):
            task_clusters[cluster_id] = [cid for cid, label in cluster_labels.items() if label == cluster_id]
        
        # Phase 4 (Laplacian) is only used for the full ATLAS method.
        # For other modes, we skip building the task graph and Laplacian updates.
        task_graph = None
        laplacian_agg = None
        if mode == 'atlas':
            # Build adjacency weights using MIRA's RBF kernel: a_kℓ = exp(-α||f_k - f_ℓ||²)
            print(f"\n[Phase 4] Building task graph with {self.config.laplacian_adjacency_method} adjacency...")

            from src.phase4_laplacian import compute_adjacency_weights

            # NOTE:
            # In this implementation, Phase 3 performs per-cluster FedAvg and assigns the *same*
            # aggregated LoRA weights to all clients inside a cluster. If we also enforce a
            # block-diagonal task graph (no cross-cluster edges), then for any client k all
            # neighbors ℓ satisfy W_k == W_ℓ and the Laplacian term becomes ~0.
            # To ensure Laplacian regularization has a measurable effect, we disable block-diagonal
            # adjacency for the Laplacian graph in `mode='atlas'`.
            laplacian_block_diagonal = False

            adjacency_weights = compute_adjacency_weights(
                task_clusters=task_clusters,
                gradient_fingerprints=fingerprints,  # Use Phase 1 fingerprints (dict)
                method=self.config.laplacian_adjacency_method,  # 'mira_rbf' (recommended)
                mira_alpha=self.config.mira_alpha,  # RBF bandwidth parameter
                block_diagonal=laplacian_block_diagonal,  # Allow cross-cluster edges for non-trivial Laplacian
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
        else:
            print(f"\n[Phase 4] Skipped (mode={mode})")
        
        # Create per-client models (MIRA approach: each client keeps own model)
        # NOTE: to avoid GPU OOM on a single GPU we keep models on CPU and move
        # a single client model to GPU only while training/evaluating it.
        client_models = {}
        for client_data in self.clients_data:
            _, _, _, num_labels = self.dataset_map[client_data.task_name]
            # Suppress transformers stdout/stderr noise during loading
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=num_labels,
                    torch_dtype=torch.float32,  # Use FP32 for stability (GPT2-XL fits in 44GB)
                    ignore_mismatched_sizes=True  # Ignore head size mismatch for LLMs
                )
            
            # Set padding token for GPT2 (required for batch processing)
            if model.config.pad_token_id is None:
                model.config.pad_token_id = model.config.eos_token_id
            
            # Reinitialize classification head with small weights for stability
            if hasattr(model, 'classifier'):
                torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
                if model.classifier.bias is not None:
                    torch.nn.init.zeros_(model.classifier.bias)
            elif hasattr(model, 'score'):  # Some models use 'score' instead
                torch.nn.init.normal_(model.score.weight, mean=0.0, std=0.02)
                if model.score.bias is not None:
                    torch.nn.init.zeros_(model.score.bias)
            
            # Enable gradient checkpointing before LoRA (saves memory during training)
            if hasattr(model, 'gradient_checkpointing_enable') and callable(getattr(model, 'gradient_checkpointing_enable', None)):
                try:
                    model.gradient_checkpointing_enable()
                except Exception as e:
                    print(f"[Warning: gradient checkpointing setup failed: {e}]")
            
            # Apply LoRA with heterogeneous ranks; keep model on CPU to save VRAM
            model = self._apply_heterogeneous_lora(model, client_data.lora_ranks)
            # Ensure model is on CPU (do not call .to(self.device) here)
            model.to('cpu')
            client_models[client_data.client_id] = model
        
        print(f"  ✓ Created {len(client_models)} personalized client models")

        # ── Build genuine split-server models (atlas / atlas_no_laplacian / fedavg_cluster) ──
        # local_only and standard_fl use full local training, so no server needed.
        use_split_learning = mode not in ('local_only', 'standard_fl')
        split_server_models: Dict[int, Dict] = {}
        if use_split_learning:
            split_server_models = self._build_split_server_models(task_clusters, device_configs)


        # Use clustering_metrics (returned from Phase 1) when available; fall back to safe defaults
        phase1_info = {
            'silhouette_score': float(clustering_metrics.get('silhouette_score', clustering_metrics.get('combined_score', 0.0))) if clustering_metrics else None,
            'davies_bouldin': float(clustering_metrics.get('davies_bouldin_index', clustering_metrics.get('davies_bouldin', 0.0))) if clustering_metrics else None,
            'num_clusters': len(task_clusters),
            'cluster_assignments': {int(k): [int(x) for x in v] for k, v in task_clusters.items()}
        }

        results = {
            'round_metrics': [],
            'final_accuracies': {},
            'cluster_labels': cluster_labels,
            'phase1_clustering': phase1_info,
            'phase2_rank_allocation': [
                {
                    'client_id': int(c.client_id),
                    'device': str(c.device_type),
                    'cluster': int(c.cluster_id) if c.cluster_id is not None else -1,
                    'ranks': [int(r) for r in (c.lora_ranks if c.lora_ranks is not None else [])],
                    'total_params': int(sum(int(r) for r in (c.lora_ranks if c.lora_ranks is not None else []))),
                    'lora_params': int(sum(int(r) * 768 * 2 for r in (c.lora_ranks if c.lora_ranks is not None else [])))  # Actual LoRA parameter count
                }
                for c in self.clients_data
            ],
            'communication_costs': {
                'per_round': [],  # Will store {round, upload_bytes, download_bytes} per client
                'total_bytes_uploaded': 0,
                'total_bytes_downloaded': 0
            },
            'time_metrics': {
                'phase1_time': 0,
                'phase2_time': 0,
                'per_round': []
            }
        }

        round_accuracies: Dict[int, float] = {}
        
        for round_idx in range(start_round, self.config.num_rounds):
            round_start = time.time()
            print(f"\n{'='*70}")
            print(f"ROUND {round_idx + 1}/{self.config.num_rounds}")
            print(f"{'='*70}\n")
            
            # Step 1: Client training
            # - local_only / standard_fl  → full local training (no split)
            # - all ATLAS modes           → genuine split learning (activations ↔ gradients)
            print(f"[Round {round_idx+1}] {'Split' if use_split_learning else 'Local'} training...")
            round_losses = {}
            # Communication counters (bytes) for this round
            comm_upload = {c.client_id: 0 for c in self.clients_data}
            comm_download = {c.client_id: 0 for c in self.clients_data}

            for client_data in self.clients_data:
                cid = client_data.client_id
                model = client_models[cid]
                model.to(self.device)

                if use_split_learning:
                    # ── Genuine split learning ──────────────────────────────
                    cluster_id = client_data.cluster_id if client_data.cluster_id is not None else 0
                    srv = split_server_models[cluster_id]
                    srv_model     = srv['model']      # already on self.device
                    srv_optimizer = srv['optimizer']
                    split_layer   = srv['split_layer']

                    loss, up_bytes, dn_bytes = self._train_client_split(
                        client_model=model,
                        dataset=client_data.train_dataset,
                        server_model=srv_model,
                        server_optimizer=srv_optimizer,
                        split_layer=split_layer,
                        client_id=cid,
                        task_name=client_data.task_name,
                    )
                    comm_upload[cid]   = int(up_bytes)
                    comm_download[cid] = int(dn_bytes)

                else:
                    # ── Full local training (local_only / standard_fl) ──────
                    loss = self._train_client_local(
                        model,
                        client_data.train_dataset,
                        client_id=cid,
                        task_name=client_data.task_name,
                    )
                    # No per-batch activation exchange → comm is 0 here;
                    # LoRA-weight upload is counted after aggregation below.
                    comm_upload[cid]   = 0
                    comm_download[cid] = 0

                round_losses[cid] = loss

                try:
                    model.to('cpu')
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                client_models[cid] = model

            # Baseline: local training only (no aggregation, no Laplacian)
            if mode == 'local_only':
                print(f"\n[Round {round_idx+1}] Aggregation skipped (mode=local_only)")
                print(f"[Round {round_idx+1}] Laplacian skipped (mode=local_only)")
            elif mode == 'standard_fl':
                # Standard FL: global FedAvg across all clients (no clustering)
                # Count LoRA-weight upload for each client (sent to server at round end).
                for cid in list(client_models.keys()):
                    up = 0
                    for name, param in client_models[cid].named_parameters():
                        if 'lora' in name.lower():
                            up += param.numel() * param.element_size()
                    comm_upload[cid] = int(up)

                print(f"\n[Round {round_idx+1}] Global FedAvg (standard FL)...")
                all_weights = [
                    {name: param.data.clone() for name, param in client_models[cid].named_parameters()}
                    for cid in range(len(client_models))
                ]
                avg_weights = self._fedavg_aggregate(all_weights)
                lora_struct = self._flat_state_to_lora(avg_weights)
                aggregated_models = {cid: lora_struct for cid in range(len(client_models))}
                print(f"[Round {round_idx+1}] Laplacian skipped (mode=standard_fl)")
                updated_models = aggregated_models

                # Measure download size per client (server -> clients) based on weights actually sent
                for cid in updated_models:
                    total_bytes = 0
                    for _layer_name, parts in updated_models[cid].items():
                        for _k, t in parts.items():
                            if isinstance(t, (np.ndarray,)):
                                total_bytes += t.nbytes
                            else:
                                try:
                                    total_bytes += int(t.numel() * t.element_size())
                                except Exception:
                                    continue
                    comm_download[cid] = int(total_bytes)

                # Apply aggregated LoRA weights back to client models
                for cid, lora_weights in updated_models.items():
                    model = client_models[cid]
                    state = model.state_dict()
                    new_state = {}
                    import re
                    for key, val in state.items():
                        key_low = key.lower()
                        if 'lora_a' in key_low or 'lora_b' in key_low:
                            m = re.search(r"\.h\.(\d+)\.", key)
                            if m:
                                layer_idx = int(m.group(1))
                                layer_name = f'layer_{layer_idx}'
                            else:
                                m2 = re.search(r'layer_(\d+)', key_low)
                                if m2:
                                    layer_name = f"layer_{int(m2.group(1))}"
                                else:
                                    layer_name = None

                            if layer_name and layer_name in lora_weights:
                                if 'lora_a' in key_low and 'A' in lora_weights[layer_name]:
                                    new_tensor = lora_weights[layer_name]['A']
                                    if new_tensor.shape == val.shape:
                                        new_state[key] = new_tensor.to(val.device)
                                    else:
                                        try:
                                            cand = new_tensor.to(val.device)
                                            new_state[key] = cand if cand.shape == val.shape else val
                                        except Exception:
                                            new_state[key] = val
                                elif 'lora_b' in key_low and 'B' in lora_weights[layer_name]:
                                    new_tensor = lora_weights[layer_name]['B']
                                    if new_tensor.shape == val.shape:
                                        new_state[key] = new_tensor.to(val.device)
                                    else:
                                        try:
                                            cand = new_tensor.to(val.device)
                                            new_state[key] = cand if cand.shape == val.shape else val
                                        except Exception:
                                            new_state[key] = val
                                else:
                                    new_state[key] = val
                            else:
                                new_state[key] = val
                        else:
                            new_state[key] = val

                    try:
                        model.load_state_dict(new_state, strict=False)
                    except Exception:
                        for name, param in model.named_parameters():
                            if name in new_state:
                                try:
                                    param.data.copy_(new_state[name])
                                except Exception:
                                    continue
            else:
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

                    # Convert flat state to LoRA-structured dict
                    lora_struct = self._flat_state_to_lora(avg_weights)
                    for cid in client_ids:
                        aggregated_models[cid] = lora_struct

                # Step 3: Optional Laplacian regularization (ATLAS only)
                if mode == 'atlas':
                    print(f"\n[Round {round_idx+1}] Applying Laplacian regularization...")
                    if laplacian_agg is not None and task_graph is not None:
                        updated_models = laplacian_agg.laplacian_update(
                            client_models=aggregated_models,
                            task_graph=task_graph
                        )
                    else:
                        print(f"[Round {round_idx+1}] Laplacian unavailable; skipping")
                        updated_models = aggregated_models
                else:
                    # Other modes: skip Laplacian (atlas_no_laplacian, fedavg_cluster)
                    print(f"\n[Round {round_idx+1}] Laplacian skipped (mode={mode})")
                    updated_models = aggregated_models

                # Measure download size per client (server -> clients) based on weights actually sent
                for cid in updated_models:
                    total_bytes = 0
                    for _layer_name, parts in updated_models[cid].items():
                        for _k, t in parts.items():
                            if isinstance(t, (np.ndarray,)):
                                total_bytes += t.nbytes
                            else:
                                try:
                                    total_bytes += int(t.numel() * t.element_size())
                                except Exception:
                                    continue
                    comm_download[cid] = int(total_bytes)

                # Update client models from server-provided LoRA weights
                for cid, lora_weights in updated_models.items():
                    model = client_models[cid]
                    state = model.state_dict()
                    new_state = {}
                    import re
                    for key, val in state.items():
                        key_low = key.lower()
                        if 'lora_a' in key_low or 'lora_b' in key_low:
                            m = re.search(r"\.h\.(\d+)\.", key)
                            if m:
                                layer_idx = int(m.group(1))
                                layer_name = f'layer_{layer_idx}'
                            else:
                                m2 = re.search(r'layer_(\d+)', key_low)
                                if m2:
                                    layer_name = f"layer_{int(m2.group(1))}"
                                else:
                                    layer_name = None

                            if layer_name and layer_name in lora_weights:
                                if 'lora_a' in key_low and 'A' in lora_weights[layer_name]:
                                    new_tensor = lora_weights[layer_name]['A']
                                    if new_tensor.shape == val.shape:
                                        new_state[key] = new_tensor.to(val.device)
                                    else:
                                        try:
                                            cand = new_tensor.to(val.device)
                                            new_state[key] = cand if cand.shape == val.shape else val
                                        except Exception:
                                            new_state[key] = val
                                elif 'lora_b' in key_low and 'B' in lora_weights[layer_name]:
                                    new_tensor = lora_weights[layer_name]['B']
                                    if new_tensor.shape == val.shape:
                                        new_state[key] = new_tensor.to(val.device)
                                    else:
                                        try:
                                            cand = new_tensor.to(val.device)
                                            new_state[key] = cand if cand.shape == val.shape else val
                                        except Exception:
                                            new_state[key] = val
                                else:
                                    new_state[key] = val
                            else:
                                new_state[key] = val
                        else:
                            new_state[key] = val

                    try:
                        model.load_state_dict(new_state, strict=False)
                    except Exception:
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
                model = client_models[cid]

                # Move model to GPU for evaluation
                model.to(self.device)
                acc, loss, f1 = self._evaluate_client(
                    model,
                    client_data.test_dataset
                )
                round_accuracies[cid] = acc
                round_f1s[cid] = f1
                print(f"  Client {cid} ({client_data.task_name}): acc={acc:.4f}, f1={f1:.4f}, loss={loss:.4f}")

                # Move model back to CPU after evaluation
                try:
                    model.to('cpu')
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                client_models[cid] = model
            
            round_time = time.time() - round_start
            
            # Calculate communication totals for this round
            round_upload_total = sum(comm_upload.values())
            round_download_total = sum(comm_download.values())
            results['communication_costs']['total_bytes_uploaded'] += round_upload_total
            results['communication_costs']['total_bytes_downloaded'] += round_download_total
            results['communication_costs']['per_round'].append({
                'round': round_idx + 1,
                'upload_bytes': comm_upload,
                'download_bytes': comm_download,
                'total_upload': round_upload_total,
                'total_download': round_download_total
            })
            results['time_metrics']['per_round'].append({
                'round': round_idx + 1,
                'time_seconds': round_time
            })
            
            # Store results
            results['round_metrics'].append({
                'round': round_idx + 1,
                'train_losses': round_losses,
                'test_accuracies': round_accuracies,
                'test_f1': round_f1s,
                'avg_accuracy': np.mean(list(round_accuracies.values())),
                'time_seconds': round_time,
                'comm_upload_bytes': comm_upload,
                'comm_download_bytes': comm_download
            })
            
            print(f"\n[Round {round_idx+1}] Avg accuracy: {np.mean(list(round_accuracies.values())):.4f}, Time: {round_time:.1f}s")
            print(f"[Round {round_idx+1}] Communication: ↑{round_upload_total/1e6:.2f}MB ↓{round_download_total/1e6:.2f}MB")
            
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
        
        # ── Final summary metrics (last round) ───────────────────────────
        results['final_accuracies'] = {int(k): float(v) for k, v in round_accuracies.items()}
        results['final_canonical']  = {int(k): float(v) for k, v in round_canonical.items()}
        results['final_f1']         = {int(k): float(v) for k, v in round_f1s.items()}

        # Per-task final scores (the proof that ATLAS preserves task identity)
        final_task_canonical: Dict[str, List[float]] = {}
        for cd in self.clients_data:
            tname = cd.task_name
            final_task_canonical.setdefault(tname, []).append(
                round_canonical.get(cd.client_id, 0.0)
            )
        results['final_task_scores'] = {
            tname: {
                'metric':     self.TASK_METRIC.get(tname, 'accuracy'),
                'mean':       float(np.mean(vals)),
                'std':        float(np.std(vals)),
                'per_client': [float(v) for v in vals],
            }
            for tname, vals in final_task_canonical.items()
        }
        results['macro_avg_canonical']    = float(np.mean([
            v['mean'] for v in results['final_task_scores'].values()
        ]))
        results['personalization_spread'] = float(np.std(
            list(results['final_canonical'].values())
        ))
        results['total_comm_mb'] = (
            results['communication_costs']['total_bytes_uploaded'] +
            results['communication_costs']['total_bytes_downloaded']
        ) / 1e6

        # Print proof-of-ATLAS summary
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS — mode={mode}")
        print(f"{'='*70}")
        for tname, info in results['final_task_scores'].items():
            print(f"  {tname:6s}: {info['metric']}={info['mean']:.4f} ± {info['std']:.4f}")
        print(f"  Macro avg canonical : {results['macro_avg_canonical']:.4f}")
        print(f"  Personalization spread (std across clients): {results['personalization_spread']:.4f}")
        print(f"  Total communication : {results['total_comm_mb']:.1f} MB")
        print(f"{'='*70}\n")

        try:
            results['run_metadata'] = {
                'ablation_mode':      str(mode),
                'laplacian_enabled':  bool(mode == 'atlas'),
                'split_learning':     bool(use_split_learning),
                'split_learning_type': 'genuine_activation_exchange' if use_split_learning else 'none',
            }
        except Exception:
            pass
        
        return results
    
    def _apply_heterogeneous_lora(self, model: PreTrainedModel, lora_ranks) -> nn.Module:
        """Apply LoRA with heterogeneous ranks per layer"""
        from peft import get_peft_model, LoraConfig, TaskType
        
        # Get unique rank (simplified - use max rank for now)
        # In full implementation, would apply different ranks per layer
        rank = 8
        if lora_ranks is not None:
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
        
        # Debug: print model type and structure
        print(f"[LoRA Debug] Model type: {type(model).__name__}")
        print(f"[LoRA Debug] Model config type: {type(model.config).__name__}")
        
        # Auto-detect target modules based on model architecture
        # Use FULL module paths, not just last component
        all_module_names = []
        all_module_types = []
        
        for name, module in model.named_modules():
            all_module_types.append((name, type(module).__name__))
            # GPT2 uses Conv1D instead of Linear!
            if isinstance(module, nn.Linear) or type(module).__name__ == 'Conv1D':
                all_module_names.append(name)
                
        print(f"[LoRA Debug] Total modules: {len(all_module_types)}")
        print(f"[LoRA Debug] First 10 modules: {all_module_types[:10]}")
        print(f"[LoRA Debug] Linear/Conv1D modules found: {len(all_module_names)}")
        print(f"[LoRA Debug] Sample linear modules: {all_module_names[:5]}")
        
        # Architecture-specific patterns (use regex-style patterns)
        target_modules = []
        
        # Check for GPT2 architecture
        if any('c_attn' in name for name in all_module_names):
            target_modules = ['c_attn', 'c_proj']
        # Check for LLaMA architecture
        elif any('q_proj' in name for name in all_module_names):
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        # Check for BERT/DistilBERT architecture
        elif any('query' in name for name in all_module_names):
            target_modules = ['query', 'key', 'value']
        else:
            # Generic fallback: find common attention patterns
            patterns = ['attn', 'attention', 'self']
            for pattern in patterns:
                matched = [n.split('.')[-1] for n in all_module_names if pattern in n and 'score' not in n and 'classifier' not in n]
                if matched:
                    target_modules = list(set(matched))[:4]
                    break
        
        # Last resort: just use first few non-classifier modules
        if not target_modules:
            target_modules = [n.split('.')[-1] for n in all_module_names if 'score' not in n and 'classifier' not in n][:3]
        
        # If STILL no targets, there's a serious problem - use 'all-linear' as emergency fallback
        if not target_modules:
            print("[LoRA Debug] WARNING: No suitable target modules found! Using emergency fallback.")
            target_modules = 'all-linear'  # PEFT special keyword
        
        print(f"[LoRA Debug] Target modules selected: {target_modules}")
        
        # Find classifier modules - these should NOT overlap with target_modules
        classifier_modules = []
        for name in all_module_names:
            module_name = name.split('.')[-1]
            if any(cls in module_name for cls in ['classifier', 'score', 'pre_classifier']):
                if module_name not in target_modules:  # Avoid overlap
                    classifier_modules.append(module_name)
        
        modules_to_save = sorted(set(classifier_modules)) if classifier_modules else None
        print(f"[LoRA Debug] Modules to save (classifier): {modules_to_save}")
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=rank,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules,
            modules_to_save=modules_to_save
        )
        
        peft_model = get_peft_model(model, lora_config)
        return cast(nn.Module, peft_model)
    
    # =========================================================================
    # Genuine Split Learning helpers (Phase 3)
    # =========================================================================

    def _split_forward_bottom(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        split_layer: int,
    ) -> torch.Tensor:
        """
        Run only the bottom `split_layer` transformer blocks of a PEFT-wrapped model.

        The returned tensor is part of the LoRA-parameter compute graph, so
        calling `.backward(activation_gradients)` on it propagates gradients
        through the client-side LoRA matrices.

        Supports DistilBERT, GPT-2, and Qwen2/LLaMA architectures.
        """
        # Unwrap PEFT shell to reach the underlying HuggingFace model.
        base = model.base_model.model if hasattr(model, 'base_model') else model # pyright: ignore[reportAttributeAccessIssue]

        # ---- DistilBERT ----
        if hasattr(base, 'distilbert'):
            x = base.distilbert.embeddings(input_ids=input_ids)
            # Additive attention mask expected by DistilBERT layers.
            ext_mask = attention_mask[:, None, None, :].float()
            ext_mask = (1.0 - ext_mask) * -10000.0
            for layer in base.distilbert.transformer.layer[:split_layer]:
                out = layer(x, attn_mask=ext_mask)
                x = out[-1] if isinstance(out, tuple) else out
            return x  # (batch, seq_len, hidden)

        # ---- GPT-2 ----
        if hasattr(base, 'transformer') and hasattr(base.transformer, 'h'):
            pos_ids = torch.arange(
                input_ids.size(1), dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
            x = base.transformer.wte(input_ids) + base.transformer.wpe(pos_ids)
            for block in base.transformer.h[:split_layer]:
                out = block(x)
                x = out[0] if isinstance(out, tuple) else out
            return x  # (batch, seq_len, hidden)

        # ---- Qwen2 / LLaMA ----
        if hasattr(base, 'model') and hasattr(base.model, 'embed_tokens'):
            x = base.model.embed_tokens(input_ids)
            for layer in base.model.layers[:split_layer]:
                out = layer(x)
                x = out[0] if isinstance(out, tuple) else out
            return x  # (batch, seq_len, hidden)

        raise ValueError(
            f"_split_forward_bottom: unsupported model architecture "
            f"'{type(base).__name__}'. Expected distilbert / transformer+h / model+embed_tokens."
        )

    def _build_split_server_models(
        self,
        task_clusters: Dict[int, List[int]],
        device_configs: Dict[int, Dict],
    ) -> Dict[int, Dict]:
        """
        Create one SplitServerWrapper per task cluster.

        Each server model holds the top (n_total - split_layer) transformer
        blocks plus the classification head. It is shared by all clients in
        a cluster and trained server-side during split learning.

        Returns:
            {cluster_id: {'model': SplitServerWrapper,
                          'optimizer': AdamW,
                          'split_layer': int}}
        """
        print("\n[Phase 3] Building server-side top-layer models (genuine split learning)...")
        server_models: Dict[int, Dict] = {}

        for cluster_id, client_ids in task_clusters.items():
            rep_cid = client_ids[0]
            rep_client = next(c for c in self.clients_data if c.client_id == rep_cid)
            _, _, _, num_labels = self.dataset_map[rep_client.task_name]

            cfg = device_configs.get(rep_cid) if isinstance(device_configs, dict) else None
            split_layer = (
                cfg.get('split_layer', self.config.split_layer)
                if isinstance(cfg, dict) else self.config.split_layer
            )

            # Load a fresh base model for the server (not PEFT-wrapped).
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=num_labels,
                    torch_dtype=torch.float32,
                    ignore_mismatched_sizes=True,
                )
            if base_model.config.pad_token_id is None:
                base_model.config.pad_token_id = base_model.config.eos_token_id

            n_total = getattr(
                base_model.config, 'n_layer',
                getattr(base_model.config, 'num_hidden_layers', 12)
            )

            try:
                server_wrapper = SplitServerWrapper(base_model, split_layer, n_total)
            except ValueError as exc:
                raise RuntimeError(
                    f"Cannot build SplitServerWrapper for cluster {cluster_id}: {exc}"
                ) from exc

            server_wrapper.to(self.device)
            optimizer = torch.optim.AdamW(
                server_wrapper.parameters(), lr=self.config.learning_rate
            )

            server_models[cluster_id] = {
                'model': server_wrapper,
                'optimizer': optimizer,
                'split_layer': split_layer,
            }
            print(
                f"  Cluster {cluster_id}: arch={server_wrapper.arch}, "
                f"split_layer={split_layer}/{n_total}, labels={num_labels}"
            )

        print(f"  ✓ Built {len(server_models)} server model(s).")
        return server_models

    def _train_client_split(
        self,
        client_model: nn.Module,
        dataset: Subset,
        server_model: 'SplitServerWrapper',
        server_optimizer: torch.optim.Optimizer,
        split_layer: int,
        client_id: int,
        task_name: str,
    ):
        """
        Genuine split federated learning training step for one client.

        Protocol (per batch):
          1. CLIENT  – run embedding + bottom split_layer transformer blocks
                       (LoRA adapters are active → activations are in their
                       compute graph).
          2. NETWORK – upload: activation tensor (batch × seq_len × hidden).
          3. SERVER  – receive activations as a new leaf (requires_grad=True),
                       run top layers + head, compute cross-entropy loss,
                       backprop → fill leaf.grad with dL/d(activations).
          4. NETWORK – download: gradient tensor (same shape as activations).
          5. CLIENT  – call split_activations.backward(activation_gradients)
                       to propagate gradients through LoRA, then optimizer.step().

        Communication is counted from the actual tensor sizes — no estimates.

        Returns:
            avg_loss (float), total_upload_bytes (int), total_download_bytes (int)
        """
        client_model.train()
        server_model.train()

        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        client_optimizer = torch.optim.AdamW(
            [p for p in client_model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )

        total_loss = 0.0
        num_batches = 0
        total_upload_bytes = 0
        total_download_bytes = 0
        nan_count = 0

        for _epoch in range(self.config.local_epochs):
            for batch in dataloader:
                input_ids      = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels         = batch['label'].to(self.device)

                client_optimizer.zero_grad()
                server_optimizer.zero_grad()

                # ── CLIENT: forward through bottom layers ──────────────────
                # split_activations is connected to the LoRA param graph.
                try:
                    split_activations = self._split_forward_bottom(
                        client_model, input_ids, attention_mask, split_layer
                    )
                except Exception as exc:
                    print(f"    ⚠️  Client {client_id} bottom-forward failed: {exc}. Skipping batch.")
                    continue

                # ── NETWORK UPLOAD: count actual bytes ─────────────────────
                upload_bytes = split_activations.numel() * split_activations.element_size()
                total_upload_bytes += int(upload_bytes)

                # ── SERVER: create a detached leaf so .grad gets filled ────
                server_input = split_activations.detach().requires_grad_(True)

                _, loss = server_model(server_input, attention_mask, labels)

                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    continue

                # Server backward: populates server_input.grad = dL/d(activations)
                loss.backward()

                # Clip server-side gradients
                torch.nn.utils.clip_grad_norm_(
                    server_model.parameters(), self.config.gradient_clip_norm
                )
                server_optimizer.step()

                if server_input.grad is None:
                    # No gradient reached the split point – skip this batch.
                    del server_input, loss
                    continue
                activation_gradients = server_input.grad.detach().clone()

                # ── NETWORK DOWNLOAD: count actual bytes ───────────────────
                download_bytes = activation_gradients.numel() * activation_gradients.element_size()
                total_download_bytes += int(download_bytes)

                # ── CLIENT: backward through LoRA using server gradients ───
                split_activations.backward(activation_gradients)
                torch.nn.utils.clip_grad_norm_(
                    client_model.parameters(), self.config.gradient_clip_norm
                )
                client_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Free GPU memory after each batch
                del input_ids, attention_mask, labels, split_activations
                del server_input, activation_gradients, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_batches, 1)
        status = f"  ⚠️ {nan_count} NaN batches skipped" if nan_count else ""
        print(
            f"    Client {client_id} ({task_name}): {num_batches} batches, "
            f"loss={avg_loss:.4f}, "
            f"↑{total_upload_bytes / 1e6:.2f} MB  ↓{total_download_bytes / 1e6:.2f} MB"
            f"{status}"
        )
        return avg_loss, total_upload_bytes, total_download_bytes

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
        nan_detected = False
        
        for epoch in range(self.config.local_epochs):
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass (no autocast for FP32)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"    ⚠️  Client {client_id}: NaN/Inf loss detected at batch {num_batches}. Skipping batch.")
                    nan_detected = True
                    continue
                
                # Standard backward (no scaler needed for FP16 models)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                
                # Optimizer step
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        status = "⚠️ NaN detected" if nan_detected else ""
        print(f"    Client {client_id} ({task_name}): {num_batches} batches, loss={avg_loss:.4f} {status}")
        return avg_loss
    
    # Task → canonical metric name (matches the paper's Table II)
    TASK_METRIC: Dict[str, str] = {
        'sst2': 'accuracy',
        'mrpc': 'f1',
        'cola': 'matthews',
        'qnli': 'accuracy',
    }

    def _evaluate_client(
        self,
        model: nn.Module,
        test_dataset,
        task_name: str = 'sst2',
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate one client on its test split.

        Returns:
            accuracy  (float) – raw classification accuracy
            avg_loss  (float) – mean cross-entropy loss
            f1        (float) – macro-F1
            canonical (float) – the task's *primary* metric used in the paper:
                                 accuracy for SST-2 / QNLI,
                                 F1        for MRPC,
                                 Matthews  for CoLA
        """
        from sklearn.metrics import matthews_corrcoef

        model.eval()
        dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size * 2)

        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        num_batches = 0
        all_preds: List[torch.Tensor] = []
        all_labels_list: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids      = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels         = batch['label'].to(self.device)

                outputs     = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss        = outputs.loss
                logits      = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                total_correct  += (predictions == labels).sum().item()
                total_samples  += labels.size(0)
                total_loss     += loss.item()
                num_batches    += 1
                all_preds.append(predictions.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        accuracy = total_correct / max(total_samples, 1)
        avg_loss = total_loss / max(num_batches, 1)

        preds_np  = torch.cat(all_preds).numpy()          if all_preds        else np.array([])
        labels_np = torch.cat(all_labels_list).numpy()   if all_labels_list  else np.array([])

        try:
            f1 = float(f1_score(labels_np, preds_np, average='macro', zero_division=0)) \
                if len(preds_np) > 0 else 0.0
        except Exception:
            f1 = 0.0

        # Task-specific canonical metric
        metric_name = self.TASK_METRIC.get(task_name, 'accuracy')
        try:
            if metric_name == 'matthews' and len(preds_np) > 0:
                canonical = float(matthews_corrcoef(labels_np, preds_np))
            elif metric_name == 'f1':
                canonical = f1
            else:
                canonical = accuracy
        except Exception:
            canonical = accuracy

        return accuracy, avg_loss, f1, canonical
    
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
        # Include ablation mode and seed in checkpoint filename to avoid overwriting
        checkpoint_path = Path(self.config.checkpoint_dir) / f"atlas_{self.config.mode}_seed{self.config.seed}_round_{round_num}.pkl"
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
    from config import get_model_config, get_model_hyperparameters
    
    parser = argparse.ArgumentParser(description="Run ATLAS integrated experiment")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("-r", "--rounds", type=int, help="Override number of rounds")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint (path to .pkl file)")
    parser.add_argument("--ablation", choices=["atlas", "atlas_no_laplacian", "fedavg_cluster", "standard_fl", "local_only"], default="atlas",
                       help="Ablation mode: atlas (full), atlas_no_laplacian, fedavg_cluster (task-aware FedAvg), standard_fl (pure FedAvg), local_only")
    parser.add_argument("--lambda-sweep", action="store_true",
                       help="Run lambda sweep over [0.0, 0.01, 0.1, 0.5, 1.0]")
    parser.add_argument("--eta", type=float, help="Override Laplacian regularization strength (lambda)")
    
    # NEW: Model and task configuration for publication experiments
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                       help="Model to use: distilbert, gpt2, gpt2-xl, qwen2.5")
    parser.add_argument("--tasks", type=str, nargs="+", default=['sst2', 'mrpc', 'cola'],
                       help="Tasks to use (space-separated): sst2 mrpc cola qnli mnli")
    parser.add_argument("--clients-per-task", type=int, default=3,
                       help="Number of clients per task")
    parser.add_argument("--samples", type=int, help="Override max_samples_per_client")
    parser.add_argument("--local-epochs", type=int, help="Override local_epochs")
    parser.add_argument("--max-rounds", type=int, help="Maximum rounds for this session (for splitting 30→15+15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Hyperparameter tuning (for breaking performance plateau)
    parser.add_argument("--lr", type=float, help="Override learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, help="Override batch size (default: 16)")
    
    # Fingerprint settings (for large models like LLaMA that need reduced memory)
    parser.add_argument("--fingerprint-samples", type=int, help="Override fingerprint_samples (default: 50)")
    parser.add_argument("--fingerprint-batches", type=int, help="Override fingerprint_batches (default: 20)")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[SEED] Set random seed to {args.seed} for reproducibility")
    
    # Get model-specific hyperparameters
    try:
        model_hparams = get_model_hyperparameters(args.model)
        print(f"\n[MODEL CONFIG] Using optimized hyperparameters for {args.model}")
        print(f"  • Batch size: {model_hparams['batch_size']}")
        print(f"  • Learning rate: {model_hparams['learning_rate']}")
        print(f"  • Local epochs: {model_hparams['local_epochs']}")
        print(f"  • Fingerprint samples: {model_hparams['fingerprint_samples']}")
        print(f"  • Hidden size: {model_hparams['hidden_size']}")
    except Exception as e:
        print(f"[WARNING] Could not load model-specific config for {args.model}: {e}")
        print("[WARNING] Using default hyperparameters")
        model_hparams = {
            'batch_size': 16,
            'learning_rate': 2e-5,
            'local_epochs': 2,
            'fingerprint_samples': 50,
            'fingerprint_batches': 20,
            'max_samples': 3000,
            'lora_ranks': [4, 8, 16, 32],
            'hidden_size': 768
        }

    # Resolve model alias -> actual HF repo id (e.g., 'distilbert' -> 'distilbert-base-uncased')
    try:
        model_repo = get_model_config(args.model)['name']
    except Exception:
        # If mapping fails, fall back to raw args.model
        model_repo = args.model
    
    if args.mode == "quick":
        # Quick test: For debugging and validation
        print("[MODE] Quick test (30-45 min on T4 GPU)")
        config = ATLASConfig(
            model_name=model_repo,
            tasks=args.tasks,
            clients_per_task=args.clients_per_task,
            num_rounds=10,  # Quick validation
            local_epochs=model_hparams['local_epochs'],
            batch_size=model_hparams['batch_size'],
            max_samples_per_client=model_hparams['max_samples'],
            fingerprint_epochs=2,
            fingerprint_batches=model_hparams['fingerprint_batches'],
            fingerprint_samples=model_hparams['fingerprint_samples'],
            learning_rate=model_hparams['learning_rate'],
            rank_candidates=model_hparams['lora_ranks'],
            mode=args.ablation,
            save_every=999,  # Only save final checkpoint
            seed=args.seed
        )
    else:
        # Full experiment: Publication-quality parameters
        print("[MODE] Full experiment - 10 rounds in one shot")
        config = ATLASConfig(
            model_name=model_repo,
            tasks=args.tasks,
            clients_per_task=args.clients_per_task,
            num_rounds=10,  # 10 rounds in one shot
            local_epochs=model_hparams['local_epochs'],
            batch_size=model_hparams['batch_size'],
            max_samples_per_client=model_hparams['max_samples'],
            fingerprint_epochs=2,
            fingerprint_batches=model_hparams['fingerprint_batches'],
            fingerprint_samples=model_hparams['fingerprint_samples'],
            learning_rate=model_hparams['learning_rate'],
            rank_candidates=model_hparams['lora_ranks'],
            mode=args.ablation,
            save_every=999,  # Only save final checkpoint
            seed=args.seed
        )
    
    # Override parameters from CLI
    if args.rounds is not None:
        config.num_rounds = int(args.rounds)
    if args.lr is not None:
        config.learning_rate = args.lr
        print(f"[OVERRIDE] learning_rate = {args.lr}")
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        print(f"[OVERRIDE] batch_size = {args.batch_size}")
    if args.fingerprint_samples is not None:
        config.fingerprint_samples = args.fingerprint_samples
        print(f"[OVERRIDE] fingerprint_samples = {args.fingerprint_samples}")
    if args.fingerprint_batches is not None:
        config.fingerprint_batches = args.fingerprint_batches
        print(f"[OVERRIDE] fingerprint_batches = {args.fingerprint_batches}")
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
        
        # Save sweep results (include ablation mode and seed to avoid overwrites)
        results_path = Path("./results") / f"lambda_sweep_{args.mode}_{config.mode}_seed{config.seed}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(_to_jsonable(sweep_results), f, indent=2)
        
        print(f"\\n[SAVED] Lambda sweep results saved to {results_path}")
        print("\\n[DONE] Lambda sweep complete!")
    
    else:
        # Single run
        trainer = ATLASIntegratedTrainer(config)

        results = trainer.run_full_pipeline(resume_from=args.resume)
        
        # Save final results using the canonical publication naming convention
        # This matches the filenames consumed by `experiments/generate_publication_plots.py`
        # and `experiments/generate_results_tables.py`.
        model_norm = str(config.model_name).replace('/', '_').replace('\\', '_')
        results_path = Path("./results") / f"atlas_{model_norm}_{config.mode}_seed{config.seed}_r{config.num_rounds}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            results_json = {
                # ── Per-round training records ──────────────────────────────────
                'round_metrics':           _to_jsonable(results.get('round_metrics', [])),
                # ── Final per-client scores ─────────────────────────────────────
                'final_accuracies':        _to_jsonable(results.get('final_accuracies', {})),
                'final_canonical':         _to_jsonable(results.get('final_canonical', {})),
                'final_f1':                _to_jsonable(results.get('final_f1', {})),
                # ── Key proof-of-ATLAS metrics ──────────────────────────────────
                # Each task's canonical metric (acc / F1 / Matthews) averaged over
                # clients that share that task.  Compare across ablation modes to
                # show ATLAS outperforms standard_fl on minority tasks (e.g. CoLA).
                'final_task_scores':       _to_jsonable(results.get('final_task_scores', {})),
                'macro_avg_canonical':     _to_jsonable(results.get('macro_avg_canonical', 0.0)),
                # Low spread → clients in same task have converged (personalization worked)
                'personalization_spread':  _to_jsonable(results.get('personalization_spread', 0.0)),
                # Total bytes exchanged across ALL rounds (split activation + LoRA uploads)
                'total_comm_mb':           _to_jsonable(results.get('total_comm_mb', 0.0)),
                # ── Phase information ───────────────────────────────────────────
                'cluster_labels':          _to_jsonable(results.get('cluster_labels', {})),
                'phase1_clustering':       _to_jsonable(results.get('phase1_clustering', {})),
                'phase2_rank_allocation':  _to_jsonable(results.get('phase2_rank_allocation', [])),
                'communication_costs':     _to_jsonable(results.get('communication_costs', {})),
                'time_metrics':            _to_jsonable(results.get('time_metrics', {})),
                'fingerprints':            _to_jsonable(results.get('fingerprints', {})),
                'clustering_metrics':      _to_jsonable(results.get('clustering_metrics', {})),
                'device_configs':          _to_jsonable(results.get('device_configs', {})),
                'layer_importances':       _to_jsonable(results.get('layer_importances', {})),
                'run_metadata':            _to_jsonable(results.get('run_metadata', {})),
                'config':                  asdict(config),
            }
            json.dump(results_json, f, indent=2)
        
        print(f"\n[SAVED] Results saved to {results_path}")
        print("\n[DONE] ATLAS integrated experiment complete!")
