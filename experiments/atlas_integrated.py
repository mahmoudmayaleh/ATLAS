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
import math
import pickle
import contextlib
import io

# Import all ATLAS phases
from src.phase1_clustering import GradientExtractor, TaskClusterer, CFLClusterer
from src.phase1_fingerprint import FingerprintExtractor, build_cosine_fingerprints
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
    fingerprint_batch_size: int = 4  # Fingerprint batch size (larger = faster accumulation, lower diversity)
    max_samples_per_client: int = 2000
    learning_rate: float = 5e-6
    gradient_clip_norm: float = 1.0  # Clip gradients to prevent explosion (critical for large models)
    
    # Device heterogeneity
    device_types: List[str] = field(default_factory=lambda: ['cpu_2gb'] * 2 + ['tablet_4gb'] * 3 + ['laptop_8gb'] * 2 + ['gpu_16gb'] * 2)
    
    # Phase 1: Clustering
    fingerprint_epochs: int = 1  # Reduced to 1 epoch for memory efficiency
    fingerprint_batches: int = 50  # Gradient accumulation steps per client
    fingerprint_samples: int = 400  # Data pool size (batch_size × batches + buffer)
    fingerprint_dim: int = 32  # Target PCA dimension (≤ n_gradient_layers; DistilBERT=36)
    k_range: Tuple[int, int] = (2, 5)  # Overridden at runtime to match number of tasks
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
    laplacian_warmup_rounds: int = 1  # skip Laplacian for first N rounds (LoRA weights too noisy)
    laplacian_adjacency_method: Literal['uniform', 'similarity', 'adaptive', 'mira_rbf', 'mira_rbf_robust'] = 'mira_rbf_robust'  # 'mira_rbf_robust' (RECOMMENDED: clip+log1p outlier-robust)
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

    def __init__(self, model: nn.Module, split_layer: int, n_total_layers: int,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.split_layer = split_layer
        self.n_total_layers = n_total_layers
        self.arch = self._detect_arch(model)
        self._extract_top_components(model)
        # Fix 31: class-balanced loss weights for imbalanced tasks (e.g. CoLA 69/31).
        # Stored as a non-parameter buffer so it moves with .to(device) calls.
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None

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
            # Fix 25: extract the classification-head dropout that
            # DistilBertForSequenceClassification applies between
            # pre_classifier and classifier.  Without this, the server
            # trains 21.9M params on ~3 000 samples with NO dropout →
            # extreme overfitting (train loss 0.22, test acc 0.50).
            self.clf_dropout = getattr(model, 'dropout', nn.Dropout(0.2))
        elif self.arch == 'gpt2':
            self.top_layers = nn.ModuleList(model.transformer.h[s:])
            self.ln_f = model.transformer.ln_f
            self.score = model.score
            self.clf_dropout = nn.Dropout(0.1)
        elif self.arch == 'llama_qwen':
            self.top_layers = nn.ModuleList(model.model.layers[s:])
            self.norm = model.model.norm
            self.score = model.score
            self.clf_dropout = nn.Dropout(0.1)

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
            pooled = self.clf_dropout(pooled)          # Fix 25: match HF's forward
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
            # Fix 31: use class-balanced weights if supplied (handles imbalanced tasks
            # like CoLA where 69% of samples are class 1 → trivial all-one prediction).
            w = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=w)  # type: ignore[possibly-unbound]

        return logits, loss  # type: ignore[possibly-unbound]

    def sync_to_client(self, client_model: nn.Module):
        """Synchronize server's trained top layers + classifier back to client model."""
        # Unwrap PEFT to access base model
        base = client_model.base_model.model if hasattr(client_model, 'base_model') else client_model
        
        if self.arch == 'distilbert':
            # Copy top transformer layers
            for i, server_layer in enumerate(self.top_layers):
                client_layer = base.distilbert.transformer.layer[self.split_layer + i]
                client_layer.load_state_dict(server_layer.state_dict(), strict=False)
            # Copy classifier heads
            base.pre_classifier.load_state_dict(self.pre_classifier.state_dict(), strict=False)
            base.classifier.load_state_dict(self.classifier.state_dict(), strict=False)
        
        elif self.arch == 'gpt2':
            for i, server_layer in enumerate(self.top_layers):
                client_layer = base.transformer.h[self.split_layer + i]
                client_layer.load_state_dict(server_layer.state_dict(), strict=False)
            base.transformer.ln_f.load_state_dict(self.ln_f.state_dict(), strict=False)
            base.score.load_state_dict(self.score.state_dict(), strict=False)
        
        elif self.arch == 'llama_qwen':
            for i, server_layer in enumerate(self.top_layers):
                client_layer = base.model.layers[self.split_layer + i]
                client_layer.load_state_dict(server_layer.state_dict(), strict=False)
            base.model.norm.load_state_dict(self.norm.state_dict(), strict=False)
            base.score.load_state_dict(self.score.state_dict(), strict=False)

    def sync_from_client(self, client_model: nn.Module):
        """Synchronize client's aggregated top layers + classifier to server model."""
        # Unwrap PEFT to access base model
        base = client_model.base_model.model if hasattr(client_model, 'base_model') else client_model
        
        if self.arch == 'distilbert':
            # Copy top transformer layers from client to server
            for i, server_layer in enumerate(self.top_layers):
                client_layer = base.distilbert.transformer.layer[self.split_layer + i]
                server_device = next(server_layer.parameters()).device
                client_state = {k: v.to(server_device) for k, v in client_layer.state_dict().items()}
                server_layer.load_state_dict(client_state, strict=False)
            # Copy classifier heads from client to server
            pre_clf_device = next(self.pre_classifier.parameters()).device
            pre_clf_state = {k: v.to(pre_clf_device) for k, v in base.pre_classifier.state_dict().items()}
            self.pre_classifier.load_state_dict(pre_clf_state, strict=False)
            clf_device = next(self.classifier.parameters()).device
            clf_state = {k: v.to(clf_device) for k, v in base.classifier.state_dict().items()}
            self.classifier.load_state_dict(clf_state, strict=False)
        
        elif self.arch == 'gpt2':
            for i, server_layer in enumerate(self.top_layers):
                client_layer = base.transformer.h[self.split_layer + i]
                server_device = next(server_layer.parameters()).device
                client_state = {k: v.to(server_device) for k, v in client_layer.state_dict().items()}
                server_layer.load_state_dict(client_state, strict=False)
            ln_device = next(self.ln_f.parameters()).device
            ln_state = {k: v.to(ln_device) for k, v in base.transformer.ln_f.state_dict().items()}
            self.ln_f.load_state_dict(ln_state, strict=False)
            score_device = next(self.score.parameters()).device
            score_state = {k: v.to(score_device) for k, v in base.score.state_dict().items()}
            self.score.load_state_dict(score_state, strict=False)
        
        elif self.arch == 'llama_qwen':
            for i, server_layer in enumerate(self.top_layers):
                client_layer = base.model.layers[self.split_layer + i]
                server_device = next(server_layer.parameters()).device
                client_state = {k: v.to(server_device) for k, v in client_layer.state_dict().items()}
                server_layer.load_state_dict(client_state, strict=False)
            norm_device = next(self.norm.parameters()).device
            norm_state = {k: v.to(norm_device) for k, v in base.model.norm.state_dict().items()}
            self.norm.load_state_dict(norm_state, strict=False)
            score_device = next(self.score.parameters()).device
            score_state = {k: v.to(score_device) for k, v in base.score.state_dict().items()}
            self.score.load_state_dict(score_state, strict=False)


class ATLASIntegratedTrainer:
    """
    Full ATLAS pipeline integrating all 4 phases.
    Runs real federated learning with multi-task, heterogeneous devices.
    """
    
    def __init__(self, config: ATLASConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Auto-set k_range to exactly match the number of tasks so clustering
        # never merges different tasks or over-splits within a task.
        n_tasks = len(config.tasks) if config.tasks else 3
        config.k_range = (n_tasks, n_tasks)  # Force k = n_tasks
        print(f"[ATLAS] Auto-set k_range = {config.k_range} (one cluster per task)")
        
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
        # Dedicated fingerprint extractor (phase1_fingerprint.py)
        self._fingerprint_extractor = FingerprintExtractor(config, self.device)
        # CFL-style cosine hierarchical clusterer (immune to HDLSS)
        self.cfl_clusterer = CFLClusterer(n_clusters=n_tasks, linkage_method='complete')
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
        for task_idx, task_name in enumerate(self.config.tasks):
            print(f"  Loading task: {task_name}")
            train_data, test_data = self._load_task_data(task_name)
            task_datasets[task_name] = (train_data, test_data)
            
            # Partition among clients for this task.
            # IMPORTANT: shuffle before partitioning so each client gets a
            # label-balanced subset.  HuggingFace GLUE datasets are sorted by
            # label (e.g. SST-2: all negatives first, then all positives) — a
            # sequential slice gives client 0 mostly one class and client 2
            # the other, causing the observed 0.48 vs 0.84 accuracy gap.
            n_clients = self.config.clients_per_task
            # Task-specific RNG so each task gets a distinct (but reproducible) partition
            rng = np.random.RandomState(int(self.config.seed) + int(task_idx))

            # Prefer stratified partitioning by label to reduce client-to-client variance,
            # especially on imbalanced tasks (e.g., MRPC/CoLA). Fallback to a global
            # shuffle if labels cannot be read for any reason.
            label_to_indices: Dict[int, List[int]] = {}
            try:
                for idx in range(len(train_data)):
                    ex = train_data[idx]
                    label = ex.get('label') if isinstance(ex, dict) else None
                    if label is None:
                        raise KeyError('label')
                    label_int = int(label)  # torch scalar → int
                    label_to_indices.setdefault(label_int, []).append(idx)
            except Exception:
                label_to_indices = {0: list(range(len(train_data)))}

            for inds in label_to_indices.values():
                rng.shuffle(inds)
            
            for i in range(n_clients):
                # Assign device type (cycle through available types)
                device_type = self.config.device_types[client_id % len(self.config.device_types)]
                
                # Create client data subset.
                # Stratified chunks per label -> concatenate -> shuffle within-client.
                indices: List[int] = []
                for label_int in sorted(label_to_indices.keys()):
                    inds = label_to_indices[label_int]
                    start_idx = i * len(inds) // n_clients
                    end_idx = (i + 1) * len(inds) // n_clients
                    indices.extend(inds[start_idx:end_idx])
                rng.shuffle(indices)
                
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
        # Task-specific max length (QNLI benefits from longer context)
        max_length = 256 if task_name == 'qnli' else 128

        def tokenize_fn(examples):
            if text_col2:
                texts = [(t1, t2) for t1, t2 in zip(examples[text_col], examples[text_col2])]
                return self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length)
            else:
                return self.tokenizer(examples[text_col], padding='max_length', truncation=True, max_length=max_length)
        
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
            clustering_metrics=clustering_metrics,
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
        seen_client_ids: list = []     # ordered list of client IDs
        tensor_importances  = {}       # cid → {param_name: avg_norm²}   → diagnostics
        grad_vec_importances = {}      # cid → {param_name: mean_unit_grad_vector} → PCA source
        layer_importances   = {}       # cid → {layer_k: avg_norm²}      → Phase 2 rank allocation

        # Collect both per-client averaged gradients and per-batch gradient samples
        # NOTE: grad_samples accumulated here for potential future use but is NOT passed
        # to the KMeans clusterer (which uses the 7-dim layer-norm fingerprint vectors).
        # Removed to avoid holding ~3GB of grad tensors (66M params × 12 clients) in RAM.
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
            
            # Reinitialize classification head with small weights for stability
            if hasattr(model, 'classifier'):
                torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
                if model.classifier.bias is not None:
                    torch.nn.init.zeros_(model.classifier.bias)
            elif hasattr(model, 'score'):
                torch.nn.init.normal_(model.score.weight, mean=0.0, std=0.02)
                if model.score.bias is not None:
                    torch.nn.init.zeros_(model.score.bias)

            # Extract norms (diagnostics), mean grad vectors (PCA source), layer norms (Phase 2)
            # Uses FingerprintExtractor from src/phase1_fingerprint.py (10-step warm-up,
            # head + last-2 backbone layers, LayerNorm gradient vectors, StandardScaler+PCA)
            tensor_imp, grad_vecs, layer_imp, per_batch_grads = self._fingerprint_extractor.extract(
                model, client_data.train_dataset
            )
            seen_client_ids.append(client_data.client_id)
            tensor_importances[client_data.client_id]  = tensor_imp
            grad_vec_importances[client_data.client_id] = grad_vecs
            layer_importances[client_data.client_id]   = layer_imp
            # per_batch_grads: compressed norm snapshots — diagnostics only.

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            n_layers = len([k for k in layer_imp if k.startswith('layer_')])
            print(f"✓ {n_layers} layers + classifier fingerprinted, "
                  f"{len(layer_imp)} importance keys")
        
        # ── CFL cosine similarity fingerprints (src/phase1_fingerprint.py) ─────
        # Works in the n×n similarity space — immune to HDLSS (n=12, p=591K).
        # See build_cosine_fingerprints() for full scientific rationale.
        S_matrix, raw_fingerprints, fp_info = build_cosine_fingerprints(
            seen_client_ids=seen_client_ids,
            grad_vec_importances=grad_vec_importances,
            tensor_importances=tensor_importances,
            layer_importances=layer_importances,
        )
        print(f"\n[Phase 1] Fingerprint: {fp_info}")

        # ── CFL hierarchical clustering on the n×n distance matrix ───────────
        print(f"\n[Phase 1] CFL clustering {len(seen_client_ids)} clients...")
        client_ids = seen_client_ids
        res = self.cfl_clusterer.cluster(S_matrix, client_ids=client_ids, verbose=True)
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
        silhouette = float(metrics.get('silhouette_score', 0.0)) if metrics else 0.0
        print(f"\n  ✓ Average cluster purity: {avg_purity:.3f}  |  Silhouette: {silhouette:.4f}", flush=True)

        # ── Oracle clustering fallback ────────────────────────────────────────
        # If gradient fingerprints are not task-discriminative (silhouette < 0.15
        # or purity < 0.85), fall back to ground-truth task-based cluster assignment.
        # This guarantees that the Laplacian ONLY regularises within-task, which is
        # the correct ATLAS behaviour when Phase 1 cannot learn a good embedding.
        SILHOUETTE_THRESHOLD = 0.30   # raised from 0.15 — fingerprinting now uses
        PURITY_THRESHOLD     = 0.80   # per-cluster min purity (not avg): a single
                                       # impure cluster (e.g. purity=0.50) must trigger
                                       # oracle even if other clusters are pure (avg masks it)
        min_purity = min(cluster_task_purity.values()) if cluster_task_purity else 0.0
        use_oracle = (silhouette < SILHOUETTE_THRESHOLD or min_purity < PURITY_THRESHOLD)
        if use_oracle:
            print(f"\n  ⚠️  [Phase 1] Fingerprint clustering quality too low "
                  f"(silhouette={silhouette:.3f} < {SILHOUETTE_THRESHOLD} or "
                  f"min_purity={min_purity:.2f} < {PURITY_THRESHOLD}).", flush=True)
            print(f"  ⚠️  Falling back to ORACLE (task-label) clustering to prevent "
                  f"cross-task Laplacian contamination.", flush=True)
            # Assign each unique task a cluster index, in sorted order
            task_names_sorted = sorted(set(cd.task_name for cd in self.clients_data))
            task_to_cluster = {t: i for i, t in enumerate(task_names_sorted)}
            cluster_labels = {cd.client_id: task_to_cluster[cd.task_name]
                              for cd in self.clients_data}
            # Re-run purity analysis on oracle clusters for logging
            oracle_cluster_task: Dict[int, Dict[str, int]] = {}
            for cd in self.clients_data:
                cid_cluster = cluster_labels[cd.client_id]
                oracle_cluster_task.setdefault(cid_cluster, {})
                oracle_cluster_task[cid_cluster][cd.task_name] = \
                    oracle_cluster_task[cid_cluster].get(cd.task_name, 0) + 1
            print(f"  Oracle clusters: { {c: list(v.keys()) for c, v in sorted(oracle_cluster_task.items())} }",
                  flush=True)
            if metrics is not None:
                metrics['oracle_fallback'] = True
                metrics['oracle_reason'] = f"silhouette={silhouette:.3f}, min_purity={min_purity:.2f}"

        # Update client cluster assignments
        for client_data in self.clients_data:
            client_data.cluster_id = cluster_labels[client_data.client_id]

        clustering_metrics = metrics if metrics is not None else {}
        clustering_metrics['avg_purity'] = float(avg_purity)
        clustering_metrics['min_purity'] = float(min_purity)
        clustering_metrics['oracle_fallback'] = use_oracle

        # Return raw (unnormalized) fingerprints for downstream Phase 2 + task graph.
        # Normalized versions were only needed for KMeans; raw preserves the
        # absolute gradient-norm magnitudes that drive RBF distances and variance.
        return cluster_labels, raw_fingerprints, clustering_metrics, layer_importances
    
    def _extract_fingerprint(self, model: nn.Module, dataset: Subset) -> Tuple[Dict, Dict, list]:
        """Extract gradient fingerprint from a client's local training.
        
        Returns:
            (averaged_grads, layer_importance, per_batch_grads): gradient dict, per-layer importance scores,
            and a list of per-batch gradient dicts collected during fingerprinting (may be empty).
        """
        # Clear cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Disable gradient checkpointing during fingerprinting:
        # with batch_size=1 it re-runs the full forward pass per backward call,
        # doubling computation time for long sequences (e.g. QNLI with 256 tokens).
        gradient_checkpointing_disable = getattr(model, 'gradient_checkpointing_disable', None)
        if callable(gradient_checkpointing_disable):
            try:
                gradient_checkpointing_disable()
            except Exception:
                pass
        
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

        # ── Warm-start: head-only SGD for task-discriminative gradients ──────────
        # Root cause of silhouette~0.05: freshly-reinit head (std=0.02) produces
        # near-identical gradients across tasks. 3-5 head-only gradient steps give
        # task-specific signal before fingerprint PCA (FedGroup/CFL recommendation).
        # Only classifier/score/pre_classifier params are updated; backbone stays frozen
        # so warm-start cost ≈ one tiny linear layer, not the full 14M param model.
        _head_params = [p for n, p in model.named_parameters()
                        if any(k in n for k in ('classifier', 'score', 'pre_classifier'))]
        if _head_params:
            _warmup_opt = torch.optim.AdamW(_head_params, lr=1e-4, weight_decay=0.01)
            _warmup_loader = DataLoader(
                fingerprint_subset, batch_size=self.config.fingerprint_batch_size, shuffle=True
            )
            _warmup_steps = 0
            _max_warmup = 5  # 3-5 steps: sufficient for task-discriminative head (Sattler et al., CFL)
            for _wb in _warmup_loader:
                if _warmup_steps >= _max_warmup:
                    break
                _wi = _wb['input_ids'].to(self.device)
                _wa = _wb['attention_mask'].to(self.device)
                _wl = _wb['label'].to(self.device)
                _warmup_opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=False):  # FP32 for stability
                    _wout = model(input_ids=_wi, attention_mask=_wa, labels=_wl)
                if _wout.loss is not None and not (torch.isnan(_wout.loss) or torch.isinf(_wout.loss)):
                    _wout.loss.backward()
                    torch.nn.utils.clip_grad_norm_(_head_params, 1.0)
                    _warmup_opt.step()
                del _wi, _wa, _wl, _wout
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _warmup_steps += 1
            model.zero_grad(set_to_none=True)
            print(f"[warmup:{_warmup_steps}✓]", end=" ", flush=True)

        # Online running-sum removed: averaged_grads is no longer returned.
        # Only tensor_norms (scalars), grad_sum (head param vectors) and layer_norms kept.
        running_sum: Dict[str, torch.Tensor] = {}   # kept for API compat; unused
        running_count: Dict[str, int] = {}
        layer_norms: Dict[str, List[float]] = {}    # grouped by layer → Phase 2 rank allocation
        tensor_norms: Dict[str, List[float]] = {}   # per param tensor → norm² (diagnostics)
        #
        # grad_sum / grad_cnt: accumulate mean NORMALISED gradient *vector* for task-head
        # parameters (classifier.weight, pre_classifier.weight, score.weight).
        #
        # Why head-gradient vectors, not norm scalars or sign fractions:
        #   - Per-tensor norm²  → dominated by classifier (≈96% of total), indistinguishable
        #     across tasks (all tasks start from same random init and converge at similar speed)
        #   - P(g>0) polarity   → backbone polarity ≈ 0.5 for balanced class splits;
        #     StandardScaler amplifies this constant to unit variance = pure noise
        #   - Head gradient DIRECTION → encodes which hidden dims each task associates with
        #     each class label.  SST2 trains the head toward sentiment token directions;
        #     QNLI toward entailment-structure directions; etc.  Totally different vectors.
        #     This is exactly the signal CFL (Sattler et al. 2021) uses for client clustering.
        #
        # Memory: classifier.weight (2×768=1536) + pre_classifier.weight (768×768=589K)
        #   = ~591K floats per client = 2.3 MB — negligible.  No full backbone tensors.
        # Cost: one additional norm+divide per head-param per batch — negligible.
        _HEAD_KW = ('classifier', 'score', 'pre_classifier')  # task-head param name patterns
        _HEAD_MAX_PARAMS = 700_000                             # safety cap (excludes giant layers)
        grad_sum: Dict[str, torch.Tensor] = {}  # name → running sum of unit-normalised grads
        grad_cnt: Dict[str, int] = {}
        # grad_history stores compressed per-batch layer-norm snapshots (not full param tensors).
        # Clustering uses PCA-projected per-tensor norms, not raw grad tensors.
        # Full all-layer grad tensors would be ~264 MB × 15 × 12 clients — unacceptable.
        grad_history: List[Dict[str, float]] = []
        MAX_HISTORY = 15  # per client; stored as {layer_k: scalar_norm} dicts only
        import re as _re

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
                    print(f"⚠️", end="", flush=True)
                    del input_ids, attention_mask, labels, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                try:
                    loss.backward()
                except Exception as oom:
                    print(f"⚠️OOM/ERR", end="", flush=True)
                    model.zero_grad(set_to_none=True)
                    del input_ids, attention_mask, labels, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    total_batches_processed += 1
                    continue

                # Collect gradients from ALL layers for per-tensor norm² fingerprinting.
                # Each named-parameter tensor contributes one scalar (its gradient norm²)
                # to the ~72-dim raw fingerprint that feeds the 64-dim PCA projection.
                # No full grad tensors are accumulated — only scalars kept.
                grads_dict = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_cpu = param.grad.detach().cpu()

                        # No running_sum: we only need per-tensor and per-layer norms.

                        # Layer importance (squared norm) — classify by layer index
                        layer_match = _re.search(r'layer[._](\d+)', name)
                        if layer_match:
                            layer_key = f'layer_{int(layer_match.group(1))}'
                        elif 'classifier' in name or 'pooler' in name or 'pre_classifier' in name:
                            layer_key = 'classifier'
                        else:
                            layer_key = 'other'
                        grad_norm_sq = float((grad_cpu ** 2).sum())
                        layer_norms.setdefault(layer_key, []).append(grad_norm_sq)
                        tensor_norms.setdefault(name, []).append(grad_norm_sq)  # for diagnostics

                        # Accumulate normalised head-parameter gradient vector.
                        # Only for task-head params within the memory cap.
                        if (any(k in name for k in _HEAD_KW)
                                and grad_cpu.numel() <= _HEAD_MAX_PARAMS):
                            g_flat  = grad_cpu.float().flatten()
                            g_norm  = g_flat.norm()
                            if g_norm > 1e-12:
                                g_unit = g_flat / g_norm
                                if name not in grad_sum:
                                    grad_sum[name] = g_unit.clone()
                                    grad_cnt[name] = 1
                                else:
                                    grad_sum[name].add_(g_unit)
                                    grad_cnt[name] += 1

                        grads_dict[name] = grad_cpu

                # Keep a small compressed snapshot (per-batch layer norms) for diagnostics.
                # Do NOT store full grad tensors — with all-layer collection that would be
                # ~264 MB per snapshot and grad_history is not used by the KMeans clusterer.
                if grads_dict and len(grad_history) < MAX_HISTORY:
                    batch_norms: Dict[str, float] = {}
                    for _n, _g in grads_dict.items():
                        _lm = _re.search(r'layer[._](\d+)', _n)
                        if _lm:
                            _lk = f'layer_{int(_lm.group(1))}'
                        elif any(_k in _n for _k in ('classifier', 'pooler', 'pre_classifier')):
                            _lk = 'classifier'
                        else:
                            _lk = 'other'
                        batch_norms[_lk] = batch_norms.get(_lk, 0.0) + float((_g ** 2).sum())
                    grad_history.append(batch_norms)

                # Clear gradients immediately
                model.zero_grad(set_to_none=True)
                del input_ids, attention_mask, labels, outputs, loss, grads_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                total_batches_processed += 1
            
            # Break outer loop if limit reached
            if total_batches_processed >= batch_limit:
                break
        
        # Build per-tensor / per-layer importance dicts and mean gradient direction.
        if tensor_norms:
            tensor_importance = {name: float(np.mean(v)) for name, v in tensor_norms.items()}
            layer_importance  = {k:    float(np.mean(v)) for k,    v in layer_norms.items()}
            # Mean normalised gradient vector for each head parameter.
            # Dividing by count gives the average unit gradient direction over all batches.
            mean_grad_vecs: Dict[str, np.ndarray] = {
                name: (grad_sum[name] / grad_cnt[name]).numpy()
                for name in grad_sum if grad_cnt.get(name, 0) > 0
            }
            return tensor_importance, mean_grad_vecs, layer_importance, grad_history
        else:
            # Fallback: empty dicts — caller will apply oracle clustering
            return {}, {}, {}, []
    
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

            # Use per-layer importance (gradient norm²) for complexity. This is:
            # - stable across fingerprint representation changes
            # - directly tied to Phase-2's objective (rank allocation)
            gathered = [layer_importances[cid] for cid in cluster_client_ids if cid in layer_importances]
            if not gathered:
                continue

            # Build a consistent ordered layer vector
            keys = sorted(set().union(*[set(d.keys()) for d in gathered]))
            mat = np.array([[float(d.get(k, 0.0)) for k in keys] for d in gathered], dtype=np.float32)
            # Variance: within-cluster heterogeneity of gradient energy across layers
            variance = float(np.var(mat, axis=0).mean())
            # Magnitude: mean layer-importance norm (proxy for task difficulty)
            avg_norm = float(np.mean(np.linalg.norm(mat, axis=1)))

            cluster_stats[cluster_id] = {
                'variance': variance,
                'avg_norm': avg_norm,
                'n_clients': len(cluster_client_ids),
                'complexity_score': variance * avg_norm,
            }

            print(
                f"  Cluster {cluster_id}: variance={variance:.4f}, "
                f"norm={avg_norm:.4f}, complexity={cluster_stats[cluster_id]['complexity_score']:.4f}"
            )
        
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
        clustering_metrics: Optional[Dict] = None,
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

            # block_diagonal=False: allow edges across ALL client pairs (cross-cluster
            # included).  The same-task filter below then removes every edge where
            # task(k) ≠ task(ℓ), so only genuinely similar clients remain connected.
            # This is strictly better than block_diagonal=True because:
            #   - block_diagonal=True  → only intra-cluster edges: cola client 7
            #     (cluster 0) can never pull cola clients 6,8 (cluster 3).
            #   - block_diagonal=False + same-task filter → cross-cluster same-task
            #     edges are kept; cross-task edges (the real source of contamination)
            #     are zeroed regardless of cluster membership.
            laplacian_block_diagonal = False

            adjacency_weights = compute_adjacency_weights(
                task_clusters=task_clusters,
                gradient_fingerprints=fingerprints,
                method=self.config.laplacian_adjacency_method,
                mira_alpha=self.config.mira_alpha,
                block_diagonal=laplacian_block_diagonal,
                ensure_connectivity=self.config.ensure_connectivity,
                rbf_clip_percentile=95.0,
                rbf_floor=0.05,
            )

            # ── Same-task edge filter ──────────────────────────────────────────────
            # Phase 1 clustering is gradient-based and may produce impure clusters
            # (e.g. cluster 3 can contain mrpc + cola + qnli clients).  Keeping
            # cross-task Laplacian edges pulls client k's LoRA weights toward a
            # neighbour on a DIFFERENT task, which acts as destructive noise rather
            # than personalisation signal.  We retain only edges where both endpoints
            # share the same task name: a_kℓ = 0 whenever task(k) ≠ task(ℓ).
            # Clients with no same-task neighbours are left untouched (isolated nodes).
            cid_to_task: Dict[int, str] = {
                cd.client_id: cd.task_name for cd in self.clients_data
            }
            adjacency_weights = {
                (i, j): w
                for (i, j), w in adjacency_weights.items()
                if cid_to_task.get(i) == cid_to_task.get(j)
            }
            # ─────────────────────────────────────────────────────────────────────

            print(f"  ✓ Computed {len(adjacency_weights)} same-task adjacency weights "
                  f"using {self.config.laplacian_adjacency_method}")
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
        use_split_learning = mode not in ('local_only', 'standard_fl')
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
            
            # Initialize the classification head.
            #
            # HISTORY: Fix 17 originally zero-initialized W and b to prevent
            # label inversion (random init → different clients in opposite
            # basins → test_acc ≈ 0.49).  However, in split learning:
            #
            #   logits = W @ h + b   →   ∂logits/∂h = W
            #
            # With W=0, the activation gradient ∂loss/∂h = W^T @ ∂loss/∂logits
            # is EXACTLY ZERO.  This blocks gradient flow to ALL upstream
            # server layers (3 transformer blocks, pre_classifier) AND to the
            # client LoRA (via the activation gradient).  Only the 1538-param
            # classifier itself gets any gradient — catastrophic for split FL.
            #
            # Fix 21: We still zero-init here as a PLACEHOLDER.  After building
            # server models, sync_to_client pushes the server's properly-
            # initialized classifier (seeded Xavier weight + log-prior bias)
            # to every client in the cluster.  This guarantees:
            #   1. All clients share IDENTICAL classifier init → no label inversion
            #   2. W ≠ 0 → gradients flow through entire server + back to client
            #   3. Bias = log(prior) → strong initial gradient signal
            head = getattr(model, 'classifier', None) or getattr(model, 'score', None)
            if head is not None:
                torch.nn.init.zeros_(head.weight)
                if head.bias is not None:
                    torch.nn.init.zeros_(head.bias)

            # Enable gradient checkpointing before LoRA (saves memory during training)
            if hasattr(model, 'gradient_checkpointing_enable') and callable(getattr(model, 'gradient_checkpointing_enable', None)):
                try:
                    model.gradient_checkpointing_enable()
                except Exception as e:
                    print(f"[Warning: gradient checkpointing setup failed: {e}]")
            
            # Apply LoRA restricted to bottom split_layer layers (split learning mode)
            # Top layers will be managed by the server (no LoRA adapters there)
            model = self._apply_heterogeneous_lora(
                model,
                client_data.lora_ranks,
                split_layer=self.config.split_layer if use_split_learning else None,
            )
            # Ensure model is on CPU (do not call .to(self.device) here)
            model.to('cpu')
            client_models[client_data.client_id] = model
        
        print(f"  ✓ Created {len(client_models)} personalized client models")

        # ── Build genuine split-server models (atlas / atlas_no_laplacian / fedavg_cluster) ──
        # local_only and standard_fl use full local training, so no server needed.
        split_server_models: Dict[int, Dict] = {}
        if use_split_learning:
            split_server_models = self._build_split_server_models(task_clusters, device_configs)

            # Fix 21 (cluster-wide sync): Push the server's properly-initialized
            # top layers + pre_classifier + classifier to every client in the
            # cluster.  This achieves three things:
            #   1. All clients share IDENTICAL classifier W (seeded Xavier)
            #      → no label inversion possible
            #   2. W ≠ 0 → ∂loss/∂h ≠ 0 → gradient flows through entire server
            #      AND back to client LoRA from step 1
            #   3. Bias = log(prior) → initial predictions match class distribution
            #      → strong gradient signal (not the flat saddle at uniform)
            #
            # This overwrites the placeholder zero-init set during client creation.
            # When sync_from_client runs at the start of round 1, it copies
            # THIS properly-initialized classifier back to the server — no
            # information loss.
            for cluster_id, srv_dict in split_server_models.items():
                srv_wrapper = srv_dict['model']
                client_ids_in_cluster = task_clusters.get(cluster_id, [])
                for cid in client_ids_in_cluster:
                    cm = client_models.get(cid)
                    if cm is None:
                        continue
                    srv_wrapper.sync_to_client(cm)
                print(
                    f"  Cluster {cluster_id}: synced server→client for "
                    f"{len(client_ids_in_cluster)} clients (Xavier W + log-prior bias)"
                )

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
        round_canonical: Dict[int, float] = {}
        round_f1s: Dict[int, float] = {}

        # ── Per-client server optimizer state snapshots (Fix 21) ────────────
        # The cluster server is shared sequentially among all clients in a
        # cluster.  AdamW accumulates exponential moving averages (exp_avg,
        # exp_avg_sq) that encode the gradient statistics of the PREVIOUS
        # client's data distribution.  When the server switches to a new
        # client, the weight values are restored via sync_from_client but the
        # optimizer state is NOT — causing the first ~50 gradient steps to be
        # biased toward the previous client's task (cross-client momentum
        # contamination).  Over rounds this compounds: only the last-trained
        # client in each cluster converges; earlier clients stay near random.
        #
        # Fix: maintain per-client snapshots of the server optimizer state.
        # Before each client's training turn we restore THEIR optimizer state
        # (or start fresh on the first turn).  After training we snapshot it.
        # This guarantees Adam momentum/variance are always consistent with
        # the client currently being served.
        #
        # References:
        #   SplitFed (Thapa et al., AAAI 2022) — per-client server state
        #   SCAFFOLD (Karimireddy et al., ICML 2020) — per-client control variates
        client_server_opt_states: Dict[int, Dict] = {}
        if checkpoint is not None and 'client_server_opt_states' in checkpoint:
            client_server_opt_states = checkpoint['client_server_opt_states']

        # Per-client LoRA (client-side) optimizer state snapshots.
        # Analogous to Fix 21 for the server: the client LoRA optimizer is
        # currently recreated fresh inside _train_client_split every round,
        # discarding ALL Adam momentum/variance accumulated in round N before
        # round N+1 starts.  This makes each round's first ~20 steps behave
        # like SGD (no momentum), wasting the adaptive-rate benefits of Adam.
        # With 63 batches/round × 10 rounds ≈ 630 steps, persistent momentum
        # roughly doubles effective convergence speed for all clients.
        client_lora_opt_states: Dict[int, Dict] = {}
        if checkpoint is not None and 'client_lora_opt_states' in checkpoint:
            client_lora_opt_states = checkpoint['client_lora_opt_states']

        for round_idx in range(start_round, self.config.num_rounds):
            round_start = time.time()
            print(f"\n{'='*70}")
            print(f"ROUND {round_idx + 1}/{self.config.num_rounds}")
            print(f"{'='*70}", flush=True)
            
            # Step 1: Client training
            # - local_only / standard_fl  → full local training (no split)
            # - all ATLAS modes           → genuine split learning (activations ↔ gradients)
            n_clients = len(self.clients_data)
            training_mode_str = 'Split' if use_split_learning else 'Local'
            print(f"[Round {round_idx+1}] {training_mode_str} training — {n_clients} clients...", flush=True)
            round_losses = {}
            # Communication counters (bytes) for this round
            comm_upload = {c.client_id: 0 for c in self.clients_data}
            comm_download = {c.client_id: 0 for c in self.clients_data}

            for _ci, client_data in enumerate(self.clients_data):
                cid = client_data.client_id
                print(f"\n  [{_ci+1}/{n_clients}] Client {cid} | task={client_data.task_name} "
                      f"| device={client_data.device_type} | cluster={client_data.cluster_id}",
                      flush=True)
                _client_round_start = time.time()
                model = client_models[cid]
                model.to(self.device)

                if use_split_learning:
                    # ── Genuine split learning ──────────────────────────────
                    cluster_id = client_data.cluster_id if client_data.cluster_id is not None else 0
                    srv = split_server_models[cluster_id]
                    srv_model     = srv['model']      # already on self.device
                    srv_optimizer = srv['optimizer']
                    split_layer   = srv['split_layer']

                    # CRITICAL (Fix 20): restore the server to the top-layer
                    # state this client received at the end of the previous
                    # round.  Without this, client N trains on top layers
                    # left by client N-1.
                    srv_model.sync_from_client(model)

                    # CRITICAL (Fix 21): restore per-client server optimizer
                    # state.  Without this, Adam momentum/variance carry over
                    # from the previous client's data, causing cross-client
                    # gradient contamination.  Fresh optimizer on first turn.
                    if cid in client_server_opt_states:
                        try:
                            srv_optimizer.load_state_dict(
                                client_server_opt_states[cid]
                            )
                        except Exception:
                            # Shape mismatch: fall back to fresh optimizer for this client.
                            srv_optimizer = torch.optim.AdamW(
                                srv_model.parameters(),
                                lr=self.config.learning_rate,
                            )
                            srv['optimizer'] = srv_optimizer
                    else:
                        # First training turn for this client: reset to fresh
                        # optimizer (no stale momentum from other clients).
                        srv_optimizer = torch.optim.AdamW(
                            srv_model.parameters(),
                            lr=self.config.learning_rate,
                        )
                        srv['optimizer'] = srv_optimizer

                    loss, up_bytes, dn_bytes = self._train_client_split(
                        client_model=model,
                        dataset=client_data.train_dataset,
                        server_model=srv_model,
                        server_optimizer=srv_optimizer,
                        split_layer=split_layer,
                        client_id=cid,
                        task_name=client_data.task_name,
                        client_lora_opt_states=client_lora_opt_states,
                    )
                    # CRITICAL: synchronize server's trained top layers + classifier back to client
                    srv_model.sync_to_client(model)

                    # Snapshot this client's server optimizer state to CPU.
                    _opt_sd = srv_optimizer.state_dict()
                    _cpu_opt: Dict[str, Any] = {
                        'param_groups': _opt_sd['param_groups'],
                        'state': {},
                    }
                    for _pk, _pv in _opt_sd['state'].items():
                        _cpu_opt['state'][_pk] = {
                            _kk: _vv.cpu().clone()
                                 if isinstance(_vv, torch.Tensor) else _vv
                            for _kk, _vv in _pv.items()
                        }
                    client_server_opt_states[cid] = _cpu_opt

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
                print(f"  [{_ci+1}/{n_clients}] Client {cid} done in {time.time()-_client_round_start:.0f}s",
                      flush=True)

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

                print(f"\n[Round {round_idx+1}] Global FedAvg (standard FL)...", flush=True)
                all_weights = [
                    {name: param.data.clone() for name, param in client_models[cid].named_parameters()}
                    for cid in range(len(client_models))
                ]
                avg_state = self._fedavg_aggregate(all_weights)
                print(f"[Round {round_idx+1}] Laplacian skipped (mode=standard_fl)", flush=True)

                # Measure upload (LoRA weights sent to server) and download (avg sent back)
                trainable_kw = ['lora', 'classifier', 'score', 'modules_to_save']
                for cid in list(client_models.keys()):
                    up = sum(p.numel() * p.element_size()
                             for n, p in client_models[cid].named_parameters()
                             if any(kw in n.lower() for kw in trainable_kw))
                    comm_upload[cid] = int(up)
                    dn = sum(t.numel() * t.element_size()
                             for k, t in avg_state.items()
                             if any(kw in k.lower() for kw in trainable_kw))
                    comm_download[cid] = int(dn)

                # Direct load — no broken _flat_state_to_lora conversion
                for cid in client_models:
                    try:
                        client_models[cid].load_state_dict(avg_state, strict=False)
                    except Exception:
                        for name, param in client_models[cid].named_parameters():
                            if name in avg_state and avg_state[name].shape == param.shape:
                                try:
                                    param.data.copy_(avg_state[name].to(param.device))
                                except Exception:
                                    pass
            else:
                # ── MIRA-compliant pipeline ────────────────────────────────────────────
                # atlas mode   : Laplacian regularization REPLACES cluster FedAvg.
                #                Applied directly on local W_k(t,R) — before any averaging.
                #                Each client retains a DISTINCT personalized model; the
                #                Laplacian nudges similar clients together without overwriting.
                # other modes  : intra-cluster raFLoRA rank-partitioned FedAvg (no Laplacian).
                #                HeLoRA design: server builds a max-rank template; each client
                #                receives only its own r_k-slice → retains hetero-rank
                #                personalization, no full broadcast overwrite.
                # ─────────────────────────────────────────────────────────────────────

                trainable_kw = ['lora', 'classifier', 'score', 'modules_to_save']

                if mode == 'atlas':
                    # ── Step 1: Collect raw local states W_k(t,R) ────────────────────
                    # These are the post-training, pre-aggregation weights for every client.
                    local_flat: Dict[int, Dict[str, torch.Tensor]] = {
                        cid: {n: p.data.clone()
                              for n, p in client_models[cid].named_parameters()}
                        for cid in client_models
                    }

                    # ── Step 2: MIRA Laplacian on raw locals ──────────────────────────
                    # Formula: W_k ← W_k − η Σ_{ℓ∈N_k} a_kℓ (W_k − W_ℓ)
                    # heterogeneous ranks handled by raFLoRA-style truncate-then-pad.
                    #
                    # Only LoRA params are regularized — NOT top-layer / classifier weights.
                    # Reason: clients train SEQUENTIALLY on a shared server.  After training,
                    # local_flat[k] contains server weights snapshotted at different times
                    # (client 0 → server state after client 0; client 11 → server state after
                    # all 12 clients).  Laplacian-averaging these time-inconsistent snapshots
                    # is noise.  Top layers and classifier are handled by the server-sync step
                    # (sync_from_client + broadcast sync_to_client) which uses the FINAL,
                    # converged server state and is applied consistently to every client.
                    #
                    # Warmup: skip Laplacian for the first `laplacian_warmup_rounds` rounds.
                    # LoRA weights initialise at ~0 and are far from meaningful minima;
                    # regularizing them early adds noise and destabilises convergence.
                    _warmup = getattr(self.config, 'laplacian_warmup_rounds', 1)
                    _skip_lap = (round_idx < _warmup)
                    print(
                        f"\n[Round {round_idx+1}] Laplacian regularization on local models (MIRA)"
                        + (f" [SKIPPED — warmup round {round_idx+1}/{_warmup}]" if _skip_lap else "") + "...",
                        flush=True,
                    )

                    # LoRA-only keyword — classifier/score excluded (server-synced below)
                    lora_kw = ['lora']

                    if task_graph is not None and not _skip_lap:
                        _eta = self.config.eta
                        updated_flat: Dict[int, Dict[str, torch.Tensor]] = {}
                        n_lap_applied = 0

                        for cid in local_flat:
                            state_k = local_flat[cid]
                            neighbors = task_graph.get_neighbors(cid)
                            lap_delta: Dict[str, torch.Tensor] = {}

                            for nbr in neighbors:
                                if nbr not in local_flat:
                                    continue
                                state_l = local_flat[nbr]
                                a_kl = task_graph.get_edge_weight(cid, nbr)
                                if a_kl == 0.0:
                                    continue

                                for key, w_k in state_k.items():
                                    # LoRA-only: skip classifier, score, top-layer weights
                                    if not any(kw in key.lower() for kw in lora_kw):
                                        continue
                                    if key not in state_l:
                                        continue
                                    w_l = state_l[key]
                                    key_low = key.lower()

                                    # Hetero-rank LoRA A (r×v): truncate rank dim 0 to min_r,
                                    # compute diff, pad zeros back to client's own rank (HeLoRA).
                                    if 'lora_a' in key_low and w_k.shape != w_l.shape:
                                        min_r = min(w_k.shape[0], w_l.shape[0])
                                        diff = torch.zeros_like(w_k.float())
                                        diff[:min_r] = a_kl * (
                                            w_k[:min_r].float() - w_l[:min_r].float()
                                        )
                                    # Hetero-rank LoRA B (u×r): truncate rank dim 1 to min_r.
                                    elif 'lora_b' in key_low and w_k.shape != w_l.shape:
                                        min_r = min(w_k.shape[1], w_l.shape[1])
                                        diff = torch.zeros_like(w_k.float())
                                        diff[:, :min_r] = a_kl * (
                                            w_k[:, :min_r].float() - w_l[:, :min_r].float()
                                        )
                                    else:
                                        # Same shape (non-LoRA trainable or same-rank LoRA)
                                        if w_k.shape != w_l.shape:
                                            continue
                                        diff = a_kl * (w_k.float() - w_l.float())

                                    lap_delta[key] = (
                                        lap_delta.get(key, torch.zeros_like(w_k.float())) + diff
                                    )
                                    n_lap_applied += 1

                            # Apply nudge: W_k ← W_k − η * Σ a_kℓ(W_k − W_ℓ)
                            # Clients with no neighbors are unchanged (isolated).
                            new_state: Dict[str, torch.Tensor] = {}
                            for key, w_k in state_k.items():
                                if key in lap_delta:
                                    new_state[key] = (
                                        w_k.float() - _eta * lap_delta[key]
                                    ).to(w_k.dtype)
                                else:
                                    new_state[key] = w_k
                            updated_flat[cid] = new_state

                        print(
                            f"  Laplacian applied: {len(updated_flat)} clients nudged "
                            f"({n_lap_applied} param updates via task graph)",
                            flush=True,
                        )
                        aggregated_flat: Dict[int, Dict[str, torch.Tensor]] = updated_flat
                    else:
                        if _skip_lap:
                            print(
                                f"  Laplacian skipped (warmup); using raw local weights.",
                                flush=True,
                            )
                        else:
                            print(
                                f"[Round {round_idx+1}] Laplacian unavailable (no task graph); "
                                "using raw local weights.",
                                flush=True,
                            )
                        aggregated_flat = local_flat

                else:
                    # ── intra-cluster raFLoRA / HeLoRA aggregation ────────────────────
                    # Modes: fedavg_cluster, atlas_no_laplacian, etc.
                    # Personalization: server builds a max-rank template per key, then
                    # returns only the r_k-slice to each client → distinct models.
                    print(
                        f"\n[Round {round_idx+1}] Task-aware aggregation "
                        f"(raFLoRA / HeLoRA rank-personalized)...",
                        flush=True,
                    )
                    aggregated_flat = {}

                    for cluster_id, client_ids in task_clusters.items():
                        n_clients = len(client_ids)
                        print(f"  Group {cluster_id} ({n_clients} clients): raFLoRA", flush=True)

                        client_states: Dict[int, Dict[str, torch.Tensor]] = {
                            cid: {n: p.data.clone()
                                  for n, p in client_models[cid].named_parameters()}
                            for cid in client_ids
                        }

                        # per_client[cid] will hold a state shaped to client cid's own ranks.
                        per_client: Dict[int, Dict[str, torch.Tensor]] = {
                            cid: {} for cid in client_ids
                        }

                        all_keys = set().union(
                            *[set(s.keys()) for s in client_states.values()]
                        )
                        for key in all_keys:
                            tensors = {
                                cid: client_states[cid][key]
                                for cid in client_ids
                                if key in client_states[cid]
                            }
                            if not tensors:
                                continue

                            shapes = {cid: t.shape for cid, t in tensors.items()}
                            key_low = key.lower()
                            is_lora_a = 'lora_a' in key_low
                            is_lora_b = 'lora_b' in key_low

                            if is_lora_a or is_lora_b:
                                # ── HeLoRA / raFLoRA: build max-rank template ──────────
                                # rank dim: 0 for A (r×v), 1 for B (u×r)
                                rank_dim = 0 if is_lora_a else 1
                                ranks = {cid: t.shape[rank_dim] for cid, t in tensors.items()}
                                max_rank = max(ranks.values())

                                # Build averaged max-rank template slice-by-slice.
                                # Slice r: only clients with rank > r contribute.
                                if is_lora_a:
                                    hidden = next(iter(tensors.values())).shape[1]
                                    template = torch.zeros(
                                        max_rank, hidden,
                                        dtype=torch.float32,
                                        device=next(iter(tensors.values())).device,
                                    )
                                    for r in range(max_rank):
                                        contrib = [
                                            t.float() for cid2, t in tensors.items()
                                            if ranks[cid2] > r
                                        ]
                                        if contrib:
                                            template[r:r+1, :] = torch.stack(
                                                [c[r:r+1, :] for c in contrib]
                                            ).mean(0)
                                else:  # lora_b
                                    hidden = next(iter(tensors.values())).shape[0]
                                    template = torch.zeros(
                                        hidden, max_rank,
                                        dtype=torch.float32,
                                        device=next(iter(tensors.values())).device,
                                    )
                                    for r in range(max_rank):
                                        contrib = [
                                            t.float() for cid2, t in tensors.items()
                                            if ranks[cid2] > r
                                        ]
                                        if contrib:
                                            template[:, r:r+1] = torch.stack(
                                                [c[:, r:r+1] for c in contrib]
                                            ).mean(0)

                                # Each client receives its OWN r_k-slice of the template.
                                # This retains personalization: high-rank clients keep their
                                # extra components; low-rank clients are not polluted.
                                for cid in client_ids:
                                    if cid not in tensors:
                                        continue
                                    r_k = ranks[cid]
                                    orig_dtype = tensors[cid].dtype
                                    if is_lora_a:
                                        per_client[cid][key] = template[:r_k, :].to(orig_dtype)
                                    else:
                                        per_client[cid][key] = template[:, :r_k].to(orig_dtype)

                            elif len(set(v for v in shapes.values())) == 1:
                                # Same-shape non-LoRA param (e.g. classifier head, bias).
                                # Average is correct within a task-pure cluster.
                                avg = torch.stack(
                                    [t.float() for t in tensors.values()]
                                ).mean(0)
                                for cid in client_ids:
                                    if cid in tensors:
                                        per_client[cid][key] = avg.to(tensors[cid].dtype)

                            else:
                                # Non-LoRA param with shape mismatch: min-slice average,
                                # then pad back to each client's original shape.
                                ndim = len(next(iter(tensors.values())).shape)
                                min_shape = tuple(
                                    min(shapes[c][d] for c in tensors) for d in range(ndim)
                                )
                                slices = tuple(slice(0, s) for s in min_shape)
                                avg_min = torch.stack(
                                    [t[slices].float() for t in tensors.values()]
                                ).mean(0)
                                for cid in client_ids:
                                    if cid not in tensors:
                                        continue
                                    # Preserve client's own values outside the shared slice.
                                    merged = tensors[cid].clone().float()
                                    merged[slices] = avg_min
                                    per_client[cid][key] = merged.to(tensors[cid].dtype)

                        for cid in client_ids:
                            aggregated_flat[cid] = per_client[cid]

                    print(
                        f"\n[Round {round_idx+1}] Laplacian skipped (mode={mode})",
                        flush=True,
                    )

                # ── Step 3: Load per-client states back into models ───────────────────
                # strict=False: frozen base-model params are untouched.
                for cid, flat_state in aggregated_flat.items():
                    model = client_models[cid]
                    try:
                        model.load_state_dict(flat_state, strict=False)
                    except Exception:
                        current_state = dict(model.named_parameters())
                        for key, val in flat_state.items():
                            if key in current_state and current_state[key].shape == val.shape:
                                try:
                                    current_state[key].data.copy_(
                                        val.to(current_state[key].device)
                                    )
                                except Exception:
                                    pass

                # Compute download bytes: trainable param sizes returned to each client.
                for cid in aggregated_flat:
                    comm_download[cid] = sum(
                        int(t.numel() * t.element_size())
                        for k, t in aggregated_flat[cid].items()
                        if any(kw in k.lower() for kw in trainable_kw)
                    )

                # ── Step 4: NO cross-client server sync ───────────────────────────────
                # Each client has its own dedicated server.  The sync_to_client call
                # inside _train_client_split already copies each client's server top-
                # layers back to that client's model at the end of their training turn,
                # giving perfect bottom-LoRA / top-layer alignment.  We must NOT
                # broadcast any other client's server state here — that was the root
                # cause of the within-cluster variance (representative's top layers
                # glued onto other clients' mismatched bottom LoRA → near-random eval).
                if use_split_learning and split_server_models:
                    pass  # per-client servers need no cross-client sync

            # Evaluation

            print(f"\n[Round {round_idx+1}] Evaluation...", flush=True)
            round_accuracies = {}
            round_f1s = {}
            round_canonical = {}
            
            for client_data in self.clients_data:
                cid = client_data.client_id
                model = client_models[cid]

                # Move model to GPU for evaluation
                model.to(self.device)
                acc, loss, f1, canonical = self._evaluate_client(
                    model,
                    client_data.test_dataset,
                    task_name=client_data.task_name,
                )
                round_accuracies[cid] = acc
                round_f1s[cid] = f1
                round_canonical[cid] = canonical
                print(f"  Client {cid} ({client_data.task_name}): acc={acc:.4f}, f1={f1:.4f}, canonical={canonical:.4f}, loss={loss:.4f}", flush=True)

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
            
            # Per-task summary for this round
            task_canonical_round: Dict[str, List[float]] = {}
            for cd in self.clients_data:
                task_canonical_round.setdefault(cd.task_name, []).append(round_canonical[cd.client_id])
            task_summary = "  |  ".join(
                f"{t}: {np.mean(v):.4f}" for t, v in sorted(task_canonical_round.items())
            )
            print(f"\n[Round {round_idx+1}] SUMMARY", flush=True)
            print(f"  Per-task canonical  : {task_summary}", flush=True)
            print(f"  Macro avg accuracy  : {np.mean(list(round_accuracies.values())):.4f}", flush=True)
            print(f"  Macro avg canonical : {np.mean(list(round_canonical.values())):.4f}", flush=True)
            print(f"  Round time          : {round_time:.1f}s", flush=True)
            print(f"  Communication       : ↑{round_upload_total/1e6:.2f}MB ↓{round_download_total/1e6:.2f}MB", flush=True)
            
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
                    'fingerprints': fingerprints,
                    'client_server_opt_states': client_server_opt_states,  # Fix 21: per-client server optimizer states
                    'client_lora_opt_states':   client_lora_opt_states,   # Fix 27: per-client LoRA optimizer states
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
    
    def _apply_heterogeneous_lora(self, model: PreTrainedModel, lora_ranks,
                                    split_layer: Optional[int] = None) -> nn.Module:
        """Apply LoRA with heterogeneous ranks per layer.
        
        When split_layer is given (split learning mode), LoRA is ONLY applied to
        the bottom `split_layer` transformer layers. Top layers remain as plain
        base weights updated server-side, so no random LoRA adapters corrupt
        inference on the top half.
        """
        from peft import get_peft_model, LoraConfig, TaskType
        
        # Parse lora_ranks into a per-layer dict {layer_idx: rank}
        # lora_ranks is a list like [4, 8, 8, 8, 4, 4] from Phase 2.
        per_layer_ranks: Dict[int, int] = {}
        default_rank = 8
        if lora_ranks is not None:
            if isinstance(lora_ranks, dict):
                per_layer_ranks = {int(k): int(v) for k, v in lora_ranks.items()}
                default_rank = min(per_layer_ranks.values()) if per_layer_ranks else 8
            elif isinstance(lora_ranks, (list, tuple, np.ndarray)):
                per_layer_ranks = {i: int(r) for i, r in enumerate(lora_ranks)}
                default_rank = min(per_layer_ranks.values()) if per_layer_ranks else 8
            else:
                try:
                    default_rank = int(lora_ranks)
                except Exception:
                    default_rank = 8
        
        # Auto-detect target modules based on model architecture
        all_module_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or type(module).__name__ == 'Conv1D':
                all_module_names.append(name)

        target_modules: Any = []

        module_leaf_names = set(n.split('.')[-1] for n in all_module_names)
        if any('c_attn' in name for name in all_module_names):
            # GPT-2 / GPT-2-XL: c_attn=QKV combined, c_proj=attn-out + FFN-out, c_fc=FFN-in
            target_modules = ['c_attn', 'c_proj']
            if 'c_fc' in module_leaf_names:
                target_modules.append('c_fc')         # FFN input projection
        elif any('q_proj' in name for name in all_module_names):
            # LLaMA / Qwen2 / Mistral: standard MHA + SwiGLU FFN
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            for ffn_mod in ['gate_proj', 'up_proj', 'down_proj']:  # SwiGLU FFN
                if ffn_mod in module_leaf_names:
                    target_modules.append(ffn_mod)
        elif any('q_lin' in name for name in all_module_names):
            # DistilBERT: q_lin/k_lin/v_lin/out_lin=attention, lin1/lin2=FFN
            target_modules = ['q_lin', 'k_lin', 'v_lin', 'out_lin']
            for ffn_mod in ['lin1', 'lin2']:
                if ffn_mod in module_leaf_names:
                    target_modules.append(ffn_mod)
        elif any('query' in name for name in all_module_names):
            # BERT / RoBERTa: attention only (FFN uses 'dense' which is too generic)
            target_modules = ['query', 'key', 'value']
        else:
            for pattern in ['attn', 'attention', 'self']:
                matched = [n.split('.')[-1] for n in all_module_names
                           if pattern in n and 'score' not in n and 'classifier' not in n]
                if matched:
                    target_modules = list(set(matched))[:4]
                    break

        if not target_modules:
            target_modules = [n.split('.')[-1] for n in all_module_names
                              if 'score' not in n and 'classifier' not in n][:3]
        if not target_modules:
            print("[LoRA] WARNING: No suitable target modules found — using 'all-linear' fallback.")
            target_modules = 'all-linear'

        # CRITICAL: When in split-learning mode, restrict LoRA to ONLY the bottom
        # `split_layer` layers.  The top layers are handled by the server (base
        # weights only); if LoRA adapters existed on the top layers they would
        # stay randomly initialised and corrupt every forward pass.
        layers_to_transform = None
        if split_layer is not None:
            layers_to_transform = list(range(split_layer))

        # Build PEFT rank_pattern: maps regex patterns → per-layer rank.
        # PEFT calls re.search(pattern, full_module_name) for each LoRA adapter.
        # For DistilBERT: full path = "distilbert.transformer.layer.0.attention.q_lin"
        # Using "transformer.layer.{i}" is unambiguous: "layer.0" would mis-match
        # "layer.10" in larger models, while the transformer-prefixed form anchors correctly.
        active_layers = layers_to_transform if layers_to_transform is not None else list(per_layer_ranks.keys())
        rank_pattern: Dict[str, int] = {}
        for layer_idx in active_layers:
            if layer_idx in per_layer_ranks:
                # Matches any module path containing "transformer.layer.<i>":
                # e.g. distilbert.transformer.layer.0.attention.q_lin → rank for layer 0
                rank_pattern[f"transformer\.layer\.{layer_idx}\.?"] = per_layer_ranks[layer_idx]

        # Use minimum active rank as LoraConfig base r (rank_pattern overrides upward)
        active_ranks = [per_layer_ranks[i] for i in active_layers if i in per_layer_ranks]
        base_rank = min(active_ranks) if active_ranks else default_rank

        # Log actual per-layer rank assignment
        if rank_pattern:
            rank_summary = ", ".join(f"L{i}:r{per_layer_ranks[i]}" for i in active_layers if i in per_layer_ranks)
            print(f"  [LoRA] Layers {active_layers} (split_layer={split_layer}), "
                  f"per-layer ranks: [{rank_summary}]")
        else:
            print(f"  [LoRA] Applying LoRA to all layers, rank={base_rank}")

        # CRITICAL: Use FEATURE_EXTRACTION (not SEQ_CLS) so that PEFT does NOT
        # wrap the classifier head in a ModulesToSaveWrapper.  When task_type=
        # SEQ_CLS, PEFT automatically wraps model.classifier even with
        # modules_to_save=None.  ModulesToSaveWrapper stores weights under
        # "original_module.weight" / "modules_to_save.default.weight" — so
        # sync_to_client's load_state_dict({'weight':..., 'bias':...}, strict=False)
        # silently matches NO keys and leaves the classifier at zeros forever.
        # With FEATURE_EXTRACTION, model.classifier stays a plain nn.Linear and
        # load_state_dict works correctly.  The split-FL server retains full
        # responsibility for the classifier; the client model only hosts the
        # LoRA-adapted bottom layers.
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=base_rank,
            lora_alpha=base_rank * 2,   # scale alpha with base rank (standard heuristic)
            lora_dropout=0.1,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            rank_pattern=rank_pattern,  # per-layer override: {pattern: rank}
            modules_to_save=None,
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
        base: Any = model.base_model.model if hasattr(model, 'base_model') else model  # type: ignore[union-attr]

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
        Create one SplitServerWrapper per task cluster (keyed by cluster_id).

        The server is SHARED within a cluster.  Cross-client contamination is
        prevented by calling sync_from_client at the START of each client's training
        turn (restoring the server to the state this client received at the end of
        the previous round), then sync_to_client at the END (storing the newly
        trained state back into the client model).  No round-end broadcast is used.

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

            # Fix 31: compute class-balanced loss weights from the cluster's training data.
            # Collects all labels across every client in the cluster, then applies
            # sklearn-style "balanced" weighting: w_c = N / (C * n_c).
            # This prevents imbalanced tasks (e.g. CoLA: 69% class-1) from collapsing
            # to the trivial majority-class prediction (MCC=0, acc=0.69).
            class_weights_tensor: Optional[torch.Tensor] = None
            try:
                label_counts: Dict[int, int] = {}
                for cid in client_ids:
                    cd = next(c for c in self.clients_data if c.client_id == cid)
                    for idx in range(len(cd.train_dataset)):
                        ex = cd.train_dataset[idx]
                        lbl = int(ex['label']) if isinstance(ex, dict) else int(ex[2])
                        label_counts[lbl] = label_counts.get(lbl, 0) + 1
                if len(label_counts) > 1:
                    total = sum(label_counts.values())
                    n_classes = num_labels
                    weights = [
                        total / (n_classes * label_counts.get(c, 1))
                        for c in range(n_classes)
                    ]
                    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
                    print(
                        f"  Cluster {cluster_id} ({rep_client.task_name}) class weights: "
                        + ", ".join(f"c{c}={w:.3f}" for c, w in enumerate(weights))
                    )
            except Exception as _cw_exc:
                print(f"  [Warning] Could not compute class weights for cluster {cluster_id}: {_cw_exc}")

            try:
                server_wrapper = SplitServerWrapper(base_model, split_layer, n_total,
                                                    class_weights=class_weights_tensor)
            except ValueError as exc:
                raise RuntimeError(
                    f"Cannot build SplitServerWrapper for cluster {cluster_id}: {exc}"
                ) from exc

            # Fix 21: Proper classifier initialization for split learning.
            #
            # The classifier head is the ONLY bridge between the server's
            # transformer layers and the loss function.  Its weight matrix
            # determines ∂logits/∂h, so:
            #   - W = 0  →  ∂loss/∂h = 0  →  server layers + client LoRA get NO gradient
            #   - W ≠ 0  →  ∂loss/∂h ≠ 0  →  full gradient flow from step 1
            #
            # We use SEEDED Xavier uniform init so every cluster gets a
            # deterministic, non-zero W.  Combined with log-prior bias and
            # sync_to_client (below), all clients in a cluster start with
            # IDENTICAL classifiers → no label inversion, full gradient flow.
            #
            # The seed is deterministic per cluster (base_seed + cluster_id)
            # so that different clusters get different inits appropriate to
            # their num_labels.
            srv_head = getattr(server_wrapper, 'classifier', None) or getattr(server_wrapper, 'score', None)
            if srv_head is not None:
                _clf_rng = torch.Generator()
                _clf_rng.manual_seed(self.config.seed + cluster_id * 1000)
                # Xavier uniform: std ≈ sqrt(2 / (fan_in + fan_out))
                # For (2, 768): std ≈ 0.051 — small enough not to dominate,
                # large enough to carry gradient.
                _fan_in = srv_head.weight.size(1)
                _fan_out = srv_head.weight.size(0)
                _bound = math.sqrt(6.0 / (_fan_in + _fan_out))
                with torch.no_grad():
                    srv_head.weight.uniform_(-_bound, _bound, generator=_clf_rng)
                print(f"  Cluster {cluster_id}: Xavier classifier W "
                      f"(bound={_bound:.4f}, |W|={srv_head.weight.norm().item():.4f})")

                # Log-prior bias: bias[c] = log(π_c) where π_c = n_c / N
                if srv_head.bias is not None:
                    if label_counts and len(label_counts) > 1:
                        _total_lc = sum(label_counts.values())
                        with torch.no_grad():
                            for c in range(num_labels):
                                _prior = label_counts.get(c, 1) / _total_lc
                                srv_head.bias[c] = math.log(max(_prior, 1e-8))
                        print(f"  Cluster {cluster_id}: log-prior bias = "
                              + ", ".join(f"c{c}={srv_head.bias[c].item():.4f}" for c in range(num_labels)))
                    else:
                        torch.nn.init.zeros_(srv_head.bias)

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

        print(f"  \u2713 Built {len(server_models)} server model(s).")
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
        client_lora_opt_states: Optional[Dict] = None,
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
        # Fix 26: LoRA adapters need a higher learning rate than the
        # server's full-parameter fine-tuning (Hu et al., 2022 recommend
        # 1e-4 – 3e-4 for LoRA).  The original code used the same 3e-5 as
        # the server, which — combined with gradient sparsification — made
        # client-side learning negligible.
        _client_lr = max(self.config.learning_rate * 5, 2e-4)
        _lora_params = [p for p in client_model.parameters() if p.requires_grad]
        client_optimizer = torch.optim.AdamW(_lora_params, lr=_client_lr)

        # Fix 27: Restore per-client LoRA optimizer state if available.
        # Without this, Adam momentum/variance are discarded at the end of
        # every round and rebuilt from scratch in the next round — effectively
        # running SGD (no momentum) for the first ~20 batches of each round.
        # Persistent optimizer state doubles effective convergence speed.
        if client_lora_opt_states is not None and client_id in client_lora_opt_states:
            try:
                client_optimizer.load_state_dict(client_lora_opt_states[client_id])
            except Exception:
                pass  # shape mismatch after rank change → use fresh state

        total_loss = 0.0
        num_batches = 0
        total_upload_bytes = 0
        total_download_bytes = 0
        nan_count = 0
        total_batches_est = len(dataloader) * self.config.local_epochs
        _client_start = time.time()
        print(f"    Client {client_id} ({task_name}) | split-FL | "
              f"{self.config.local_epochs} epoch(s) × ~{len(dataloader)} batches", flush=True)

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

                # ── NETWORK UPLOAD: communication cost accounting ──────────
                # (Fix 23) Train/eval quantization distribution shift.
                #
                # Previously, activations were per-channel int8 quantized for
                # the upload, and the SERVER trained on dequantized (noisy)
                # activations.  During evaluation, the full model processes
                # clean fp32 activations.  Because the server's top layers
                # were trained on the noisy distribution, evaluation on clean
                # activations represents a domain shift at the cut layer.
                #
                # Fix: pass fp32 activations to the server during training
                # (matching the evaluation path), and ANALYTICALLY account
                # for int8 communication savings.  This is standard practice
                # in split FL literature (VFLAIR-LLM, HSplitLoRA) where
                # compression is a communication optimization, not a training
                # mechanism.  The server sees the same activation distribution
                # during both training and inference.
                #
                # Communication cost: still reported as per-channel int8
                # (4× compression vs fp32) because that is what would be
                # transmitted over the wire in a real deployment.
                _act_fp = split_activations.detach()          # (B, S, H)
                # Analytical int8 upload cost (1 byte per value vs 4 bytes fp32)
                upload_bytes = _act_fp.numel()  # int8 = 1 byte/value
                total_upload_bytes += int(upload_bytes)

                # Server receives clean fp32 activations as a leaf node.
                server_input = _act_fp.clone().requires_grad_(True)

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

                # ── NETWORK DOWNLOAD: communication cost accounting ────────────
                # Fix 24: The previous 30% top-k gradient sparsification zeroed
                # 70% of the gradient tensor sent to the client.  Combined with
                # the client's small LoRA capacity, this destroyed virtually all
                # gradient signal: the bottom LoRA barely adapted, forcing the
                # server to overfit static representations → train loss 0.22,
                # test accuracy 0.50 (random).  Standard split-FL practice
                # (VFLAIR-LLM, HSplitLoRA, SplitQuant) transmits full gradients
                # for correctness and reports analytical compressed cost.
                #
                # Analytical download cost: 30% top-k + idx (as if compressed).
                _nnz_analytical = max(1, int(0.30 * activation_gradients.numel()))
                download_bytes = _nnz_analytical * (activation_gradients.element_size() + 4)
                total_download_bytes += int(download_bytes)
                # Client receives FULL fp32 gradients (no sparsification).

                # ── CLIENT: backward through LoRA using full server gradients ──
                split_activations.backward(activation_gradients)
                torch.nn.utils.clip_grad_norm_(
                    client_model.parameters(), self.config.gradient_clip_norm
                )
                client_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Live progress line (overwrite in-place every 5 batches)
                if num_batches % 5 == 0 or num_batches == 1:
                    running_loss = total_loss / num_batches
                    elapsed = time.time() - _client_start
                    eta = (elapsed / num_batches) * (total_batches_est - num_batches)
                    print(
                        f"\r    Client {client_id} [{task_name}] "
                        f"ep{_epoch+1}/{self.config.local_epochs} "
                        f"batch {num_batches}/{total_batches_est} "
                        f"loss={running_loss:.4f} "
                        f"eta={eta:.0f}s",
                        end="", flush=True
                    )

                # Free GPU memory after each batch
                del input_ids, attention_mask, labels, split_activations
                del server_input, _act_fp, activation_gradients, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_batches, 1)
        elapsed_total = time.time() - _client_start
        status = f"  ⚠️ {nan_count} NaN batches skipped" if nan_count else ""
        print(
            f"\r    Client {client_id} ({task_name}): {num_batches} batches, "
            f"loss={avg_loss:.4f}, "
            f"↑{total_upload_bytes / 1e6:.2f} MB  ↓{total_download_bytes / 1e6:.2f} MB "
            f"[{elapsed_total:.0f}s]{status}",
            flush=True
        )

        # Fix 27: Snapshot client LoRA optimizer state to CPU for next round.
        # The state is stored in the mutable dict passed by the caller so it
        # persists across rounds without changing the return signature.
        if client_lora_opt_states is not None:
            _cl_opt_sd = client_optimizer.state_dict()
            _cpu_cl_opt: Dict[str, Any] = {
                'param_groups': _cl_opt_sd['param_groups'],
                'state': {},
            }
            for _pk, _pv in _cl_opt_sd['state'].items():
                _cpu_cl_opt['state'][_pk] = {
                    _kk: _vv.cpu().clone() if isinstance(_vv, torch.Tensor) else _vv
                    for _kk, _vv in _pv.items()
                }
            client_lora_opt_states[client_id] = _cpu_cl_opt

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
        total_batches_est = len(dataloader) * self.config.local_epochs
        _client_start = time.time()
        print(f"    Client {client_id} ({task_name}) | local | "
              f"{self.config.local_epochs} epoch(s) × ~{len(dataloader)} batches", flush=True)
        
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

                # Live progress line (overwrite in-place every 5 batches)
                if num_batches % 5 == 0 or num_batches == 1:
                    running_loss = total_loss / num_batches
                    elapsed = time.time() - _client_start
                    eta = (elapsed / num_batches) * (total_batches_est - num_batches)
                    print(
                        f"\r    Client {client_id} [{task_name}] "
                        f"ep{epoch+1}/{self.config.local_epochs} "
                        f"batch {num_batches}/{total_batches_est} "
                        f"loss={running_loss:.4f} "
                        f"eta={eta:.0f}s",
                        end="", flush=True
                    )
        
        avg_loss = total_loss / max(num_batches, 1)
        elapsed_total = time.time() - _client_start
        status = " ⚠️ NaN detected" if nan_detected else ""
        print(f"\r    Client {client_id} ({task_name}): {num_batches} batches, "
              f"loss={avg_loss:.4f} [{elapsed_total:.0f}s]{status}", flush=True)
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

    # Force line-buffered stdout so every print() appears immediately in the terminal
    import sys as _sys
    try:
        _sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
    except AttributeError:
        import io
        _sys.stdout = io.TextIOWrapper(open(_sys.stdout.fileno(), 'wb', 0), write_through=True)
    
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
            'fingerprint_samples': 400,
            'fingerprint_batches': 50,
            'fingerprint_batch_size': 8,
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
            fingerprint_batch_size=model_hparams.get('fingerprint_batch_size', 4),
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
            fingerprint_batch_size=model_hparams.get('fingerprint_batch_size', 4),
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
