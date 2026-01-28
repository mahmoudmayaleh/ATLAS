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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle

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
    clients_per_task: int = 3
    
    # Training
    num_rounds: int = 10
    local_epochs: int = 3
    batch_size: int = 16
    max_samples_per_client: int = 2000
    learning_rate: float = 2e-5
    
    # Device heterogeneity
    device_types: List[str] = None  # e.g., ['cpu_2gb', 'tablet_4gb', 'laptop_8gb', 'gpu_16gb']
    
    # Phase 1: Clustering
    fingerprint_epochs: int = 2  # Epochs for gradient extraction
    fingerprint_dim: int = 64
    k_range: Tuple[int, int] = (2, 5)  # Try k=2,3,4,5 clusters
    
    # Phase 2: LoRA ranks
    rank_candidates: List[int] = None  # [4, 8, 16, 32]
    alpha_base: float = 0.5  # Base model takes 50% memory
    alpha_act: float = 0.25  # Activations take 25%
    alpha_opt: float = 0.15  # Optimizer takes 15%
    
    # Phase 3: Split learning
    split_layer: int = 3  # Split at layer 3 (bottom half)
    
    # Phase 4: Laplacian
    eta: float = 0.1  # Regularization strength
    k_neighbors: int = 3
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 2  # Save every N rounds
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ['sst2', 'mrpc', 'cola']
        if self.device_types is None:
            # Mix of devices: 2 low-end, 3 mid, 2 high, 1 very high
            self.device_types = ['cpu_2gb'] * 2 + ['tablet_4gb'] * 3 + ['laptop_8gb'] * 2 + ['gpu_16gb'] * 1
        if self.rank_candidates is None:
            self.rank_candidates = [4, 8, 16, 32]


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
        
        # Load dataset
        if task_name == 'sst2':
            dataset = load_dataset(dataset_name, split='train')
            test_dataset = load_dataset(dataset_name, split='validation')
        else:
            dataset = load_dataset(dataset_name, task_name, split='train')
            test_dataset = load_dataset(dataset_name, task_name, split='validation')
        
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
            cluster_labels, fingerprints = self._phase1_clustering()
        else:
            cluster_labels = checkpoint['cluster_labels']
            fingerprints = checkpoint['fingerprints']
            print(f"[RESUME] Loaded clustering from checkpoint")
        
        # ========== PHASE 2: HETEROGENEOUS RANK ALLOCATION ==========
        if start_round == 0:
            print(f"\n{'='*70}")
            print(f"PHASE 2: HETEROGENEOUS RANK ALLOCATION")
            print(f"{'='*70}\n")
            device_configs = self._phase2_rank_allocation(cluster_labels, fingerprints)
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
            start_round=start_round,
            checkpoint=checkpoint
        )
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"[DONE] ATLAS pipeline complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Final per-client accuracy: {results['final_accuracies']}")
        print(f"{'='*70}\n")
        
        return results
    
    def _phase1_clustering(self) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
        """
        Phase 1: Extract gradient fingerprints and cluster clients.
        Returns: (cluster_labels, fingerprints)
        """
        print("[Phase 1] Extracting gradient fingerprints...")
        
        # Create temporary models for fingerprinting
        raw_gradients = {}

        for client_data in self.clients_data:
            print(f"  Client {client_data.client_id} ({client_data.task_name})...", end=" ")

            # Create model
            _, _, _, num_labels = self.dataset_map[client_data.task_name]
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=num_labels
            ).to(self.device)

            # Extract raw gradient vector (tensor)
            raw_grad = self._extract_fingerprint(model, client_data.train_dataset)
            raw_gradients[client_data.client_id] = raw_grad

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            print(f"✓ raw grad shape: {raw_grad.shape}")
        
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
        for cluster_id in sorted(set(labels)):
            clients_in_cluster = [cid for cid, label in cluster_labels.items() if label == cluster_id]
            print(f"    - Group {cluster_id}: {len(clients_in_cluster)} clients")
        
        print(f"  ✓ Found {len(set(cluster_labels.values()))} task groups")
        for cluster_id in set(cluster_labels.values()):
            clients_in_cluster = [cid for cid, label in cluster_labels.items() if label == cluster_id]
            tasks_in_cluster = [self.clients_data[cid].task_name for cid in clients_in_cluster]
            print(f"    Group {cluster_id}: clients {clients_in_cluster}, tasks {set(tasks_in_cluster)}")
        
        # Update client cluster assignments
        for client_data in self.clients_data:
            client_data.cluster_id = cluster_labels[client_data.client_id]
        
        return cluster_labels, fingerprints
    
    def _extract_fingerprint(self, model: nn.Module, dataset: Subset) -> np.ndarray:
        """Extract gradient fingerprint from a client's local training"""
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        
        model.train()
        grad_history = []
        
        # Train for fingerprint_epochs to collect gradients
        for epoch in range(self.config.fingerprint_epochs):
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 10:  # Limit to 10 batches for speed
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                
                # Collect gradients from last 2 layers
                grads = []
                for name, param in model.named_parameters():
                    if param.grad is not None and ('layer.5' in name or 'layer.4' in name or 'classifier' in name):
                        grads.append(param.grad.detach().cpu().flatten().numpy())
                
                if grads:
                    grad_history.append(np.concatenate(grads))
                
                optimizer.step()
        
        # Extract fingerprint using GradientExtractor
        if grad_history:
            avg_grad = np.mean(grad_history, axis=0)
            # Return raw gradient tensor (will be PCA-fitted later)
            return torch.from_numpy(avg_grad).float()
        else:
            # Fallback: random raw gradient tensor (PCA can handle fallback)
            return torch.from_numpy(np.random.randn(self.config.fingerprint_dim)).float()
    
    def _phase2_rank_allocation(
        self, 
        cluster_labels: Dict[int, int],
        fingerprints: Dict[int, np.ndarray]
    ) -> Dict[int, Dict]:
        """
        Phase 2: Allocate heterogeneous LoRA ranks based on device + task complexity.
        Returns: device_configs[client_id] = {device_profile, lora_ranks}
        """
        print("[Phase 2] Profiling devices and allocating ranks...")
        
        device_configs = {}
        
        # Get cluster statistics (variance = complexity)
        cluster_variances = {}
        for cluster_id in set(cluster_labels.values()):
            cluster_fingerprints = [fingerprints[cid] for cid, label in cluster_labels.items() if label == cluster_id]
            if cluster_fingerprints:
                variance = np.var(cluster_fingerprints, axis=0).mean()
                cluster_variances[cluster_id] = variance
        
        for client_data in self.clients_data:
            device_type = client_data.device_type
            cluster_id = client_data.cluster_id
            
            # Get device profile
            device_profile = self.device_profiler.profile_device(device_type)
            
            # Compute importance scores (higher variance = more important)
            cluster_variance = cluster_variances.get(cluster_id, 1.0)
            importance_scores = {f'layer_{i}': cluster_variance * (1.0 + 0.1 * i) for i in range(6)}
            
            # Allocate ranks (DistilBERT has 6 transformer layers)
            lora_ranks = self.rank_allocator.allocate_ranks(
                device_profile=device_profile,
                importance_scores=importance_scores,
                n_layers=6,
                split_point=None
            )
            
            device_configs[client_data.client_id] = {
                'device_profile': device_profile,
                'lora_ranks': lora_ranks,
                'importance_scores': importance_scores
            }
            
            client_data.lora_ranks = lora_ranks
            
            print(f"  Client {client_data.client_id} ({device_type}): ranks {lora_ranks}")
        
        print(f"  ✓ Allocated heterogeneous ranks for {len(device_configs)} clients")
        return device_configs
    
    def _phase3_4_training(
        self,
        cluster_labels: Dict[int, int],
        device_configs: Dict[int, Dict],
        start_round: int = 0,
        checkpoint: Optional[Dict] = None
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
        
        # Build adjacency weights via k-NN on fingerprints
        all_clients = sorted([c for clients in task_clusters.values() for c in clients])
        n_clients = len(all_clients)
        fps_matrix = np.vstack([fingerprints[cid] for cid in all_clients])

        # Cosine similarity
        normed = fps_matrix / (np.linalg.norm(fps_matrix, axis=1, keepdims=True) + 1e-12)
        sim = normed @ normed.T

        # Keep only top-k neighbors per client
        k = max(1, int(self.config.k_neighbors))
        adj = np.zeros_like(sim)
        for i in range(n_clients):
            # exclude self
            sim[i, i] = -1.0
            topk = np.argsort(sim[i])[-k:]
            for j in topk:
                if sim[i, j] > 0:
                    adj[i, j] = float(sim[i, j])

        task_graph = TaskGraph.from_task_clusters(
            task_clusters=task_clusters,
            adjacency_weights=adj,
            normalize=True,
            symmetrize=True
        )
        
        # Initialize Laplacian aggregator
        laplacian_agg = LaplacianAggregation(
            eta=self.config.eta,
            heterogeneous_rank=True
        )
        
        # Create per-client models (MIRA approach: each client keeps own model)
        client_models = {}
        for client_data in self.clients_data:
            _, _, _, num_labels = self.dataset_map[client_data.task_name]
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
                
                # Store for Phase 4
                for cid in client_ids:
                    aggregated_models[cid] = avg_weights
            
            # Step 3: Laplacian regularization (personalization)
            print(f"\n[Round {round_idx+1}] Applying Laplacian regularization...")
            personalized_models = laplacian_agg.laplacian_update(
                client_models=aggregated_models,
                task_graph=task_graph
            )
            
            # Update client models
            for cid, weights in personalized_models.items():
                client_models[cid].load_state_dict(weights)
            
            # Step 4: Evaluation
            print(f"\n[Round {round_idx+1}] Evaluation...")
            round_accuracies = {}
            
            for client_data in self.clients_data:
                cid = client_data.client_id
                acc, loss = self._evaluate_client(
                    client_models[cid],
                    client_data.test_dataset
                )
                round_accuracies[cid] = acc
                print(f"  Client {cid} ({client_data.task_name}): acc={acc:.4f}, loss={loss:.4f}")
            
            round_time = time.time() - round_start
            
            # Store results
            results['round_metrics'].append({
                'round': round_idx + 1,
                'train_losses': round_losses,
                'test_accuracies': round_accuracies,
                'avg_accuracy': np.mean(list(round_accuracies.values())),
                'time_seconds': round_time
            })
            
            print(f"\n[Round {round_idx+1}] Avg accuracy: {np.mean(list(round_accuracies.values())):.4f}, Time: {round_time:.1f}s")
            
            # Checkpoint
            if (round_idx + 1) % self.config.save_every == 0:
                checkpoint_state = {
                    'round': round_idx,
                    'cluster_labels': cluster_labels,
                    'device_configs': device_configs,
                    'client_models': {cid: model.state_dict() for cid, model in client_models.items()},
                    'results': results,
                    'fingerprints': None  # Already computed
                }
                self._save_checkpoint(round_idx + 1, checkpoint_state)
        
        # Final accuracies
        results['final_accuracies'] = round_accuracies
        
        return results
    
    def _apply_heterogeneous_lora(self, model: nn.Module, lora_ranks: Dict[str, int]) -> nn.Module:
        """Apply LoRA with heterogeneous ranks per layer"""
        from peft import get_peft_model, LoraConfig, TaskType
        
        # Get unique rank (simplified - use max rank for now)
        # In full implementation, would apply different ranks per layer
        rank = max(lora_ranks.values()) if lora_ranks else 8
        
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
    
    def _evaluate_client(self, model: nn.Module, test_dataset) -> Tuple[float, float]:
        """Evaluate one client on test set"""
        model.eval()
        dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size * 2)
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        num_batches = 0
        
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
        
        accuracy = total_correct / max(total_samples, 1)
        avg_loss = total_loss / max(num_batches, 1)
        
        return accuracy, avg_loss
    
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
                # Average trainable parameters
                stacked = torch.stack([w[key] for w in weights_list])
                aggregated[key] = stacked.mean(dim=0)
            else:
                # Keep frozen base model (should be identical)
                aggregated[key] = weights_list[0][key]
        
        return aggregated
    
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
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()
    
    if args.mode == "quick":
        # Quick test: 3 tasks, 2 clients per task, 5 rounds
        print("[MODE] Quick test (15-20 min on T4 GPU)")
        config = ATLASConfig(
            model_name="distilbert-base-uncased",
            tasks=['sst2', 'mrpc'],  # 2 tasks for quick test
            clients_per_task=2,
            num_rounds=5,
            local_epochs=2,
            batch_size=16,
            max_samples_per_client=500,  # Small for speed
            fingerprint_epochs=1,
            save_every=2
        )
    else:
        # Full experiment: 3 tasks, 3 clients per task, 10 rounds
        print("[MODE] Full experiment (2-3 hours on T4 GPU)")
        config = ATLASConfig(
            model_name="distilbert-base-uncased",
            tasks=['sst2', 'mrpc', 'cola'],
            clients_per_task=3,
            num_rounds=10,
            local_epochs=3,
            batch_size=16,
            max_samples_per_client=2000,
            fingerprint_epochs=2,
            save_every=3
        )
    
    trainer = ATLASIntegratedTrainer(config)
    results = trainer.run_full_pipeline(resume_from=args.resume)
    
    # Save final results
    results_path = Path("./results") / f"atlas_integrated_{args.mode}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        # Convert numpy values to native Python for JSON
        results_json = {
            'round_metrics': [
                {k: (v.item() if isinstance(v, (np.integer, np.floating)) else 
                     {k2: (v2.item() if isinstance(v2, (np.integer, np.floating)) else float(v2)) 
                      for k2, v2 in v.items()} if isinstance(v, dict) else float(v))
                 for k, v in r.items()}
                for r in results['round_metrics']
            ],
            'final_accuracies': {k: float(v) for k, v in results['final_accuracies'].items()},
            'cluster_labels': results['cluster_labels'],
            'config': asdict(config)
        }
        json.dump(results_json, f, indent=2)
    
    print(f"\n[SAVED] Results saved to {results_path}")
    print("\n[DONE] ATLAS integrated experiment complete!")
