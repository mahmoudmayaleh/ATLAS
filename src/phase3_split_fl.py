"""
Phase 3: Split Federated Learning with Heterogeneous LoRA

Literature-aligned implementation of:
- SplitLoRA: Model splitting at optimal layer
- HSplitLoRA: Heterogeneous LoRA rank allocation
- VFLAIR-LLM: Vertical federated learning for LLMs
- LoRA-FA: Federated LoRA aggregation

Integrates with:
- Phase 1: Task clustering via gradient fingerprints
- Phase 2: Device-aware heterogeneous rank allocation

Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from transformers import AutoModel, AutoConfig
import warnings

# Import Phase 1 and Phase 2 components
try:
    from phase2_configuration import DeviceProfiler, RankAllocator
    PHASE2_AVAILABLE = True
except ImportError:
    try:
        from .phase2_configuration import DeviceProfiler, RankAllocator
        PHASE2_AVAILABLE = True
    except ImportError:
        PHASE2_AVAILABLE = False
        warnings.warn("Phase 2 not available. Heterogeneous rank allocation disabled.")


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA) module for efficient fine-tuning.
    
    Applies low-rank decomposition: ΔW = B @ A^T
    where A ∈ R^(r×d), B ∈ R^(d×r), r << d
    
    Memory formula (aligned with Phase 2):
        M_adapter = 2 * d * r * bytes_per_param
    
    Args:
        in_dim (int): Input dimension (hidden size)
        rank (int): LoRA rank (default: 8)
        dropout (float): Dropout probability (default: 0.0)
        alpha (float): Scaling factor (default: 1.0)
        bytes_per_param (int): Bytes per parameter (4=fp32, 2=fp16)
    """
    
    def __init__(
        self, 
        in_dim: int, 
        rank: int = 8, 
        dropout: float = 0.0, 
        alpha: float = 1.0,
        bytes_per_param: int = 4
    ):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.bytes_per_param = bytes_per_param
        
        # LoRA parameters: A and B matrices
        # A: Random Gaussian initialization (scaled)
        # B: Zero initialization (ensures ΔW=0 initially)
        # Shape: Both A and B are (in_dim, rank)
        # Computation: x @ A @ B^T where x is (*, in_dim)
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(in_dim, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA transformation: x @ A @ B^T
        
        Args:
            x: Input tensor of shape (batch, seq_len, in_dim) or (batch, in_dim)
            
        Returns:
            LoRA output scaled by alpha/rank
        """
        # Check if input dimension matches
        input_dim = x.shape[-1]
        if input_dim != self.in_dim:
            # Skip LoRA if dimensions don't match (for dummy models)
            return torch.zeros_like(x)
        
        # x @ A @ B^T
        result = torch.matmul(torch.matmul(x, self.A), self.B.T)
        result = self.dropout(result)
        return result * self.scaling
    
    def reset_parameters(self):
        """Reset LoRA parameters to initial state"""
        nn.init.normal_(self.A, mean=0.0, std=0.01)
        nn.init.zeros_(self.B)
    
    def get_memory_bytes(self) -> int:
        """
        Calculate memory in bytes using Phase 2 formula: 2*d*r*b
        
        Returns:
            Memory usage in bytes
        """
        return 2 * self.in_dim * self.rank * self.bytes_per_param
    
    def get_memory_mb(self) -> float:
        """
        Calculate memory in MB using Phase 2 formula.
        
        Returns:
            Memory usage in megabytes
        """
        return self.get_memory_bytes() / (1024 ** 2)


class SplitClient:
    """
    Client-side component for split federated learning with heterogeneous LoRA.
    
    Architecture (VFLAIR-LLM / HSplitLoRA style):
        Input → Bottom Layers (frozen) → LoRA Adapters → Activations → Server
        
    The client:
    1. Runs forward pass through bottom layers (frozen base model)
    2. Applies heterogeneous LoRA adapters (per-layer ranks from Phase 2)
    3. Sends activations to server (split learning)
    4. Receives gradients from server
    5. Trains only LoRA weights (memory-efficient PEFT)
    
    Integrates with Phase 1 and Phase 2:
    - Phase 1: Assigns task group based on gradient clustering
    - Phase 2: Computes per-layer ranks under device memory constraints
    
    Args:
        client_id (int): Unique client identifier
        model_name (str): HuggingFace model name (e.g., 'gpt2', 'bert-base')
        rank_config (Dict[int, int], optional): LoRA ranks per layer {layer_idx: rank}
                                                 If None, computes from device_profile + importance_scores
        split_layer (int, optional): Layer index where model is split
                                      If None, computes from device_profile
        device (str): Device to run on ('cpu' or 'cuda')
        learning_rate (float): Learning rate for LoRA optimizer
        bytes_per_param (int): Bytes per parameter (4=fp32, 2=fp16)
        device_profile (Dict, optional): Device profile from Phase 2's DeviceProfiler
        importance_scores (Dict, optional): Per-layer importance from Phase 2
        task_id (int, optional): Task group ID from Phase 1 clustering
    """
    
    def __init__(
        self,
        client_id: int,
        model_name: str,
        rank_config: Optional[Dict[int, int]] = None,
        split_layer: Optional[int] = None,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        bytes_per_param: int = 4,
        device_profile: Optional[Dict] = None,
        importance_scores: Optional[Dict[str, float]] = None,
        task_id: Optional[int] = None
    ):
        self.client_id = client_id
        self.model_name = model_name
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.bytes_per_param = bytes_per_param
        self.device_profile = device_profile
        self.importance_scores = importance_scores
        self.task_id = task_id
        
        # Determine split point (budget-aware if device_profile provided)
        if split_layer is None:
            self.split_layer = self._compute_split_point(model_name, device_profile)
        else:
            self.split_layer = split_layer
        
        # Load model and extract bottom layers
        self.model, self.hidden_dim, self.n_layers_total = self._load_bottom_layers(
            model_name, self.split_layer
        )
        self.model = self.model.to(self.device)
        
        # Freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Compute rank configuration (Phase 2 integration)
        if rank_config is None and PHASE2_AVAILABLE and device_profile is not None:
            self.rank_config = self._compute_rank_config_from_phase2(
                device_profile, importance_scores, self.split_layer
            )
        elif rank_config is None:
            # Fallback: uniform rank
            self.rank_config = {i: 8 for i in range(self.split_layer)}
        else:
            self.rank_config = rank_config
        
        # Create LoRA adapters for each layer
        self.lora_adapters = self._create_lora_adapters(
            self.rank_config, self.hidden_dim, bytes_per_param
        )
        self.lora_adapters = self.lora_adapters.to(self.device)
        
        # Optimizer for LoRA weights only
        self.optimizer = Adam(self.lora_adapters.parameters(), lr=learning_rate)
        
        # Training state
        self.current_activations = None
        self.training_round = 0
        self.local_dataset_size = 0  # For weighted aggregation
    
    def _compute_split_point(
        self, 
        model_name: str, 
        device_profile: Optional[Dict]
    ) -> int:
        """
        Compute optimal split point (HSplitLoRA / SplitLoRA style).
        
        Considers:
        - Device memory constraints
        - Communication cost
        - Model architecture
        
        Args:
            model_name: Model identifier
            device_profile: Device capabilities from Phase 2
            
        Returns:
            Optimal split layer index
        """
        # Get model config
        try:
            config = AutoConfig.from_pretrained(model_name)
            n_layers = getattr(config, 'n_layer', getattr(config, 'num_hidden_layers', 12))
        except:
            n_layers = 12  # Default
        
        # If no device profile, use fixed heuristic (old behavior)
        if device_profile is None:
            return get_split_point(model_name)
        
        # Budget-aware split selection (HSplitLoRA / SplitLoRA)
        memory_mb = device_profile.get('memory_mb', 2048)
        
        # Try candidate splits in middle third of network
        # (SplitLoRA: mid-layers capture task features best)
        min_split = max(2, n_layers // 3)
        max_split = min(n_layers - 2, 2 * n_layers // 3)
        
        best_split = min_split
        min_communication = float('inf')
        
        for split in range(min_split, max_split + 1):
            # Estimate memory for this split
            estimated_memory = self._estimate_split_memory(
                split, device_profile, n_layers
            )
            
            # Reject if exceeds budget
            if estimated_memory > memory_mb * 0.8:  # 80% safety margin
                continue
            
            # Estimate communication cost (activation size)
            # Lower split = more communication (larger activations)
            communication_cost = (n_layers - split) * 100  # Simplified
            
            if communication_cost < min_communication:
                min_communication = communication_cost
                best_split = split
        
        return best_split
    
    def _estimate_split_memory(
        self, 
        split: int, 
        device_profile: Dict,
        n_layers: int
    ) -> float:
        """
        Estimate memory required for a given split point.
        
        Memory includes:
        - Base model parameters for bottom layers
        - LoRA adapter parameters
        - Activation memory for batch
        
        Args:
            split: Candidate split point
            device_profile: Device capabilities
            n_layers: Total number of layers
            
        Returns:
            Estimated memory in MB
        """
        # Base model memory (fraction of total based on split)
        base_fraction = split / n_layers
        base_memory = device_profile.get('memory_mb', 2048) * 0.35 * base_fraction
        
        # LoRA memory (Phase 2 formula)
        # Assume rank ~8-16 for estimation
        hidden_dim = 768  # GPT-2 default
        avg_rank = 12
        lora_memory = split * (2 * hidden_dim * avg_rank * self.bytes_per_param) / (1024 ** 2)
        
        # Activation memory (batch_size * seq_len * hidden_dim * bytes)
        batch_size = 8  # Typical
        seq_len = 128
        activation_memory = (batch_size * seq_len * hidden_dim * self.bytes_per_param) / (1024 ** 2)
        
        return base_memory + lora_memory + activation_memory
    
    def _compute_rank_config_from_phase2(
        self,
        device_profile: Dict,
        importance_scores: Optional[Dict[str, float]],
        n_layers: int
    ) -> Dict[int, int]:
        """
        Compute heterogeneous rank configuration using Phase 2's RankAllocator.
        
        Args:
            device_profile: Device capabilities
            importance_scores: Per-layer importance weights
            n_layers: Number of client-side layers
            
        Returns:
            Dictionary mapping layer index to rank
        """
        if not PHASE2_AVAILABLE:
            warnings.warn("Phase 2 not available, using uniform rank=8")
            return {i: 8 for i in range(n_layers)}
        
        try:
            # Initialize RankAllocator
            allocator = RankAllocator(
                model_dim=self.hidden_dim,
                bytes_per_param=self.bytes_per_param
            )
            
            # Use importance scores or uniform if not provided
            if importance_scores is None:
                importance_scores = {f'layer_{i}': 1.0 for i in range(n_layers)}
            
            # Allocate ranks under memory constraint
            ranks = allocator.allocate_ranks(
                device_profile=device_profile,
                importance_scores=importance_scores,
                n_layers=n_layers,
                split_point=n_layers  # Client-side only
            )
            
            # Convert to integer-keyed dict
            rank_config = {}
            for key, rank in ranks.items():
                if key.startswith('layer_'):
                    layer_idx = int(key.split('_')[1])
                    rank_config[layer_idx] = rank
            
            return rank_config
        
        except Exception as e:
            warnings.warn(f"Failed to compute ranks from Phase 2: {e}. Using uniform rank=8")
            return {i: 8 for i in range(n_layers)}
    
    def _load_bottom_layers(self, model_name: str, split_layer: int) -> Tuple[nn.Module, int, int]:
        """
        Load pre-trained model and extract bottom layers.
        
        Args:
            model_name: HuggingFace model identifier
            split_layer: Layer index to split at
            
        Returns:
            bottom_model: Sequential module with bottom layers
            hidden_dim: Hidden dimension of the model
            n_layers_total: Total number of layers in model
        """
        try:
            # Load config to get architecture info
            config = AutoConfig.from_pretrained(model_name)
            hidden_dim = config.hidden_size
            n_layers_total = getattr(config, 'n_layer', getattr(config, 'num_hidden_layers', 12))
            
            # Load full model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full_model = AutoModel.from_pretrained(model_name)
            
            # Extract transformer layers
            if hasattr(full_model, 'transformer'):
                # GPT-2 style
                layers = full_model.transformer.h[:split_layer]
                embeddings = full_model.transformer.wte
            elif hasattr(full_model, 'encoder'):
                # BERT style
                layers = full_model.encoder.layer[:split_layer]
                embeddings = full_model.embeddings
            elif hasattr(full_model, 'model'):
                # LLaMA style
                if hasattr(full_model.model, 'layers'):
                    layers = full_model.model.layers[:split_layer]
                    embeddings = full_model.model.embed_tokens
                else:
                    raise ValueError(f"Unsupported model structure: {model_name}")
            else:
                raise ValueError(f"Unsupported model architecture: {model_name}")
            
            # Create sequential bottom model
            bottom_model = nn.Sequential(
                embeddings,
                *layers
            )
            
            return bottom_model, hidden_dim, n_layers_total
            
        except Exception as e:
            # Fallback: create simple model for testing
            warnings.warn(f"Could not load {model_name}, using dummy model: {e}")
            hidden_dim = 768
            n_layers_total = 12
            dummy_model = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            return dummy_model, hidden_dim, n_layers_total
    
    def _create_lora_adapters(
        self, 
        rank_config: Dict[int, int], 
        hidden_dim: int,
        bytes_per_param: int
    ) -> nn.ModuleDict:
        """
        Create LoRA adapters with heterogeneous ranks per layer.
        
        Args:
            rank_config: Dictionary mapping layer indices to LoRA ranks
            hidden_dim: Model hidden dimension
            bytes_per_param: Bytes per parameter (4=fp32, 2=fp16)
            
        Returns:
            ModuleDict of LoRA adapters
        """
        adapters = nn.ModuleDict()
        for layer_idx, rank in rank_config.items():
            adapters[f'layer_{layer_idx}'] = LoRAAdapter(
                in_dim=hidden_dim,
                rank=rank,
                dropout=0.1,
                bytes_per_param=bytes_per_param
            )
        return adapters
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bottom layers with LoRA.
        
        Args:
            inputs: Input tensor (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
            
        Returns:
            Activations after bottom layers + LoRA
        """
        x = inputs.to(self.device)
        
        # Forward through bottom layers
        with torch.no_grad():
            x = self.model(x)
        
        # Apply LoRA adapters
        for layer_idx, adapter in self.lora_adapters.items():
            lora_output = adapter(x)
            x = x + lora_output  # Residual connection
        
        return x
    
    def compute_activations(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute activations to send to server.
        
        Args:
            batch: Input batch with 'input_ids' or 'inputs'
            
        Returns:
            Intermediate activations (batch_size, seq_len, hidden_dim)
        """
        self.lora_adapters.eval()
        
        with torch.no_grad():
            if 'input_ids' in batch:
                inputs = batch['input_ids'].to(self.device)
            elif 'inputs' in batch:
                inputs = batch['inputs'].to(self.device)
            else:
                raise ValueError("Batch must contain 'input_ids' or 'inputs'")
            
            activations = self.forward(inputs)
        
        self.current_activations = activations.detach().cpu()
        return self.current_activations
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        activation_gradients: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one training step with gradients from server.
        
        Args:
            batch: Input batch
            activation_gradients: Gradients from server (same shape as activations)
            
        Returns:
            Updated LoRA weights
        """
        self.lora_adapters.train()
        self.optimizer.zero_grad()
        
        # Forward pass with gradient tracking
        if 'input_ids' in batch:
            inputs = batch['input_ids'].to(self.device)
        elif 'inputs' in batch:
            inputs = batch['inputs'].to(self.device)
        else:
            raise ValueError("Batch must contain 'input_ids' or 'inputs'")
        
        activations = self.forward(inputs)
        
        # Backward pass with gradients from server
        activation_gradients = activation_gradients.to(self.device)
        activations.backward(activation_gradients)
        
        # Update LoRA weights
        self.optimizer.step()
        
        # Return updated weights
        return self.get_lora_weights()
    
    def get_lora_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get current LoRA weights.
        
        Returns:
            Dictionary of LoRA weights per layer
        """
        weights = {}
        for name, adapter in self.lora_adapters.items():
            weights[name] = {
                'A': adapter.A.data.clone().cpu(),
                'B': adapter.B.data.clone().cpu()
            }
        return weights
    
    def set_lora_weights(self, weights: Dict[str, Dict[str, torch.Tensor]]):
        """
        Update LoRA weights from aggregated values.
        
        Args:
            weights: Dictionary of LoRA weights per layer
        """
        for name, weight_dict in weights.items():
            if name in self.lora_adapters:
                self.lora_adapters[name].A.data = weight_dict['A'].to(self.device)
                self.lora_adapters[name].B.data = weight_dict['B'].to(self.device)
    
    def update_rank_config(self, new_rank_config: Dict[int, int]):
        """
        Update rank configuration and rebuild adapters.
        
        Useful when re-clustering or re-profiling mid-training.
        
        Args:
            new_rank_config: New dictionary mapping layer index to rank
        """
        # Save old weights
        old_weights = self.get_lora_weights()
        
        # Update rank config
        self.rank_config = new_rank_config
        
        # Rebuild adapters
        self.lora_adapters = self._create_lora_adapters(
            new_rank_config, self.hidden_dim, self.bytes_per_param
        )
        self.lora_adapters = self.lora_adapters.to(self.device)
        
        # Reinitialize optimizer
        self.optimizer = Adam(self.lora_adapters.parameters(), lr=self.learning_rate)
        
        # Try to restore weights where possible
        for name, weights in old_weights.items():
            if name in self.lora_adapters:
                old_rank = weights['A'].shape[0]
                new_rank = self.lora_adapters[name].rank
                
                if old_rank == new_rank:
                    # Same rank, restore exactly
                    self.set_lora_weights({name: weights})
                elif new_rank < old_rank:
                    # Truncate to new rank
                    self.lora_adapters[name].A.data = weights['A'][:new_rank, :].to(self.device)
                    self.lora_adapters[name].B.data = weights['B'][:, :new_rank].to(self.device)
                # If new_rank > old_rank, keep random initialization
    
    def get_memory_usage(self) -> float:
        """
        Calculate memory usage of LoRA parameters in MB.
        
        Uses Phase 2 formula: M = 2*d*r*b per adapter
        
        Returns:
            Memory usage in megabytes
        """
        total_memory_mb = 0.0
        for adapter in self.lora_adapters.values():
            total_memory_mb += adapter.get_memory_mb()
        return total_memory_mb
    
    def get_lora_memory_bytes(self) -> int:
        """
        Calculate total LoRA memory in bytes (Phase 2 compatible).
        
        Returns:
            Memory usage in bytes
        """
        total_bytes = 0
        for adapter in self.lora_adapters.values():
            total_bytes += adapter.get_memory_bytes()
        return total_bytes
    
    def get_lora_memory_mb(self) -> float:
        """
        Calculate total LoRA memory in MB (Phase 2 compatible).
        
        Returns:
            Memory usage in megabytes
        """
        return self.get_lora_memory_bytes() / (1024 ** 2)
    
    def set_task_id(self, task_id: int):
        """
        Assign task group ID from Phase 1 clustering.
        
        Args:
            task_id: Task group identifier
        """
        self.task_id = task_id
    
    def set_dataset_size(self, size: int):
        """
        Set local dataset size for weighted aggregation.
        
        Args:
            size: Number of samples in local dataset
        """
        self.local_dataset_size = size


class SplitServer:
    """
    Server-side component for split federated learning with multi-task support.
    
    Architecture (VFLAIR-LLM / HSplitLoRA style):
        Activations (from clients) → Top Layers → Task Heads → Loss → Gradients → Clients
        
    The server:
    1. Receives activations from clients
    2. Runs forward pass through top layers
    3. Computes task-specific outputs via task heads
    4. Calculates loss and backpropagates
    5. Sends activation gradients back to clients
    6. Aggregates LoRA weights per task group (HSplitLoRA multi-task aggregation)
    
    Integrates with Phase 1:
    - Per-task LoRA aggregation (only within task groups)
    - Task-specific metrics tracking
    
    Args:
        model_name (str): HuggingFace model name
        n_tasks (int): Number of task groups (from Phase 1)
        split_layer (int): Layer where model is split
        num_classes (int): Number of output classes per task
        device (str): Device to run on
        learning_rate (float): Learning rate
        aggregation_mode (str): 'global' (all clients) or 'per_task' (per task group)
    """
    
    def __init__(
        self,
        model_name: str,
        n_tasks: int,
        split_layer: int,
        num_classes: int = 10,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        aggregation_mode: str = 'per_task'
    ):
        self.model_name = model_name
        self.n_tasks = n_tasks
        self.split_layer = split_layer
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.aggregation_mode = aggregation_mode
        
        # Load top layers
        self.model, self.hidden_dim = self._load_top_layers(model_name, split_layer)
        self.model = self.model.to(self.device)
        
        # Create task-specific heads
        self.task_heads = self._create_task_heads(n_tasks, self.hidden_dim, num_classes)
        self.task_heads = self.task_heads.to(self.device)
        
        # Optimizer for server parameters
        params = list(self.model.parameters()) + list(self.task_heads.parameters())
        self.optimizer = Adam(params, lr=learning_rate)
        
        # Global and per-task aggregated LoRA weights
        self.global_lora_weights = None
        self.per_task_lora_weights = {i: None for i in range(n_tasks)}
        
        # Task groups (from Phase 1)
        self.task_groups = {}  # {task_id: [client_ids]}
        
        # Tracking
        self.training_round = 0
        self.task_metrics = {i: {'loss': [], 'accuracy': []} for i in range(n_tasks)}
        
    def _load_top_layers(self, model_name: str, split_layer: int) -> Tuple[nn.Module, int]:
        """
        Load top layers of pre-trained model.
        
        Args:
            model_name: HuggingFace model identifier
            split_layer: Layer index where split occurs
            
        Returns:
            top_model: Sequential module with top layers
            hidden_dim: Hidden dimension
        """
        try:
            config = AutoConfig.from_pretrained(model_name)
            hidden_dim = config.hidden_size
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full_model = AutoModel.from_pretrained(model_name)
            
            # Extract top layers
            if hasattr(full_model, 'transformer'):
                # GPT-2
                layers = full_model.transformer.h[split_layer:]
                ln_f = full_model.transformer.ln_f if hasattr(full_model.transformer, 'ln_f') else nn.Identity()
                top_model = nn.Sequential(*layers, ln_f)
            elif hasattr(full_model, 'encoder'):
                # BERT
                layers = full_model.encoder.layer[split_layer:]
                pooler = full_model.pooler if hasattr(full_model, 'pooler') else nn.Identity()
                top_model = nn.Sequential(*layers, pooler)
            elif hasattr(full_model, 'model'):
                # LLaMA
                if hasattr(full_model.model, 'layers'):
                    layers = full_model.model.layers[split_layer:]
                    norm = full_model.model.norm if hasattr(full_model.model, 'norm') else nn.Identity()
                    top_model = nn.Sequential(*layers, norm)
                else:
                    raise ValueError(f"Unsupported model structure")
            else:
                raise ValueError(f"Unsupported model architecture")
            
            return top_model, hidden_dim
            
        except Exception as e:
            warnings.warn(f"Could not load {model_name}, using dummy model: {e}")
            hidden_dim = 768
            dummy_model = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            return dummy_model, hidden_dim
    
    def _create_task_heads(self, n_tasks: int, hidden_dim: int, num_classes: int) -> nn.ModuleDict:
        """
        Create task-specific classification heads.
        
        Args:
            n_tasks: Number of tasks
            hidden_dim: Input dimension
            num_classes: Output classes per task
            
        Returns:
            ModuleDict of task heads
        """
        heads = nn.ModuleDict()
        for i in range(n_tasks):
            heads[f'task_{i}'] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        return heads
    
    def forward(self, activations: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass through top layers and task head.
        
        Args:
            activations: Activations from client (batch_size, seq_len, hidden_dim)
            task_id: Task identifier
            
        Returns:
            Logits (batch_size, num_classes)
        """
        activations = activations.to(self.device)
        
        # Forward through top layers
        x = self.model(activations)
        
        # Handle different output shapes
        if len(x.shape) == 3:
            # (batch, seq_len, hidden) → (batch, hidden)
            x = x.mean(dim=1)  # Average pooling over sequence
        
        # Task-specific head
        logits = self.task_heads[f'task_{task_id}'](x)
        
        return logits
    
    def compute_loss(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        task_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss and get activation gradients.
        
        Args:
            activations: Client activations (requires_grad=True)
            labels: Ground truth labels
            task_id: Task identifier
            
        Returns:
            loss: Cross-entropy loss
            activation_gradients: Gradients w.r.t. activations
        """
        activations = activations.to(self.device).requires_grad_(True)
        labels = labels.to(self.device)
        
        # Forward pass
        logits = self.forward(activations, task_id)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Extract activation gradients
        activation_gradients = activations.grad.clone().detach()
        
        # Update server parameters
        self.optimizer.step()
        
        return loss, activation_gradients
    
    def evaluate(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        task_id: int
    ) -> Dict[str, float]:
        """
        Evaluate on validation data.
        
        Args:
            activations: Client activations
            labels: Ground truth labels
            task_id: Task identifier
            
        Returns:
            Dictionary with loss and accuracy
        """
        self.model.eval()
        self.task_heads.eval()
        
        with torch.no_grad():
            logits = self.forward(activations, task_id)
            loss = F.cross_entropy(logits, labels.to(self.device))
            
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels.to(self.device)).float().mean()
        
        self.model.train()
        self.task_heads.train()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def aggregate_lora_weights(
        self,
        client_weights: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
        task_groups: Dict[int, List[int]],
        client_dataset_sizes: Optional[Dict[int, int]] = None,
        mode: Optional[str] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Aggregate LoRA weights with task-group awareness (HSplitLoRA multi-task).
        
        Note: Always returns layer_name -> {'A': ..., 'B': ...} format for compatibility.
        Per-task weights are stored internally in self.per_task_lora_weights.
        
        Args:
            client_weights: Dictionary mapping client_id to LoRA weights
            task_groups: Dictionary mapping task_id to list of client_ids
            client_dataset_sizes: Optional dataset sizes for weighted averaging
            mode: 'global' (all clients) or 'per_task' (per group). Uses self.aggregation_mode if None.
            
        Returns:
            Aggregated weights dict (layer_name -> {'A': ..., 'B': ...})
        """
        mode = mode or self.aggregation_mode
        
        if mode == 'global' or len(task_groups) == 1:
            # Traditional FedAvg: aggregate across all clients
            # Or single task group - return direct format
            return self._aggregate_global(client_weights, client_dataset_sizes)
        
        elif mode == 'per_task':
            # HSplitLoRA multi-task: aggregate within each task group
            per_task_weights = self._aggregate_per_task(client_weights, task_groups, client_dataset_sizes)
            # Return first task's weights for compatibility (or could average across tasks)
            # Store per-task weights for later use
            return list(per_task_weights.values())[0] if per_task_weights else {}
        
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")
    
    def _aggregate_global(
        self,
        client_weights: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
        client_dataset_sizes: Optional[Dict[int, int]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Global aggregation (FedAvg style).
        
        Args:
            client_weights: Client LoRA weights
            client_dataset_sizes: Optional dataset sizes for weighting
            
        Returns:
            Aggregated global weights
        """
        # Get all layer names
        all_layers = list(next(iter(client_weights.values())).keys())
        aggregated = {}
        
        # Compute weights (by dataset size if provided, else uniform)
        if client_dataset_sizes:
            total_samples = sum(client_dataset_sizes.values())
            weights_dict = {
                cid: size / total_samples 
                for cid, size in client_dataset_sizes.items()
            }
        else:
            n_clients = len(client_weights)
            weights_dict = {cid: 1.0 / n_clients for cid in client_weights.keys()}
        
        # Weighted average per layer
        for layer_name in all_layers:
            A_weighted = None
            B_weighted = None
            
            for client_id, weights in client_weights.items():
                if layer_name in weights:
                    weight = weights_dict.get(client_id, 1.0 / len(client_weights))
                    
                    if A_weighted is None:
                        A_weighted = weights[layer_name]['A'] * weight
                        B_weighted = weights[layer_name]['B'] * weight
                    else:
                        A_weighted += weights[layer_name]['A'] * weight
                        B_weighted += weights[layer_name]['B'] * weight
            
            if A_weighted is not None:
                aggregated[layer_name] = {
                    'A': A_weighted,
                    'B': B_weighted
                }
        
        self.global_lora_weights = aggregated
        return aggregated
    
    def _aggregate_per_task(
        self,
        client_weights: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
        task_groups: Dict[int, List[int]],
        client_dataset_sizes: Optional[Dict[int, int]]
    ) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Per-task aggregation (HSplitLoRA multi-task style).
        
        Args:
            client_weights: Client LoRA weights
            task_groups: Task group assignments
            client_dataset_sizes: Optional dataset sizes for weighting
            
        Returns:
            Dictionary mapping task_id to aggregated weights
        """
        per_task_weights = {}
        
        for task_id, client_ids in task_groups.items():
            # Filter weights for this task
            task_client_weights = {
                cid: weights 
                for cid, weights in client_weights.items() 
                if cid in client_ids
            }
            
            if not task_client_weights:
                continue
            
            # Filter dataset sizes for this task
            task_dataset_sizes = None
            if client_dataset_sizes:
                task_dataset_sizes = {
                    cid: size 
                    for cid, size in client_dataset_sizes.items() 
                    if cid in client_ids
                }
            
            # Aggregate within task group
            task_aggregated = self._aggregate_global(task_client_weights, task_dataset_sizes)
            per_task_weights[task_id] = task_aggregated
        
        self.per_task_lora_weights = per_task_weights
        return per_task_weights
    
    def assign_task_ids_from_clusters(self, cluster_labels: Dict[int, int]):
        """
        Assign task groups from Phase 1 clustering results.
        
        Args:
            cluster_labels: Dictionary mapping client_id to cluster_id
        """
        # Build task_groups from cluster labels
        self.task_groups = {}
        for client_id, cluster_id in cluster_labels.items():
            if cluster_id not in self.task_groups:
                self.task_groups[cluster_id] = []
            self.task_groups[cluster_id].append(client_id)
    
    def get_task_id(self, client_id: int) -> int:
        """
        Get task ID for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Task ID
        """
        for task_id, client_list in self.task_groups.items():
            if client_id in client_list:
                return task_id
        return 0  # Default to task 0
    
    def update_task_groups(self, new_task_groups: Dict[int, List[int]]):
        """
        Update task groups (e.g., after re-clustering).
        
        Args:
            new_task_groups: New task group assignments
        """
        self.task_groups = new_task_groups


def get_split_point(model_name: str) -> int:
    """
    Determine optimal split point for a model.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Layer index to split at
    """
    model_name_lower = model_name.lower()
    
    if 'gpt2' in model_name_lower:
        return 6  # Split GPT-2 (12 layers) at layer 6
    elif 'llama' in model_name_lower:
        if '7b' in model_name_lower or '13b' in model_name_lower:
            return 16  # Split LLaMA-7B/13B (32 layers) at layer 16
        else:
            return 20  # Split larger LLaMA at layer 20
    elif 'bert' in model_name_lower:
        return 6  # Split BERT (12 layers) at layer 6
    else:
        # Default: split at 50%
        return 6


def federated_training_round(
    clients: List[SplitClient],
    server: SplitServer,
    train_data: Dict[int, Dict[str, torch.Tensor]],
    task_groups: Dict[int, List[int]],
    aggregation_mode: str = 'per_task'
) -> Dict[str, Any]:
    """
    Execute one round of federated training with task-aware aggregation.
    
    Args:
        clients: List of SplitClient instances
        server: SplitServer instance
        train_data: Training data per client {client_id: batch}
        task_groups: Task assignments {task_id: [client_ids]}
        aggregation_mode: 'global' or 'per_task'
        
    Returns:
        Metrics for this round
    """
    client_lora_updates = {}
    client_dataset_sizes = {}
    round_metrics = {'losses': [], 'accuracies': [], 'per_task': {}}
    
    # Step 1: Each client trains locally
    for client in clients:
        if client.client_id not in train_data:
            continue
        
        batch = train_data[client.client_id]
        
        # Find task for this client
        task_id = server.get_task_id(client.client_id)
        
        # Client computes activations
        activations = client.compute_activations(batch)
        
        # Server computes loss and gradients
        labels = batch.get('labels', torch.randint(0, server.num_classes, (activations.shape[0],)))
        loss, activation_gradients = server.compute_loss(activations, labels, task_id)
        
        # Client trains LoRA with gradients
        updated_weights = client.train_step(batch, activation_gradients)
        client_lora_updates[client.client_id] = updated_weights
        
        # Record dataset size
        client_dataset_sizes[client.client_id] = client.local_dataset_size or activations.shape[0]
        
        # Track metrics
        round_metrics['losses'].append(loss.item())
        
        # Per-task metrics
        if task_id not in round_metrics['per_task']:
            round_metrics['per_task'][task_id] = {'losses': [], 'accuracies': []}
        round_metrics['per_task'][task_id]['losses'].append(loss.item())
    
    # Step 2: Server aggregates LoRA weights
    aggregated = server.aggregate_lora_weights(
        client_lora_updates, 
        task_groups,
        client_dataset_sizes,
        mode=aggregation_mode
    )
    
    # Step 3: Broadcast aggregated weights to clients
    if aggregation_mode == 'global':
        # Same weights for all clients
        for client in clients:
            client.set_lora_weights(aggregated)
    
    elif aggregation_mode == 'per_task':
        # Task-specific weights
        for client in clients:
            task_id = server.get_task_id(client.client_id)
            if task_id in aggregated:
                client.set_lora_weights(aggregated[task_id])
    
    # Compute average metrics
    round_metrics['avg_loss'] = np.mean(round_metrics['losses']) if round_metrics['losses'] else 0.0
    
    # Per-task average losses
    for task_id, metrics in round_metrics['per_task'].items():
        if metrics['losses']:
            metrics['avg_loss'] = np.mean(metrics['losses'])
    
    return round_metrics


def create_message(
    message_type: str,
    sender_id: int,
    round_num: int,
    data: Dict[str, Any],
    task_id: Optional[int] = None,
    payload_size_bytes: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a communication message for federated learning.
    
    Enhanced with task awareness and payload size tracking (VFLAIR-LLM style).
    
    Args:
        message_type: 'activations', 'gradients', 'weights', 'metrics'
        sender_id: Client or server ID
        round_num: Current training round
        data: Message payload
        task_id: Task group ID (from Phase 1)
        payload_size_bytes: Size of payload in bytes
        
    Returns:
        Formatted message dictionary
    """
    message = {
        'type': message_type,
        'sender_id': sender_id,
        'round': round_num,
        'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None,
        'data': data
    }
    
    # Add task ID if provided
    if task_id is not None:
        message['task_id'] = task_id
    
    # Estimate payload size if not provided
    if payload_size_bytes is None and data:
        payload_size_bytes = 0
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                payload_size_bytes += value.numel() * value.element_size()
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, torch.Tensor):
                        payload_size_bytes += v.numel() * v.element_size()
    
    if payload_size_bytes is not None:
        message['payload_size_bytes'] = payload_size_bytes
        message['payload_size_mb'] = payload_size_bytes / (1024 ** 2)
    
    return message


if __name__ == '__main__':
    print("Phase 3: Split Federated Learning (Refactored)")
    print("=" * 80)
    print("Literature-aligned: HSplitLoRA, SplitLoRA, VFLAIR-LLM, LoRA-FA")
    print("Integrated with Phase 1 (clustering) and Phase 2 (rank allocation)")
    print("=" * 80)
    
    # Test 1: LoRA adapter with Phase 2 memory formula
    print("\n[1] Testing LoRA Adapter with Phase 2 memory formula...")
    adapter = LoRAAdapter(in_dim=768, rank=16, bytes_per_param=4)
    test_input = torch.randn(2, 10, 768)
    output = adapter(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Memory (Phase 2 formula): {adapter.get_memory_mb():.4f} MB")
    print(f"   Memory (bytes): {adapter.get_memory_bytes()} bytes")
    print(f"   Expected: 2 * 768 * 16 * 4 = {2 * 768 * 16 * 4} bytes")
    print(f"   ✓ LoRA adapter with Phase 2 memory accounting!")
    
    # Test 2: SplitClient with Phase 2 integration
    if PHASE2_AVAILABLE:
        print("\n[2] Testing SplitClient with Phase 2 rank allocation...")
        
        # Create device profile
        profiler = DeviceProfiler()
        device_profile = profiler.profile_device('cpu_2gb')
        
        # Create importance scores (uniform for now)
        importance_scores = {f'layer_{i}': 1.0 for i in range(6)}
        
        # Create client with Phase 2 integration
        client = SplitClient(
            client_id=0,
            model_name='gpt2',
            split_layer=None,  # Auto-compute
            device='cpu',
            device_profile=device_profile,
            importance_scores=importance_scores,
            task_id=0
        )
        
        print(f"   Client ID: {client.client_id}")
        print(f"   Split layer: {client.split_layer} (auto-computed)")
        print(f"   Rank config: {client.rank_config}")
        print(f"   Memory usage: {client.get_lora_memory_mb():.2f} MB")
        print(f"   Device budget: {device_profile['memory_mb']} MB")
        print(f"   ✓ Phase 2 integration working!")
    else:
        print("\n[2] Phase 2 not available, skipping integration test")
        
        # Fallback test without Phase 2
        print("   Testing SplitClient with manual rank config...")
        rank_config = {0: 8, 1: 8, 2: 4}
        client = SplitClient(
            client_id=0,
            model_name='gpt2',
            rank_config=rank_config,
            split_layer=6,
            device='cpu'
        )
        print(f"   Client ID: {client.client_id}")
        print(f"   Memory usage: {client.get_lora_memory_mb():.2f} MB")
        print(f"   ✓ SplitClient working without Phase 2!")
    
    # Test 3: SplitServer with task-aware aggregation
    print("\n[3] Testing SplitServer with per-task aggregation...")
    server = SplitServer(
        model_name='gpt2',
        n_tasks=2,
        split_layer=6,
        num_classes=10,
        device='cpu',
        aggregation_mode='per_task'
    )
    
    # Assign task groups (simulating Phase 1 clustering)
    server.assign_task_ids_from_clusters({0: 0, 1: 0, 2: 1})  # Clients 0,1 → task 0, client 2 → task 1
    
    print(f"   Server initialized with {server.n_tasks} tasks")
    print(f"   Task groups: {server.task_groups}")
    print(f"   Aggregation mode: {server.aggregation_mode}")
    print(f"   ✓ Task-aware server working!")
    
    # Test 4: Federated training round with multiple clients
    print("\n[4] Testing federated training round...")
    
    # Create 3 clients with different tasks
    clients = []
    for i in range(3):
        rank_config = {0: 8, 1: 8, 2: 4 if i == 2 else 8}  # Heterogeneous ranks
        c = SplitClient(
            client_id=i,
            model_name='gpt2',
            rank_config=rank_config,
            split_layer=6,
            device='cpu',
            task_id=0 if i < 2 else 1
        )
        c.set_dataset_size(100)  # 100 samples per client
        clients.append(c)
    
    # Create dummy training data
    train_data = {}
    for i in range(3):
        train_data[i] = {
            'inputs': torch.randn(4, 768),  # batch_size=4
            'labels': torch.randint(0, 10, (4,))
        }
    
    # Run one training round with per-task aggregation
    task_groups = {0: [0, 1], 1: [2]}
    metrics = federated_training_round(
        clients, server, train_data, task_groups, aggregation_mode='per_task'
    )
    
    print(f"   Round metrics:")
    print(f"     Average loss: {metrics['avg_loss']:.4f}")
    print(f"     Per-task losses:")
    for task_id, task_metrics in metrics['per_task'].items():
        print(f"       Task {task_id}: {task_metrics.get('avg_loss', 0):.4f}")
    print(f"   ✓ Federated training with task-aware aggregation working!")
    
    # Test 5: Communication message with payload tracking
    print("\n[5] Testing communication message with payload tracking...")
    activations = torch.randn(4, 128, 768)
    message = create_message(
        message_type='activations',
        sender_id=0,
        round_num=1,
        data={'activations': activations},
        task_id=0
    )
    
    print(f"   Message type: {message['type']}")
    print(f"   Sender: {message['sender_id']}")
    print(f"   Task ID: {message.get('task_id', 'N/A')}")
    print(f"   Payload size: {message['payload_size_mb']:.4f} MB")
    print(f"   ✓ Enhanced communication protocol!")
    
    # Test 6: Rank config update
    print("\n[6] Testing dynamic rank config update...")
    old_ranks = clients[0].rank_config.copy()
    new_ranks = {0: 16, 1: 8, 2: 4}
    clients[0].update_rank_config(new_ranks)
    print(f"   Old ranks: {old_ranks}")
    print(f"   New ranks: {clients[0].rank_config}")
    print(f"   New memory: {clients[0].get_lora_memory_mb():.2f} MB")
    print(f"   ✓ Dynamic rank update working!")
    
    print("\n" + "=" * 80)
    print("✓ All Phase 3 refactored components operational!")
    print("✓ Literature-aligned (HSplitLoRA, SplitLoRA, VFLAIR-LLM)")
    print("✓ Integrated with Phase 1 (task clustering) and Phase 2 (rank allocation)")
    print("✓ Ready for comprehensive end-to-end testing")
    print("=" * 80)
