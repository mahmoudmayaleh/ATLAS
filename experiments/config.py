"""
Experiment Configuration for ATLAS Validation

Defines datasets, models, baselines, and hyperparameters for comprehensive experiments.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import torch


# ========== DATASET CONFIGURATIONS ==========

GLUE_TASKS = {
    "sst2": {
        "name": "sst2",  # Sentiment analysis
        "num_classes": 2,
        "metric": "accuracy",
        "max_length": 128,
        "n_clients": 10
    },
    "mrpc": {
        "name": "mrpc",  # Paraphrase detection
        "num_classes": 2,
        "metric": "f1",
        "max_length": 128,
        "n_clients": 8
    },
    "cola": {
        "name": "cola",  # Linguistic acceptability
        "num_classes": 2,
        "metric": "matthews_correlation",
        "max_length": 128,
        "n_clients": 8
    },
    "qnli": {
        "name": "qnli",  # Question NLI
        "num_classes": 2,
        "metric": "accuracy",
        "max_length": 256,
        "n_clients": 10
    }
}

SUMMARIZATION_TASKS = {
    "xsum": {
        "name": "xsum",
        "metric": "rouge",
        "max_length": 512,
        "n_clients": 8
    }
}


# ========== MODEL CONFIGURATIONS ==========

MODELS = {
    "distilbert": {
        "name": "distilbert-base-uncased",
        "type": "sequence_classification",
        "hidden_size": 768,
        "num_layers": 6,
        "full_params": 66_000_000,
        "full_size_mb": 268,
        # Experiment-specific hyperparameters
        "batch_size": 16,
        "learning_rate": 3e-5,
        "local_epochs": 3,
        "fingerprint_samples": 100,
        "fingerprint_batches": 30,
        "max_samples": 3000,
        "lora_ranks": [4, 8, 16, 32]
    },
    "gpt2": {
        "name": "gpt2",
        "type": "causal_lm",
        "hidden_size": 768,
        "num_layers": 12,
        "full_params": 124_000_000,
        "full_size_mb": 500,
        # Experiment-specific hyperparameters
        "batch_size": 16,
        "learning_rate": 3e-5,
        "local_epochs": 3,
        "fingerprint_samples": 100,
        "fingerprint_batches": 30,
        "max_samples": 3000,
        "lora_ranks": [4, 8, 16, 32]
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "type": "causal_lm",
        "hidden_size": 1024,
        "num_layers": 24,
        "full_params": 355_000_000,
        "full_size_mb": 1400,
        "local_epochs": 3
    },
    "gpt2-xl": {
        "name": "gpt2-xl",
        "type": "causal_lm",
        "hidden_size": 1600,
        "num_layers": 48,
        "full_params": 1_558_000_000,
        "full_size_mb": 6200,
        # Experiment-specific hyperparameters
        "batch_size": 8,
        "learning_rate": 2e-5,
        "local_epochs": 3,
        "fingerprint_samples": 25,
        "fingerprint_batches": 20,
        "max_samples": 3000,
        "lora_ranks": [4, 8, 16, 32, 64]
    },
    "qwen2.5": {
        "name": "Qwen/Qwen2.5-0.5B",
        "type": "causal_lm",
        "hidden_size": 1024,
        "num_layers": 24,
        "full_params": 494_000_000,
        "full_size_mb": 1976,
        # Experiment-specific hyperparameters
        "batch_size": 12,
        "learning_rate": 3e-5,
        "local_epochs": 3,
        "fingerprint_samples": 80,
        "fingerprint_batches": 25,
        "max_samples": 3000,
        "lora_ranks": [4, 8, 16, 32, 48]
    },
    "bert-base": {
        "name": "bert-base-uncased",
        "type": "sequence_classification",
        "hidden_size": 768,
        "num_layers": 12,
        "full_params": 110_000_000,
        "full_size_mb": 440,
        "local_epochs": 3
    }
}


# ========== DEVICE PROFILES ==========

DEVICE_PROFILES = {
    "smartphone": {
        "name": "smartphone",
        "total_memory_mb": 2048,
        "compute_capability": 0.3,
        "bandwidth_mbps": 10,
        "description": "Low-end mobile device"
    },
    "tablet": {
        "name": "tablet",
        "total_memory_mb": 4096,
        "compute_capability": 0.5,
        "bandwidth_mbps": 20,
        "description": "Mid-tier mobile device"
    },
    "laptop_cpu": {
        "name": "laptop_cpu",
        "total_memory_mb": 8192,
        "compute_capability": 1.0,
        "bandwidth_mbps": 50,
        "description": "Consumer laptop (CPU only)"
    },
    "laptop_gpu": {
        "name": "laptop_gpu",
        "total_memory_mb": 16384,
        "compute_capability": 2.0,
        "bandwidth_mbps": 100,
        "description": "Gaming laptop with GPU"
    },
    "workstation": {
        "name": "workstation",
        "total_memory_mb": 32768,
        "compute_capability": 4.0,
        "bandwidth_mbps": 200,
        "description": "High-end workstation"
    }
}


# ========== TRAINING HYPERPARAMETERS ==========

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # General
    num_rounds: int = 50
    local_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-4
    
    # LoRA
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Split Learning
    split_layer: int = 6  # Middle of 12-layer model
    
    # Clustering
    n_clusters_range: tuple = (2, 5)
    fingerprint_dim: int = 64
    
    # Laplacian
    lambda_laplacian: float = 0.1
    k_neighbors: int = 3
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ========== BASELINE CONFIGURATIONS ==========

BASELINES = {
    "atlas": {
        "name": "ATLAS (Full System)",
        "description": "Task clustering + heterogeneous ranks + split learning + Laplacian",
        "heterogeneous": True,
        "task_aware": True,
        "split_learning": True,
        "laplacian": True
    },
    "atlas_no_laplacian": {
        "name": "ATLAS without Laplacian",
        "description": "Task clustering + heterogeneous ranks + split learning (no Laplacian)",
        "heterogeneous": True,
        "task_aware": True,
        "split_learning": True,
        "laplacian": False
    },
    "fedavg_cluster": {
        "name": "Task-aware FedAvg",
        "description": "Task clustering with FedAvg (no heterogeneous, no split, no Laplacian)",
        "heterogeneous": False,
        "task_aware": True,
        "split_learning": False,
        "laplacian": False
    },
    "standard_fl": {
        "name": "Standard Federated Learning (FedAvg)",
        "description": "Pure global FedAvg across all clients (no clustering, no heterogeneous, no split, no Laplacian)",
        "heterogeneous": False,
        "task_aware": False,
        "split_learning": False,
        "laplacian": False
    },
    "local_only": {
        "name": "Local Training Only",
        "description": "No aggregation (baseline for comparison)",
        "heterogeneous": False,
        "task_aware": False,
        "split_learning": False,
        "laplacian": False
    }
}


# ========== EXPERIMENT CONFIGURATIONS ==========

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str
    dataset: str
    model: str
    baseline: str
    device_mix: List[str]  # Device types for clients
    training: TrainingConfig
    
    def __post_init__(self):
        if self.training is None:
            self.training = TrainingConfig()


# Predefined experiment sets
QUICK_EXPERIMENTS = [
    ExperimentConfig(
        name="quick_sst2_atlas",
        dataset="sst2",
        model="gpt2",
        baseline="atlas",
        device_mix=["smartphone"] * 3 + ["tablet"] * 3 + ["laptop_cpu"] * 4,
        training=TrainingConfig(num_rounds=20)
    ),
    ExperimentConfig(
        name="quick_sst2_baseline",
        dataset="sst2",
        model="gpt2",
        baseline="homogeneous_lora",
        device_mix=["laptop_cpu"] * 10,
        training=TrainingConfig(num_rounds=20)
    )
]

FULL_EXPERIMENTS = [
    # SST-2 experiments (all baselines)
    ExperimentConfig(
        name="sst2_standard_fl",
        dataset="sst2",
        model="gpt2",
        baseline="standard_fl",
        device_mix=["laptop_cpu"] * 10,
        training=TrainingConfig()
    ),
    ExperimentConfig(
        name="sst2_homogeneous_lora",
        dataset="sst2",
        model="gpt2",
        baseline="homogeneous_lora",
        device_mix=["laptop_cpu"] * 10,
        training=TrainingConfig()
    ),
    ExperimentConfig(
        name="sst2_hsplitlora",
        dataset="sst2",
        model="gpt2",
        baseline="hsplitlora",
        device_mix=["smartphone"] * 3 + ["tablet"] * 3 + ["laptop_cpu"] * 4,
        training=TrainingConfig()
    ),
    ExperimentConfig(
        name="sst2_atlas",
        dataset="sst2",
        model="gpt2",
        baseline="atlas",
        device_mix=["smartphone"] * 3 + ["tablet"] * 3 + ["laptop_cpu"] * 4,
        training=TrainingConfig()
    ),
    
    # Multi-task experiments (MRPC, CoLA, QNLI)
    ExperimentConfig(
        name="multitask_atlas",
        dataset="multitask",  # Special handling for multiple GLUE tasks
        model="gpt2",
        baseline="atlas",
        device_mix=["smartphone"] * 8 + ["tablet"] * 8 + ["laptop_cpu"] * 12,
        training=TrainingConfig(num_rounds=100)
    ),
]


# ========== EVALUATION METRICS ==========

METRICS = {
    "accuracy": "Classification accuracy",
    "f1": "F1 score",
    "matthews_correlation": "Matthews correlation coefficient",
    "rouge": "ROUGE score for summarization",
    "memory_mb": "Peak memory usage (MB)",
    "comm_mb_per_round": "Communication cost per round (MB)",
    "training_time_sec": "Training time (seconds)",
    "convergence_rounds": "Rounds to convergence",
    "privacy_dcs": "Data Characteristic Score (privacy metric)"
}


# ========== UTILITY FUNCTIONS ==========

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for a dataset"""
    if dataset_name in GLUE_TASKS:
        return GLUE_TASKS[dataset_name]
    elif dataset_name in SUMMARIZATION_TASKS:
        return SUMMARIZATION_TASKS[dataset_name]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a model"""
    # Map common aliases to our model keys
    model_map = {
        "distilbert": "distilbert",
        "distilbert-base-uncased": "distilbert",
        "gpt2": "gpt2",
        "gpt2-xl": "gpt2-xl",
        "gpt2xl": "gpt2-xl",
        "qwen2.5": "qwen2.5",
        "qwen": "qwen2.5",
        "Qwen/Qwen2.5-0.5B": "qwen2.5",
        "qwen-0.5b": "qwen2.5",
        "bert-base-uncased": "bert-base",
        "bert-base": "bert-base"
    }
    
    # Normalize model name
    normalized_name = model_map.get(model_name, model_name)
    
    if normalized_name in MODELS:
        return MODELS[normalized_name]
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")


def get_model_hyperparameters(model_name: str) -> Dict[str, Any]:
    """
    Get experiment-specific hyperparameters for a model.
    Returns defaults if model doesn't have specific hyperparameters.
    """
    config = get_model_config(model_name)
    
    # Extract hyperparameters with defaults
    return {
        'batch_size': config.get('batch_size', 16),
        'learning_rate': config.get('learning_rate', 2e-5),
        'local_epochs': config.get('local_epochs', 3),
        'fingerprint_samples': config.get('fingerprint_samples', 50),
        'fingerprint_batches': config.get('fingerprint_batches', 20),
        'max_samples': config.get('max_samples', 3000),
        'lora_ranks': config.get('lora_ranks', [4, 8, 16, 32]),
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers']
    }


def get_device_profile(device_type: str) -> Dict[str, Any]:
    """Get device profile"""
    if device_type in DEVICE_PROFILES:
        return DEVICE_PROFILES[device_type]
    else:
        raise ValueError(f"Unknown device type: {device_type}")


def create_heterogeneous_clients(device_mix: List[str], n_clients: int = None):
    """
    Create heterogeneous client configuration
    
    Args:
        device_mix: List of device types for each client
        n_clients: Number of clients (if None, use len(device_mix))
    
    Returns:
        List of device profiles for each client
    """
    if n_clients is None:
        n_clients = len(device_mix)
    
    client_configs = []
    for i in range(n_clients):
        device_type = device_mix[i % len(device_mix)]
        profile = get_device_profile(device_type).copy()
        profile['client_id'] = i
        client_configs.append(profile)
    
    return client_configs


if __name__ == "__main__":
    # Test configurations
    print("=" * 60)
    print("ATLAS EXPERIMENT CONFIGURATION")
    print("=" * 60)
    
    print("\n📊 Available Datasets:")
    for task in GLUE_TASKS:
        print(f"  - {task}: {GLUE_TASKS[task]['metric']}")
    
    print("\n🤖 Available Models:")
    for model in MODELS:
        print(f"  - {model}: {MODELS[model]['full_params']:,} params ({MODELS[model]['full_size_mb']} MB)")
    
    print("\n📱 Device Profiles:")
    for device in DEVICE_PROFILES:
        print(f"  - {device}: {DEVICE_PROFILES[device]['total_memory_mb']} MB, "
              f"{DEVICE_PROFILES[device]['bandwidth_mbps']} Mbps")
    
    print("\n🔬 Baselines:")
    for baseline in BASELINES:
        print(f"  - {baseline}: {BASELINES[baseline]['description']}")
    
    print("\n✅ Quick experiments:", len(QUICK_EXPERIMENTS))
    print("✅ Full experiments:", len(FULL_EXPERIMENTS))
    print("\n" + "=" * 60)
