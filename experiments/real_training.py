"""
Real Training Implementation for ATLAS
Optimized for Colab T4 GPU (3-4 hour window)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass


@dataclass
class TrainingResult:
    """Results from training a model"""
    accuracy: float
    loss: float
    time_seconds: float
    memory_mb: float
    communication_mb: float


class RealFederatedTrainer:
    """
    Real federated learning trainer with actual PyTorch training
    Optimized for T4 GPU constraints (16GB VRAM, 3-4 hour time limit)
    """
    
    def __init__(
        self,
        model_name: str,
        task_name: str,
        num_clients: int = 10,
        local_epochs: int = 1,
        batch_size: int = 8,
        max_samples: int = 500,  # Limit samples per client for speed
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.num_clients = num_clients
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.device = device
        
        print(f"[INIT] Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Dataset mapping
        self.dataset_map = {
            'sst2': ('stanfordnlp/sst2', 'sentence', None, 2),
            'mrpc': ('nyu-mll/glue', 'sentence1', 'sentence2', 2),
            'cola': ('nyu-mll/glue', 'sentence', None, 2),
            'qnli': ('nyu-mll/glue', 'question', 'sentence', 2),
        }
        
        # Load and prepare dataset
        self.train_dataset, self.test_dataset = self._load_dataset()
        
    def _load_dataset(self):
        """Load and tokenize dataset"""
        if self.task_name not in self.dataset_map:
            raise ValueError(f"Unknown task: {self.task_name}")
        
        dataset_name, text_col, text_col2, num_labels = self.dataset_map[self.task_name]
        
        print(f"[DATA] Loading {dataset_name} for task {self.task_name}...")
        
        # Load dataset
        if self.task_name == 'sst2':
            dataset = load_dataset(dataset_name, split='train')
            test_dataset = load_dataset(dataset_name, split='validation')
        else:
            dataset = load_dataset(dataset_name, self.task_name, split='train')
            test_dataset = load_dataset(dataset_name, self.task_name, split='validation')
        
        # Tokenize
        def tokenize_fn(examples):
            if text_col2:
                texts = [(t1, t2) for t1, t2 in zip(examples[text_col], examples[text_col2])]
                return self.tokenizer(
                    texts,
                    padding='max_length',
                    truncation=True,
                    max_length=128
                )
            else:
                return self.tokenizer(
                    examples[text_col],
                    padding='max_length',
                    truncation=True,
                    max_length=128
                )
        
        print(f"[DATA] Tokenizing...")
        dataset = dataset.map(tokenize_fn, batched=True, load_from_cache_file=False)
        test_dataset = test_dataset.map(tokenize_fn, batched=True, load_from_cache_file=False)
        
        # Set format
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        print(f"[DATA] Train size: {len(dataset)}, Test size: {len(test_dataset)}")
        
        return dataset, test_dataset
    
    def _create_model(self) -> nn.Module:
        """Create a fresh model instance"""
        _, _, _, num_labels = self.dataset_map[self.task_name]
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        
        # For GPT-2, set pad token id
        if 'gpt2' in self.model_name.lower():
            model.config.pad_token_id = self.tokenizer.pad_token_id
        
        return model.to(self.device)
    
    def _partition_data(self) -> List[Subset]:
        """Partition data across clients (IID for now)"""
        indices = list(range(len(self.train_dataset)))
        np.random.shuffle(indices)
        
        # Limit total samples to speed up training
        total_samples = min(len(indices), self.max_samples * self.num_clients)
        indices = indices[:total_samples]
        
        # Partition
        client_size = len(indices) // self.num_clients
        client_datasets = []
        
        for i in range(self.num_clients):
            start = i * client_size
            end = start + client_size if i < self.num_clients - 1 else len(indices)
            client_indices = indices[start:end]
            client_datasets.append(Subset(self.train_dataset, client_indices))
        
        print(f"[DATA] Partitioned into {self.num_clients} clients, ~{client_size} samples each")
        
        return client_datasets
    
    def train_client(
        self,
        model: nn.Module,
        client_dataset: Subset,
        learning_rate: float = 2e-5
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Train a single client and return updated weights"""
        
        model.train()
        dataloader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Return updated weights
        return {name: param.data.clone() for name, param in model.named_parameters()}, avg_loss
    
    def aggregate_weights(
        self,
        client_weights_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """FedAvg aggregation"""
        
        if not client_weights_list:
            return {}
        
        # Average weights
        aggregated = {}
        for key in client_weights_list[0].keys():
            aggregated[key] = torch.stack([w[key] for w in client_weights_list]).mean(dim=0)
        
        return aggregated
    
    def evaluate(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluate model on test set"""
        
        model.eval()
        dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size * 2)
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
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
    
    def run_federated_training(
        self,
        num_rounds: int = 10,
        clients_per_round: int = 10,
        learning_rate: float = 2e-5
    ) -> List[Dict[str, float]]:
        """Run federated training for specified rounds"""
        
        print(f"\n[TRAIN] Starting federated training for {num_rounds} rounds")
        print(f"[TRAIN] {clients_per_round} clients per round, {self.local_epochs} local epochs")
        
        # Create global model
        global_model = self._create_model()
        
        # Partition data
        client_datasets = self._partition_data()
        
        # Training loop
        results = []
        
        for round_idx in range(num_rounds):
            round_start = time.time()
            
            print(f"\n[ROUND {round_idx + 1}/{num_rounds}]")
            
            # Select clients
            selected_clients = np.random.choice(
                self.num_clients,
                size=min(clients_per_round, self.num_clients),
                replace=False
            )
            
            # Train clients
            client_weights_list = []
            client_losses = []
            
            for client_id in selected_clients:
                # Create client model (copy of global)
                client_model = self._create_model()
                client_model.load_state_dict(global_model.state_dict())
                
                # Train client
                weights, loss = self.train_client(
                    client_model,
                    client_datasets[client_id],
                    learning_rate
                )
                
                client_weights_list.append(weights)
                client_losses.append(loss)
                
                # Clear memory
                del client_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Aggregate
            aggregated_weights = self.aggregate_weights(client_weights_list)
            global_model.load_state_dict(aggregated_weights)
            
            # Evaluate
            accuracy, test_loss = self.evaluate(global_model)
            
            round_time = time.time() - round_start
            
            # Get memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_mb = 0.0
            
            # Estimate communication cost (model size)
            param_count = sum(p.numel() for p in global_model.parameters())
            comm_mb = (param_count * 4 * len(selected_clients)) / (1024 * 1024)  # 4 bytes per float32
            
            result = {
                'round': round_idx + 1,
                'accuracy': accuracy,
                'loss': test_loss,
                'train_loss': np.mean(client_losses),
                'time_seconds': round_time,
                'memory_mb': memory_mb,
                'communication_mb': comm_mb
            }
            
            results.append(result)
            
            print(f"  Accuracy: {accuracy:.4f}, Loss: {test_loss:.4f}, Time: {round_time:.1f}s")
            
        return results


class LoRAFederatedTrainer(RealFederatedTrainer):
    """
    Federated learning with LoRA adapters
    More memory efficient for T4 GPU
    """
    
    def __init__(self, rank: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
    
    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA to model"""
        from peft import get_peft_model, LoraConfig, TaskType
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.rank,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"] if "bert" in self.model_name.lower() 
                          else ["c_attn", "c_proj"]
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def _create_model(self) -> nn.Module:
        """Create model with LoRA adapters"""
        model = super()._create_model()
        model = self._apply_lora(model)
        return model
    
    def aggregate_weights(
        self,
        client_weights_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate only LoRA parameters"""
        
        if not client_weights_list:
            return {}
        
        # Only aggregate trainable parameters (LoRA adapters)
        lora_keys = [k for k in client_weights_list[0].keys() if 'lora' in k.lower()]
        
        aggregated = {}
        for key in client_weights_list[0].keys():
            if key in lora_keys:
                # Average LoRA parameters
                aggregated[key] = torch.stack([w[key] for w in client_weights_list]).mean(dim=0)
            else:
                # Keep base model parameters from first client (they should be identical)
                aggregated[key] = client_weights_list[0][key]
        
        return aggregated


def run_quick_experiment(
    experiment_name: str,
    model_name: str = "distilbert-base-uncased",  # Faster than BERT/GPT-2
    task_name: str = "sst2",
    num_rounds: int = 5,
    num_clients: int = 5,
    use_lora: bool = False,
    lora_rank: int = 8
) -> Dict[str, any]:
    """
    Run a quick experiment optimized for Colab T4 GPU (3-4 hours)
    
    Args:
        experiment_name: Name for saving results
        model_name: Model to use (smaller = faster)
        task_name: GLUE task
        num_rounds: Number of FL rounds (5-10 for quick experiments)
        num_clients: Number of clients (5-10 for quick experiments)
        use_lora: Whether to use LoRA (recommended for memory efficiency)
        lora_rank: LoRA rank (lower = faster)
    """
    
    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"Model: {model_name}, Task: {task_name}")
    print(f"Rounds: {num_rounds}, Clients: {num_clients}")
    print(f"LoRA: {use_lora}" + (f" (rank={lora_rank})" if use_lora else ""))
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Create trainer
    if use_lora:
        trainer = LoRAFederatedTrainer(
            model_name=model_name,
            task_name=task_name,
            num_clients=num_clients,
            rank=lora_rank,
            local_epochs=1,
            batch_size=16,  # Larger batch for LoRA
            max_samples=300  # Limit samples for speed
        )
    else:
        trainer = RealFederatedTrainer(
            model_name=model_name,
            task_name=task_name,
            num_clients=num_clients,
            local_epochs=1,
            batch_size=8,
            max_samples=300
        )
    
    # Run training
    results = trainer.run_federated_training(
        num_rounds=num_rounds,
        clients_per_round=num_clients
    )
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"[DONE] Total time: {total_time/60:.1f} minutes")
    print(f"[DONE] Final accuracy: {results[-1]['accuracy']:.4f}")
    print(f"{'='*70}\n")
    
    # Return summary
    return {
        'experiment_name': experiment_name,
        'model_name': model_name,
        'task_name': task_name,
        'num_rounds': num_rounds,
        'num_clients': num_clients,
        'use_lora': use_lora,
        'lora_rank': lora_rank if use_lora else None,
        'total_time_minutes': total_time / 60,
        'final_accuracy': results[-1]['accuracy'],
        'final_loss': results[-1]['loss'],
        'round_results': results
    }
