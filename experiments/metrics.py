"""
Metrics Collection and Analysis for ATLAS Experiments

Provides tools for tracking, logging, and analyzing performance metrics
during federated learning experiments.
"""

import torch
import psutil
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    current_mb: float
    peak_mb: float
    allocated_mb: float  # GPU only
    reserved_mb: float  # GPU only
    device: str
    
    @staticmethod
    def capture(device: str = "cpu"):
        """Capture current memory metrics"""
        if device == "cuda" and torch.cuda.is_available():
            return MemoryMetrics(
                current_mb=torch.cuda.memory_allocated() / (1024**2),
                peak_mb=torch.cuda.max_memory_allocated() / (1024**2),
                allocated_mb=torch.cuda.memory_allocated() / (1024**2),
                reserved_mb=torch.cuda.memory_reserved() / (1024**2),
                device="cuda"
            )
        else:
            process = psutil.Process()
            mem_info = process.memory_info()
            return MemoryMetrics(
                current_mb=mem_info.rss / (1024**2),
                peak_mb=mem_info.rss / (1024**2),
                allocated_mb=0,
                reserved_mb=0,
                device="cpu"
            )


@dataclass
class CommunicationMetrics:
    """Communication cost metrics"""
    upload_mb: float  # Client -> Server
    download_mb: float  # Server -> Client
    total_mb: float
    num_parameters: int
    compression_ratio: float = 1.0
    
    @staticmethod
    def compute(model_params: Dict[str, torch.Tensor], 
                compression: bool = False) -> 'CommunicationMetrics':
        """Compute communication cost for model parameters"""
        total_bytes = 0
        num_params = 0
        
        for name, param in model_params.items():
            param_bytes = param.numel() * param.element_size()
            total_bytes += param_bytes
            num_params += param.numel()
        
        total_mb = total_bytes / (1024**2)
        compression_ratio = 0.5 if compression else 1.0
        
        return CommunicationMetrics(
            upload_mb=total_mb * compression_ratio,
            download_mb=total_mb * compression_ratio,
            total_mb=total_mb * 2 * compression_ratio,
            num_parameters=num_params,
            compression_ratio=compression_ratio
        )


@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    loss: float
    accuracy: float
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    perplexity: Optional[float] = None
    
    def to_dict(self):
        return {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'perplexity': self.perplexity
        }


@dataclass
class RoundMetrics:
    """Metrics for a single federated learning round"""
    round_num: int
    training: TrainingMetrics
    memory: MemoryMetrics
    communication: CommunicationMetrics
    time_sec: float
    num_clients: int
    
    # Task clustering info (if applicable)
    task_group_id: Optional[int] = None
    silhouette_score: Optional[float] = None
    
    def to_dict(self):
        return {
            'round': self.round_num,
            'training': self.training.to_dict(),
            'memory': {
                'current_mb': self.memory.current_mb,
                'peak_mb': self.memory.peak_mb,
                'device': self.memory.device
            },
            'communication': {
                'upload_mb': self.communication.upload_mb,
                'download_mb': self.communication.download_mb,
                'total_mb': self.communication.total_mb,
                'num_parameters': self.communication.num_parameters
            },
            'time_sec': self.time_sec,
            'num_clients': self.num_clients,
            'task_group_id': self.task_group_id,
            'silhouette_score': self.silhouette_score
        }


class MetricsLogger:
    """Comprehensive metrics logger for experiments"""
    
    def __init__(self, experiment_name: str, save_dir: str = "./results"):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.round_metrics: List[RoundMetrics] = []
        self.client_metrics: Dict[int, List[Dict]] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Tracking
        self.start_time = time.time()
        self.round_start_time = None
        
    def log_round_start(self, round_num: int):
        """Mark the start of a training round"""
        self.round_start_time = time.time()
        
    def log_round_end(self, round_num: int,
                     training_metrics: TrainingMetrics,
                     memory_metrics: MemoryMetrics,
                     comm_metrics: CommunicationMetrics,
                     num_clients: int,
                     task_group_id: Optional[int] = None,
                     silhouette_score: Optional[float] = None):
        """Log metrics at the end of a training round"""
        round_time = time.time() - self.round_start_time
        
        metrics = RoundMetrics(
            round_num=round_num,
            training=training_metrics,
            memory=memory_metrics,
            communication=comm_metrics,
            time_sec=round_time,
            num_clients=num_clients,
            task_group_id=task_group_id,
            silhouette_score=silhouette_score
        )
        
        self.round_metrics.append(metrics)
        
    def log_client_metrics(self, client_id: int, metrics: Dict[str, Any]):
        """Log per-client metrics"""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = []
        
        self.client_metrics[client_id].append({
            'timestamp': time.time(),
            **metrics
        })
    
    def set_metadata(self, key: str, value: Any):
        """Set experiment metadata"""
        self.metadata[key] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.round_metrics:
            return {}
        
        # Extract metrics
        accuracies = [m.training.accuracy for m in self.round_metrics]
        losses = [m.training.loss for m in self.round_metrics]
        memory_peaks = [m.memory.peak_mb for m in self.round_metrics]
        comm_costs = [m.communication.total_mb for m in self.round_metrics]
        round_times = [m.time_sec for m in self.round_metrics]
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_rounds': len(self.round_metrics),
            'total_time_sec': time.time() - self.start_time,
            
            # Accuracy stats
            'final_accuracy': accuracies[-1],
            'best_accuracy': max(accuracies),
            'worst_accuracy': min(accuracies),
            'avg_accuracy': np.mean(accuracies),
            
            # Loss stats
            'final_loss': losses[-1],
            'best_loss': min(losses),
            'avg_loss': np.mean(losses),
            
            # Memory stats
            'peak_memory_mb': max(memory_peaks),
            'avg_memory_mb': np.mean(memory_peaks),
            
            # Communication stats
            'total_comm_mb': sum(comm_costs),
            'avg_comm_per_round_mb': np.mean(comm_costs),
            
            # Time stats
            'avg_round_time_sec': np.mean(round_times),
            'total_training_time_sec': sum(round_times),
            
            # Convergence
            'convergence_round': self._find_convergence_round(accuracies),
            
            # Metadata
            'metadata': self.metadata
        }
        
        return summary
    
    def _find_convergence_round(self, accuracies: List[float], 
                               threshold: float = 0.95) -> Optional[int]:
        """Find the round where model converged (reached threshold of best accuracy)"""
        if not accuracies:
            return None
        
        best_acc = max(accuracies)
        target_acc = best_acc * threshold
        
        for i, acc in enumerate(accuracies):
            if acc >= target_acc:
                return i + 1
        
        return len(accuracies)
    
    def save(self):
        """Save all metrics to disk"""
        # Save round-by-round metrics
        rounds_path = self.save_dir / f"{self.experiment_name}_rounds.json"
        with open(rounds_path, 'w') as f:
            json.dump([m.to_dict() for m in self.round_metrics], f, indent=2)
        
        # Save client metrics
        if self.client_metrics:
            clients_path = self.save_dir / f"{self.experiment_name}_clients.json"
            with open(clients_path, 'w') as f:
                json.dump(self.client_metrics, f, indent=2)
        
        # Save summary
        summary_path = self.save_dir / f"{self.experiment_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        
        print(f"✅ Metrics saved to {self.save_dir}")
        print(f"   - {rounds_path.name}")
        print(f"   - {summary_path.name}")
        if self.client_metrics:
            print(f"   - {clients_path.name}")
    
    def print_summary(self):
        """Print summary to console"""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print(f"📊 EXPERIMENT SUMMARY: {self.experiment_name}")
        print("=" * 70)
        
        print(f"\n🎯 Performance:")
        print(f"   Final Accuracy: {summary['final_accuracy']:.4f}")
        print(f"   Best Accuracy:  {summary['best_accuracy']:.4f}")
        print(f"   Final Loss:     {summary['final_loss']:.4f}")
        print(f"   Convergence:    Round {summary['convergence_round']}")
        
        print(f"\n💾 Memory:")
        print(f"   Peak Memory:    {summary['peak_memory_mb']:.1f} MB")
        print(f"   Avg Memory:     {summary['avg_memory_mb']:.1f} MB")
        
        print(f"\n📡 Communication:")
        print(f"   Total Comm:     {summary['total_comm_mb']:.2f} MB")
        print(f"   Per Round:      {summary['avg_comm_per_round_mb']:.2f} MB")
        
        print(f"\n⏱️  Time:")
        print(f"   Total Time:     {summary['total_time_sec']:.1f} sec")
        print(f"   Per Round:      {summary['avg_round_time_sec']:.2f} sec")
        print(f"   Total Rounds:   {summary['total_rounds']}")
        
        print("=" * 70 + "\n")


class ComparisonAnalyzer:
    """Analyze and compare multiple experiments"""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.experiments = {}
        
    def load_experiments(self, experiment_names: List[str]):
        """Load multiple experiment results"""
        for name in experiment_names:
            summary_path = self.results_dir / f"{name}_summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    self.experiments[name] = json.load(f)
            else:
                print(f"⚠️  Warning: {summary_path} not found")
    
    def compare_accuracy(self) -> Dict[str, float]:
        """Compare final accuracies"""
        return {name: exp['final_accuracy'] 
                for name, exp in self.experiments.items()}
    
    def compare_memory(self) -> Dict[str, float]:
        """Compare peak memory usage"""
        return {name: exp['peak_memory_mb'] 
                for name, exp in self.experiments.items()}
    
    def compare_communication(self) -> Dict[str, float]:
        """Compare total communication cost"""
        return {name: exp['total_comm_mb'] 
                for name, exp in self.experiments.items()}
    
    def compare_convergence(self) -> Dict[str, int]:
        """Compare convergence rounds"""
        return {name: exp['convergence_round'] 
                for name, exp in self.experiments.items()}
    
    def print_comparison_table(self):
        """Print comparison table"""
        if not self.experiments:
            print("No experiments loaded!")
            return
        
        print("\n" + "=" * 100)
        print("📊 EXPERIMENT COMPARISON")
        print("=" * 100)
        
        # Header
        print(f"{'Experiment':<30} {'Accuracy':<12} {'Memory (MB)':<15} "
              f"{'Comm (MB)':<15} {'Conv. Round':<12}")
        print("-" * 100)
        
        # Rows
        for name, exp in self.experiments.items():
            print(f"{name:<30} {exp['final_accuracy']:>10.4f}  "
                  f"{exp['peak_memory_mb']:>13.1f}  "
                  f"{exp['total_comm_mb']:>13.2f}  "
                  f"{exp['convergence_round']:>10}")
        
        print("=" * 100 + "\n")
        
        # Compute improvements
        if 'atlas' in self.experiments and 'standard_fl' in self.experiments:
            atlas = self.experiments['atlas']
            baseline = self.experiments['standard_fl']
            
            acc_gain = (atlas['final_accuracy'] - baseline['final_accuracy']) * 100
            mem_reduction = (1 - atlas['peak_memory_mb'] / baseline['peak_memory_mb']) * 100
            comm_reduction = (1 - atlas['total_comm_mb'] / baseline['total_comm_mb']) * 100
            
            print("🎯 ATLAS vs Standard FL:")
            print(f"   Accuracy Gain:        {acc_gain:+.2f}%")
            print(f"   Memory Reduction:     {mem_reduction:.1f}%")
            print(f"   Communication Saving: {comm_reduction:.1f}%")
            print()


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("METRICS COLLECTION DEMO")
    print("=" * 70)
    
    # Create logger
    logger = MetricsLogger("demo_experiment")
    logger.set_metadata("model", "gpt2")
    logger.set_metadata("dataset", "sst2")
    
    # Simulate training rounds
    for round_num in range(1, 6):
        logger.log_round_start(round_num)
        
        # Simulate training
        time.sleep(0.1)
        
        # Create metrics
        training = TrainingMetrics(
            loss=1.0 / round_num,
            accuracy=0.6 + 0.05 * round_num
        )
        memory = MemoryMetrics.capture()
        comm = CommunicationMetrics(
            upload_mb=5.0,
            download_mb=5.0,
            total_mb=10.0,
            num_parameters=1_000_000
        )
        
        logger.log_round_end(round_num, training, memory, comm, num_clients=10)
    
    # Print and save
    logger.print_summary()
    logger.save()
    
    print("✅ Demo complete!")
