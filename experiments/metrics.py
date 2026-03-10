"""
Metrics Collection and Analysis for ATLAS Experiments

Provides tools for tracking, logging, and analyzing performance metrics
during federated learning experiments.

Includes:
- Classification metrics (accuracy, F1, Matthews)
- NLG metrics (BLEU, NIST, METEOR, ROUGE-L) for SplitLoRA/HSplitLoRA comparison
- Perplexity tracking for causal LM tasks
- Convergence & efficiency helpers (trainable params, memory, wall-clock)
"""

import torch
import torch.nn.functional as F
import psutil
import time
import json
from typing import Dict, List, Any, Optional, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


# ========== PERPLEXITY ==========

def compute_perplexity_from_loss(avg_loss: float) -> float:
    """Convert average cross-entropy loss to perplexity: PPL = exp(loss).
    
    Works for both classification and causal-LM tasks.
    Clamps loss to [0, 100] to avoid overflow.
    """
    return float(np.exp(min(avg_loss, 100.0)))


def compute_perplexity(model, dataloader, device: str = "cuda") -> float:
    """Compute perplexity of a causal LM on a dataloader.
    
    Args:
        model: HuggingFace causal LM (or any model with .logits output)
        dataloader: DataLoader yielding batches with 'input_ids' and 'labels'
        device: torch device string
    
    Returns:
        Perplexity (float). Lower = better.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch.get('labels', input_ids).to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )
            # Count non-padding tokens
            valid_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return float(np.exp(min(avg_loss, 100.0)))


# ========== NLG METRICS (SplitLoRA / HSplitLoRA comparison) ==========

def _ensure_nltk_data():
    """Download required NLTK corpora once (no-op if already present)."""
    import nltk
    for pkg in ('wordnet', 'punkt', 'punkt_tab', 'omw-1.4'):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def compute_nlg_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute NLG evaluation metrics matching SplitLoRA/HSplitLoRA Table I/II.

    Metrics: BLEU, NIST, METEOR, ROUGE-L.
    Library mapping (sacrebleu 2.x dropped corpus_nist / corpus_meteor):
      BLEU   → sacrebleu.corpus_bleu
      NIST   → nltk.translate.nist_score.corpus_nist
      METEOR → nltk.translate.meteor_score.meteor_score  (averaged, ×100)
      ROUGE-L→ rouge_score.rouge_scorer

    Args:
        predictions: List of generated text strings
        references:  List of reference text strings (one per prediction)

    Returns:
        Dict with keys: BLEU, NIST, METEOR, ROUGE-L  (all 0–100 scale)
    """
    _ensure_nltk_data()
    results: Dict[str, float] = {}

    # ── BLEU via sacrebleu ────────────────────────────────────────────────────
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        results['BLEU'] = float(bleu.score)
    except Exception:
        results['BLEU'] = 0.0

    # ── NIST via nltk ─────────────────────────────────────────────────────────
    # nltk.corpus_nist expects: list-of-list-of-refs, list-of-hyps (tokenised)
    try:
        from nltk.translate.nist_score import corpus_nist as nltk_nist
        hyps = [p.split() for p in predictions]
        refs_tok = [[r.split()] for r in references]   # one ref per hypothesis
        results['NIST'] = float(nltk_nist(refs_tok, hyps, n=5))
    except Exception:
        results['NIST'] = 0.0

    # ── METEOR via nltk ──────────────────────────────────────────────────────
    # meteor_score takes tokenised lists; average over corpus, then ×100
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        scores = [
            nltk_meteor([ref.split()], pred.split())
            for pred, ref in zip(predictions, references)
        ]
        results['METEOR'] = float(np.mean(scores)) * 100.0
    except Exception:
        results['METEOR'] = 0.0

    # ── ROUGE-L via rouge_score ───────────────────────────────────────────────
    try:
        from rouge_score import rouge_scorer as rs_module
        scorer = rs_module.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [
            scorer.score(ref, pred)['rougeL'].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        results['ROUGE-L'] = float(np.mean(rouge_scores)) * 100.0
    except Exception:
        results['ROUGE-L'] = 0.0

    return results


# ========== CONVERGENCE TRACKING ==========

def find_convergence_round(
    values: List[float],
    target: Optional[float] = None,
    threshold_frac: float = 0.95,
    mode: str = 'max',
) -> Optional[int]:
    """Find the round where a metric converges.
    
    Args:
        values: Per-round metric values (accuracy, PPL, loss, etc.)
        target: Absolute target value. If None, uses threshold_frac of best.
        threshold_frac: Fraction of best value to consider converged (for mode='max').
            For mode='min', convergence = value <= target or best / threshold_frac.
        mode: 'max' (accuracy-like) or 'min' (loss/PPL-like)
    
    Returns:
        1-based round number, or None if never converged.
    """
    if not values:
        return None

    if target is None:
        if mode == 'max':
            best = max(values)
            target = best * threshold_frac
        else:
            best = min(values)
            target = best / threshold_frac  # e.g. converged when PPL <= best*1.05

    for i, v in enumerate(values):
        if mode == 'max' and v >= target:
            return i + 1
        elif mode == 'min' and v <= target:
            return i + 1

    return None


# ========== EFFICIENCY HELPERS ==========

def count_trainable_params(model) -> Dict[str, int]:
    """Count trainable vs total parameters, split by LoRA / non-LoRA.
    
    Returns:
        Dict with keys: total, trainable, frozen, lora_trainable, non_lora_trainable
    """
    total = 0
    trainable = 0
    lora_trainable = 0

    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            if any(kw in name.lower() for kw in ['lora_a', 'lora_b', 'lora']):
                lora_trainable += param.numel()

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'lora_trainable': lora_trainable,
        'non_lora_trainable': trainable - lora_trainable,
    }


def capture_memory_stats(device: str = "cuda") -> Dict[str, float]:
    """Capture GPU/CPU memory statistics in MB.
    
    Returns:
        Dict with allocated_mb, peak_mb, reserved_mb (GPU) or rss_mb (CPU)
    """
    if device == "cuda" and torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / (1024**2),
            'peak_mb': torch.cuda.max_memory_allocated() / (1024**2),
            'reserved_mb': torch.cuda.memory_reserved() / (1024**2),
            'device': 'cuda',
        }
    else:
        process = psutil.Process()
        mem = process.memory_info()
        return {
            'rss_mb': mem.rss / (1024**2),
            'peak_mb': mem.rss / (1024**2),
            'device': 'cpu',
        }


def compute_comm_cost_mb(state_dict: Dict[str, torch.Tensor],
                         quantized: bool = True) -> Dict[str, float]:
    """Compute communication cost of a state dict in MB.
    
    Args:
        state_dict: Model state dict (or subset — e.g. LoRA params only)
        quantized: If True, assume INT8 quantization (1 byte/param)
    
    Returns:
        Dict with total_mb, num_params, bytes_per_param
    """
    total_bytes = 0
    num_params = 0
    for name, param in state_dict.items():
        n = param.numel()
        num_params += n
        if quantized:
            total_bytes += n  # 1 byte per param (INT8)
        else:
            total_bytes += n * param.element_size()

    return {
        'total_mb': total_bytes / (1024**2),
        'num_params': num_params,
        'bytes_per_param': 1 if quantized else 4,
    }


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
