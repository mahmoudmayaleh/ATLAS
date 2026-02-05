# ATLAS Experiments

This directory contains the complete experimental framework for validating and benchmarking the ATLAS federated learning system.

## Structure

```
experiments/
├── __init__.py              # Package initialization
├── config.py                # Experiment configurations (datasets, models, baselines)
├── run_experiments.py       # Main experiment runner
├── metrics.py               # Metrics collection and logging
├── visualize.py             # Visualization utilities
└── README.md                # This file
```

## Quick Start

### Run Quick Experiments (Fast validation)

```powershell
cd c:\Users\Hp\Downloads\Advanced_project\ATLAS
python experiments\run_experiments.py --mode quick
```

This runs 2 quick experiments (~5 minutes):

- `quick_sst2_atlas`: ATLAS on SST-2 (20 rounds)
- `quick_sst2_baseline`: Homogeneous LoRA baseline

### Run Full Experiment Suite

```powershell
python experiments\run_experiments.py --mode full --save-dir ./results
```

This runs comprehensive experiments:

- Standard FL baseline
- Homogeneous LoRA
- HSplitLoRA (heterogeneous without task clustering)
- ATLAS (full system)

### Run Single Experiment

```powershell
python experiments\run_experiments.py --mode single --name sst2_atlas
```

## Experiment Configurations

### Datasets

- **GLUE Tasks**: SST-2, MRPC, CoLA, QNLI
- **Multi-task**: Combined GLUE tasks for task clustering validation

### Models

- **GPT-2**: 124M parameters, 500MB
- **GPT-2 Medium**: 355M parameters, 1.4GB
- **BERT-base**: 110M parameters, 440MB

### Device Profiles

- **Smartphone**: 2GB RAM, low compute
- **Tablet**: 4GB RAM, medium compute
- **Laptop (CPU)**: 8GB RAM, standard compute
- **Laptop (GPU)**: 16GB RAM, high compute
- **Workstation**: 32GB RAM, very high compute

### Baselines

1. **Standard FL**: Full model training with FedAvg
2. **Homogeneous LoRA**: Same rank for all clients
3. **HSplitLoRA**: Heterogeneous ranks without task clustering
4. **ATLAS**: Full system (task clustering + heterogeneous + split learning + Laplacian)

## Metrics Collected

### Performance

- Accuracy (classification tasks)
- F1 score, precision, recall
- Loss
- Perplexity

### Resources

- Peak memory usage (MB)
- Average memory usage
- Communication cost per round (MB)
- Total communication (MB)

### Training

- Convergence round
- Time per round (seconds)
- Total training time

### System

- Task clustering quality (Silhouette score)
- Number of task groups
- Per-client metrics

## Visualization

Generate visualizations from experiment results:

```powershell
python experiments\visualize.py
```

Or in Python:

```python
from experiments import ExperimentVisualizer

viz = ExperimentVisualizer(results_dir="./results", save_dir="./figures")
viz.load_all_experiments()
viz.create_all_plots()
```

### Generated Plots

- `convergence_accuracy.png`: Training accuracy curves
- `convergence_loss.png`: Training loss curves
- `comparison_accuracy.png`: Accuracy comparison bar chart
- `comparison_memory.png`: Memory usage comparison
- `comparison_communication.png`: Communication cost comparison
- `memory_accuracy_tradeoff.png`: Memory vs accuracy scatter plot
- `communication_comparison.png`: Communication breakdown
- `dashboard.png`: Comprehensive dashboard with all metrics

## Python API

### Run Experiments Programmatically

```python
from experiments import ExperimentRunner, ExperimentConfig, TrainingConfig

# Create custom experiment
config = ExperimentConfig(
    name="my_experiment",
    dataset="sst2",
    model="gpt2",
    baseline="atlas",
    device_mix=["smartphone"] * 5 + ["laptop_cpu"] * 5,
    training=TrainingConfig(
        num_rounds=30,
        batch_size=8,
        learning_rate=1e-4
    )
)

# Run
runner = ExperimentRunner(config, save_dir="./my_results")
results = runner.run()

# Access results
print(f"Final Accuracy: {results.final_accuracy:.4f}")
print(f"Memory Usage: {results.peak_memory_mb:.1f} MB")
```

### Metrics Logging

```python
from experiments import MetricsLogger, TrainingMetrics, MemoryMetrics, CommunicationMetrics

logger = MetricsLogger("my_experiment")

for round_num in range(10):
    logger.log_round_start(round_num)

    # Train...

    # Log metrics
    training = TrainingMetrics(loss=0.5, accuracy=0.85)
    memory = MemoryMetrics.capture()
    comm = CommunicationMetrics.compute(model_params)

    logger.log_round_end(round_num, training, memory, comm, num_clients=10)

# Save and print
logger.print_summary()
logger.save()
```

### Compare Results

```python
from experiments import ComparisonAnalyzer

analyzer = ComparisonAnalyzer(results_dir="./results")
analyzer.load_experiments(["sst2_atlas", "sst2_baseline", "sst2_standard_fl"])
analyzer.print_comparison_table()

# Get specific comparisons
accuracies = analyzer.compare_accuracy()
memory = analyzer.compare_memory()
communication = analyzer.compare_communication()
```

## 📝 Results Format

Results are saved in JSON format:

### Summary (`{experiment}_summary.json`)

```json
{
  "experiment_name": "sst2_atlas",
  "total_rounds": 50,
  "final_accuracy": 0.8543,
  "best_accuracy": 0.8621,
  "peak_memory_mb": 256.3,
  "total_comm_mb": 125.4,
  "convergence_round": 38,
  ...
}
```

### Round-by-Round (`{experiment}_rounds.json`)

```json
[
  {
    "round": 1,
    "training": {"loss": 1.2, "accuracy": 0.65},
    "memory": {"peak_mb": 245.1},
    "communication": {"total_mb": 2.5},
    "time_sec": 8.3
  },
  ...
]
```

## 🎯 Key Experiments

### 1. Quick Validation

**Purpose**: Fast validation that system works  
**Runtime**: ~5 minutes

```powershell
python experiments\run_experiments.py --mode quick
```

### 2. Full Baseline Comparison

**Purpose**: Compare ATLAS vs all baselines  
**Runtime**: ~2 hours

```powershell
python experiments\run_experiments.py --mode full
```

### 3. Multi-Task Learning

**Purpose**: Validate task clustering on multiple GLUE tasks  
**Config**: `FULL_EXPERIMENTS["multitask_atlas"]`

### 4. Device Heterogeneity

**Purpose**: Test adaptation to different device capabilities  
**Focus**: Memory usage across smartphone, tablet, laptop

### 5. Ablation Study

**Purpose**: Test contribution of each component  
**Variants**:

- No task clustering (HSplitLoRA)
- No heterogeneous ranks (Homogeneous LoRA)
- No split learning (Standard FL)

## 🔧 Configuration

Edit [config.py](config.py) to customize:

- Add new datasets
- Define new device profiles
- Create custom experiments
- Adjust hyperparameters

Example:

```python
# Add custom device
DEVICE_PROFILES["custom_device"] = {
    "name": "custom_device",
    "total_memory_mb": 6144,
    "compute_capability": 0.8,
    "bandwidth_mbps": 30,
    "description": "Custom device profile"
}

# Create custom experiment
custom_exp = ExperimentConfig(
    name="custom_experiment",
    dataset="sst2",
    model="gpt2",
    baseline="atlas",
    device_mix=["custom_device"] * 10,
    training=TrainingConfig(num_rounds=40)
)
```

## 📊 Expected Results

Based on similar systems (MIRA, HSplitLoRA, SplitLoRA papers):

| Metric        | Standard FL  | Homogeneous LoRA | ATLAS      | Improvement   |
| ------------- | ------------ | ---------------- | ---------- | ------------- |
| Accuracy      | 82%          | 83%              | 85%        | +3%           |
| Memory        | 500 MB       | 200 MB           | 150 MB     | 70% reduction |
| Communication | 250 MB/round | 5 MB/round       | 2 MB/round | 98% reduction |
| Convergence   | 60 rounds    | 45 rounds        | 35 rounds  | 42% faster    |

## 🐛 Troubleshooting

### Out of Memory

- Reduce `batch_size` in `TrainingConfig`
- Use smaller model (gpt2 instead of gpt2-medium)
- Increase split layer (more on server side)

### Slow Execution

- Use `--mode quick` for faster validation
- Reduce `num_rounds`
- Use CPU if GPU is slow

### Missing Dependencies

```powershell
pip install -r requirements.txt
```

## 📚 References

- MIRA: Multi-task federated learning with task clustering
- HSplitLoRA: Heterogeneous split learning with LoRA
- SplitLoRA: Communication-efficient federated learning

---

**Ready to run experiments! 🚀**

For questions or issues, check the main [ATLAS README](../README.md) or implementation roadmap.
