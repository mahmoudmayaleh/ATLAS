# Statistical Experiments & Model Expansion Guide

## 🎯 Overview

This guide covers:

1. **Multi-seed statistical experiments** (3-5 seeds with t-tests, Wilcoxon)
2. **Expanded model support** (BERT-base/large, RoBERTa, GPT-2)
3. **Expanded GLUE tasks** (QNLI, QQP, MNLI)
4. **Improved split layer selection** (bandwidth-aware, task-specific)

---

## 📊 1. Multi-Seed Statistical Experiments

### Quick Start

```bash
# Run 5 seeds for all configurations (ATLAS, FedAvg, Local-only)
python experiments/run_statistical_experiments.py \
    --seeds 5 \
    --configs atlas fedavg_cluster local_only \
    --rounds 15 \
    --samples 3000
```

### Outputs

1. **Individual results**: `results/statistical/{config}_seed{N}.json`
2. **Statistical summary**: `results/statistical/statistical_summary.csv`
3. **Statistical tests**: `results/statistical/statistical_tests.csv`

### Statistical Summary Includes:

- **Mean ± Std Dev** for: Final accuracy, Personalization, Communication
- **Paired t-tests**: Parametric significance tests
- **Wilcoxon tests**: Non-parametric significance tests
- **Cohen's d**: Effect size (small/medium/large)
- **LaTeX table**: Ready for IEEE paper

### Example Output:

```
[ATLAS] (5 seeds)
  Final Accuracy:    0.8123 ± 0.0045
  Accuracy Range:    [0.8061, 0.8189]
  Personalization:   0.0312 ± 0.0021
  Communication:     245.3 ± 12.7 MB

[ATLAS vs FEDAVG_CLUSTER]
  Accuracy:
    Δμ = +0.0067
    t-test:    p=0.0234 ✓ SIGNIFICANT
    Wilcoxon:  p=0.0312 ✓ SIGNIFICANT
    Cohen's d: 0.634 (medium)
```

---

## 🤖 2. Expanded Model Support

### Currently Supported Models

| Model               | Parameters | Layers | Use Case          |
| ------------------- | ---------- | ------ | ----------------- |
| **DistilBERT-base** | 66M        | 6      | Fast, lightweight |
| **BERT-base**       | 110M       | 12     | Standard baseline |
| **BERT-large**      | 340M       | 24     | High accuracy     |
| **RoBERTa-base**    | 125M       | 12     | Better NLU        |
| **RoBERTa-large**   | 355M       | 24     | State-of-the-art  |
| **GPT-2**           | 124M       | 12     | Generative tasks  |
| **GPT-2-medium**    | 355M       | 24     | Larger generative |

### Usage Examples

```bash
# BERT-base
python experiments/atlas_integrated.py \
    --mode full \
    --model bert-base-uncased \
    --tasks sst2 mrpc cola \
    --rounds 15 \
    --seed 42

# RoBERTa-base
python experiments/atlas_integrated.py \
    --mode full \
    --model roberta-base \
    --tasks sst2 qnli mnli \
    --rounds 15 \
    --seed 42

# GPT-2
python experiments/atlas_integrated.py \
    --mode full \
    --model gpt2 \
    --tasks sst2 cola \
    --rounds 15 \
    --seed 42
```

### Multi-Seed Experiments with Different Models

```bash
# Run BERT-base with 5 seeds
python experiments/run_statistical_experiments.py \
    --seeds 5 \
    --model bert-base-uncased \
    --tasks sst2 mrpc cola qnli \
    --rounds 15

# Run RoBERTa-base with 5 seeds
python experiments/run_statistical_experiments.py \
    --seeds 5 \
    --model roberta-base \
    --tasks sst2 mrpc cola qnli mnli \
    --rounds 15
```

---

## 📝 3. Expanded GLUE Task Support

### All Supported Tasks

| Task      | Type        | Metric   | Description                            |
| --------- | ----------- | -------- | -------------------------------------- |
| **SST-2** | Sentiment   | Accuracy | Stanford Sentiment Treebank            |
| **MRPC**  | Paraphrase  | F1/Acc   | Microsoft Research Paraphrase          |
| **CoLA**  | Grammar     | Matthews | Corpus of Linguistic Acceptability     |
| **QNLI**  | QA/NLI      | Accuracy | Question Natural Language Inference    |
| **QQP**   | Duplicate   | F1/Acc   | Quora Question Pairs                   |
| **MNLI**  | NLI         | Accuracy | Multi-Genre Natural Language Inference |
| **RTE**   | Entailment  | Accuracy | Recognizing Textual Entailment         |
| **WNLI**  | Coreference | Accuracy | Winograd Schema Challenge              |

### Multi-Task Heterogeneous Experiments

```bash
# 6 diverse tasks across 18 clients (3 per task)
python experiments/atlas_integrated.py \
    --mode full \
    --model distilbert-base-uncased \
    --tasks sst2 mrpc cola qnli qqp mnli \
    --clients-per-task 3 \
    --rounds 30 \
    --samples 5000

# Statistical experiments with expanded tasks
python experiments/run_statistical_experiments.py \
    --seeds 5 \
    --tasks sst2 mrpc cola qnli mnli \
    --clients-per-task 3 \
    --rounds 15
```

---

## 🔧 4. Improved Split Layer Selection

### Current Implementation

**Location**: `src/improved_split_selection.py`

**Features**:

- ✅ Memory-aware (device constraints)
- ✅ Communication-aware (bandwidth, activation sizes)
- ✅ Layer importance-aware (gradient magnitudes)
- ✅ Workload balancing
- ✅ Task-specific optimization

### How It Works

The split point is computed using a **weighted scoring function**:

```
Total Score = 0.35 × Memory Score
            + 0.30 × Communication Score
            + 0.25 × Layer Importance Score
            + 0.10 × Workload Balance Score
```

#### 1. **Memory Score** (35%)

- Estimates memory for: base model layers + LoRA adapters + activations
- Soft constraint with safety margin (80% of available memory)
- Exponential penalty for exceeding budget

#### 2. **Communication Score** (30%)

- Models activation transfer time (upload + download)
- Considers bandwidth and compression ratio
- Prefers higher splits (more layers on client = less communication)

#### 3. **Layer Importance Score** (25%)

- Uses gradient fingerprints to identify important layers
- Keeps task-specific features on client for personalization
- Fallback: prefers middle layers (most task-relevant)

#### 4. **Workload Balance Score** (10%)

- Balances computation between client and server
- Prefers splits near 50/50

### Integration with ATLAS

The improved split selector is already integrated:

- Called during Phase 3 initialization
- Per-client adaptive splitting based on device profile
- Can be task-specific if fingerprints provided

### Advanced Usage

```python
from src.improved_split_selection import ImprovedSplitSelector

# Define device profiles
device_profiles = {
    0: {'memory_mb': 2048, 'device_type': 'cpu_2gb'},
    1: {'memory_mb': 4096, 'device_type': 'tablet_4gb'},
    2: {'memory_mb': 8192, 'device_type': 'laptop_8gb'},
}

# Define tasks per client
task_assignments = {
    0: 'sst2',
    1: 'mrpc',
    2: 'sst2'
}

# Create selector
selector = ImprovedSplitSelector(
    model_name='bert-base-uncased',
    device_profiles=device_profiles,
    task_assignments=task_assignments,
    bandwidth_mbps=10.0,  # Network speed
    compression_ratio=0.5  # 50% activation compression
)

# Compute optimal splits
for client_id in device_profiles.keys():
    split = selector.compute_optimal_split(client_id)
    print(f"Client {client_id}: Split at layer {split}")
```

### Output Example

```
[SPLIT] Client 0: Optimal split = Layer 4/12
        Score: 0.782 (mem=0.95, comm=0.68, imp=0.82)
[SPLIT] Client 1: Optimal split = Layer 6/12
        Score: 0.845 (mem=1.00, comm=0.75, imp=0.88)
[SPLIT] Client 2: Optimal split = Layer 8/12
        Score: 0.891 (mem=1.00, comm=0.82, imp=0.91)
```

### Further Improvements (Future Work)

**Potential enhancements**:

1. **Dynamic splitting**: Adjust split point during training
2. **Gradient-based optimization**: Learn optimal split via meta-learning
3. **QoS constraints**: Consider latency, energy, fairness
4. **Layer clustering**: Group similar layers for coarser splits
5. **Knowledge distillation**: Compress activations with learned codecs

---

## 🧪 Complete Publication Experiment Workflow

### Step 1: Quick Validation (30 min)

```bash
# Validate setup with quick mode
python experiments/atlas_integrated.py --mode quick --seed 42
```

### Step 2: Multi-Seed Statistical Experiments (8-12 hours)

```bash
# Run 5 seeds × 3 configs = 15 experiments
python experiments/run_statistical_experiments.py \
    --seeds 5 \
    --configs atlas fedavg_cluster local_only \
    --model distilbert-base-uncased \
    --tasks sst2 mrpc cola qnli \
    --rounds 15 \
    --samples 3000
```

### Step 3: Model Comparison (per model: 8-12 hours)

```bash
# BERT-base
python experiments/run_statistical_experiments.py \
    --seeds 5 --model bert-base-uncased --rounds 15

# RoBERTa-base
python experiments/run_statistical_experiments.py \
    --seeds 5 --model roberta-base --rounds 15

# GPT-2
python experiments/run_statistical_experiments.py \
    --seeds 5 --model gpt2 --rounds 15
```

### Step 4: Lambda (η) Sweep (2-4 hours)

```bash
# Test regularization strengths
for eta in 0.0 0.01 0.1 0.5 1.0; do
    python experiments/atlas_integrated.py \
        --mode full \
        --ablation atlas \
        --eta $eta \
        --rounds 15 \
        --seed 42
done
```

### Step 5: Extended Training (30+ rounds, 4-6 hours per config)

```bash
# Session 1: Rounds 1-15
python experiments/atlas_integrated.py \
    --mode full --rounds 15 --max-rounds 15 --seed 42

# Session 2: Rounds 16-30 (resume)
python experiments/atlas_integrated.py \
    --mode full --rounds 30 \
    --resume checkpoints/atlas_round_15.pkl --seed 42
```

### Step 6: Analyze Results

```python
# Load and compare all results
import json
import pandas as pd
from pathlib import Path

results_dir = Path('results/statistical')

# Load statistical summary
summary = pd.read_csv(results_dir / 'statistical_summary.csv')
print(summary)

# Load statistical tests
tests = pd.read_csv(results_dir / 'statistical_tests.csv')
significant = tests[tests['significant'] == True]
print(f"\n{len(significant)} significant comparisons found")
```

---

## 📈 Expected Results (Publication Quality)

### ATLAS vs Baselines (Mean ± Std, 5 seeds)

| Method             | Final Acc     | Personalization | Comm (MB) | Sig?      |
| ------------------ | ------------- | --------------- | --------- | --------- |
| **ATLAS**          | 0.812 ± 0.005 | 0.031 ± 0.002   | 245 ± 13  | -         |
| **FedAvg/Cluster** | 0.805 ± 0.007 | 0.048 ± 0.003   | 242 ± 15  | p<0.05 ✓  |
| **Local Only**     | 0.789 ± 0.009 | 0.092 ± 0.005   | 0 ± 0     | p<0.01 ✓✓ |

### Model Comparison

| Model            | Params | ATLAS Acc | FedAvg Acc | Improvement |
| ---------------- | ------ | --------- | ---------- | ----------- |
| **DistilBERT**   | 66M    | 0.812     | 0.805      | +0.7%       |
| **BERT-base**    | 110M   | 0.851     | 0.842      | +0.9%       |
| **RoBERTa-base** | 125M   | 0.867     | 0.858      | +0.9%       |
| **GPT-2**        | 124M   | 0.824     | 0.817      | +0.7%       |

---

## ✅ Checklist for IEEE Publication

- [ ] Multi-seed experiments (5 seeds minimum)
- [ ] Statistical significance tests (t-test + Wilcoxon)
- [ ] Reported with mean ± std dev
- [ ] Tested on 3+ models (DistilBERT, BERT, RoBERTa)
- [ ] Tested on 4+ GLUE tasks
- [ ] Ablation studies (all 4 phases)
- [ ] Lambda sweep (5 values)
- [ ] 30+ round convergence experiments
- [ ] Communication cost analysis
- [ ] Per-task and per-client metrics
- [ ] High-resolution plots (300 DPI)
- [ ] Results saved as JSON (reproducible)

---

## 📚 Citation

```bibtex
@article{atlas2026,
  title={ATLAS: Adaptive Task and Layer-Aware Split Learning for Federated Fine-Tuning of Large Language Models},
  author={Your Name et al.},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2026}
}
```

---

## 🔗 References

1. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
2. **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
3. **HSplitLoRA**: Heterogeneous Split Learning with LoRA, 2024
4. **MIRA**: Personalized FL via Laplacian Regularization, ICML 2023
5. **VFLAIR-LLM**: Vertical Federated Learning for LLMs, 2024
