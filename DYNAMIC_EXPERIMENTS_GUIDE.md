# ATLAS Dynamic Experiment Runner

## Overview

The ATLAS framework supports **dynamic model configuration** for running experiments across multiple models without code modification. All experiments run **10 rounds in one shot** with automatic hyperparameter optimization.

## Supported Models

| Model ID | Full Name | Parameters | Notes |
|----------|-----------|------------|-------|
| `distilbert` | DistilBERT-base-uncased | 66M | Fastest, lightweight |
| `gpt2` | GPT-2 (124M) | 124M | Fast experiments |
| `gpt2-xl` | GPT-2 XL (1.5B) | 1.558B | Publication quality |
| `qwen2.5` | Qwen/Qwen2.5-0.5B | 494M | Mid-size, efficient |

## Quick Start

### Generic Script (Any Model, Any Seed)
```bash
./run_experiment.sh <model> <seed> <method>
```

### GPU-Specific Scripts (Fixed Seeds)

| Script | Seed |
|--------|------|
| `gpu0_seed42.sh` | 42 |
| `gpu1_seed123.sh` | 123 |
| `gpu2_seed456.sh` | 456 |

**Usage:**
```bash
./gpu0_seed42.sh <model> <method>
./gpu1_seed123.sh <model> <method>
./gpu2_seed456.sh <model> <method>
```

## Parameters

### `<model>` (Required)
- `distilbert` - DistilBERT 66M (fastest)
- `gpt2` - GPT-2 124M
- `gpt2-xl` - GPT-2 XL 1.5B (default)
- `qwen2.5` - Qwen2.5 494M

### `<seed>` (Required for generic script)
- Any integer (e.g., 42, 123, 456)

### `<method>` (Required)
- `atlas` - Full ATLAS pipeline (default)
- `fedavg_cluster` - FedAvg within task clusters
- `local_only` - Local training only

## Examples

```bash
# DistilBERT with ATLAS
./gpu0_seed42.sh distilbert atlas

# GPT-2 with FedAvg
./gpu1_seed123.sh gpt2 fedavg_cluster

# GPT-2 XL with ATLAS
./gpu0_seed42.sh gpt2-xl atlas

# Qwen2.5 with Local-Only
./gpu2_seed456.sh qwen2.5 local_only

# Custom seed
./run_experiment.sh gpt2 999 atlas
```

## Model Configurations

Each model has optimized hyperparameters in `experiments/config.py`:

**DistilBERT (66M):**
- Batch: 16, LR: 3e-5, Epochs: 2, Samples: 100, Ranks: [4,8,16,32]

**GPT-2 (124M):**
- Batch: 16, LR: 3e-5, Epochs: 2, Samples: 100, Ranks: [4,8,16,32]

**GPT-2 XL (1.5B):**
- Batch: 8, LR: 2e-5, Epochs: 3, Samples: 25, Ranks: [4,8,16,32,64]

**Qwen2.5 (494M):**
- Batch: 12, LR: 3e-5, Epochs: 2, Samples: 80, Ranks: [4,8,16,32,48]

**Fixed for all models:**
- Rounds: 10
- Tasks: sst2, mrpc, cola, qnli
- Clients per task: 3
- Total clients: 12

## Results

Results saved as: `results/atlas_<model>_<method>_seed<seed>_r10.json`

**Examples:**
- `atlas_distilbert_atlas_seed42_r10.json`
- `atlas_gpt2_fedavg_cluster_seed123_r10.json`
- `atlas_gpt2-xl_atlas_seed42_r10.json`
- `atlas_qwen2.5_local_only_seed456_r10.json`

**Contents include:**
- Round-by-round metrics (accuracy, loss)
- Final accuracies per client
- Cluster assignments
- Rank allocations
- Communication costs
- Time metrics
- Complete config

## Running All Experiments

### Parallel (3 GPUs)
```bash
# Terminal 1
./gpu0_seed42.sh gpt2-xl atlas

# Terminal 2
./gpu1_seed123.sh gpt2 atlas

# Terminal 3
./gpu2_seed456.sh qwen2.5 atlas
```

### Sequential (All Models)
```bash
# All models with ATLAS across three seeds
./gpu0_seed42.sh distilbert atlas
./gpu0_seed42.sh gpt2 atlas
./gpu0_seed42.sh gpt2-xl atlas
./gpu0_seed42.sh qwen2.5 atlas

./gpu1_seed123.sh distilbert atlas
./gpu1_seed123.sh gpt2 atlas
./gpu1_seed123.sh gpt2-xl atlas
./gpu1_seed123.sh qwen2.5 atlas

./gpu2_seed456.sh distilbert atlas
./gpu2_seed456.sh gpt2 atlas
./gpu2_seed456.sh gpt2-xl atlas
./gpu2_seed456.sh qwen2.5 atlas
```

## Advanced Usage

### Override Hyperparameters
```bash
python experiments/atlas_integrated.py \
    --mode full \
    --model gpt2 \
    --ablation atlas \
    --seed 42 \
    --rounds 10 \
    --batch-size 32 \
    --lr 1e-5
```

### Lambda Sweep
```bash
python experiments/atlas_integrated.py \
    --mode full \
    --model gpt2-xl \
    --lambda-sweep \
    --seed 42
```

## Performance

### Runtime (T4 GPU, 10 rounds)

| Model | Time |
|-------|------|
| DistilBERT | 30-45 min |
| GPT-2 | 45-60 min |
| GPT-2 XL | 2-3 hours |
| Qwen2.5 | 1-1.5 hours |

### Memory Requirements

| Model | GPU Memory |
|-------|------------|
| DistilBERT | 4-6 GB |
| GPT-2 | 6-8 GB |
| GPT-2 XL | 10-14 GB |
| Qwen2.5 | 8-10 GB |

## Troubleshooting

**Python not found:**
```bash
conda activate atlas_env
```

**CUDA OOM:**
- Use smaller model (distilbert or gpt2)
- Reduce: `--batch-size 4 --fingerprint-samples 10`

**Results not found:**
- Check `results/` directory
- Verify experiment completed successfully
- Check disk space

## Directory Structure

```
ATLAS/
├── run_experiment.sh           # Generic runner
├── gpu0_seed42.sh              # Seed 42
├── gpu1_seed123.sh             # Seed 123
├── gpu2_seed456.sh             # Seed 456
├── experiments/
│   ├── config.py               # Model configs
│   ├── atlas_integrated.py     # Main runner
│   └── ...
├── results/                    # JSON outputs
├── checkpoints/                # Training checkpoints
└── logs/                       # Execution logs
```

---
**Version:** 2.0 (Dynamic Model Support)  
**Date:** February 2026
