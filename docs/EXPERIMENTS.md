# ATLAS Experiments (Professional, Reproducible)

This document is the **single source of truth** for how to run experiments in the current repo.

## 1) Supported tasks (current implementation)

`experiments/atlas_integrated.py` currently wires the following tasks in its dataset mapping:

- `sst2`
- `mrpc`
- `cola`
- `qnli`

Passing other tasks will require extending the dataset mapping and preprocessing.

## 2) Primary experiment entrypoints

### A) Single-run experiments

Use `experiments/atlas_integrated.py` for a single configuration and seed.

Key flags:

- `--mode quick|full`
- `--ablation atlas|fedavg_cluster|local_only`
- `--seed N`
- `--rounds N`
- `--samples N`
- `--local-epochs N`
- `--tasks ...`

Baseline semantics (as implemented):

- `--ablation local_only`: local training only (no aggregation; communication is 0)
- `--ablation fedavg_cluster`: FedAvg **within each Phase-1 cluster only** (no cross-cluster averaging; no Laplacian)
- `--ablation atlas`: per-cluster FedAvg + Laplacian personalization

#### Quick validation

```bash
python experiments/atlas_integrated.py --mode quick --ablation atlas --seed 42
```

#### Paper baseline (single run)

```bash
python experiments/atlas_integrated.py \
  --mode full \
  --ablation atlas \
  --rounds 15 \
  --samples 3000 \
  --local-epochs 3 \
  --model distilbert-base-uncased \
  --tasks sst2 mrpc cola qnli \
  --clients-per-task 3 \
  --seed 42
```

Baselines:

```bash
python experiments/atlas_integrated.py --mode full --ablation fedavg_cluster --rounds 15 --samples 3000 --local-epochs 3 --tasks sst2 mrpc cola qnli --seed 42
python experiments/atlas_integrated.py --mode full --ablation local_only     --rounds 15 --samples 3000 --local-epochs 3 --tasks sst2 mrpc cola qnli --seed 42
```

### B) Multi-seed statistical experiments (recommended for papers)

Use `experiments/run_statistical_experiments.py` to run multiple seeds and automatically produce publication tables.

```bash
python experiments/run_statistical_experiments.py \
  --seeds 3 \
  --configs atlas fedavg_cluster local_only \
  --mode full \
  --rounds 15 \
  --samples 3000 \
  --local-epochs 3 \
  --model distilbert-base-uncased \
  --tasks sst2 mrpc cola qnli \
  --clients-per-task 3
```

#### Seed semantics

`--seeds` is a **count**, not a list. The runner uses predetermined seeds for reproducibility:

- 1 seed: `[42]`
- 3 seeds: `[42, 123, 456]`
- 5 seeds: `[42, 123, 456, 789, 1024]`

## 3) Outputs and file naming

### Single-run outputs (`atlas_integrated.py`)

- Results JSON:
  - `results/atlas_integrated_{mode}_{ablation}_seed{seed}.json`
- Checkpoint:
  - `checkpoints/atlas_{ablation}_seed{seed}_round_{round}.pkl`
  - The code is configured to save **final-only** checkpoints (to reduce disk usage) while still supporting resume.

### Statistical runner outputs

Default output directory: `results/statistical/`

- Per-run copies: `results/statistical/{config}_seed{seed}.json`
- Aggregate tables:
  - `results/statistical/statistical_summary.csv`
  - `results/statistical/statistical_tests.csv`

The runner also prints a LaTeX table to stdout.

## 4) Multi-session runs (resume)

Use this pattern when your environment has runtime limits.

```bash
# Session 1
python experiments/atlas_integrated.py --mode full --ablation atlas --rounds 30 --max-rounds 15 --seed 42 --tasks sst2 mrpc cola qnli

# Session 2 (use the printed checkpoint path)
python experiments/atlas_integrated.py --mode full --ablation atlas --rounds 30 --resume checkpoints/atlas_atlas_seed42_round_15.pkl --seed 42 --tasks sst2 mrpc cola qnli
```

## 5) Reporting guidance (paper-facing)

Recommended reporting for main tables:

- mean ± std of **final average accuracy** across seeds
- mean ± std of **total communication (MB)** across seeds
- significance testing (paired t-test + Wilcoxon when applicable)
- effect sizes (Cohen’s d)

## 6) Sanity-check protocol before overnight runs

1. Run one quick validation (`--mode quick --seed 42`).
2. Run one full baseline (`--mode full --rounds 5` if needed).
3. Only then run multi-seed experiments.
