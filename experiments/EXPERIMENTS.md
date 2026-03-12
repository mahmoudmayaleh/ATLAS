# Experiments Package

This document describes the maintained experiment interfaces in the public repository.

## Maintained Entry Points

- `experiments/atlas_integrated.py`: primary runner for classification experiments
- `experiments/generate_results_tables.py`: table generation from local result JSON files
- `experiments/generate_publication_plots.py`: plot generation from local result JSON files
- `experiments/generate_all_publication_materials.py`: convenience wrapper for tables and plots

## Configuration Sources

- `experiments/config.py` defines task metadata, model aliases, device profiles, and baseline metadata.
- `experiments/metrics.py` contains metric collection helpers used during and after runs.

## Supported Tasks

The current classification workflow supports:

- `sst2`
- `mrpc`
- `qnli`

## Main Runner

Use `experiments/atlas_integrated.py` for classification experiments.

Common flags:

- `--mode quick|full`
- `--ablation atlas|atlas_no_laplacian|fedavg_cluster|standard_fl|local_only`
- `--model ...`
- `--tasks ...`
- `--clients-per-task N`
- `--rounds N`
- `--samples N`
- `--local-epochs N`
- `--seed N`
- `--resume path.pkl`
- `--max-rounds N`

### Quick Validation

```bash
python experiments/atlas_integrated.py --mode quick --ablation atlas --tasks sst2 mrpc qnli --rounds 2 --samples 80 --local-epochs 1 --seed 42
```

### Full Run

```bash
python experiments/atlas_integrated.py \
  --mode full \
  --ablation atlas \
  --model distilbert-base-uncased \
  --tasks sst2 mrpc qnli \
  --clients-per-task 3 \
  --rounds 10 \
  --samples 3000 \
  --local-epochs 3 \
  --seed 42
```

## Baseline Semantics

- `local_only`: no aggregation and no communication
- `fedavg_cluster`: aggregation within Phase 1 clusters only
- `standard_fl`: task-agnostic FedAvg baseline
- `atlas_no_laplacian`: ATLAS pipeline without Laplacian personalization
- `atlas`: full ATLAS pipeline

## Resume Workflow

```bash
python experiments/atlas_integrated.py --mode full --ablation atlas --rounds 30 --max-rounds 15 --seed 42 --tasks sst2 mrpc qnli
python experiments/atlas_integrated.py --mode full --ablation atlas --rounds 30 --resume checkpoints/atlas_atlas_seed42_round_15.pkl --seed 42 --tasks sst2 mrpc qnli
```

## Output Naming

Single-run outputs follow the canonical pattern:

```text
results/atlas_{MODEL}_{ABLATION}_seed{SEED}_r{ROUNDS}.json
```

Example:

```text
results/atlas_gpt2_atlas_seed42_r10.json
```

Checkpoints are written under `checkpoints/` and logs under `logs/`.

## Post-Processing

Generate result tables and figures locally with:

```bash
python experiments/generate_results_tables.py --results-dir results --output-dir results
python experiments/generate_publication_plots.py --results-dir results --output-dir results
python experiments/generate_all_publication_materials.py --results-dir results --figures-dir results --tables-dir results
```

These products are generated artifacts and are not intended to be committed to the public repository.

## Statistical and Publication Utilities

If you have a local `experiments/run_statistical_experiments.py` in your private workflow, keep it outside the public source tree. The open-source repository keeps the core runner and post-processing scripts only.

Publication materials are generated from local result files, not stored as source artifacts.
