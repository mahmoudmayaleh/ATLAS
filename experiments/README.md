# Experiments Package

This directory contains the maintained experiment entrypoints and result-generation utilities for ATLAS.

## Maintained Entry Points

- `experiments/atlas_integrated.py`: primary runner for classification experiments
- `experiments/generate_results_tables.py`: table generation from local result JSON files
- `experiments/generate_publication_plots.py`: plot generation from local result JSON files
- `experiments/generate_all_publication_materials.py`: convenience wrapper for tables and plots

## Configuration Sources

- `experiments/config.py` defines task metadata, model aliases, device profiles, and baseline metadata.
- `experiments/metrics.py` contains metric collection helpers used during and after runs.

## Classification Workflow

Quick validation:

```bash
python experiments/atlas_integrated.py --mode quick --ablation atlas --rounds 2 --samples 80 --local-epochs 1 --tasks sst2 mrpc qnli --seed 42
```

Full run:

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

## Baselines

The integrated runner currently exposes:

- `atlas`
- `atlas_no_laplacian`
- `fedavg_cluster`
- `standard_fl`
- `local_only`

## Outputs

Generated outputs are written locally under:

- `results/`
- `checkpoints/`
- `logs/`

The canonical single-run filename pattern is:

```text
results/atlas_{MODEL}_{ABLATION}_seed{SEED}_r{ROUNDS}.json
```

Example:

```text
results/atlas_distilbert-base-uncased_atlas_seed42_r10.json
```

## Statistical and Publication Utilities

If you have a local `experiments/run_statistical_experiments.py` in your private workflow, keep it outside the public source tree. The open-source repository keeps the core runner and post-processing scripts only.

Publication materials are generated from local result files, not stored as source artifacts.
