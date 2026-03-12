# ATLAS Run Instructions

This document covers the maintained execution paths in the repository.

## Setup

```bash
cd /path/to/ATLAS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Conda instead of `venv`, activate the environment before running any scripts.

## Primary Entrypoint

Use `experiments/atlas_integrated.py` for classification experiments.

Help:

```bash
python experiments/atlas_integrated.py --help
```

### Supported tasks

- `sst2`
- `mrpc`
- `qnli`

### Supported ablations

- `atlas`
- `atlas_no_laplacian`
- `fedavg_cluster`
- `standard_fl`
- `local_only`

## Quick Sanity Run

```bash
python experiments/atlas_integrated.py \
  --mode quick \
  --ablation atlas \
  --tasks sst2 mrpc qnli \
  --rounds 2 \
  --samples 80 \
  --local-epochs 1 \
  --seed 42
```

The first run may take longer because the required models and datasets are downloaded on demand.

## Full Classification Run

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
  --eta 0.1 \
  --seed 42
```

Baseline examples:

```bash
python experiments/atlas_integrated.py --mode full --ablation fedavg_cluster --model distilbert-base-uncased --tasks sst2 mrpc qnli --clients-per-task 3 --rounds 10 --samples 3000 --local-epochs 3 --seed 42
python experiments/atlas_integrated.py --mode full --ablation local_only --model distilbert-base-uncased --tasks sst2 mrpc qnli --clients-per-task 3 --rounds 10 --samples 3000 --local-epochs 3 --seed 42
```

Laplacian sweep:

```bash
python experiments/atlas_integrated.py --mode full --ablation atlas --tasks sst2 mrpc qnli --clients-per-task 3 --rounds 10 --seed 42 --lambda-sweep
```

## Multi-Session Resume

Use `--max-rounds` to split a longer job into multiple sessions.

Session 1:

```bash
python experiments/atlas_integrated.py --mode full --ablation atlas --rounds 30 --max-rounds 15 --seed 42 --tasks sst2 mrpc qnli
```

Session 2:

```bash
python experiments/atlas_integrated.py --mode full --ablation atlas --rounds 30 --resume checkpoints/atlas_atlas_seed42_round_15.pkl --seed 42 --tasks sst2 mrpc qnli
```

## Seeded GPU Helper Scripts

The repository ships with thin wrappers around `experiments/atlas_integrated.py` for the common three-seed setup:

- `gpu0_seed42.sh`
- `gpu1_seed123.sh`
- `gpu2_seed456.sh`

Examples:

```bash
./gpu0_seed42.sh distilbert-base-uncased atlas
./gpu1_seed123.sh gpt2 standard_fl
./gpu2_seed456.sh Qwen/Qwen2.5-0.5B atlas
```

## Generated Outputs

Generated files are written under:

- `results/`
- `checkpoints/`
- `logs/`

These locations are treated as local build artifacts and are ignored by default.

## Publication Materials

Once result JSON files exist locally, generate tables and figures with:

```bash
python experiments/generate_results_tables.py --results-dir results --output-dir results
python experiments/generate_publication_plots.py --results-dir results --output-dir results
python experiments/generate_all_publication_materials.py --results-dir results --figures-dir results --tables-dir results
```
