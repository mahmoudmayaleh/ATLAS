# ATLAS: Adaptive Task-aware Federated Learning (Multi-Task NLP)

ATLAS is a research codebase for multi-task federated learning with:

- task clustering (Phase 1)
- device-aware LoRA configuration (Phase 2)
- split federated learning (Phase 3)
- Laplacian regularization for personalization (Phase 4)

This README is aligned with the current, runnable experiment entrypoints.

## What is implemented in this repo

### Primary experiment scripts

- `experiments/atlas_integrated.py`: single-run experiments (supports `--seed`, `--resume`, `--max-rounds`)
- `experiments/run_statistical_experiments.py`: multi-seed runner with significance tests and CSV summaries

### Implemented tasks

The current dataset mapping in `experiments/atlas_integrated.py` includes:

- `sst2`
- `mrpc`
- `cola`
- `qnli`

Passing other task names will require extending the dataset mapping.

## Quick start

### Install

```bash
pip install -r requirements.txt
```

### Quick validation run

```bash
python experiments/atlas_integrated.py --mode quick --ablation atlas --seed 42
```

### Paper baseline single run

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

## Multi-seed statistical experiments (recommended for papers)

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

Outputs:

- `results/statistical/statistical_summary.csv`
- `results/statistical/statistical_tests.csv`

## Resume (multi-session)

For long runs under runtime limits, split into sessions.

```bash
# Session 1
python experiments/atlas_integrated.py --mode full --ablation atlas --rounds 30 --max-rounds 15 --seed 42 --tasks sst2 mrpc cola qnli

# Session 2
python experiments/atlas_integrated.py --mode full --ablation atlas --rounds 30 --resume checkpoints/atlas_atlas_seed42_round_15.pkl --seed 42 --tasks sst2 mrpc cola qnli
```

Note: checkpoint naming includes ablation + seed (the exact path is printed during the run).

## Outputs

Single runs write:

- Results JSON: `results/atlas_integrated_{mode}_{ablation}_seed{seed}.json`
- Final-only checkpoint: `checkpoints/atlas_{ablation}_seed{seed}_round_{round}.pkl`

## Documentation

Documentation is consolidated into three professional docs:

- `docs/METHODOLOGY.md` — detailed methodology (phases + split selection)
- `docs/EXPERIMENTS.md` — reproducible commands, outputs, and reporting protocol
- `docs/PROBLEMS_AND_SOLUTIONS.md` — practical issues + mitigations (paper-relevant)

The Colab notebook `atlas_publication.ipynb` includes optional Google Drive backup helpers.
