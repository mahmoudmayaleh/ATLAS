# IEEE Publication Guide

This document is a reproducibility checklist for paper-facing experiments run from the public ATLAS codebase.

## Recommended Configuration

Use a fixed configuration across baselines before comparing results.

| Parameter | Recommended value |
| --- | --- |
| Rounds | 30 |
| Samples per client | 5000 |
| Local epochs | 3 |
| Tasks | `sst2 mrpc qnli` |
| Clients per task | 3 |
| Seeds | at least `42`, `123`, `456` |

## Core Experiments

Report at minimum:

1. `atlas`
2. `atlas_no_laplacian`
3. `fedavg_cluster`
4. `standard_fl`
5. `local_only`

Suggested command template:

```bash
python experiments/atlas_integrated.py \
  --mode full \
  --ablation atlas \
  --model distilbert-base-uncased \
  --tasks sst2 mrpc qnli \
  --clients-per-task 3 \
  --rounds 30 \
  --samples 5000 \
  --local-epochs 3 \
  --seed 42
```

## Resume Pattern for Long Runs

If your environment has runtime limits, split the run into two sessions.

Session 1:

```bash
python experiments/atlas_integrated.py --mode full --ablation atlas --rounds 30 --max-rounds 15 --tasks sst2 mrpc qnli --seed 42
```

Session 2:

```bash
python experiments/atlas_integrated.py --mode full --ablation atlas --rounds 30 --resume checkpoints/atlas_atlas_seed42_round_15.pkl --tasks sst2 mrpc qnli --seed 42
```

## What to Report

For each baseline, record:

- final average task metric
- per-task metric breakdown
- communication totals
- runtime per round and total runtime
- personalization spread or per-client dispersion

For multi-seed reporting, use mean and standard deviation across seeds.

## Local Artifact Handling

Keep the following outputs local to the experiment environment:

- `results/*.json`
- `results/*.pdf`
- `results/*.tex`
- `checkpoints/`
- `logs/`

The repository is structured so that source code remains public while experimental outputs and paper assets stay outside the public tree.

## Pre-Submission Checklist

1. Run the full baseline set with the same task list, model, and seed set.
2. Regenerate plots and tables from the local result files.
3. Verify that all claims in the manuscript are backed by reproducible commands.
4. Keep paper drafts, slide decks, and external reference PDFs outside the public repository.
