# ATLAS Run Instructions (Demo-Friendly, Current)

This file is a **clean, up-to-date** guide for running ATLAS from this repository.

If you only remember one thing:

- Single run (one config/seed): `experiments/atlas_integrated.py`
- Multi-seed paper runs: `experiments/run_statistical_experiments.py`
- Tables/figures: `experiments/generate_results_tables.py`, `experiments/generate_publication_plots.py`

---

## 0) Setup

### Windows (PowerShell)

```powershell
cd C:\Users\Hp\Downloads\Advanced_project\ATLAS

# If you already have a venv, just activate it:
.\.venv\Scripts\Activate.ps1

# Install deps
pip install -r requirements.txt
```

### Linux/macOS (bash)

```bash
cd /path/to/ATLAS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1) Quick demo run (fast sanity check)

This is the best command for a supervisor demo because it exercises the full pipeline but keeps the run small.

```powershell
python experiments\atlas_integrated.py \
	--mode quick \
	--ablation atlas \
	--tasks sst2 mrpc cola \
	--rounds 2 \
	--samples 80 \
	--local-epochs 1 \
	--seed 42
```

Notes:

- First run takes longer due to HuggingFace model/dataset downloads.
- If you have a GPU and want to force GPU selection: set `CUDA_VISIBLE_DEVICES`.

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python experiments\atlas_integrated.py --mode quick --ablation atlas --rounds 2 --samples 80 --local-epochs 1 --clients-per-task 1 --tasks sst2 mrpc cola --seed 42
```

---

## 2) Primary entrypoint: `atlas_integrated.py`

Help:

```powershell
python experiments\atlas_integrated.py --help
```

### Supported tasks (current dataset mapping)

- `sst2`
- `mrpc`
- `cola`
- `qnli`

### Ablation modes

- `atlas`: full ATLAS (clustering + hetero ranks + split FL + Laplacian)
- `atlas_no_laplacian`: disables Laplacian personalization
- `fedavg_cluster`: FedAvg within each cluster (task-aware baseline)
- `standard_fl`: pure FedAvg (may be heavier)
- `local_only`: no aggregation baseline

### Common parameter patterns

**ATLAS run (DistilBERT, 4 tasks):**

```powershell
python experiments\atlas_integrated.py \
	--mode full \
	--ablation atlas \
	--model distilbert-base-uncased \
	--tasks sst2 mrpc cola qnli \
	--rounds 10 \
	--samples 3000 \
	--local-epochs 3 \
	--eta 0.1 \
	--seed 42
```

**Baselines:**

```powershell
python experiments\atlas_integrated.py --mode full --ablation fedavg_cluster --model distilbert-base-uncased --tasks sst2 mrpc cola qnli --clients-per-task 3 --rounds 10 --samples 3000 --local-epochs 3 --seed 42
python experiments\atlas_integrated.py --mode full --ablation local_only     --model distilbert-base-uncased --tasks sst2 mrpc cola qnli --clients-per-task 3 --rounds 10 --samples 3000 --local-epochs 3 --seed 42
```

**Laplacian strength sweep:**

```powershell
python experiments\atlas_integrated.py --mode full --ablation atlas --tasks sst2 mrpc cola qnli --clients-per-task 3 --rounds 10 --seed 42 --lambda-sweep
```

### CLI flags (reference)

| Flag                      | Meaning                                                                                       |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| `--mode quick\|full`      | Run mode (quick = sanity, full = paper-style defaults)                                        |
| `--ablation ...`          | Which method/baseline to run                                                                  |
| `--model ...`             | Model name or alias (e.g., `distilbert-base-uncased`, `gpt2`, `gpt2-xl`, `Qwen/Qwen2.5-0.5B`) |
| `--tasks ...`             | Space-separated tasks (see supported list)                                                    |
| `--clients-per-task N`    | Number of clients per task                                                                    |
| `--rounds N`              | Total rounds to run                                                                           |
| `--max-rounds N`          | Limit rounds for the _current session_ (used with resume)                                     |
| `--resume path.pkl`       | Resume from a checkpoint file                                                                 |
| `--samples N`             | Max samples per client (reduce for faster demo)                                               |
| `--local-epochs N`        | Local epochs per round                                                                        |
| `--eta X`                 | Laplacian regularization strength                                                             |
| `--lambda-sweep`          | Runs `eta` sweep over `[0.0, 0.01, 0.1, 0.5, 1.0]`                                            |
| `--seed N`                | Random seed                                                                                   |
| `--lr X`                  | Override learning rate                                                                        |
| `--batch-size N`          | Override batch size                                                                           |
| `--fingerprint-samples N` | Reduce for very large models to avoid OOM                                                     |
| `--fingerprint-batches N` | Reduce for very large models to avoid OOM                                                     |

---

## 3) Outputs (where files go)

### Results JSON

Single-run results are saved under `results/` using the canonical naming convention:

```
results/atlas_{MODEL}_{ABLATION}_seed{SEED}_r{ROUNDS}.json
```

Example:

```
results/atlas_distilbert-base-uncased_atlas_seed42_r10.json
```

### Checkpoints (resume)

Checkpoints are written under `checkpoints/`:

```
checkpoints/atlas_{ABLATION}_seed{SEED}_round_{ROUND}.pkl
```

---

## 4) Multi-session runs (resume pattern)

Use this when you have time limits (Colab, shared GPU, etc.).

### Session 1 (first N rounds)

```powershell
python experiments\atlas_integrated.py --mode full --ablation atlas --rounds 30 --max-rounds 15 --seed 42 --tasks sst2 mrpc cola qnli
```

This produces a checkpoint like:

```
checkpoints/atlas_atlas_seed42_round_15.pkl
```

### Session 2 (resume and continue)

```powershell
python experiments\atlas_integrated.py --mode full --ablation atlas --rounds 30 --resume checkpoints\atlas_atlas_seed42_round_15.pkl --seed 42 --tasks sst2 mrpc cola qnli
```

---

## 5) Statistical (multi-seed) runner

This is the paper-style runner that repeats experiments over multiple predetermined seeds.

```powershell
python experiments\run_statistical_experiments.py \
	--seeds 3 \
	--configs atlas fedavg_cluster local_only \
	--mode full \
	--model distilbert-base-uncased \
	--tasks sst2 mrpc cola qnli \
	--clients-per-task 3 \
	--rounds 10 \
	--samples 3000 \
	--local-epochs 3 \
	--eta 0.1
```

Output directory (default):

- `results/statistical/` (summary CSVs + per-seed JSON copies)

---

## 6) Generate publication tables and plots

These scripts read the `results/*.json` files and generate:

- LaTeX + CSV tables under `results/`
- Figures under `figures/`

### Plots only

```powershell
python experiments\generate_publication_plots.py --results-dir results --output-dir figures
```

### Tables only

```powershell
python experiments\generate_results_tables.py --results-dir results --output-dir results
```

### Everything (plots + tables)

Run from repo root (recommended):

```powershell
python experiments\generate_all_publication_materials.py --results-dir results --figures-dir figures --tables-dir results
```

---

## 7) Notes on helper scripts

- Linux helper scripts exist (e.g., `gpu0_seed42.sh`, `gpu1_seed123.sh`, `gpu2_seed456.sh`). They are thin wrappers around `experiments/atlas_integrated.py`.
- Some older scripts in the repo may reference legacy filenames; the **canonical** output naming is the one shown in this document.
