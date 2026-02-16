# Problems and Solutions (Detailed, Paper-Relevant)

This document collects practical problems encountered in ATLAS-style experiments and the corresponding mitigations. It is written in a “journal engineering notes” style so you can directly reuse phrasing in the paper.

## 1) Runtime limits (Colab / session-based environments)

**Symptom**: full runs do not finish in a single session.

**Root cause**: end-to-end FL is expensive; external environments impose hard runtime limits.

**Solution**:

- Use `--max-rounds` to cap a session and `--resume` to continue.
- Prefer 15 rounds for paper tables (and optionally add 30-round extended runs).

Command pattern:

```bash
python experiments/atlas_integrated.py --mode full --ablation atlas --rounds 30 --max-rounds 15 --seed 42 --tasks sst2 mrpc cola qnli
python experiments/atlas_integrated.py --mode full --ablation atlas --rounds 30 --resume checkpoints/atlas_atlas_seed42_round_15.pkl --seed 42 --tasks sst2 mrpc cola qnli
```

## 2) Checkpoint explosion / storage pressure

**Symptom**: large numbers of checkpoint files; disk fills up.

**Root cause**: saving checkpoints at frequent intervals is expensive for split FL and transformer backbones.

**Solution**:

- The integrated experiment script is configured to save **final-only** checkpoints (while preserving resume capability).
- For long runs, rely on session-based resume instead of frequent checkpointing.

## 3) Result overwriting across runs

**Symptom**: result files get overwritten when re-running experiments.

**Root cause**: filenames that do not encode the run identity (config/seed).

**Solution**:

- Current scripts include `ablation` and `seed` in filenames:
  - `results/atlas_integrated_{mode}_{ablation}_seed{seed}.json`
  - `checkpoints/atlas_{ablation}_seed{seed}_round_{round}.pkl`

## 4) “Task not supported” failures

**Symptom**: passing extra tasks causes runtime errors.

**Root cause**: tasks must be explicitly wired in the dataset mapping and preprocessing.

**Solution**:

- For the current repo, use only: `sst2`, `mrpc`, `cola`, `qnli`.
- If you need additional tasks, extend the dataset map in `experiments/atlas_integrated.py` and ensure preprocessing matches dataset fields.

## 5) GPU / CUDA memory errors (OOM)

**Symptom**: CUDA out-of-memory during client updates or server pass.

**Common causes**:

- large `--samples` with nontrivial batch sizes
- large backbone models without adjusting compute budget

**Solutions**:

- Reduce `--samples` (e.g., 2000–3000)
- Reduce `--rounds` for initial validation
- If debugging stability, run one seed first before multi-seed runs

## 6) Statistical runs timing out

**Symptom**: `run_statistical_experiments.py` marks runs as failed after a fixed duration.

**Root cause**: subprocess timeout is fixed to 2 hours per run.

**Solutions**:

- Reduce `--rounds` and/or `--samples`
- Use fewer seeds, then scale up
- Run single-run commands to identify bottlenecks before multi-seed sweeps

## 7) Non-determinism across seeds

**Symptom**: results vary across runs more than expected.

**Root cause**: GPU kernels, data loading order, and stochastic optimization.

**Solutions**:

- Always set `--seed` for single runs.
- For paper tables, use the multi-seed runner and report mean ± std, plus significance tests and effect sizes.

## 8) Communication accounting confusion

**Symptom**: communication numbers are not comparable between methods.

**Root cause**: different methods exchange different payloads (activations vs full model vs LoRA-only parameters).

**Solutions**:

- Use the repo’s recorded per-round upload/download bytes and aggregate totals.
- Compare methods under the same tasks, rounds, and seeds.

## 9) Paper wording guidance (professional framing)

When writing the paper, it is both accurate and professional to state:

- “We enforce an operationally realistic runtime budget by using session-based resume and reporting 15-round results with multi-seed statistics.”
- “We avoid checkpoint storage explosion by saving final-only checkpoints and relying on explicit resume for long runs.”
- “We report mean ± std across fixed seeds, along with paired significance tests and effect sizes.”
