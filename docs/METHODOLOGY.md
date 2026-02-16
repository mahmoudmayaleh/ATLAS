# ATLAS Methodology (Detailed)

This document describes the **current** ATLAS methodology as implemented in this repository. It is written to be suitable for a journal/IEEE methods section and is intentionally aligned with runnable code.

## 1) Problem setting and notation

We consider multi-task federated learning (FL) where clients are partitioned across tasks.

- Let $\mathcal{T}$ be a set of tasks (e.g., GLUE-style classification tasks).
- Each client $i$ is associated with a task $t(i)\in\mathcal{T}$.
- Each client holds local data $\mathcal{D}_i$.

ATLAS focuses on **heterogeneity** in:

- task distributions (non-IID across tasks)
- client resources (memory/compute)
- communication constraints

The implementation integrates four phases (see `src/phase1_clustering.py`, `src/phase2_configuration.py`, `src/phase3_split_fl.py`, `src/phase4_laplacian.py`). The end-to-end runnable pipeline is orchestrated by `experiments/atlas_integrated.py`.

## 2) End-to-end pipeline overview

ATLAS is designed as a **pipeline** that transforms a set of heterogeneous clients into a structured training process:

1. **Phase 1 (Task clustering)**: derive task/client similarity via gradient fingerprints and form clusters.
2. **Phase 2 (Heterogeneous LoRA configuration)**: allocate LoRA ranks under per-client resource budgets.
3. **Phase 3 (Split federated learning)**: train a split model with task-aware aggregation and heterogeneous client splits.
4. **Phase 4 (Laplacian regularization)**: enforce smoothness/personalization using a graph regularizer.

In the repo, the integrated “paper-style” runs and baselines are executed via:

- `experiments/atlas_integrated.py --ablation atlas` (full ATLAS)
- `experiments/atlas_integrated.py --ablation fedavg_cluster` (baseline)
- `experiments/atlas_integrated.py --ablation local_only` (baseline)

## 3) Phase 1 — Task clustering via fingerprints

Goal: group clients/tasks that should share information while avoiding harmful mixing.

High-level steps:

1. **Fingerprint extraction**: for each client, compute a compact representation of update/gradient behavior on a small number of batches.
2. **Similarity computation**: compute pairwise similarity/distance between fingerprints.
3. **Cluster selection**: choose $k$ using internal clustering metrics (implementation-dependent), then compute cluster labels.

Practical motivation: in multi-task settings, a single global model aggregation can be suboptimal when tasks differ in label semantics and feature requirements.

## 4) Phase 2 — Device-aware heterogeneous LoRA rank allocation

Goal: assign LoRA ranks per client (and potentially per layer group) that fit within device constraints while preserving adaptation capacity.

Inputs:

- client device profile (e.g., memory budget)
- model dimensionality / hidden size
- optional importance information from clustering or observed statistics

Output:

- per-client rank configuration used in training

ATLAS uses this phase to make training feasible across a mix of low-memory and higher-memory clients.

## 5) Phase 3 — Split federated learning with adaptive split selection

Goal: reduce client memory footprint and manage client-server compute by splitting the model at a layer boundary.

### 5.1 Split learning structure

Let the model be decomposed into a client-side prefix $f_{\le s}$ and a server-side suffix $f_{>s}$ where $s$ is the split layer.

- Client computes activations: $h = f_{\le s}(x)$
- Client sends $h$ to server
- Server computes loss and backprop through suffix, returns gradients w.r.t. $h$
- Client backprops through prefix

This reduces on-client memory/compute compared to full fine-tuning.

### 5.2 Improved split selection (implemented)

The improved split selector is inlined in `src/phase3_split_fl.py` as `ImprovedSplitSelector`.

For each candidate split point $s$, it computes a weighted score:

$$
\text{score}(s) = 0.35\,\text{Mem}(s) + 0.30\,\text{Comm}(s) + 0.25\,\text{Imp}(s) + 0.10\,\text{Bal}(s)
$$

Where:

- **Mem(s)**: soft penalty if estimated memory exceeds the client’s budget (with a safety margin)
- **Comm(s)**: proxy for activation transfer time given bandwidth and optional compression
- **Imp(s)**: optional importance proxy (can use fingerprint gradients when available)
- **Bal(s)**: encourages balanced client/server workload

The public helper `get_split_point(model_name, device_profile=None)` provides a backward-compatible API: it returns a heuristic split when no `device_profile` is supplied, and uses the improved selector when it is.

### 5.3 Communication accounting

Communication is dominated by activation transfer for split learning (and any metadata exchange). The experimental pipeline records per-round upload/download byte counts and aggregates them into MB totals.

## 6) Phase 4 — Laplacian regularization for personalization

Goal: improve personalization and stability by encouraging clients that are “close” in a similarity graph to have similar updates while allowing task-specific deviations.

Typical structure:

- build adjacency/weight matrix from fingerprints or embeddings
- add regularization term scaled by a hyperparameter (exposed as `--eta` in integrated runs)

This phase is designed to improve per-client consistency (reduce variance) without collapsing all clients to identical behavior.

## 7) What is “paper-ready” in this repo

ATLAS includes a multi-seed runner for reporting statistical results:

- `experiments/run_statistical_experiments.py` runs multiple seeds per configuration
- computes mean ± std across seeds
- runs paired significance tests (t-test and Wilcoxon when applicable)
- reports effect sizes (Cohen’s d)

See `docs/EXPERIMENTS.md` for exact commands and outputs.

## 8) Implementation alignment (source of truth)

- End-to-end experiment pipeline: `experiments/atlas_integrated.py`
- Multi-seed statistical runner: `experiments/run_statistical_experiments.py`
- Split selection + split FL: `src/phase3_split_fl.py`

This document intentionally avoids claims that are not wired in the current code (e.g., tasks or modalities not present in the dataset mapping).
