#!/usr/bin/env python3
"""
ATLAS E2E NLG Experiment Runner
================================
Dedicated script for E2E NLG experiments that directly compare ATLAS against
SplitLoRA / HSplitLoRA baselines on the E2E NLG Challenge dataset.

This produces the tables and PPL curves matching:
  - SplitLoRA Table I/II: BLEU, NIST, METEOR, ROUGE-L
  - HSplitLoRA Table I:   same NLG metrics under homo/hetero settings
  - HSplitLoRA Fig 18:    PPL vs round curves
  - SplitLoRA Fig 2/3:    PPL convergence curves

Dataset : tuetschek/e2e_nlg  (parquet mirror, no legacy script required)
          Columns: meaning_representation, human_reference
Models  : gpt2, gpt2-medium  (LLaMA-7B infeasible on T4 GPU)
Clients : N=3-5
LoRA ranks: [2, 4, 8]
Settings: homo (uniform rank), hetero (mixed ranks via ATLAS Phase 2)
Baselines: local_only, standard_fl (FedAvg-LoRA), atlas

Usage:
    python experiments/run_e2e_nlg.py --model gpt2 --clients 5 --rounds 15
    python experiments/run_e2e_nlg.py --model gpt2 --setting hetero --rounds 10
    python experiments/run_e2e_nlg.py --model gpt2-medium --clients 3 --rounds 10 --ablation standard_fl
"""

import sys
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import gc
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

from metrics import (
    compute_perplexity_from_loss,
    compute_nlg_metrics,
    count_trainable_params,
    capture_memory_stats,
    find_convergence_round,
)
from src.phase2_configuration import DeviceProfiler, RankAllocator


# ── Rank allocation helpers ────────────────────────────────────────────────────

# Device types used to simulate heterogeneous clients.
# homo  → all clients share the same profile (edge_gpu).
# hetero → clients span a realistic capability range.
_HOMO_DEVICE   = 'edge_gpu'
_HETERO_DEVICES = ['smartphone', 'tablet', 'laptop_cpu', 'gpu', 'gpu_16gb']

_MODEL_DIMS = {'gpt2': 768, 'gpt2-medium': 1024}

def assign_ranks_atlas(num_clients: int, model_name: str,
                       setting: str, seed: int) -> List[int]:
    """Assign one LoRA rank per client using ATLAS DeviceProfiler + RankAllocator.

    homo  → every client gets the same device profile → same rank.
    hetero → each client gets a different device profile → different ranks,
             mirroring the HSplitLoRA heterogeneous setting.

    The RankAllocator respects the memory budget constraint
      Σ(2·d·r·b) ≤ C_mem
    and returns a per-layer rank list; we take the mode (most common rank)
    as the single rank for that client's uniform LoRA model.
    """
    profiler  = DeviceProfiler()
    model_dim = _MODEL_DIMS.get(model_name.lower(), 768)
    allocator = RankAllocator(model_dim=model_dim)

    # Uniform importance: all layers equally important at initialisation
    # (no gradient info yet). n_layers=12 for GPT-2, 24 for gpt2-medium.
    n_layers  = 12 if 'medium' not in model_name.lower() else 24
    uniform_importance = {f'layer_{i}': 1.0 / n_layers for i in range(n_layers)}

    rng = np.random.RandomState(seed)
    ranks = []
    for i in range(num_clients):
        if setting == 'homo':
            device_type = _HOMO_DEVICE
        else:
            # Cycle through heterogeneous device types
            pool = _HETERO_DEVICES * ((num_clients // len(_HETERO_DEVICES)) + 1)
            rng.shuffle(pool)
            device_type = pool[i]

        profile = profiler.profile_device(device_type)
        per_layer_ranks = allocator.allocate_ranks(
            device_profile=profile,
            importance_scores=uniform_importance,
            n_layers=n_layers,
        )
        # Use the median rank (rounded down to nearest power-of-2 candidate)
        median_r = int(np.median(per_layer_ranks))
        # Clamp to PEFT-friendly values
        clamped = max(2, min(median_r, max(profile.get('suggested_ranks', [8]))))
        ranks.append(clamped)
        print(f"  Client {i} → device={device_type:12s} | "
              f"per-layer ranks={sorted(set(per_layer_ranks))} → model rank={clamped}")

    return ranks


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class E2ENLGConfig:
    """Configuration for E2E NLG experiments."""
    model_name: str = "gpt2"
    num_clients: int = 5
    num_rounds: int = 15
    local_epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 5e-5
    gradient_clip_norm: float = 1.0
    max_seq_length: int = 256         # covers MR + reference in one pass
    max_train_samples: int = 3000     # per client
    lora_rank: int = 4                # default; overridden in hetero mode
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    noniid_alpha: float = 0.3         # Dirichlet concentration (lower = more non-IID)
    eta_laplacian: float = 0.3        # MIRA Laplacian regularisation strength
    setting: str = "homo"             # "homo" or "hetero"
    ablation: str = "atlas"           # "atlas", "standard_fl", "local_only"
    seed: int = 42
    generate_max_new_tokens: int = 80
    generate_num_beams: int = 4
    eval_gen_samples: int = 200       # max samples used for NLG generation eval


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_e2e_dataset(tokenizer, config: E2ENLGConfig):
    """Load and tokenize the E2E NLG dataset for causal LM training.

    Dataset columns:
        meaning_representation  – structured restaurant attribute string (input)
        human_reference         – natural language description (target)

    Format fed to the model: "MR: <meaning_repr> Text: <reference>"
    Labels mask the prompt tokens (-100) so loss is only on the generated text.

    Loading strategy: use the official CSV files from the tuetschek/e2e-dataset
    GitHub repo (raw parquet/csv path that avoids legacy dataset scripts).  Both
    the HuggingFace `tuetschek/e2e_nlg` and `GEM/e2e_nlg` repos have a
    `e2e_nlg.py` dataset script that newer `datasets` (≥2.20) refuses to run.
    Loading from CSV skips that script entirely.
    """
    _BASE = "https://github.com/tuetschek/e2e-dataset/raw/master/"
    # Only load train and validation — the testset CSV has no 'ref' column
    # (it is MR-only, distributed without references).  All evaluation uses
    # the validation split which does have human_reference strings.
    _SPLITS = {
        'train':      _BASE + 'trainset.csv',
        'validation': _BASE + 'devset.csv',
    }
    print("[DATA] Loading E2E NLG from CSV (tuetschek/e2e-dataset on GitHub) ...")
    from datasets import DatasetDict

    def _load_and_normalise(split_name, url):
        """Load one CSV split and rename columns to a canonical schema."""
        ds = load_dataset('csv', data_files={split_name: url}, split=split_name)
        # Build a case-insensitive map: lower(col) → actual col name
        col_map = {c.lower(): c for c in ds.column_names}
        orig_mr  = col_map.get('mr')
        orig_ref = col_map.get('ref')
        if orig_mr and orig_mr != 'meaning_representation':
            ds = ds.rename_column(orig_mr, 'meaning_representation')
        if orig_ref and orig_ref != 'human_reference':
            ds = ds.rename_column(orig_ref, 'human_reference')
        return ds

    dataset = DatasetDict({
        name: _load_and_normalise(name, url)
        for name, url in _SPLITS.items()
    })

    def tokenize_fn(examples):
        input_texts  = [f"MR: {mr} Text: {ref}"
                        for mr, ref in zip(examples['meaning_representation'],
                                           examples['human_reference'])]
        prompt_texts = [f"MR: {mr} Text:"
                        for mr in examples['meaning_representation']]

        full_enc   = tokenizer(input_texts,  padding='max_length', truncation=True,
                               max_length=config.max_seq_length, return_tensors=None)
        prompt_enc = tokenizer(prompt_texts, padding=False, truncation=True,
                               max_length=config.max_seq_length, return_tensors=None,
                               add_special_tokens=False)

        labels = []
        for ids, mask, prompt_ids in zip(full_enc['input_ids'],
                                          full_enc['attention_mask'],
                                          prompt_enc['input_ids']):
            # Mask prompt tokens and padding; only compute loss on generated tokens
            prompt_len = len(prompt_ids)
            lab = [-100] * prompt_len + [
                tok if mask[i] == 1 else -100
                for i, tok in enumerate(ids[prompt_len:], start=prompt_len)
            ]
            # Ensure same length as ids
            lab = lab[:len(ids)]
            while len(lab) < len(ids):
                lab.append(-100)
            labels.append(lab)

        full_enc['labels'] = labels
        return full_enc

    col_names = dataset['train'].column_names
    train_ds = dataset['train'].map(
        tokenize_fn, batched=True, remove_columns=col_names,
        load_from_cache_file=False)
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    val_ds = dataset['validation'].map(
        tokenize_fn, batched=True, remove_columns=col_names,
        load_from_cache_file=False)
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Keep raw strings for NLG generation eval
    raw_val_mrs  = dataset['validation']['meaning_representation']
    raw_val_refs = dataset['validation']['human_reference']

    return train_ds, val_ds, raw_val_refs, raw_val_mrs


def partition_dataset(dataset, num_clients: int, max_per_client: int,
                      seed: int, alpha: float = 100.0):
    """Partition dataset among clients using Dirichlet allocation.

    Lower alpha = more non-IID (heterogeneous) data distributions.
    alpha >= 100 approximates IID.  alpha ~ 0.3 creates strong skew.
    Sorts samples into shards, then distributes shards via Dirichlet(alpha)
    so that different clients receive different amounts & content clusters.
    """
    rng = np.random.RandomState(seed)
    n = len(dataset)
    num_shards = num_clients * 4
    shard_size = max(1, n // num_shards)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for s in range(num_shards):
        start = s * shard_size
        end = min(start + shard_size, n) if s < num_shards - 1 else n
        shard = list(range(start, end))
        rng.shuffle(shard)
        props = rng.dirichlet([alpha] * num_clients)
        splits = (np.cumsum(props) * len(shard)).astype(int)
        prev = 0
        for cid in range(num_clients):
            client_indices[cid].extend(shard[prev:splits[cid]])
            prev = splits[cid]

    subsets = []
    for cid in range(num_clients):
        inds = client_indices[cid][:max_per_client]
        if not inds:
            inds = list(rng.choice(n, size=min(50, n), replace=False))
        subsets.append(Subset(dataset, inds))
    return subsets


# ── Model helpers ──────────────────────────────────────────────────────────────

def create_lora_model(model_name: str, rank: int, alpha: int = 16,
                      dropout: float = 0.05) -> nn.Module:
    """Load a causal LM and apply LoRA."""
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    # Newer Transformers versions warn when loss_type=None is found in the config.
    # Setting it explicitly suppresses the warning without changing behaviour.
    if getattr(model.config, 'loss_type', None) is None:
        model.config.loss_type = "ForCausalLMLoss"
    target_modules = ["c_attn", "c_proj"] if 'gpt2' in model_name.lower() else ["q_proj", "v_proj"]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_cfg)


# ── Training & evaluation ──────────────────────────────────────────────────────

def train_one_round(model: nn.Module, dataloader: DataLoader,
                    config: E2ENLGConfig, device: str) -> float:
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate, weight_decay=0.01)

    total_loss, n_batches = 0.0, 0
    for _ in range(config.local_epochs):
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

    model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total_loss / max(n_batches, 1)


def evaluate_ppl(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    model.eval()
    model.to(device)
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            out   = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            valid = (labels != -100).sum().item()
            total_loss   += out.loss.item() * valid
            total_tokens += valid

    model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    avg_loss = total_loss / max(total_tokens, 1)
    return compute_perplexity_from_loss(avg_loss)


def generate_texts(model: nn.Module, tokenizer, mrs: List[str],
                   config: E2ENLGConfig, device: str) -> List[str]:
    model.eval()
    model.to(device)

    # GPT-2 sets pad_token = eos_token (both id=50256).  When pad==eos we must
    # NOT pass eos_token_id to generate(): HuggingFace stops generation at the
    # very first EOS token it produces regardless of early_stopping, which
    # makes every output an empty string.  Instead we let generation run for
    # the full max_new_tokens budget and rely on skip_special_tokens=True to
    # strip any stray EOS/pad tokens from the decoded text.
    pad_equals_eos = tokenizer.pad_token_id == tokenizer.eos_token_id
    gen_cfg = GenerationConfig(
        max_new_tokens=config.generate_max_new_tokens,
        num_beams=1 if pad_equals_eos else config.generate_num_beams,
        do_sample=False,
        early_stopping=False if pad_equals_eos else True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=None if pad_equals_eos else tokenizer.eos_token_id,
    )

    predictions = []
    with torch.no_grad():
        for mr in mrs[:config.eval_gen_samples]:
            prompt  = f"MR: {mr} Text:"
            inputs  = tokenizer(prompt, return_tensors='pt', truncation=True,
                                max_length=config.max_seq_length).to(device)
            out_ids = model.generate(**inputs, generation_config=gen_cfg)
            generated = tokenizer.decode(
                out_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True).strip()
            predictions.append(generated)

    model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Diagnostic: warn if most predictions are empty
    empty_count = sum(1 for p in predictions if not p.strip())
    if empty_count > 0:
        print(f"  [WARN] {empty_count}/{len(predictions)} predictions are empty strings")
    if predictions:
        print(f"  [GEN SAMPLE] predictions[0]: {repr(predictions[0][:120])}")

    return predictions


# ── FedAvg / ATLAS aggregation ─────────────────────────────────────────────────

def fedavg_aggregate(models: List[nn.Module]):
    avg = {}
    n = len(models)
    for name, param in models[0].named_parameters():
        if not param.requires_grad:
            continue
        avg[name] = param.data.clone()
        for m in models[1:]:
            for n2, p2 in m.named_parameters():
                if n2 == name:
                    avg[name] += p2.data
                    break
        avg[name] /= n
    for model in models:
        for name, param in model.named_parameters():
            if name in avg:
                param.data.copy_(avg[name])


def atlas_aggregate(models: List[nn.Module], ranks: List[int],
                    eta: float = 0.3):
    """ATLAS aggregation: MIRA Laplacian regularisation + hetero-rank
    pad-average-truncate.

    Step 1 – Laplacian nudge on local LoRA params:
        W_k <- W_k - eta * sum_l a_kl (W_k - W_l)
    Step 2 – Pad smaller LoRA matrices to max rank, average, truncate back.
    """
    n_clients = len(models)
    max_r = max(ranks)

    # ── Step 1: MIRA Laplacian regularisation ──────────────────────────────
    if eta > 0:
        # Flatten each client's LoRA params for pairwise cosine similarity
        vecs = []
        for model in models:
            parts = []
            for name, p in model.named_parameters():
                if p.requires_grad and 'lora' in name.lower():
                    parts.append(p.data.flatten().float())
            vecs.append(torch.cat(parts) if parts else torch.zeros(1))

        # Cosine-similarity adjacency (negative similarities clamped to 0)
        adj = torch.zeros(n_clients, n_clients)
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                ml = min(len(vecs[i]), len(vecs[j]))
                sim = Fn.cosine_similarity(
                    vecs[i][:ml].unsqueeze(0), vecs[j][:ml].unsqueeze(0)
                ).item()
                w = max(sim, 0.0)
                adj[i][j] = w
                adj[j][i] = w

        # Snapshot local states before nudging
        local_states = []
        for model in models:
            state = {}
            for name, p in model.named_parameters():
                if p.requires_grad:
                    state[name] = p.data.clone()
            local_states.append(state)

        # Laplacian nudge: W_k <- W_k - eta * sum_l a_kl (W_k - W_l)
        for i, model in enumerate(models):
            for name, p in model.named_parameters():
                if not p.requires_grad or 'lora' not in name.lower():
                    continue
                w_k = local_states[i][name].float()
                lap = torch.zeros_like(w_k)
                for j in range(n_clients):
                    if i == j or adj[i][j] < 1e-6:
                        continue
                    a_ij = adj[i][j].item()
                    w_l = local_states[j][name].float()
                    if 'lora_A' in name and w_k.shape[0] != w_l.shape[0]:
                        mr = min(w_k.shape[0], w_l.shape[0])
                        lap[:mr] += a_ij * (w_k[:mr] - w_l[:mr])
                    elif 'lora_B' in name and w_k.shape[-1] != w_l.shape[-1]:
                        mr = min(w_k.shape[-1], w_l.shape[-1])
                        lap[..., :mr] += a_ij * (w_k[..., :mr] - w_l[..., :mr])
                    else:
                        lap += a_ij * (w_k - w_l)
                p.data = (w_k - eta * lap).to(p.dtype)

    # ── Step 2: heterogeneous-rank pad-average-truncate ────────────────────
    acc    = {}
    counts = {}

    for i, model in enumerate(models):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            data = param.data.clone()
            if 'lora_A' in name and data.shape[0] < max_r:
                pad = torch.zeros(max_r, data.shape[1], device=data.device)
                pad[:data.shape[0]] = data
                data = pad
            elif 'lora_B' in name and data.shape[-1] < max_r:
                pad = torch.zeros(*data.shape[:-1], max_r, device=data.device)
                pad[..., :data.shape[-1]] = data
                data = pad
            acc[name]    = acc.get(name, torch.zeros_like(data)) + data
            counts[name] = counts.get(name, 0) + 1

    for name in acc:
        acc[name] /= counts[name]

    for i, model in enumerate(models):
        r = ranks[i]
        for name, param in model.named_parameters():
            if name not in acc:
                continue
            agg = acc[name]
            if 'lora_A' in name:
                param.data.copy_(agg[:r])
            elif 'lora_B' in name:
                param.data.copy_(agg[..., :r])
            else:
                param.data.copy_(agg)


# ── Main experiment ────────────────────────────────────────────────────────────

def run_e2e_experiment(config: E2ENLGConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"E2E NLG — {config.model_name} | {config.ablation} | {config.setting}")
    rank_info = f"base={config.lora_rank} (hetero per-client)" if config.setting == 'hetero' and config.ablation != 'standard_fl' else str(config.lora_rank)
    print(f"Clients={config.num_clients} | Rounds={config.num_rounds} | Rank={rank_info}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    import random
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds, raw_val_refs, raw_val_mrs = load_e2e_dataset(tokenizer, config)
    client_datasets = partition_dataset(train_ds, config.num_clients,
                                        config.max_train_samples, config.seed,
                                        alpha=config.noniid_alpha)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size * 2, shuffle=False) # type: ignore

    print(f"  Train: {len(train_ds)} total | {[len(c) for c in client_datasets]} per client")
    print(f"  Val  : {len(val_ds)} samples")

    # Assign LoRA ranks per client using ATLAS DeviceProfiler + RankAllocator
    print(f"\n[RANKS] Computing ATLAS rank allocation ({config.setting} setting) ...")
    atlas_ranks = assign_ranks_atlas(config.num_clients, config.model_name,
                                     config.setting, config.seed)
    if config.ablation == 'standard_fl' and config.setting == 'hetero':
        # Standard FL cannot handle heterogeneous ranks — all clients must use
        # the minimum affordable rank (weakest-device constraint)
        min_rank = min(atlas_ranks)
        ranks = [min_rank] * config.num_clients
        print(f"  standard_fl (hetero): forced to uniform rank={min_rank} "
              f"(weakest device constraint, ATLAS would use {atlas_ranks})")
    elif config.ablation == 'standard_fl':
        # Homo: all devices identical → use same common rank
        ranks = atlas_ranks
        print(f"  standard_fl (homo): uniform rank={ranks[0]}")
    else:
        # ATLAS and local_only use device-aware heterogeneous ranks
        ranks = atlas_ranks
    print(f"  Final ranks: {ranks}")

    print(f"\n[MODELS] Creating {config.num_clients} client models ...")
    client_models = [create_lora_model(config.model_name, r, config.lora_alpha,
                                       config.lora_dropout)
                     for r in ranks]

    results = {
        'config': asdict(config),
        'client_ranks': ranks,
        'round_metrics': [],
        'ppl_curve': [],
        'nlg_metrics': {},
        'trainable_params': {i: count_trainable_params(m)
                             for i, m in enumerate(client_models)},
    }

    start_time = time.time()
    ppl_last = [None] * config.num_clients

    for round_idx in range(config.num_rounds):
        round_start = time.time()
        print(f"\n[Round {round_idx+1}/{config.num_rounds}]")

        round_losses = {}
        for cid in range(config.num_clients):
            loader = DataLoader(client_datasets[cid], batch_size=config.batch_size,
                                shuffle=True)
            loss = train_one_round(client_models[cid], loader, config, device)
            round_losses[cid] = loss
            print(f"  Client {cid}: train_loss={loss:.4f}")

        if config.ablation == 'standard_fl':
            fedavg_aggregate(client_models)
        elif config.ablation == 'atlas':
            atlas_aggregate(client_models, ranks, eta=config.eta_laplacian)
        # local_only: no aggregation

        ppls = [evaluate_ppl(m, val_loader, device) for m in client_models]
        ppl_last = ppls
        avg_ppl   = float(np.mean(ppls))
        round_time = time.time() - round_start

        results['round_metrics'].append({
            'round': round_idx + 1,
            'train_losses': round_losses,
            'client_ppls': {i: p for i, p in enumerate(ppls)},
            'avg_ppl': avg_ppl,
            'avg_train_loss': float(np.mean(list(round_losses.values()))),
            'time_seconds': round_time,
        })
        results['ppl_curve'].append(avg_ppl)
        print(f"  Avg PPL={avg_ppl:.2f} | time={round_time:.1f}s")

    # ── Final NLG eval ─────────────────────────────────────────────────────────
    print(f"\n[NLG EVAL] Generating outputs ...")
    best_cid   = int(np.argmin(ppl_last)) if ppl_last[0] is not None else 0 # type: ignore
    eval_model = client_models[best_cid]
    print(f"  Using client {best_cid} (PPL={ppl_last[best_cid]:.2f})")

    preds    = generate_texts(eval_model, tokenizer, raw_val_mrs, config, device)
    refs_sub = raw_val_refs[:len(preds)]
    nlg      = compute_nlg_metrics(preds, refs_sub)
    results['nlg_metrics'] = nlg
    results['final_ppl']   = float(ppl_last[best_cid]) # type: ignore
    results['memory_stats'] = capture_memory_stats(device)
    results['convergence']  = {
        'ppl_convergence_round': find_convergence_round(
            results['ppl_curve'], mode='min', threshold_frac=0.95),
        'total_wall_clock_seconds': time.time() - start_time,
    }

    print(f"\n{'='*70}")
    print(f"FINAL — {config.model_name} | {config.ablation} | {config.setting}")
    print(f"  Final PPL : {results['final_ppl']:.2f}")
    for k, v in nlg.items():
        print(f"  {k:10s}: {v:.2f}")
    conv_r = results['convergence']['ppl_convergence_round']
    print(f"  Convergence round: {conv_r}")
    print(f"  Wall-clock: {results['convergence']['total_wall_clock_seconds']:.1f}s")
    print(f"{'='*70}\n")

    # ── Save ───────────────────────────────────────────────────────────────────
    def _jsonable(obj):
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)): return obj.item()
        if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
        if isinstance(obj, dict):  return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_jsonable(v) for v in obj]
        return obj

    model_norm = config.model_name.replace('/', '_')
    # Use the actual assigned ranks (not the CLI default lora_rank which is stale
    # once ATLAS DeviceProfiler takes over rank assignment).
    # Format: rUNIQ if all clients share one rank, rMIN-MAX otherwise.
    unique_ranks = sorted(set(ranks))
    rank_tag = (f"r{unique_ranks[0]}" if len(unique_ranks) == 1
                else f"r{unique_ranks[0]}-{unique_ranks[-1]}")
    out_path = (Path("./results") /
                f"e2e_{model_norm}_{config.ablation}_{config.setting}"
                f"_{rank_tag}_seed{config.seed}_rounds{config.num_rounds}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(_jsonable(results), f, indent=2)
    print(f"[SAVED] {out_path}")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATLAS E2E NLG Experiment")
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium"])
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--setting", default="homo", choices=["homo", "hetero"])
    parser.add_argument("--ablation", default="atlas",
                        choices=["atlas", "standard_fl", "local_only"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noniid-alpha", type=float, default=0.3,
                        help="Dirichlet concentration for non-IID partition "
                             "(lower = more skewed, >=100 ≈ IID)")
    parser.add_argument("--eta-laplacian", type=float, default=0.3,
                        help="MIRA Laplacian regularisation strength")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--eval-gen-samples", type=int, default=200)
    args = parser.parse_args()

    cfg = E2ENLGConfig(
        model_name=args.model,
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        noniid_alpha=args.noniid_alpha,
        eta_laplacian=args.eta_laplacian,
        setting=args.setting,
        ablation=args.ablation,
        seed=args.seed,
        max_train_samples=args.max_samples,
        eval_gen_samples=args.eval_gen_samples,
    )
    run_e2e_experiment(cfg)
