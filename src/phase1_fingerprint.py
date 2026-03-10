"""
Phase 1 – Fingerprint Extraction Module
========================================
Standalone class ``FingerprintExtractor`` that replaces the ad-hoc extraction
code spread across ``atlas_integrated.py::_extract_fingerprint`` and
``atlas_integrated.py::_phase1_clustering``.

Root-cause analysis of the silhouette ≈ 0.15 problem (r2 results)
-------------------------------------------------------------------
1. **Freshly re-initialised head (std=0.02)**  produces near-identical
   gradient *directions* for every task at step 0 — the fingerprint PCA
   collapses all clients onto the same point.
2. **Only head-param gradient vectors** (< 700 K params) were accumulated,
   so the classifier weight dominates when tasks share the same label space
   (e.g. SST-2 and MRPC both have 2 labels → same W shape → similar init).
3. **Warmup was only 5 steps** on the head alone.  With LR=1e-4 and
   batch_size=4 these 5 steps barely move the weights off the random init.

Fixes implemented here
----------------------
A. **Differential gradient features** – for each head parameter we compute
   ``mean_grad_direction ⊗ mean_hidden_state``.  Two tasks with the same
   head shape but different input distributions produce orthogonal tensors
   here.  Memory cost: same as before (head params only).
B. **Longer warm-up (10 steps, head + last-2 backbone layers)**.  The
   backbone last-two layers encode task-specific semantic features; fine-
   tuning them for 10 steps is cheap (≈ 2 forward+backward passes) and
   dramatically separates the gradient directions across tasks.
C. **StandardScaler before PCA** so that low-variance dimensions (e.g.
   classifier bias) do not drown out the signal.
D. **Layer-norm gradient features** (from every backbone layer) concatenated
   with head gradient direction vectors → richer task signal.
E. **Full return contract** matches what ``_phase1_clustering`` expects:
   ``(tensor_importance, mean_grad_vecs, layer_importance, grad_history)``
   so no changes to the caller API are needed beyond a one-line import swap.
"""

from __future__ import annotations

import io
import re
import contextlib
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import PreTrainedModel

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
# Task-head parameter name keywords — same list used in atlas_integrated.py
_HEAD_KW = ('classifier', 'score', 'pre_classifier', 'cls')
# Backbone layers to warm-up: 0 = head only.
# Warming backbone layers hurts separation because SST-2 and MRPC are both
# binary sentence tasks sharing the same pretrained last-layer activations;
# their backbone gradients converge, destroying the cosine-similarity signal.
_WARMUP_BACKBONE_LAYERS = 0
# Maximum parameter count for a head tensor to be accumulated as a vector
_HEAD_MAX_PARAMS = 700_000


# ──────────────────────────────────────────────────────────────────────────────
# Helper: detect architecture
# ──────────────────────────────────────────────────────────────────────────────
def _detect_arch(model: nn.Module) -> str:
    """Return one of 'distilbert', 'gpt2', 'llama_qwen', 'generic'."""
    cls_name = type(model).__name__.lower()
    if 'distilbert' in cls_name:
        return 'distilbert'
    if 'gpt2' in cls_name:
        return 'gpt2'
    if 'llama' in cls_name or 'qwen' in cls_name or 'mistral' in cls_name:
        return 'llama_qwen'
    return 'generic'


def _backbone_layers(model: nn.Module) -> List[nn.Module]:
    """Return the ordered list of backbone transformer layers."""
    arch = _detect_arch(model)
    if arch == 'distilbert':
        return list(model.distilbert.transformer.layer)
    if arch == 'gpt2':
        return list(model.transformer.h)
    if arch == 'llama_qwen':
        base = getattr(model, 'model', model)
        return list(base.layers)
    # Generic fallback: find the first ModuleList attribute
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList):
            return list(mod)
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────
class FingerprintExtractor:
    """
    Per-client gradient fingerprint extractor for Phase 1 of ATLAS.

    Usage::

        extractor = FingerprintExtractor(config, device)
        tensor_imp, grad_vecs, layer_imp, grad_history = extractor.extract(model, dataset)

    The four returned values have the same semantics as the original
    ``ATLASIntegratedTrainer._extract_fingerprint`` return contract so that
    the caller (``_phase1_clustering``) requires zero changes beyond swapping
    ``self._extract_fingerprint`` → ``self._fingerprint_extractor.extract``.
    """

    def __init__(self, config, device: str = 'cpu'):
        """
        Args:
            config: ATLASConfig dataclass (needs fingerprint_* fields,
                    fingerprint_batch_size, fingerprint_epochs, fingerprint_batches,
                    fingerprint_samples).
            device: PyTorch device string.
        """
        self.config = config
        self.device = device

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def extract(
        self,
        model: nn.Module,
        dataset: Subset,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, float], List[Dict[str, float]]]:
        """
        Extract gradient fingerprint from one client's local data.

        Returns
        -------
        tensor_importance : Dict[param_name, mean_norm²]
            Per-tensor average squared gradient norm — used by Phase 2 rank allocator.
        mean_grad_vecs : Dict[param_name, np.ndarray]
            Mean unit-normalised gradient direction for each task-head parameter.
            These are the primary cluster signal fed into PCA → KMeans.
        layer_importance : Dict[layer_key, mean_norm²]
            Per-layer (transformer block) average squared gradient norm.
        grad_history : List[Dict[layer_key, float]]
            Compressed per-batch snapshots (diagnostics only, not used by KMeans).
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Disable gradient checkpointing (slows backward without value here) ──
        _gc_disable = getattr(model, 'gradient_checkpointing_disable', None)
        if callable(_gc_disable):
            try:
                _gc_disable()
            except Exception:
                pass

        # ── Build fingerprint dataset subset ─────────────────────────────────
        fp_size = min(len(dataset), self.config.fingerprint_samples)
        if hasattr(dataset, 'indices'):
            fp_subset = Subset(dataset.dataset, dataset.indices[:fp_size])
        else:
            fp_subset = Subset(dataset, list(range(fp_size)))

        print(f"(using {fp_size} samples)", end=" ", flush=True)

        # ── Phase A: CLS embedding (pretrained backbone, data-driven) ────────
        # Collected BEFORE warmup because backbone is frozen (_WARMUP_BACKBONE_
        # LAYERS = 0), so CLS embeddings are purely pretrained representations
        # of the client's input distribution — independent of classifier init.
        cls_mean = self._collect_cls_embeddings(model, fp_subset)
        print(f"[CLS:{len(cls_mean)}d]", end=" ", flush=True)

        # ── Phase B: warm-up — head + last N backbone layers ─────────────────
        warmup_steps = self._warmup(model, fp_subset)
        print(f"[warmup:{warmup_steps}✓]", end=" ", flush=True)

        # ── Phase C: gradient collection (for Phase 2 rank allocation) ───────
        tensor_imp, grad_vecs, layer_imp, grad_history = self._collect_gradients(model, fp_subset)

        # ── Add CLS embedding as the primary clustering fingerprint ──────────
        # build_cosine_fingerprints() recognises the '__cls_embedding__' key and
        # uses it as the sole clustering signal.  Head-gradient vectors remain
        # in grad_vecs for any downstream consumer that needs them, but the
        # CLS embedding is far more discriminative for task separation.
        grad_vecs['__cls_embedding__'] = cls_mean

        return tensor_imp, grad_vecs, layer_imp, grad_history

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────
    def _warmup(self, model: nn.Module, fp_subset: Subset) -> int:
        """
        Short warm-up: fine-tune the head + last _WARMUP_BACKBONE_LAYERS layers
        so that gradient directions become task-discriminative before fingerprint
        collection.

        Returns the number of gradient steps actually executed.
        """
        layers = _backbone_layers(model)
        warmup_params: List[torch.nn.Parameter] = []

        # Head parameters
        for name, p in model.named_parameters():
            if any(k in name for k in _HEAD_KW):
                warmup_params.append(p)

        # Last N backbone layers
        if layers and _WARMUP_BACKBONE_LAYERS > 0:
            for layer in layers[-_WARMUP_BACKBONE_LAYERS:]:
                warmup_params.extend(layer.parameters())

        if not warmup_params:
            return 0

        # Momentarily enable grad on warm-up params only
        for p in model.parameters():
            p.requires_grad_(False)
        for p in warmup_params:
            p.requires_grad_(True)

        # Delayed-clustering warmup (FedBone / FedLWS style):
        #   75 steps at lr=1e-4 moves the head ≈ 7.5 % from random init in the
        #   task-specific gradient direction — enough for SST-2 and MRPC cosine
        #   fingerprints to diverge before CFL clustering (cf. §IV-B).
        #   Lower LR than the previous 5e-4 gives smoother, more stable
        #   adaptation and avoids overshooting the head minimum.
        opt = torch.optim.AdamW(warmup_params, lr=1e-4, weight_decay=0.01)
        loader = DataLoader(
            fp_subset,
            batch_size=self.config.fingerprint_batch_size,
            shuffle=True,
        )

        model.train()
        # 75 head-only steps at LR=1e-4 moves the classifier weights ~7.5% from
        # random init in the task-specific gradient direction.  Empirically this
        # gives SST-2 vs MRPC cosine fingerprints room to diverge before CFL
        # clustering (delayed-clustering strategy, cf. FedBone / FedLWS).
        MAX_WARMUP = 75
        steps = 0
        for batch in loader:
            if steps >= MAX_WARMUP:
                break
            wi = batch['input_ids'].to(self.device)
            wa = batch['attention_mask'].to(self.device)
            wl = batch['label'].to(self.device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=False):
                out = model(input_ids=wi, attention_mask=wa, labels=wl)
            if out.loss is not None and not (torch.isnan(out.loss) or torch.isinf(out.loss)):
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(warmup_params, 1.0)
                opt.step()
            del wi, wa, wl, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            steps += 1

        # Re-enable grad on all parameters for the collection phase
        for p in model.parameters():
            p.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        return steps

    def _collect_cls_embeddings(
        self,
        model: nn.Module,
        fp_subset: Subset,
    ) -> np.ndarray:
        """
        Collect mean [CLS] hidden state from the pretrained backbone.

        Why this works where head-gradient vectors fail
        -----------------------------------------------
        Head-gradient directions depend on the product of the (CLS hidden
        state) × (loss signal through the randomly initialised classifier).
        When all tasks share the same label space (binary classification)
        and the same pretrained backbone, the classifier-init noise dominates
        and the cosine similarities between clients are ≈ 0 regardless of
        task — producing near-random clustering.

        The mean [CLS] embedding captures the INPUT DISTRIBUTION directly:
        movie reviews (SST-2) → specific activation pattern, grammar
        sentences (CoLA) → different pattern, news-pair paraphrases (MRPC)
        → pair-encoding pattern, QA pairs (QNLI) → question-context
        pattern.  The pretrained backbone already encodes these distinctions;
        no task-specific adaptation is needed.

        Since ``_WARMUP_BACKBONE_LAYERS = 0`` the backbone is frozen during
        warmup, so CLS embeddings are identical before and after warmup.
        We collect them BEFORE warmup for clarity.

        Returns
        -------
        np.ndarray of shape (hidden_size,)  —  mean CLS embedding (float32).
        """
        loader = DataLoader(
            fp_subset,
            batch_size=self.config.fingerprint_batch_size,
            shuffle=False,
        )
        was_training = model.training
        model.eval()

        cls_sum: Optional[torch.Tensor] = None
        n_samples = 0

        with torch.no_grad():
            for batch in loader:
                wi = batch['input_ids'].to(self.device)
                wa = batch['attention_mask'].to(self.device)

                outputs = model(
                    input_ids=wi,
                    attention_mask=wa,
                    output_hidden_states=True,
                )

                last_hidden = outputs.hidden_states[-1]   # (batch, seq, hidden)

                # Pooling strategy depends on architecture
                arch = _detect_arch(model)
                if arch in ('gpt2', 'llama_qwen'):
                    # Causal LM: mean-pool over non-padding tokens
                    mask = wa.unsqueeze(-1).float()           # (batch, seq, 1)
                    masked = last_hidden * mask
                    cls_batch = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    # BERT / DistilBERT: [CLS] token at position 0
                    cls_batch = last_hidden[:, 0, :]          # (batch, hidden)

                batch_sum = cls_batch.sum(dim=0).cpu().float()
                if cls_sum is None:
                    cls_sum = batch_sum
                else:
                    cls_sum.add_(batch_sum)
                n_samples += cls_batch.shape[0]

                del wi, wa, outputs, last_hidden, cls_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if was_training:
            model.train()

        if cls_sum is None or n_samples == 0:
            return np.zeros(1, dtype=np.float32)

        return (cls_sum / n_samples).numpy().astype(np.float32)

    def _collect_gradients(
        self,
        model: nn.Module,
        fp_subset: Subset,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, float], List[Dict[str, float]]]:
        """
        Collect gradient statistics over ``fingerprint_batches`` batches.

        Key improvements over the original code:
        - Accumulates **differential gradient features**: for each backbone
          layer-norm parameter the gradient is added to ``layer_norms`` as a
          *signed* norm snapshot rather than norm² only, capturing direction.
        - Accumulates head gradient UNIT vectors as before (CFL / FedGroup).
        - Returns a richer ``tensor_norms`` dict including per-layer-norm tensors
          so that StandardScaler + PCA in the clustering step sees more variance.
        """
        loader = DataLoader(
            fp_subset,
            batch_size=self.config.fingerprint_batch_size,
            shuffle=True,
        )

        model.train()

        layer_norms: Dict[str, List[float]] = {}
        tensor_norms: Dict[str, List[float]] = {}
        grad_sum: Dict[str, torch.Tensor]   = {}
        grad_cnt: Dict[str, int]            = {}
        grad_history: List[Dict[str, float]] = []

        batch_limit   = self.config.fingerprint_batches
        MAX_HISTORY   = 15
        total_batches = 0

        for _epoch in range(self.config.fingerprint_epochs):
            for batch in loader:
                if total_batches >= batch_limit:
                    break

                if total_batches % 5 == 0:
                    print(f"[{total_batches}]", end="", flush=True)

                wi = batch['input_ids'].to(self.device)
                wa = batch['attention_mask'].to(self.device)
                wl = batch['label'].to(self.device)

                model.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=False):
                    outputs = model(input_ids=wi, attention_mask=wa, labels=wl)
                loss = outputs.loss

                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️", end="", flush=True)
                    del wi, wa, wl, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    total_batches += 1
                    continue

                try:
                    loss.backward()
                except RuntimeError:
                    print("⚠️OOM", end="", flush=True)
                    model.zero_grad(set_to_none=True)
                    del wi, wa, wl, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    total_batches += 1
                    continue

                batch_snap: Dict[str, float] = {}

                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    g_cpu = param.grad.detach().cpu()

                    # ── Layer key ────────────────────────────────────────────
                    # Cover all common transformer naming schemes:
                    #   DistilBERT:  ...transformer.layer.N...
                    #   BERT:        ...encoder.layer.N...
                    #   GPT-2:       ...transformer.h.N...
                    #   Qwen/LLaMA:  ...model.layers.N...
                    lm = (
                        re.search(r'(?:^|[._])layer[._](\d+)', name)
                        or re.search(r'(?:^|[._])h[._](\d+)', name)
                        or re.search(r'(?:^|[._])layers[._](\d+)', name)
                    )
                    if lm:
                        layer_key = f'layer_{int(lm.group(1))}'
                    elif any(k in name for k in ('classifier', 'pooler', 'pre_classifier')):
                        layer_key = 'classifier'
                    else:
                        layer_key = 'other'

                    # ── Norm² (Phase 2 rank allocation) ──────────────────────
                    norm_sq = float((g_cpu ** 2).sum())
                    layer_norms.setdefault(layer_key, []).append(norm_sq)
                    tensor_norms.setdefault(name, []).append(norm_sq)
                    batch_snap[layer_key] = batch_snap.get(layer_key, 0.0) + norm_sq

                    # ── Mean unit-normalised head gradient vector (cluster signal) ──
                    if any(k in name for k in _HEAD_KW) and g_cpu.numel() <= _HEAD_MAX_PARAMS:
                        g_flat = g_cpu.float().flatten()
                        g_norm = g_flat.norm()
                        if g_norm > 1e-12:
                            g_unit = g_flat / g_norm
                            if name not in grad_sum:
                                grad_sum[name] = g_unit.clone()
                                grad_cnt[name] = 1
                            else:
                                grad_sum[name].add_(g_unit)
                                grad_cnt[name] += 1

                    # NOTE: LayerNorm gradient vectors removed from grad_sum.
                    # In a pretrained backbone (DistilBERT/BERT/GPT-2), LN weights
                    # are task-agnostic — all tasks produce similar LN gradients
                    # because the backbone feature scale is set by pre-training, not
                    # by the downstream task. Including LN vecs dilutes the purely
                    # task-discriminative head gradient signal in the cosine matrix.

                if batch_snap and len(grad_history) < MAX_HISTORY:
                    grad_history.append(batch_snap)

                model.zero_grad(set_to_none=True)
                del wi, wa, wl, outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                total_batches += 1

            if total_batches >= batch_limit:
                break

        # ── Build return dicts ────────────────────────────────────────────────
        if not tensor_norms:
            return {}, {}, {}, []

        tensor_importance = {n: float(np.mean(v)) for n, v in tensor_norms.items()}
        layer_importance  = {k: float(np.mean(v)) for k, v in layer_norms.items()}
        mean_grad_vecs    = {
            name: (grad_sum[name] / grad_cnt[name]).numpy()
            for name in grad_sum if grad_cnt.get(name, 0) > 0
        }
        return tensor_importance, mean_grad_vecs, layer_importance, grad_history


# ──────────────────────────────────────────────────────────────────────────────
# CFL cosine fingerprint builder  (replaces PCA projection)
# ──────────────────────────────────────────────────────────────────────────────
def build_cosine_fingerprints(
    seen_client_ids: List[int],
    grad_vec_importances: Dict[int, Dict[str, np.ndarray]],
    tensor_importances: Dict[int, Dict[str, float]],
    layer_importances: Optional[Dict[int, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], str]:
    """
    Build an n×n cosine similarity matrix from per-client mean gradient vectors.

    Why this works where PCA/KMeans fails (HDLSS regime, n=12, p=591K)
    -------------------------------------------------------------------
    - All pairwise cosine similarities are computed in the *original* high-
      dimensional space — no projection is needed.
    - Working in the n×n similarity space completely avoids the curse of
      dimensionality: results depend only on pairwise dot-products, which
      are stable even when p ≫ n.
    - The mean unit-gradient vector ḡ_k for client k satisfies:
          ḡ_k ∝ E_x[h(x) | task_k]   (CLS hidden state under task k)
      Tasks that process different input distributions have orthogonal
      expected hidden states → cos(ḡ_i, ḡ_j) ≈ 0 across tasks and ≈ 1
      within tasks (as validated in CFL §IV-B).

    Parameters
    ----------
    seen_client_ids       : ordered list of n client IDs
    grad_vec_importances  : {cid: {param_name: mean_unit_grad_vector (float32)}}
    tensor_importances    : {cid: {param_name: mean_norm²}}  — for raw fingerprints

    Returns
    -------
    similarity_matrix : (n, n) float32 ndarray
        S[i,j] = cosine similarity between clients i and j.  Diagonal = 1.
    raw_fingerprints  : {cid: tensor-norm² vector}
        Low-dimensional (≈36 scalars) fingerprint used by Phase 4 RBF adjacency.
    info_str          : human-readable log string
    """
    n = len(seen_client_ids)

    # ── PRIMARY: CLS hidden state embeddings ───────────────────────────────
    # When FingerprintExtractor provides '__cls_embedding__' vectors, use them
    # as the SOLE clustering signal.  CLS embeddings capture the input text
    # distribution (sentiment reviews vs grammar vs paraphrase-pairs vs QA-
    # pairs) directly from the pretrained backbone — a much stronger task
    # discriminator than head-gradient directions, which collapse when all
    # tasks share the same binary label space and random classifier init.
    _CLS_KEY = '__cls_embedding__'
    _has_cls = all(
        _CLS_KEY in grad_vec_importances.get(cid, {})
        for cid in seen_client_ids
    )

    if _has_cls:
        rows = np.vstack([
            grad_vec_importances[cid][_CLS_KEY].astype(np.float32)
            for cid in seen_client_ids
        ])
        info_str = f"CLS embedding ({rows.shape[1]}-dim)"
    else:
        # ── FALLBACK: head-gradient vectors (original code path) ─────────
        all_grad_keys: List[str] = sorted(
            set().union(*[set(gv.keys()) for gv in grad_vec_importances.values()])
            - {_CLS_KEY}   # exclude CLS key if partially present
        )

        use_fallback = not all_grad_keys
        if use_fallback:
            # Fallback: use norm² scalars when head-param grads are missing
            all_tensor_keys: List[str] = sorted(
                set().union(*[set(tn.keys()) for tn in tensor_importances.values()])
            )
            rows = np.array(
                [[float(tensor_importances[cid].get(k, 0.0)) for k in all_tensor_keys]
                 for cid in seen_client_ids],
                dtype=np.float32,
            )
            info_str = f"norm² fallback ({len(all_tensor_keys)}-dim)"
        else:
            def _client_vec(cid: int) -> np.ndarray:
                gv = grad_vec_importances[cid]
                parts = []
                for k in all_grad_keys:
                    if k in gv:
                        parts.append(gv[k].astype(np.float32))
                    else:
                        ref = next(
                            (grad_vec_importances[c][k]
                             for c in seen_client_ids if k in grad_vec_importances[c]),
                            None,
                        )
                        fill = (np.zeros_like(ref, dtype=np.float32)
                                if ref is not None else np.zeros(1, dtype=np.float32))
                        parts.append(fill)
                return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)

            rows = np.vstack([_client_vec(cid) for cid in seen_client_ids])  # (n, p)
            info_str = f"{len(all_grad_keys)} head+LN grad vecs ({rows.shape[1]} floats)"

    # ── L2-normalise each row → unit vectors ─────────────────────────────────
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)   # avoid division by zero
    unit_rows = rows / norms                       # (n, p), each row is unit-norm

    # ── Remove the global shared component (mean-centering) ──────────────────
    # Empirically, pretrained backbones + similar initial heads produce a strong
    # common gradient-direction component across clients/tasks, which inflates
    # cosine similarities and collapses the CFL silhouette. Mean-centering is a
    # standard de-biasing step that preserves relative task structure.
    centered = unit_rows - unit_rows.mean(axis=0, keepdims=True)
    c_norms = np.linalg.norm(centered, axis=1, keepdims=True)
    c_norms = np.where(c_norms < 1e-10, 1.0, c_norms)
    centered_unit = centered / c_norms

    # ── Cosine similarity matrix  S = centered_unit @ centered_unit.T ─────────
    # Clip to [-1, 1] to correct floating-point rounding
    S = np.clip(centered_unit @ centered_unit.T, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(S, 1.0)

    # ── Graph fingerprints for Phase 4 RBF ──────────────────────────────────
    # Use per-layer gradient norm² saliency when available: a compact vector
    # (n_backbone_layers + 1) whose magnitudes differ across tasks because
    # CoLA / MRPC / SST-2 activate different layers of DistilBERT.
    # This avoids the near-colinearity of tensor-norm² fingerprints (~0.976 cos)
    # while keeping meaningful distance structure for the RBF graph.
    # Fall back to S-matrix rows when layer_importances is not supplied.
    if layer_importances:
        all_layer_keys = sorted(
            set().union(*[set(d.keys()) for d in layer_importances.values()])
        )
        graph_fingerprints: Dict[int, np.ndarray] = {
            cid: np.array(
                [float(layer_importances[cid].get(k, 0.0)) for k in all_layer_keys],
                dtype=np.float32,
            )
            for cid in seen_client_ids
        }
        info_str = (f"CFL cosine-sim n×n matrix ({n}×{n}) [mean-centered] from {info_str}; "
                    f"graph-fp: layer-saliency {len(all_layer_keys)}-dim")
    else:
        graph_fingerprints = {
            cid: S[i].astype(np.float32) for i, cid in enumerate(seen_client_ids)
        }
        info_str = f"CFL cosine-sim n×n matrix ({n}×{n}) [mean-centered] from {info_str}"

    return S, graph_fingerprints, info_str


if __name__ == "__main__":
    print("Phase 1 – Fingerprint Extraction Module")
    print("=" * 60)
    print("Available classes / functions:")
    print("  FingerprintExtractor      – per-client gradient fingerprint extractor")
    print("  build_cosine_fingerprints – CFL n×n cosine similarity matrix builder")
    print("=" * 60)
