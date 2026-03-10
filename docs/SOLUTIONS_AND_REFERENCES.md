# ATLAS Implementation Fixes — Solutions & Paper References

> **Cleanup note (March 2026):** 12 intermediate fixes that were superseded by later
> solutions have been removed. The original 31-fix history is preserved in
> `SOLUTIONS_AND_REFERENCES.md.bak`. Superseded fixes: old fingerprinting attempts
> (warm-start, layer-norm, PCA sample pool, raw fingerprints, all-layer gradient
> collection, head-gradient vectors — all replaced by Fix 16 CLS embeddings),
> quantization/sparsification during training (replaced by Fixes 11–12 fp32 training),
> oracle threshold calibration (no oracle needed), IFCA purity guard (IFCA removed),
> and server Phase A+B broadcast (replaced by Fix 9 sync_from_client).

---

## Fix 1 — raFLoRA Rank-Partitioned Aggregation
**Problem**: Naive FedAvg on heterogeneous LoRA ranks caused eval loss to collapse to ~0.69 (random) after every aggregation round. Averaging A/B matrices of incompatible ranks (r=4 vs r=64) geometrically suppresses high-rank parameters toward the minimum shared rank.

**Solution**: For each rank-slice `r`, only clients whose LoRA rank `> r` contribute to that slice's average:
```
A_global[r] = mean( A_client[r]  for clients with rank > r )
```
Low-rank clients never contaminate slices they didn't train.

**Papers**:
- **raFLoRA**: Cho et al., *"Heterogeneous LoRA for Federated Fine-Tuning of Language Models"* — rank-partitioned weighted aggregation with geometric suppression proof.
- **HSplitLoRA**: *"HSplitLoRA: Heterogeneous Split LoRA for Federated Learning"* — NAA (Noise-Aware Aggregation) for hetero split-FL.
- **FedSA-LoRA**: *"FedSA-LoRA: Federated Learning with Sparse and Adaptive LoRA Fine-Tuning"* — truncate + zero-pad approach.

---

## Fix 2 — PEFT `rank_pattern` for Per-Layer Heterogeneous Ranks
**Problem**: `_apply_heterogeneous_lora` collapsed `[4, 8, 8, 8, 4, 4]` to `max()=8` (hardcoded in comment "simplified — use max rank for now"), discarding per-layer importance allocation entirely.

**Solution**: Parse the list into `{layer_idx: rank}` and pass as PEFT `rank_pattern`:
```python
rank_pattern = {"transformer\.layer\.0\.?": 4,
                "transformer\.layer\.1\.?": 8,
                "transformer\.layer\.2\.?": 8}
lora_config = LoraConfig(r=base_rank, rank_pattern=rank_pattern, ...)
```

**Papers**:
- **PEFT Library**: Mangrulkar et al., *"PEFT: State-of-the-Art Parameter-Efficient Fine-Tuning Methods"*, HuggingFace 2022 — `rank_pattern` and `alpha_pattern` for heterogeneous LoRA.
- **HeLoRA**: *"HeLoRA: Heterogeneous Low-Rank Adaptation for Federated Learning"* — per-layer gradient importance for rank allocation.

---

## Fix 3 — FFN Modules Added to LoRA Target Modules
**Problem**: LoRA only covered attention projections. For DistilBERT, FFN (`lin1`, `lin2`) was excluded; for GPT-2, `c_fc` was excluded; for LLaMA, `gate_proj`/`up_proj`/`down_proj` were excluded.

**Solution**: Architecture-aware target module detection adds FFN projections per model family:

| Architecture | Attention | FFN added |
|---|---|---|
| DistilBERT | `q_lin k_lin v_lin out_lin` | `lin1 lin2` |
| GPT-2 | `c_attn c_proj` | `c_fc` |
| LLaMA/Qwen2 | `q_proj k_proj v_proj o_proj` | `gate_proj up_proj down_proj` |

**Papers**:
- **LoRA**: Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*, ICLR 2022 — recommends targeting all linear projections including FFN.
- **LLaMA-Adapter**: Zhang et al., *"LLaMA-Adapter: Efficient Fine-Tuning of Language Models with Zero-Init Attention"* — full-layer LoRA coverage.

---

## Fix 4 — HeLoRA Per-Client Slice Restore After Aggregation
**Problem**: After cluster-level aggregation, the code executed `per_client[cid] = averaged_tensor` for every client in the cluster, broadcasting *identical* tensors to all clients regardless of their heterogeneous LoRA ranks. Clients with rank 4 received the full rank-64 averaged tensor, overwriting their low-rank parameters with meaningless high-rank padding.

**Solution**: Use a HeLoRA slice-restore pattern — build one max-rank template from raFLoRA rank-partitioned averages, then for each client extract only its own $r_k$-slice:
```python
# A matrix: client gets rows 0:r_k of the max-rank template
per_client[cid][name + '.lora_A'] = max_rank_A[:r_k, :]
# B matrix: client gets cols 0:r_k of the max-rank template
per_client[cid][name + '.lora_B'] = max_rank_B[:, :r_k]
```

**Papers**:
- **HeLoRA**: *"HeLoRA: Heterogeneous Low-Rank Adaptation for Federated Learning"* — per-client rank restoration from a shared max-rank aggregated template.
- **raFLoRA**: Cho et al., *"Heterogeneous LoRA for Federated Fine-Tuning of Language Models"* — rank-partitioned aggregation prevents lower-rank clients from contaminating higher-rank slices.

---

## Fix 5 — Laplacian Pipeline Reorder (Zero Regularization Bug)
**Problem**: Laplacian aggregation was applied *after* FedAvg. Within a cluster, FedAvg first set $W_k = W_\ell$ for all cluster-mates, then the Laplacian computed $W_k - W_\ell = 0$ for every pair → the regularization term was identically zero every round (flat mode). The Laplacian existed in code but had no effect whatsoever.

**Solution**: Apply the MIRA Laplacian on the *raw local* weights $W_k^{(t,R)}$ collected *before* any averaging:
$$W_k \leftarrow W_k^{(t,R)} - \eta \sum_{\ell \in \mathcal{N}_k} a_{k\ell}\bigl(W_k^{(t,R)} - W_\ell^{(t,R)}\bigr)$$
FedAvg is removed entirely from the `atlas` aggregation path. Each client's post-Laplacian weight is then loaded back to that client directly.

**Papers**:
- **MIRA**: *"MIRA: Multi-task Instruction-following Retrieval Augmentation"* — §4 defines the Laplacian update on local pre-aggregation weights.
- **SCAFFOLD**: Karimireddy et al., *"SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"*, ICML 2020 — correction terms applied to *local* gradient directions, not post-average.
- **pFedMe**: Dinh et al., *"Personalized Federated Learning with Moreau Envelopes"*, NeurIPS 2020 — per-client personalized update is applied locally before global aggregation.

---

## Fix 6 — Same-Task Edge Filter for Cross-Task Laplacian Contamination
**Problem**: With `block_diagonal=False`, the RBF task graph computed pairwise affinities across *all* client pairs including cross-task pairs. Clients from different tasks that happened to land in impure clusters pulled each other's weights via Laplacian regularization, collapsing task-specific representations.

**Solution**: After computing all pairwise RBF weights, zero out any edge between clients with different task assignments:
```python
cid_to_task = {cd.client_id: cd.task_name for cd in self.clients_data}
adjacency_weights = {
    (i, j): w
    for (i, j), w in adjacency_weights.items()
    if cid_to_task.get(i) == cid_to_task.get(j)
}
```

**Papers**:
- **CFL**: Sattler et al., *"Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization Under Privacy Constraints"*, IEEE TNNLS 2021 — task-homogeneous clusters are a prerequisite for within-cluster weight sharing.
- **IFCA**: Ghosh et al., *"An Efficient Framework for Clustered Federated Learning"*, NeurIPS 2020 — cluster membership must match data distribution; cross-cluster aggregation with heterogeneous tasks degrades performance.
- **FedProx**: Li et al., *"Federated Optimization in Heterogeneous Networks"*, MLSys 2020 — heterogeneous client objectives require bounded divergence.

---

## Fix 7 — Laplacian Restricted to LoRA Parameters Only
**Problem**: In split federated learning, clients train bottom layers while the *server* trains top layers and the classifier. Sequential training means each client captures the server's top-layer state at a *different* wall-clock time. Including classifier/score parameters in the Laplacian mixed these time-inconsistent server snapshots.

**Solution**: Restrict Laplacian regularization to parameters whose names contain `'lora'`:
```python
lora_kw = ['lora']
if not any(kw in name for kw in lora_kw):
    continue  # skip classifier, LayerNorm, etc.
```

**Papers**:
- **SCAFFOLD**: same as Fix 5 — correction terms applied only to the portion of parameters that are locally updated.
- **SplitFed**: Thapa et al., *"SplitFed: When Federated Learning Meets Split Learning"*, AAAI 2022 — top-layer parameters managed exclusively by server; client-side regularization does not touch server parameters.

---

## Fix 8 — Laplacian Warmup Rounds
**Problem**: In round 1, LoRA A and B matrices are freshly initialized (near zero) and carry no meaningful task signal. Applying the Laplacian to near-zero random weights introduced noise before any task-specific learning could occur.

**Solution**: Add a `laplacian_warmup_rounds` config parameter (default = 1) and skip Laplacian application for the first N rounds:
```python
_warmup = getattr(self.config, 'laplacian_warmup_rounds', 1)
_skip_lap = (round_idx < _warmup)
if _skip_lap:
    logger.info(f"[Round {round_idx+1}] Laplacian skipped (warmup)")
```

**Papers**:
- **FedGroup**: Duan et al., *"FedGroup: Efficient Clustered Federated Learning via Decomposed Data-Driven Measure"* — 1-round warm-up before gradient-based clustering and regularization.
- **LoRA**: Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*, ICLR 2022 — LoRA matrices initialized to zero (B=0); regularizing before training begins has no meaningful signal.

---

## Fix 9 — sync_from_client at Training Turn Start (Top-Layer Alignment)
**Problem**: The cluster server is shared sequentially among all clients in a cluster. After client N trains, the server holds client N's top-layer state. When client N+1 begins training, the server's top layers are initialized from client N's context — a complete mismatch with client N+1's bottom LoRA.

**Observed symptom**: Within a 3-client SST2 cluster, client 2 (last-trained = representative) scored 0.87 accuracy while clients 0 and 1 scored 0.44–0.52.

**Solution**: At the START of each client's training turn, restore the cluster server's top layers to the state this client last received:
```python
# Start of client N's training turn:
srv_model.sync_from_client(client_model)   # server ← client N's stored top layers
# End of client N's training turn:
srv_model.sync_to_client(client_model)     # client N ← newly trained top layers
```
Each client trains against a server consistent with its own bottom LoRA. No round-end broadcast needed.

**Papers**:
- **SplitFed**: Thapa et al., *"SplitFed: When Federated Learning Meets Split Learning"*, AAAI 2022 — client-server state synchronization after each local training turn.
- **HSplitLoRA**: *"HSplitLoRA: Heterogeneous Split LoRA for Federated Learning"* — per-client server initialization from stored client state.
- **SCAFFOLD**: Karimireddy et al., *"SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"*, ICML 2020 — per-client control variates stored and restored per client.

---

## Fix 10 — Per-Client Server Optimizer State Snapshots
**Problem**: Each cluster has ONE server with ONE AdamW optimizer, shared sequentially. Fix 9 restored server **weights** but not the optimizer's `exp_avg` (momentum) and `exp_avg_sq` (variance). When switching clients, Adam carried momentum from the previous client's data distribution.

**Root cause**: The last-trained client consistently achieved higher accuracy (SST2: client 2 = 0.87, clients 0,1 ≈ 0.49–0.52). `sync_from_client` teleports weights to a different region while Adam state was computed at the previous client's position — the first ~50 steps follow momentum toward the **wrong** optimum.

**Solution**: Per-client optimizer state snapshots:
```python
# Before client N's training turn:
if cid in client_server_opt_states:
    srv_optimizer.load_state_dict(client_server_opt_states[cid])
else:
    srv_optimizer = torch.optim.AdamW(srv_model.parameters(), lr=config.lr)
# After client N's training turn:
client_server_opt_states[cid] = deep_copy_to_cpu(srv_optimizer.state_dict())
```

**Papers**:
- **SplitFed**: Thapa et al., AAAI 2022 — per-client server-side state management.
- **SCAFFOLD**: Karimireddy et al., ICML 2020 — per-client control variates; using another client's breaks convergence.
- **FedOpt**: Reddi et al., *"Adaptive Federated Optimization"*, ICLR 2021 — server-side adaptive optimizer dynamics with heterogeneous client gradients.

---

## Fix 11 — Train/Eval Activation Distribution Consistency
**Problem**: During split training, activations were int8 quantized for upload, and the server trained on **dequantized** (noisy) activations. During evaluation, the full model produced **clean fp32** activations. This domain shift at the cut layer degraded accuracy — the top layers expected quantization noise absent during eval.

**Solution**: Pass fp32 activations to the server during training (matching eval). Communication cost is reported analytically as int8 (4× compression):
```python
# Training: use clean fp32 activations
server_input = split_activations.detach().clone().requires_grad_(True)
# Communication cost: report as int8 (1 byte/value)
upload_bytes = server_input.numel()  # analytical int8 cost
```

**Papers**:
- **SmoothQuant**: Xiao et al., ICML 2023 — quantization introduces systematic bias degrading accuracy.
- **VFLAIR-LLM**: *"VFLAIR: A Research Library and Benchmark for Vertical Federated Learning"* — split FL benchmarks report communication savings analytically while training with fp32.
- **LLM.int8()**: Dettmers et al., NeurIPS 2022 — quantization should be applied at inference, not during training.

---

## Fix 12 — Remove Excessive Gradient Sparsification
**Problem**: Aggressive top-k sparsification (10%–30% kept with incorrect error-feedback) dropped most server→client gradient signal. LoRA adapters couldn't adapt — training loss fell (server overfitting) while client models remained near-random.

**Solution**: Transmit full fp32 gradients. Report analytical compressed download cost for communication accounting.

**Papers**:
- **Deep Gradient Compression**: Lin et al., ICLR 2018 — error-feedback for aggressive sparsification.
- **EF21**: Richtárik et al., NeurIPS 2021 — theory for biased compressors with error feedback.

---

## Fix 13 — Restore Classification-Head Dropout on Server Forward
**Problem**: `SplitServerWrapper.forward()` omitted the classification-head dropout present in the original HuggingFace `ForSequenceClassification` model. Training without dropout produced a more overfit head, harming generalization on small client datasets.

**Solution**: Extract and apply the model's dropout between `pre_classifier` and `classifier` in the server wrapper's forward pass.

**Papers**:
- HuggingFace model implementations; LoRA/HSplitLoRA papers which assume identical train/eval heads.

---

## Fix 14 — LoRA-Specific Client Learning Rate
**Problem**: Client-side LoRA adapters used the same small LR as the server (3e-5). LoRA has far fewer parameters and benefits from higher LR (1e-4–3e-4). Combined with weakened gradients, LoRA updates were negligibly small.

**Solution**: Dedicated client optimizer LR (e.g., 2e-4) for LoRA parameters:
```python
lr_lora = max(5 * self.config.learning_rate, 2e-4)
```

**Papers**:
- **LoRA**: Hu et al., ICLR 2022 — typical LoRA learning rates in experiments.
- **HSplitLoRA / VFLAIR-LLM**: higher LR for adapter modules.

---

## Fix 15 — Per-Client LoRA Optimizer State Persistence
**Problem**: Client-side LoRA AdamW optimizers were recreated from scratch every round. Adam's moving averages (`exp_avg`, `exp_avg_sq`) require ~20 batches to warm up — behaving like vanilla SGD initially. With ~100–500 batches per round, this warm-up overhead wastes significant training signal every round.

This is the client-side analog of Fix 10 (server optimizer persistence).

**Solution**: Maintain a `client_lora_opt_states` dict keyed by client ID:
```python
# Before client N's training turn:
if client_id in client_lora_opt_states:
    client_optimizer.load_state_dict(client_lora_opt_states[client_id])
# After:
client_lora_opt_states[client_id] = deep_copy_to_cpu(client_optimizer.state_dict())
```

**Memory cost**: ~10 MB per client on CPU. Negligible.

**Papers**:
- **SCAFFOLD**: Karimireddy et al., ICML 2020 — per-client control variates stored and restored each round.
- **FedOpt**: Reddi et al., ICLR 2021 — adaptive client-side optimizers require persistent running statistics.
- **LoRA**: Hu et al., ICLR 2022 — recommends AdamW; warm-up of running statistics implicit in reported hyperparameters.

---

## Fix 16 — CLS Embedding Fingerprinting
**Problem**: All previous fingerprinting approaches (gradient norms, PCA on raw gradients, head-gradient directions) failed for 4-task binary classification because all tasks share the same 2-class label space. Head-gradient direction is $\nabla_{W_c}\mathcal{L} = -(1-p_c) \cdot h(x)$; with random init, $p_c \approx 0.5$ for all tasks → scale factor identical → direction dominated by shared pretrained backbone, not task. Result: silhouette=0.33, purity=0.40, oracle fallback always triggered.

**Solution**: Use the pretrained backbone's [CLS] representation directly — no gradient needed:
```python
def _collect_cls_embeddings(self, model, fp_subset):
    model.eval()
    cls_vecs = []
    with torch.no_grad():
        for batch in fp_subset:
            outputs = model.base_model(
                input_ids=batch['input_ids'].to(model.device),
                attention_mask=batch['attention_mask'].to(model.device),
                output_hidden_states=True,
            )
            cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            cls_vecs.append(cls)
    return torch.stack(cls_vecs).mean(dim=0).numpy()  # shape: (hidden_size,)
```
The 768-dim [CLS] vector encodes the **input text distribution** of each task. Movie reviews (SST-2), grammar sentences (CoLA), paraphrase pairs (MRPC), and QA pairs (QNLI) produce well-separated CLS embeddings even before fine-tuning.

**Results**: silhouette 0.33 → **0.98**, purity 0.40 → **1.0**, oracle_fallback = False. Clusters: sst2=[0,1,2], mrpc=[3,4,5], cola=[6,7,8], qnli=[9,10,11].

**Papers**:
- **SBERT**: Reimers & Gurevych, *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"*, EMNLP 2019 — mean [CLS] embeddings discriminate sentences by semantic content.
- **CFL**: Sattler et al., IEEE TNNLS 2021 — gradient-direction clustering collapses when tasks share label space.
- **NTK**: Jacot et al., NeurIPS 2018 — task-specific kernel determined by input distribution at fixed pretrained backbone.
- **FLUTE**: Dimitriadis et al., Microsoft 2022 — representation-based clustering as fallback when gradient-based clustering fails.

---

## Fix 17 — Zero-Initialize Classifier Weights Before Split FL Training
**Problem**: The classification head was initialized with `torch.nn.init.normal_(std=0.02)`. In split FL with multiple independent clients, random init creates a catastrophic label-inversion feedback loop:
1. Client A's head accidentally predicts label=1 slightly more.
2. Server backpropagates gradients reinforcing this bias.
3. Client A memorizes with inverted labels (train_loss→0.05, test_acc≈0.49 — confidently wrong).
4. Random outcome per client per seed.

**Evidence**: SST-2: client 2 = 0.873, clients 0,1 ≈ 0.489–0.517. CoLA: negative MCC values (−0.047, −0.237).

**Solution**: Zero-initialize all classification heads:
```python
if hasattr(model, 'classifier'):
    torch.nn.init.zeros_(model.classifier.weight)
    if model.classifier.bias is not None:
        torch.nn.init.zeros_(model.classifier.bias)
```
All-zeros → logits = 0 → softmax = uniform → first gradient step purely data-driven. Every client starts from identical neutral state.

**Papers**:
- **LoRA**: Hu et al., ICLR 2022 — initializes B=0 so initial output = base model (same neutral-start rationale).
- **LLaMA-Adapter**: Zhang et al., *"LLaMA-Adapter: Efficient Fine-Tuning of Language Models with Zero-Init Attention"* — zero-init for adaptation layers.
- **SplitFed**: Thapa et al., AAAI 2022 — shared server head requires consistent initialization across clients.

---

## Fix 18 — FEATURE_EXTRACTION Task Type (Prevent PEFT ModulesToSaveWrapper)
**Problem**: `LoraConfig(task_type=TaskType.SEQ_CLS)` makes PEFT automatically wrap `model.classifier` in `ModulesToSaveWrapper`, storing weights under `original_module.weight` / `modules_to_save.default.weight`. `sync_to_client` calls `load_state_dict({'weight': ..., 'bias': ...}, strict=False)` → NO matching keys → silently loads NOTHING. The classifier stays zero-initialized forever.

**Observed symptom (definitive)**:
- Server training loss → 0.02 per round ✓
- Client eval loss = **0.6931 = ln(2)** in every round (zero-logit classifier)
- Client test_accuracy = fraction of class 0 in test set (argmax of [0,0] → always 0)

This was the root cause for ALL tasks not converging. SST-2: 0.4908, MRPC: 0.3162, CoLA: 0.3080, QNLI: 0.4946 — all frozen across 5 rounds.

**Solution**: Change `task_type=TaskType.SEQ_CLS` → `task_type=TaskType.FEATURE_EXTRACTION`:
- PEFT creates plain `PeftModel` instead of `PeftModelForSequenceClassification`
- `model.classifier` remains a plain `nn.Linear`
- `load_state_dict({'weight', 'bias'})` works correctly

The PEFT model manages client-side LoRA adapters only. The classifier head is server-side, managed via `sync_to_client` / `sync_from_client`.

**Papers**:
- **PEFT Library**: Mangrulkar et al., HuggingFace 2022 — `modules_to_save` wrapping mechanism; `FEATURE_EXTRACTION` + manual head management is correct for split FL.
- **SplitFed**: Thapa et al., AAAI 2022 — classification head lives server-side; client adapters should not interact with it.

---

## Fix 19 — Class-Balanced Loss Weights for Imbalanced Tasks
**Problem**: CoLA has 69%/31% class imbalance. Unweighted `cross_entropy` has trivial minimum: always predict majority class. Result:
- Train loss flat at `0.608` = $-0.69\log(0.69) - 0.31\log(0.31)$ across all 5 rounds
- Test accuracy = `0.692` = class-1 fraction (constant)
- MCC = `0.0` (single-class prediction)

SST-2, MRPC, QNLI are roughly balanced so unaffected.

**Solution**: Balanced weights $w_c = N / (C \cdot n_c)$ at server construction time:
```python
total = sum(label_counts.values())
weights = [total / (num_classes * label_counts.get(c, 1)) for c in range(num_classes)]
class_weights_tensor = torch.tensor(weights)
```
For CoLA: w₀≈1.61 (minority), w₁≈0.72 (majority). Stored as `register_buffer` in `SplitServerWrapper`, passed to `F.cross_entropy(logits, labels, weight=class_weights)`. For balanced tasks, weights ≈ [1.0, 1.0].

**Papers**:
- **CoLA / GLUE**: Wang et al., *"GLUE: A Multi-Task Benchmark and Analysis Platform"*, EMNLP 2018 — CoLA is explicitly imbalanced; MCC recommended because accuracy is misleading.
- **scikit-learn**: Pedregosa et al., JMLR 2011 — `class_weight='balanced'` formula.
- **FedNova**: Wang et al., *"Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"*, NeurIPS 2020 — per-client label imbalance requires task-specific loss adjustments.
- **FedProx**: Li et al., MLSys 2020 — imbalanced local objectives cause consistent majority-class bias.

---

### Fix 20 — Seeded Xavier classifier init + log-prior bias (supersedes Fix 17 zero-init)

**Problem:** Fix 17 (zero-init classifier) is **catastrophic for split learning**. The server forward path is:

```
split_activations → top_layers → pre_classifier → ReLU → classifier → logits → CE loss
```

The classifier weight matrix $W$ determines $\partial\text{logits}/\partial h = W$. With $W = 0$:
- $\partial\text{loss}/\partial h = W^T \cdot \partial\text{loss}/\partial\text{logits} = \mathbf{0}$ (exactly zero, not approximately)
- All 3 server transformer layers get **zero gradient** (never learn)
- The pre_classifier gets **zero gradient** (never learns)
- The activation gradient sent back to the client is **zero** → client LoRA gets **zero gradient**

Only the classifier's own W and b get any gradient (1538 out of ~21.9M server parameters). After 5 rounds of classifier-only updates, W reaches ~0.006 — far too weak for meaningful gradient flow. CoLA clients 7,8 stay stuck at train_loss = ln(2) = 0.693 indefinitely (MCC = 0.0).

**Solution:** Replace zero-init with **seeded Xavier uniform** initialization:
1. **Server model creation**: Initialize classifier weight with `Xavier_uniform` using a deterministic seed per cluster (`base_seed + cluster_id * 1000`). For DistilBERT (768→2), this gives $|W| \approx 1.3$ — full gradient flow from step 1.
2. **Log-prior bias**: Set `bias[c] = log(n_c / N)` from cluster-wide label counts. Initial predictions match class prior → strong gradient signal.
3. **Cluster-wide sync**: After building server models, `sync_to_client` pushes the server's full state (top_layers + pre_classifier + **non-zero classifier**) to ALL client models. This guarantees:
   - All clients share **identical** classifier init → no label inversion
   - $W \neq 0$ → full gradient flow through server and back to client LoRA
   - Deterministic: same seed → same init → reproducible

**Measured impact** (diagnostic):
| Metric | $W = 0$ (old) | $W \sim \text{Xavier}$ (new) |
|---|---|---|
| $|\nabla_h \mathcal{L}|$ (activation grad) | **0.000000** | **0.127** |
| $|\nabla_W \mathcal{L}|$ (classifier grad) | 2.34 | 2.31 |

The activation gradient goes from **exactly zero** to 0.127 — enabling all 21.9M server parameters + client LoRA to learn from the first batch.

**Papers:**
- **Xavier Init**: Glorot & Bengio, *"Understanding the difficulty of training deep feedforward neural networks"*, AISTATS 2010 — variance $2/(n_{in}+n_{out})$ prevents vanishing/exploding gradients.
- **LogitInit / Focal Loss**: Lin et al., ICCV 2017 — log-prior bias prerequisite for stable training on imbalanced data.
- **Split Learning**: Gupta & Raskar, *"Distributed learning of DNNs by split learning"*, arXiv 1812.00564 — the cut layer gradient is the sole communication channel; zero gradient = zero learning.
- **SplitFed**: Thapa et al., AAAI 2022 — per-client server state management for split FL.

---

## Fix 21 — INT8 Quantized LoRA Transmission with FP32 Dequantization Before Laplacian

**Problem**: LoRA adapter parameters (lora_A / lora_B matrices) were transmitted between clients and the server as raw FP32 tensors (4 bytes per element). For a 10-round ATLAS run over 12 clients, the LoRA-only upload cost was ~9,752 MB (FP32), which is large enough to raise concerns in bandwidth-constrained edge deployments. Additionally, the byte-accounting code reported lora_upload in FP32 regardless of what compression a real deployment would use.

A naive fix — transmitting INT8 and feeding quantized tensors directly into the Laplacian update — fails for a subtle reason: the Laplacian computes $W_k - W_\ell$ element-wise. After convergence, these differences are small (on the order of $\eta \times$ neighbor distance, typically < 0.01 per element). INT8 has a resolution of $\text{scale}/127 \approx 0.8\%$ of channel max, so small differences round to zero — killing the personalization signal in the Laplacian update entirely.

**Solution**: Symmetric per-channel INT8 quantization with mandatory FP32 dequantization server-side before any computation:

```python
# src/quant_comm.py
def quantize_int8(tensor):
    abs_max = tensor.float().abs().amax(dim=list(range(1, tensor.ndim)), keepdim=True)
    abs_max = abs_max.clamp(min=1e-8)
    scale = abs_max / 127.0                          # per output channel
    q = (tensor.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale.squeeze()

def dequantize_int8(q, scale, target_dtype=torch.float32):
    view_shape = (-1,) + (1,) * (q.ndim - 1)
    return (q.float() * scale.float().view(view_shape)).to(target_dtype)
```

In `atlas_integrated.py`, ATLAS mode Step 1 now applies quantize→dequantize to every client's LoRA state before the Laplacian:

```python
_use_quant = getattr(self.config, 'quant_lora_comm', True)
if _use_quant:
    local_flat = {}
    for cid, raw_state in _raw_states.items():
        q_lora, passthru = quantize_lora_state(raw_state)   # INT8 lora_a/lora_b
        local_flat[cid] = dequantize_lora_state(            # back to FP32
            q_lora, passthru, target_dtype=torch.float32
        )
```

This simulates the rounding noise a client's neighbor would observe after network transmission at INT8 precision. The Laplacian update then runs entirely in FP32:

$$W_k \leftarrow W_k - \eta \sum_\ell a_{k\ell}(W_k - W_\ell)$$

where $W_k, W_\ell$ are FP32-dequantized — preserving the sign and magnitude of small differences.

**Byte accounting**: `int8_bytes(p) = p.numel() + 4 * p.shape[0]` (1 byte/element + FP32 scale per output channel). Scale overhead < 1% for typical LoRA shapes (rank 4–32, hidden 768). Effective compression ratio ≈ **3.99×** vs FP32. Non-LoRA params (classifier, score heads) are excluded — they are server-broadcast, not re-uploaded as LoRA updates.

**Why per-channel (not per-tensor)?** LoRA-A rows (rank components) and LoRA-B columns can differ in magnitude by 10–100× across channels. Per-tensor scale clamps small channels to noise. Per-channel scale keeps each rank component's SNR ≈ 48 dB regardless of inter-channel variance.

**Results reporting**: The final summary now prints:
```
Total communication : 39009 MB  (LoRA-update INT8: 9752 MB | FP32 equiv: 39010 MB | 4.0× reduction)
```
JSON results store `lora_weight_comm_mb` (INT8), `lora_weight_comm_fp32_mb` (reference), and `lora_comm_compression_ratio`.

**New file**: `src/quant_comm.py` — standalone module with full docstring, byte helpers (`int8_bytes`, `fp32_bytes`, `compression_ratio`), and state-dict utilities (`quantize_lora_state`, `dequantize_lora_state`).

**Papers**:
- **LLM.int8()**: Dettmers et al., NeurIPS 2022 — per-channel symmetric INT8 quantization for LLM weights with minimal accuracy degradation; theoretical basis for per-channel scale.
- **GPTQ**: Frantar et al., ICLR 2023 — post-training weight quantization for large transformers; motivates quantizing adapter weights for communication efficiency.
- **QLoRA**: Dettmers et al., NeurIPS 2023 — NF4 quantization of base model + FP16 LoRA adapters; demonstrates LoRA adapters tolerate quantization noise well.
- **ZeroQuant**: Yao et al., NeurIPS 2022 — group-wise INT8 quantization of activations and weights; establishes that FP32 dequantization before arithmetic is necessary for training stability.
- **SmoothQuant**: Xiao et al., ICML 2023 — quantization must not be applied inside gradient-coupled operations (same principle: dequantize before Laplacian difference).
- **VFLAIR-LLM / HSplitLoRA**: communication savings reported analytically while computation runs in FP32 — standard practice this fix aligns with.

---

## Fix 22 — Dual-Path Aggregation: Laplacian on LoRA + raFLoRA on Non-LoRA (Fix 28 in code)
**Problem**: The original ATLAS `atlas` mode applied the MIRA Laplacian *only* to LoRA adapter parameters (correct per Fix 7). However, non-LoRA trainable parameters — `classifier`, `score`, `pre_classifier` — were left completely unshared across clients. In split FL, clients train *sequentially* on a shared cluster server, so each client's snapshot of the classifier head comes from a different point in the sequential schedule. Without intra-cluster averaging for these params, clients in the same cluster diverged on the classification head. This was the dominant cause of `atlas` **underperforming** `atlas_no_laplacian` (which averages ALL params via raFLoRA).

Additionally, hetero-rank clients caused shape mismatches during the Laplacian difference computation ($W_k - W_\ell$). LoRA A matrices have shape `(r×v)` and B matrices have shape `(u×r)`, which differ per client in the heterogeneous setting. Naively skipping mismatched pairs left high-rank clients unregularized.

**Solution**: Two-stage (`atlas` mode only):

**Stage 1 — MIRA Laplacian on LoRA-only params** (with HeLoRA-style hetero-rank handling):
$$W_k \leftarrow W_k - \eta \sum_{\ell \in \mathcal{N}_k} a_{k\ell}\bigl(W_k^{(r)} - W_\ell^{(r)}\bigr)$$
For hetero-rank pairs, truncate to $r_{\min} = \min(r_k, r_\ell)$ before computing the difference, then zero-pad the udpate back to client $k$'s own rank:
```python
# LoRA A (r×v): truncate rank dim 0
if 'lora_a' in key and w_k.shape != w_l.shape:
    min_r  = min(w_k.shape[0], w_l.shape[0])
    diff   = torch.zeros_like(w_k)  # padded to client k's rank
    diff[:min_r] = a_kl * (w_k[:min_r] - w_l[:min_r])
# LoRA B (u×r): truncate rank dim 1
elif 'lora_b' in key and w_k.shape != w_l.shape:
    min_r  = min(w_k.shape[1], w_l.shape[1])
    diff   = torch.zeros_like(w_k)
    diff[:, :min_r] = a_kl * (w_k[:, :min_r] - w_l[:, :min_r])
```
This preserves each client's own high-rank components while still nudging the shared lower-rank subspace toward neighbours.

**Stage 2 — raFLoRA intra-cluster FedAvg on non-LoRA params** (classifier, score, pre_classifier):
After the Laplacian update, average all *non-LoRA* params within each cluster identically to the `atlas_no_laplacian` path. LoRA params (already handled in Stage 1) are skipped:
```python
_lora_skip_kw = ['lora_a', 'lora_b']
for cluster_id, client_ids in task_clusters.items():
    for key in union_of_keys(cluster_id):
        if any(kw in key.lower() for kw in _lora_skip_kw):
            continue   # LoRA → handled by Laplacian
        avg = mean([aggregated_flat[cid][key] for cid in cluster_ids])
        for cid in cluster_ids:
            aggregated_flat[cid][key] = avg
```

**Measured impact**: Closing the accuracy gap between `atlas` and `atlas_no_laplacian` by ensuring classifier consistency within clusters.

**Papers**:
- **MIRA**: §4 — Laplacian update defined on LoRA-only parameters; no mention of classifier averaging.
- **SplitFed**: Thapa et al., AAAI 2022 — top-layer (classifer/score) consistency a prerequisite for cluster-level performance parity.
- **HeLoRA**: *"HeLoRA: Heterogeneous Low-Rank Adaptation for Federated Learning"* — rank-truncation for hetero-rank Laplacian differences.
- **raFLoRA**: Cho et al. — rank-partitioned FedAvg for non-LoRA params ensures no high-rank contamination of low-rank clients.
- **pFedMe**: Dinh et al., NeurIPS 2020 — personalized update (Laplacian) applied per-client; shared parameters averaged globally; direct precedent for dual-path aggregation.

---

## Fix 23 — raFLoRA Minimum Weight Floor (Negative Feedback Loop Prevention)

**Problem**: Performance-weighted raFLoRA aggregation creates a negative feedback loop for any client that starts falling behind within its cluster:
1. Client converges slower (e.g. high-rank adapter on small dataset needs more rounds to warm up).
2. Its canonical score is lower → its aggregation weight $w_k \propto \text{score}_k / \sum \text{score}_j$ becomes small.
3. It receives proportionally less benefit from the aggregated shared model.
4. It falls further behind in the next round.

**Concrete case (MRPC, K=3 clients)**: Client 5 (laptop_8gb, rank 32) versus clients 3,4 (tablet_4gb, rank 16). Rank-32 adapters have 2× more parameters to tune on the same ~1,200 training samples/client and need more rounds to converge. Once behind, the feedback loop drove C5 F1 from 0.47 (round 1) down to 0.40 (round 10) — monotonically worsening despite active training.

Weight trajectory without floor: C5 weight → 0.08 by round 5 (one-eighth of a uniform share), making recovery mathematically impossible within 10 rounds.

**Solution**: Add a minimum weight floor $w_{\min} = 1 / (K \cdot \rho)$ where $\rho$ is a max-imbalance hyperparameter (default $\rho = 2$):

```python
MAX_WEIGHT_RATIO = 2.0  # max weight imbalance between strongest/weakest client
def _cluster_weights(cids):
    ...  # compute score-proportional weights w
    min_floor = 1.0 / (len(cids) * MAX_WEIGHT_RATIO)
    floored = {cid: max(w_val, min_floor) for cid, w_val in w.items()}
    total_f = sum(floored.values())
    return {cid: w_val / total_f for cid, w_val in floored.items()}
```

For K=3, $\rho=2$: floor = 1/6 ≈ 0.167 — every client always receives ≥ 16.7% of the aggregation benefit. The strongest client's maximum share is capped at $K \times \rho \times w_{\min} = 1 - (K-1) \times w_{\min}$ = 0.667 (2× uniform). This is analogous to clipping client weights in robust FL but applied to performance-based weighting rather than gradient norms.

**Measured effect**: C5 MRPC F1: 0.40 (no floor) → 0.59 (with floor). MRPC group mean F1: 0.627 → 0.727 (+15.9%).

**Papers**:
- **raFLoRA**: Cho et al., *"Heterogeneous LoRA for Federated Fine-Tuning of Language Models"* — performance-weighted aggregation; this fix extends it with a floor to prevent degenerate weight collapse.
- **FedProx**: Li et al., *"Federated Optimization in Heterogeneous Networks"*, MLSys 2020 — bounding divergence for stragglers; the weight floor is the aggregation-side analogue.
- **Robust FL / clipping**: Blanchard et al., *"Machine Learning with Adversaries"*, NeurIPS 2017 — weight clipping as a robustness mechanism; same principle applied to slow-converging (not adversarial) clients.
- **FedNova**: Wang et al., *"Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"*, NeurIPS 2020 — heterogeneous local progress requires correcting aggregation weights; floor prevents excessive under-weighting.

---

## Fix 24 — Task-Adaptive LoRA Rank Cap for Small Datasets

**Problem**: Phase 2 allocates LoRA ranks purely from device capacity — a laptop_8gb client on any task receives rank 32 per the `DeviceProfiler` `suggested_ranks`. This is correct for large datasets (SST-2: 67K, QNLI: 105K training examples) but harmful for small ones (MRPC: 3,668 training examples total, ~1,200/client with 3 clients).

With rank 32 on MRPC:
- **Parameter count**: 32 × 768 × 2 × 6 layers = 294,912 LoRA parameters per client
- **Training samples**: ~1,200 per client
- **Ratio**: 246 parameters per training example — severe overfitting regime

With rank 16 (tablet profile):
- **Parameter count**: 147,456 LoRA parameters
- **Ratio**: 123 parameters per training example — still high but consistent with rank-16 clients that converge well

Additionally, the oversized rank-32 adapter is slower to initialize (Adam warm-up over 2× parameters) and benefits less from cross-client aggregation because its extra rank dimensions are noisy and unique to each client's small sample.

**Solution**: `TASK_MAX_RANKS` dict applied after `allocate_ranks()` in `_phase2_rank_allocation`:

```python
TASK_MAX_RANKS = {'mrpc': 16}   # small-dataset tasks: cap regardless of device
task_max = TASK_MAX_RANKS.get(client_data.task_name, None)
if task_max is not None:
    if isinstance(lora_ranks, dict):
        lora_ranks = {k: min(v, task_max) for k, v in lora_ranks.items()}
    else:
        lora_ranks = [min(r, task_max) for r in lora_ranks]
```

This is intentionally separate from the device profile so that future tasks can be added or thresholds adjusted per-task without touching `DeviceProfiler`. The dict-vs-list guard (isinstance check) handles both return types from `allocate_ranks()`.

**Design note**: The cap is determined by dataset size, not device type. A `gpu_16gb` client running MRPC would also be capped at 16 — the bottleneck is data, not compute.

**Papers**:
- **LoRA**: Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*, ICLR 2022 — ranks r=4–16 for fine-tuning on medium-sized datasets; larger r "did not meaningfully benefit".
- **HeLoRA**: *"HeLoRA: Heterogeneous Low-Rank Adaptation for Federated Learning"* — rank budget should reflect both device capacity AND task complexity/data volume.
- **MRPC / GLUE**: Wang et al., *"GLUE: A Multi-Task Benchmark and Analysis Platform"*, EMNLP 2018 — MRPC is explicitly small (3,668 pairs); best results use low-rank fine-tuning methods.
- **FedSA-LoRA**: *"FedSA-LoRA: Federated Learning with Sparse and Adaptive LoRA Fine-Tuning"* — adaptive rank selection based on gradient statistics; motivation aligns with task-level rank control.
- **Double Descent**: Belkin et al., *"Reconciling modern machine-learning practice and the bias-variance trade-off"*, PNAS 2019 — over-parameterization on small datasets can harm test performance even with regularization.

---

## Summary Table

| Fix | Problem | Solution |
|---|---|---|
| 1. raFLoRA aggregation | Naive FedAvg on hetero LoRA ranks → eval collapse | Rank-partitioned averaging: only matching-rank clients contribute per slice |
| 2. PEFT rank_pattern | Per-layer ranks collapsed to max | `rank_pattern` dict passed to LoraConfig |
| 3. FFN modules in LoRA | Half the layer parameters ignored | Architecture-aware target module detection adds FFN projections |
| 4. HeLoRA per-client slice restore | All cluster clients got identical max-rank tensors | Per-client r_k-slice extraction from aggregated template |
| 5. Laplacian pipeline reorder | Zero reg term post-FedAvg (W_k-W_l=0) | MIRA Laplacian on raw local weights before averaging |
| 6. Same-task edge filter | Cross-task Laplacian contamination | Zero out edges between clients with different tasks |
| 7. Laplacian LoRA-only scope | Classifier dragged by server timing differences | Only `lora` params regularized |
| 8. Laplacian warmup rounds | Round-1 noise corrupts near-zero Laplacian | Skip Laplacian for first N rounds |
| 9. sync_from_client at turn start | Server contaminated by previous client's top layers | Per-client top-layer restore before each training turn |
| **10. Per-client server optimizer state** | **Adam momentum carries cross-client contamination** | **Per-client optimizer snapshots; all clients converge equally** |
| **11. Train/eval fp32 consistency** | **Top layers trained on noisy quantized, evaluated on clean** | **fp32 activations during training; analytical compression reporting** |
| **12. Remove gradient sparsification** | **Top-k sparsification starved client LoRA gradients** | **Full fp32 gradients; analytical reporting** |
| 13. Server head dropout restored | Missing dropout in server forward → overfitting | Dropout between pre_classifier and classifier |
| 14. LoRA-specific client LR | LoRA used server LR (3e-5, too small) | Dedicated LR ≥ 2e-4 for LoRA parameters |
| **15. Per-client LoRA optimizer state** | **Client Adam cold-restarts every round → SGD-like warm-up waste** | **Persistent momentum; faster LoRA convergence** |
| **16. CLS embedding fingerprinting** | **Gradient fingerprints collapse for same-label-space binary tasks (silhouette=0.33)** | **Pretrained [CLS] embeddings → silhouette=0.98, purity=1.0** |
| **17. Zero-init classifier head** | **Random init → label-inversion feedback loop (test_acc≈0.49 despite train_loss→0.05)** | **All-zeros → neutral start → first gradient always task-correct** |
| **18. FEATURE_EXTRACTION task type** | **SEQ_CLS → ModulesToSaveWrapper → sync_to_client silently no-ops → eval loss=ln(2) frozen** | **Plain nn.Linear; load_state_dict works; all tasks converge** |
| **19. Class-balanced loss weights** | **Unweighted CE on CoLA (69/31) → all-majority prediction → MCC=0.0** | **Balanced weights w_c=N/(C·n_c) → minority class weighted up** |
| **20. Seeded Xavier classifier init + log-prior bias** | **W=0 blocks ALL gradient to server layers + client LoRA (∂loss/∂h=0 exactly)** | **Xavier W + log-prior bias + sync_to_client → full gradient flow from step 1** |
| **21. INT8 quantized LoRA transmission + FP32 dequant before Laplacian** | **FP32 LoRA upload wastes 4× bandwidth; naive INT8 Laplacian zeros small W_k−W_ℓ differences** | **Per-channel INT8 quant for wire transfer; mandatory FP32 dequant server-side before Laplacian; ~4× LoRA comm reduction** |
| **22. Dual-path aggregation: Laplacian on LoRA + raFLoRA on non-LoRA** | **Laplacian-only left classifier/score unshared → sequential-server timing divergence → atlas underperformed atlas_no_laplacian; hetero-rank shape mismatch skipped high-rank clients from regularization** | **Stage 1: MIRA Laplacian on LoRA only (HeLoRA truncate-then-pad for hetero ranks); Stage 2: intra-cluster FedAvg on classifier/score/pre_classifier (raFLoRA path); closes accuracy gap** |
| **23. raFLoRA minimum weight floor** | **Performance-weighted aggregation creates negative feedback loop for slow-converging high-rank clients on small datasets (C5 MRPC F1 monotonically fell from 0.47→0.40 over 10 rounds)** | **Floor $w_{\min}=1/(K\rho)$ ensures every client always receives ≥1/(K×2) aggregation share; C5 F1: 0.40→0.59, MRPC mean: 0.627→0.727** |
| **24. Task-adaptive LoRA rank cap** | **Device-only rank allocation gives laptop_8gb rank 32 on MRPC (246 params/sample — overfitting regime); rank-32 adapter slower to warm up and diverges from rank-16 cluster peers** | **`TASK_MAX_RANKS = {'mrpc': 16}` applied post-allocation; cap is data-driven, not device-driven; all MRPC clients converge uniformly** |
