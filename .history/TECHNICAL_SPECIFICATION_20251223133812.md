# ATLAS PROJECT: TECHNICAL SPECIFICATIONS & IMPLEMENTATION GUIDE

## 1. PHASE-BY-PHASE TECHNICAL REQUIREMENTS

### PHASE 1: TASK DISCOVERY AND CLUSTERING

#### Input

- N clients, each with local dataset D_i
- Pretrained LLM model (GPT-2 or LLaMA)
- Hyperparameters: fingerprint dim = 64, num clusters k ∈ {3, 5, 10}

#### Process

**Step 1.1: Gradient Computation**

```python
# For each client i:
for batch in local_dataset_D_i:
    x, y = batch
    loss = model(x, y)
    gradients = ∇_W loss
    # Extract from selected layers (e.g., first 3 attention layers)
```

**Step 1.2: Fingerprint Generation (Dimensionality Reduction)**

```python
# PCA-based compression
from sklearn.decomposition import PCA

all_gradients = [grad_1, grad_2, ..., grad_N]
pca = PCA(n_components=64)
fingerprints = pca.fit_transform(all_gradients)
# Output: fingerprints shape (N, 64)
```

**Step 1.3: Clustering**

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100)
cluster_assignments = kmeans.fit_predict(fingerprints)
# Output: cluster_assignments shape (N,) - each element ∈ {0, 1, ..., k-1}
```

**Step 1.4: Task Similarity Graph**

```python
# Construct adjacency matrix
A[i][j] = 1 if cluster(i) == cluster(j) else 0
# Or weighted: A[i][j] = exp(-distance(fingerprint_i, fingerprint_j)^2 / sigma^2)
```

#### Output

- Cluster assignments: C ∈ ℝ^N (each element = cluster_id)
- Task similarity graph: A ∈ ℝ^{N×N}
- Cluster centroids: μ_k ∈ ℝ^64 for k ∈ {1, ..., K}

#### Validation Metrics

- **Silhouette Score:** Higher = better cluster separation (target > 0.4)
- **Davies-Bouldin Index:** Lower = better cluster compactness
- **Within-cluster consistency:** Low variance of fingerprints within clusters

---

### PHASE 2: HETEROGENEOUS CONFIGURATION ASSIGNMENT

#### Input

- Cluster assignments C from Phase 1
- Device properties: memory budgets {M_1, M_2, ..., M_N}
- Model layer information: layer sizes, parameter counts
- Hyperparameters: rank budget per device

#### Process

**Step 2.1: Weight Importance Scoring**

Method A: Gradient Magnitude (Simple)

```python
# For each weight matrix W_l:
importance[l] = mean(|∇W_l|) or max(|∇W_l|)
```

Method B: Hessian-based (More Principled)

```python
# For each weight w_ij:
importance[i][j] = |hessian[i][j] * w[i][j]|
# Hessian = d²loss/dw² (expensive to compute)
```

Method C: Layer-wise Contribution

```python
# Ablation: remove layer, measure accuracy drop
importance[l] = accuracy_drop(remove layer l)
```

**Recommendation:** Use Method A (gradient magnitude) for ATLAS Phase 2

```python
import torch

importance_scores = {}
for layer_name, layer_module in model.named_modules():
    if isinstance(layer_module, torch.nn.Linear):
        importance_scores[layer_name] = torch.abs(layer_module.weight.grad).mean()
# Sort layers by importance
sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
```

**Step 2.2: Rank Assignment Per Device**

```python
# For each device i in cluster c:
memory_budget_i = M_i
total_model_size = sum(layer.parameters())

# Determine max total LoRA rank
max_total_rank_i = int((memory_budget_i / total_model_size) * max_rank)

# Distribute ranks proportional to importance
ranks_i = {}
for layer, importance in sorted_layers:
    rank_fraction = importance / sum(importance_scores.values())
    rank_i[layer] = max(1, int(max_total_rank_i * rank_fraction))

# Ensure within memory constraint
total_rank = sum(ranks_i.values())
if total_rank > max_total_rank_i:
    # Scale down all ranks
    scale_factor = max_total_rank_i / total_rank
    ranks_i = {l: max(1, int(r * scale_factor)) for l, r in ranks_i.items()}
```

**Step 2.3: Split Point Selection**

```python
# For each device i with memory M_i:
# Find deepest split point respecting memory

total_model_size = sum(layer.numel() for layer in model.parameters())

for split_point in range(num_layers, 0, -1):  # Scan from deep to shallow
    client_layers = model.layers[:split_point]
    server_layers = model.layers[split_point:]

    client_memory = sum(p.numel() for p in client_layers.parameters())
    + sum(rank_i[l] * layer_dim for l in client_layers)  # LoRA overhead

    if client_memory <= M_i * 0.9:  # 90% budget utilization
        split_point_i = split_point
        break
```

#### Output

- Rank configuration: ranks_i ∈ ℤ^num_layers for each device i
- Split points: split_point_i ∈ {1, ..., num_layers} for each device i
- Memory utilization report

#### Validation

- Total memory per device ≤ budget
- Ranks > 0 for important layers
- Similar rank patterns within same cluster (verify via cosine similarity)

---

### PHASE 3: SPLIT LEARNING TRAINING LOOP

#### Input

- Cluster assignments, rank configurations from Phase 1-2
- Pretrained LLM model
- Local dataset D_i for each client
- Hyperparameters: learning rate η, num rounds T, local steps L

#### Architecture

```
Client i:                           Server:
┌─────────────────┐
│ Embedding       │
│ + LoRA adapt    │
│                 │
│ Layers 1..s_i   │
└────────┬────────┘
         │ h_i (intermediate)
         ├──────────────────────────→
                                   ┌──────────────────────┐
                                   │ Layers (s_i+1)..n    │
                                   │ + Task Head          │
                                   │ + LoRA adapters      │
                                   │                      │
                                   │ Compute loss         │
                                   │ Backward pass        │
                                   └────────┬─────────────┘
         ← ∇h_i (gradients) ────────────────┘
         │
         │ Update local LoRA adapters
         └─────────────────┘
```

#### Algorithm

**Client-Side (per round t, local steps l=1..L):**

```python
def client_forward_backward(x_batch, y_batch):
    # Forward pass (head only)
    h = client_head(x_batch)  # Intermediate activation

    return h

def client_backward(grad_h, optimizer):
    # Backward pass (head only)
    loss_client = local_loss_fn(grad_h)  # Head reconstruction
    loss_client.backward()

    # Update LoRA adapters on client
    optimizer.step()
    optimizer.zero_grad()

# Main loop
for round_t in range(num_rounds):
    for local_step_l in range(L):
        for x_batch, y_batch in local_dataloader:
            # Forward pass
            h = client_forward_backward(x_batch, y_batch)

            # Send h to server, receive grad_h
            grad_h = send_to_server(h)  # Blocks until server responds

            # Backward pass
            client_backward(grad_h, local_optimizer)

    # Aggregation (every I rounds, handled separately)
```

**Server-Side (per intermediate h received):**

```python
def server_forward_backward(h, y_batch):
    # Forward pass (tail only)
    logits = server_tail(h)
    loss = loss_fn(logits, y_batch)

    # Backward pass
    grad_h = torch.autograd.grad(loss, h, create_graph=True)[0]

    return grad_h

# Main loop
for round_t in range(num_rounds):
    for x_batch, y_batch in aggregated_dataloader:
        # Receive h from client
        h = receive_from_client(client_id)

        # Forward/backward pass
        grad_h = server_forward_backward(h, y_batch)

        # Send grad_h back to client
        send_to_client(grad_h, client_id)

        # Update server LoRA adapters
        server_optimizer.step()
        server_optimizer.zero_grad()
```

#### LoRA Implementation Details

**LoRA Adapter:**

```python
class LoRAAdapter(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = torch.nn.Linear(in_dim, rank, bias=False)
        self.B = torch.nn.Linear(rank, out_dim, bias=False)
        self.rank = rank

    def forward(self, x):
        return self.B(self.A(x))

class LoRALinear(torch.nn.Module):
    def __init__(self, original_linear, rank):
        super().__init__()
        self.original = original_linear
        self.lora = LoRAAdapter(
            original_linear.in_features,
            original_linear.out_features,
            rank
        )

    def forward(self, x):
        # Standard forward: W @ x
        # LoRA forward: W @ x + (B @ A @ x)
        return self.original(x) + self.lora(x)

# Usage:
model.attention.q_proj = LoRALinear(model.attention.q_proj, rank=8)
model.attention.v_proj = LoRALinear(model.attention.v_proj, rank=8)
```

#### Output

- Trained LoRA adapters for each client and server
- Converged model accuracy on validation set
- Convergence curves (loss per round)

---

### PHASE 4: CLUSTER-AWARE AGGREGATION

#### Input

- Trained LoRA adapters from Phase 3
  - Client-side: B_i^c, A_i^c per device i
  - Server-side: B_i^s, A_i^s per device i
- Cluster assignments from Phase 1
- Hyperparameter: aggregation frequency (every I rounds)

#### Process

**Step 4.1: Separate Aggregation Per Cluster**

```python
# Group devices by cluster
clusters = {}
for i in range(N):
    c = cluster_assignments[i]
    if c not in clusters:
        clusters[c] = []
    clusters[c].append(i)

# Aggregate per cluster
aggregated_adapters = {}
for cluster_id, device_ids in clusters.items():
    # Client-side aggregation
    adapters_c = [get_client_adapter(i) for i in device_ids]
    adapters_c_agg = aggregate_adapters(adapters_c, weights='uniform')
    aggregated_adapters[f'client_{cluster_id}'] = adapters_c_agg

    # Server-side aggregation (if centralized server)
    # OR skip if server stays separate
```

**Step 4.2: Noise-Free Heterogeneous Aggregation**

Problem: Different clients have different LoRA ranks

```
Client 1: B₁ ∈ ℝ^{d×4}, A₁ ∈ ℝ^{4×d'}
Client 2: B₂ ∈ ℝ^{d×6}, A₂ ∈ ℝ^{6×d'}
```

Solution: Concatenation + Merge

```python
def aggregate_heterogeneous_lora(adapters_list, device_ids, weights=None):
    """
    adapters_list: List of (B_i, A_i) tuples with different ranks
    device_ids: List of device identifiers
    weights: Aggregation weights (default: uniform)
    """

    if weights is None:
        weights = [1.0 / len(adapters_list)] * len(adapters_list)

    # Extract dimensions
    d = adapters_list[0][0].shape[0]  # Input dimension
    d_prime = adapters_list[0][1].shape[1]  # Output dimension

    # Step 1: Concatenate all B and A matrices
    B_concat = torch.cat([B for B, A in adapters_list], dim=1)
    # B_concat shape: (d, sum_of_ranks)

    A_concat = torch.cat([A for B, A in adapters_list], dim=0)
    # A_concat shape: (sum_of_ranks, d')

    # Step 2: Weighted aggregate (on concatenated space)
    total_weight = sum(weights)
    weighted_B = sum(w * B for w, (B, A) in zip(weights, adapters_list)) / total_weight
    weighted_A = sum(w * A for w, (B, A) in zip(weights, adapters_list)) / total_weight

    # Alternative: Direct weighted sum on concatenated
    # This preserves all information (no dimensionality loss)

    # Step 3: Decompose back to original ranks (optional, for memory efficiency)
    # Use SVD to compress or keep concatenated form
    # For simplicity, keep concatenated form (merged rank = sum of individual ranks)

    return weighted_B, weighted_A

# Usage:
client_adapters = [(B_i_c, A_i_c) for i in cluster_devices]
B_agg_c, A_agg_c = aggregate_heterogeneous_lora(client_adapters, cluster_devices)

# Broadcast back
for i in cluster_devices:
    decompose_and_store(B_agg_c, A_agg_c, rank_i, device=i)
```

**Step 4.3: Broadcasting and Local Decomposition**

```python
def decompose_merged_adapter(B_merged, A_merged, target_rank):
    """
    B_merged: (d, merged_rank)
    A_merged: (merged_rank, d')
    target_rank: Original rank for this device

    Returns: (B_decomposed, A_decomposed) with shape constraints
    """

    # Option 1: Keep first target_rank columns of B, first target_rank rows of A
    B_decomposed = B_merged[:, :target_rank]
    A_decomposed = A_merged[:target_rank, :]

    # Option 2: SVD-based decomposition (better quality)
    # Compute SVD of product: B_merged @ A_merged
    X = B_merged @ A_merged  # (d, d')
    U, S, Vt = torch.svd(X)

    B_decomposed = U[:, :target_rank] @ torch.diag(S[:target_rank])
    A_decomposed = torch.diag(torch.ones(target_rank)) @ Vt[:target_rank, :]

    return B_decomposed, A_decomposed

# Broadcast to clients
for i in cluster_devices:
    B_i_decomposed, A_i_decomposed = decompose_merged_adapter(B_agg_c, A_agg_c, rank_i)
    send_to_client(B_i_decomposed, A_i_decomposed, device_id=i)
```

#### Output

- Cluster-level aggregated LoRA adapters
- Broadcasted adapters to all devices
- Aggregation quality metrics

#### Validation

- Verify no rank mismatch after decomposition
- Check aggregation quality (reconstruction error)
- Validate convergence improved after aggregation

---

### PHASE 5: PRIVACY EVALUATION

#### Input

- Trained ATLAS model from Phase 3-4
- VFLAIR-LLM framework (available at https://github.com/FLAIR-THU/VFLAIR-LLM)
- Test datasets

#### Process

**Step 5.1: Run Privacy Attacks**

```python
from vflair_llm import attacks

# Vanilla Model Inversion (VMI)
vmi_attack = attacks.VanillaModelInversion(model, device)
recovered_data_vmi = vmi_attack.attack(intermediate_h)
recall_score_vmi = compute_text_recall(recovered_data_vmi, ground_truth)

# Recursive Model Inversion (RMI)
rmi_attack = attacks.RecursiveModelInversion(model, device)
recovered_data_rmi = rmi_attack.attack(intermediate_h)
recall_score_rmi = compute_text_recall(recovered_data_rmi, ground_truth)

# Bidirectional Semantic Reconstruction (BiSR)
bisr_attack = attacks.BiSR(model, device)
recovered_data_bisr = bisr_attack.attack(intermediate_h)
recall_score_bisr = compute_text_recall(recovered_data_bisr, ground_truth)

# Label Inference Attacks (LIA)
lia_attack = attacks.LabelInferenceAttack(model, device)
inferred_labels = lia_attack.attack(intermediate_h)
label_accuracy = compute_accuracy(inferred_labels, ground_truth_labels)
```

**Step 5.2: Apply Privacy Defenses**

```python
from vflair_llm import defenses

# Differential Privacy (DP)
dp_defense = defenses.DifferentialPrivacy(epsilon=100, delta=1e-5)

# Mutual Information Disentanglement (MID) - RECOMMENDED
mid_defense = defenses.MID(lambda_param=0.5)

# Adversarial Training (AT)
at_defense = defenses.AdversarialTraining(lambda_param=0.1)

# Apply defenses during training
for h in intermediate_activations:
    h_defended = dp_defense(h)
    h_defended = mid_defense(h_defended)
```

**Step 5.3: Compute Privacy Metrics**

```python
def compute_dcs(main_task_performance, attack_performance, alpha=0.5):
    """
    Defense Capability Score

    Args:
        main_task_performance (MP): Final accuracy (0-1)
        attack_performance (AP): Attack success rate (0-1)
        alpha: Weight for utility vs privacy (default 0.5)

    Returns:
        DCS score (0-1)
    """
    dcs = alpha * main_task_performance + (1 - alpha) * (1 - attack_performance)
    return dcs

# Evaluate all attack-defense combinations
results = {}
for attack_name in ['VMI', 'RMI', 'BiSR', 'LIA']:
    for defense_name in ['None', 'DP', 'MID', 'AT', 'SP']:
        attack = get_attack(attack_name)
        defense = get_defense(defense_name)

        # Run attack with defense
        mp = evaluate_main_task(model, defense)
        ap = evaluate_attack(attack, model, defense)
        dcs = compute_dcs(mp, ap)

        results[f'{attack_name}_{defense_name}'] = {
            'MP': mp,
            'AP': ap,
            'DCS': dcs
        }

# Create DCS ranking
ranking = sorted(results.items(), key=lambda x: x[1]['DCS'], reverse=True)
print("Top 10 attack-defense pairs:")
for i, (pair, metrics) in enumerate(ranking[:10]):
    print(f"{i+1}. {pair}: DCS={metrics['DCS']:.4f}")
```

**Step 5.4: Generate Privacy Report**

```python
# Create visualization: MP vs AP scatter plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for pair, metrics in results.items():
    ax.scatter(metrics['MP'], metrics['AP'], label=pair, s=100)
    ax.annotate(pair, (metrics['MP'], metrics['AP']), fontsize=8)

ax.set_xlabel('Main Task Performance (MP)', fontsize=12)
ax.set_ylabel('Attack Performance (AP)', fontsize=12)
ax.set_title('Privacy-Utility Tradeoff')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid()
plt.tight_layout()
plt.savefig('privacy_utility_tradeoff.pdf')

# Recommendation
print("\n=== Privacy Recommendation ===")
best_defense = ranking[0][0].split('_')[1]
best_dcs = ranking[0][1]['DCS']
print(f"Recommended defense: {best_defense}")
print(f"Expected DCS: {best_dcs:.4f}")

if best_dcs < 0.7:
    print("WARNING: DCS < 0.7. Consider stronger privacy measures.")
else:
    print("✓ Privacy-Utility balance is acceptable (DCS ≥ 0.7)")
```

#### Output

- DCS rankings for all attack-defense pairs
- Privacy-utility tradeoff visualization
- Defense recommendations
- Privacy assessment report

---

## 2. KEY DATA STRUCTURES

### Client Context

```python
class ClientContext:
    client_id: int
    cluster_id: int
    dataset: Dataset  # Local training data
    memory_budget: float  # GB
    model_head: torch.nn.Module
    lora_adapter_client: dict  # {layer_name: LoRALinear}
    lora_ranks: dict  # {layer_name: int}
    split_point: int
    optimizer: torch.optim.Optimizer

    def forward_pass(self, x_batch):
        """Compute intermediate activations"""
        h = self.model_head(x_batch)
        return h

    def backward_pass(self, grad_h):
        """Update client-side LoRA adapters"""
        # Compute gradients for client parameters
        loss = torch.sum(grad_h * forward_output)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

class ServerContext:
    model_tail: torch.nn.Module
    lora_adapters_server: dict  # Per-device servers
    loss_fn: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def forward_backward_pass(self, h, y_batch):
        """Compute loss and gradients"""
        logits = self.model_tail(h)
        loss = self.loss_fn(logits, y_batch)
        grad_h = torch.autograd.grad(loss, h)[0]
        # Update server LoRA
        self.optimizer.step()
        self.optimizer.zero_grad()
        return grad_h

class ClusterContext:
    cluster_id: int
    member_ids: list  # Device IDs in this cluster
    aggregated_lora_adapters: dict

    def aggregate(self, adapters_per_device):
        """Heterogeneous aggregation"""
        # Concatenate, merge, decompose
        pass
```

---

## 3. COMMUNICATION PROTOCOL

### Round Structure (for T rounds)

```
Round t:
├── Phase 3 Training (L local steps per client)
│   ├── Client i: Forward pass (h_i)
│   │   ├── Send: h_i to server
│   │   ├── Receive: grad_h_i from server
│   │   └── Backward pass: Update LoRA adapters
│   │
│   └── Server: Forward/Backward pass
│       ├── Receive: h_i from client
│       ├── Compute: loss, grad_h_i
│       ├── Send: grad_h_i to client
│       └── Update: Server LoRA adapters
│
└── Aggregation (every I rounds)
    ├── Client i: Upload LoRA adapters to aggregator
    │   └── Send: B_i^c, A_i^c
    │
    ├── Aggregator: Cluster-level aggregation
    │   ├── Group by cluster
    │   ├── Noise-free concatenation + merge
    │   └── Decompose back to original ranks
    │
    └── Client i: Download aggregated adapters
        └── Receive: B_i^c_agg, A_i^c_agg
```

### Message Sizes

```
Forward (Client → Server):
├── Intermediate h: (batch_size, hidden_dim) = 32 × 768 = 24,576 floats
├── Size: 24,576 × 4 bytes = ~96 KB

Backward (Server → Client):
├── Gradients grad_h: (batch_size, hidden_dim) = 32 × 768 = 24,576 floats
├── Size: 24,576 × 4 bytes = ~96 KB

Aggregation (Client → Aggregator → Client):
├── Per adapter: B_i ∈ ℝ^{d×r}, A_i ∈ ℝ^{r×d'}
├── Example: d=768, d'=768, r=8
├── B_i size: 768 × 8 × 4 = 24,576 bytes
├── A_i size: 8 × 768 × 4 = 24,576 bytes
├── Total per device: ~50 KB
├── Total for 100 devices: ~5 MB (manageable)
```

---

## 4. ERROR HANDLING & ROBUSTNESS

### Client Dropout Handling

```python
def handle_client_dropout(active_clients, num_expected_clients):
    """
    During training, some clients may disconnect
    """
    dropout_rate = 1 - len(active_clients) / num_expected_clients

    if dropout_rate > 0.3:
        print(f"WARNING: {dropout_rate*100:.1f}% clients dropped")
        # Strategy 1: Wait for stragglers (timeout)
        # Strategy 2: Continue with active clients only
        # Strategy 3: Intra-cluster re-pairing (use backup clients)
        return True

    return False

def intra_cluster_repair(cluster_members, active_clients):
    """
    If client drops, reassign to another client in same cluster
    """
    for client_id in cluster_members:
        if client_id not in active_clients:
            # Find replacement in same cluster
            for backup_id in cluster_members:
                if backup_id in active_clients and backup_id != client_id:
                    reassign_data(client_id, backup_id)
                    break
```

### Gradient Validation

```python
def validate_gradients(grad_h):
    """Check for NaN/Inf"""
    if torch.isnan(grad_h).any():
        print("ERROR: NaN in gradients!")
        return False
    if torch.isinf(grad_h).any():
        print("ERROR: Inf in gradients!")
        return False
    return True
```

---

## 5. HYPERPARAMETER TUNING

| Hyperparameter     | Range        | Default | Notes                       |
| ------------------ | ------------ | ------- | --------------------------- |
| Fingerprint dim    | 32-256       | 64      | Higher = more information   |
| Num clusters k     | 2-20         | 5       | Elbow method to select      |
| LoRA rank          | 1-32         | 4-16    | Per device/layer            |
| Learning rate      | 1e-5 to 1e-2 | 1e-3    | Task/model dependent        |
| Local steps L      | 1-10         | 5       | Client-side batches         |
| Aggregation freq I | 1-50         | 10      | Rounds between aggregations |
| Num rounds T       | 100-1000     | 500     | Total training rounds       |
| Batch size         | 8-64         | 32      | GPU memory constraint       |
| DP epsilon (ε)     | 50-500       | 100     | Lower = stronger privacy    |
| MID lambda (λ)     | 0.001-1.0    | 0.5     | Privacy-utility tradeoff    |

---

## 6. TESTING CHECKLIST

- [ ] Phase 1: Cluster quality (silhouette > 0.4)
- [ ] Phase 1: Task fingerprint validity (no NaNs)
- [ ] Phase 2: Rank assignment respects memory
- [ ] Phase 2: Ranks > 0 for all layers
- [ ] Phase 3: Training loop converges
- [ ] Phase 3: Client-server communication works
- [ ] Phase 3: LoRA adapters update correctly
- [ ] Phase 4: Aggregation produces valid adapters
- [ ] Phase 4: Decomposed ranks match originals
- [ ] Phase 5: Privacy attacks run without error
- [ ] Phase 5: Defenses reduce attack success
- [ ] Phase 5: DCS ≥ 0.7 with recommended defense

---

## 7. PERFORMANCE BENCHMARKING

### Metrics to Track

**Accuracy:**

- Per-task classification/QA accuracy
- Convergence curve (accuracy vs. round)

**Efficiency:**

- Communication: Total MB transmitted per round
- Computation: Seconds per round (client + server)
- Memory: Peak GPU usage per device

**Privacy:**

- Attack success rate per attack type
- Defense Capability Score (DCS)
- Privacy-utility tradeoff curves

**Convergence:**

- Rounds to target accuracy
- Sensitivity to hyperparameters

### Comparison Baselines

1. **Centralized** (upper bound): Full model on central server
2. **Federated Learning** (FedAvg): Standard FL without split
3. **SplitLoRA** (homogeneous baseline): Without task clustering
4. **HSplitLoRA** (heterogeneous baseline): Without task clustering
5. **MIRA** (task-aware baseline): Without split learning

---

## CONCLUSION

This technical specification provides:

1. ✓ Detailed algorithms for each phase
2. ✓ Code templates and pseudocode
3. ✓ Data structures and communication protocols
4. ✓ Error handling and robustness measures
5. ✓ Hyperparameter ranges and tuning guidance
6. ✓ Testing and benchmarking procedures

Ready for implementation beginning Week 1.
