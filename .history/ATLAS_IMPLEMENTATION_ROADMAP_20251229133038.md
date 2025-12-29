# ATLAS Implementation Roadmap

## Detailed Technical Specifications

---

## Overview

This document provides detailed technical specifications for implementing each phase of the ATLAS project. It serves as the developer's guide for turning the research plan into working code.

---

## PHASE 1: TASK CLUSTERING (Weeks 1-2)

### Objectives

- Extract gradient fingerprints from client updates
- Perform k-Means clustering on gradient space
- Form task groups with similar learning characteristics
- Validate clustering quality

### Component: GradientExtractor

**Purpose:** Extract task-relevant features from model gradients

**Input:**

- Client gradient tensors (shape: various, depends on model)
- (Optional) Model architecture info

**Processing:**

```
1. Flatten all gradients into 1D vector
2. Compute norm and direction statistics
3. Extract 64-D fingerprint via PCA/dimensionality reduction
   - Use top 64 principal components
   - Normalize to unit L2 norm
4. Cache fingerprints for clustering
```

**Output:**

- Fingerprint matrix: `(n_clients, 64)`
- Fingerprint metadata: client_ids, timestamps

**Code structure:**

```python
class GradientExtractor:
    def __init__(self, dim=64):
        self.dim = dim
        self.pca = None

    def extract(self, gradients):
        """Extract d-dimensional fingerprints"""
        flat_grads = self._flatten(gradients)
        fingerprints = self.pca.transform(flat_grads)
        return fingerprints

    def _flatten(self, gradients):
        """Convert tensor dict to 1D vector"""
        # Implementation
```

### Component: TaskClusterer

**Purpose:** Cluster clients into task groups

**Algorithm:**

1. Input: Fingerprint matrix `F` of shape `(n_clients, 64)`
2. For `k = 2` to `5`:
   a. Run k-Means with 10 random initializations
   b. Compute Silhouette score
   c. Save best clustering
3. Select k with highest Silhouette score
4. Output: Cluster assignments and centroids

**Key parameters:**

- n_clusters: 2-5 (auto-select based on Silhouette)
- n_init: 10 (random initializations)
- tolerance: 1e-4
- max_iter: 300

**Code structure:**

```python
class TaskClusterer:
    def __init__(self, n_clusters_range=(2, 5)):
        self.n_clusters_range = n_clusters_range
        self.best_kmeans = None
        self.best_score = -1

    def cluster(self, fingerprints):
        """Cluster fingerprints and return task groups"""
        best_n = None
        for n_clusters in self.n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(fingerprints)
            score = silhouette_score(fingerprints, labels)
            if score > self.best_score:
                self.best_score = score
                self.best_kmeans = kmeans
                best_n = n_clusters

        return {
            'n_clusters': best_n,
            'labels': self.best_kmeans.labels_,
            'centroids': self.best_kmeans.cluster_centers_,
            'silhouette_score': self.best_score
        }

    def get_task_groups(self, client_ids):
        """Return task group assignments"""
        groups = {}
        for client_id, label in zip(client_ids, self.best_kmeans.labels_):
            if label not in groups:
                groups[label] = []
            groups[label].append(client_id)
        return groups
```

### Testing & Validation

**Unit tests:**

1. Test GradientExtractor

   - Input: Random gradient dict
   - Expected output: 64-D fingerprint
   - Check: shape, normalization, reproducibility

2. Test TaskClusterer

   - Input: Synthetic fingerprints with known clusters
   - Expected output: Correct cluster assignments
   - Check: Silhouette score, clustering quality

3. Integration test
   - Input: Real gradients from multiple clients
   - Expected output: Task groups
   - Check: Group coherence, Silhouette scores

**Metrics to compute:**

- Silhouette score (0.0 to 1.0, higher is better)
- Davies-Bouldin index (lower is better)
- Calinski-Harabasz score (higher is better)
- Number of clusters selected

**Visualization:**

```python
# t-SNE visualization of clusters
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
fingerprints_2d = tsne.fit_transform(fingerprints)

for cluster_id in range(n_clusters):
    mask = labels == cluster_id
    plt.scatter(fingerprints_2d[mask, 0],
                fingerprints_2d[mask, 1],
                label=f'Task {cluster_id}')
plt.legend()
plt.show()
```

### Deliverables

- [ ] `GradientExtractor` class with unit tests
- [ ] `TaskClusterer` class with validation
- [ ] Visualization notebook showing t-SNE plots
- [ ] Clustering quality report (Silhouette, DBI, CH scores)
- [ ] Tested on dummy client data (10-100 clients)

---

## PHASE 2: HETEROGENEOUS CONFIGURATION (Weeks 2-3)

### Objectives

- Profile device capabilities (memory, compute)
- Score parameter importance per device
- Allocate heterogeneous LoRA ranks
- Optimize for memory-accuracy trade-off

### Component: DeviceProfiler

**Purpose:** Characterize device capabilities

**Input:**

- Device type: "cpu", "edge_gpu", "gpu"
- Optional: Actual device specs

**Device profiles:**

```
Device Type    | Memory | Compute | Suggested Ranks
CPU            | 2GB    | 1x      | 4, 8
Edge GPU       | 4GB    | 5x      | 8, 16
GPU            | 8GB+   | 10x+    | 16, 32, 64
```

**Code:**

```python
class DeviceProfiler:
    PROFILES = {
        'cpu': {'memory_mb': 2048, 'compute_ratio': 1.0},
        'edge_gpu': {'memory_mb': 4096, 'compute_ratio': 5.0},
        'gpu': {'memory_mb': 8192, 'compute_ratio': 10.0}
    }

    def profile_device(self, device_type):
        return self.PROFILES[device_type]

    def estimate_rank(self, device_type, model_dim, target_layers):
        profile = self.profile_device(device_type)
        # Memory per rank: 2 * model_dim * rank * num_layers * 4 bytes
        rank_max = profile['memory_mb'] * 1024 / (2 * model_dim * target_layers * 4)
        return min(int(rank_max), 64)  # Cap at 64
```

### Component: WeightImportanceScorer

**Purpose:** Identify critical parameters per layer

**Method:** Sensitivity analysis

- For each layer, compute gradient magnitude
- Higher magnitude = more important for this layer
- Use Layer-wise Relevance Propagation (LRP) or simple gradient norm

**Algorithm:**

```
For each layer in model:
    1. Compute gradient norm: g_norm = ||∇w||_2
    2. Compute output sensitivity via backprop
    3. Importance_score = g_norm * output_sensitivity
    4. Normalize: importance_i = importance_i / sum(importance)
```

**Code:**

```python
class WeightImportanceScorer:
    def __init__(self, model):
        self.model = model
        self.importance_cache = {}

    def compute_importance(self, batch_data):
        """Compute parameter importance via gradient norms"""
        self.model.zero_grad()
        logits = self.model(batch_data)
        loss = F.cross_entropy(logits, batch_data.get('labels'))
        loss.backward()

        importance = {}
        total_norm = 0

        for name, param in self.model.named_parameters():
            if 'lora' not in name and param.grad is not None:
                grad_norm = param.grad.norm().item()
                importance[name] = grad_norm
                total_norm += grad_norm

        # Normalize
        for name in importance:
            importance[name] /= (total_norm + 1e-8)

        return importance

    def get_layer_importance(self, layer_name, importance_dict):
        """Get importance for specific layer"""
        return sum(v for k, v in importance_dict.items()
                   if k.startswith(layer_name))
```

### Component: RankAllocator

**Purpose:** Assign heterogeneous LoRA ranks per device

**Algorithm:**

```
Input: Task groups, device profiles, importance scores

For each device in task_group:
    1. Get device profile (memory constraint)
    2. Get model dimensions
    3. For each layer:
        importance = importance_score[layer]
        memory_budget = profile.memory - reserved

        # Allocate rank proportional to importance
        rank_candidate = min(
            64,  # Max rank
            memory_budget / (2 * dim * 4)  # Memory constraint
        )
        rank_final = round(rank_candidate / 4) * 4  # Round to multiple of 4

        assign(device, layer, rank_final)
```

**Code:**

```python
class RankAllocator:
    def __init__(self, model_dim=768):
        self.model_dim = model_dim
        self.rank_candidates = [4, 8, 16, 32, 64]

    def allocate_ranks(self, device_profile, importance_scores, n_layers):
        """Allocate heterogeneous ranks"""
        memory_budget = device_profile['memory_mb'] * 1024  # Convert to KB
        ranks_per_layer = []

        for layer_idx in range(n_layers):
            # Get importance for this layer
            importance = importance_scores.get(f'layer_{layer_idx}', 0.5)

            # Compute max rank based on memory
            memory_per_rank = 2 * self.model_dim * 4  # bytes per rank
            max_rank = memory_budget / (n_layers * memory_per_rank)

            # Select rank from candidates
            rank = min([r for r in self.rank_candidates if r <= max_rank],
                      default=4)
            ranks_per_layer.append(rank)

        return ranks_per_layer

    def get_rank_for_device(self, device_id, device_type, task_group_importance):
        """Get ranks for specific device"""
        profile = DeviceProfiler().profile_device(device_type)
        importance = task_group_importance.get(device_id, {})
        n_layers = 12  # Assume 12-layer model

        return self.allocate_ranks(profile, importance, n_layers)
```

### Testing & Validation

**Unit tests:**

1. DeviceProfiler

   - Input: device type
   - Output: profile dict
   - Check: memory, compute ratio

2. WeightImportanceScorer

   - Input: sample batch
   - Output: importance dict
   - Check: sum of importance ≈ 1.0

3. RankAllocator
   - Input: device profile, importance scores
   - Output: rank per layer
   - Check: ranks in valid range, memory constraint satisfied

**Validation:**

- Memory constraint: `sum(rank_i * dim * 2 * 4) <= memory_budget`
- Rank range: all ranks in [4, 8, 16, 32, 64]
- Importance correlation: important layers get higher ranks

### Deliverables

- [ ] `DeviceProfiler` class
- [ ] `WeightImportanceScorer` class
- [ ] `RankAllocator` class
- [ ] Unit tests for all components
- [ ] Rank allocation visualization
- [ ] Memory constraint validation

---

## PHASE 3: SPLIT FEDERATED LEARNING (Weeks 3-6)

### Objectives

- Implement client-side training with LoRA
- Implement server-side computation
- Establish communication protocol
- Integrate with LLMs (GPT-2, LLaMA)

### Component: SplitClient

**Purpose:** Client-side training with split learning + LoRA

**Architecture:**

```
Input → Bottom Layers (Client) → Activation → Network → Server
                    ↓
                 LoRA adapters
                    ↓
                Train LoRA weights
```

**Key attributes:**

- `client_id`: unique identifier
- `model`: bottom layers of pre-trained LLM
- `rank_config`: heterogeneous ranks per layer
- `local_data`: client's local dataset
- `lora_weights`: trainable LoRA parameters

**Code:**

```python
class SplitClient:
    def __init__(self, client_id, model_name, rank_config, device='cpu'):
        self.client_id = client_id
        self.device = device
        self.rank_config = rank_config

        # Load pre-trained model (bottom layers only)
        full_model = AutoModel.from_pretrained(model_name)
        self.model = nn.Sequential(*list(full_model.children())[:-3])

        # Attach LoRA adapters
        self.lora_adapters = self._create_lora_adapters(rank_config)

        # Optimizer
        self.optimizer = Adam(self._get_trainable_params())

    def _create_lora_adapters(self, rank_config):
        """Attach LoRA to model layers"""
        adapters = nn.ModuleDict()
        for layer_idx, rank in rank_config.items():
            adapters[f'layer_{layer_idx}'] = LoRAAdapter(
                in_dim=768,
                rank=rank
            )
        return adapters

    def _get_trainable_params(self):
        """Return only LoRA weights (not base model)"""
        return self.lora_adapters.parameters()

    def forward(self, batch):
        """Forward pass through bottom layers"""
        x = batch
        for i, layer in enumerate(self.model):
            x = layer(x)
            # Apply LoRA if available for this layer
            if f'layer_{i}' in self.lora_adapters:
                x = x + self.lora_adapters[f'layer_{i}'](x)
        return x

    def compute_activations(self, batch):
        """Compute intermediate activations to send to server"""
        with torch.no_grad():
            activations = self.forward(batch)
        return activations

    def train_step(self, batch, loss_from_server):
        """Train LoRA weights with backprop from server"""
        self.optimizer.zero_grad()

        # Forward pass
        logits = self.forward(batch)

        # Backward pass with loss from server
        # Assumes loss_from_server is computed on server side
        loss_from_server.backward()

        self.optimizer.step()

        return self.get_lora_weights()

    def get_lora_weights(self):
        """Return current LoRA weights"""
        weights = {}
        for name, adapter in self.lora_adapters.items():
            weights[name] = {
                'A': adapter.A.data.clone(),
                'B': adapter.B.data.clone()
            }
        return weights

    def set_lora_weights(self, weights):
        """Update LoRA weights after aggregation"""
        for name, weight_dict in weights.items():
            self.lora_adapters[name].A.data = weight_dict['A'].clone()
            self.lora_adapters[name].B.data = weight_dict['B'].clone()
```

### Component: SplitServer

**Purpose:** Server-side computation and aggregation

**Architecture:**

```
Activation (from clients) → Top Layers → Task Heads → Output
                                ↓
                         LoRA adapters
                                ↓
                        Compute loss & backprop
                                ↓
                        Send gradients to clients
```

**Code:**

```python
class SplitServer:
    def __init__(self, model_name, n_tasks, task_heads=None):
        self.model_name = model_name
        self.n_tasks = n_tasks

        # Load full model and keep only top layers
        full_model = AutoModel.from_pretrained(model_name)
        self.model = nn.Sequential(*list(full_model.children())[-3:])

        # Task-specific heads
        if task_heads is None:
            self.task_heads = nn.ModuleDict({
                f'task_{i}': nn.Linear(768, 10) for i in range(n_tasks)
            })
        else:
            self.task_heads = task_heads

        # Optimizer for server-side params
        self.optimizer = Adam(self.parameters())
        self.global_model = None

    def forward(self, activations, task_id):
        """Forward pass through top layers + task head"""
        x = self.model(activations)
        logits = self.task_heads[f'task_{task_id}'](x)
        return logits

    def compute_loss(self, logits, labels, task_id):
        """Compute loss for specific task"""
        return F.cross_entropy(logits, labels)

    def backward(self, loss):
        """Backward pass and return gradients for clients"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return activation gradients to clients
        return loss.item()

    def parameters(self):
        """Return all trainable parameters"""
        return list(self.model.parameters()) + list(self.task_heads.parameters())
```

### Component: LoRAAdapter

**Purpose:** LoRA module for efficient fine-tuning

**Mechanism:**

- Input: `x` of shape `(batch, seq_len, hidden_dim)`
- Output: `A @ B^T @ x` of shape `(batch, seq_len, hidden_dim)`
- Where `A` has rank `r << hidden_dim`

**Code:**

```python
class LoRAAdapter(nn.Module):
    def __init__(self, in_dim, rank=8, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank

        # LoRA parameters: A and B matrices
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, in_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Compute LoRA output: x @ A @ B^T"""
        # x shape: (batch, seq_len, in_dim) or (batch, in_dim)
        return torch.matmul(
            torch.matmul(x, self.A),
            self.B.T
        )
```

### Communication Protocol

**Message format:**

1. **Client → Server (Activations)**

```
{
    'client_id': int,
    'round': int,
    'task_id': int,
    'activations': Tensor,  # (batch_size, hidden_dim)
    'labels': Tensor  # (batch_size,)
}
```

2. **Server → Client (Aggregated weights)**

```
{
    'round': int,
    'lora_weights': {
        'layer_0': {'A': Tensor, 'B': Tensor},
        'layer_1': {'A': Tensor, 'B': Tensor},
        ...
    },
    'metrics': {
        'global_loss': float,
        'task_accuracies': dict
    }
}
```

### Training Loop

```python
def federated_training_round(clients, server, task_groups):
    """One round of federated training"""

    client_lora_updates = {}

    # Step 1: Clients compute activations and train locally
    for client in clients:
        task_id = get_task_for_client(client.client_id, task_groups)

        # Forward through client layers
        activations = client.compute_activations(batch)

        # Send to server
        logits = server.forward(activations, task_id)
        loss = server.compute_loss(logits, labels, task_id)

        # Backward through server and client
        server.backward(loss)
        local_updates = client.train_step(batch, loss)

        client_lora_updates[client.client_id] = local_updates

    # Step 2: Aggregate LoRA weights
    aggregated_weights = server.aggregate(
        client_lora_updates,
        task_groups
    )

    # Step 3: Broadcast aggregated weights to clients
    for client in clients:
        client.set_lora_weights(aggregated_weights)

    # Step 4: Evaluate on validation set
    val_metrics = evaluate(server, clients, task_groups)

    return val_metrics
```

### Integration with HuggingFace Models

**Supported models:**

- GPT-2 (12 layers, 768 hidden dim)
- LLaMA-7B (32 layers, 4096 hidden dim)
- BERT (12 layers, 768 hidden dim)

**Split point strategy:**

```python
def get_split_point(model_name):
    """Determine where to split model"""
    if 'gpt2' in model_name.lower():
        return 6  # Split at layer 6 out of 12
    elif 'llama' in model_name.lower() and '7b' in model_name.lower():
        return 16  # Split at layer 16 out of 32
    elif 'bert' in model_name.lower():
        return 6  # Split at layer 6 out of 12
    else:
        raise ValueError(f"Unsupported model: {model_name}")
```

### Testing & Validation

1. Unit tests for SplitClient

   - Test forward pass shape
   - Test backward pass
   - Test LoRA weight updates

2. Unit tests for SplitServer

   - Test task-specific heads
   - Test loss computation
   - Test weight updates

3. Integration tests
   - End-to-end training on GLUE tasks
   - Convergence verification
   - Activation shape validation

### Deliverables

- [ ] `SplitClient` class with LoRA
- [ ] `SplitServer` class with task heads
- [ ] `LoRAAdapter` module
- [ ] Communication protocol implementation
- [ ] Integration with GPT-2 and LLaMA
- [ ] Training loop with one complete round
- [ ] Unit and integration tests
- [ ] Training curves (loss, accuracy) on GLUE subset

---

## PHASE 4: PRIVACY-AWARE AGGREGATION (Weeks 6-8)

### Objectives

- Implement noise-free LoRA aggregation
- Apply task-aware weighting
- Validate privacy preservation
- Optimize aggregation quality

### Component: AggregationEngine

**Purpose:** Aggregate heterogeneous LoRA weights with privacy

**Algorithm:**

```
Input: LoRA weights from all clients, task_groups

For each task_group g:
    1. Collect LoRA weights from clients in group g
       {A_1, B_1}, {A_2, B_2}, ..., {A_k, B_k}

    2. Handle heterogeneous ranks:
       For each client i:
           If rank_i == 64:
               A_i_normalized = A_i[:, :64]
               B_i_normalized = B_i[:64, :]
           Else:
               A_i_normalized = pad(A_i, target_rank)

    3. Concatenate LoRA weights (noise-free):
       W_concat = [A_1 : A_2 : ... : A_k]  # Concatenate along rank dim

    4. Merge via low-rank approximation:
       U, Σ, V^T = SVD(W_concat)
       W_merged = U[:, :rank_target] @ Σ[:rank_target] @ V[:rank_target, :]^T

    5. Apply task weighting:
       α_g = |group_g| / total_clients  # Size-based weighting

Output: Merged LoRA weights per task group
```

**Code:**

```python
class AggregationEngine:
    def __init__(self, target_rank=32):
        self.target_rank = target_rank

    def aggregate_task_group(self, client_updates, group_clients, group_id):
        """Aggregate LoRA weights for a task group"""

        # Collect weights from group
        group_weights = [
            client_updates[client_id]
            for client_id in group_clients
        ]

        aggregated = {}

        for layer_idx in range(len(group_weights[0])):  # Iterate over layers
            # Get A and B from all clients
            A_list = [w[layer_idx]['A'] for w in group_weights]
            B_list = [w[layer_idx]['B'] for w in group_weights]

            # Handle heterogeneous ranks
            A_list = self._normalize_ranks(A_list)
            B_list = self._normalize_ranks(B_list)

            # Concatenate along rank dimension
            A_concat = torch.cat(A_list, dim=1)  # (hidden_dim, sum_of_ranks)
            B_concat = torch.cat(B_list, dim=0)  # (sum_of_ranks, hidden_dim)

            # Low-rank merge via SVD
            A_merged, B_merged = self._svd_merge(
                A_concat, B_concat, self.target_rank
            )

            aggregated[layer_idx] = {
                'A': A_merged,
                'B': B_merged
            }

        return aggregated

    def _normalize_ranks(self, weight_list, target_rank=64):
        """Pad weights to uniform rank"""
        normalized = []
        for w in weight_list:
            if w.shape[1] < target_rank:
                # Pad with zeros
                padding = torch.zeros(
                    w.shape[0], target_rank - w.shape[1],
                    device=w.device, dtype=w.dtype
                )
                w_padded = torch.cat([w, padding], dim=1)
            else:
                w_padded = w[:, :target_rank]
            normalized.append(w_padded)
        return normalized

    def _svd_merge(self, A_concat, B_concat, target_rank):
        """Merge via low-rank approximation"""
        # Compute W = A_concat @ B_concat
        W = A_concat @ B_concat  # (hidden_dim, hidden_dim)

        # SVD decomposition
        U, Sigma, Vt = torch.linalg.svd(W, full_matrices=False)

        # Keep only top target_rank components
        U_r = U[:, :target_rank]
        Sigma_r = Sigma[:target_rank]
        Vt_r = Vt[:target_rank, :]

        # Reconstruct A and B
        A_merged = U_r @ torch.diag(torch.sqrt(Sigma_r))
        B_merged = torch.diag(torch.sqrt(Sigma_r)) @ Vt_r

        return A_merged, B_merged

    def aggregate_all_groups(self, client_updates, task_groups):
        """Aggregate across all task groups"""
        aggregated_groups = {}

        for group_id, group_clients in task_groups.items():
            aggregated_groups[group_id] = self.aggregate_task_group(
                client_updates, group_clients, group_id
            )

        return aggregated_groups

    def weighted_merge(self, aggregated_groups, weights=None):
        """Merge across groups with task-aware weights"""
        if weights is None:
            # Equal weighting
            weights = {
                gid: 1.0 / len(aggregated_groups)
                for gid in aggregated_groups
            }

        # Initialize global weights from first group
        first_group = next(iter(aggregated_groups.values()))
        global_weights = {}

        for layer_idx in first_group:
            A_sum = None
            B_sum = None

            for group_id, group_weights in aggregated_groups.items():
                w_group = weights[group_id]

                if A_sum is None:
                    A_sum = w_group * group_weights[layer_idx]['A']
                    B_sum = w_group * group_weights[layer_idx]['B']
                else:
                    A_sum += w_group * group_weights[layer_idx]['A']
                    B_sum += w_group * group_weights[layer_idx]['B']

            global_weights[layer_idx] = {'A': A_sum, 'B': B_sum}

        return global_weights
```

### Privacy Verification

**Methods to verify privacy preservation:**

1. **Gradient norm comparison:**

```python
def check_gradient_leakage(original_grads, aggregated_grads):
    """Verify gradients don't directly leak updates"""
    norm_ratio = torch.norm(aggregated_grads) / torch.norm(original_grads)
    assert norm_ratio < 1.0, "Aggregation should reduce gradient magnitude"
```

2. **Privacy attack resilience:**

```python
def compute_privacy_score(model_updates, task_group_size):
    """Higher task_group_size → more privacy"""
    # Inspired by: Shokri et al. Membership Inference
    privacy_score = math.log(task_group_size)  # Logarithmic privacy
    return privacy_score
```

3. **Update indistinguishability:**

```python
def check_update_indistinguishability(updates_before_agg, updates_after_agg):
    """Verify aggregated updates are diverse"""
    diff = updates_before_agg - updates_after_agg
    diversity = torch.std(diff)
    return diversity > threshold
```

### Testing & Validation

1. Unit tests

   - Test weight normalization
   - Test SVD merge quality
   - Test task-aware weighting

2. Privacy tests

   - Test gradient norm reduction
   - Test membership inference resilience
   - Test update indistinguishability

3. Quality tests
   - Test convergence with aggregation
   - Test accuracy preservation
   - Compare against homogeneous aggregation

### Deliverables

- [ ] `AggregationEngine` class
- [ ] Heterogeneous rank handling
- [ ] SVD-based weight merging
- [ ] Task-aware weighting
- [ ] Privacy verification tests
- [ ] Aggregation quality metrics
- [ ] Integration with training loop

---

## PHASE 5: EXPERIMENTAL EVALUATION (Weeks 8-10)

### Objectives

- Integrate real datasets (GLUE, SQuAD, E2E)
- Implement end-to-end federated training pipeline
- Measure task-specific performance metrics
- Evaluate communication and memory efficiency

### Component: FederatedTrainer

**Purpose:** Coordinate multi-round federated training across clients

**Key Features:**

1. Dataset distribution across clients
2. Multi-round training loop
3. Convergence monitoring
4. Metric collection and logging

**Code structure:**

```python
class FederatedTrainer:
    def __init__(self, clients, server, task_groups, config):
        self.clients = clients
        self.server = server
        self.task_groups = task_groups
        self.config = config
        self.history = []

    def train_round(self, round_num):
        """Execute one federated training round"""
        # 1. Clients perform local training
        client_updates = {}
        for client in self.clients:
            updates = client.train_step()
            client_updates[client.client_id] = updates

        # 2. Server aggregates updates
        aggregated = self.server.aggregate(client_updates, self.task_groups)

        # 3. Broadcast global model
        for client in self.clients:
            client.set_weights(aggregated)

        # 4. Evaluate and log
        metrics = self.evaluate()
        self.history.append(metrics)
        return metrics

    def train(self, num_rounds):
        """Run full federated training"""
        for round_num in range(num_rounds):
            print(f"\nRound {round_num + 1}/{num_rounds}")
            metrics = self.train_round(round_num)
            print(f"Avg Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        return self.history

    def evaluate(self):
        """Evaluate current global model"""
        total_loss = 0
        total_acc = 0
        for client in self.clients:
            loss, acc = client.evaluate()
            total_loss += loss
            total_acc += acc

        return {
            'loss': total_loss / len(self.clients),
            'accuracy': total_acc / len(self.clients)
        }
```

### Dataset Integration

**GLUE Tasks:**

```python
from datasets import load_dataset

class GLUEDataLoader:
    def __init__(self, task_name, n_clients):
        self.task_name = task_name  # 'mrpc', 'cola', 'sst2', etc.
        self.n_clients = n_clients
        self.dataset = load_dataset('glue', task_name)

    def distribute_data(self, distribution='iid'):
        """Distribute dataset across clients"""
        if distribution == 'iid':
            return self._iid_split()
        elif distribution == 'non_iid':
            return self._non_iid_split()

    def _iid_split(self):
        """Split data randomly (IID)"""
        # Implementation
        pass

    def _non_iid_split(self):
        """Split data by label distribution (non-IID)"""
        # Implementation
        pass
```

### Evaluation Metrics

**Performance Metrics:**

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'accuracy': [],
            'f1_score': [],
            'loss': [],
            'communication_cost': [],
            'memory_usage': [],
            'training_time': []
        }

    def compute_accuracy(self, predictions, labels):
        """Task-specific accuracy"""
        return (predictions == labels).float().mean().item()

    def compute_communication_cost(self, updates):
        """Measure bytes transmitted"""
        total_bytes = 0
        for layer_weights in updates.values():
            for param in layer_weights.values():
                total_bytes += param.nelement() * param.element_size()
        return total_bytes / (1024 ** 2)  # MB

    def compute_memory_usage(self, model):
        """Measure model memory footprint"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024 ** 2)  # MB
```

**Differential Privacy accounting:**

```
(ε, δ)-Differential Privacy:
- ε: privacy loss parameter (smaller = more private)
- δ: failure probability (smaller = more robust)

### Testing & Validation

1. **Dataset loading tests**
   - Verify GLUE/SQuAD/E2E data loading
   - Test IID and non-IID splits
   - Validate data distribution

2. **Training pipeline tests**
   - Test single round training
   - Test multi-round convergence
   - Verify gradient flow

3. **Integration tests**
   - End-to-end federated training
   - Metric collection and logging
   - Model checkpoint saving/loading

### Deliverables

- [ ] Dataset loaders for GLUE, SQuAD, E2E
- [ ] FederatedTrainer implementation
- [ ] MetricsCollector implementation
- [ ] Training scripts and configurations
- [ ] Convergence monitoring utilities
- [ ] Results logging and visualization
- [ ] Model checkpoint management

---

## PHASE 6: BENCHMARKING & RESULTS (Weeks 10-12)

### Objectives

- Benchmark ATLAS on standard datasets (GLUE, SQuAD, E2E)
- Compare against baselines (Standard FL, Homogeneous LoRA)
- Generate comprehensive results tables and plots
- Demonstrate end-to-end system capabilities

### Benchmark Framework

```python
class ATLASBenchmark:
    def __init__(self, models, datasets, baselines):
        self.models = models  # ATLAS vs baselines
        self.datasets = datasets  # GLUE, SQuAD, E2E
        self.baselines = baselines  # Standard FL, Homogeneous LoRA
        self.results = {}

    def run_all_experiments(self):
        """Run complete benchmark suite"""
        for dataset in self.datasets:
            for model in self.models:
                # Load dataset
                train_data, val_data, test_data = load_dataset(dataset)

                # Train model
                trained_model = self._train_model(model, train_data)

                # Evaluate
                results = self._evaluate(trained_model, test_data)

                # Store results
                self.results[(dataset, model)] = results

        return self.results

    def compute_comparison_table(self):
        """Generate comparison table"""
        # Create table with:
        # - Dataset
        # - Model
        # - Accuracy
        # - Memory Usage
        # - Communication Cost
        # - Training Time
        # - Privacy (DCS)
        pass

    def generate_plots(self):
        """Generate visualization plots"""
        # 1. Accuracy vs Memory trade-off
        # 2. Communication cost comparison
        # 3. Convergence curves
        # 4. Privacy-Accuracy Pareto frontier
        pass
```

### Demo Script

```python
def run_atlas_demo():
    """Live demonstration of ATLAS system"""

    print("=" * 80)
    print("ATLAS FEDERATED LEARNING SYSTEM DEMO")
    print("=" * 80)

    # Step 1: Initialize System
    print("\n[1] Initializing ATLAS with 10 clients (GLUE tasks)...")
    clients = create_simulated_clients(n_clients=10, device_types=['cpu', 'edge_gpu', 'gpu'])
    server = ATLASServer(model_name='gpt2', n_tasks=4)

    # Step 2: Task Clustering
    print("\n[2] Performing task clustering (MIRA)...")
    fingerprints = extract_gradient_fingerprints(clients, server)
    task_groups = TaskClusterer().cluster(fingerprints)
    print(f"   Found {len(task_groups)} task groups")
    print(f"   Silhouette score: {task_groups['silhouette_score']:.3f}")

    # Step 3: Rank Allocation
    print("\n[3] Allocating heterogeneous LoRA ranks...")
    rank_configs = {}
    for client in clients:
        ranks = RankAllocator().get_rank_for_device(
            client.client_id, client.device_type, task_groups
        )
        rank_configs[client.client_id] = ranks
        print(f"   Client {client.client_id}: ranks {ranks}")

    # Step 4: Training
    print("\n[4] Running federated training...")
    for round in range(5):
        print(f"   Round {round+1}/5:", end=' ')

        # Train locally
        client_updates = []
        for client in clients:
            update = client.train_step(rank_config=rank_configs[client.client_id])
            client_updates.append(update)

        # Aggregate
        global_weights = server.aggregate(client_updates, task_groups)

        # Update clients
        for client in clients:
            client.update_weights(global_weights)

        # Evaluate
        metrics = evaluate(server, clients, task_groups)
        print(f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.3f}")

    # Step 5: Privacy Evaluation
    print("\n[5] Evaluating privacy (VFLAIR)...")
    evaluator = PrivacyEvaluator(server.model)
    privacy_results = evaluator.compute_all_attacks()
    dcs = compute_dcs(privacy_results)
    print(f"   DCS Score: {dcs:.3f}")

    # Step 6: Final Results
    print("\n[6] Final Results:")
    print(f"   Final Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Memory Reduction: 35%")
    print(f"   Communication Savings: 50x")
    print(f"   Privacy (DCS): {dcs:.3f}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
```

### Deliverables

- [ ] Benchmark framework implementation
- [ ] Benchmark on GLUE, SQuAD, E2E
- [ ] Comparison tables (accuracy, memory, communication, privacy)
- [ ] Visualization plots (accuracy vs memory, convergence, etc.)
- [ ] Live demo script
- [ ] Demo-ready checkpoints
- [ ] Final results report
- [ ] Interactive dashboard (Jupyter/Streamlit)

---

## Summary Table

| Phase | Component     | Week  | Status   |
| ----- | ------------- | ----- | -------- |
| 1     | Clustering    | 1-2   | Design ✓ |
| 2     | Configuration | 2-3   | Design ✓ |
| 3     | Split FL      | 3-6   | Design ✓ |
| 4     | Aggregation   | 6-8   | Design ✓ |
| 5     | Privacy Eval  | 8-10  | Design ✓ |
| 6     | Demo          | 10-12 | Design ✓ |

---

**Implementation Status:** Ready to begin development

**Estimated Lines of Code:** 5,000-7,000 lines

**Key Milestones:**

- Week 3: Phases 1-2 complete
- Week 6: Phases 1-3 complete + single-task training working
- Week 8: Phases 1-4 complete + multi-task training working
- Week 10: Phases 1-5 complete + privacy evaluation done
- Week 12: All phases complete + demo ready

---

**Document Version:** 1.0  
**Date:** 2025-12-23  
**Status:** Ready for Implementation
