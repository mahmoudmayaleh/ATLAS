# ATLAS Literature-Grounded Improvements

**Date:** February 3, 2026  
**Status:** Implemented and ready for experimentation

## Overview

Based on detailed analysis of your Round-15 results showing weak clustering (silhouette ≈ 0.01), underutilized heterogeneity, and sub-optimal MIRA regularization, we have implemented comprehensive, literature-grounded improvements across all 4 phases of ATLAS.

---

## Phase 1: Task Clustering

### Issues Identified

- Silhouette score of 0.011 indicates essentially no cluster structure
- Too few gradient samples (10 batches) led to high noise
- Fingerprint extraction not following design spec (last 2 layers, per-layer norm, PCA to 64D)
- Fixed k=5 not optimal for 3 tasks with 9 clients

### Improvements Implemented

#### 1.1 Strengthened Gradient Fingerprinting

**File:** `src/phase1_clustering.py`, `experiments/atlas_integrated.py`

- **Exact last-2-layer extraction**: Properly identifies last 2 transformer blocks:
  - DistilBERT: `transformer.layer.4`, `transformer.layer.5`, `classifier`
  - BERT: `encoder.layer.10`, `encoder.layer.11`, `classifier`
- **Increased samples**: 64 batches (up from 10) for better signal-to-noise ratio

  ```python
  for batch_idx, batch in enumerate(dataloader):
      if batch_idx >= 64:  # Increased from 10
          break
  ```

- **Per-layer L2 normalization**: Each layer's gradient is normalized before concatenation

  ```python
  if self.layer_normalize:
      norm = np.linalg.norm(layer_flat)
      if norm > 1e-8:
          layer_flat = layer_flat / norm
  ```

- **Variance explained reporting**: Warns if explained variance < 0.7
  ```python
  if total_variance < 0.7:
      warnings.warn(f"Low explained variance ({total_variance:.2f}). "
                    f"Fingerprints may have weak signal.")
  ```

#### 1.2 Multi-Metric k-Selection

**File:** `src/phase1_clustering.py`

Already implemented! Uses weighted combination of:

- **Silhouette** (maximize, range [-1, 1])
- **Davies-Bouldin** (minimize, range [0, ∞))
- **Calinski-Harabasz** (maximize, range [0, ∞))

```python
def _compute_combined_score(self, fingerprints, labels, kmeans):
    sil = silhouette_score(fingerprints, labels)
    db = davies_bouldin_score(fingerprints, labels)
    ch = calinski_harabasz_score(fingerprints, labels)

    # Normalize and combine
    sil_norm = (sil + 1) / 2
    db_norm = 1 / (1 + db)
    ch_norm = ch / (ch + 1000)

    combined = 0.5 * sil_norm + 0.3 * db_norm + 0.2 * ch_norm
    return combined
```

#### 1.3 Cluster-Task Alignment Validation

**File:** `experiments/atlas_integrated.py`

Added sanity check that prints cluster→task mapping with purity metric:

```python
# Compute purity: fraction of clients belonging to dominant task
cluster_task_purity = {}
for cluster_id in sorted(set(labels)):
    tasks_in_cluster = [client.task_name for client in cluster_clients]
    task_counts = Counter(tasks_in_cluster)
    dominant_task = task_counts.most_common(1)[0][0]
    purity = task_counts[dominant_task] / len(cluster_clients)
    cluster_task_purity[cluster_id] = purity

avg_purity = np.mean(list(cluster_task_purity.values()))
if avg_purity < 0.8:
    warnings.warn(f"Low cluster-task alignment (purity={avg_purity:.2f})")
```

---

## Phase 2: Heterogeneous LoRA Rank Allocation

### Issues Identified

- Coarse granularity: all layers same rank (e.g., `[8,8,8,8,8,8]` or `[32,32,32,32,32,32]`)
- Not using layer importance from gradient norms
- Not coupling with cluster statistics (difficult clusters should get more capacity)

### Improvements Implemented

#### 2.1 Greedy Importance-Aware Allocator

**File:** `src/phase2_configuration.py` (RankAllocator class)

Already implemented! Uses HSplitLoRA formulation:

- **Memory constraint**: $\sum_\ell 2 \cdot d \cdot r_\ell \cdot b \leq C_{mem}$
- **Greedy allocation**: Sort layers by importance, assign highest rank that fits budget
- **Rank candidates**: {4, 8, 16, 32, 64}

```python
def allocate_ranks(self, device_profile, importance_scores, n_layers, split_point=None):
    C_mem = self.compute_adapter_memory_budget(device_profile['memory_mb'])
    max_device_rank = max(device_profile['suggested_ranks'])

    # Sort layers by importance (descending)
    layer_importance.sort(key=lambda x: x[1], reverse=True)

    # Greedy allocation
    for layer_idx, importance in layer_importance:
        for rank in reversed(feasible_ranks):
            marginal_memory = new_rank_memory - old_rank_memory
            if current_memory + marginal_memory <= C_mem:
                ranks[layer_idx] = rank
                current_memory += marginal_memory
                break

    return ranks
```

#### 2.2 Cluster-Coupled Rank Allocation

**File:** `experiments/atlas_integrated.py`

Allocates higher ranks to clusters with:

- Higher within-cluster variance (heterogeneity)
- Higher gradient norms (task difficulty)

```python
# Compute cluster complexity score
cluster_stats[cluster_id] = {
    'variance': np.var(cluster_fingerprints).mean(),
    'avg_norm': np.mean([np.linalg.norm(fp) for fp in cluster_fingerprints]),
    'complexity_score': variance * avg_norm
}

# Normalize across clusters
max_complexity = max(stats['complexity_score'] for stats in cluster_stats.values())
normalized_complexity = cluster_stats[cluster_id]['complexity_score'] / max_complexity

# Scale layer importance by cluster complexity
for i in range(6):
    layer_importance = 0.5 + (i / 6.0)  # Depth-based importance
    importance_scores[f'layer_{i}'] = layer_importance * (0.5 + 0.5 * normalized_complexity)
```

---

## Phase 4: MIRA Laplacian Regularization

### Issues Identified

- Not using MIRA's RBF kernel for adjacency
- No literature-grounded formula: $a_{k\ell} = \exp(-\alpha \|f_k - f_\ell\|^2)$
- Lambda (η) not tunable for experimentation
- Using cosine similarity + top-k instead of exponential decay

### Improvements Implemented

#### 4.1 MIRA RBF Adjacency from Fingerprints

**File:** `src/phase4_laplacian.py`

Implemented MIRA's exact formula using Phase 1 fingerprints:

```python
def compute_adjacency_weights(
    task_clusters,
    gradient_fingerprints,  # Dict[client_id, fingerprint]
    method='mira_rbf',
    mira_alpha=1.0
):
    if method == 'mira_rbf' and gradient_fingerprints is not None:
        # Convert fingerprints to array
        fp_array = np.vstack([gradient_fingerprints[cid] for cid in sorted_clients])

        # Compute pairwise L2 distances: ||f_k - f_ℓ||²
        from scipy.spatial.distance import cdist
        pairwise_distances_sq = cdist(fp_array, fp_array, metric='sqeuclidean')

        # MIRA's RBF kernel: a_kℓ = exp(-α ||f_k - f_ℓ||²)
        idx_i = client_to_idx[client_i]
        idx_j = client_to_idx[client_j]
        dist_sq = pairwise_distances_sq[idx_i, idx_j]
        weight = np.exp(-mira_alpha * dist_sq)

    # Normalize: Σ_ℓ a_kℓ = 1
    total_weight = sum(w for _, w in neighbor_weights)
    neighbor_weights = [(j, w / total_weight) for j, w in neighbor_weights]

    return weights
```

**Integration:** Updated training loop to use fingerprints directly:

```python
# experiments/atlas_integrated.py
adjacency_weights = compute_adjacency_weights(
    task_clusters=task_clusters,
    gradient_fingerprints=fingerprints,  # From Phase 1
    method=self.config.laplacian_adjacency_method,  # 'mira_rbf'
    mira_alpha=self.config.mira_alpha  # Tunable α
)
```

#### 4.2 Configurable Lambda (η) Tuning

**File:** `experiments/atlas_integrated.py`

Added tunable regularization strength to ATLASConfig:

```python
@dataclass
class ATLASConfig:
    # ...
    # Phase 4: Laplacian regularization (MIRA)
    eta: float = 0.1  # Regularization strength λ (tune: {0.01, 0.1, 0.5})
    laplacian_adjacency_method: str = 'mira_rbf'  # RECOMMENDED
    mira_alpha: float = 1.0  # RBF kernel bandwidth
```

**Recommended sweep:** {0.01, 0.1, 0.5, 1.0}

- **Low η (0.01)**: More local training, high personalization
- **Mid η (0.1)**: Balanced personalization + convergence
- **High η (0.5+)**: Stronger regularization, approaching global model

---

## Phase 3: Improvements

### 3.1 Strict Per-Cluster Aggregation

Already implemented: FedAvg only within clusters, no cross-cluster averaging.

### 3.2 Memory Validation

Added memory constraint validation and reporting:

```python
is_valid, adapter_mb = self.rank_allocator.validate_memory_constraint(
    lora_ranks, device_profile
)

device_configs[client_id] = {
    'lora_ranks': lora_ranks,
    'memory_valid': is_valid,
    'adapter_memory_mb': adapter_mb
}
```

---

## Notebook Enhancements

### New Experiment Cells

**File:** `atlas_colab.ipynb`

#### 1. Oracle Clustering Baseline

Test with perfect (task-based) clustering to isolate Phase 2-4 improvements.

#### 2. Lambda (η) Sweep

Automated sweep across {0.01, 0.1, 0.5} to find optimal regularization strength.

#### 3. Cluster Quality Metrics Visualization

- Per-cluster accuracy over rounds
- Clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)

#### 4. MIRA Adjacency Heatmap

Visualize RBF kernel weights $a_{k\ell} = \exp(-\alpha \|f_k - f_\ell\|^2)$

- Raw adjacency matrix
- Row-normalized (stochastic) matrix
- Cluster boundaries overlaid

#### 5. Rank Allocation Visualization

- Per-client min/avg/max ranks
- Rank vs device memory (colored by cluster)

---

## Expected Improvements

### Phase 1: Clustering Quality

- **Before:** Silhouette ≈ 0.01 (no structure)
- **After:** Expected 0.3-0.6 (moderate-to-good structure)
- **Validation:** Purity metric should be >0.8 for GLUE tasks

### Phase 2: Rank Utilization

- **Before:** Uniform ranks across all layers
- **After:** Heterogeneous ranks based on layer importance and cluster difficulty
- **Example:** Client in difficult cluster (CoLA) with 8GB device:
  - Before: `[32, 32, 32, 32, 32, 32]` (192 total rank)
  - After: `[8, 16, 16, 32, 32, 32]` (136 total, optimized)

### Phase 4: Personalization

- **Before:** Cosine similarity + top-k neighbors
- **After:** MIRA's RBF kernel with exponential decay
- **Benefit:** Smoother gradient-based regularization, better alignment with task similarity

### Overall Performance

- **Cluster-task alignment:** >80% purity
- **Per-client accuracy:** Higher for difficult tasks (MRPC, CoLA)
- **Convergence:** Faster with tuned λ
- **Communication:** More efficient with importance-aware ranks

---

## Usage Instructions

### Quick Experimentation

```bash
cd ATLAS
python experiments/atlas_integrated.py --mode quick --rounds 5
```

### With Custom Lambda

```bash
python experiments/atlas_integrated.py --mode full --rounds 15 \
    --laplacian_eta 0.1 \
    --mira_alpha 1.0
```

### In Colab Notebook

```python
# Load improved notebook
!git pull origin main
# Run cells under "Literature-Grounded Improvements & Experiments"
```

---

## Files Modified

### Core Implementation

1. `src/phase1_clustering.py`
   - Strengthened fingerprint extraction
   - Added variance explained reporting
   - Multi-metric k-selection (already present, verified)

2. `src/phase2_configuration.py`
   - Greedy importance-aware allocator (already present, verified)
   - Memory validation methods (already present, verified)

3. `src/phase4_laplacian.py`
   - Added `mira_rbf` method to `compute_adjacency_weights`
   - RBF kernel: $a_{k\ell} = \exp(-\alpha \|f_k - f_\ell\|^2)$

4. `experiments/atlas_integrated.py`
   - Updated fingerprint extraction (64 batches, last-2-layers, dict format)
   - Enhanced Phase 2 allocation (cluster-coupled importance)
   - Cluster-task alignment validation
   - MIRA adjacency integration with fingerprints
   - Tunable η parameter

### Notebook & Docs

5. `atlas_colab.ipynb`
   - Literature-grounded improvements section
   - Oracle clustering experiment
   - Lambda sweep cell
   - Enhanced visualizations (cluster metrics, adjacency heatmap, rank allocation)

6. `docs/LITERATURE_IMPROVEMENTS.md` (this file)

---

## Next Steps

### 1. Debug Cluster Quality

Run clustering-only experiment to verify Phase 1 improvements:

```python
# In atlas_integrated.py, add early exit after Phase 1
cluster_labels, fingerprints, metrics = self._phase1_clustering()
print(f"Silhouette: {metrics['silhouette_score']:.3f}")
print(f"Cluster purity: {compute_purity(cluster_labels, task_labels):.3f}")
return  # Exit before training
```

### 2. Oracle Clustering Baseline

Compare with perfect clustering:

```python
# Force cluster_labels from task names
cluster_labels = {cid: task_to_cluster[client.task_name]
                  for cid, client in enumerate(clients_data)}
```

### 3. Lambda Sweep

Automated sweep:

```bash
for eta in 0.01 0.1 0.5; do
    python experiments/atlas_integrated.py --mode quick --rounds 10 \
        --laplacian_eta $eta \
        --output results/atlas_eta_${eta}.json
done
```

### 4. Visualization

Run all new visualization cells in notebook to validate:

- Cluster quality metrics
- MIRA adjacency heatmap
- Rank allocation patterns

---

## References

1. **MIRA Paper:** Gradient-based task similarity, RBF kernel adjacency
2. **HSplitLoRA Paper:** Greedy importance-aware rank allocation under memory constraint
3. **FedAvg:** Per-cluster aggregation for task-aware FL
4. **Cluster Validation:** Silhouette + Davies-Bouldin + Calinski-Harabasz multi-metric

---

## Contact

For questions or issues, refer to:

- Main README: `README.md`
- Colab Quickstart: `docs/COLAB_QUICKSTART.md`
- MIRA Visual Explanation: `docs/MIRA_VISUAL_EXPLANATION.md`
