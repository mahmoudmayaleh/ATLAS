# 🔧 Per-Layer Rank Allocation Fix

## 🚨 **The Problem**

Despite computing importance scores correctly (layer_3=0.272 vs layer_5=0.007, a **36× difference**), all layers got **uniform ranks**:

```
Client 0 (2GB): [8, 8, 8, 8, 8, 8]  ❌ All same!
Client 7 (16GB): [64, 64, 64, 64, 64, 64]  ❌ All same!
```

Expected heterogeneous allocation:

```
Client 0 (2GB): [4, 8, 16, 16, 8, 4]  ✅ Varies by importance!
```

---

## 🔍 **Root Cause: Flawed Greedy Algorithm**

### Old Algorithm (BROKEN)

```python
# 1. Start all layers at min_rank=8
ranks = [8, 8, 8, 8, 8, 8]

# 2. Sort layers by importance
# [layer_3 (0.272), layer_2 (0.226), layer_1 (0.181), ...]

# 3. For each layer (in importance order):
for layer_idx, importance in sorted_layers:
    # Try to upgrade to highest rank that fits
    for rank in [64, 32, 16, 8, 4]:  # descending
        if current_memory + memory(rank) <= budget:
            ranks[layer_idx] = rank
            break
```

### Why It Failed

1. **Layer 3** (most important): upgrades from 8→64 ✅ Fits!
2. **Layer 2**: upgrades from 8→64 ✅ Still fits!
3. **Layer 1**: upgrades from 8→64 ✅ Still fits!
4. **...all layers**: upgrade to 64 ✅ All fit!

**Result**: If budget allows upgrading one layer to max rank, it allows upgrading ALL layers to max rank → uniform allocation!

---

## ✅ **The Fix: Budget-Proportional Allocation**

### New Algorithm (CORRECT)

```python
# 1. Find maximum uniform rank (baseline)
uniform_rank = max(r for r in [4,8,16,32,64]
                   if n_layers * memory(r) <= budget)
# Example: For 2GB with 6 layers, uniform_rank = 8

# 2. Define total rank budget
total_budget = n_layers × uniform_rank
# Example: 6 × 8 = 48 total ranks

# 3. Distribute proportionally to importance
for layer_idx, importance in layers:
    target_rank = importance × total_budget
    ranks[layer_idx] = round_to_valid(target_rank)

# Example for 6 layers with importances:
# layer_0: 0.136 × 48 = 6.5 → rank 8
# layer_1: 0.181 × 48 = 8.7 → rank 8
# layer_2: 0.226 × 48 = 10.8 → rank 16
# layer_3: 0.272 × 48 = 13.0 → rank 16  ⭐ Highest!
# layer_4: 0.035 × 48 = 1.7 → rank 4
# layer_5: 0.007 × 48 = 0.3 → rank 4   ⭐ Lowest!

# 4. Adjust if over budget (downgrade least important layers)
```

---

## 📊 **Expected Results After Fix**

### Small Devices (2GB CPU)

```python
Importance:  [0.136, 0.181, 0.226, 0.272, 0.035, 0.007]
Old ranks:   [8,     8,     8,     8,     8,     8    ]  ❌ Uniform
New ranks:   [4,     8,     16,    16,    4,     4    ]  ✅ Heterogeneous!
```

### Medium Devices (8GB Laptop)

```python
Importance:  [0.136, 0.181, 0.226, 0.272, 0.035, 0.007]
Old ranks:   [32,    32,    32,    32,    32,    32   ]  ❌ Uniform
New ranks:   [16,    32,    64,    64,    16,    8    ]  ✅ Varies 8-64!
```

### Large Devices (16GB GPU)

```python
Importance:  [0.136, 0.181, 0.226, 0.272, 0.035, 0.007]
Old ranks:   [64,    64,    64,    64,    64,    64   ]  ❌ Uniform
New ranks:   [32,    64,    128,   128,   32,    16   ]  ✅ Varies 16-128!
```

---

## 🔬 **Why This Fix Works**

### **1. Fixed Total Budget**

Instead of "upgrade until out of memory", we define:

```
Total budget = n_layers × best_uniform_rank
```

This ensures heterogeneous allocation uses **same memory** as uniform allocation.

### **2. Proportional Distribution**

High-importance layers get **proportionally more** of the budget:

```
layer_3 (imp=0.272): 27.2% of budget → high rank
layer_5 (imp=0.007): 0.7% of budget → low rank
```

### **3. Respects Memory Constraint**

After proportional allocation, we validate:

```python
if memory(ranks) > budget:
    downgrade_least_important_layers()
```

---

## 🎯 **Alignment with HSplitLoRA & MIRA**

| Paper          | Method                                                       | ATLAS Implementation          |
| -------------- | ------------------------------------------------------------ | ----------------------------- |
| **HSplitLoRA** | "Greedy allocation: assign higher ranks to important layers" | ✅ Proportional to importance |
| **MIRA**       | "Per-layer importance from gradient norms"                   | ✅ Computed correctly         |
| **MIRA**       | "Memory-constrained heterogeneous allocation"                | ✅ Budget-proportional        |

---

## 📝 **Implementation Details**

### File Changed

[src/phase2_configuration.py](../src/phase2_configuration.py) - `RankAllocator.allocate_ranks()`

### Key Changes

1. Replaced incremental greedy with budget-proportional
2. Added total rank budget computation
3. Added downgrade logic for over-budget scenarios
4. Enhanced logging to show importance vs rank

### Verification Command

```bash
python experiments/atlas_integrated.py --quick --num-rounds 3
```

### Look For

```
[Phase 2] Sample importance scores:
  layer_0: 0.136, layer_1: 0.181, layer_2: 0.226
  layer_3: 0.272 ⭐ Highest
  layer_4: 0.035, layer_5: 0.007 ⭐ Lowest

[Phase 2] Client 0 (cpu_2gb): ranks=[4, 8, 16, 16, 4, 4]  ✅ Heterogeneous!
[Phase 2] Client 7 (gpu_16gb): ranks=[32, 64, 128, 128, 32, 16]  ✅ Varies!
```

---

## 🚀 **Testing the Fix**

### Quick Test (3 rounds)

```bash
python experiments/atlas_integrated.py --quick --num-rounds 3
```

### Expected Improvements

1. **Heterogeneous ranks across layers** (not uniform)
2. **Higher ranks for important layers** (layer_2, layer_3)
3. **Lower ranks for less important layers** (layer_4, layer_5)
4. **Same total memory** as before (no increase in communication cost)

### Success Criteria

- ✅ At least 3 different rank values per client
- ✅ Highest rank assigned to layer_3 (most important)
- ✅ Lowest rank assigned to layer_5 (least important)
- ✅ Memory usage ≤ budget for all clients

---

## 🎓 **Conceptual Explanation**

### Analogy: Budget Allocation

Imagine you have $100 to spend on 6 employees based on performance:

**Old Method (Broken)**:

1. Start: everyone gets $10 (min salary)
2. Top performer: can I give them $100? Yes → $100
3. 2nd best: can I give them $100? Yes → $100
4. ...everyone gets $100! 💸 Over budget!

**New Method (Correct)**:

1. Total budget: $100
2. Top performer (30% importance): gets $30
3. 2nd best (25% importance): gets $25
4. ...lowest (5% importance): gets $5
5. Sum = $100 ✅ On budget!

Same principle applies to rank allocation:

- **Total rank budget** = 48 (6 layers × 8 uniform rank)
- **Layer 3** (27.2% importance) → 13 ranks → round to 16
- **Layer 5** (0.7% importance) → 0.3 ranks → round to 4

---

## ✅ **Commit**

```bash
git commit -m "Fix per-layer rank allocation: Replace broken greedy with budget-proportional algorithm"
```

---

## 📖 **Further Reading**

- HSplitLoRA paper: Section 4.2 "Importance-Aware Rank Allocation"
- MIRA paper: Section 3.3 "Per-Layer Gradient Norms"
- [PUBLICATION_READY_FIXES.md](PUBLICATION_READY_FIXES.md): Overview of all fixes
