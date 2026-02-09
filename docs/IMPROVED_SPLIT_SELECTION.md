# Improved Split Selection Integration

**Date**: February 9, 2026  
**Status**: ✅ Integrated into Phase 3

## Overview

The improved split selection module (`src/improved_split_selection.py`) has been integrated into `src/phase3_split_fl.py` to provide adaptive, device-aware split point selection for federated learning with split models.

## Key Features

### 1. Multi-Factor Scoring

The improved selector evaluates split points based on:

- **Memory constraints** (35% weight): Ensures split fits in client device memory
- **Communication cost** (30% weight): Minimizes activation transfer size
- **Layer importance** (25% weight): Keeps important layers on client for personalization
- **Workload balance** (10% weight): Balances client/server computation

### 2. Adaptive Selection

- **With device profile**: Uses `ImprovedSplitSelector` for optimal split based on device capabilities
- **Without device profile**: Falls back to heuristic-based selection (BERT→6, GPT-2→6, LLaMA-7B→16)

### 3. Task-Specific Optimization

- Supports task-specific split points for heterogeneous multi-task FL
- Uses gradient fingerprints to compute layer importance per task

## API Changes

### `get_split_point(model_name, device_profile=None)`

**Before**:

```python
split = get_split_point('distilbert-base-uncased')  # Returns 6 (heuristic)
```

**After** (backward compatible):

```python
# Heuristic (unchanged behavior)
split = get_split_point('distilbert-base-uncased')  # Returns 6

# Adaptive (new)
device_profile = {'memory_mb': 4096, 'device_type': 'tablet'}
split = get_split_point('distilbert-base-uncased', device_profile)  # Returns optimal split (e.g., 3)
```

### `SplitClient._compute_split_point()`

Updated to use `ImprovedSplitSelector` when:

1. `IMPROVED_SPLIT_AVAILABLE = True` (import successful)
2. `device_profile` is provided
3. No exceptions during selector initialization

Falls back to legacy method if improved selector unavailable or fails.

## Integration Points

### Files Modified

1. **`src/phase3_split_fl.py`**:
   - Added import for `ImprovedSplitSelector`
   - Updated `_compute_split_point()` method
   - Updated `get_split_point()` function

2. **`tests/test_phase3.py`**:
   - Added `test_get_split_point_with_device_profile()` test

### Dependent Files (No Changes Required)

- `experiments/atlas_integrated.py`: Uses `SplitClient` (internal logic improved automatically)
- `experiments/run_experiments.py`: Uses `SplitClient` and `SplitServer` (no API changes)
- Tests remain backward compatible

## Example Usage

### Basic Usage (Legacy)

```python
from src.phase3_split_fl import get_split_point

# Simple heuristic
split = get_split_point('gpt2')  # 6
split = get_split_point('llama-7b')  # 16
```

### Adaptive Usage (New)

```python
from src.phase3_split_fl import get_split_point

# Device-aware adaptive split
device_profile = {
    'memory_mb': 2048,
    'device_type': 'mobile'
}

split = get_split_point('distilbert-base-uncased', device_profile=device_profile)
# Output: [SPLIT] Client 0: Optimal split = Layer 3/6
#         Score: 0.920 (mem=1.00, comm=0.73, imp=1.00)
```

### SplitClient Automatic Integration

```python
from src.phase3_split_fl import SplitClient

# Client will automatically use improved selector if device_profile provided
client = SplitClient(
    client_id=0,
    model_name='distilbert-base-uncased',
    split_layer=None,  # Will compute optimal split
    device_profile={'memory_mb': 4096},  # Triggers improved selector
    device='cuda'
)

print(f"Computed split: {client.split_layer}")
```

## Validation

### Import Check

```bash
python -c "from src.phase3_split_fl import IMPROVED_SPLIT_AVAILABLE; print(f'Available: {IMPROVED_SPLIT_AVAILABLE}')"
# Output: Available: True
```

### Split Comparison

```bash
python -c "from src.phase3_split_fl import get_split_point; \
profile = {'memory_mb': 4096}; \
print(f'With profile: {get_split_point(\"distilbert-base-uncased\", profile)}'); \
print(f'Without: {get_split_point(\"distilbert-base-uncased\")}')"

# Output:
# [SPLIT] Client 0: Optimal split = Layer 3/6
#         Score: 0.920 (mem=1.00, comm=0.73, imp=1.00)
# With profile: 3
# Without: 6
```

### Run Tests

```bash
cd c:\Users\Hp\Downloads\Advanced_project\ATLAS
python -m pytest tests/test_phase3.py::TestUtilityFunctions::test_get_split_point_with_device_profile -v
```

## Technical Details

### Scoring Formula

For each candidate split point `s`:

```
total_score = 0.35 * memory_score(s)
            + 0.30 * comm_score(s)
            + 0.25 * importance_score(s)
            + 0.10 * balance_score(s)
```

Where:

- **memory_score**: `exp(-(memory_ratio - 1) * 3)` if over budget, else 1.0
- **comm_score**: Normalized transfer time penalty + split ratio bonus
- **importance_score**: Proportion of important layers on client
- **balance_score**: Distance from 50/50 client/server split

### Memory Estimation

```
total_memory = base_model_memory
             + lora_adapter_memory
             + activation_memory

base_model_memory = split * 50 MB/layer
lora_adapter_memory = split * (2 * hidden_size * rank * bytes_per_param)
activation_memory = batch_size * seq_len * hidden_size * bytes_per_param
```

### Communication Cost

```
activation_size = batch_size * seq_len * hidden_size * bytes_per_param * compression_ratio
transfer_time = (2 * activation_size * 8) / bandwidth_mbps  # Upload + download
```

## LLaMA-7B Support

### Memory Requirements

For LLaMA-7B (32 layers, hidden=4096):

- **Full model**: ~28 GB (fp32) or ~14 GB (fp16)
- **With optimal split (layer 16)**:
  - Client: ~7 GB (16 layers + LoRA)
  - Server: ~7 GB (16 layers)
- **LoRA memory** (rank=16): ~4 MB per layer

### Recommended Device Profiles

```python
# Mobile/Edge device
mobile_profile = {'memory_mb': 2048, 'device_type': 'mobile'}
# → Split at layer 8-10 (keep fewer layers on client)

# Tablet
tablet_profile = {'memory_mb': 4096, 'device_type': 'tablet'}
# → Split at layer 12-14

# Laptop/Desktop
laptop_profile = {'memory_mb': 8192, 'device_type': 'laptop'}
# → Split at layer 16-18

# High-end GPU
gpu_profile = {'memory_mb': 16384, 'device_type': 'gpu_16gb'}
# → Split at layer 20-24 (more on client for personalization)
```

## Benefits

1. **Better memory utilization**: Adapts split to device constraints
2. **Reduced communication**: Balances memory vs communication tradeoff
3. **Improved personalization**: Keeps important layers on client when possible
4. **Backward compatible**: Existing code works without modifications
5. **Automatic fallback**: Gracefully degrades to heuristics if improved selector unavailable

## Next Steps

To leverage improved split selection in your experiments:

1. **Provide device profiles** when creating `SplitClient` instances
2. **Pass fingerprints** from Phase 1 clustering for task-specific importance
3. **Tune bandwidth** parameter if network conditions differ from default (10 Mbps)
4. **Enable compression** by setting `compression_ratio < 1.0` in selector

## References

- **SplitLoRA**: Optimal memory-communication tradeoff
- **HSplitLoRA**: Heterogeneous device-aware splitting
- **VFLAIR-LLM**: Vertical federated learning for LLMs
- Phase 1: Task clustering via gradient fingerprints
- Phase 2: Device-aware rank allocation
