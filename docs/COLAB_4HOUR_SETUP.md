# ATLAS 4-Hour Colab Setup for Publication Runs

**Last Updated**: February 5, 2026

## Overview

This document describes the modifications made to ATLAS for running comprehensive experiments within a 4-hour Colab session limit, suitable for paper publication.

## Key Changes

### 1. Checkpoint Strategy Change

**Before**:

- Saved checkpoints every 5 rounds during training
- Created many intermediate checkpoint files
- Useful for resuming interrupted multi-hour runs

**After**:

- Disabled per-round checkpointing (`save_every = 999999`)
- Only saves final results after each complete experiment
- Saves results immediately when experiment completes
- Much cleaner and Colab-optimized

**Rationale**:

- Colab sessions can disconnect unpredictably
- If a run gets interrupted, it's better to restart the experiment than resume from a checkpoint
- Saves disk space and reduces I/O overhead
- Results are saved after each complete run, so you don't lose data between experiments

### 2. Optimized Experimental Parameters

**Full Mode Configuration**:

```python
num_rounds: 18          # Reduced from 30 (sufficient for convergence)
max_samples: 1500       # Reduced from 2000 (balanced quality/speed)
clients: 9              # 3 tasks × 3 clients each
local_epochs: 2         # Kept moderate
batch_size: 16          # Optimal for T4 GPU
```

**Time Estimates**:

- ATLAS Full: ~75 minutes
- FedAvg Cluster: ~65 minutes
- Local Only: ~50 minutes
- Lambda variants: ~70 minutes each

**Total for all experiments**: ~3.5-4 hours

### 3. Comprehensive Batch Runner

Added a new cell in `atlas_colab.ipynb` that:

**Features**:

- Runs multiple experiments sequentially
- Tracks elapsed time and remaining time
- Estimates time for each experiment
- Skips experiments if insufficient time remains
- Saves results after each experiment completes
- Provides comprehensive summary at the end

**Experiments Included**:

1. **ATLAS Full** - Complete 4-phase pipeline
2. **FedAvg per Cluster** - Ablation: no Laplacian (Phase 1-3 only)
3. **Local Only** - Ablation: no aggregation baseline
4. **Lambda 0.01** - Low regularization
5. **Lambda 0.5** - High regularization

**Usage**:

```python
# Just run the comprehensive experiment runner cell
# It will automatically run all experiments and track time
```

### 4. Enhanced Results Analysis

Added comprehensive results comparison cell that:

**Automatic Analysis**:

- Loads all experiment results
- Creates comparison table (CSV)
- Generates 4-panel visualization:
  1. Average accuracy with error bars
  2. Communication overhead
  3. Training time
  4. Convergence curves

**Metrics Compared**:

- Average accuracy
- Standard deviation (personalization quality)
- Min/Max accuracy across clients
- Total communication cost
- Total training time
- Number of rounds

**Output Files**:

- `results/experiment_comparison.csv`
- `results/experiment_comparison.png`

### 5. Individual Experiment Cells

For selective/debugging runs, added individual cells for:

```bash
# ATLAS Full
!python experiments/atlas_integrated.py --mode full --rounds 18 --ablation atlas

# FedAvg Cluster
!python experiments/atlas_integrated.py --mode full --rounds 18 --ablation fedavg_cluster

# Local Only
!python experiments/atlas_integrated.py --mode full --rounds 18 --ablation local_only

# Lambda Sweep
!python experiments/atlas_integrated.py --mode full --rounds 15 --lambda-sweep
```

## Modified Files

### 1. `experiments/atlas_integrated.py`

**Changes**:

```python
# Line 124: Disabled per-round checkpointing
save_every: int = 999999  # Was: 5

# Lines 1230-1260: Updated experiment configurations
# Quick mode: 10 rounds, 500 samples (~15-20 min)
# Full mode: 18 rounds, 1500 samples (~60-80 min per run)
```

### 2. `atlas_colab.ipynb`

**New Cells Added**:

1. **Top markdown** - Updated introduction explaining 4-hour setup
2. **Comprehensive batch runner** - Python cell that runs all experiments with time tracking
3. **Individual experiment cells** - Cells for running experiments one at a time
4. **Results comparison cell** - Automatic analysis and visualization

**Modified Cells**:

- Disabled old "quick mode" cell (replaced with batch runner)
- Updated markdown descriptions

## Usage Instructions

### Option 1: Run All Experiments (Recommended)

1. **Setup** (run once):

   ```python
   # Install dependencies
   !pip install -q torch transformers datasets peft scikit-learn scipy numpy

   # Clone/update repo
   !git clone https://github.com/mahmoudmayaleh/ATLAS.git
   %cd ATLAS
   !git pull origin main
   ```

2. **Run comprehensive suite**:
   - Find the cell with "Comprehensive Experiment Runner"
   - Run it (will take ~3.5-4 hours)
   - Results are saved after each experiment completes

3. **Analyze results**:
   - Run the "Results Analysis & Comparison" cell
   - View comparison table and plots

### Option 2: Run Individual Experiments

Useful for debugging or selective runs:

```python
# Run just ATLAS
!python experiments/atlas_integrated.py --mode full --rounds 18 --ablation atlas

# Run just one ablation
!python experiments/atlas_integrated.py --mode full --rounds 18 --ablation local_only
```

### Option 3: Quick Testing

Before running long experiments:

```python
# Quick 15-minute test
!python experiments/atlas_integrated.py --mode quick
```

## Expected Results Structure

After running experiments, you'll have:

```
results/
├── atlas_integrated_full_atlas.json              # ATLAS full results
├── atlas_integrated_full_fedavg_cluster.json     # FedAvg cluster results
├── atlas_integrated_full_local_only.json         # Local only results
├── experiment_comparison.csv                      # Comparison table
├── experiment_comparison.png                      # Comparison plots
└── experiment_suite_summary_YYYYMMDD_HHMMSS.json # Batch run summary
```

## Time Management Strategy

The batch runner implements smart time management:

1. **Estimates time** for each experiment before starting
2. **Tracks elapsed time** and remaining time
3. **Skips experiments** if insufficient time (with 20% buffer)
4. **Stops at 3h 55min** to ensure results are saved
5. **Saves progress** after each experiment completes

**Example Output**:

```
Total time: 3.75 hours (225 minutes)

Results Summary:
  ATLAS Full Pipeline: success (Time: 78.2 min)
  FedAvg per Cluster: success (Time: 67.5 min)
  Local Training Only: success (Time: 51.3 min)
  ATLAS Lambda=0.01: skipped (insufficient time)
  ATLAS Lambda=0.5: skipped (insufficient time)
```

## Tips for Successful 4-Hour Runs

### 1. Use Runtime Management

```python
# Check GPU assignment
!nvidia-smi

# Monitor memory usage
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
```

### 2. Save Intermediate Artifacts

The batch runner automatically saves:

- Results after each experiment
- Summary JSON with timing info
- Status for each experiment (success/failed/timeout/skipped)

### 3. Handle Disconnections

If Colab disconnects:

- Results from completed experiments are saved in `results/`
- Check `experiment_suite_summary_*.json` for what completed
- Re-run only missing experiments using individual cells

### 4. Prioritize Experiments

If short on time, prioritize:

1. ATLAS Full (most important for paper)
2. FedAvg Cluster (key ablation)
3. Local Only (baseline)
4. Lambda variants (supplementary)

## Troubleshooting

### "Out of Memory" Errors

Reduce batch size or samples:

```python
# In experiments/atlas_integrated.py
batch_size=12,              # Was: 16
max_samples_per_client=1000 # Was: 1500
```

### "Timeout" on Experiments

Reduce rounds:

```python
--rounds 15  # Instead of 18
```

### "No Results Found" Error

Check if experiments completed:

```bash
!ls -lh results/
```

## Publication-Ready Outputs

After successful runs, you'll have:

### 1. Performance Comparison

- Table with accuracy, communication cost, time
- Statistical comparison (mean ± std)
- Min/max accuracy (personalization quality)

### 2. Visualizations

- 4-panel comparison plot
- Convergence curves
- Bar charts with error bars

### 3. Raw Data

- JSON files with all metrics per round
- Per-client accuracies
- Communication logs
- Clustering results
- Device configurations

### 4. Ablation Studies

- ATLAS vs FedAvg vs Local Only
- With/without Laplacian regularization
- Different lambda values

## Next Steps

After running experiments:

1. **Download results**:

   ```python
   !zip -r atlas_results.zip results/
   from google.colab import files
   files.download('atlas_results.zip')
   ```

2. **Analyze locally**: Use Jupyter/Python to create publication plots

3. **Run additional experiments** if needed (lambda sweep, etc.)

## Contact & Support

For issues or questions:

- Check experiment logs in stdout
- Review `experiment_suite_summary_*.json`
- Examine individual result JSON files

---

**Happy Experimenting!**
