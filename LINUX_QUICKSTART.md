# Quick Start for Linux GPU Server

## Setup (One-time)

```bash
# Pull latest code
git pull

# Make scripts executable
chmod +x gpu0_seed42.sh gpu1_seed123.sh gpu2_seed456.sh

# Verify conda environment is active
conda activate atlas_env  # or your environment name
```

## Run Session 1 (Rounds 1-3) - ATLAS Method

Open 3 separate terminals (or use tmux/screen) and run:

### Terminal 1 - GPU 0:
```bash
export CUDA_VISIBLE_DEVICES=0
./gpu0_seed42.sh 1 atlas
```

### Terminal 2 - GPU 1:
```bash
export CUDA_VISIBLE_DEVICES=1
./gpu1_seed123.sh 1 atlas
```

### Terminal 3 - GPU 2:
```bash
export CUDA_VISIBLE_DEVICES=2
./gpu2_seed456.sh 1 atlas
```

**Expected time:** ~4 hours per GPU

## Run Session 1 - Other Methods

After ATLAS completes, run the same session for other methods:

```bash
# GPU 0
export CUDA_VISIBLE_DEVICES=0
./gpu0_seed42.sh 1 fedavg_cluster
./gpu0_seed42.sh 1 local_only

# GPU 1
export CUDA_VISIBLE_DEVICES=1
./gpu1_seed123.sh 1 fedavg_cluster
./gpu1_seed123.sh 1 local_only

# GPU 2
export CUDA_VISIBLE_DEVICES=2
./gpu2_seed456.sh 1 fedavg_cluster
./gpu2_seed456.sh 1 local_only
```

## Resume Session 2 (Rounds 4-7)

After Session 1 completes, scripts automatically resume:

```bash
# GPU 0
export CUDA_VISIBLE_DEVICES=0
./gpu0_seed42.sh 2 atlas

# GPU 1
export CUDA_VISIBLE_DEVICES=1
./gpu1_seed123.sh 2 atlas

# GPU 2
export CUDA_VISIBLE_DEVICES=2
./gpu2_seed456.sh 2 atlas
```

Then run `fedavg_cluster` and `local_only` the same way.

**Expected time:** ~5.3 hours per GPU

## Resume Session 3 (Rounds 8-10)

```bash
# GPU 0
export CUDA_VISIBLE_DEVICES=0
./gpu0_seed42.sh 3 atlas

# GPU 1
export CUDA_VISIBLE_DEVICES=1
./gpu1_seed123.sh 3 atlas

# GPU 2
export CUDA_VISIBLE_DEVICES=2
./gpu2_seed456.sh 3 atlas
```

Then run `fedavg_cluster` and `local_only` the same way.

**Expected time:** ~4 hours per GPU

## Using tmux (Recommended for Remote Server)

If your SSH connection might drop:

```bash
# Start tmux session for GPU 0
tmux new -s gpu0
export CUDA_VISIBLE_DEVICES=0
./gpu0_seed42.sh 1 atlas
# Detach: Ctrl+B then D

# Start tmux session for GPU 1
tmux new -s gpu1
export CUDA_VISIBLE_DEVICES=1
./gpu1_seed123.sh 1 atlas
# Detach: Ctrl+B then D

# Start tmux session for GPU 2
tmux new -s gpu2
export CUDA_VISIBLE_DEVICES=2
./gpu2_seed456.sh 1 atlas
# Detach: Ctrl+B then D

# Reattach later to check progress
tmux attach -t gpu0
tmux attach -t gpu1
tmux attach -t gpu2

# List all tmux sessions
tmux ls
```

## Check Results

Results are saved to:
- `results/atlas_integrated_quick_atlas_seed42.json`
- `results/atlas_integrated_quick_fedavg_cluster_seed123.json`
- etc.

Checkpoints are saved to:
- `checkpoints/atlas_atlas_seed42_round_3.pkl`
- `checkpoints/atlas_atlas_seed42_round_7.pkl`
- etc.

## Troubleshooting

**Error: Permission denied**
```bash
chmod +x gpu0_seed42.sh gpu1_seed123.sh gpu2_seed456.sh
```

**Error: Checkpoint not found**
→ You skipped a session. Run the previous session first.

**Out of memory**
→ Edit the scripts and reduce `BATCH_SIZE=8` to `BATCH_SIZE=4`

**Wrong GPU used**
→ Double-check `export CUDA_VISIBLE_DEVICES=X` before running script
