# Multi-GPU Multi-Session Training Instructions

## Setup: 3 GPUs × 3 Seeds × 3 Methods × 3 Sessions

Each GPU handles one seed across all methods. Split 10 rounds into 3 sessions (3+4+3 rounds).

**GPU Assignment:**

- GPU 0 → Seed 42
- GPU 1 → Seed 123
- GPU 2 → Seed 456

**Methods** (run separately): `atlas`, `fedavg_cluster`, `local_only`

**Session Breakdown:**

- Session 1: Rounds 1-3 (~4 hours/method)
- Session 2: Rounds 4-7 (~5.3 hours/method)
- Session 3: Rounds 8-10 (~4 hours/method)

---

## Platform: Choose Your System

**Linux/Mac users:** Use `*.sh` scripts (bash)  
**Windows users:** Use `*.ps1` scripts (PowerShell)

---

## Session 1: Initial Training

### Linux/Mac:

Open 3 terminal sessions (one per GPU) and run:

```bash
# GPU 0 (Terminal 1)
export CUDA_VISIBLE_DEVICES=0
chmod +x gpu0_seed42.sh
./gpu0_seed42.sh 1 atlas

# GPU 1 (Terminal 2)
export CUDA_VISIBLE_DEVICES=1
chmod +x gpu1_seed123.sh
./gpu1_seed123.sh 1 atlas

# GPU 2 (Terminal 3)
export CUDA_VISIBLE_DEVICES=2
chmod +x gpu2_seed456.sh
./gpu2_seed456.sh 1 atlas
```

**Wait for all 3 to complete (~4 hours each).**

Then run `fedavg_cluster`:

```bash
# GPU 0
./gpu0_seed42.sh 1 fedavg_cluster

# GPU 1
./gpu1_seed123.sh 1 fedavg_cluster

# GPU 2
./gpu2_seed456.sh 1 fedavg_cluster
```

Then run `local_only`:

```bash
# GPU 0
./gpu0_seed42.sh 1 local_only

# GPU 1
./gpu1_seed123.sh 1 local_only

# GPU 2
./gpu2_seed456.sh 1 local_only
```

### Windows (PowerShell):

Open 3 PowerShell terminals (one per GPU) and run:

```powershell
# GPU 0 (Terminal 1)
$env:CUDA_VISIBLE_DEVICES="0"
.\gpu0_seed42.ps1 -Session 1 -Method atlas

# GPU 1 (Terminal 2)
$env:CUDA_VISIBLE_DEVICES="1"
.\gpu1_seed123.ps1 -Session 1 -Method atlas

# GPU 2 (Terminal 3)
$env:CUDA_VISIBLE_DEVICES="2"
.\gpu2_seed456.ps1 -Session 1 -Method atlas
```

**Wait for all 3 to complete (~4 hours each).**

Then run `fedavg_cluster`:

```powershell
# GPU 0
.\gpu0_seed42.ps1 -Session 1 -Method fedavg_cluster

# GPU 1
.\gpu1_seed123.ps1 -Session 1 -Method fedavg_cluster

# GPU 2
.\gpu2_seed456.ps1 -Session 1 -Method fedavg_cluster
```

Then run `local_only`:

```powershell
# GPU 0
.\gpu0_seed42.ps1 -Session 1 -Method local_only

# GPU 1
.\gpu1_seed123.ps1 -Session 1 -Method local_only

# GPU 2
.\gpu2_seed456.ps1 -Session 1 -Method local_only
```

---

## Session 2: Resume Training (Rounds 4-7)

After Session 1 completes, start Session 2:

### Linux/Mac:

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

**Wait for all 3 to complete (~5.3 hours each).**

Then run `fedavg_cluster` and `local_only` Session 2 the same way.

### Windows (PowerShell):

```powershell
# GPU 0
$env:CUDA_VISIBLE_DEVICES="0"
.\gpu0_seed42.ps1 -Session 2 -Method atlas

# GPU 1
$env:CUDA_VISIBLE_DEVICES="1"
.\gpu1_seed123.ps1 -Session 2 -Method atlas

# GPU 2
$env:CUDA_VISIBLE_DEVICES="2"
.\gpu2_seed456.ps1 -Session 2 -Method atlas
```

**Wait for all 3 to complete (~5.3 hours each).**

Then run `fedavg_cluster` and `local_only` Session 2 the same way.

---

## Session 3: Final Training (Rounds 8-10)

After Session 2 completes:

### Linux/Mac:

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

**Wait for all 3 to complete (~4 hours each).**

Then run `fedavg_cluster` and `local_only` Session 3.

### Windows (PowerShell):

```powershell
# GPU 0
$env:CUDA_VISIBLE_DEVICES="0"
.\gpu0_seed42.ps1 -Session 3 -Method atlas

# GPU 1
$env:CUDA_VISIBLE_DEVICES="1"
.\gpu1_seed123.ps1 -Session 3 -Method atlas

# GPU 2
$env:CUDA_VISIBLE_DEVICES="2"
.\gpu2_seed456.ps1 -Session 3 -Method atlas
```

**Wait for all 3 to complete (~4 hours each).**

Then run `fedavg_cluster` and `local_only` Session 3.

---

## Quick Run (All Methods, One GPU at a Time)

If running one GPU sequentially instead of parallel:

### Linux/Mac - GPU 0 - All Methods - All Sessions:

```bash
export CUDA_VISIBLE_DEVICES=0

# Session 1
./gpu0_seed42.sh 1 atlas
./gpu0_seed42.sh 1 fedavg_cluster
./gpu0_seed42.sh 1 local_only

# Session 2
./gpu0_seed42.sh 2 atlas
./gpu0_seed42.sh 2 fedavg_cluster
./gpu0_seed42.sh 2 local_only

# Session 3
./gpu0_seed42.sh 3 atlas
./gpu0_seed42.sh 3 fedavg_cluster
./gpu0_seed42.sh 3 local_only
```

Repeat for GPU 1 and GPU 2 with their respective scripts.

### Windows (PowerShell) - GPU 0 - All Methods - All Sessions:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"

# Session 1
.\gpu0_seed42.ps1 -Session 1 -Method atlas
.\gpu0_seed42.ps1 -Session 1 -Method fedavg_cluster
.\gpu0_seed42.ps1 -Session 1 -Method local_only

# Session 2
.\gpu0_seed42.ps1 -Session 2 -Method atlas
.\gpu0_seed42.ps1 -Session 2 -Method fedavg_cluster
.\gpu0_seed42.ps1 -Session 2 -Method local_only

# Session 3
.\gpu0_seed42.ps1 -Session 3 -Method atlas
.\gpu0_seed42.ps1 -Session 3 -Method fedavg_cluster
.\gpu0_seed42.ps1 -Session 3 -Method local_only
```

Repeat for GPU 1 and GPU 2 with their respective scripts.

---

## Checkpoints

Checkpoints are automatically saved after each session:

- `checkpoints/atlas_atlas_seed42_round_3.pkl` (after Session 1)
- `checkpoints/atlas_atlas_seed42_round_7.pkl` (after Session 2)
- `checkpoints/atlas_atlas_seed42_round_10.pkl` (after Session 3, final)

Same pattern for `fedavg_cluster` and `local_only`.

---

## Results

Final results are saved to:

- `results/atlas_integrated_quick_atlas_seed42.json`
- `results/atlas_integrated_quick_fedavg_cluster_seed42.json`
- `results/atlas_integrated_quick_local_only_seed42.json`

Same pattern for seeds 123 and 456.

---

## Troubleshooting

**Error: "Checkpoint not found"**
→ You skipped a session. Run the previous session first.

**Error: GPU out of memory**
→ Reduce `--batch-size` from 8 to 4 in the scripts.

**Need to change GPU assignment:**
Change `$env:CUDA_VISIBLE_DEVICES="X"` before running the script.

---

## Total Time Estimate

- **Per method per seed:** ~13 hours (3 sessions)
- **All 3 methods per seed:** ~39 hours (if sequential)
- **All 3 seeds (parallel on 3 GPUs):** ~39 hours wall time
- **Full experiment (3 seeds × 3 methods):** ~39 hours with 3 GPUs

---

## After Completion

Aggregate results using the statistical runner:

```powershell
python experiments/run_statistical_experiments.py --help
```

Or manually compare the 9 result JSONs (3 seeds × 3 methods).
