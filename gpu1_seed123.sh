#!/bin/bash
# GPU 1 - Seed 123 - Dynamic Model Experiments
# Supports: gpt2, gpt2-xl, qwen-0.5b
# Runs 10 rounds in one shot with all professional metrics

set -euo pipefail
IFS=$'\n\t'


# Detect Python command (python3 or python). Allow override via PYTHON_CMD env.
PYTHON_CMD=${PYTHON_CMD:-}
if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "Error: Python not found. Please activate your conda/venv environment first or set PYTHON_CMD."
        echo "Example: conda activate atlas_env  # or: export PYTHON_CMD=/path/to/python"
        exit 1
    fi
fi

echo "Using Python: $PYTHON_CMD ($(command -v "$PYTHON_CMD"))"

# Parse arguments
MODEL=${1:-gpt2-xl}
METHOD=${2:-atlas}

# GPU selection: default to GPU 1 for this script. Override with GPU_ID env var.
GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

# Optional conda env name to auto-activate (set CONDA_ENV env var), or
# set VENV_PATH to point to a Python venv directory (then PYTHON_CMD will be set).
if [ -n "${CONDA_ENV:-}" ]; then
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)" || true
        conda activate "${CONDA_ENV}" || true
        echo "Activated conda env: ${CONDA_ENV}"
    else
        echo "CONDA_ENV set but conda not found in PATH"
    fi
fi
if [ -n "${VENV_PATH:-}" ]; then
    if [ -f "${VENV_PATH}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${VENV_PATH}/bin/activate"
        echo "Activated venv: ${VENV_PATH}"
    else
        echo "VENV_PATH set but no activate script found at ${VENV_PATH}/bin/activate"
    fi
fi

# Validate arguments (simple check)
VALID_MODELS=("distilbert-base-uncased" "gpt2" "gpt2-xl" "Qwen/Qwen2.5-0.5B")
if [[ ! " ${VALID_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "Warning: Model '${MODEL}' not in the quick-validated list. Proceeding anyway."
fi

if [[ ! "$METHOD" =~ ^(atlas|atlas_no_laplacian|fedavg_cluster|standard_fl|local_only)$ ]]; then
    echo "Error: Method must be one of: atlas, atlas_no_laplacian, fedavg_cluster, standard_fl, local_only"
    echo "Usage: $0 <model> <method>"
    exit 1
fi

# Configuration
SEED=123
MODEL_NORMALIZED="${MODEL//\//_}"  # Replace / with _ for file paths
TASKS="sst2 mrpc cola qnli"
CLIENTS_PER_TASK=3
ROUNDS=10  # 10 rounds in one shot

# Create directories
mkdir -p results
mkdir -p checkpoints
mkdir -p logs

echo "========================================"
echo "GPU 1 - Seed $SEED"
echo "Model: $MODEL | Method: $METHOD"
echo "Rounds: $ROUNDS (one shot)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "========================================"

# Build command - model-specific hyperparameters loaded automatically
OUTPUT_FILE="results/atlas_${MODEL_NORMALIZED}_${METHOD}_seed${SEED}_r${ROUNDS}.json"

CMD=("$PYTHON_CMD" "experiments/atlas_integrated.py"
    --mode full
    --ablation "$METHOD"
    --model "$MODEL"
    --tasks $TASKS
    --clients-per-task "$CLIENTS_PER_TASK"
    --rounds "$ROUNDS"
    --seed "$SEED")

# Log file (includes timestamp)
TS=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/gpu1_${MODEL_NORMALIZED}_${METHOD}_seed${SEED}_r${ROUNDS}_${TS}.log"

echo ""
echo "[START] ${CMD[*]}"
echo "Logging to: $LOGFILE"
echo ""

# Run and tee output
"${CMD[@]}" 2>&1 | tee -a "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]:-0}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "[SUCCESS] Experiment complete for $METHOD (seed $SEED, model $MODEL)"
    
    # The runner writes the canonical output filename directly.
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Results saved: $OUTPUT_FILE"
    else
        echo "[WARN] Expected results file not found: $OUTPUT_FILE"
        echo "[INFO] Checking for recent result files (last 3 hours)..."
        find results/ -name "*${METHOD}*seed${SEED}*.json" -type f -mmin -180 -print || true
    fi
    
    echo ""
    echo "[COMPLETE] GPU 1 experiment done!"
    echo "Model: $MODEL | Seed: $SEED | Method: $METHOD | Rounds: $ROUNDS"
else
    echo ""
    echo "[FAILED] Experiment failed (exit code: $EXIT_CODE)"
    exit $EXIT_CODE
fi
