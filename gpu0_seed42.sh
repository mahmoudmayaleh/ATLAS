#!/bin/bash
# GPU 0 - Seed 42 - Dynamic Model Experiments
# Supports: gpt2, gpt2-xl, qwen-0.5b
# Runs 10 rounds in one shot with all professional metrics

set -e

export HUGGINGFACE_HUB_TOKEN="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"

# Detect Python command (python3 or python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please activate your conda environment first."
    echo "Try: conda activate atlas_env"
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($(which $PYTHON_CMD))"

# Parse arguments
MODEL=${1:-gpt2-xl}
METHOD=${2:-atlas}

# Validate arguments
VALID_MODELS=("distilbert-base-uncased" "gpt2" "gpt2-xl" "Qwen/Qwen2.5-0.5B")
if [[ ! " ${VALID_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "Error: Invalid model '${MODEL}'"
    echo "Valid models: distilbert-base-uncased, gpt2, gpt2-xl, Qwen/Qwen2.5-0.5B"
    echo "Usage: $0 <model> <method>"
    echo "  model:  distilbert-base-uncased | gpt2 | gpt2-xl | Qwen/Qwen2.5-0.5B (default: gpt2-xl)"
    echo "  method: atlas | fedavg_cluster | local_only (default: atlas)"
    exit 1
fi

if [[ ! "$METHOD" =~ ^(atlas|atlas_no_laplacian|fedavg_cluster|standard_fl|local_only)$ ]]; then
    echo "Error: Method must be one of: atlas, atlas_no_laplacian, fedavg_cluster, standard_fl, local_only"
    echo "Usage: $0 <model> <method>"
    exit 1
fi

# Configuration
SEED=42
MODEL_NORMALIZED="${MODEL//\//_}"  # Replace / with _ for file paths
TASKS="sst2 mrpc cola qnli"
CLIENTS_PER_TASK=3
ROUNDS=10  # 10 rounds in one shot

# Create directories
mkdir -p results
mkdir -p checkpoints
mkdir -p logs

echo "========================================"
echo "GPU 0 - Seed $SEED"
echo "Model: $MODEL | Method: $METHOD"
echo "Rounds: $ROUNDS (one shot)"
echo "========================================"

# Build command - model-specific hyperparameters loaded automatically
OUTPUT_FILE="results/atlas_${MODEL_NORMALIZED}_${METHOD}_seed${SEED}_r${ROUNDS}.json"

CMD="$PYTHON_CMD experiments/atlas_integrated.py \
    --mode full \
    --ablation $METHOD \
    --model $MODEL \
    --tasks $TASKS \
    --clients-per-task $CLIENTS_PER_TASK \
    --rounds $ROUNDS \
    --seed $SEED"

# Run experiment
echo ""
echo "[START] $CMD"
echo ""
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "[SUCCESS] Experiment complete for $METHOD (seed $SEED, model $MODEL)"
    
    # Find and rename the generated results file
    GENERATED="results/atlas_integrated_full_${METHOD}_seed${SEED}.json"
    if [ -f "$GENERATED" ]; then
        mv "$GENERATED" "$OUTPUT_FILE"
        echo "Results saved: $OUTPUT_FILE"
    else
        echo "[WARN] Expected results file not found: $GENERATED"
        echo "[INFO] Checking for alternative result files..."
        find results/ -name "*${METHOD}*seed${SEED}*.json" -type f -mmin -10
    fi
    
    echo ""
    echo "[COMPLETE] GPU 0 experiment done!"
    echo "Model: $MODEL | Seed: $SEED | Method: $METHOD | Rounds: $ROUNDS"
else
    echo ""
    echo "[FAILED] Experiment failed"
    exit 1
fi
