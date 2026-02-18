#!/bin/bash
# GPU 0 - Seed 42 - GPT-2 XL FULL Experiment
# 10 rounds, single session, saves results at the end

set -e

# Detect Python command
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

# Configuration - FULL MODE with DistilBERT settings
SEED=42
MODEL="gpt2-xl"
TASKS="sst2 mrpc cola qnli"
CLIENTS_PER_TASK=3
SAMPLES=3000           # Same as DistilBERT
LOCAL_EPOCHS=3         # Same as DistilBERT
ROUNDS=10              # 10 rounds total
BATCH_SIZE=8
FP_SAMPLES=25
FP_BATCHES=20

echo "========================================"
echo "GPU 0 - GPT-2 XL FULL Experiment"
echo "Seed: $SEED"
echo "Mode: FULL (not quick!)"
echo "Rounds: $ROUNDS (single session)"
echo "Samples: $SAMPLES"
echo "Local Epochs: $LOCAL_EPOCHS"
echo "========================================"

# Build command - FULL MODE
CMD="$PYTHON_CMD experiments/atlas_integrated.py \
    --mode full \
    --ablation atlas \
    --model $MODEL \
    --tasks $TASKS \
    --clients-per-task $CLIENTS_PER_TASK \
    --rounds $ROUNDS \
    --samples $SAMPLES \
    --local-epochs $LOCAL_EPOCHS \
    --batch-size $BATCH_SIZE \
    --fingerprint-samples $FP_SAMPLES \
    --fingerprint-batches $FP_BATCHES \
    --seed $SEED"

# Run experiment
echo ""
echo "[START] Running FULL experiment with $ROUNDS rounds"
echo "$CMD"
echo ""
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "[SUCCESS] FULL experiment complete!"
    echo "Seed: $SEED"
    echo "Results: results/atlas_integrated_full_atlas_seed${SEED}.json"
    echo "========================================"
else
    echo ""
    echo "[FAILED] Experiment failed"
    exit 1
fi
