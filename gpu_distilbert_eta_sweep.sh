#!/bin/bash
# GPU script - DistilBERT (seed 42) - ETA sweep (0.1, 0.5, 1.0)
# Usage: ./gpu_distilbert_eta_sweep.sh [method]
# Example: ./gpu_distilbert_eta_sweep.sh atlas

set -e

# Detect Python command (python3 or python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please activate your conda environment first."
    exit 1
fi

MODEL="distilbert-base-uncased"
METHOD=${1:-atlas}
SEED=42
MODEL_NORMALIZED="${MODEL//\//_}"
TASKS="sst2 mrpc cola qnli"
CLIENTS_PER_TASK=3
ROUNDS=10

mkdir -p results
mkdir -p checkpoints
mkdir -p logs

ETAS=(0.1 0.5 1.0)

for ETA in "${ETAS[@]}"; do
    echo "========================================"
    echo "Model: $MODEL | Method: $METHOD | eta=$ETA | Seed: $SEED"
    echo "========================================"

    OUTPUT_FILE="results/atlas_${MODEL_NORMALIZED}_${METHOD}_seed${SEED}_eta${ETA}_r${ROUNDS}.json"

    CMD="$PYTHON_CMD experiments/atlas_integrated.py \
        --mode full \
        --ablation $METHOD \
        --model $MODEL \
        --tasks $TASKS \
        --clients-per-task $CLIENTS_PER_TASK \
        --rounds $ROUNDS \
        --seed $SEED \
        --eta $ETA"

    echo ""
    echo "[START] $CMD"
    echo ""
    eval $CMD

    if [ $? -eq 0 ]; then
        echo ""
        echo "[SUCCESS] Completed run for eta=$ETA"

        GENERATED="results/atlas_integrated_full_${METHOD}_seed${SEED}.json"
        if [ -f "$GENERATED" ]; then
            mv "$GENERATED" "$OUTPUT_FILE"
            echo "Results saved: $OUTPUT_FILE"
        else
            # Fallback: try to find recently created result file containing method and seed
            alt=$(find results/ -maxdepth 1 -type f -name "*${METHOD}*seed${SEED}*.json" -printf "%p\n" | tail -n 1)
            if [ -n "$alt" ]; then
                mv "$alt" "$OUTPUT_FILE"
                echo "Results saved (alt): $OUTPUT_FILE"
            else
                echo "[WARN] Expected results file not found: $GENERATED"
                echo "[INFO] Check results/ for output files"
            fi
        fi

        echo "[COMPLETE] Run for eta=$ETA finished"
    else
        echo ""
        echo "[FAILED] Run failed for eta=$ETA"
        exit 1
    fi

    echo "" && sleep 2
done

echo "All runs finished. Files in results/ with _eta<value>_ in names."
