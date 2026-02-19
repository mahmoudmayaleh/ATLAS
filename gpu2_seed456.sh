#!/bin/bash
# GPU 2 - Seed 456 - GPT-2 XL Experiments
# Run each method separately: atlas, fedavg_cluster, local_only
# Split into 3 sessions: 3 rounds, 4 rounds, 3 rounds

set -e

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
SESSION=${1:-1}
METHOD=${2:-atlas}

# Validate arguments
if [[ ! "$SESSION" =~ ^[1-3]$ ]]; then
    echo "Error: Session must be 1, 2, or 3"
    echo "Usage: $0 <session> <method>"
    exit 1
fi

if [[ ! "$METHOD" =~ ^(atlas|fedavg_cluster|local_only)$ ]]; then
    echo "Error: Method must be atlas, fedavg_cluster, or local_only"
    echo "Usage: $0 <session> <method>"
    exit 1
fi

# Configuration
SEED=456
MODEL="gpt2-xl"
TASKS="sst2 mrpc cola qnli"
CLIENTS_PER_TASK=3
SAMPLES=3000
LOCAL_EPOCHS=3
BATCH_SIZE=8
FP_SAMPLES=25
FP_BATCHES=20

# Session configurations
declare -A SESSION_START SESSION_ROUNDS SESSION_TOTAL
SESSION_START[1]=0
SESSION_ROUNDS[1]=3
SESSION_TOTAL[1]=3

SESSION_START[2]=3
SESSION_ROUNDS[2]=4
SESSION_TOTAL[2]=7

SESSION_START[3]=7
SESSION_ROUNDS[3]=3
SESSION_TOTAL[3]=10

START=${SESSION_START[$SESSION]}
TOTAL_ROUNDS=${SESSION_TOTAL[$SESSION]}
CHECKPOINT_PATH="checkpoints/atlas_${METHOD}_seed${SEED}_round_${START}.pkl"

echo "========================================"
echo "GPU 2 - Seed $SEED - Method: $METHOD"
echo "Session $SESSION - Rounds $((START + 1)) to $TOTAL_ROUNDS"
echo "========================================"

# Build command with session-specific output
OUTPUT_FILE="results/atlas_integrated_full_atlas_gpt2xl_${METHOD}_seed${SEED}_session${SESSION}.json"

CMD="$PYTHON_CMD experiments/atlas_integrated.py \
    --mode full \
    --ablation $METHOD \
    --model $MODEL \
    --tasks $TASKS \
    --clients-per-task $CLIENTS_PER_TASK \
    --rounds $TOTAL_ROUNDS \
    --samples $SAMPLES \
    --local-epochs $LOCAL_EPOCHS \
    --batch-size $BATCH_SIZE \
    --fingerprint-samples $FP_SAMPLES \
    --fingerprint-batches $FP_BATCHES \
    --seed $SEED"

# Add resume flag if not first session
if [ "$SESSION" != "1" ]; then
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "[RESUME] Loading checkpoint: $CHECKPOINT_PATH"
        CMD="$CMD --resume $CHECKPOINT_PATH"
    else
        echo "[ERROR] Checkpoint not found: $CHECKPOINT_PATH"
        echo "Run Session $((SESSION - 1)) first!"
        exit 1
    fi
fi

# Run experiment
echo ""
echo "[START] $CMD"
echo ""
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "[SUCCESS] Session $SESSION complete for $METHOD (seed $SEED)"
    GENERATED="results/atlas_integrated_full_${METHOD}_seed${SEED}.json"
    if [ -f "$GENERATED" ]; then
        mv "$GENERATED" "$OUTPUT_FILE"
        echo "Results: $OUTPUT_FILE"
    else
        echo "[WARN] Expected results file not found: $GENERATED"
    fi

    if [ "$SESSION" != "3" ]; then
        NEXT_CHECKPOINT="checkpoints/atlas_${METHOD}_seed${SEED}_round_${TOTAL_ROUNDS}.pkl"
        echo "Checkpoint saved: $NEXT_CHECKPOINT"
        echo ""
        echo "Next: ./gpu2_seed456.sh $((SESSION + 1)) $METHOD"
    else
        echo ""
        echo "[COMPLETE] All sessions done for $METHOD!"
    fi
else
    echo ""
    echo "[FAILED] Session $SESSION failed"
    exit 1
fi
