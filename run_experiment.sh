#!/bin/bash
# Dynamic ATLAS Experiment Runner
# Supports multiple models (gpt2, gpt2-xl, qwen-0.5b) and seeds
# Runs 10 rounds in one shot with all professional metrics

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect Python command (python3 or python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python not found. Please activate your conda environment first.${NC}"
    echo "Try: conda activate atlas_env"
    exit 1
fi

echo -e "${BLUE}Using Python: $PYTHON_CMD ($(which $PYTHON_CMD))${NC}"

# Parse arguments
MODEL=${1:-gpt2-xl}
SEED=${2:-42}
METHOD=${3:-atlas}

# Validate MODEL argument
VALID_MODELS=("distilbert" "gpt2" "gpt2-xl" "qwen2.5")
if [[ ! " ${VALID_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo -e "${RED}Error: Invalid model '${MODEL}'${NC}"
    echo "Valid models: distilbert, gpt2, gpt2-xl, qwen2.5"
    echo ""
    echo "Usage: $0 <model> <seed> <method>"
    echo "  model:  distilbert | gpt2 | gpt2-xl | qwen2.5 (default: gpt2-xl)"
    echo "  seed:   42 | 123 | 456 | any integer (default: 42)"
    echo "  method: atlas | fedavg_cluster | local_only (default: atlas)"
    echo ""
    echo "Examples:"
    echo "  $0 gpt2-xl 42 atlas"
    echo "  $0 gpt2 123 fedavg_cluster"
    echo "  $0 qwen2.5 456 local_only"
    exit 1
fi

# Validate METHOD argument
if [[ ! "$METHOD" =~ ^(atlas|fedavg_cluster|local_only)$ ]]; then
    echo -e "${RED}Error: Method must be atlas, fedavg_cluster, or local_only${NC}"
    echo "Usage: $0 <model> <seed> <method>"
    exit 1
fi

# Normalize model name for file paths
MODEL_NORMALIZED="${MODEL//\//_}"  # Replace / with _ for Qwen path

# Configuration - Standard for all models
TASKS="sst2 mrpc cola qnli"
CLIENTS_PER_TASK=3
ROUNDS=10  # 10 rounds in one shot

# Create necessary directories
mkdir -p results
mkdir -p checkpoints
mkdir -p logs

# Generate timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${MODEL_NORMALIZED}_seed${SEED}_${METHOD}_${TIMESTAMP}.log"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ATLAS Experiment Configuration${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  ${BLUE}Model:${NC}          ${MODEL}"
echo -e "  ${BLUE}Seed:${NC}           ${SEED}"
echo -e "  ${BLUE}Method:${NC}         ${METHOD}"
echo -e "  ${BLUE}Rounds:${NC}         ${ROUNDS} (in one shot)"
echo -e "  ${BLUE}Tasks:${NC}          ${TASKS}"
echo -e "  ${BLUE}Clients/Task:${NC}   ${CLIENTS_PER_TASK}"
echo -e "  ${BLUE}Total Clients:${NC}  $((CLIENTS_PER_TASK * 4))"
echo -e "  ${BLUE}Log File:${NC}       ${LOG_FILE}"
echo -e "${GREEN}========================================${NC}"

# Build output filename
OUTPUT_FILE="results/atlas_${MODEL_NORMALIZED}_${METHOD}_seed${SEED}_r${ROUNDS}.json"

# Build command - Model-specific hyperparameters are loaded automatically from config.py
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
echo -e "${YELLOW}[START] Running experiment...${NC}"
echo -e "${BLUE}Command: $CMD${NC}"
echo ""

# Execute with tee to show output and save to log
eval $CMD 2>&1 | tee "$LOG_FILE"

# Check if experiment succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[SUCCESS] Experiment completed successfully!${NC}"
    
    # Find and rename the generated results file
    GENERATED_FILE="results/atlas_integrated_full_${METHOD}_seed${SEED}.json"
    if [ -f "$GENERATED_FILE" ]; then
        mv "$GENERATED_FILE" "$OUTPUT_FILE"
        echo -e "${GREEN}Results saved to: ${OUTPUT_FILE}${NC}"
    else
        echo -e "${YELLOW}[WARNING] Expected results file not found: $GENERATED_FILE${NC}"
        echo -e "${YELLOW}Looking for alternative locations...${NC}"
        # Check for other possible locations
        find results/ -name "*${METHOD}*seed${SEED}*.json" -type f -mmin -10
    fi
    
    # Display results summary
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Experiment Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "  Model:       ${MODEL}"
    echo -e "  Seed:        ${SEED}"
    echo -e "  Method:      ${METHOD}"
    echo -e "  Rounds:      ${ROUNDS}"
    echo -e "  Results:     ${OUTPUT_FILE}"
    echo -e "  Log:         ${LOG_FILE}"
    echo -e "${BLUE}========================================${NC}"
    
else
    echo ""
    echo -e "${RED}[FAILED] Experiment failed!${NC}"
    echo -e "${RED}Check log file for details: ${LOG_FILE}${NC}"
    exit 1
fi
