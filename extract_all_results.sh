#!/bin/bash
# Extract results from all checkpoint files for all seeds
# Usage: ./extract_all_results.sh [model_name]

set -e

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found"
    exit 1
fi

# Model name (default: gpt2-xl)
MODEL_NAME=${1:-gpt2-xl}

echo "========================================"
echo "Extracting results from checkpoints"
echo "Model: $MODEL_NAME"
echo "========================================"

# Extract all checkpoints matching the pattern
# Looks for: atlas_*_seed*_round_*.pkl
$PYTHON_CMD tools/extract_checkpoint_results.py "checkpoints/*_seed*_round_*.pkl" --model "$MODEL_NAME"

echo ""
echo "✓ All results extracted!"
echo "Output files saved to: results/"
echo ""
echo "To view results:"
echo "  ls -lh results/atlas_integrated_quick_*_${MODEL_NAME//-/_}_seed*.json"
