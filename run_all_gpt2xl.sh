#!/bin/bash
# Master script to launch all 3 GPT-2 XL experiments in parallel
# Each seed runs on a different GPU using tmux

set -e

echo "=========================================="
echo "Launching GPT-2 XL FULL Experiments"
echo "=========================================="
echo "Configuration:"
echo "  Model: gpt2-xl"
echo "  Mode: FULL (not quick)"
echo "  Rounds: 10 (single session)"
echo "  Samples: 3000"
echo "  Local Epochs: 3"
echo ""
echo "GPU Assignment:"
echo "  GPU 0: Seed 42"
echo "  GPU 1: Seed 123"
echo "  GPU 2: Seed 456"
echo "=========================================="
echo ""

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux not found. Install with: sudo apt-get install tmux"
    exit 1
fi

# Make scripts executable
chmod +x run_gpt2xl_seed42.sh
chmod +x run_gpt2xl_seed123.sh
chmod +x run_gpt2xl_seed456.sh

# Launch each experiment in a separate tmux session
echo "[GPU 0] Starting seed 42 in tmux session 'gpt2xl_seed42'"
tmux new-session -d -s gpt2xl_seed42 "CUDA_VISIBLE_DEVICES=0 ./run_gpt2xl_seed42.sh; exec bash"

echo "[GPU 1] Starting seed 123 in tmux session 'gpt2xl_seed123'"
tmux new-session -d -s gpt2xl_seed123 "CUDA_VISIBLE_DEVICES=1 ./run_gpt2xl_seed123.sh; exec bash"

echo "[GPU 2] Starting seed 456 in tmux session 'gpt2xl_seed456'"
tmux new-session -d -s gpt2xl_seed456 "CUDA_VISIBLE_DEVICES=2 ./run_gpt2xl_seed456.sh; exec bash"

echo ""
echo "=========================================="
echo "All experiments launched!"
echo "=========================================="
echo ""
echo "To monitor experiments:"
echo "  tmux attach -t gpt2xl_seed42   # GPU 0"
echo "  tmux attach -t gpt2xl_seed123  # GPU 1"
echo "  tmux attach -t gpt2xl_seed456  # GPU 2"
echo ""
echo "To detach from tmux: Press Ctrl+B then D"
echo ""
echo "To check if experiments are running:"
echo "  tmux ls"
echo ""
echo "Results will be saved to:"
echo "  results/atlas_integrated_full_atlas_seed42.json"
echo "  results/atlas_integrated_full_atlas_seed123.json"
echo "  results/atlas_integrated_full_atlas_seed456.json"
echo "=========================================="
