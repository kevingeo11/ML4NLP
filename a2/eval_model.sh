#!/bin/bash

#SBATCH --job-name=eval_model
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_model_%j.out
#SBATCH --error=logs/eval_model_%j.err

set -euo pipefail

# Move to project directory
cd "$HOME/ML4NLP/a2"

# Activate Python environment
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

echo "================================"
echo "Starting Transformer model evaluation"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================"

# Verify GPU allocation
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs

# Run model evaluation
# Default parameters (can be overridden via command line)
python -u eval_model.py \
    --tokenizer "tokenizers/tokenizer_512.pkl" \
    --model "models/transformer_small" \
    --val_file "val.txt" \
    --batch_size 64 \
    --max_length 100 \
    --temperature 1.0 \
    --top_k 5 \
    --output "output.txt" \
    "$@"

echo "================================"
echo "Finished at: $(date)"
echo "Results saved to output.txt"
echo "================================"
