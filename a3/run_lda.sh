#!/bin/bash

#SBATCH --job-name=eval_model
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/run_lda_%j.out
#SBATCH --error=logs/run_lda_%j.err

set -euo pipefail

# Move to project directory
cd "$HOME/ML4NLP/a3"

# Activate Python environment
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

echo "================================"
echo "Starting LDA topic modelling"
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
python -u lda.py \
    --k 50 \
    --alpha 0.01 \
    --beta 0.01 \
    --output "output_$SLURM_JOB_ID.txt" \
    "$@"

echo "================================"
echo "Finished at: $(date)"
echo "Results saved to output_$SLURM_JOB_ID.txt"
echo "================================"
