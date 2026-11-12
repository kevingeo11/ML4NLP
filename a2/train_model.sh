#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:59:00
#SBATCH --output=logs/train_model_%j.out
#SBATCH --error=logs/train_model_%j.err

set -euo pipefail

# Move to project directory
cd "$HOME/ML4NLP/a2"

# Activate Python environment
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

echo "================================"
echo "Starting Transformer model training"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================"

# Verify GPU allocation
nvidia-smi

# Run model training
python -u train_model.py "$@"

echo "Finished at: $(date)"
