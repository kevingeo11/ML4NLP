#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=23:59:00
#SBATCH --output=tokenizer_%j.out
#SBATCH --error=tokenizer_%j.err

set -euo pipefail

# Move to project directory
cd "$HOME/ML4NLP/a1"

# Activate Python environment
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

echo "================================"
echo "Starting BPE tokenizer training"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================"

python -u train_model.py

echo "Finished at: $(date)"