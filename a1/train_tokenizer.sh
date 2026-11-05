#!/bin/bash

#SBATCH --job-name=bpe-tokenizer
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=23:59:00
#SBATCH --output=logs/tokenizer_%j.out
#SBATCH --error=logs/tokenizer_%j.err

set -euo pipefail

# Read vocabulary size from first argument
VOCAB_SIZE=${1:?Usage: sbatch train_tokenizer.sh VOCAB_SIZE}
MAX_LENGTH=${2:-256}

# Move to project directory
cd "$HOME/ML4NLP/a1"

# Activate Python environment
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

echo "================================"
echo "Starting BPE tokenizer training"
echo "Vocabulary size: $VOCAB_SIZE"
echo "Max sequence length: $MAX_LENGTH"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================"

python -u train_tokenizer.py \
    --vocab_size "$VOCAB_SIZE" \
    --max_length "$MAX_LENGTH"

echo "Finished at: $(date)"