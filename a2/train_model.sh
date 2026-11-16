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

VOCAB_SIZE=512
LEARNING_RATE=1e-4
EPOCHS=2
TRAIN_BATCH_SIZE=128
EVAL_BATCH_SIZE=128
HIDDEN_SIZE=512
INTERMEDIATE_SIZE=1024
LAYERS=2
NUM_ATTENTION_HEADS=8
MAX_POSITION_EMBEDDINGS=2048
ROPE_THETA=10000.0
RMS_NORM_EPS=1e-5

tokenizer="tokenizers/tokenizer_${VOCAB_SIZE}.pkl"
model_name="transformer_v${VOCAB_SIZE}_h${HIDDEN_SIZE}_l${LAYERS}_ah${NUM_ATTENTION_HEADS}_bs${TRAIN_BATCH_SIZE}_ep${EPOCHS}"

# Run model training
python -u train_model.py \
    --tokenizer "$tokenizer" \
    --epochs 10 \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --hidden_size "$HIDDEN_SIZE" \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --num_layers "$LAYERS" \
    --num_attention_heads "$NUM_ATTENTION_HEADS" \
    --max_position_embeddings "$MAX_POSITION_EMBEDDINGS" \
    --rope_theta "$ROPE_THETA" \
    --rms_norm_eps "$RMS_NORM_EPS" \
    --output_dir models \
    --model_name "$model_name" \
    "$@"

echo "Finished at: $(date)"
