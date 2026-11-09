
#!/bin/bash

set -euo pipefail

cd "$HOME/ML4NLP/a1"

# Experiment 3 configuration
VOCAB_SIZE=512
EMBEDDING_SIZE=512
HIDDEN_SIZE=1024

LAYERS=2
EPOCHS=100
TRAIN_BATCH_SIZE=128
EVAL_BATCH_SIZE=128

mkdir -p logs models

tokenizer="tokenizers/tokenizer_${VOCAB_SIZE}.pkl"

if [[ ! -f "$tokenizer" ]]; then
    echo "Missing tokenizer: $tokenizer"
    exit 1
fi

model_name="lstm_v${VOCAB_SIZE}_e${EMBEDDING_SIZE}_h${HIDDEN_SIZE}_l${LAYERS}_ep${EPOCHS}_bs${TRAIN_BATCH_SIZE}"

echo "Submitting Experiment 3: $model_name"

sbatch \
    --job-name="exp3_v512_l2" \
    train_model.sh \
    --tokenizer "$tokenizer" \
    --model_name "$model_name" \
    --epochs "$EPOCHS" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --embedding_size "$EMBEDDING_SIZE" \
    --hidden_size "$HIDDEN_SIZE" \
    --num_layers "$LAYERS" \
    --output_dir models

echo "Experiment 3 submitted."
