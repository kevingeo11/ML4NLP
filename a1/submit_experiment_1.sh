#!/bin/bash

set -euo pipefail

cd "$HOME/ML4NLP/a1"

VOCAB_SIZES=(512 1024 2048 4096)
LAYERS=(2 3 5)

EPOCHS=10
TRAIN_BATCH_SIZE=128
EVAL_BATCH_SIZE=128
EMBEDDING_SIZE=256
HIDDEN_SIZE=512

mkdir -p logs models

for vocab in "${VOCAB_SIZES[@]}"; do

    tokenizer="tokenizers/tokenizer_${vocab}.pkl"

    if [[ ! -f "$tokenizer" ]]; then
        echo "Missing tokenizer: $tokenizer"
        exit 1
    fi

    for layers in "${LAYERS[@]}"; do

        model_name="lstm_v${vocab}_e${EMBEDDING_SIZE}_h${HIDDEN_SIZE}_l${layers}_ep${EPOCHS}_bs${TRAIN_BATCH_SIZE}"

        echo "Submitting: $model_name"

        sbatch \
            --job-name="lstm_v${vocab}_l${layers}" \
            train_model.sh \
            --tokenizer "$tokenizer" \
            --model_name "$model_name" \
            --epochs "$EPOCHS" \
            --train_batch_size "$TRAIN_BATCH_SIZE" \
            --eval_batch_size "$EVAL_BATCH_SIZE" \
            --embedding_size "$EMBEDDING_SIZE" \
            --hidden_size "$HIDDEN_SIZE" \
            --num_layers "$layers" \
            --output_dir models

    done
done

echo "All 12 experiments submitted."