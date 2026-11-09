
#!/bin/bash

set -euo pipefail

cd "$HOME/ML4NLP/a1"

# Experiment 2 configuration
VOCAB_SIZES=(512 1024 2048)

EMBEDDING_SIZES=(128 256 512)
HIDDEN_SIZES=(256 512 1024)

LAYERS=2
EPOCHS=50
TRAIN_BATCH_SIZE=128
EVAL_BATCH_SIZE=128

mkdir -p logs models

job_count=0

for vocab in "${VOCAB_SIZES[@]}"; do

    tokenizer="tokenizers/tokenizer_${vocab}.pkl"

    if [[ ! -f "$tokenizer" ]]; then
        echo "Missing tokenizer: $tokenizer"
        exit 1
    fi

    for i in "${!EMBEDDING_SIZES[@]}"; do

        embedding=${EMBEDDING_SIZES[$i]}
        hidden=${HIDDEN_SIZES[$i]}

        model_name="lstm_v${vocab}_e${embedding}_h${hidden}_l${LAYERS}_ep${EPOCHS}_bs${TRAIN_BATCH_SIZE}"

        echo "Submitting: $model_name"

        sbatch \
            --job-name="exp2_v${vocab}_h${hidden}" \
            train_model.sh \
            --tokenizer "$tokenizer" \
            --model_name "$model_name" \
            --epochs "$EPOCHS" \
            --train_batch_size "$TRAIN_BATCH_SIZE" \
            --eval_batch_size "$EVAL_BATCH_SIZE" \
            --embedding_size "$embedding" \
            --hidden_size "$hidden" \
            --num_layers "$LAYERS" \
            --output_dir models

        job_count=$((job_count + 1))

    done
done

echo "All $job_count Experiment 2 jobs submitted."
