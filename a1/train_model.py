import argparse
import time

import torch
from datasets import load_dataset
from transformers import TrainingArguments

from A1_skeleton import A1Tokenizer, A1RNNModelConfig, A1RNNModel, A1Trainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="tokenizer_512.pickle", help="Pickled tokenizer file")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--output_dir", default="a1_model")
    parser.add_argument("--use_cpu", action="store_true")
    args = parser.parse_args()

    tokenizer = A1Tokenizer.from_file(args.tokenizer)
    print(f"Loaded tokenizer {args.tokenizer} (vocab size {len(tokenizer)})")

    dataset = load_dataset(
            'text',
            data_files={'train': "train.txt", 'val': "val.txt"}
        )
    
    dataset = dataset.filter(lambda x: x['text'].strip() != '')

    config = A1RNNModelConfig(
        vocab_size=len(tokenizer),
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        pad_token_id=tokenizer.pad_token_id
    )
    model = A1RNNModel(config)

    args = TrainingArguments(
        output_dir=args.output_dir,
        optim='adamw_torch',
        eval_strategy='epoch',
        use_cpu=args.use_cpu,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
    )

    start = time.perf_counter()

    trainer = A1Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"Training took {time.perf_counter() - start:.1f}s")