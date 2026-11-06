import argparse
import time
import os

from datasets import load_dataset
from transformers import TrainingArguments

from A1_skeleton import A1Tokenizer, A1RNNModelConfig, A1RNNModel, A1Trainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer_512.pkl", help="Pickled tokenizer file")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--model_name", default=None, help="Optional model name")
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

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)

    print(f"Model type:           {model.__class__.__name__}")
    print(f"Vocabulary size:      {len(tokenizer):,}")
    print(f"Embedding size:       {config.embedding_size}")
    print(f"Hidden size:          {config.hidden_size}")
    print(f"LSTM layers:          {model.rnn.num_layers}")
    print(f"Bidirectional:        {model.rnn.bidirectional}")
    print(f"Max sequence length:  {tokenizer.model_max_length}")

    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nLayer parameters:")
    for name, module in [
        ("Embedding", model.embedding),
        ("LSTM", model.rnn),
        ("Unembedding", model.unembedding)
    ]:
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<16} {n_params:>12,}")

    print("=" * 50)

    if args.model_name is None:
        args.model_name = (
            f"lstm_v{len(tokenizer)}"
            f"_e{args.embedding_size}"
            f"_h{args.hidden_size}"
            f"_l{args.num_layers}"
            f"_ep{args.epochs}"
            f"_bs{args.train_batch_size}"
        )

    model_dir = os.path.join(args.output_dir, args.model_name)

    if os.path.exists(model_dir):
        raise FileExistsError(
            f"Model directory already exists: {model_dir}"
        )

    os.makedirs(model_dir)
    print("Model will be saved to:", model_dir)

    args = TrainingArguments(
        output_dir=model_dir,
        optim='adamw_torch',
        eval_strategy='epoch',
        use_cpu=args.use_cpu,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        report_to="none"
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