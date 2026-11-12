import argparse
import time
import os

from datasets import load_dataset
from transformers import TrainingArguments

from A2_skeleton import *
from tokenizer import A1Tokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="tokenizers/tokenizer_16384.pkl", help="Pickled tokenizer file")

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)

    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--intermediate_size", type=int, default=11008)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=32)
    parser.add_argument("--max_position_embeddings", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--rms_norm_eps", type=float, default=1e-5)

    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--model_name", default=None, help="Optional model name")
    parser.add_argument("--use_cpu", action="store_true")
    args = parser.parse_args()

    # loading tokeniser
    tokenizer = A1Tokenizer.from_file(args.tokenizer)
    print(f"Loaded tokenizer {args.tokenizer} (vocab size {len(tokenizer)})")

    # loading dataset
    dataset = load_dataset(
                    'text',
                    data_files={'train': "train.txt", 'val': "val.txt"}
                )
    dataset = dataset.filter(lambda x: x['text'].strip() != '')

    # Tokenize each text example
    def tokenize_function(example):
        encoded = tokenizer(
            example["text"],
            truncation=True
        )

        return {"input_ids": encoded["input_ids"][0]}

    dataset = dataset.map(
        tokenize_function,
        remove_columns=["text"],
    )

    # dataset = dataset.map(
    #     lambda x: tokenizer(x["text"]),
    #     remove_columns=["text"],
    # )

    config = A2ModelConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_layers,
        rope_theta=args.rope_theta,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=args.rms_norm_eps,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = A2Transformer(config)

    # sanity checks
    assert len(model.decoder_layers) == args.num_layers, (
        f"Requested {args.num_layers} layers, "
        f"but model has {len(model.decoder_layers)}"
    )

    assert args.hidden_size % args.num_attention_heads == 0, (
        "hidden_size must be divisible by num_attention_heads"
    )

    # model statistics
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)

    print(f"Model type:           {model.__class__.__name__}")
    print(f"Vocabulary size:      {len(tokenizer):,}")
    print(f"Hidden size:          {config.hidden_size}")
    print(f"Intermediate size:    {config.intermediate_size}")
    print(f"Attention heads:      {config.num_attention_heads}")
    print(f"Transformer layers:   {config.num_hidden_layers}")
    print(f"Head dimension:       {config.hidden_size // config.num_attention_heads}")
    print(f"Max sequence length:  {config.max_position_embeddings}")

    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nLayer parameters:")

    for name, module in [
        ("Embedding", model.embedding),
        ("Decoder layers", model.decoder_layers),
        ("Final norm", model.norm),
        ("Unembedding", model.unembedding),
    ]:
        n_params = sum(
            p.numel() for p in module.parameters()
        )
        print(f"  {name:<16} {n_params:>12,}")

    print("=" * 50)

    # set output directory 
    if args.model_name is None:
        args.model_name = (
            f"transformer_v{len(tokenizer)}"
            f"_h{args.hidden_size}"
            f"_i{args.intermediate_size}"
            f"_l{args.num_layers}"
            f"_a{args.num_attention_heads}"
            f"_ep{args.epochs}"
            f"_bs{args.train_batch_size}"
        )

    model_dir = os.path.join(
        args.output_dir,
        args.model_name
    )

    if os.path.exists(model_dir):
        raise FileExistsError(
            f"Model directory already exists: {model_dir}"
        )

    os.makedirs(model_dir)
    print("Model will be saved to:", model_dir)

    # training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        optim="adamw_torch",
        eval_strategy="epoch",
        use_cpu=args.use_cpu,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        report_to="none",
    )
    
    # training model 
    start = time.perf_counter()

    trainer = A1Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(model_dir)

    print(
        f"Training took {time.perf_counter() - start:.1f}s"
    )