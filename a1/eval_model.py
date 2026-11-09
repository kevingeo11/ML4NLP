
import argparse
import math
import os
import sys

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from A1_skeleton import A1Tokenizer, A1RNNModel


def evaluate(model, tokenizer, val_loader, device):
    """Compute validation cross-entropy and perplexity."""

    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:

            labels = tokenizer(
                batch["text"],
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).input_ids

            labels[labels == tokenizer.pad_token_id] = -100

            n_tokens = (labels[:, 1:] != -100).sum().item()

            if n_tokens == 0:
                continue

            labels = labels.to(device)
            output = model(labels=labels)

            total_loss += output.loss.item()

    val_loss = total_loss / len(val_loader)
    perplexity = math.exp(val_loss)

    return val_loss, perplexity


def next_token_predictions(model, tokenizer, prompt, device, k=5):
    """Print the top-k next BPE token predictions."""

    model.eval()

    input_ids = tokenizer(
        prompt, return_tensors='pt'
    ).input_ids.to(device)

    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits

        next_token_logits = logits[0, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=-1)

        scores, indices = torch.topk(probabilities, k=k, dim=-1)

    print(f"\nPrompt: {prompt!r}")
    print(f"Top {k} next BPE tokens:")

    for score, idx in zip(scores, indices):
        token = tokenizer.int_to_token[idx.item()]
        print(f"  {token!r:<22} probability: {score.item():.4f}")


def generate(model, tokenizer, prompt, device, max_new_tokens=100):
    """Generate text autoregressively using greedy decoding."""

    model.eval()

    input_ids = tokenizer(
        prompt, return_tensors='pt'
    ).input_ids.to(device)

    stopped_by_eos = False
    with torch.no_grad():
        for _ in range(max_new_tokens):

            # Forward pass
            output = model(input_ids)

            # Predict next token
            logits = output.logits[0, -1, :]
            next_id = torch.argmax(logits).item()

            # Stop at EOS
            if next_id == tokenizer.eos_token_id:
                stopped_by_eos = True
                break

            # Insert predicted token before the dummy EOS
            next_token = torch.tensor(
                [[next_id]], dtype=torch.long, device=device
            )

            input_ids = torch.cat([
                input_ids[:, :-1],
                next_token,
                input_ids[:, -1:]
            ], dim=1)

    return tokenizer.decode(input_ids[0]), stopped_by_eos


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer",
        default="tokenizers/tokenizer_512.pkl"
    )
    parser.add_argument(
        "--model",
        default="models/lstm_v512_e512_h1024_l2_ep50_bs128"
    )
    parser.add_argument("--val_file", default="val.txt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--output", default="output.txt")
    parser.add_argument("--use_cpu", action="store_true")

    args = parser.parse_args()

    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("Batch size and max_new_tokens must be positive.")

    device = torch.device(
        "cpu" if args.use_cpu
        else "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load tokenizer and model
    tokenizer = A1Tokenizer.from_file(args.tokenizer)
    model = A1RNNModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    assert len(tokenizer) == model.config.vocab_size

    # Load validation dataset
    dataset = load_dataset(
        "text",
        data_files={"val": args.val_file}
    )["val"]

    dataset = dataset.filter(
        lambda x: x["text"].strip() != ""
    )

    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Print to terminal and save to output file
    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, text):
            for stream in self.streams:
                stream.write(text)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    with open(args.output, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, f)

        try:
            print("Output")
            print("-" * 20)
            print(" ")
            print("Model:", args.model)
            print("Tokenizer:", args.tokenizer)
            print("Device:", device)
            print("Vocabulary size:", len(tokenizer))
            print("Embedding size:", model.embedding.embedding_dim)
            print("Hidden size:", model.rnn.hidden_size)
            print("LSTM layers:", model.rnn.num_layers)
            print("Validation paragraphs:", len(dataset))

            print("\nEvaluating validation dataset...", flush=True)

            val_loss, perplexity = evaluate(
                model, tokenizer, val_loader, device
            )

            print("\nValidation Results")
            print(f"Cross-entropy loss: {val_loss:.4f}")
            print(f"Perplexity:         {perplexity:.2f}")

            prompts = [
                "She lives in San ",
                "The history of",
                "Gothenburg is ",
                "I like to eat ",
                "Scientists discovered that"
            ]

            
            print("\nNext token predictions")
            print("-" * 20)

            for prompt in prompts:
                next_token_predictions(
                    model, tokenizer, prompt, device, k=5
                )

            print("\n")
            print("\nSome Auto Completions (max 100 tokens using argmax)")
            print("-" * 20)

            for prompt in prompts:
                text, stopped_by_eos = generate(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    max_new_tokens=args.max_new_tokens
                )

                print(f"\nPrompt: {prompt}")
                print(f"Completion: {text}")
                print(
                    "Stopped by:",
                    "<EOS>" if stopped_by_eos else "token limit"
                )

        finally:
            sys.stdout = original_stdout

    print(f"\nEvaluation results saved to {args.output}")


if __name__ == "__main__":
    main()
