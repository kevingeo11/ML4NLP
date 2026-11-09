import os
import time
import argparse
from tokenizer import build_tokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Maximum BPE vocabulary size"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum model sequence length"
    )

    args = parser.parse_args()

    if args.vocab_size < 260:
        parser.error("Vocabulary size must be at least 260")

    if args.max_length < 2:
        parser.error("Maximum model length must be at least 2")

    print(f"Training tokenizer with vocab size {args.vocab_size}")

    start = time.perf_counter()

    tokenizer = build_tokenizer(
        "train.txt",
        max_voc_size=args.vocab_size,
        model_max_length=args.max_length
    )

    print(f"Training time: {time.perf_counter() - start:.2f} seconds")

    os.makedirs("tokenizers", exist_ok=True)

    filename = f"tokenizers/tokenizer_{args.vocab_size}.pkl"
    tokenizer.save(filename)

    print(f"Tokenizer saved to {filename}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Number of merges: {len(tokenizer.merge_rules)}")
