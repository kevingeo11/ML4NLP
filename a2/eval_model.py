import argparse
import math
import sys

import torch
from datasets import load_dataset
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from A2_skeleton import *
from tokenizer import A1Tokenizer


def evaluate(model, val_loader, device):
    """Compute validation cross-entropy and perplexity."""

    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            
            output = model(input_ids)
            loss = output.loss
            
            total_loss += loss.item()
            total_batches += 1
    
    val_loss = total_loss / total_batches
    perplexity = math.exp(val_loss)
    
    return val_loss, perplexity


def next_token_predictions(model, tokenizer, prompt, device):
    """Print the next BPE token prediction."""
    
    model.eval()

    input_ids = tokenizer(
        prompt, return_tensors='pt'
    ).input_ids.to(device)
    
    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits
        
        next_token_logits = logits[0, -1, :]
        
        predicted_token_idx = torch.argmax(next_token_logits)
    
    predicted_token = tokenizer.int_to_token[predicted_token_idx.item()]
    
    print(f"\nPrompt: {prompt!r}")
    print(f"Most likely next token: {predicted_token!r}")


def generate(model, tokenizer, prompt, device, max_length, temperature, topk):
    """Generate text given a prompt using top-k sampling."""

    model.eval()

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    stopped_by_eos = False
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_ids)
            logits = output.logits
            next_token_logits = logits[0, -1, :] / temperature

            top_k_logits, top_k_indices = torch.topk(next_token_logits, topk)

            filtered_logits = torch.full_like(next_token_logits, float('-inf'))

            filtered_logits[top_k_indices] = top_k_logits

            next_token_logits = filtered_logits

            distribution = Categorical(logits=next_token_logits)
            next_token_id = distribution.sample().unsqueeze(0)

            if next_token_id.item() == eos_token_id:
                stopped_by_eos = True
                break

            input_ids = torch.cat([
                input_ids[:, :-1],
                next_token_id,
                input_ids[:, -1:]
            ], dim=1)

    return tokenizer.decode(input_ids[0]), stopped_by_eos

 
if __name__ == "__main__":

    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="tokenizers/tokenizer_512.pkl")
    parser.add_argument("--model", default="models/lstm_v512_e512_h1024_l2_ep50_bs128")
    parser.add_argument("--val_file", default="val.txt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output", default="output.txt")
    parser.add_argument("--use_cpu", action="store_true")
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = A1Tokenizer.from_file(args.tokenizer)
    model = A2Transformer.from_pretrained(args.model)
    device = "cpu" if args.use_cpu else "cuda"
    model.to(device)
    model.eval()
 
    # Load validation dataset
    dataset = load_dataset("text", data_files={"val": args.val_file})["val"]
    dataset = dataset.filter(lambda x: x["text"].strip() != "")

    # Tokenize the validation dataset
    dataset = dataset.map(
        lambda batch: dict(tokenizer(batch['text'], truncation=True, padding=False)),
        batched=True,
        remove_columns=['text'],
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

    # Generate outputs
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

 
            print("\nEvaluating validation dataset...", flush=True)
 
            val_loss, perplexity = evaluate(
                model, val_loader, device
            )
 
            print("\nValidation Results")
            print(f"Cross-entropy loss: {val_loss:.4f}")
            print(f"Perplexity:         {perplexity:.2f}")
 
            prompts = [
                'She lives in San',
                'Gothenburg is',
                'In natural language processing, a Transformer',
                'Is Stockholm the capital of Sweden? Answer yes or no. The answer is',
                'Write a Python program that reverses a list.'
            ]
 
            print("\nNext token predictions")
            print("-" * 20)
 
            for prompt in prompts:
                next_token_predictions(model, tokenizer, prompt, device)
 
            print("\n")
            print("\nSome Auto Completions (max 100 tokens using argmax)")
            print("-" * 20)
 
            for prompt in prompts:
                text, stopped_by_eos = generate(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    args.max_length,
                    args.temperature,
                    args.top_k,
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
