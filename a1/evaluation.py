import torch
import matplotlib.pyplot as plt

from A1_skeleton import A1Tokenizer, A1RNNModel


def predict_next_token(model, tokenizer, sentence, device, top_k=10):
    model.eval()

    # Tokenize input
    input_ids = tokenizer(sentence, return_tensors='pt').input_ids

    # Remove EOS to predict the next token
    input_ids = input_ids[:, :-1].to(device)

    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits

        # Last logit corresponds to the prediction after the last input token
        next_token_logits = logits[0, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=-1)

        # Get top k predictions
        top_probs, top_ids = torch.topk(probabilities, top_k)

    # Decode predicted tokens
    top_tokens = [
        tokenizer.decode([token_id.item()])
        for token_id in top_ids
    ]

    # Print results
    print(f"\nInput: {sentence}")
    print(f"Most likely next token: {repr(top_tokens[0])}")
    print(f"Probability: {top_probs[0].item():.4f}")

    print("\nTop predictions:")
    for token, prob in zip(top_tokens, top_probs):
        print(f"{repr(token):20s} {prob.item():.4f}")

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.bar(top_tokens, top_probs.cpu().numpy())
    plt.xlabel("Predicted token")
    plt.ylabel("Probability")
    plt.title(f"Top {top_k} next-token predictions")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = A1Tokenizer.from_file("tokenizers/tokenizer_1024.pkl")
    model = A1RNNModel.from_pretrained("models/lstm_v1024_e256_h512_l5_ep10_bs128")
    model.to(device)

    sentences = [
        "She lives in San",
        "The capital of France is",
        "I like to eat",
    ]

    for sentence in sentences:
        predict_next_token(model, tokenizer, sentence, device)
        