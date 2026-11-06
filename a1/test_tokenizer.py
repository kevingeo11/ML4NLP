
import argparse
import os
import torch

from A1_skeleton import A1Tokenizer


def test_tokenizer(tokenizer):

    print("\n" + "=" * 50)
    print("TOKENIZER SUMMARY")
    print("=" * 50)

    print(f"Vocabulary size:    {len(tokenizer):,}")
    print(f"BPE merge rules:    {len(tokenizer.merge_rules):,}")
    print(f"Max model length:   {tokenizer.model_max_length}")

    print("\nSpecial tokens:")
    print(f"PAD: {tokenizer.pad_token_id}")
    print(f"UNK: {tokenizer.unk_token_id}")
    print(f"BOS: {tokenizer.bos_token_id}")
    print(f"EOS: {tokenizer.eos_token_id}")

    # 1. Vocabulary sanity checks
    assert len(tokenizer) >= 260
    assert len(tokenizer.merge_rules) == len(tokenizer) - 260

    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3

    for i in range(256):
        assert bytes([i]) in tokenizer.voc

    print("\n[PASS] Vocabulary and special tokens")

    # 2. Encoding and decoding
    texts = [
        "Hello world!",
        "This is a test.",
        "Natural language processing is interesting.",
        "café",
        "😊",
        "日本語",
        "electrochemistry"
    ]

    print("\n" + "=" * 50)
    print("ENCODING / DECODING")
    print("=" * 50)

    for text in texts:
        encoded = tokenizer(
            [text],
            truncation=False,
            padding=False
        )

        ids = encoded["input_ids"][0]
        decoded = tokenizer.decode(ids)

        print(f"\nOriginal: {text}")
        print(f"Token IDs: {ids}")
        print(f"Decoded:  {decoded}")

        assert decoded == text

    print("\n[PASS] Encoding and decoding")

    # 3. Verify inverse vocabulary reconstruction
    text = "This is a test."

    ids = tokenizer([text])["input_ids"][0]

    tokens = [
        tokenizer.int_to_token[i]
        for i in ids
        if isinstance(tokenizer.int_to_token[i], bytes)
    ]

    reconstructed = b"".join(tokens).decode("utf-8")

    assert reconstructed == text

    print("\n[PASS] Inverse vocabulary reconstruction")

    # 4. Padding and attention mask
    texts = [
        "Hello",
        "Hello world! This is a longer sentence."
    ]

    encoded = tokenizer(
        texts,
        padding=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.dtype == torch.long

    assert input_ids.shape == attention_mask.shape
    assert input_ids.shape[0] == len(texts)

    assert torch.equal(
        attention_mask,
        (input_ids != tokenizer.pad_token_id).long()
    )

    for text, ids in zip(texts, input_ids):
        assert tokenizer.decode(ids) == text

    print("\n[PASS] Padding and attention mask")
    print("Padded batch shape:", tuple(input_ids.shape))

    # 5. Truncation
    if tokenizer.model_max_length is not None:

        text = "Hello world! " * (tokenizer.model_max_length + 10)

        encoded = tokenizer(
            [text],
            truncation=True,
            padding=False
        )

        ids = encoded["input_ids"][0]

        assert len(ids) <= tokenizer.model_max_length
        assert ids[0] == tokenizer.bos_token_id

        print("\n[PASS] Truncation")
        print("Truncated length:", len(ids))

    # 6. Save and reload
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test_tokenizer.pkl")

        tokenizer.save(filename)
        loaded = A1Tokenizer.from_file(filename)

        assert len(loaded) == len(tokenizer)
        assert loaded.voc == tokenizer.voc
        assert loaded.merge_rules == tokenizer.merge_rules

        test_text = "Testing saved tokenizer."

        assert (
            loaded([test_text])["input_ids"]
            == tokenizer([test_text])["input_ids"]
        )

    print("\n[PASS] Save and reload")

    print("\n" + "=" * 50)
    print("ALL TOKENIZER TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to saved tokenizer"
    )

    args = parser.parse_args()

    tokenizer = A1Tokenizer.from_file(args.tokenizer)

    print(f"Loaded tokenizer: {args.tokenizer}")

    test_tokenizer(tokenizer)
