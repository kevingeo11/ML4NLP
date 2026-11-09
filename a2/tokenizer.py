import re
from collections import Counter
import pickle
import torch
from transformers import BatchEncoding


def split_into_bytes(text):
    """Split a string into individual UTF-8 byte tokens.

    Args:
        text: The input string to encode.

    Returns:
        A list containing one ``bytes`` object for each byte in the text.
    """
    return [bytes([b]) for b in text.encode("utf-8")]


PATTERN = re.compile(r' ?[^\W\d]+| ?\d+| ?[^\w\s]+|\s+')
def build_tokenizer(train_file, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """

    # making sure a maximal vocabulary size is provided
    if max_voc_size is None:
        raise NameError("A maximal vocabulary size (>260) must be provided.")
    if max_voc_size < 260:
        raise ValueError("Vocabulary must contain at least 260 tokens.")

    # Initialize vocabulary with 4 special tokens
    voc = {
        pad_token: 0,
        unk_token: 1,
        bos_token: 2,
        eos_token: 3
    }
    for i in range(256):
        voc[bytes([i])] = len(voc)

    # splitting each line into chunks (words, numbers, punctuation) and counting them.
    # Each unique chunk is stored once, together with how often it occurs.
    word_freq = Counter()
    with open(train_file, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                word_freq.update(PATTERN.findall(l.strip()))

    # converting each unique chunk to byte sequence
    seq = [split_into_bytes(w) for w in word_freq]   # byte tokens of each unique chunk
    freq = list(word_freq.values())                  # freq[k] = occurrences of seq[k]

    # byte pair encoding
    merge = []

    while len(voc) < max_voc_size:

        pairs = Counter()
        for s, f in zip(seq, freq):
            for i in range(len(s) - 1):
                pair = (s[i], s[i + 1])
                pairs[pair] += f           # a chunk seen f times contributes f times

        if not pairs:
            break

        most_frequent_pair = pairs.most_common(1)[0][0]
        new_token = most_frequent_pair[0] + most_frequent_pair[1]

        # Two different pairs can spell the same bytes, e.g. (ab, c) and (a, bc).
        # They are the same token, so it keeps its existing ID.
        if new_token not in voc:
            voc[new_token] = len(voc) # updating vocabulary to add new token
        merge.append(most_frequent_pair) # storing the merging rule in order of appearance

        first = most_frequent_pair[0]
        new_seq = [] # updating sequence, replacing all pairs with new token
        for s in seq:
            if len(s) < 2 or first not in s:
                new_seq.append(s)          # pair cannot occur here: keep as is
                continue
            updated_s = []
            i = 0
            while i < len(s):
                if (
                    i < len(s) - 1
                    and (s[i], s[i + 1]) == most_frequent_pair
                ):
                    updated_s.append(new_token)
                    i += 2  # skipping both merged tokens
                else:
                    updated_s.append(s[i])
                    i += 1

            new_seq.append(updated_s)

        seq = new_seq

        if len(merge) % 100 == 0:
            total_merges = max_voc_size - 260
            progress = len(merge) / total_merges * 100

            print(
                f"Progress: {len(merge)}/{total_merges} merges "
                f"({progress:.2f}%) | "
                f"Vocabulary: {len(voc)}/{max_voc_size}",
                flush=True
            )

    return A1Tokenizer(
        voc=voc,
        merge_rules=merge,
        model_max_length=model_max_length,
        pad_token=pad_token,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token
        )


class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self,
                 voc,
                 merge_rules,
                 model_max_length,
                 pad_token='<PAD>',
                 unk_token='<UNK>',
                 bos_token='<BOS>',
                 eos_token='<EOS>'):
        self.voc = voc
        self.merge_rules = merge_rules

        self.pad_token_id = voc[pad_token]     # Compulsory attribute.
        self.unk_token_id = voc[unk_token]
        self.bos_token_id = voc[bos_token]
        self.eos_token_id = voc[eos_token]

        self.model_max_length = model_max_length # Needed for truncation.

        self.int_to_token = {v: k for k, v in self.voc.items()}

        # NEW: the same word-splitting regex that build_tokenizer used. Stored on the
        # object so a saved tokenizer always splits the way it was trained.
        self.pattern = PATTERN

        # NEW: word -> list of token IDs. Each distinct word goes through BPE once;
        # every later occurrence is a dictionary lookup.
        self.cache = {}

    def _bpe(self, word):
        """NEW: BPE-encode one word into token IDs.

        This is the original rule loop, unchanged, applied to one word instead of a
        whole paragraph. The result is the same, because build_tokenizer learned
        every merge inside a single word: no rule joins two words.
        """
        # splitting word into bytes
        seq = split_into_bytes(word)

        # applying merge rules in order of appearance
        for pair in self.merge_rules:
            new_seq = []
            i = 0
            while i < len(seq):
                if (
                    i < len(seq) - 1
                    and (seq[i], seq[i + 1]) == pair
                ):
                    new_seq.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            seq = new_seq

        # converting tokens to integer IDs
        return [self.voc.get(token, self.unk_token_id) for token in seq]

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.

           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')

        if truncation and self.model_max_length is None:
            raise ValueError("model_max_length must be provided for truncation.")

        if isinstance(texts, str):
            texts = [texts]

        input_ids = []
        for text in texts:
            # CHANGED: splitting the text into words with the same regex as in training,
            # encoding each word (from the cache if seen before) and joining the IDs
            ids = []
            for word in self.pattern.findall(text.strip()):
                if word not in self.cache:
                    self.cache[word] = self._bpe(word)
                ids.extend(self.cache[word])

            # adding BOS/EOS tokens
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

            if truncation:
                ids = ids[:self.model_max_length]

            input_ids.append(ids)

        if padding and input_ids:
            max_length = max(len(t) for t in input_ids)
            input_ids = [
                t + [self.pad_token_id] * (max_length - len(t))
                for t in input_ids
            ]

        # creating attention mask
        attention_mask = [
            [0 if token == self.pad_token_id else 1 for token in t]
            for t in input_ids
        ]

        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

        return BatchEncoding({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.voc)

    def decode(self, ids):
        """Return a string decoded from given input ids"""

        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # removing special tokens and joining bytes
        tokens = [
            self.int_to_token[i]
            for i in ids
            if i in self.int_to_token
            and isinstance(self.int_to_token[i], bytes)
        ]

        return b''.join(tokens).decode("utf-8", errors="replace")

    def save(self, filename):
        """Save the tokenizer to the given file."""
        self.cache.clear()
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load the tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)