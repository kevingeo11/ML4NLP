import torch, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os

###
### Part 1. Tokenization.
###

# Byte-pair encoding (BPE): the main idea
# ---------------------------------------
# A token is one piece of text (not necessarily a word/subword/"token").
# In this tokenizer, every token is stored as one
# or more bytes. A pair means two tokens that appear next to each other, and a
# merge means joining such a pair to make one longer token.
#
# Training BPE:
#   1. Split every line of the training text into individual UTF-8 bytes.
#   2. Start the vocabulary with the four special tokens and all 256 possible
#      byte values. The vocabulary maps each token to an integer ID.
#   3. Count how often each pair of neighboring tokens occurs.
#   4. Join the most common pair everywhere it occurs. Add the joined token to
#      the vocabulary and save the pair as a merge rule.
#   5. Repeat steps 3 and 4 until the requested vocabulary size is reached
#      (NB: make sure it is provided).

#
# Encoding new text:
#   1. Remove whitespace from the beginning and end.
#   2. Split the text into individual UTF-8 bytes.
#   3. Apply the saved merge rules in the same order in which they were learned.
#   4. Replace the resulting tokens with their vocabulary IDs.
#   5. Add BOS/EOS IDs, and optionally truncate or pad the sequence.
#
# This simple version treats a whole non-empty line as one sequence. Therefore,
# it may learn tokens that contain spaces or parts of more than one word.

def split_into_bytes(text):
    """Split a string into individual UTF-8 byte tokens.

    Args:
        text: The input string to encode.

    Returns:
        A list containing one ``bytes`` object for each byte in the text.
    """
    return [bytes([b]) for b in text.encode("utf-8")]


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

    # converting each line to byte sequence
    with open(train_file, "r", encoding="utf-8") as f:
        seq = [
            split_into_bytes(l.strip())
            for l in f
            if l.strip()
        ]

    # byte pair encoding
    merge = []

    while len(voc) < max_voc_size:

        pairs = Counter()
        for s in seq:
            for i in range(len(s) - 1):
                pair = (s[i], s[i + 1])
                pairs[pair] += 1

        if not pairs:
            break

        most_frequent_pair = pairs.most_common(1)[0][0]
        new_token = most_frequent_pair[0] + most_frequent_pair[1]

        if new_token in voc:
            raise ValueError(f"Duplicate BPE token: {new_token!r}")

        voc[new_token] = len(voc) # updating vocabulary to add new token
        merge.append(most_frequent_pair) # storing the merging rule in order of appearance

        new_seq = [] # updating sequence, replacing all pairs with new token
        for s in seq:
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

        if len(merge) % 10 == 0:
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
            # splitting text into bytes
            seq = split_into_bytes(text.strip())

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
            ids = [self.voc.get(token, self.unk_token_id) for token in seq]

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
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load the tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)


###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=None, embedding_size=None, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.LSTM(config.embedding_size, config.hidden_size, 
                            num_layers=1, batch_first=True)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init() # for compatibility 
        
    def forward(self, labels):
        """The forward pass of the RNN-based language model.
        
           Args:
             X:  The input tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             CausalLMOutputWithPast object that stores loss and logits (the output tensor (3D) for all token positions for all vocabulary items.)
        """

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        X = labels[:, :-1] # all columns in input_ids except the last one
        Y = labels[:, 1:] # all columns in input_ids except the first one

        X = X.masked_fill(X == -100, self.config.pad_token_id) # self.config.pad_token_id

        assert(X.shape == Y.shape)

        embedded = self.embedding(X)
        rnn_out, _ = self.rnn(embedded)
        out = self.unembedding(rnn_out)

        loss = loss_func(
            out.view(-1, self.config.vocab_size), # out.view(-1, out.shape[-1])
            Y.reshape(-1)
        )

        #  compute the loss for the model output and Y 
        return CausalLMOutputWithPast(
            loss=loss,
            logits=out
        )


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, processing_class):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             processing_class:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate) 
        
        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        val_loader = DataLoader(self.eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)
        
        # TODO: Your work here is to implement the training loop.
        #       
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       labels = apply your tokenizer to B
        #       replace padding tokens with value -100 so loss function will ignore them
	    #       put labels onto the GPU (or whatever device you use)
        #       apply the model to labels
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward() # loss is stored as an attribute of model's output
        #       optimizer.step()

        tokenizer = self.processing_class

        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            total_loss = 0.0

            for step, batch in enumerate(train_loader):
                labels = tokenizer(batch['text'], truncation=True, padding=True, return_tensors='pt').input_ids
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.to(device)
                output = self.model(labels)

                optimizer.zero_grad()
                output.loss.backward() # loss is stored as an attribute of model's output
                optimizer.step()

                total_loss += output.loss.item()
                if step % 100 == 0:
                    print(f'epoch {epoch+1} step {step}/{len(train_loader)} loss {output.loss.item():.4f}')

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    labels = tokenizer(batch['text'], truncation=True, padding=True, return_tensors='pt').input_ids
                    labels[labels == tokenizer.pad_token_id] = -100
                    val_loss += self.model(labels.to(device)).loss.item()
            
            val_loss /= len(val_loader)

            print(f'epoch {epoch+1}: train loss {total_loss/len(train_loader):.4f}, '
                  f'val loss {val_loss:.4f}, perplexity {np.exp(val_loss):.2f}')
                    

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)

    
