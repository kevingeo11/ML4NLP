# Names: Kevin George, Sofia Larsson, Thémis Halka

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import SiLU
from torch.utils.data import DataLoader

import numpy as np
import os
import json

class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None,
                 pad_token_id=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id



class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        # TODO: initalize components here
        self.gated_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.activation = SiLU()
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        

    def forward(self, hidden_states):
        hidden_states1 = self.gated_proj(hidden_states)
        hidden_states1 = self.activation(hidden_states1)

        hidden_states2 = self.up_proj(hidden_states)

        hidden_states = hidden_states1 * hidden_states2
        hidden_states = self.down_proj(hidden_states)

        return hidden_states  
    
# This is optional, since you can use PyTorch's RMSNorm.
# class A2RMSNorm(nn.Module):
#     """RMS layer normalization."""
#     def __init__(self, config):
#         super().__init__()
#         # TODO: Use config.rms_norm_eps
#         # TODO: initalize weights here

#     def forward(self, hidden_states):
#         ...


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        # TODO: set up W_q, W_k, W_v, W_o here
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size

        assert self.hidden_size % self.num_attention_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.W_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # TODO: set up normalizers here
        self.norm_q = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.norm_k = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)


    def forward(self, hidden_states, rope_rotations):

        batch_size, seq_length, _ = hidden_states.size()

        q = self.W_q(hidden_states)
        k = self.W_k(hidden_states)
        v = self.W_v(hidden_states)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, rope_rotations)

        attn_output = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.hidden_size)
        output = self.W_o(attn_output)
        return output


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        # TODO: set up attention, MLP, and normalizers here.
        self.mha = A2Attention(config)
        self.mlp = A2MLP(config)
        self.post_mha_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(self, hidden_states, rope_rotations):
        hidden_states_res = self.mha(hidden_states, rope_rotations)
        hidden_states_res = self.post_mha_norm(hidden_states_res)
        hidden_states = hidden_states + hidden_states_res
        hidden_states_res = self.mlp(hidden_states)
        hidden_states_res = self.post_mlp_norm(hidden_states_res)
        hidden_states = hidden_states + hidden_states_res

        return hidden_states


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        # TODO: Set up the other components here.
        # TODO: put all transformer decoder layers in a ModuleList.
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder_layers = nn.ModuleList([A2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.pad_token_id = config.pad_token_id

        # This line should be called after you have set up all components.
        self.post_init()


    def forward(self, labels):
        """The forward pass of the Transformer language model.

           Args:
             labels:  The input tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             CausalLMOutputWithPast object that stores loss and logits (the output tensor (3D) for all token positions for all vocabulary items.)
        """

        loss_func = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        X = labels[:,:-1] # all columns in labels except the last one
        Y = labels[:, 1:] # all columns in labels except the first one

        rope_rotations = self.rotary_emb(X) # pass this to all the transformer decoder layers

        # TODO: Call embedding, transformer decoder layers, last normalizer, and unembedding.
        hidden_states = self.embedding(X)
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states, rope_rotations)
        hidden_states = self.norm(hidden_states)
        logits = self.unembedding(hidden_states)

        loss = loss_func(
            logits.reshape(-1, self.config.vocab_size),
            Y.reshape(-1)
        )
        
        # TODO: compute the loss for the model output and Y
        return CausalLMOutputWithPast(loss=loss, logits=logits)


#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin



# The trainer from the prev assg
class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, processing_class):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class

        assert args.optim == 'adamw_torch'
        assert args.eval_strategy == 'epoch'

    def select_device(self):
        """Return the device to use for training."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def collate_fn(self, batch):
        """Pad already-tokenized sequences to the longest sequence in the batch."""

        sequences = [item['input_ids'] for item in batch]
        max_length = max(len(seq) for seq in sequences)

        padded_sequences = []

        for seq in sequences:
            pad_length = max_length - len(seq)

            padded_seq = seq + [
                self.processing_class.pad_token_id
            ] * pad_length

            padded_sequences.append(padded_seq)

        return torch.tensor(padded_sequences, dtype=torch.long)

    def train(self):
        """Train the model using pre-tokenized datasets."""

        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        val_loader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        history = []

        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            total_loss = 0.0

            for step, batch in enumerate(train_loader):

                # Already tokenized and padded
                labels = batch.to(device)

                output = self.model(labels)

                optimizer.zero_grad()
                output.loss.backward()
                optimizer.step()

                total_loss += output.loss.item()

                if step % 100 == 0:
                    print(
                        f'epoch {epoch+1} step {step}/{len(train_loader)} '
                        f'loss {output.loss.item():.4f}'
                    )

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:

                    labels = batch.to(device)

                    output = self.model(labels)
                    val_loss += output.loss.item()

            val_loss /= len(val_loader)

            train_loss = total_loss / len(train_loader)

            print(
                f'epoch {epoch+1}: '
                f'train loss {train_loss:.4f}, '
                f'val loss {val_loss:.4f}, '
                f'perplexity {np.exp(val_loss):.2f}'
            )

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "perplexity": float(np.exp(val_loss))
            })

            # Save history after every epoch
            os.makedirs(args.output_dir, exist_ok=True)

            with open(
                os.path.join(args.output_dir, "training_history.json"), "w"
            ) as f:
                json.dump(history, f, indent=4)

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)
        