import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from .configuration import MinjaLMConfig


class MinjaLM(PreTrainedModel):
    """Minimal GPT-style Transformer decoder model."""

    config_class = MinjaLMConfig

    def __init__(self, config):
        super().__init__(config)

        vocab_size = config.vocab_size
        n_embd = config.n_embd
        n_layer = config.n_layer
        n_head = config.n_head
        block_size = config.block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)  # Token embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))  # Positional embedding
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=n_embd, nhead=n_head, batch_first=True, activation="gelu"
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)  # Output projection

    def forward(self, idx):
        # idx: (batch, seq_len)
        _B, T = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, input_ids, max_new_tokens=20, temperature=0.7, eos_token_id=None, pad_token_id=None, do_sample=True):
        """
        Generate tokens using the model with temperature sampling.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Temperature for sampling (higher = more random)
            eos_token_id (int, optional): Token ID to stop generation
            pad_token_id (int, optional): Padding token ID (unused for now)
            do_sample (bool): Whether to use sampling (True) or greedy decoding (False)

        Returns:
            torch.Tensor: Generated token IDs of shape (batch_size, original_seq_len + generated_tokens)
        """
        self.eval()
        device = input_ids.device
        self.to(device)

        # Ensure input_ids has the right shape
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        idx = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to the last block_size tokens if sequence is too long
                idx_cond = idx[:, -self.config.block_size:] if idx.size(1) > self.config.block_size else idx
                logits = self(idx_cond)
                logits = logits[:, -1, :]  # Get the last token's logits

                if do_sample:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

                idx = torch.cat([idx, next_id], dim=1)

                # Stop if we hit the end-of-sequence token
                if eos_token_id is not None and next_id.item() == eos_token_id:
                    break

        return idx
