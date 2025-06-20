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

    def generate(self, tokenizer, prompt, max_new_tokens=20, temperature=0.7, device="cpu"):
        """
        Generate text using the model and tokenizer with temperature sampling.
        """
        self.eval()
        self.to(device)
        idx = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(idx[:, -self.config.block_size:])
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
                if next_id.item() == tokenizer.eos_token_id:
                    break
        return tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)
