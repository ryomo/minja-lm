import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Configuration
model_path = "minja_lm.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 16


class Model(nn.Module):
    """Minimal GPT-style Transformer decoder model."""

    def __init__(self, vocab_size, n_embd=128, n_layer=2, n_head=2, block_size=16):
        super().__init__()
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


def generate(model, tokenizer, prompt, max_new_tokens=20, temperature=0.7):
    """
    Generate text using the model and tokenizer with temperature sampling.
    Args:
        model: Trained language model
        tokenizer: Corresponding tokenizer
        prompt: Initial text to start generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random, lower = more deterministic)
    Returns:
        Generated text as a string
    """
    model.eval()
    idx = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])  # Get logits for next token
        logits = logits[:, -1, :] / temperature  # Apply temperature scaling
        probs = torch.softmax(logits, dim=-1)  # Convert to probability distribution
        next_id = torch.multinomial(probs, num_samples=1)  # Sample next token
        idx = torch.cat([idx, next_id], dim=1)  # Append to sequence
        if next_id.item() == tokenizer.eos_token_id:
            break  # Stop if end-of-sequence token is generated
    return tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)  # Convert to text


# Japanese GPT-2 tokenizer setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress parallelism warnings
# Load pre-trained Japanese GPT-2 tokenizer
# Set pad_token to eos_token for compatibility
# (rinna/japanese-gpt2-medium is a Japanese language model)
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", legacy=False)
tokenizer.pad_token = tokenizer.eos_token
