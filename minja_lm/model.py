import json
import torch
import torch.nn as nn


class SimpleConfig:
    def __init__(self, config_path=None):
        if not config_path:
            config_path = "config.json"

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        for key, value in config_dict.items():
            setattr(self, key, value)


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


class MinjaLMForCausalLM(Model):
    """
    CausalLM wrapper compatible with HuggingFace Transformers.
    """

    def __init__(self, config):
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config["vocab_size"]
        n_embd = getattr(config, "n_embd", 128)
        n_layer = getattr(config, "n_layer", 2)
        n_head = getattr(config, "n_head", 2)
        block_size = getattr(config, "block_size", 16)
        super().__init__(vocab_size, n_embd, n_layer, n_head, block_size)
        self.config = config
        self.block_size = block_size

    @classmethod
    def from_pretrained(cls, model_path, *args, config=None, **kwargs):
        import torch

        if config is None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_path)
        model = cls(config)
        state_dict = torch.load(f"{model_path}/minja_lm.pth", map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def forward(self, input_ids, **kwargs):
        return super().forward(input_ids)

    def generate(self, tokenizer, prompt, max_new_tokens=20, temperature=0.7, device="cpu"):
        """
        Generate text using the model and tokenizer with temperature sampling.
        """
        self.eval()
        self.to(device)
        idx = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(idx[:, -self.block_size:])
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
                if next_id.item() == tokenizer.eos_token_id:
                    break
        return tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)
