import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# 設定
model_path = "minja_lm.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 16


class Model(nn.Module):
    """最小構成のGPT風Transformerデコーダモデル"""

    def __init__(self, vocab_size, n_embd=128, n_layer=2, n_head=2, block_size=16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
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
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

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
    モデルとトークナイザーを使ってテキストを自動生成する関数。
    temperature付きサンプリングによりランダム性を持たせる。
    - model: 学習済みモデル
    - tokenizer: 対応するトークナイザー
    - prompt: 生成のきっかけとなるテキスト
    - max_new_tokens: 生成する最大トークン数
    - temperature: サンプリングの多様性（大きいほど多様、小さいほど確定的）
    """
    model.eval()
    idx = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])  # 次トークンのロジット取得
        logits = logits[:, -1, :] / temperature  # temperatureでスケーリング
        probs = torch.softmax(logits, dim=-1)  # 確率分布に変換
        next_id = torch.multinomial(probs, num_samples=1)  # サンプリング
        idx = torch.cat([idx, next_id], dim=1)  # 生成文に追加
        if next_id.item() == tokenizer.eos_token_id:
            break  # 終端トークンなら終了
    return tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)  # テキスト化


# 日本語対応トークナイザー
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 警告対策
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", legacy=False)
tokenizer.pad_token = tokenizer.eos_token
