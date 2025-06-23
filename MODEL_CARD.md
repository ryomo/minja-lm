---
language:
  - ja
license: mit
---

# MinjaLM Model Card

## Model Description

MinjaLM is a minimal Japanese language model based on a GPT-style transformer decoder architecture. This model is designed for educational purposes and demonstrates how to create custom models with the Transformers library.

## Model Details

- **Architecture**: GPT-style transformer decoder
- **Parameters**:
  - Vocabulary size: 32,000
  - Embedding dimension: 128
  - Number of layers: 2
  - Number of attention heads: 2
  - Block size (context length): 16
- **Training data**: Small Japanese text dataset
- **Tokenizer**: Uses `rinna/japanese-gpt2-medium` tokenizer

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("your-username/minja-lm", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("your-username/minja-lm")

# Generate text
prompt = "お気に入りの音楽を"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=20, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Training

The model was trained using the provided training script with the following configuration:
- Optimizer: AdamW (lr=0.0001, weight_decay=0.01)
- Training epochs: 20
- Batch size: 4
- Loss function: CrossEntropyLoss

## Limitations

This is a minimal model designed for educational purposes:
- Very small model size (limited capacity)
- Short context length
- Trained on a small dataset
- May generate repetitive or nonsensical text

## License

This project is open source and available under the MIT License.

## How to Train Your Own

1. Clone [ryomo/minja-lm](https://github.com/ryomo/minja-lm) from GitHub.
2. Install dependencies: `uv sync --frozen`
3. Train the model: `uv run scripts/training.py`
4. Run inference: `uv run scripts/inference.py`
