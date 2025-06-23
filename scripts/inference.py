import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import minja_lm  # noqa: F401  # Import to register MinjaLM


PROJECT_ROOT = Path(__file__).parents[1]
model_dir = str(PROJECT_ROOT / "src" / "minja_lm")

if __name__ == "__main__":
    # Set working directory to the script's location
    current_dir = Path(__file__).parent
    os.chdir(current_dir)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Japanese GPT-2 tokenizer setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress parallelism warnings
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", legacy=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the saved model and run inference
    # NOTE: `trust_remote_code=True` is not needed here since this project uses a local model
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    print(model.generate(tokenizer, "お気に入りの音楽を", 20, device=device))
