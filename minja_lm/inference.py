import os
from pathlib import Path
import torch
from transformers import AutoTokenizer
from model import MinjaLMForCausalLM, SimpleConfig, Model

if __name__ == "__main__":
    # Set working directory to the script's location
    current_dir = Path(__file__).parent
    os.chdir(current_dir)

    # Configuration
    config = SimpleConfig()
    model_file = "minja_lm.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Japanese GPT-2 tokenizer setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress parallelism warnings
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", legacy=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the saved model and run inference
    minja_model = MinjaLMForCausalLM(config)
    print(minja_model.generate(tokenizer, "お気に入りの音楽を", 20, device=device))
