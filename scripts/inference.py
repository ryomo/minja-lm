import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import minja_lm  # noqa: F401  # Import to register MinjaLM


PROJECT_ROOT = Path(__file__).parents[1]
MODEL_DIR = str(PROJECT_ROOT / "model")

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
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.to(device)

    # Prepare input
    prompt = "お気に入りの音楽を"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate tokens
    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=20,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True
    )

    # Decode and print the result
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(generated_text)
