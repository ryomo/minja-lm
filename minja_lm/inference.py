import os
from pathlib import Path
import torch
from model import Model, block_size, device, generate, model_path, tokenizer

if __name__ == "__main__":
    # Set working directory to the script's location
    current_dir = Path(__file__).parent
    os.chdir(current_dir)

    # Load the trained model for inference
    loaded_model = Model(len(tokenizer), block_size=block_size).to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()

    # Example: Generate text using the loaded model
    print(generate(loaded_model, tokenizer, "お気に入りの", 20, temperature=0.7))
