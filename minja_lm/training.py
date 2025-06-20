import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from model import Model, block_size, device, generate, model_path, tokenizer
from torch.utils.data import DataLoader

# Set working directory to the script's location
current_dir = Path(__file__).parent
os.chdir(current_dir)

# Load dataset from CSV file (expects a 'text' column)
dataset = load_dataset("csv", data_files="dataset.csv")


class CustomDataset(torch.utils.data.Dataset):
    """Dataset class holding pairs of input and label sequences."""

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])


if __name__ == "__main__":

    # Convert text to token ID sequences
    def tokenize_function(example):
        ids = tokenizer.encode(example["text"])
        return {"ids": ids}

    # Tokenize the entire dataset
    tokenized = dataset["train"].map(tokenize_function)
    all_ids = sum(tokenized["ids"], [])  # Concatenate all token IDs

    # Create input and label sequences in blocks of block_size
    inputs = []
    labels = []
    for i in range(0, len(all_ids) - block_size, block_size):
        x = all_ids[i : i + block_size]
        y = all_ids[i + 1 : i + block_size + 1]
        if len(x) == block_size and len(y) == block_size:
            inputs.append(x)
            labels.append(y)

    # Create PyTorch Dataset and DataLoader
    ds = CustomDataset(inputs, labels)
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=8)

    # Prepare model, optimizer, and loss function
    model = Model(vocab_size=len(tokenizer), block_size=block_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(20):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}, loss: {total_loss/len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # Load the saved model and run inference
    loaded_model = Model(len(tokenizer), block_size=block_size).to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()

    # Example: Generate text using the loaded model
    print(generate(loaded_model, tokenizer, "お気に入りの音楽を", 20))
