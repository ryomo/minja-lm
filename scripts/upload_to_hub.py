"""
Upload the trained MinjaLM model to Hugging Face Hub.

Before running this script:
1. Create an Access Token with `write` permissions on Hugging Face Hub.
2. Login to Hugging Face: huggingface-cli login
3. Make sure your model is trained and saved in the `model` directory.
"""
import os
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import to register the model
import minja_lm  # noqa: F401


REPO_NAME = "ryomo/minja-lm"
PROJECT_ROOT = Path(__file__).parents[1]
MODEL_DIR = str(PROJECT_ROOT / "model")

def upload_model_card(_repo_name):
    """Upload the model card to Hugging Face Hub."""

    # Path to model card
    model_card_path = PROJECT_ROOT / "MODEL_CARD.md"

    if not model_card_path.exists():
        print(f"Error: Model card not found at {model_card_path}")
        return

    # Initialize Hugging Face API
    api = HfApi()

    print(f"Uploading model card to {_repo_name}...")

    # Upload the model card as README.md
    api.upload_file(
        path_or_fileobj=str(model_card_path),
        path_in_repo="README.md",
        repo_id=_repo_name,
        commit_message="Add model card documentation",
    )


def main():
    # Set working directory to the script's location
    current_dir = Path(__file__).parent
    os.chdir(current_dir)

    # Check if model files exist
    model_path = Path(MODEL_DIR)
    if not (model_path / "config.json").exists():
        print("Error: config.json not found. Please train the model first.")
        return

    if not (model_path / "model.safetensors").exists():
        print("Error: model.safetensors not found. Please train the model first.")
        return

    print("Loading model and tokenizer...")

    # Load the trained model
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    # Load the tokenizer (using the same one as training)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(
        "rinna/japanese-gpt2-medium", legacy=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Uploading model to {REPO_NAME}...")

    try:
        # Upload model to Hub
        model.push_to_hub(REPO_NAME)

        # Upload tokenizer to Hub
        tokenizer.push_to_hub(REPO_NAME)

        # Upload model card
        upload_model_card(REPO_NAME)

        print(f"✅ Model successfully uploaded to: https://huggingface.co/{REPO_NAME}")

    except Exception as e:
        print(f"❌ Error uploading model: {e}")


if __name__ == "__main__":
    main()
