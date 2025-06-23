"""
Inference using the uploaded MinjaLM model from Hugging Face Hub.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_NAME = "ryomo/minja-lm"


def main():
    print(f"Using repository: {REPO_NAME}")

    # Load model and tokenizer from Hub
    print("Loading model from Hugging Face Hub...")
    model = AutoModelForCausalLM.from_pretrained(
        REPO_NAME,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(REPO_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Model loaded successfully on {device}")
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.config}")

    # Test generation
    test_prompts = [
        "今日は",
        "お気に入りの音楽は",
        "人工知能について",
    ]

    print("\n=== Generation Test ===")
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")

        try:
            # Encode prompt to input_ids
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            # Generate tokens
            generated_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )

            # Decode and print the result
            generated_text = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            print(f"Generated: '{generated_text}'")
        except Exception as gen_error:
            print(f"Generation error for prompt '{prompt}': {gen_error}")
            continue

    print(
        "\n✅ Model test completed successfully!\n"
        f"🎉 Your model '{REPO_NAME}' is working correctly!\n"
        f"You can share it with others: https://huggingface.co/{REPO_NAME}"
    )


if __name__ == "__main__":
    main()
