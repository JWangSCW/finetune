#!/usr/bin/env python3
"""
push_to_hub.py

Push the fine-tuned model to Hugging Face Hub securely.
"""
from huggingface_hub import HfApi, HfFolder
from transformers import AutoTokenizer

# Load tokenizer from base model to save it
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)



# Define repository and model path
repo_id = "JQXavier/scaleway-agent"
model_dir = "mistral_lora_merged"

# Ensure you have your HF token (environment variable or login)
token = HfFolder.get_token()
if not token:
    raise ValueError("Hugging Face token not found. Set HF_TOKEN or use `huggingface-cli login`.")

api = HfApi()
# Create the repo if it doesn't exist (private by default)
api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
# Save tokenizer (important for future use or upload)
tokenizer.save_pretrained("mistral_lora_merged")

# Upload all files in model_dir to the repo (transformers files: config, pytorch_model.bin or safetensors, tokenizer, etc.)
api.upload_folder(
    repo_id=repo_id,
    folder_path=model_dir,
    path_in_repo="", 
    token=token
)
print(f"Pushed the model to https://huggingface.co/{repo_id}")
