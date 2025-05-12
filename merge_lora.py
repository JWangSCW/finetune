from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Path to the base model and LoRA adapter
base_model = "JQXavier/scaleway-agent" # or "Mistral-7B-Instruct" if you want to use the original model
# revision = "13c7d575e52205157085cffb9a7c5de260a8a650"  # Uncomment if you want to use a specific branch/commit
adapter_path = "mistral_lora_output/final_checkpoint"
merged_model_path = "mistral_lora_merged"

# Load the base model and LoRA adapter
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16) # Add revision=revision if needed
model = PeftModel.from_pretrained(model, adapter_path)

# Move model to bfloat16 explicitly (just to be sure that we are not using float32, need to explicitly set it)
model = model.to(torch.bfloat16)

# Merge LoRA weights into the base model
model.save_pretrained(merged_model_path) # If you need save the model in PyTorch (.bin), you can add safe_serialization=False

# Save the tokenizer
# Load the tokenizer from the base model and save it to the merged model path
tokenizer = AutoTokenizer.from_pretrained(base_model) # Add revision=revision if needed
tokenizer.save_pretrained(merged_model_path)

