from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Base + adapter paths
base_model = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "mistral_lora_output/final_checkpoint"
merged_model_path = "mistral_lora_merged"

# Load base model and LoRA adapter, merge them
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
model.save_pretrained(merged_model_path)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(merged_model_path)
