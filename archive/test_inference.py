#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "JQXavier/scaleway-agent"  # or "mistral_lora_merged" for local test
# model_id = "mistral_lora_merged"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Sample test prompt
prompt = "Create a virtual instance named resource-42 with image ubuntu-focal and type GP1-M in region fr-par-1."
formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\n--- Model Response ---\n")
print(response)
