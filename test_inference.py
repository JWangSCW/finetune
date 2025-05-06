# !/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# model_id = "JQXavier/scaleway-agent"  # or "mistral_lora_merged" for local test
model_id = "mistral_lora_merged"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Sample test prompt
prompt = "Create a Kubernetes Kapsule cluster named kapsule-90bd8d in region pl-waw with version 1.26.9"
formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=2560, do_sample=False)

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\nModel Response\n")
print(response)
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import sys

# # Load your fine-tuned model and tokenizer
# model_dir = "mistral_lora_merged"
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")

# # Read the prompt from CLI or ask interactively
# if len(sys.argv) > 1:
#     raw_prompt = " ".join(sys.argv[1:])
# else:
#     raw_prompt = input("Enter your instruction: ").strip()

# # Format the prompt exactly like during training
# formatted_prompt = f"""### Instruction:
# {raw_prompt}

# ### Response:
# CLI:"""

# # Tokenize input
# inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

# # Generate output deterministically
# output = model.generate(
#     **inputs,
#     max_new_tokens=256,
#     do_sample=False,            # no randomness
#     num_beams=1,                # greedy decoding
#     eos_token_id=tokenizer.eos_token_id,
#     pad_token_id=tokenizer.pad_token_id
# )

# # Decode and print cleanly
# decoded = tokenizer.decode(output[0], skip_special_tokens=True)
# # Optional: strip anything generated after the next instruction block
# response = decoded.split("### Instruction:")[0].strip()

# print("\n🔹 Model Response:\n")
# print(response)
