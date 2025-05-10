from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the merged model and tokenizer
# model_id = "JQXavier/scaleway-agent"  # or "mistral_lora_merged" for local test
model_id = "mistral_lora_merged"
revision= "main"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", revision=revision)

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
