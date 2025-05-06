# Scaleway Agent - Fine-Tuned Mistral 7B with LoRA

This repository contains a fine-tuned version of [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) using **LoRA** adapters for the task of generating **Scaleway CLI and Terraform commands** from natural language instructions. This model has been trained on 1000 examples describing cloud infrastructure tasks (e.g., "Create a bucket in region fr-par").

---

## Why This Project?

Cloud infrastructure is complex. This project enables an **AI assistant to generate Scaleway infrastructure code (CLI + Terraform)** from a simple prompt like:

```

Create a virtual instance named resource-42 with image ubuntu-focal and type GP1-M in zone fr-par-1

```

Result:
```
CLI: scw rdb instance create ...
Terraform:
resource "scaleway_rdb_instance" ...
```

---

## How It Works
### 1. **Install dependencies (`setup.sh`)**

- Execute this script to install all dependancies (python,Hugginface cli,CUDA etc)

  ```bash
  bash setup.sh
  ```

- Switch to the virtual environment `source /root/finetune-venv/bin/activate`
- Login to Huggingface `huggingface-cli login`, enter your huggingface token


### 2. **Prepare Dataset (`preprocess_and_tokenize.py`)**
Each dataset item is a JSON object like:

```json
{
  "prompt": "Create a bucket in fr-par",
  "cli": "scw object create-bucket name=b1 region=fr-par",
  "terraform": "resource scaleway_object_bucket ..."
}
```

We concatenate the prompt and response into:

```
### Instruction:
<prompt>

### Response:
CLI: <cli>
Terraform: <tf>
```

Then, we tokenize and align labels, by executing `python3 process_and_tokenize.py` masking the instruction part (`labels = -100`) to only train on the response.

---

### 3. **Train with LoRA (`train_mistral_lora.py`)**

We apply [PEFT](https://github.com/huggingface/peft) + [LoRA](https://arxiv.org/abs/2106.09685) to adapt Mistral-7B efficiently:

- Base model is **frozen** (no training)
- We only train small LoRA adapter weights (`~0.02%` of the model)
- Optimizer: **Adafactor**
- Save adapters after training

Execute `python3 train_mistral_lora.py`
---

### 4. **Merge LoRA Adapters (Optional)**

To export a standalone model (without PEFT dependency), we merge the LoRA adapters into the base model:

Execute 

```bash 
python3 merge_lora_and_push.py
```

---

### 5. **Push to Hugging Face Hub (`push_to_hub.py`)**

To share the model, execute:

```bash
huggingface-cli login
python3 push_to_hub.py
```

---

## 6. Inference Example (`test_inference.py`)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "JQXavier/scaleway-agent"  # or "mistral_lora_merged" if you wish to test it locally
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "Create a bucket called data-b1 in region fr-par"
instruction = f"### Instruction:\n{prompt}\n\n### Response:\n"
inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Folder Structure

```
.
├── preprocess_and_tokenize.py  # Tokenizes the JSONL dataset
├── train_mistral_lora.py       # Trains Mistral with LoRA
├── push_to_hub.py              # Uploads model to Hugging Face Hub
├── test_inference.py           # Runs inference locally
├── mistral_lora_output/        # Training outputs and checkpoints
├── mistral_lora_merged/        # Final merged model (ready to push or use)
└── setup.sh                    # Install all dependancies
```

---

## Requirements

```bash
pip install transformers==4.39.3 \
            peft==0.10.0 \
            datasets==2.18.0 \
            accelerate==0.28.0 \
            bitsandbytes==0.43.1 \
            sentencepiece==0.1.99 \
            protobuf==4.25.3 \
            huggingface_hub==0.22.2
```

---

## Results

After 3 epochs on 1000 examples:
- **Training loss:** ~0.02
- **Eval loss:** ~0.0013
- Model can generate **valid CLI + Terraform** with high accuracy

---

## Next Steps

- Integrate with LangChain or a chatbot interface
- Expand to support more Scaleway services

---

## Credits

Built by [JQXavier](https://huggingface.co/JQXavier) using 🤗 Transformers + PEFT on **Scaleway L4 GPU**.

---
