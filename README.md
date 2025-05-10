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
### Requirements
- Scaleway account & API key
- [HuggingFace account, personal token, model repository](https://huggingface.co/docs/huggingface_hub/quick-start)
- Scaleway VPC, Private Network(optional), GPU Instance (starting from Nvidia L4)+ BlockStorage, PGW (optional) or Flexible IP to access in GPU instance to fine-tune model
- Scaleway Managed Inference instance to inferent model through public endpoint or private endpoint, using BYOM (Bring your own model) to import the post-finetuned model from HuggingFace

### 1. **Install dependencies (`setup.sh`)**
- Execute this script to install all dependancies (python,Hugginface cli,CUDA etc)
  ```bash
  bash setup.sh
  ```
- Switch to the virtual environment `source ~/finetune-venv/bin/activate`
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
Then, we tokenize and align labels, by executing `python3 process_and_tokenize.py`

---

### 3. **Train with LoRA (`train_mistral_lora.py`)**
We apply [PEFT](https://github.com/huggingface/peft) + [LoRA](https://arxiv.org/abs/2106.09685) to adapt Mistral-7B efficiently:
- Base model is **frozen** (no training)
- We only train small LoRA adapter weights (`~0.02%` of the model)
- Optimizer: **Adafactor**
- Save adapters after training under the folder **`mistral_lora_output`**

Execute `python3 train_mistral_lora.py`

---

### 4. **Merge LoRA Adapters**
To export a standalone model (without PEFT dependency), we merge the LoRA adapters into the base model:
Execute 
```bash 
python3 merge_lora_and_push.py
```
The merged model is under the folder **`mistral_lora_merged`**
---

### 5. **Push to Hugging Face Hub (`push_to_hub.py`)**
To share the model, execute:
```bash
huggingface-cli login
python3 push_to_hub.py
```
---

## Inference Example 
### Option1. Through python script (`test_inference.py`) 
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "JQXavier/scaleway-agent"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "Create a Kubernetes Kapsule cluster named kapsule-90bd8d in region pl-waw with version 1.26.9"
instruction = f"### Instruction:\n{prompt}\n\n### Response:\n"
inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
You will receive the response:
```
Model Response

### Instruction:
Create a Kubernetes Kapsule cluster named kapsule-90bd8d in region pl-waw with version 1.26.9

### Response:
CLI:
```bash
kubermatic cluster create name=kapsule-90bd8d version=1.26.9 region=pl-waw
```
Terraform:
```hcl
resource "kubermatic_cluster" "kapsule-90bd8d" {
  name    = "kapsule-90bd8d"
  version = "1.26.9"
  region  = "pl-waw"
}
```
## Option2. Through Curl POST, after import the model into Scaleway Managed Inference
```bash
curl -X POST https://<YOUR_INFERENCE_ENDPOINT_URL>/v1/completions \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "scaleway-agent-v1",
    "prompt": "### Instruction:\nCreate a virtual instance named random-42 with image ubuntu and type GP1-L in zone fr-par-2, in the private network xyz.\n\n### Response:\n",
    "max_tokens": 300,
    "temperature": 0.0
  }'
```
You will receive response:
```bash
{"id":"cmpl-3ff3ee11eec0498ba55719e479b6d6bc","object":"text_completion","created":1746534178,"model":"scaleway-agent-v1","choices":[{"index":0,"text":" CLI: scw instance create name=random-42 image=ubuntu type=GP1-L zone=fr-par-2 network=xyz\nTerraform: resource \"scaleway_instance\" \"random-42\" {\n  name  = \"random-42\"\n  image = \"ubuntu\"\n  type  = \"GP1-L\"\n  zone  = \"fr-par-2\"\n  network {\n    id = scaleway_network.xyz.id\n  }\n}","logprobs":null,"finish_reason":"stop","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":48,"total_tokens":168,"completion_tokens":120,"prompt_tokens_details":null}}%  
```
---

## Folder Structure
```
├── preprocess_and_tokenize.py  # Tokenizes the JSONL dataset
├── train_mistral_lora.py       # Trains Mistral with LoRA
├── push_to_hub.py              # Uploads model to Hugging Face Hub
├── test_inference.py           # Runs inference locally
├── mistral_lora_output/        # Training outputs and checkpoints => Generated by train_mistral_lora.py
├── mistral_lora_merged/        # Final merged model (ready to push or use) => Generated by merge_lora.py
├── setup.sh                    # Install all dependancies
├── merge_lora.py               # Merge LoRA Adaptor layers to frozen base model
└── scaleway_3000_dataset.jsonl # Dataset sample (used for previous fine-tuning)
```
---

## Roadmaps
### Shortterm
#### V1.n
- Expand to support more Scaleway services
  - Databases (RDB, Redis)
  - Kubernetes, Serverless, Cockpit
  - IAM or billing commands

### Middleterm
#### V2
- Can search for the latest version of commands or syntax
  - Thinking to add feature of RAG to check the latest version of **Terraform** and **scw cli** as the evolution is fast
  - Add a agent or frontend to connect with

#### Further
- Can help Devops engineer work ONLY at IDE, rather than spending time on research and correction across different websites
  - Add features to check out via **Scaleway API** to avoid any possible error such as CIDR overlap or namespace conflicts before generate the response
  - Able to find out the reason of error while or before doing terraform plan/apply throughout the terraform script at local workspace or integrate into CI pipeline 
---

## About more information regarding why did we start this project and what did we recognise as constraints, please checkout in [Thinktank](./ThinkTank.md)

---

## Credits
Built by [JQXavier](https://huggingface.co/JQXavier) using Transformers + PEFT on **Scaleway L4 GPU**. Attention: as fine-tuning progresses with more datasets, the model size may increase, potentially requiring a more powerful instance to handle the task efficiently.

---