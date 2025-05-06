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

## Inference Example 

### Option1. Through python script (`test_inference.py`) 

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "JQXavier/scaleway-agent"  # or "mistral_lora_merged" if you wish to test it locally
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
└── merge_lora.py               # Merge LoRA Adaptor layers to frozen base model
```

---

## Next Steps

- Integrate with a OpenWebUI to have a decent frontend interface
- Expand to support more Scaleway services
  - Databases (RDB, Redis)
  - Kubernetes, Serverless, Cockpit
  - IAM or billing commands
  - Extending dataset coverage will require preparing **validated JSONL files** in the same prompt/response format.

---

## Thinktank

### Why we chose an Instruction-based model over a Chat-native one ?
The core objective of this project is to generate deterministic, copy-paste-ready Scaleway CLI and Terraform commands from natural language prompts. Our users aren’t asking for conversational insights — they need infrastructure scripts they can trust and use immediately.
That’s why we opted for an instruction-tuned model (Mistral-7B-Instruct) over chat-centric ones. This enables us to build a prompt-response interface that behaves more like a doc-generation assistant than a chatbot.
A future evolution toward a chat model could make sense (e.g. with tool usage or contextual memory), but for our MVP — predictability and structure matter more than interactivity.

### Why we used LoRA + PEFT instead of full fine-tuning ?
Full fine-tuning a 7B model is resource-intensive, requiring multiple high-memory GPUs and long training cycles. Instead, we used LoRA (Low-Rank Adaptation) in conjunction with PEFT to freeze the base Mistral model and only train lightweight adapter layers.
This allowed us to:
- Train on a single NVIDIA L4 GPU (24GB vRAM)
- Process 3000+ samples in under 30 minutes
- Fine-tune just 0.02% of the model parameters
- Merge the adapters post-training for standalone usage
This approach is ideal for narrow-domain adaptation (Scaleway infra tasks) where speed and compute efficiency matter more than generalization.

### Key constraints and challenges
#### Dataset quality matters more than quantity
In our context, prompt engineering + data curation define success. Poor data leads to unusable generations (“garbage in, garbage out”).
Unlike AWS, Scaleway does not have a public Knowledge Graph or schema registry. That’s why we bootstrapped:
- A high-quality dataset of 1000 handcrafted examples covering VPC, instances, volumes, and buckets.
- A second wave of 3000+ semi-automated examples to scale coverage — though some may be outdated or require review due to API drift.

####  Dependency and versioning issues
Initially, we used a naive install method (pip install --upgrade ...) which led to compatibility issues between transformers, peft, datasets, and bitsandbytes. We then locked specific library versions in setup.sh to maintain stability — but these will require future review to keep up with breaking changes.

#### Output format & inference constraints
- Model outputs vary unless temperature=0.0 is used. This is mandatory for deterministic CLI/Terraform generation.
- Even then, small variations in naming, structure, or whitespace may occur — consider adding post-processing or regex cleanup if strict formatting is required.
- The model does not enforce correctness regarding Scaleway resource availability, quotas, or dependencies — outputs should be reviewed before deployment.

#### Dataset coverage limitation:
As of now, the model has been trained and validated primarily on:
- Compute: Instances, Volumes
- Storage: Object Storage (buckets)
- Network: VPC, Public Gateways, Private Networks, IPs


#### CLI & Terraform format strictness:
- The CLI commands are trained to follow [`scaleway-cli`](https://github.com/scaleway/scaleway-cli/tree/master/docs/commands)
- The Terraform output follows [`scaleway/scaleway` provider syntax](https://registry.terraform.io/providers/scaleway/scaleway/latest/docs)
Despite this, hallucinations may occur — especially for deprecated fields or older syntax patterns — due to outdated examples in the dataset.

#### Inference Testing
- Always use temperature=0.0 for deterministic output
- Prompt format must follow ### Instruction: and ### Response: even when using curl
- Confirm padding token is set (tokenizer.pad_token = tokenizer.eos_token)

---

## Credits

Built by [JQXavier](https://huggingface.co/JQXavier) using 🤗 Transformers + PEFT on **Scaleway L4 GPU**.

---