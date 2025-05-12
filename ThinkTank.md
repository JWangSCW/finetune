# Thinktank

---
## Why we chose an Instruction-based model over a Chat-native one or just do RAG?

We chose an instruction-based model because our users need precise, copy-paste-ready CLI and Terraform commands. They aren’t looking for a conversation — they want immediate, reliable outputs. An instruction-tuned model like Mistral-7B-Instruct fits this perfectly. It gives us one clear answer for one clear prompt, plus it behaves more like a doc generator than a chatbot.
A chat model could be useful later — for tool use, memory, or multi-turn flows.But for this case, we need simplicity, speed, and control — not interactivity. 
RAG is also powerful, but it needs more infrastructure: a vector database, a retriever, more development time. We already have the tools at Scaleway **Jean-Cloud**, but it's not the fastest way to deliver value.
With this approach, we fine-tune quickly on a single GPU. It’s cost-effective, deterministic, and fully aligned with our use case.

##

## Why we used LoRA + PEFT instead of full fine-tuning ?

Full fine-tuning a 7B model is resource-intensive, requiring multiple high-memory GPUs and long training cycles. Instead, we used LoRA (Low-Rank Adaptation) in conjunction with PEFT to freeze the base Mistral model and only train lightweight adapter layers.
This allowed us to:
- Train on a single NVIDIA L4 GPU (24GB vRAM)
- Process 3000+ samples in under 30 minutes
- Fine-tune just 0.02% of the model parameters
- Merge the adapters post-training for standalone usage
This approach is ideal for narrow-domain adaptation (Scaleway infra tasks) where speed and compute efficiency matter more than generalization.

## **Key constraints and challenges**

### Dataset quality matters more than quantity

In our context, prompt engineering + data curation define success. Poor data leads to unusable generations (*garbage in, garbage out*).
Unlike AWS, Scaleway does not have a public Knowledge Graph or schema registry. That’s why we bootstrapped:
- A high-quality dataset of 1000 handcrafted examples covering VPC, instances, volumes, and buckets.
- A second wave of 3000+ semi-automated examples to scale coverage — though some may be outdated or require review due to API drift.

### Dataset coverage limitation:

As of now, the model has been trained and validated primarily on:
- Compute: Instances, Volumes
- Storage: Object Storage (buckets)
- Network: VPC, Public Gateways, Private Networks, IPs

### Dependency and versioning issues

Initially, we used a naive install method (`pip install --upgrade ...`) which led to compatibility issues between transformers, peft, datasets, and bitsandbytes. We then locked specific library versions in setup.sh to maintain stability — but these will require future review to keep up with breaking changes.

### Output format & inference constraints

- Model outputs vary unless `temperature=0.0` is used. This is mandatory for deterministic CLI/Terraform generation.
- Even then, small variations in naming, structure, or whitespace may occur — consider adding post-processing or regex cleanup if strict formatting is required.
- The model does not enforce correctness regarding Scaleway resource availability, quotas, or dependencies — outputs should be reviewed before deployment.

### Require tokenizer.pad_token = tokenizer.eos_token for training and inference

A little bit knowledge to share about these two:
- **pad_token**: Fills empty space when we need to make all sequences the same length in a batch. Example: <pad>
- **eos_token**	End of Sequence — tells the model “this is the end of the real content.” Example: <eos>
Suppose we have 2 tokenized sentences:

```
"create server"
"delete"
```

After tokenization (with made-up token IDs):

```
"create server" => [101, 202]
"delete"        => [303]
```

They’re different lengths, so we need to pad. if we set `tokenizer.pad_token = tokenizer.eos_token`, let's say <eos>=999, we get padded results:

```
[
 [101, 202, 999],     # "create server" + <eos>
 [303, 999, 999]      # "delete" + <eos> <eos>
]
```

If we don’t set pad_token, or it's undefined, Hugging Face Trainer may crash or warn: **pad_token is not set**, or it may use token ID 0, which might not be safe for your model, Example:

```
[101, 202,   0]
[303,   0,   0]
```

### Storage limlitation

As you know fine-tune a model can be a huge workload, which consume a lot storage space in an instance, if you see this kind of error which means you are running out of disk, instead of mounting another one, you can checkout firstly the disk space (`df -h`)

```
Filesystem      Size  Used Avail Use% Mounted on
tmpfs           4.8G  1.2M  4.8G   1% /run
efivarfs         56K   18K   34K  35% /sys/firmware/efi/efivars
/dev/sda1       112G  112G   12M 100% /
tmpfs            24G     0   24G   0% /dev/shm
tmpfs           5.0M     0  5.0M   0% /run/lock
/dev/sda16      881M  127M  693M  16% /boot
/dev/sda15      105M  6.1M   99M   6% /boot/efi
tmpfs           4.8G   12K  4.8G   1% /run/user/0
```

Then clean out the huggingface caching checkpoint (`rm -rf ~/.cache/huggingface`), we only need to upload final checkpoint to HuggingFace Hub

```
Filesystem      Size  Used Avail Use% Mounted on
tmpfs           4.8G  1.2M  4.8G   1% /run
efivarfs         56K   18K   34K  35% /sys/firmware/efi/efivars
/dev/sda1       112G   31G   82G  28% /
tmpfs            24G     0   24G   0% /dev/shm
tmpfs           5.0M     0  5.0M   0% /run/lock
/dev/sda16      881M  127M  693M  16% /boot
/dev/sda15      105M  6.1M   99M   6% /boot/efi
tmpfs           4.8G   12K  4.8G   1% /run/user/0
```

### CLI & Terraform format strictness:

- The CLI commands are trained to follow [`scaleway-cli`](https://github.com/scaleway/scaleway-cli/tree/master/docs/commands)
- The Terraform output follows [`scaleway/scaleway` provider syntax](https://registry.terraform.io/providers/scaleway/scaleway/latest/docs)
Despite this, hallucinations may occur — especially for deprecated fields or older syntax patterns — due to outdated examples in the dataset.

### Inference Testing

- Always use `temperature=0.0` for deterministic output
- Prompt format must follow **### Instruction:** and **### Response:** even when using curl

### Limitation of Scaleway's BYOM (beta version) regarding PyTorch .bin shards

- We can import both *.safetensors* shards and PyTorch *.bin* shards. However, since .bin does not explicitly store metadata, you cannot detect if the model is quantized from the file itself.
- Managed Inference fully supports models with *.safetensors* shards. When attempting to deploy models saved in *.bin* format, you may not find any compatible GPU offer.The Managed Inference team has confirmed they will investigate the format differences, and plan to improve the import experience: if a model format is unsupported, the platform will block the upload upfront to avoid confusion.

### Save LoRA adaptor in frozen model in .safetensor or in .bin

- If you want to know why we save the same model in different format:
Both formats are used to **save and load model weights** in PyTorch. 
- *.safetensor* is the default format when you call `model.save_pretrained()` in Hugging Face Transformers, you will see the saved shards such as:
```
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
model.safetensors.index.json
```
- Otherwise you need to add `safe_serialization=False`. You will see shards saved in *.bin*, for example:
```
pytorch_model-00001-of-00003.bin
pytorch_model-00002-of-00003.bin
pytorch_model-00003-of-00003.bin
pytorch_model.bin.index.json
```

The key differences:

|         Format         |                      *.safetensors*                   |              *.bin* (PyTorch default)            |
|------------------------|-------------------------------------------------------|--------------------------------------------------|
| Safety                 | Secure, prevents arbitrary code execution             | Standard, no extra safety                        |
| Speed                  | Faster loading                                        | Slightly slower on large models                  |
| Large model support    | Optimized for multi-shard large models (7B, 13B…)     | Works but less optimized                         |
| Corruption protection  | Immutable once created                                | Modifiable, higher risk of accidental corruption |
| Ecosystem              | Supported by Hugging Face `transformers`, `diffusers` | Native in PyTorch                                |
| File extension         | `model.safetensors`                                   | `pytorch_model.bin`                              |


### Upload fine-tuned model in .safetensor or in .bin

When you use `AutoModelForCausalLM.from_pretrained().push_to_hub()`, it internally calls `AutoModelForCausalLM.from_pretrained().save_pretrained()` without specifying the *safe_serialization* parameter. By default, *safe_serialization* is set to True, which means the model is saved in the *.safetensors* format.
If you want to save and upload your model in the *.bin* format, you need to explicitly set `safe_serialization=False` when calling `save_pretrained()`. However, `push_to_hub()` doesn't provide a direct way to pass this parameter. Therefore, to upload a model in *.bin* format, you should first save the model locally with `safe_serialization=False` and then use `upload_folder()` to upload the saved files to the Hugging Face Hub.

---

# Some interesting articles

- [LoRA and PEFT: Fine-Tuning Large Language Models in a Cost-Effective Way](https://medium.com/@camillanawaz/lora-and-peft-fine-tuning-large-language-models-in-a-cost-effective-way-2340f88c77a5)
- [Fine-Tuning Large Language Models with PEFT (LoRA) and Rouge Score: A Comprehensive Hands-On Guide](https://bobrupakroy.medium.com/fine-tuning-large-language-models-with-peft-lora-and-rogue-score-a-comprehensive-hands-on-guide-3d54179125f0)
- [LoRA and PEFT: Fine-Tuning Large Language Models in a Cost-Effective Way](https://ai.plainenglish.io/understanding-low-rank-adaptation-lora-for-efficient-fine-tuning-of-large-language-models-082d223bb6db)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/en/main_classes/model)
- [How to fine-tune: Focus on effective datasets](https://ai.meta.com/blog/how-to-fine-tune-llms-peft-dataset-curation/)

