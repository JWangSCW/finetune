#!/usr/bin/env python3
"""
preprocess_and_tokenize.py

Load the JSONL dataset, format each example as an instruction-response pair,
and tokenize using the Mistral-7B-Instruct tokenizer.
"""
from datasets import load_dataset
from transformers import AutoTokenizer

# Load the Hugging Face tokenizer for Mistral-7B-Instruct
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # ✅ Fix: set pad_token

# Load the validated JSONL dataset
dataset = load_dataset("json", data_files={"train": "scaleway_1000_create_dataset_validated.jsonl"})["train"]

def preprocess_fn(example):
    prompt = example["prompt"]
    cli = example["cli"]
    tf_code = example["terraform"]
    instruction = f"### Instruction:\n{prompt}\n\n### Response:\n"
    response = f"CLI: {cli}\nTerraform: {tf_code}"

    instr_tokens = tokenizer(
        instruction,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )

    resp_tokens = tokenizer(
        response,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )

    input_ids = instr_tokens["input_ids"] + resp_tokens["input_ids"] + [tokenizer.eos_token_id]
    labels = [-100] * len(instr_tokens["input_ids"]) + resp_tokens["input_ids"] + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "labels": labels}

# Tokenize and remove raw columns
tokenized_dataset = dataset.map(preprocess_fn, remove_columns=dataset.column_names)

# ❌ DO NOT manually call tokenizer.pad (it injects bad attention_mask)
# Let Hugging Face Trainer + default_data_collator handle it dynamically
print("Tokenized dataset ready. Example:", tokenized_dataset[0])
tokenized_dataset.save_to_disk("tokenized_scaleway_dataset")
