#!/usr/bin/env python3
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    default_data_collator
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

# Load tokenized dataset
dataset = load_from_disk("tokenized_scaleway_dataset")
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# Model and Tokenizer Setup
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # just to avoid warning
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA Adapter Setup
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training Arguments
training_args = TrainingArguments(
    output_dir="mistral_lora_output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    bf16=True,
    optim="adafactor",
    logging_dir="logs",
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    report_to="none",
    remove_unused_columns=False
)

# Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator
)

# Launch Training
trainer.train()

# Save LoRA Fine-Tuned Model
trainer.save_model("mistral_lora_output/final_checkpoint")
