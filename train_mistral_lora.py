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
model_name = "JQXavier/scaleway-agent"   # TBD if you want to use a different model from scratch or another model, such as "Mistral-7B-Instruct"
# revision = "13c7d575e52205157085cffb9a7c5de260a8a650" # Optional: branch/commit of the model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False) # Optional: revision=revision if necessary
tokenizer.pad_token = tokenizer.eos_token  # Avoid warnings about pad token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  ## Use efficient 16-bit format
    device_map="auto"   # Automatically place model on available GPU(s)
)  # Optional: revision=revision if necessary

# Set padding token ID explicitly in the model config, avoid warnings
model.config.pad_token_id = tokenizer.pad_token_id

# Configure LoRA adapters: small, trainable weights injected into key model layers
lora_config = LoraConfig(
    r=4,                         # Rank of the adapter
    lora_alpha=16,              # Scaling factor
    lora_dropout=0.05,          # Dropout to avoid overfitting
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific attention layers
    bias="none",                # Don’t train bias
    task_type="CAUSAL_LM"       # Causal Language Modeling task
)

# Inject LoRA adapters into the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Logs how many parameters are actually being trained

# Training arguments
training_args = TrainingArguments(
    output_dir="mistral_lora_output",         # Where to save checkpoints
    per_device_train_batch_size=1,            # Very small batch size (due to large model)
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,            # Accumulate gradients over 8 steps
    num_train_epochs=3,                       # Number of training epochs
    bf16=True,                                # Use bfloat16 precision (efficient and supported on modern GPUs)
    optim="adafactor",                        # Lightweight optimizer (better than Adam for large models)
    logging_dir="logs",                       # Logging directory
    logging_steps=50,
    evaluation_strategy="steps",              # Run evaluation every few steps
    eval_steps=200,
    save_strategy="steps",                    # Save checkpoint every N steps
    save_steps=200,
    save_total_limit=2,                       # Keep only the 2 latest checkpoints
    learning_rate=1e-4,
    lr_scheduler_type="cosine",               # Learning rate schedule
    report_to="none",                         # Don’t report to W&B or TensorBoard
    remove_unused_columns=False               # Required for PEFT compatibility
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
