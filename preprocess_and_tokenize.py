from datasets import load_dataset
from transformers import AutoTokenizer

# Load the Hugging Face tokenizer and model
# model_id = "mistral-7B-Instruct"  # or "JQXavier/scaleway-agent" if you want to use the original model
model_id = "JQXavier/scaleway-agent"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load the validated JSONL dataset
dataset = load_dataset("json", data_files={"train": "scaleway_3000_dataset.jsonl"})["train"]

def tokenize(ds):
    prompt = ds["prompt"]
    cli = ds["cli"]
    tf_code = ds["terraform"]
    instruction = f"### Instruction:\n{prompt}\n\n### Response:\n"
    response = f"CLI: {cli}\nTerraform: {tf_code}"

    # Tokenize the instruction and response
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

    # Concatenate input IDs and create labels
    input_ids = instr_tokens["input_ids"] + resp_tokens["input_ids"] + [tokenizer.eos_token_id]
    # Labels are -100 for the instruction part and the response part is the same as input_ids
    labels = [-100] * len(instr_tokens["input_ids"]) + resp_tokens["input_ids"] + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "labels": labels}

# Tokenize the dataset, using the preprocess function and removing the original columns
tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Hugging Face Trainer + default_data_collator
print("Tokenized dataset ready:", tokenized_dataset[0])
# Save the tokenized dataset locally to disk for training
tokenized_dataset.save_to_disk("tokenized_scaleway_dataset")
