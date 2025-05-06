#!/usr/bin/env python3
"""
validate_dataset.py

Validate and clean the JSONL dataset to ensure each line is valid JSON
with keys 'prompt', 'cli', and 'terraform'. We also remove any blank lines or
malformed entries.
"""
import json

INPUT_FILE = "scaleway_1000_create_dataset.jsonl"
OUTPUT_FILE = "scaleway_1000_create_dataset_validated.jsonl"

required_keys = {"prompt", "cli", "terraform"}
valid_entries = []

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for lineno, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            # Skip empty lines
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Line {lineno}: invalid JSON, skipping. Error: {e}")
            continue
        # Check for required keys
        if not required_keys.issubset(entry.keys()):
            print(f"Line {lineno}: missing keys {required_keys - entry.keys()}, skipping.")
            continue
        # Optionally validate types (strings)
        if not all(isinstance(entry[k], str) for k in required_keys):
            print(f"Line {lineno}: some fields are not strings, skipping.")
            continue
        # Trim whitespace in fields (cleaning)
        entry = {k: entry[k].strip() for k in entry}
        valid_entries.append(entry)

print(f"Validated {len(valid_entries)} entries out of {lineno} lines.")

# Write cleaned data to a new JSONL file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
    for entry in valid_entries:
        json.dump(entry, fout, ensure_ascii=False)
        fout.write("\n")
