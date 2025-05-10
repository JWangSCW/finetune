from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi

local_model_dir = "mistral_lora_merged"
repo_id = "JQXavier/scaleway-agent"
branch_name = "main"  # Attention: HuggingFace does't support pull request as in Github, you cannot merge from "xx" to "main" directly

# Create the branch if it doesn't exist
# api = HfApi()
# branches = api.list_repo_refs(repo_id=repo_id).branches
# branch_names = [branch.name for branch in branches]
# if branch_name not in branch_names:
#     api.create_branch(repo_id=repo_id, branch=branch_name)

model = AutoModelForCausalLM.from_pretrained(local_model_dir)
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

model.push_to_hub(repo_id,commit_message="Pushing clean v1 model", revision=branch_name) 
tokenizer.push_to_hub(repo_id, commit_message="Pushing clean v1 model",revision=branch_name) 
print(f"Model and tokenizer pushed to: https://huggingface.co/{repo_id}/tree/{branch_name}")
