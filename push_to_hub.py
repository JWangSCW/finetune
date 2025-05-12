# Method 1. To save the model and push to Hugging Face Hub in format .safetensors
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

#----------------------------------------------------------------------------------------------
# Method 2. To save the model and push to Hugging Face Hub in format .bin

# from huggingface_hub import HfApi, upload_folder

# repo_id= "JQXavier/Scaleway-Agent-Stable"
# branch_name = "main" # Attention: Hugging Face does't support pull request as in Github, you cannot merge from "xx" to "main" directly
# local_model_dir = "mistral_lora_merged"

# # Create the branch if it doesn't exist
# # api = HfApi()
# # branches = api.list_repo_refs(repo_id=repo_id).branches
# # branch_names = [branch.name for branch in branches]
# # if branch_name not in branch_names:
# #     api.create_branch(repo_id=repo_id, branch=branch_name)

# # Push the local .bin-based model folder as-is (no .safetensors will be created)
# upload_folder(
#     repo_id=repo_id,
#     folder_path=local_model_dir,
#     path_in_repo=".",
#     repo_type="model",
#     revision=branch_name,
#     commit_message="Clean .bin push only – remove .safetensors",
#     delete_patterns=["*.safetensors", "*.safetensors.index.json"],
#     allow_patterns=["*.bin", "*.json", "*.model", "*.txt"]
# )
