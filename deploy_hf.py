import os
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN") # Securely handle token via environment variable
username = "RHAMPRASSATH"
repo_name = "MTDNet-Alzheimer-Detector"
repo_id = f"{username}/{repo_name}"

api = HfApi(token=token)

print(f"Creating Hugging Face Space: {repo_id}...")
try:
    create_repo(repo_id, repo_type="space", space_sdk="docker", token=token, private=False)
    print("Space created successfully.")
except Exception as e:
    print("Space might already exist or error:", e)

# Files to upload
files_mapping = {
    "api.py": "api.py",
    "predict.py": "predict.py",
    "model.py": "model.py",
    "train_v4.py": "train_v4.py",
    "utils.py": "utils.py",
    "Dockerfile": "Dockerfile",
    "requirements_hf.txt": "requirements.txt",
    "mtdnet_best_npz.pth": "mtdnet_best_npz.pth",
    "mtdnet_v4_best.pth": "mtdnet_v4_best.pth",
    "mtdnet_v5_best.pth": "mtdnet_v5_best.pth"
}

print("Uploading files to Space...")
base_path = r"d:\al project"

for local_name, remote_name in files_mapping.items():
    local_path = os.path.join(base_path, local_name)
    if os.path.exists(local_path):
        print(f"Uploading {local_name} as {remote_name}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="space",
            commit_message=f"Deploying {remote_name}"
        )
    else:
        print(f"WARNING: File {local_name} not found locally.")

print("All done! Your Space is being built.")
