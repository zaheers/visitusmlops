from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Configuration
repo_id = "zaheergshaikh/visitusdata"
repo_type = "dataset"

# Use environment variable instead of Google Colab's userdata
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set!")

# Dynamic pathing: Use current directory structure
# Assumes the 'data' folder is at the root of your repo
data_folder = os.path.join(os.getcwd(), "data") 

# Initialize API client
api = HfApi(token=hf_token)

# Step 1: Check/Create Repository
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"⚠️ Repository not found. Creating '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"✅ Repository '{repo_id}' created.")

# Step 2: Upload folder
print(f"📤 Uploading data from {data_folder}...")
api.upload_folder(
    folder_path=data_folder,
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"🚀 Data successfully registered to {repo_id}")
