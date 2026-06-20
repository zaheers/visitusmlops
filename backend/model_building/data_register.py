from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os
from google.colab import userdata

# Configuration
repo_id = "zaheergshaikh/visitusdata"
repo_type = "dataset"
# Pointing to our verified data folder
data_folder = "/content/drive/MyDrive/visitusmlops/data" 

# Initialize API client using Colab secrets
api = HfApi(token=userdata.get('HF_TOKEN'))

# Step 1: Check/Create Repository
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"⚠️ Repository not found. Creating '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"✅ Repository '{repo_id}' created.")

# Step 2: Upload folder
api.upload_folder(
    folder_path=data_folder,
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"🚀 Data successfully registered to {repo_id}")
