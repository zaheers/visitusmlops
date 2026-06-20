import shutil
import os
from huggingface_hub import HfApi

# Setup
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set!")
api = HfApi(token=hf_token)

# Paths
backend_deploy_path = os.path.join(os.getcwd(), "backend", "deployment")
model_src = os.path.join(os.getcwd(), "backend", "model_building", "visitus_xgb_model.joblib")
model_dest = os.path.join(backend_deploy_path, "visitus_xgb_model.joblib")

# Robust Copy Logic
if os.path.exists(model_src):
    shutil.copy(model_src, model_dest)
    print(f"✅ Model successfully copied to {model_dest}")
else:
    raise FileNotFoundError(f"❌ Critical: Model file not found at {model_src}. Pipeline artifacts failed to transfer.")

# Upload Backend
print("🚀 Uploading Backend...")
api.upload_folder(
    folder_path=backend_deploy_path,
    repo_id="zaheergshaikh/visitusapi",
    repo_type="space",
    commit_message="CI/CD: Deployment of backend API and model"
)

# Upload Frontend
print("🚀 Uploading Frontend...")
api.upload_folder(
    folder_path=os.path.join(os.getcwd(), "frontend"),
    repo_id="zaheergshaikh/visitusapp",
    repo_type="space",
    commit_message="CI/CD: Deployment of frontend UI"
)

# Fix README
readme_content = "---\ntitle: Visitus UI\nemoji: 🚀\nsdk: streamlit\napp_file: app.py\n---"
api.upload_file(
    path_or_fileobj=readme_content.encode(),
    path_in_repo="README.md",
    repo_id="zaheergshaikh/visitusapp",
    repo_type="space"
)
