import shutil
import os
from huggingface_hub import HfApi

# 1. Setup
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set!")
api = HfApi(token=hf_token)

# 2. Paths (Using absolute paths for reliability in CI/CD)
cwd = os.getcwd()
backend_deploy_path = os.path.join(cwd, "backend", "deployment")
model_src = os.path.join(cwd, "backend", "model_building", "visitus_xgb_model.joblib")
model_dest = os.path.join(backend_deploy_path, "visitus_xgb_model.joblib")

# Debugging information (Visible in GitHub Actions logs)
print(f"DEBUG: Current Working Directory: {cwd}")
print(f"DEBUG: Looking for model at: {model_src}")

# 3. Robust Copy Logic
if os.path.exists(model_src):
    shutil.copy(model_src, model_dest)
    print(f"✅ Model successfully copied to {model_dest}")
else:
    # This listing helps us see if the artifact was downloaded to the wrong subfolder
    print("❌ Model not found! Listing files in backend/model_building/ to debug:")
    try:
        print(os.listdir(os.path.join(cwd, "backend", "model_building")))
    except FileNotFoundError:
        print("Directory 'backend/model_building/' does not exist.")
    raise FileNotFoundError(f"Critical: Model file not found at {model_src}.")

# 4. Upload Backend
print("🚀 Uploading Backend...")
api.upload_folder(
    folder_path=backend_deploy_path,
    repo_id="zaheergshaikh/visitusapi",
    repo_type="space",
    commit_message="CI/CD: Deployment of backend API and model"
)

# 5. Upload Frontend
print("🚀 Uploading Frontend...")
api.upload_folder(
    folder_path=os.path.join(cwd, "frontend"),
    repo_id="zaheergshaikh/visitusapp",
    repo_type="space",
    commit_message="CI/CD: Deployment of frontend UI"
)

# 6. Fix README
readme_content = "---\ntitle: Visitus UI\nemoji: 🚀\nsdk: streamlit\napp_file: app.py\n---"
api.upload_file(
    path_or_fileobj=readme_content.encode(),
    path_in_repo="README.md",
    repo_id="zaheergshaikh/visitusapp",
    repo_type="space"
)
