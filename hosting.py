 
# hosting.py - Updated Deployment Logic
import shutil
import os
from huggingface_hub import HfApi

# 1. Setup
hf_token = os.getenv('HF_TOKEN')
api = HfApi(token=hf_token)

# 2. Prepare Backend (Copy Model)
backend_path = os.path.join(os.getcwd(), "backend", "deployment")
model_src = os.path.join(os.getcwd(), "backend", "model_building", "visitus_xgb_model.joblib")

if os.path.exists(model_src):
    shutil.copy(model_src, os.path.join(backend_path, "visitus_xgb_model.joblib"))
    print("✅ Model copied to deployment folder.")

# 3. Upload Backend
api.upload_folder(
    folder_path=backend_path, 
    repo_id="zaheergshaikh/visitusapi", 
    repo_type="space" 
)

# 4. Upload Frontend
# This will put app.py and requirements.txt at the ROOT of the space
frontend_path = os.path.join(os.getcwd(), "frontend")
api.upload_folder(
    folder_path=frontend_path, 
    repo_id="zaheergshaikh/visitusapp", 
    repo_type="space" 
)

# 5. Fix Config: Point to app.py at the ROOT
readme_content = """---
title: Visitus UI
emoji: 🚀
sdk: streamlit
app_file: app.py
---
"""
api.upload_file(
    path_or_fileobj=readme_content.encode(),
    path_in_repo="README.md",
    repo_id="zaheergshaikh/visitusapp",
    repo_type="space"
)
