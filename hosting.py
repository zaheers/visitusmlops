from huggingface_hub import HfApi
from google.colab import userdata
import os

# 1. Get your secure token
hf_token = userdata.get('HF_TOKEN')
api = HfApi(token=hf_token)

print("🚀 Starting Automated Deployment to Hugging Face...")

# 2. Upload the Backend Server
print("⏳ Uploading Backend API...")
api.upload_folder(
    folder_path="/content/drive/MyDrive/visitusmlops/backend/deployment", 
    repo_id="zaheergshaikh/visitusapi", 
    repo_type="space" 
)
print("✅ Backend deployed successfully.")

# 3. Upload the Frontend UI
print("⏳ Uploading Streamlit UI...")
api.upload_folder(
    folder_path="/content/drive/MyDrive/visitusmlops/frontend", 
    repo_id="zaheergshaikh/visitus-ui", 
    repo_type="space" 
)
print("✅ Frontend deployed successfully.")
