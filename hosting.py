from huggingface_hub import HfApi
import os

# 1. Get your secure token from environment variables
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set!")

api = HfApi(token=hf_token)

print("🚀 Starting Automated Deployment to Hugging Face...")

# 2. Define relative paths (Assuming host.py runs from the root of the repo)
backend_path = os.path.join(os.getcwd(), "backend", "deployment")
frontend_path = os.path.join(os.getcwd(), "frontend")

# 3. Upload the Backend Server
print(f"⏳ Uploading Backend API from {backend_path}...")
api.upload_folder(
    folder_path=backend_path, 
    repo_id="zaheergshaikh/visitusapi", 
    repo_type="space" 
)
print("✅ Backend deployed successfully.")

# 4. Upload the Frontend UI
print(f"⏳ Uploading Streamlit UI from {frontend_path}...")
api.upload_folder(
    folder_path=frontend_path, 
    repo_id="zaheergshaikh/visitusapp", 
    repo_type="space" 
)
print("✅ Frontend deployed successfully.")
