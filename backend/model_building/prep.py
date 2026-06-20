import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# 1. Environment-Aware Configuration
# Use environment variable for HF_TOKEN
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set!")

api = HfApi(token=hf_token)

# 2. Path Handling
# If running in CI/CD, use the working directory; otherwise, use Drive
base_dir = os.getcwd() if os.getenv("GITHUB_ACTIONS") else "/content/drive/MyDrive/visitusmlops"
data_path = os.path.join(base_dir, "data", "tourism.csv")
output_dir = os.path.join(base_dir, "data")

tourism_dataset = pd.read_csv(data_path)
tourism_dataset = tourism_dataset.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore')

# 3. Data Processing
target = 'ProdTaken'
numeric_features = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 
                    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 
                    'Passport', 'PitchSatisfactionScore', 'OwnCar', 
                    'NumberOfChildrenVisiting', 'MonthlyIncome']
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 
                        'ProductPitched', 'MaritalStatus', 'Designation']

X = tourism_dataset[numeric_features + categorical_features]
y = tourism_dataset[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Save and Upload
os.makedirs(output_dir, exist_ok=True)
files_to_upload = {"Xtrain.csv": Xtrain, "Xtest.csv": Xtest, "ytrain.csv": ytrain, "ytest.csv": ytest}

for file_name, df in files_to_upload.items():
    local_path = os.path.join(output_dir, file_name)
    df.to_csv(local_path, index=False)
    
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=file_name,  
        repo_id="zaheergshaikh/visitusdata",
        repo_type="dataset",
    )
    print(f"✅ Uploaded {file_name}")

print("🚀 All files processed and uploaded!")
