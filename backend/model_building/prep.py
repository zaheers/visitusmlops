# for data manipulation
import pandas as pd
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import HfApi
from google.colab import userdata

# Initialize API client using Colab Secrets
api = HfApi(token=userdata.get('HF_TOKEN'))

# Define constants for the dataset path (pointing to your local data folder)
DATASET_PATH = "/content/drive/MyDrive/visitusmlops/data/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)

# Drop identifier columns that shouldn't be in the model
tourism_dataset = tourism_dataset.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore')
print("✅ Dataset loaded and cleaned successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the tourism dataset
numeric_features = [
    'Age', 
    'CityTier', 
    'DurationOfPitch', 
    'NumberOfPersonVisiting', 
    'NumberOfFollowups', 
    'PreferredPropertyStar', 
    'NumberOfTrips', 
    'Passport', 
    'PitchSatisfactionScore', 
    'OwnCar', 
    'NumberOfChildrenVisiting', 
    'MonthlyIncome'
]

# List of categorical features in the tourism dataset
categorical_features = [
    'TypeofContact', 
    'Occupation', 
    'Gender', 
    'ProductPitched', 
    'MaritalStatus', 
    'Designation'
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]

# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

# Set the output directory for the split files to stay organized
output_dir = "/content/drive/MyDrive/visitusmlops/data/"

# Save locally to your data folder before uploading
Xtrain.to_csv(os.path.join(output_dir, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(output_dir, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(output_dir, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(output_dir, "ytest.csv"), index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload to your VisitUs Hugging Face dataset repository
for file_name in files:
    local_path = os.path.join(output_dir, file_name)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=file_name,  
        repo_id="zaheergshaikh/visitusdata", # Your correct HF repo!
        repo_type="dataset",
    )
print("🚀 Train/Test splits successfully uploaded to Hugging Face: zaheergshaikh/visitusdata!")
