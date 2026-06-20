import pandas as pd
import os
import mlflow
import joblib
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report

# 1. Environment-Aware Pathing
# Use GITHUB_WORKSPACE for GitHub Actions or default to Drive for local dev
if os.getenv("GITHUB_ACTIONS"):
    # GitHub Action runner's temporary directory
    db_path = os.path.join(os.getcwd(), "mlruns.db")
else:
    db_path = "/content/drive/MyDrive/visitusmlops/backend/model_building/mlruns.db"

# Ensure the database directory exists
os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("visitus_training_experiment")

# 2. Load Datasets
print("⬇️ Loading datasets...")
Xtrain = pd.read_csv("hf://datasets/zaheergshaikh/visitusdata/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/zaheergshaikh/visitusdata/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/zaheergshaikh/visitusdata/ytrain.csv").squeeze()
ytest = pd.read_csv("hf://datasets/zaheergshaikh/visitusdata/ytest.csv").squeeze()

# 3. Model Pipeline
numeric_features = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 
                    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 
                    'Passport', 'PitchSatisfactionScore', 'OwnCar', 
                    'NumberOfChildrenVisiting', 'MonthlyIncome']
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 
                        'ProductPitched', 'MaritalStatus', 'Designation']

num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
preprocessor = make_column_transformer((num_pipeline, numeric_features), (cat_pipeline, categorical_features))

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)
model_pipeline = make_pipeline(preprocessor, xgb_model)

# 4. Training
print("⚙️ Starting XGBoost Training...")
with mlflow.start_run(run_name="XGBoost_CI_Run"):
    model_pipeline.fit(Xtrain, ytrain)
    
    # Simple evaluation
    y_pred = model_pipeline.predict(Xtest)
    report = classification_report(ytest, y_pred, output_dict=True)
    mlflow.log_metrics({"test_accuracy": report['accuracy']})
    mlflow.sklearn.log_model(model_pipeline, "visitus_best_xgb_model")

# 5. Save Artifact for Deployment
joblib.dump(model_pipeline, "visitus_xgb_model.joblib")
print("🚀 Model saved as visitus_xgb_model.joblib")
