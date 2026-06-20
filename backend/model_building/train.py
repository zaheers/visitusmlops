import pandas as pd
import os
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import joblib
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# 1. Network-Aware Tracking (Environment-aware to prevent CI/CD breakage)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"🔗 Connecting to remote MLflow at: {tracking_uri}")
else:
    print("🏠 No remote MLflow URI found. Using local ./mlruns directory.")

mlflow.set_experiment("visitus_training_experiment_01")

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

# 4. Hyperparameter Grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__learning_rate': [0.01, 0.1]
}

# 5. Training with GridSearchCV and Autologging
print("⚙️ Starting XGBoost GridSearchCV Training with Autologging...")

# Enable autologging: Automatically logs best_params, CV scores, and the best model
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="XGBoost_CI_Run"):
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)
    
    # Extract the best model
    best_model = grid_search.best_estimator_
    
    # Manual evaluation logging
    y_pred = best_model.predict(Xtest)
    report = classification_report(ytest, y_pred, output_dict=True)
    mlflow.log_metrics({"test_accuracy": report['accuracy']})
    
    print(f"✅ Training complete. Best Params: {grid_search.best_params_}")
    print("✅ Run logged successfully to tracking server.")

# 6. Save Artifact for local use (Matches hosting.py expectations)
os.makedirs("backend/model_building", exist_ok=True)
joblib.dump(best_model, "backend/model_building/visitus_xgb_model.joblib")
print("🚀 Model saved to backend/model_building/visitus_xgb_model.joblib")
