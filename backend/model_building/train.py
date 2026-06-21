import pandas as pd
import os
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import joblib
import xgboost as xgb
import datetime
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from mlflow.models import infer_signature

# 1. Setup Tracking
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("visitus_training_experiment_01")

# 2. Load Datasets
Xtrain = pd.read_csv("hf://datasets/zaheergshaikh/visitusdata/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/zaheergshaikh/visitusdata/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/zaheergshaikh/visitusdata/ytrain.csv").squeeze()
ytest = pd.read_csv("hf://datasets/zaheergshaikh/visitusdata/ytest.csv").squeeze()

# 3. Preprocessing & Pipeline
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

# 4. Training Execution
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__learning_rate': [0.01, 0.1]
}

# Enable autologging for GridSearch child runs
mlflow.sklearn.autolog()

run_name = f"XGBoost_Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name):
    # Log Data Prep params for comparison
    mlflow.log_params({"num_imputer": "median", "cat_imputer": "most_frequent"})
    
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)
    
    best_model = grid_search.best_estimator_
    
    # Advanced Evaluation
    eval_data = Xtest.copy()
    eval_data["label"] = ytest
    signature = infer_signature(Xtrain, best_model.predict(Xtrain))
    model_info = mlflow.sklearn.log_model(best_model, "model", signature=signature)
    
    result = mlflow.evaluate(
        model=model_info.model_uri,
        data=eval_data,
        targets="label",
        model_type="classifier"
    )
    
    print(f"✅ Training complete. Best Params: {grid_search.best_params_}")

# 5. Local Export
os.makedirs("backend/model_building", exist_ok=True)
joblib.dump(best_model, "backend/model_building/visitus_xgb_model.joblib")
