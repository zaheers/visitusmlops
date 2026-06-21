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
from sklearn.metrics import classification_report, accuracy_score, f1_score
from mlflow.models import infer_signature

# 1. Setup Tracking
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri and tracking_uri.strip():
    mlflow.set_tracking_uri(tracking_uri)
    print(f"🔗 Connecting to remote MLflow at: {tracking_uri}")
else:
    print("🏠 No remote URI found. Using local ./mlruns directory.")

mlflow.set_experiment("visitus_training_experiment_01")

# 2. Load Datasets
Xtrain = pd.read_csv("hf://datasets/zaheergshaikh/visitusdata/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/zaheergshaikh/visitusdata/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/zaheergshaikh/visitusdata/ytrain.csv").squeeze()
ytest = pd.read_csv("hf://datasets/zaheergshaikh/visitusdata/ytest.csv").squeeze()

# 3. Pipeline
numeric_features = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 
                    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 
                    'Passport', 'PitchSatisfactionScore', 'OwnCar', 
                    'NumberOfChildrenVisiting', 'MonthlyIncome']
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 
                        'ProductPitched', 'MaritalStatus', 'Designation']

preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric_features),
    (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')), categorical_features)
)

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
model_pipeline = make_pipeline(preprocessor, xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42))

# 4. Training
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__learning_rate': [0.01, 0.1]
}

run_name = f"XGBoost_Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name):
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)
    best_model = grid_search.best_estimator_
    
    # Log Parameters AFTER fitting
    mlflow.log_params({
        "num_imputer": "median", 
        "cat_imputer": "most_frequent",
        "best_n_estimators": grid_search.best_params_['xgbclassifier__n_estimators'],
        "best_max_depth": grid_search.best_params_['xgbclassifier__max_depth'],
        "best_learning_rate": grid_search.best_params_['xgbclassifier__learning_rate']
    })
    
    # Log Metrics
    y_pred = best_model.predict(Xtest)
    mlflow.log_metrics({
        "test_accuracy": accuracy_score(ytest, y_pred),
        "test_f1": f1_score(ytest, y_pred, average='weighted')
    })
    
    # Conditional Advanced Evaluation
    if tracking_uri and tracking_uri.startswith("http"):
        print("📊 Running advanced evaluation...")
        eval_data = Xtest.copy()
        eval_data["label"] = ytest
        signature = infer_signature(Xtrain, best_model.predict(Xtrain))
        model_info = mlflow.sklearn.log_model(best_model, "model", signature=signature)
        mlflow.evaluate(model=model_info.model_uri, data=eval_data, targets="label", model_type="classifier")
    
    print(f"✅ Training complete. Best Params: {grid_search.best_params_}")

# 5. Local Export
os.makedirs("backend/model_building", exist_ok=True)
joblib.dump(best_model, "backend/model_building/visitus_xgb_model.joblib")
print("🚀 Model saved to backend/model_building/visitus_xgb_model.joblib")
