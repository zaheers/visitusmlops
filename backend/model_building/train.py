import pandas as pd
import os
import mlflow
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.metrics import classification_report

# 1. Point to your local Google Drive MLflow Database
db_path = "/content/drive/MyDrive/visitusmlops/backend/model_building/mlruns.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# Log to the experiment
mlflow.set_experiment("visitus_training_experiment")

# 2. Load Pre-Split Dataset from Hugging Face
print("⬇️ Downloading datasets from Hugging Face...")
Xtrain_path = "hf://datasets/zaheergshaikh/visitusdata/Xtrain.csv"
Xtest_path = "hf://datasets/zaheergshaikh/visitusdata/Xtest.csv"
ytrain_path = "hf://datasets/zaheergshaikh/visitusdata/ytrain.csv"
ytest_path = "hf://datasets/zaheergshaikh/visitusdata/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)

# .squeeze() converts the single-column DataFrame into a 1D Series for XGBoost
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()
print("✅ Cloud datasets loaded successfully.")

# 3. Calculate Class Weights for Imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# 4. Define Features
numeric_features = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 
                    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 
                    'Passport', 'PitchSatisfactionScore', 'OwnCar', 
                    'NumberOfChildrenVisiting', 'MonthlyIncome']

categorical_features = ['TypeofContact', 'Occupation', 'Gender', 
                        'ProductPitched', 'MaritalStatus', 'Designation']

# 5. Preprocessing
num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))

preprocessor = make_column_transformer(
    (num_pipeline, numeric_features),
    (cat_pipeline, categorical_features)
)

# 6. Pipeline & Parameter Grid
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)
model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3],
    'xgbclassifier__colsample_bytree': [0.4, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.1],
}

# 7. Grid Search & MLflow Tracking
print("⚙️ Starting XGBoost Grid Search...")
with mlflow.start_run(run_name="XGBoost_Tuning_Grid"):
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all nested parameter combinations
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_cv_accuracy", results['mean_test_score'][i])

    # Log the overall best parameters
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Evaluate the best model
    classification_threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })
    
    # Save to MLflow
    mlflow.sklearn.log_model(best_model, "visitus_best_xgb_model")

# 8. Save standard Joblib artifact for FastAPI (Updated to .joblib extension)
model_path = "/content/drive/MyDrive/visitusmlops/backend/model_building/visitus_xgb_model.joblib"
joblib.dump(best_model, model_path)
print(f"🚀 Best XGBoost model successfully saved to: {model_path}")
