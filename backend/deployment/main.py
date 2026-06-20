import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="VisitUs AI API")

# Load the model
# Ensure this path matches where your hosting.py copies the file
model = joblib.load("visitus_xgb_model.joblib")

class CustomerData(BaseModel):
    Age: int
    CityTier: int
    DurationOfPitch: float
    # ... add all your features here ...

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
