import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="VisitUs AI API")

# Load the model
#hosting.py places the model in the same directory as main.py
model = joblib.load("visitus_xgb_model.joblib")

class CustomerData(BaseModel):
    Age: int
    CityTier: int
    DurationOfPitch: float
    NumberOfPersonVisiting: int
    NumberOfFollowups: float
    PreferredPropertyStar: float
    NumberOfTrips: float
    Passport: int
    PitchSatisfactionScore: int
    OwnCar: int
    NumberOfChildrenVisiting: float
    MonthlyIncome: float
    TypeofContact: str
    Occupation: str
    Gender: str
    ProductPitched: str
    MaritalStatus: str
    Designation: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: CustomerData):
    # Convert incoming request to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Perform prediction
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]
    
    return {
        "prediction_code": int(prediction[0]),
        "prediction_label": "Purchased" if prediction[0] == 1 else "Not Purchased",
        "purchase_probability": float(probability)
    }
