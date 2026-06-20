import streamlit as st
import requests
import os

# Define the FastAPI backend URL (defaults to localhost for testing)
# In a Docker Compose network, this would be "http://backend:8000/predict"
API_URL = os.getenv("API_URL", "https://zaheergshaikh-visitusapi.hf.space/predict")

st.set_page_config(page_title="VisitUs Package Predictor", layout="wide")

st.title("✈️ VisitUs Travel Package Prediction")
st.write("Enter the customer's interaction details below to predict if they will purchase the travel package.")

# UI Layout using columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Demographics")
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    MonthlyIncome = st.number_input("Monthly Income", min_value=10000.0, value=25000.0)
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

with col2:
    st.subheader("Professional Details")
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])

with col3:
    st.subheader("Pitch & Interaction")
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, value=15.0)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=1.0, value=3.0)

st.divider()

col4, col5, col6, col7 = st.columns(4)
with col4:
    NumberOfPersonVisiting = st.number_input("Total Persons Visiting", min_value=1, value=2)
with col5:
    NumberOfChildrenVisiting = st.number_input("Children Visiting", min_value=0.0, value=0.0)
with col6:
    NumberOfTrips = st.number_input("Previous Trips", min_value=0.0, value=2.0)
with col7:
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    Passport = st.selectbox("Has Passport?", ["Yes", "No"])

# Predict Button
if st.button("🔮 Predict Conversion", use_container_width=True):
    # Map 'Yes'/'No' to 1/0 for the API
    payload = {
        "Age": Age,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "PreferredPropertyStar": PreferredPropertyStar,
        "NumberOfTrips": NumberOfTrips,
        "Passport": 1 if Passport == "Yes" else 0,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": 1 if OwnCar == "Yes" else 0,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "MonthlyIncome": MonthlyIncome,
        "TypeofContact": TypeofContact,
        "Occupation": Occupation,
        "Gender": Gender,
        "ProductPitched": ProductPitched,
        "MaritalStatus": MaritalStatus,
        "Designation": Designation
    }
    
    try:
        with st.spinner("Analyzing customer profile...") :
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            label = result["prediction_label"]
            prob = result["purchase_probability"] * 100
            
            if result["prediction_code"] == 1:
                st.success(f"### 🎉 High Intent: {label}")
                st.write(f"Probability of purchasing: **{prob:.1f}%**")
            else:
                st.error(f"### ❌ Low Intent: {label}")
                st.write(f"Probability of purchasing: **{prob:.1f}%**")
        else:
            st.error(f"Backend API Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("🚨 Could not connect to the Backend API. Make sure your FastAPI server is running!")
