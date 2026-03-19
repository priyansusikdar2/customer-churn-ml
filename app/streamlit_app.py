import streamlit as st
import requests
import pandas as pd

st.title("📊 Customer Churn Dashboard")

API_URL = "http://127.0.0.1:8000/predict"

# Inputs
tenure = st.slider("Tenure", 0, 72, 12)
MonthlyCharges = st.slider("Monthly Charges", 0, 200, 70)
TotalCharges = st.slider("Total Charges", 0, 10000, 2000)

threshold = st.slider("Churn Threshold", 0.0, 1.0, 0.5)

payload = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": tenure,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

if st.button("Predict"):
    res = requests.post(API_URL, json=payload).json()

    st.write(res)

    if res["probability"] > threshold:
        st.error(f"⚠️ Churn Risk ({res['probability']:.2f})")
    else:
        st.success(f"✅ Safe ({res['probability']:.2f})")

# 📊 Feature Importance
st.subheader("📊 Feature Importance")

try:
    df = pd.read_csv("models/feature_importance.csv")
    st.bar_chart(df.head(10).set_index("feature"))
except:
    st.warning("Run training first")