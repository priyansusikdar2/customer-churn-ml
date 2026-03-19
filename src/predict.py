import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = joblib.load(model_path)

def predict(data: dict):
    df = pd.DataFrame([data])

    df["TotalSpend"] = df["MonthlyCharges"] * df["tenure"]
    df["AvgMonthlyValue"] = df["TotalSpend"] / (df["tenure"] + 1)
    df["HighValue"] = (df["TotalSpend"] > df["TotalSpend"].median()).astype(int)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": prediction,
        "churn_flag": 1 if prediction == "Yes" else 0,
        "probability": float(probability)
    }