import shap
import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = joblib.load(model_path)

explainer = shap.Explainer(model.named_steps["model"])

def get_shap_values(data: dict):
    df = pd.DataFrame([data])

    df["TotalSpend"] = df["MonthlyCharges"] * df["tenure"]
    df["AvgMonthlyValue"] = df["TotalSpend"] / (df["tenure"] + 1)
    df["HighValue"] = (df["TotalSpend"] > df["TotalSpend"].median()).astype(int)

    processed = model.named_steps["preprocessor"].transform(df)

    shap_values = explainer(processed)

    return shap_values.values.tolist()