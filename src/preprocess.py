import pandas as pd

def load_data(path):
    df = pd.read_csv(r"C:\Users\Priyansu Sikdar\Downloads\customer-churn-ml\data\archive\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def feature_engineering(df):
    df["TotalSpend"] = df["MonthlyCharges"] * df["tenure"]
    df["AvgMonthlyValue"] = df["TotalSpend"] / (df["tenure"] + 1)
    df["HighValue"] = (df["TotalSpend"] > df["TotalSpend"].median()).astype(int)

    return df