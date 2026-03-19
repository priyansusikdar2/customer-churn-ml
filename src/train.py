import pandas as pd
import joblib
import os

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_data, feature_engineering

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

df = load_data(data_path)
df = feature_engineering(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]

cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

param_grid = {
    "model__n_estimators": [100],
    "model__max_depth": [5, 10]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)

print("Training model...")
grid.fit(X, y)

os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(grid.best_estimator_, model_path)

# 🔥 Feature importance
model = grid.best_estimator_.named_steps["model"]
feature_names = grid.best_estimator_.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

importance_df.to_csv(os.path.join(BASE_DIR, "models", "feature_importance.csv"), index=False)

print("✅ Model + feature importance saved")