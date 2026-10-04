import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/sales_data.csv")

# Date features
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# Convert categorical columns
df = pd.get_dummies(
    df,
    columns=[
        "brand",
        "product",
        "category",
        "market_condition",
        "competitor_pressure"
    ],
    dtype=int
)

# Features and target
X = df.drop(columns=["date", "profit"])
y = df["profit"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/profit_model.pkl")

# Save feature names
joblib.dump(list(X.columns), "model/model_features.pkl")

print("Model saved successfully!")
print("Saved: model/profit_model.pkl")
print("Saved: model/model_features.pkl")