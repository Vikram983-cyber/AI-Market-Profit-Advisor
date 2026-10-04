import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("data/sales_data.csv")

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# =========================================================
# 2. DATE PROCESSING
# =========================================================

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day


# =========================================================
# 3. CONVERT CATEGORICAL COLUMNS
# =========================================================

categorical_columns = [
    "brand",
    "product",
    "category",
    "market_condition",
    "competitor_pressure"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    dtype=int
)


# =========================================================
# 4. CREATE FEATURES AND TARGET
# =========================================================

drop_columns = [
    "date",
    "profit"
]

X = df.drop(columns=drop_columns)
y = df["profit"]

print("\nFeatures prepared successfully!")
print("Number of features:", len(X.columns))


# =========================================================
# 5. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 6. TRAIN RANDOM FOREST MODEL
# =========================================================

print("\nTraining model...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# =========================================================
# 7. PREDICTION
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 8. MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========== MODEL RESULTS ==========")

print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# =========================================================
# 9. SAMPLE PREDICTIONS
# =========================================================

results = pd.DataFrame({
    "Actual Profit": y_test.values,
    "Predicted Profit": y_pred
})

print("\nModel training completed successfully!")

print("\n========== SAMPLE PREDICTIONS ==========")

print(results.head(10))


# =========================================================
# 10. CREATE MODEL FOLDER
# =========================================================

os.makedirs("model", exist_ok=True)


# =========================================================
# 11. SAVE TRAINED MODEL
# =========================================================

model_path = "model/profit_model.pkl"

joblib.dump(
    model,
    model_path
)


# =========================================================
# 12. SAVE MODEL FEATURES
# =========================================================

features_path = "model/model_features.pkl"

joblib.dump(
    X.columns.tolist(),
    features_path
)


# =========================================================
# 13. SUCCESS MESSAGE
# =========================================================

print("\n========================================")
print("MODEL SAVED SUCCESSFULLY!")
print("========================================")

print("Model  :", model_path)
print("Features:", features_path)

print("\nAI Market Profit Advisor model is ready!")