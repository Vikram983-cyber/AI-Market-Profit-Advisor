import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/profit_model.pkl")
model_features = joblib.load("model/model_features.pkl")

# Load original dataset
df = pd.read_csv("data/sales_data.csv")

# Take one existing record as an example
sample = df.iloc[0].copy()

# Convert date
date = pd.to_datetime(sample["date"])

# Create input dataframe
input_data = pd.DataFrame([sample])

input_data["year"] = date.year
input_data["month"] = date.month
input_data["day"] = date.day

# Remove target and date
input_data = input_data.drop(columns=["date", "profit"], errors="ignore")

# One-hot encoding
input_data = pd.get_dummies(
    input_data,
    columns=[
        "brand",
        "product",
        "category",
        "market_condition",
        "competitor_pressure"
    ],
    dtype=int
)

# Match training features
input_data = input_data.reindex(
    columns=model_features,
    fill_value=0
)

# Prediction
prediction = model.predict(input_data)[0]

print("\n========== AI PROFIT PREDICTION ==========")

print("Product:", sample["product"])
print("Brand:", sample["brand"])
print("Category:", sample["category"])
print("Units Sold:", sample["units_sold"])
print("Selling Price:", sample["selling_price"])
print("Market Condition:", sample["market_condition"])

print("\nPredicted Profit/Loss:", prediction)

if prediction > 0:
    print("Recommendation: PROFITABLE ✅")
else:
    print("Recommendation: LOSS ⚠️")

print("\n==========================================")