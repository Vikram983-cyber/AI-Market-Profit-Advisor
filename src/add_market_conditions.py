import pandas as pd
import numpy as np

np.random.seed(42)

# Load existing dataset
file_path = "data/sales_data.csv"
df = pd.read_csv(file_path)

# Convert numeric columns to float
numeric_columns = [
    "product_cost",
    "marketing_cost",
    "other_cost",
    "selling_price"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Add market condition columns
df["market_condition"] = "Normal"
df["competitor_pressure"] = "Low"
df["competitor_demand_index"] = 100.0
df["competitor_price_change_percent"] = 0.0

# Difficult market periods
bad_market_months = [
    "2024-06",
    "2024-07",
    "2025-06",
    "2025-07",
    "2025-08"
]

# Apply market conditions
for i in df.index:

    month = df.loc[i, "date"].strftime("%Y-%m")

    if month in bad_market_months:

        df.loc[i, "market_condition"] = "Difficult"
        df.loc[i, "competitor_pressure"] = "High"

        df.loc[i, "competitor_demand_index"] = np.random.uniform(110, 140)

        df.loc[i, "competitor_price_change_percent"] = np.random.uniform(
            -8, 5
        )

        # Increase costs safely as float
        df.loc[i, "product_cost"] = (
            float(df.loc[i, "product_cost"])
            * np.random.uniform(1.08, 1.15)
        )

        df.loc[i, "marketing_cost"] = (
            float(df.loc[i, "marketing_cost"])
            * np.random.uniform(1.05, 1.20)
        )

        df.loc[i, "other_cost"] = (
            float(df.loc[i, "other_cost"])
            * np.random.uniform(1.05, 1.15)
        )

        # Reduce selling price slightly
        df.loc[i, "selling_price"] = (
            float(df.loc[i, "selling_price"])
            * np.random.uniform(0.92, 0.98)
        )

    else:

        df.loc[i, "market_condition"] = "Normal"

        df.loc[i, "competitor_pressure"] = np.random.choice(
            ["Low", "Medium"]
        )

        df.loc[i, "competitor_demand_index"] = np.random.uniform(
            90, 110
        )

        df.loc[i, "competitor_price_change_percent"] = np.random.uniform(
            -3, 3
        )

# Recalculate revenue
df["revenue"] = df["units_sold"] * df["selling_price"]

# Recalculate total cost
df["total_cost"] = (
    df["units_sold"] * df["product_cost"]
    + df["marketing_cost"]
    + df["other_cost"]
)

# Recalculate profit
df["profit"] = df["revenue"] - df["total_cost"]

# Save dataset
df.to_csv(file_path, index=False)

print("Market conditions added successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {file_path}")

print("\nUpdated columns:")
print(df.columns.tolist())

print("\nMarket condition counts:")
print(df["market_condition"].value_counts())

print("\nProfit/Loss Summary:")
print(df["profit"].describe())