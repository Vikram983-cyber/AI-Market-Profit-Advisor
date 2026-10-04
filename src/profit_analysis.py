import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/sales_data.csv")

# Make sure numeric columns are numeric
numeric_cols = [
    "units_sold",
    "selling_price",
    "revenue",
    "product_cost",
    "marketing_cost",
    "other_cost",
    "total_cost",
    "profit"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove missing profit values
df = df.dropna(subset=["profit"])

print("\n========== PROFIT ANALYSIS ==========\n")

# Overall summary
print("Total Revenue:", df["revenue"].sum())
print("Total Cost:", df["total_cost"].sum())
print("Total Profit/Loss:", df["profit"].sum())
print("Average Profit/Loss:", df["profit"].mean())

# Profit/Loss status
total_profit = df["profit"].sum()

if total_profit > 0:
    print("\nOverall Status: PROFIT")
else:
    print("\nOverall Status: LOSS")


# ------------------------------------
# 1. Profit by Brand
# ------------------------------------

brand_profit = df.groupby("brand")["profit"].sum().sort_values()

print("\n--- Profit by Brand ---")
print(brand_profit)

plt.figure(figsize=(8, 5))
brand_profit.plot(kind="bar")
plt.title("Total Profit by Brand")
plt.xlabel("Brand")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ------------------------------------
# 2. Profit by Market Condition
# ------------------------------------

market_profit = df.groupby("market_condition")["profit"].sum().sort_values()

print("\n--- Profit by Market Condition ---")
print(market_profit)

plt.figure(figsize=(8, 5))
market_profit.plot(kind="bar")
plt.title("Profit by Market Condition")
plt.xlabel("Market Condition")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ------------------------------------
# 3. Profit Distribution
# ------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(df["profit"], bins=20)
plt.title("Profit/Loss Distribution")
plt.xlabel("Profit/Loss")
plt.ylabel("Number of Sales")
plt.tight_layout()
plt.show()


# ------------------------------------
# 4. Top Products by Profit
# ------------------------------------

product_profit = (
    df.groupby("product")["profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\n--- Top 10 Products by Profit ---")
print(product_profit.head(10))


# ------------------------------------
# 5. Most Loss-Making Products
# ------------------------------------

print("\n--- Top 10 Loss-Making Products ---")
print(product_profit.sort_values().head(10))


# ------------------------------------
# 6. Profit by Category
# ------------------------------------

category_profit = (
    df.groupby("category")["profit"]
    .sum()
    .sort_values()
)

print("\n--- Profit by Category ---")
print(category_profit)


# ------------------------------------
# 7. Difficult vs Normal Market
# ------------------------------------

if "market_condition" in df.columns:

    difficult = df[df["market_condition"] == "Difficult"]["profit"].sum()
    normal = df[df["market_condition"] == "Normal"]["profit"].sum()

    print("\n--- Market Comparison ---")
    print("Difficult Market Profit/Loss:", difficult)
    print("Normal Market Profit/Loss:", normal)


# ------------------------------------
# Final Recommendation
# ------------------------------------

print("\n========== RECOMMENDATION ==========\n")

if total_profit > 0:
    print("Business is currently profitable.")
else:
    print("Business is currently operating at a loss.")

best_brand = brand_profit.idxmax()
worst_brand = brand_profit.idxmin()

print("Best performing brand:", best_brand)
print("Worst performing brand:", worst_brand)

print("\nAnalysis completed successfully!")