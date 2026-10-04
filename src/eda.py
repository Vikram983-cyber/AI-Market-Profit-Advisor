import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/sales_data.csv")

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Basic information
print("\nDataset Information:")
print(df.info())

# Profit summary
print("\nProfit Summary:")
print(df["profit"].describe())

# Total profit and loss
total_profit = df["profit"].sum()
average_profit = df["profit"].mean()

print("\nTotal Profit/Loss:", total_profit)
print("Average Profit/Loss:", average_profit)

# Profit by brand
print("\nProfit by Brand:")
print(df.groupby("brand")["profit"].agg(["sum", "mean", "count"]))

# Profit by product
print("\nProfit by Product:")
print(df.groupby("product")["profit"].agg(["sum", "mean", "count"]))

# Profit by market condition
print("\nProfit by Market Condition:")
print(df.groupby("market_condition")["profit"].agg(["sum", "mean", "count"]))

# Profit/Loss graph
plt.figure(figsize=(10, 5))
plt.hist(df["profit"], bins=20)
plt.title("Profit/Loss Distribution")
plt.xlabel("Profit/Loss")
plt.ylabel("Number of Sales")
plt.axvline(0, linestyle="--")
plt.tight_layout()
plt.show()

# Brand profit graph
brand_profit = df.groupby("brand")["profit"].sum()

plt.figure(figsize=(8, 5))
brand_profit.plot(kind="bar")
plt.title("Total Profit by Brand")
plt.xlabel("Brand")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Market condition profit graph
market_profit = df.groupby("market_condition")["profit"].sum()

plt.figure(figsize=(8, 5))
market_profit.plot(kind="bar")
plt.title("Profit by Market Condition")
plt.xlabel("Market Condition")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()