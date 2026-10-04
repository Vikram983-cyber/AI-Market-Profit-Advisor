import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range(
    start="2024-01-01",
    end="2025-12-01",
    freq="MS"
)

products = [
    ("Samsung", "Galaxy A55", 28000, 22000),
    ("Samsung", "Galaxy S24", 70000, 58000),
    ("Samsung", "Galaxy S24 Ultra", 125000, 105000),
    ("Apple", "iPhone 15", 70000, 57000),
    ("Apple", "iPhone 15 Pro", 120000, 99000),
    ("Apple", "iPhone 15 Pro Max", 145000, 120000),
]

rows = []

for date in dates:

    month = date.month

    # Seasonal demand
    seasonal_factor = 1.0

    if month in [10, 11]:
        seasonal_factor = 1.30
    elif month in [12, 1]:
        seasonal_factor = 1.10
    elif month in [6, 7]:
        seasonal_factor = 0.90

    # Market growth
    market_growth = np.random.normal(0.02, 0.05)

    # Market condition
    if market_growth < 0:
        market_condition = "Difficult"
    else:
        market_condition = "Normal"

    # Competitor pressure
    competitor_pressure = np.random.choice(
        ["Low", "Medium", "High"],
        p=[0.30, 0.45, 0.25]
    )

    for brand, product, price, base_cost in products:

        # Product demand
        base_units = np.random.randint(80, 180)

        units_sold = int(
            base_units
            * seasonal_factor
            * (1 + market_growth)
        )

        units_sold = max(units_sold, 20)

        # Competitor pressure effect
        if competitor_pressure == "High":
            units_sold = int(units_sold * 0.85)

        elif competitor_pressure == "Medium":
            units_sold = int(units_sold * 0.95)

        # Discount
        discount = np.random.uniform(3, 15)

        selling_price = price * (1 - discount / 100)

        revenue = units_sold * selling_price

        # Marketing cost
        marketing_cost = np.random.randint(30000, 100000)

        # Other operating cost
        other_cost = np.random.randint(10000, 30000)

        total_cost = (
            units_sold * base_cost
            + marketing_cost
            + other_cost
        )

        profit = revenue - total_cost

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "brand": brand,
            "product": product,
            "category": "Smartphone",

            "units_sold": units_sold,
            "selling_price": round(selling_price, 2),

            "revenue": round(revenue, 2),
            "product_cost": round(units_sold * base_cost, 2),

            "discount_percent": round(discount, 2),
            "marketing_cost": marketing_cost,
            "other_cost": other_cost,

            "total_cost": round(total_cost, 2),
            "profit": round(profit, 2),

            "market_growth_percent": round(
                market_growth * 100, 2
            ),

            "market_condition": market_condition,
            "competitor_pressure": competitor_pressure
        })


df = pd.DataFrame(rows)

output_path = "data/sales_data.csv"

df.to_csv(output_path, index=False)

print("Dataset created successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {output_path}")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nProfit/Loss Summary:")
print(df["profit"].describe())