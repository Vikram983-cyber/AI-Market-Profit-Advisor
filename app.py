import os
import hashlib
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Market Profit Advisor",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = os.path.join("data", "sales_data.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "profit_model.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "model_features.pkl")

os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

REQUIRED_COLUMNS = [
    "date",
    "brand",
    "product",
    "category",
    "units_sold",
    "selling_price",
    "revenue",
    "product_cost",
    "discount_percent",
    "marketing_cost",
    "other_cost",
    "total_cost",
    "profit",
    "market_growth_percent",
    "market_condition",
    "competitor_pressure",
]

CATEGORICAL_COLUMNS = [
    "brand",
    "product",
    "category",
    "market_condition",
    "competitor_pressure",
]

NUMERIC_COLUMNS = [
    "units_sold",
    "selling_price",
    "revenue",
    "product_cost",
    "discount_percent",
    "marketing_cost",
    "other_cost",
    "total_cost",
    "profit",
    "market_growth_percent",
]

MONTHS = list(range(1, 13))
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

POSITIVE_WORDS = {
    "growth", "growing", "strong", "surge", "surging", "demand",
    "popular", "sales", "launch", "launched", "success", "record",
    "upgrade", "premium", "trend", "trending", "increase", "increased",
    "rising", "rise", "best", "leader", "leadership", "innovation",
    "expansion", "opportunity", "positive", "boost",
}

NEGATIVE_WORDS = {
    "decline", "declining", "drop", "dropped", "fall", "falling", "weak",
    "slow", "slowing", "loss", "losses", "recall", "issue", "issues",
    "problem", "problems", "lawsuit", "ban", "banned", "delay", "delays",
    "negative", "pressure", "competition", "competitive", "down", "risk",
}

# ============================================================
# SMALL HELPERS
# ============================================================
def money(value):
    try:
        return f"₹ {float(value):,.0f}"
    except Exception:
        return "₹ 0"


def pct(value):
    try:
        return f"{float(value):+.2f}%"
    except Exception:
        return "+0.00%"


def safe_growth(current, previous):
    try:
        current = float(current)
        previous = float(previous)
    except Exception:
        return 0.0
    if not np.isfinite(previous) or previous == 0:
        return 0.0
    if not np.isfinite(current):
        return 0.0
    return (current - previous) / abs(previous) * 100.0


def dataframe_fingerprint(df):
    """Stable fingerprint used to avoid unnecessary model retraining."""
    try:
        values = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha256(values).hexdigest()
    except Exception:
        return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


# ============================================================
# DOMAIN DETECTION
# ============================================================
def infer_domain_label(df):
    values = []
    for col in ("category", "product"):
        if col in df.columns:
            values.extend(df[col].dropna().astype(str).str.lower().tolist())
    text = " ".join(values)

    rules = [
        ("food", [
            "food", "snack", "beverage", "drink", "juice", "dairy", "rice",
            "noodle", "namkeen", "paneer", "grocery", "bakery",
        ]),
        ("clothing", [
            "clothing", "apparel", "wear", "shirt", "t-shirt", "jeans",
            "hoodie", "kurti", "dress", "fashion", "footwear", "shoe",
        ]),
        ("electronics", [
            "electronic", "laptop", "mobile", "smartphone", "tablet",
            "television", " tv", "camera", "headphone", "earphone",
            "computer", "monitor", "printer",
        ]),
        ("automotive", [
            "automotive", "car", "bike", "motorcycle", "vehicle", "tyre", "tire",
        ]),
        ("cosmetics", [
            "cosmetic", "makeup", "skincare", "skin care", "shampoo",
            "conditioner", "beauty", "lipstick", "cream",
        ]),
        ("furniture", [
            "furniture", "chair", "table", "sofa", "bed", "desk", "cabinet",
        ]),
        ("stationery", [
            "stationery", "notebook", "pen", "pencil", "book", "paper",
        ]),
        ("sports", [
            "sports", "cricket", "football", "basketball", "bat", "ball",
            "jersey", "fitness", "gym",
        ]),
        ("healthcare", [
            "health", "medical", "medicine", "supplement", "pharma", "wellness",
        ]),
        ("appliances", [
            "appliance", "refrigerator", "fridge", "washing machine", "microwave",
            "oven", "air conditioner", "fan", "cooler",
        ]),
    ]
    for label, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return label
    return "product"


def domain_title(domain):
    labels = {
        "food": ("🍲", "Food Product"),
        "clothing": ("👕", "Clothing Product"),
        "electronics": ("💻", "Electronics Product"),
        "automotive": ("🚗", "Automotive Product"),
        "cosmetics": ("💄", "Cosmetics Product"),
        "furniture": ("🪑", "Furniture Product"),
        "stationery": ("📚", "Stationery Product"),
        "sports": ("🏏", "Sports Product"),
        "healthcare": ("🏥", "Healthcare Product"),
        "appliances": ("🏠", "Appliance Product"),
        "product": ("📦", "Product"),
    }
    return labels.get(domain, labels["product"])


# ============================================================
# DATA VALIDATION / PREPARATION
# ============================================================
def validate_and_clean(df, strict_schema=False):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    uploaded_columns = list(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in uploaded_columns]
    extra = [c for c in uploaded_columns if c not in REQUIRED_COLUMNS]

    if strict_schema and (missing or extra or uploaded_columns != REQUIRED_COLUMNS):
        messages = []
        if missing:
            messages.append("Missing columns: " + ", ".join(missing))
        if extra:
            messages.append("Extra columns: " + ", ".join(extra))
        if not missing and not extra and uploaded_columns != REQUIRED_COLUMNS:
            messages.append("Column order is incorrect. Use the exact 16-column order shown in the app.")
        raise ValueError(" | ".join(messages))

    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_date = int(df["date"].isna().sum())
    invalid_numeric = {
        col: int(df[col].isna().sum())
        for col in NUMERIC_COLUMNS
        if df[col].isna().any()
    }

    if strict_schema and (invalid_date > 0 or invalid_numeric):
        details = []
        if invalid_date:
            details.append(f"Invalid/blank dates: {invalid_date}")
        if invalid_numeric:
            details.append(
                "Invalid/blank numeric values: "
                + ", ".join(f"{k} ({v})" for k, v in invalid_numeric.items())
            )
        raise ValueError(" | ".join(details))

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col] == "", col] = "Unknown"

    # Base/trusted dataset: drop records missing the fields needed for analysis,
    # then fill other numeric gaps so model training remains stable.
    df = df.dropna(subset=["date", "units_sold", "selling_price", "profit"])
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].fillna(0.0)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["date", "units_sold", "selling_price", "profit"])
    df = df.reset_index(drop=True)
    return df


def prepare_data(df):
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])

    for col in CATEGORICAL_COLUMNS:
        if col not in work.columns:
            work[col] = "Unknown"
        work[col] = work[col].fillna("Unknown").astype(str)

    for col in NUMERIC_COLUMNS:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    work["year"] = work["date"].dt.year.astype(int)
    work["month"] = work["date"].dt.month.astype(int)
    work["day"] = work["date"].dt.day.astype(int)

    encoded = pd.get_dummies(
        work,
        columns=CATEGORICAL_COLUMNS,
        dtype=np.int8,
    )
    return encoded


# ============================================================
# MODEL
# ============================================================
def train_model(df):
    prepared = prepare_data(df)
    if "profit" not in prepared.columns:
        raise ValueError("Profit column is required for model training.")

    X = prepared.drop(columns=["date", "profit"], errors="ignore")
    y = pd.to_numeric(prepared["profit"], errors="coerce").fillna(0.0)

    if len(X) < 10:
        raise ValueError("At least 10 valid rows are required for AI model training.")

    test_size = max(2, int(round(len(X) * 0.20)))
    test_size = min(test_size, len(X) - 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    # Kept deliberately lightweight for Render free instances.
    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else 0.0

    # Saving is best-effort. The app still works if the runtime filesystem is read-only.
    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(X.columns.tolist(), FEATURE_PATH)
    except Exception:
        pass

    return model, X.columns.tolist(), mae, rmse, r2


def load_or_train_model(df):
    current_features = prepare_data(df).drop(
        columns=["date", "profit"],
        errors="ignore",
    ).columns.tolist()

    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
        try:
            saved_model = joblib.load(MODEL_PATH)
            saved_features = list(joblib.load(FEATURE_PATH))
            if saved_features == current_features:
                return saved_model, saved_features, None, None, None
        except Exception:
            pass

    return train_model(df)


def get_model_for_dataframe(df):
    """Reuse the current model during Streamlit reruns; retrain only when data changes."""
    fp = dataframe_fingerprint(df)
    if st.session_state.get("model_fingerprint") != fp:
        with st.spinner("🤖 Preparing AI profit model..."):
            model, features, mae, rmse, r2 = load_or_train_model(df)
        st.session_state["model_fingerprint"] = fp
        st.session_state["model"] = model
        st.session_state["features"] = features
        st.session_state["model_metrics"] = (mae, rmse, r2)

    return (
        st.session_state["model"],
        st.session_state["features"],
        *st.session_state.get("model_metrics", (None, None, None)),
    )


# ============================================================
# MONTHLY / BRAND / PRODUCT ANALYSIS
# ============================================================
def monthly_actual(df, year):
    data = df[df["date"].dt.year == year].copy()
    if data.empty:
        return pd.DataFrame(columns=["MonthNo", "Month", "Units", "Revenue", "Profit"])

    grouped = (
        data.groupby(data["date"].dt.month)
        .agg(
            Units=("units_sold", "sum"),
            Revenue=("revenue", "sum"),
            Profit=("profit", "sum"),
        )
        .reindex(MONTHS, fill_value=0)
        .reset_index()
    )
    grouped = grouped.rename(columns={grouped.columns[0]: "MonthNo"})
    grouped["Month"] = MONTH_NAMES
    grouped["MonthNo"] = MONTHS
    return grouped[["MonthNo", "Month", "Units", "Revenue", "Profit"]]


def monthly_chart(data, y, title, y_label):
    if data.empty:
        st.info("No monthly data available.")
        return

    chart_data = data.copy()
    chart_data["Month"] = pd.Categorical(
        chart_data["Month"].astype(str),
        categories=MONTH_NAMES,
        ordered=True,
    )

    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Month:N",
                sort=MONTH_NAMES,
                axis=alt.Axis(title="Month"),
            ),
            y=alt.Y(f"{y}:Q", title=y_label),
            tooltip=[
                alt.Tooltip("Month:N", title="Month"),
                alt.Tooltip(f"{y}:Q", title=y_label, format=",.0f"),
            ],
        )
        .properties(title=title, height=400)
    )
    st.altair_chart(chart, use_container_width=True)


def brand_actual(df, year):
    data = df[df["date"].dt.year == year].copy()
    if data.empty:
        return pd.DataFrame()

    out = (
        data.groupby("brand")
        .agg(
            Units=("units_sold", "sum"),
            Revenue=("revenue", "sum"),
            Profit=("profit", "sum"),
        )
        .reset_index()
        .sort_values(["Revenue", "Profit"], ascending=False)
        .reset_index(drop=True)
    )
    out["Profit_Margin"] = np.where(
        out["Revenue"] != 0,
        out["Profit"] / out["Revenue"] * 100,
        0,
    )
    return out


def product_actual(df, year):
    data = df[df["date"].dt.year == year].copy()
    if data.empty:
        return pd.DataFrame()
    return (
        data.groupby(["brand", "product", "category"])
        .agg(
            Units=("units_sold", "sum"),
            Revenue=("revenue", "sum"),
            Profit=("profit", "sum"),
        )
        .reset_index()
    )


def product_sales_actual(df, year):
    data = df[df["date"].dt.year == year].copy()
    if data.empty:
        return pd.DataFrame()

    out = (
        data.groupby(["product", "category"])
        .agg(
            Brands=("brand", "nunique"),
            Units_Sold=("units_sold", "sum"),
            Revenue=("revenue", "sum"),
            Profit=("profit", "sum"),
            Avg_Selling_Price=("selling_price", "mean"),
        )
        .reset_index()
        .sort_values(["Units_Sold", "Revenue"], ascending=False)
        .reset_index(drop=True)
    )
    out["Profit_Margin"] = np.where(
        out["Revenue"] != 0,
        out["Profit"] / out["Revenue"] * 100,
        0,
    )
    return out


def monthly_product_sales(df, year, product=None):
    data = df[df["date"].dt.year == year].copy()
    if product is not None:
        data = data[data["product"] == product]
    if data.empty:
        return pd.DataFrame()

    out = (
        data.groupby(data["date"].dt.month)
        .agg(
            Units_Sold=("units_sold", "sum"),
            Revenue=("revenue", "sum"),
            Profit=("profit", "sum"),
        )
        .reindex(MONTHS, fill_value=0)
        .reset_index()
    )
    out = out.rename(columns={out.columns[0]: "MonthNo"})
    out["MonthNo"] = MONTHS
    out["Month"] = MONTH_NAMES
    return out[["MonthNo", "Month", "Units_Sold", "Revenue", "Profit"]]


# ============================================================
# 2027 PRODUCT FORECAST / RECOMMENDATION
# ============================================================
def build_2027_product_forecast(df, online_scores=None):
    p25 = product_actual(df, 2025)
    p26 = product_actual(df, 2026)

    if p26.empty:
        return pd.DataFrame()

    keys = ["brand", "product", "category"]
    left = p26.rename(
        columns={
            "Units": "2026 Units",
            "Revenue": "2026 Revenue",
            "Profit": "2026 Profit",
        }
    )

    if p25.empty:
        merged = left.copy()
        merged["2025 Units"] = np.nan
        merged["2025 Revenue"] = np.nan
        merged["2025 Profit"] = np.nan
    else:
        right = p25.rename(
            columns={
                "Units": "2025 Units",
                "Revenue": "2025 Revenue",
                "Profit": "2025 Profit",
            }
        )
        merged = left.merge(right, on=keys, how="left")

    for metric in ["Units", "Revenue", "Profit"]:
        prev = merged[f"2025 {metric}"].fillna(0)
        curr = merged[f"2026 {metric}"].fillna(0)
        merged[f"{metric} Growth 25-26 (%)"] = np.where(
            prev != 0,
            (curr - prev) / prev.abs() * 100,
            0,
        )

    revenue_growth = (
        merged["Revenue Growth 25-26 (%)"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(-25, 35)
    )
    profit_growth = (
        merged["Profit Growth 25-26 (%)"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(-35, 40)
    )
    units_growth = (
        merged["Units Growth 25-26 (%)"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(-25, 35)
    )

    merged["Forecast Revenue Growth 2027 (%)"] = revenue_growth
    merged["Forecast Profit Growth 2027 (%)"] = profit_growth
    merged["Forecast Units Growth 2027 (%)"] = units_growth

    merged["Predicted Units 2027"] = (
        merged["2026 Units"] * (1 + units_growth / 100)
    ).clip(lower=0)
    merged["Predicted Revenue 2027"] = (
        merged["2026 Revenue"] * (1 + revenue_growth / 100)
    ).clip(lower=0)
    merged["Predicted Profit 2027"] = (
        merged["2026 Profit"] * (1 + profit_growth / 100)
    )

    if online_scores is None:
        merged["Online Market Score"] = 50.0
    else:
        merged["Online Market Score"] = (
            merged.apply(
                lambda r: online_scores.get(
                    (str(r["brand"]), str(r["product"])),
                    online_scores.get(str(r["product"]), 50.0),
                ),
                axis=1,
            )
            .fillna(50.0)
            .astype(float)
        )

    web_adjustment = ((merged["Online Market Score"] - 50.0) * 0.10).clip(-5, 5)
    merged["Web Growth Adjustment 2027 (%)"] = web_adjustment

    merged["Predicted Revenue 2027"] = (
        merged["2026 Revenue"]
        * (1 + (merged["Forecast Revenue Growth 2027 (%)"] + web_adjustment) / 100)
    ).clip(lower=0)
    merged["Predicted Profit 2027"] = (
        merged["2026 Profit"]
        * (1 + (merged["Forecast Profit Growth 2027 (%)"] + web_adjustment) / 100)
    )
    merged["Predicted Units 2027"] = (
        merged["2026 Units"]
        * (1 + (merged["Forecast Units Growth 2027 (%)"] + web_adjustment) / 100)
    ).clip(lower=0)

    merged["Predicted Margin 2027 (%)"] = np.where(
        merged["Predicted Revenue 2027"] != 0,
        merged["Predicted Profit 2027"] / merged["Predicted Revenue 2027"] * 100,
        0,
    )
    merged["Revenue Growth 2026→2027 (%)"] = np.where(
        merged["2026 Revenue"] != 0,
        (merged["Predicted Revenue 2027"] - merged["2026 Revenue"])
        / merged["2026 Revenue"].abs()
        * 100,
        0,
    )

    # Financial performance is the primary recommendation driver.
    profit_rank = merged["Predicted Profit 2027"].rank(pct=True)
    margin_rank = merged["Predicted Margin 2027 (%)"].rank(pct=True)
    growth_rank = merged["Revenue Growth 2026→2027 (%)"].rank(pct=True)
    merged["Financial Score"] = (
        0.65 * profit_rank + 0.20 * margin_rank + 0.15 * growth_rank
    ) * 100
    merged["Recommendation Score"] = (
        merged["Financial Score"] * 0.85
        + merged["Online Market Score"] * 0.15
    )

    merged["Recommendation"] = np.select(
        [
            merged["Predicted Profit 2027"] <= 0,
            merged["Recommendation Score"] >= 70,
            merged["Recommendation Score"] >= 45,
        ],
        [
            "🔴 Avoid / Review",
            "🟢 HIGH — Focus More",
            "🟡 MEDIUM — Monitor",
        ],
        default="🟠 LOW — Review",
    )

    return merged.sort_values(
        ["Recommendation Score", "Predicted Profit 2027"],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# ONLINE MARKET INTELLIGENCE
# ============================================================
def _sentiment_score(text):
    words = set(re.findall(r"[a-zA-Z]+", str(text).lower()))
    return len(words & POSITIVE_WORDS), len(words & NEGATIVE_WORDS)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_news(query, max_items=5):
    """Fetch public Google News RSS headlines without blocking app startup."""
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(str(query))
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )
    try:
        response = requests.get(
            url,
            timeout=4,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception:
        return []

    items = []
    for item in root.findall(".//item")[:max_items]:
        items.append(
            {
                "title": item.findtext("title", default="").strip(),
                "date": item.findtext("pubDate", default="").strip(),
                "link": item.findtext("link", default="").strip(),
                "source": item.findtext("source", default="").strip(),
            }
        )
    return items


def build_online_market_signals(product_rows, max_products=3):
    """Create small, best-effort web/news signals for top 2026 products."""
    if product_rows.empty:
        return {}, pd.DataFrame(), []

    top_products = (
        product_rows.sort_values("2026 Revenue", ascending=False)
        .drop_duplicates(subset=["brand", "product"])
        .head(max_products)
    )

    scores = {}
    summary = []
    headlines = []

    for _, row in top_products.iterrows():
        brand = str(row["brand"])
        product = str(row["product"])
        query = f'"{product}" {brand} market sales demand'
        news = fetch_market_news(query, max_items=5)

        pos = 0
        neg = 0
        for article in news:
            p, n = _sentiment_score(article["title"])
            pos += p
            neg += n
            headlines.append(
                {
                    "Product": product,
                    "Brand": brand,
                    "Headline": article["title"],
                    "Source": article["source"] or "Google News",
                    "Date": article["date"],
                    "Link": article["link"],
                }
            )

        raw = 50 + (pos - neg) * 6 + min(len(news), 5) * 1.5
        score = float(np.clip(raw, 0, 100))
        scores[(brand, product)] = score
        scores[product] = max(float(scores.get(product, 0)), score)

        if score >= 65:
            signal = "🟢 Positive"
        elif score <= 35:
            signal = "🔴 Negative"
        else:
            signal = "🟡 Neutral / Mixed"

        summary.append(
            {
                "Brand": brand,
                "Product": product,
                "Articles Found": len(news),
                "Online Market Score": round(score, 1),
                "Signal": signal,
            }
        )

    return scores, pd.DataFrame(summary), headlines


# ============================================================
# UI / DATA LOAD
# ============================================================
st.title("📊 AI Market Profit Advisor")
st.subheader(
    "AI-Powered Profit/Loss Prediction, Market Analysis & 2027 Business Recommendation"
)
st.caption(
    "Works with product datasets such as clothing, food, cars, electronics, cosmetics and other sales businesses."
)
st.divider()

if not os.path.exists(DATA_PATH):
    st.error(f"❌ {DATA_PATH} not found. Put your CSV inside the data folder.")
    st.stop()

try:
    df = validate_and_clean(pd.read_csv(DATA_PATH))
except Exception as exc:
    st.error(f"❌ Dataset error: {exc}")
    st.stop()

if df.empty:
    st.error("❌ Dataset contains no usable records.")
    st.stop()

domain_key = infer_domain_label(df)
domain_emoji, domain_product_label = domain_title(domain_key)

# Reset online signals whenever dataset changes.
current_dataset_fp = dataframe_fingerprint(df)
if st.session_state.get("dataset_fingerprint") != current_dataset_fp:
    st.session_state["dataset_fingerprint"] = current_dataset_fp
    st.session_state["online_scores"] = {}
    st.session_state["market_signal_table"] = pd.DataFrame()
    st.session_state["market_headlines"] = []

# ============================================================
# DATASET MANAGEMENT
# ============================================================
st.header("📂 Dataset Management")
st.caption("Upload a CSV using the exact fixed 16-column format.")

with st.expander("📋 Required CSV Format", expanded=False):
    st.code(",".join(REQUIRED_COLUMNS))
    st.info(
        "A valid upload replaces the current dataset. Invalid files are rejected and are not saved."
    )


def show_invalid_csv_popup(message, missing=None, extra=None):
    @st.dialog("🚨 INVALID CSV FILE")
    def popup():
        st.error("❌ This CSV cannot be analyzed.")
        st.write(message)
        if missing:
            st.warning("Missing Columns")
            st.code("\n".join(missing))
        if extra:
            st.warning("Extra Columns")
            st.code("\n".join(extra))
        st.info("Use the exact 16-column format shown above.")

    popup()


uploaded_file = st.file_uploader(
    "Upload CSV for Analysis",
    type=["csv"],
    key="csv_upload",
)

if "processed_upload_hash" not in st.session_state:
    st.session_state["processed_upload_hash"] = None

if uploaded_file is not None:
    try:
        upload_bytes = uploaded_file.getvalue()
        upload_hash = hashlib.sha256(upload_bytes).hexdigest()

        if st.session_state["processed_upload_hash"] != upload_hash:
            raw_upload = pd.read_csv(pd.io.common.BytesIO(upload_bytes))
            raw_upload.columns = [str(c).strip() for c in raw_upload.columns]

            upload_columns = list(raw_upload.columns)
            missing_upload = [c for c in REQUIRED_COLUMNS if c not in upload_columns]
            extra_upload = [c for c in upload_columns if c not in REQUIRED_COLUMNS]

            if missing_upload or extra_upload or upload_columns != REQUIRED_COLUMNS:
                order_issue = (
                    not missing_upload
                    and not extra_upload
                    and upload_columns != REQUIRED_COLUMNS
                )
                msg = "The uploaded CSV does not match the required 16-column schema."
                if order_issue:
                    msg += " Column names are correct, but their order is incorrect."
                show_invalid_csv_popup(msg, missing_upload, extra_upload)
                st.stop()

            new_df = validate_and_clean(raw_upload, strict_schema=True)
            if new_df.empty:
                show_invalid_csv_popup("The uploaded CSV contains no valid records.")
                st.stop()

            new_df = new_df.drop_duplicates().reset_index(drop=True)
            new_df.to_csv(DATA_PATH, index=False)
            df = new_df
            domain_key = infer_domain_label(df)
            domain_emoji, domain_product_label = domain_title(domain_key)

            st.session_state["processed_upload_hash"] = upload_hash
            st.session_state["online_scores"] = {}
            st.session_state["market_signal_table"] = pd.DataFrame()
            st.session_state["market_headlines"] = []
            st.session_state.pop("model_fingerprint", None)

            st.success(f"✅ CSV loaded successfully: {len(df):,} valid records.")
            st.info("🤖 The AI model will retrain once for this new dataset.")
        else:
            st.success("✅ Uploaded CSV is already loaded; no repeated retraining on refresh.")

    except pd.errors.EmptyDataError:
        show_invalid_csv_popup("The uploaded CSV is empty.")
        st.stop()
    except Exception as exc:
        show_invalid_csv_popup(str(exc))
        st.stop()

# ============================================================
# CURRENT DATASET SUMMARY
# ============================================================
st.header("📈 Current Dataset")
a, b, c, d = st.columns(4)
a.metric("Total Records", f"{len(df):,}")
b.metric("Years", f"{df['date'].dt.year.nunique()}")
c.metric("Total Revenue", money(df["revenue"].sum()))
d.metric("Total Profit", money(df["profit"].sum()))

with st.expander("👀 View Dataset"):
    st.dataframe(df, use_container_width=True, hide_index=True)

# ============================================================
# MODEL
# ============================================================
try:
    model, features, mae, rmse, r2 = get_model_for_dataframe(df)
except Exception as exc:
    st.error(f"❌ AI model error: {exc}")
    st.stop()

# ============================================================
# 2025 ACTUAL
# ============================================================
st.divider()
st.header("📌 2025 Actual Business Analysis")
df_2025 = df[df["date"].dt.year == 2025].copy()

rev25 = profit25 = units25 = margin25 = 0.0
if df_2025.empty:
    st.warning("No 2025 actual data found in the dataset.")
else:
    rev25 = float(df_2025["revenue"].sum())
    profit25 = float(df_2025["profit"].sum())
    units25 = float(df_2025["units_sold"].sum())
    margin25 = profit25 / rev25 * 100 if rev25 else 0

    a, b, c, d = st.columns(4)
    a.metric("2025 Revenue", money(rev25))
    b.metric("2025 Profit", money(profit25))
    c.metric("2025 Units", f"{units25:,.0f}")
    d.metric("2025 Margin", f"{margin25:.2f}%")

    m25 = monthly_actual(df, 2025)
    monthly_chart(m25, "Profit", "📈 2025 Monthly Actual Profit", "Profit (₹)")

    st.subheader("🏷️ 2025 Brand-wise Actual Sales")
    b25 = brand_actual(df, 2025)
    if not b25.empty:
        display_b25 = b25.rename(
            columns={
                "brand": "Brand",
                "Units": "Units Sold",
                "Revenue": "Revenue",
                "Profit": "Profit",
                "Profit_Margin": "Profit Margin %",
            }
        )
        st.dataframe(
            display_b25.style.format(
                {
                    "Units Sold": "{:,.0f}",
                    "Revenue": "₹ {:,.0f}",
                    "Profit": "₹ {:,.0f}",
                    "Profit Margin %": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(b25.set_index("brand")["Revenue"], y_label="Revenue (₹)")

# Product-wise 2025
st.divider()
st.header(f"{domain_emoji} 2025 {domain_product_label} Sales — What Was Actually Sold?")
if not df_2025.empty:
    product25 = product_sales_actual(df, 2025)
    if not product25.empty:
        st.success(
            f"✅ 2025 me {product25['product'].nunique()} {domain_product_label.lower()}s ki actual sales record hui."
        )
        display25 = product25.rename(
            columns={
                "product": domain_product_label,
                "category": "Category",
                "Brands": "Brands Selling",
                "Units_Sold": "Units Sold",
                "Revenue": "Revenue",
                "Profit": "Profit",
                "Avg_Selling_Price": "Avg Selling Price",
                "Profit_Margin": "Profit Margin %",
            }
        )
        st.dataframe(
            display25.style.format(
                {
                    "Units Sold": "{:,.0f}",
                    "Revenue": "₹ {:,.0f}",
                    "Profit": "₹ {:,.0f}",
                    "Avg Selling Price": "₹ {:,.2f}",
                    "Profit Margin %": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader(f"📦 2025 {domain_product_label}-wise Units Sold")
        st.bar_chart(product25.set_index("product")["Units_Sold"], y_label="Units Sold")

        selected_product_25 = st.selectbox(
            f"{domain_emoji} Select {domain_product_label.lower()} to see month-wise sales (2025)",
            sorted(product25["product"].tolist()),
            key="selected_product_2025",
        )
        mp25 = monthly_product_sales(df, 2025, selected_product_25)
        st.subheader(f"📅 {selected_product_25} — 2025 Month-wise Sales")
        st.dataframe(
            mp25[["Month", "Units_Sold", "Revenue", "Profit"]]
            .rename(columns={"Units_Sold": "Units Sold"})
            .style.format(
                {
                    "Units Sold": "{:,.0f}",
                    "Revenue": "₹ {:,.0f}",
                    "Profit": "₹ {:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# 2026 ACTUAL
# ============================================================
st.divider()
st.header("📌 2026 Actual Business Analysis")
df_2026 = df[df["date"].dt.year == 2026].copy()

rev26 = profit26 = units26 = margin26 = 0.0
rev_growth_25_26 = profit_growth_25_26 = units_growth_25_26 = 0.0

if df_2026.empty:
    st.warning(
        "No 2026 actual data found. Upload 2026 actual data before using 2027 prediction/recommendation."
    )
else:
    rev26 = float(df_2026["revenue"].sum())
    profit26 = float(df_2026["profit"].sum())
    units26 = float(df_2026["units_sold"].sum())
    margin26 = profit26 / rev26 * 100 if rev26 else 0
    rev_growth_25_26 = safe_growth(rev26, rev25) if not df_2025.empty else 0
    profit_growth_25_26 = safe_growth(profit26, profit25) if not df_2025.empty else 0
    units_growth_25_26 = safe_growth(units26, units25) if not df_2025.empty else 0

    a, b, c, d = st.columns(4)
    a.metric("2026 Actual Revenue", money(rev26), pct(rev_growth_25_26) + " vs 2025")
    b.metric("2026 Actual Profit", money(profit26), pct(profit_growth_25_26) + " vs 2025")
    c.metric("2026 Actual Units", f"{units26:,.0f}", pct(units_growth_25_26) + " vs 2025")
    d.metric("2026 Actual Margin", f"{margin26:.2f}%")

    m26 = monthly_actual(df, 2026)
    monthly_chart(m26, "Profit", "📈 2026 Monthly Actual Profit", "Profit (₹)")
    monthly_chart(m26, "Revenue", "💰 2026 Monthly Actual Revenue", "Revenue (₹)")

    st.subheader("🏷️ 2026 Brand-wise Actual Sales")
    b26 = brand_actual(df, 2026)
    if not b26.empty:
        display_b26 = b26.rename(
            columns={
                "brand": "Brand",
                "Units": "Units Sold",
                "Revenue": "Revenue",
                "Profit": "Profit",
                "Profit_Margin": "Profit Margin %",
            }
        )
        st.dataframe(
            display_b26.style.format(
                {
                    "Units Sold": "{:,.0f}",
                    "Revenue": "₹ {:,.0f}",
                    "Profit": "₹ {:,.0f}",
                    "Profit Margin %": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(b26.set_index("brand")["Revenue"], y_label="Revenue (₹)")
        st.subheader("🏷️ 2026 Brand-wise Profit")
        st.bar_chart(b26.set_index("brand")["Profit"], y_label="Profit (₹)")

# Product-wise 2026
st.subheader(f"{domain_emoji} 2026 {domain_product_label} Sales — What Was Actually Sold?")
if not df_2026.empty:
    product26 = product_sales_actual(df, 2026)
    if not product26.empty:
        st.success(
            f"✅ 2026 me {product26['product'].nunique()} {domain_product_label.lower()}s ki actual sales record hui."
        )
        display26 = product26.rename(
            columns={
                "product": domain_product_label,
                "category": "Category",
                "Brands": "Brands Selling",
                "Units_Sold": "Units Sold",
                "Revenue": "Revenue",
                "Profit": "Profit",
                "Avg_Selling_Price": "Avg Selling Price",
                "Profit_Margin": "Profit Margin %",
            }
        )
        st.dataframe(
            display26.style.format(
                {
                    "Units Sold": "{:,.0f}",
                    "Revenue": "₹ {:,.0f}",
                    "Profit": "₹ {:,.0f}",
                    "Avg Selling Price": "₹ {:,.2f}",
                    "Profit Margin %": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader(f"📦 2026 {domain_product_label}-wise Units Sold")
        st.bar_chart(product26.set_index("product")["Units_Sold"], y_label="Units Sold")

        selected_product_26 = st.selectbox(
            f"{domain_emoji} Select {domain_product_label.lower()} to see month-wise sales (2026)",
            sorted(product26["product"].tolist()),
            key="selected_product_2026",
        )
        mp26 = monthly_product_sales(df, 2026, selected_product_26)
        st.subheader(f"📅 {selected_product_26} — 2026 Month-wise Sales")
        st.dataframe(
            mp26[["Month", "Units_Sold", "Revenue", "Profit"]]
            .rename(columns={"Units_Sold": "Units Sold"})
            .style.format(
                {
                    "Units Sold": "{:,.0f}",
                    "Revenue": "₹ {:,.0f}",
                    "Profit": "₹ {:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# 2025 VS 2026
# ============================================================
st.divider()
st.header("📊 2025 vs 2026 Growth Analysis")
if not df_2025.empty and not df_2026.empty:
    comparison = pd.DataFrame(
        {
            "2025 Actual": [rev25, profit25, units25],
            "2026 Actual": [rev26, profit26, units26],
        },
        index=["Revenue", "Profit", "Units"],
    )
    st.dataframe(
        comparison.style.format(
            {
                "2025 Actual": "{:,.0f}",
                "2026 Actual": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )
    a, b, c = st.columns(3)
    a.metric("Revenue Growth", pct(rev_growth_25_26))
    b.metric("Profit Growth", pct(profit_growth_25_26))
    c.metric("Units Growth", pct(units_growth_25_26))
else:
    st.info("Upload both 2025 and 2026 actual data to calculate year-over-year growth.")

# ============================================================
# AI EXECUTIVE BUSINESS SUMMARY
# ============================================================
st.divider()
st.header("🤖 AI Executive Business Summary")

try:
    latest_year = int(df["date"].dt.year.max())
    latest_df = df[df["date"].dt.year == latest_year].copy()

    summary_lines = []
    action_lines = []

    if not latest_df.empty:
        latest_revenue = float(latest_df["revenue"].sum())
        latest_profit = float(latest_df["profit"].sum())
        latest_units = float(latest_df["units_sold"].sum())
        latest_margin = latest_profit / latest_revenue * 100 if latest_revenue else 0

        prod_latest = latest_df.groupby("product").agg(
            Units=("units_sold", "sum"),
            Revenue=("revenue", "sum"),
            Profit=("profit", "sum"),
        )
        prod_latest["Margin"] = np.where(
            prod_latest["Revenue"] != 0,
            prod_latest["Profit"] / prod_latest["Revenue"] * 100,
            0,
        )

        brand_latest = latest_df.groupby("brand").agg(
            Units=("units_sold", "sum"),
            Revenue=("revenue", "sum"),
            Profit=("profit", "sum"),
        )

        top_product = prod_latest["Profit"].idxmax()
        top_product_profit = float(prod_latest.loc[top_product, "Profit"])
        top_product_margin = float(prod_latest.loc[top_product, "Margin"])
        top_brand = brand_latest["Profit"].idxmax()
        top_brand_profit = float(brand_latest.loc[top_brand, "Profit"])

        summary_lines.append(
            f"📌 In {latest_year}, total {domain_product_label.lower()} sales were "
            f"{latest_units:,.0f} units with revenue of {money(latest_revenue)} "
            f"and profit of {money(latest_profit)} (margin {latest_margin:.2f}%)."
        )
        summary_lines.append(
            f"🏆 Best-performing {domain_product_label.lower()}: **{top_product}** "
            f"with profit {money(top_product_profit)} and margin {top_product_margin:.2f}%."
        )
        summary_lines.append(
            f"🏢 Best brand: **{top_brand}** with profit {money(top_brand_profit)}."
        )

        positive_revenue = prod_latest[prod_latest["Revenue"] > 0]
        if not positive_revenue.empty:
            risk_product = positive_revenue["Margin"].idxmin()
            risk_margin = float(positive_revenue.loc[risk_product, "Margin"])
            if risk_margin < 10:
                action_lines.append(
                    f"⚠️ Review **{risk_product}**: its {latest_year} profit margin is only {risk_margin:.2f}%."
                )

        if latest_margin >= 20:
            action_lines.append(
                "✅ Overall profitability is healthy; protect margin while scaling the strongest products."
            )
        elif latest_margin >= 10:
            action_lines.append(
                "🟡 Profitability is moderate; optimize pricing, discount and operating costs before aggressive expansion."
            )
        else:
            action_lines.append(
                "🔴 Overall margin is low; prioritize cost control and pricing review before increasing volume."
            )

    if not df_2025.empty and not df_2026.empty:
        summary_lines.append(
            f"📈 2025→2026 actual revenue growth is **{pct(rev_growth_25_26)}**, "
            f"while profit growth is **{pct(profit_growth_25_26)}**."
        )

        p25 = df_2025.groupby("product")["profit"].sum()
        p26 = df_2026.groupby("product")["profit"].sum()
        merged_profit = p25.to_frame("Profit_25").join(
            p26.to_frame("Profit_26"), how="inner"
        )
        if not merged_profit.empty:
            merged_profit["Growth"] = merged_profit.apply(
                lambda r: safe_growth(r["Profit_26"], r["Profit_25"]),
                axis=1,
            )
            fastest = merged_profit["Growth"].idxmax()
            fastest_growth = float(merged_profit.loc[fastest, "Growth"])
            summary_lines.append(
                f"🚀 Fastest profit-growing product: **{fastest}** at {pct(fastest_growth)} from 2025 to 2026."
            )
            if fastest_growth > 0:
                action_lines.append(
                    f"🎯 Increase focus on **{fastest}** because its historical profit trend is positive."
                )

    if summary_lines:
        st.success("\n\n".join(summary_lines))
    else:
        st.info("Upload usable product, sales and profit records to generate the executive summary.")
    if action_lines:
        st.info("\n\n".join(action_lines))
except Exception as exc:
    st.warning(f"⚠️ Executive summary could not be generated: {exc}")

# ============================================================
# 2027 SALES & PROFIT FORECAST
# ============================================================
st.divider()
st.header("🔮 2027 Sales & Profit Prediction")
st.info(
    "2025 and 2026 are treated as actual values from the dataset. 2027 is a forecast based on the observed 2025→2026 monthly trend."
)

pred_units27 = pred_rev27 = pred_profit27 = pred_margin27 = 0.0
rev_growth_26_27 = profit_growth_26_27 = 0.0

if df_2026.empty:
    st.warning("2027 prediction requires 2026 actual data.")
else:
    m26 = monthly_actual(df, 2026)
    m25 = monthly_actual(df, 2025) if not df_2025.empty else None

    forecast_rows = []
    for i, month_name in enumerate(MONTH_NAMES):
        base26 = m26.iloc[i]
        base25 = m25.iloc[i] if m25 is not None else None

        unit_growth = safe_growth(base26["Units"], base25["Units"]) if base25 is not None and base25["Units"] else 0
        revenue_growth = safe_growth(base26["Revenue"], base25["Revenue"]) if base25 is not None and base25["Revenue"] else 0
        profit_growth = safe_growth(base26["Profit"], base25["Profit"]) if base25 is not None and base25["Profit"] else 0

        unit_growth = float(np.clip(unit_growth, -25, 35))
        revenue_growth = float(np.clip(revenue_growth, -25, 35))
        profit_growth = float(np.clip(profit_growth, -35, 40))

        forecast_rows.append(
            {
                "Month": month_name + " 2027",
                "MonthNo": i + 1,
                "Expected Units": round(base26["Units"] * (1 + unit_growth / 100)),
                "Expected Revenue": base26["Revenue"] * (1 + revenue_growth / 100),
                "Expected Profit": base26["Profit"] * (1 + profit_growth / 100),
                "Revenue Growth vs 2026 (%)": revenue_growth,
                "Profit Growth vs 2026 (%)": profit_growth,
            }
        )

    forecast_2027 = pd.DataFrame(forecast_rows)
    pred_units27 = float(forecast_2027["Expected Units"].sum())
    pred_rev27 = float(forecast_2027["Expected Revenue"].sum())
    pred_profit27 = float(forecast_2027["Expected Profit"].sum())
    pred_margin27 = pred_profit27 / pred_rev27 * 100 if pred_rev27 else 0
    rev_growth_26_27 = safe_growth(pred_rev27, rev26)
    profit_growth_26_27 = safe_growth(pred_profit27, profit26)

    a, b, c, d = st.columns(4)
    a.metric("2027 Predicted Units", f"{pred_units27:,.0f}")
    b.metric("2027 Predicted Revenue", money(pred_rev27), pct(rev_growth_26_27) + " vs 2026")
    c.metric("2027 Predicted Profit", money(pred_profit27), pct(profit_growth_26_27) + " vs 2026")
    d.metric("2027 Predicted Margin", f"{pred_margin27:.2f}%")

    st.subheader("📅 2027 Month-wise Prediction")
    st.dataframe(
        forecast_2027.drop(columns=["MonthNo"]).style.format(
            {
                "Expected Units": "{:,.0f}",
                "Expected Revenue": "₹ {:,.0f}",
                "Expected Profit": "₹ {:,.0f}",
                "Revenue Growth vs 2026 (%)": "{:+.2f}%",
                "Profit Growth vs 2026 (%)": "{:+.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    monthly_chart(
        forecast_2027.rename(columns={"Expected Profit": "Profit", "Month": "Month"}),
        "Profit",
        "📈 2027 Predicted Profit Trend",
        "Profit (₹)",
    )

    if not df_2025.empty:
        growth_table = pd.DataFrame(
            {
                "2025 Actual": [rev25, profit25],
                "2026 Actual": [rev26, profit26],
                "2027 Predicted": [pred_rev27, pred_profit27],
            },
            index=["Revenue", "Profit"],
        )
        st.subheader("📊 Revenue & Profit: 2025 → 2026 → 2027")
        st.dataframe(
            growth_table.style.format("₹ {:,.0f}"),
            use_container_width=True,
        )

# ============================================================
# OPTIONAL LIVE ONLINE MARKET ANALYSIS
# ============================================================
st.divider()
st.header("🌐 Live Online Market Intelligence")
st.caption(
    "This uses public Google News RSS headlines as a supporting market signal. It is not direct customer-sales data and is not guaranteed demand."
)

if "online_scores" not in st.session_state:
    st.session_state["online_scores"] = {}
if "market_signal_table" not in st.session_state:
    st.session_state["market_signal_table"] = pd.DataFrame()
if "market_headlines" not in st.session_state:
    st.session_state["market_headlines"] = []

if not df_2026.empty:
    base_products_2027 = build_2027_product_forecast(
        df,
        online_scores=None,
    )
    if not base_products_2027.empty:
        if st.button("🌐 Run / Refresh Online Market Analysis", key="run_market_analysis"):
            with st.spinner("🌐 Checking current online market/news signals..."):
                scores, signal_table, headlines = build_online_market_signals(
                    base_products_2027,
                    max_products=3,
                )
            st.session_state["online_scores"] = scores
            st.session_state["market_signal_table"] = signal_table
            st.session_state["market_headlines"] = headlines
            st.rerun()

        market_signal_table = st.session_state["market_signal_table"]
        market_headlines = st.session_state["market_headlines"]

        if market_signal_table.empty:
            st.info(
                "Click the button above to run live market analysis. The rest of the app works without internet/news access."
            )
        else:
            avg_online = float(market_signal_table["Online Market Score"].mean())
            st.metric("Overall Online Market Signal", f"{avg_online:.1f}/100")
            st.dataframe(
                market_signal_table.sort_values(
                    "Online Market Score",
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("📰 Latest Market Headlines Used"):
                if market_headlines:
                    headline_df = pd.DataFrame(market_headlines)
                    st.dataframe(
                        headline_df[["Product", "Brand", "Headline", "Source", "Date"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No headlines were returned for the selected products.")

# ============================================================
# 2027 PRODUCT RECOMMENDATION
# ============================================================
st.divider()
st.header("🏆 2027 Best Product Recommendation")

online_scores = st.session_state.get("online_scores", {})
product_2027 = build_2027_product_forecast(df, online_scores=online_scores)

if product_2027.empty:
    st.warning("Product recommendation requires actual 2026 product data.")
else:
    best = product_2027.iloc[0]

    a, b, c, d = st.columns(4)
    a.metric("🥇 Best Product", str(best["product"]))
    b.metric("Brand", str(best["brand"]))
    c.metric("Predicted 2027 Profit", money(best["Predicted Profit 2027"]))
    d.metric("Recommendation Score", f"{best['Recommendation Score']:.1f}/100")

    if best["Predicted Profit 2027"] > 0:
        st.success(
            f"🟢 **Recommended for 2027: {best['product']} ({best['brand']})** — "
            f"predicted profit {money(best['Predicted Profit 2027'])}, "
            f"margin {best['Predicted Margin 2027 (%)']:.2f}%, "
            f"revenue growth {best['Revenue Growth 2026→2027 (%)']:+.2f}% vs 2026."
        )
    else:
        st.error(
            f"🔴 No product has positive predicted 2027 profit. Review {best['product']} before increasing production."
        )

    display = product_2027[
        [
            "brand",
            "product",
            "category",
            "2025 Units",
            "2026 Units",
            "2025 Revenue",
            "2026 Revenue",
            "2025 Profit",
            "2026 Profit",
            "Predicted Units 2027",
            "Predicted Revenue 2027",
            "Predicted Profit 2027",
            "Predicted Margin 2027 (%)",
            "Revenue Growth 25-26 (%)",
            "Profit Growth 25-26 (%)",
            "Revenue Growth 2026→2027 (%)",
            "Online Market Score",
            "Recommendation Score",
            "Recommendation",
        ]
    ].copy()

    st.subheader("📋 2027 Product Ranking")
    st.dataframe(
        display.style.format(
            {
                "2025 Units": "{:,.0f}",
                "2026 Units": "{:,.0f}",
                "2025 Revenue": "₹ {:,.0f}",
                "2026 Revenue": "₹ {:,.0f}",
                "2025 Profit": "₹ {:,.0f}",
                "2026 Profit": "₹ {:,.0f}",
                "Predicted Units 2027": "{:,.0f}",
                "Predicted Revenue 2027": "₹ {:,.0f}",
                "Predicted Profit 2027": "₹ {:,.0f}",
                "Predicted Margin 2027 (%)": "{:.2f}%",
                "Revenue Growth 25-26 (%)": "{:+.2f}%",
                "Profit Growth 25-26 (%)": "{:+.2f}%",
                "Revenue Growth 2026→2027 (%)": "{:+.2f}%",
                "Online Market Score": "{:.1f}",
                "Recommendation Score": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🥇 Top 3 Products to Focus on in 2027")
    for i, (_, row) in enumerate(product_2027.head(3).iterrows(), start=1):
        st.markdown(
            f"**{i}. {row['product']} ({row['brand']})** — "
            f"2026 profit {money(row['2026 Profit'])}, "
            f"predicted 2027 profit {money(row['Predicted Profit 2027'])}, "
            f"2025→2026 profit growth {row['Profit Growth 25-26 (%)']:+.2f}%, "
            f"{row['Recommendation']}"
        )

    st.subheader("📈 Product Profit: 2026 Actual vs 2027 Predicted")
    comparison = product_2027.set_index("product")[[
        "2026 Profit", "Predicted Profit 2027"
    ]]
    st.bar_chart(comparison)

# ============================================================
# WHAT-IF PROFIT SIMULATOR
# ============================================================
st.divider()
st.header("🧪 What-If Profit Simulator")
st.caption("Change business assumptions and compare a scenario with the current product economics.")

products = sorted(df["product"].dropna().unique().tolist())
if products:
    selected_sim_product = st.selectbox("Simulator Product", products, key="sim_product")
    product_rows = df[df["product"] == selected_sim_product].copy()

    default_units = int(max(1, round(product_rows["units_sold"].median())))
    default_price = float(max(1, product_rows["selling_price"].median()))
    default_discount = float(np.clip(product_rows["discount_percent"].median(), 0, 100))
    default_marketing = float(max(0, product_rows["marketing_cost"].median()))
    default_other = float(max(0, product_rows["other_cost"].median()))
    default_growth = float(product_rows["market_growth_percent"].median())

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_units = st.number_input("Scenario Units Sold", min_value=1, value=default_units, step=1, key="sim_units")
        sim_price = st.number_input("Scenario Selling Price (₹)", min_value=1.0, value=default_price, step=100.0, key="sim_price")
        sim_discount = st.number_input("Scenario Discount (%)", min_value=0.0, max_value=100.0, value=default_discount, step=0.5, key="sim_discount")
    with col2:
        sim_marketing = st.number_input("Scenario Marketing Cost (₹)", min_value=0.0, value=default_marketing, step=1000.0, key="sim_marketing")
        sim_other = st.number_input("Scenario Other Cost (₹)", min_value=0.0, value=default_other, step=1000.0, key="sim_other")
        sim_growth = st.number_input("Scenario Market Growth (%)", min_value=-100.0, max_value=100.0, value=default_growth, step=0.5, key="sim_growth")
    with col3:
        sim_condition_options = sorted(df["market_condition"].dropna().unique().tolist())
        sim_comp_options = sorted(df["competitor_pressure"].dropna().unique().tolist())
        sim_condition = st.selectbox("Market Condition", sim_condition_options, key="sim_condition")
        sim_comp = st.selectbox("Competitor Pressure", sim_comp_options, key="sim_comp")
        run_sim = st.button("🧪 Run Scenario", use_container_width=True, key="run_sim")

    if run_sim:
        try:
            unit_cost_values = np.where(
                product_rows["units_sold"] > 0,
                product_rows["product_cost"] / product_rows["units_sold"],
                np.nan,
            )
            unit_cost = float(np.nanmedian(unit_cost_values)) if np.isfinite(np.nanmedian(unit_cost_values)) else 0.0

            effective_price = sim_price * (1 - sim_discount / 100)
            scenario_revenue = sim_units * effective_price
            scenario_product_cost = sim_units * unit_cost
            scenario_total_cost = scenario_product_cost + sim_marketing + sim_other
            scenario_profit = scenario_revenue - scenario_total_cost
            scenario_margin = scenario_profit / scenario_revenue * 100 if scenario_revenue else 0

            base_profit = float(product_rows["profit"].median())
            profit_change = safe_growth(scenario_profit, base_profit)

            a, b, c, d = st.columns(4)
            a.metric("Scenario Revenue", money(scenario_revenue))
            b.metric("Scenario Cost", money(scenario_total_cost))
            c.metric("Scenario Profit/Loss", money(scenario_profit))
            d.metric("Scenario Margin", f"{scenario_margin:.2f}%")

            st.metric("Profit Change vs Product Median", pct(profit_change))

            if scenario_profit > 0:
                st.success("🟢 Scenario is profitable under the entered assumptions.")
            else:
                st.error("🔴 Scenario results in a loss. Reduce cost/discount or improve price/volume.")
        except Exception as exc:
            st.error(f"❌ Simulator error: {exc}")

# ============================================================
# INDIVIDUAL AI PROFIT PREDICTOR
# ============================================================
st.divider()
st.header("🤖 Individual Product Profit/Loss Predictor")

if products:
    predictor_product = st.selectbox("Product", products, key="predictor_product")
    product_data = df[df["product"] == predictor_product].copy()

    product_brands = sorted(product_data["brand"].dropna().unique().tolist()) or sorted(df["brand"].dropna().unique().tolist())
    product_categories = sorted(product_data["category"].dropna().unique().tolist()) or sorted(df["category"].dropna().unique().tolist())
    market_conditions = sorted(df["market_condition"].dropna().unique().tolist())
    competitor_pressures = sorted(df["competitor_pressure"].dropna().unique().tolist())

    a, b, c = st.columns(3)
    with a:
        predictor_brand = st.selectbox("Brand", product_brands, key="predictor_brand")
        predictor_category = st.selectbox("Category", product_categories, key="predictor_category")
    with b:
        predictor_units = st.number_input("Units Sold", min_value=1, value=100, step=1, key="predictor_units")
        predictor_price = st.number_input("Selling Price (₹)", min_value=1.0, value=1000.0, step=100.0, key="predictor_price")
        predictor_discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="predictor_discount")
    with c:
        predictor_marketing = st.number_input("Marketing Cost (₹)", min_value=0.0, value=5000.0, step=500.0, key="predictor_marketing")
        predictor_other = st.number_input("Other Cost (₹)", min_value=0.0, value=2000.0, step=500.0, key="predictor_other")
        predictor_growth = st.number_input("Market Growth (%)", min_value=-100.0, max_value=100.0, value=4.0, step=0.5, key="predictor_growth")

    predictor_condition = st.selectbox("Market Condition", market_conditions, key="predictor_condition")
    predictor_competitor = st.selectbox("Competitor Pressure", competitor_pressures, key="predictor_competitor")

    if st.button("🔮 Predict Profit/Loss", use_container_width=True, key="predict_profit"):
        try:
            unit_cost_array = np.where(
                product_data["units_sold"] > 0,
                product_data["product_cost"] / product_data["units_sold"],
                np.nan,
            )
            unit_cost = float(np.nanmedian(unit_cost_array)) if np.isfinite(np.nanmedian(unit_cost_array)) else 0.0

            effective_price = predictor_price * (1 - predictor_discount / 100)
            revenue = predictor_units * effective_price
            product_cost = predictor_units * unit_cost
            total_cost = product_cost + predictor_marketing + predictor_other

            input_data = pd.DataFrame(
                {
                    "date": [pd.Timestamp.today()],
                    "brand": [predictor_brand],
                    "product": [predictor_product],
                    "category": [predictor_category],
                    "units_sold": [predictor_units],
                    "selling_price": [effective_price],
                    "revenue": [revenue],
                    "product_cost": [product_cost],
                    "discount_percent": [predictor_discount],
                    "marketing_cost": [predictor_marketing],
                    "other_cost": [predictor_other],
                    "total_cost": [total_cost],
                    "profit": [0.0],
                    "market_growth_percent": [predictor_growth],
                    "market_condition": [predictor_condition],
                    "competitor_pressure": [predictor_competitor],
                }
            )

            prepared_input = prepare_data(input_data)
            encoded_input = prepared_input.reindex(columns=features, fill_value=0)
            prediction = float(model.predict(encoded_input)[0])
            margin = prediction / revenue * 100 if revenue else 0

            a, b, c = st.columns(3)
            a.metric("Predicted Profit/Loss", money(prediction))
            b.metric("Expected Revenue", money(revenue))
            c.metric("Estimated Total Cost", money(total_cost))
            st.metric("Expected Profit Margin", f"{margin:.2f}%")

            if prediction < 0:
                st.error("🔴 LOSS — Review price, costs, discount and competitor pressure.")
            elif margin < 5:
                st.warning("🟠 Low margin — profitable but needs cost/price optimization.")
            elif margin < 15:
                st.info("🟡 Moderate margin — monitor market conditions.")
            else:
                st.success("🟢 PROFIT — Strong predicted margin under the entered assumptions.")
        except Exception as exc:
            st.error(f"❌ Prediction error: {exc}")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "AI Market Profit Advisor | 2025 Actual + 2026 Actual + 2027 Prediction + Product Recommendation + Optional Live Market Signals"
)
