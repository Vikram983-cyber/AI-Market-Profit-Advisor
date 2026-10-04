import os
import hashlib
import re
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Market Profit Advisor",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = "data/sales_data.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "profit_model.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "model_features.pkl")

os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

REQUIRED_COLUMNS = [
    "date", "brand", "product", "category", "units_sold",
    "selling_price", "revenue", "product_cost", "discount_percent",
    "marketing_cost", "other_cost", "total_cost", "profit",
    "market_growth_percent", "market_condition", "competitor_pressure"
]

CATEGORICAL_COLUMNS = [
    "brand", "product", "category", "market_condition", "competitor_pressure"
]
NUMERIC_COLUMNS = [
    "units_sold", "selling_price", "revenue", "product_cost",
    "discount_percent", "marketing_cost", "other_cost", "total_cost",
    "profit", "market_growth_percent"
]
MONTHS = list(range(1, 13))
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


def money(value):
    return f"₹ {float(value):,.0f}"


def pct(value):
    return f"{float(value):+.2f}%"


def safe_growth(current, previous):
    if previous is None or pd.isna(previous) or previous == 0:
        return 0.0
    return (current - previous) / abs(previous) * 100.0



def infer_domain_label(df):
    """Infer a broad business domain from category/product text."""
    values = []
    for col in ("category", "product"):
        if col in df.columns:
            values.extend(df[col].dropna().astype(str).str.lower().tolist())
    text = " ".join(values)
    rules = [
        ("food", ["food", "snack", "beverage", "drink", "juice", "dairy", "rice", "noodle", "namkeen", "paneer", "grocery", "bakery"]),
        ("clothing", ["clothing", "apparel", "wear", "shirt", "t-shirt", "jeans", "hoodie", "kurti", "dress", "fashion", "footwear", "shoe"]),
        ("electronics", ["electronic", "laptop", "mobile", "smartphone", "tablet", "television", " tv", "camera", "headphone", "earphone", "computer"]),
        ("automotive", ["automotive", "car", "bike", "motorcycle", "vehicle", "tyre", "tire"]),
        ("cosmetics", ["cosmetic", "makeup", "skincare", "skin care", "shampoo", "conditioner", "beauty", "lipstick", "cream"]),
        ("furniture", ["furniture", "chair", "table", "sofa", "bed", "desk", "cabinet"]),
        ("stationery", ["stationery", "notebook", "pen", "pencil", "book", "paper"]),
    ]
    for label, keywords in rules:
        if any(k in text for k in keywords):
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
        "product": ("📦", "Product"),
    }
    return labels.get(domain, labels["product"])

def validate_and_clean(df, strict_schema=False):
    """Validate CSV and return a clean, numeric dataframe.

    strict_schema=True is used for uploads so the uploaded CSV must contain
    exactly the fixed 16-column schema in the required order. No data is saved
    or merged until this validation succeeds.
    """
    df = df.copy()
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
    invalid_numeric = {c: int(df[c].isna().sum()) for c in NUMERIC_COLUMNS if df[c].isna().any()}

    if strict_schema and (invalid_date > 0 or invalid_numeric):
        details = []
        if invalid_date:
            details.append(f"Invalid/blank dates: {invalid_date}")
        if invalid_numeric:
            details.append("Invalid/blank numeric values: " + ", ".join(f"{k} ({v})" for k, v in invalid_numeric.items()))
        raise ValueError(" | ".join(details))

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("Unknown").astype(str)

    # Drop unusable records only for the trusted/base dataset.
    df = df.dropna(subset=["date", "units_sold", "selling_price", "profit"])
    df = df.reset_index(drop=True)
    return df


def prepare_data(df):
    """Prepare training/prediction data with stable one-hot columns."""
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])
    for col in CATEGORICAL_COLUMNS:
        if col not in work.columns:
            work[col] = "Unknown"
        work[col] = work[col].fillna("Unknown").astype(str)

    work["year"] = work["date"].dt.year
    work["month"] = work["date"].dt.month
    work["day"] = work["date"].dt.day

    encoded = pd.get_dummies(work, columns=CATEGORICAL_COLUMNS, dtype=int)
    return encoded


def train_model(df):
    prepared = prepare_data(df)
    if "profit" not in prepared.columns:
        raise ValueError("Profit column is required for model training.")

    X = prepared.drop(columns=["date", "profit"], errors="ignore")
    y = prepared["profit"]

    if len(X) < 10:
        raise ValueError("At least 10 valid rows are required for model training.")

    test_size = max(2, int(round(len(X) * 0.20)))
    test_size = min(test_size, len(X) - 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURE_PATH)
    return model, X.columns.tolist(), mae, rmse, r2


def load_or_train_model(df, force=False):
    if not force and os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
        try:
            saved_model = joblib.load(MODEL_PATH)
            saved_features = joblib.load(FEATURE_PATH)
            current_features = prepare_data(df).drop(columns=["date", "profit"], errors="ignore").columns.tolist()
            if list(saved_features) == list(current_features):
                return saved_model, saved_features, None, None, None
        except Exception:
            pass
    return train_model(df)


def monthly_actual(df, year):
    """Return monthly totals in guaranteed calendar order Jan -> Dec."""
    data = df[df["date"].dt.year == year].copy()
    out = data.groupby(data["date"].dt.month).agg(
        Units=("units_sold", "sum"),
        Revenue=("revenue", "sum"),
        Profit=("profit", "sum"),
    ).reindex(MONTHS, fill_value=0).reset_index().rename(columns={"date": "MonthNo"})
    out["Month"] = MONTH_NAMES
    out["MonthNo"] = MONTHS
    return out[["MonthNo", "Month", "Units", "Revenue", "Profit"]]


def monthly_chart(data, y, title, y_label):
    """Calendar-ordered monthly line chart; avoids alphabetical month sorting."""
    chart = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("MonthNo:O", sort=MONTHS, axis=alt.Axis(labelExpr="['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datum.value-1]", title="Month")),
        y=alt.Y(f"{y}:Q", title=y_label),
        tooltip=[
            alt.Tooltip("Month:N", title="Month"),
            alt.Tooltip(f"{y}:Q", title=y_label, format=",.0f"),
        ],
    ).properties(title=title, height=420)
    st.altair_chart(chart, use_container_width=True)


def product_actual(df, year):
    data = df[df["date"].dt.year == year].copy()
    if data.empty:
        return pd.DataFrame()
    return (
        data.groupby(["brand", "product", "category"])
        .agg(Units=("units_sold", "sum"), Revenue=("revenue", "sum"), Profit=("profit", "sum"))
        .reset_index()
    )


def brand_actual(df, year):
    """Aggregate actual sales/profit by brand for a given year."""
    data = df[df["date"].dt.year == year].copy()
    if data.empty:
        return pd.DataFrame()
    return (
        data.groupby("brand")
        .agg(Units=("units_sold", "sum"), Revenue=("revenue", "sum"), Profit=("profit", "sum"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )


def product_sales_actual(df, year):
    """Aggregate product sales for a year so product names are visible."""
    data = df[df["date"].dt.year == year].copy()
    if data.empty:
        return pd.DataFrame()

    data["selling_price"] = pd.to_numeric(data["selling_price"], errors="coerce")
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
        out["Revenue"] != 0, out["Profit"] / out["Revenue"] * 100, 0
    )
    return out


def monthly_product_sales(df, year, product=None):
    """Return Jan-Dec product sales with month names in calendar order."""
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
    out["MonthNo"] = MONTHS
    out["Month"] = MONTH_NAMES
    return out[["MonthNo", "Month", "Units_Sold", "Revenue", "Profit"]]


def build_2027_product_forecast(df, online_scores=None):
    """Forecast each product using 2025->2026 actual growth, then project one more year."""
    p25 = product_actual(df, 2025)
    p26 = product_actual(df, 2026)
    if p26.empty:
        return pd.DataFrame()

    keys = ["brand", "product", "category"]
    if p25.empty:
        merged = p26.rename(columns={
            "Units": "2026 Units",
            "Revenue": "2026 Revenue",
            "Profit": "2026 Profit",
        }).copy()
        merged["2025 Units"] = np.nan
        merged["2025 Revenue"] = np.nan
        merged["2025 Profit"] = np.nan
    else:
        left = p26.rename(columns={"Units": "2026 Units", "Revenue": "2026 Revenue", "Profit": "2026 Profit"})
        right = p25.rename(columns={"Units": "2025 Units", "Revenue": "2025 Revenue", "Profit": "2025 Profit"})
        merged = left.merge(right, on=keys, how="left")

    # Keep actual 2026 values as the base.
    if "2026 Units" not in merged:
        merged["2026 Units"] = merged["Units"]
        merged["2026 Revenue"] = merged["Revenue"]
        merged["2026 Profit"] = merged["Profit"]

    # Product-level growth from 2025 to 2026.
    for metric in ["Units", "Revenue", "Profit"]:
        merged[f"{metric} Growth 25-26 (%)"] = np.where(
            merged[f"2025 {metric}"].fillna(0) != 0,
            (merged[f"2026 {metric}"] - merged[f"2025 {metric}"]) /
            merged[f"2025 {metric}"].abs() * 100,
            0,
        )

    # Clip extreme growth so one unusual record cannot create absurd 2027 values.
    revenue_growth = merged["Revenue Growth 25-26 (%)"].replace([np.inf, -np.inf], np.nan).fillna(0)
    profit_growth = merged["Profit Growth 25-26 (%)"].replace([np.inf, -np.inf], np.nan).fillna(0)
    units_growth = merged["Units Growth 25-26 (%)"].replace([np.inf, -np.inf], np.nan).fillna(0)

    merged["Forecast Revenue Growth 2027 (%)"] = revenue_growth.clip(-25, 35)
    merged["Forecast Profit Growth 2027 (%)"] = profit_growth.clip(-35, 40)
    merged["Forecast Units Growth 2027 (%)"] = units_growth.clip(-25, 35)

    merged["Predicted Units 2027"] = merged["2026 Units"] * (1 + merged["Forecast Units Growth 2027 (%)"] / 100)
    merged["Predicted Revenue 2027"] = merged["2026 Revenue"] * (1 + merged["Forecast Revenue Growth 2027 (%)"] / 100)
    merged["Predicted Profit 2027"] = merged["2026 Profit"] * (1 + merged["Forecast Profit Growth 2027 (%)"] / 100)

    merged["Predicted Margin 2027 (%)"] = np.where(
        merged["Predicted Revenue 2027"] != 0,
        merged["Predicted Profit 2027"] / merged["Predicted Revenue 2027"] * 100,
        0,
    )
    merged["Revenue Growth 2026→2027 (%)"] = np.where(
        merged["2026 Revenue"] != 0,
        (merged["Predicted Revenue 2027"] - merged["2026 Revenue"]) /
        merged["2026 Revenue"].abs() * 100,
        0,
    )

    # Online market signal is a supporting factor only; historical financial
    # performance remains the main driver of the recommendation.
    if online_scores is None:
        merged["Online Market Score"] = 50.0
    else:
        merged["Online Market Score"] = merged["product"].map(online_scores).fillna(50.0)

    # Convert online signal into a small growth adjustment for 2027.
    # Score 50 = neutral; positive/negative web signal changes growth modestly.
    online_growth_adjustment = (merged["Online Market Score"] - 50.0) * 0.10
    merged["Web Growth Adjustment 2027 (%)"] = online_growth_adjustment.clip(-5, 5)

    merged["Predicted Revenue 2027"] = (
        merged["2026 Revenue"] *
        (1 + (merged["Forecast Revenue Growth 2027 (%)"] +
              merged["Web Growth Adjustment 2027 (%)"]) / 100)
    )
    merged["Predicted Profit 2027"] = (
        merged["2026 Profit"] *
        (1 + (merged["Forecast Profit Growth 2027 (%)"] +
              merged["Web Growth Adjustment 2027 (%)"]) / 100)
    )
    merged["Predicted Units 2027"] = (
        merged["2026 Units"] *
        (1 + (merged["Forecast Units Growth 2027 (%)"] +
              merged["Web Growth Adjustment 2027 (%)"]) / 100)
    )

    merged["Predicted Margin 2027 (%)"] = np.where(
        merged["Predicted Revenue 2027"] != 0,
        merged["Predicted Profit 2027"] / merged["Predicted Revenue 2027"] * 100,
        0,
    )
    merged["Revenue Growth 2026→2027 (%)"] = np.where(
        merged["2026 Revenue"] != 0,
        (merged["Predicted Revenue 2027"] - merged["2026 Revenue"]) /
        merged["2026 Revenue"].abs() * 100,
        0,
    )

    # Recommendation score: finance 70%, growth 15%, online market signal 15%.
    profit_rank = merged["Predicted Profit 2027"].rank(pct=True)
    margin_rank = merged["Predicted Margin 2027 (%)"].rank(pct=True)
    growth_rank = merged["Revenue Growth 2026→2027 (%)"].rank(pct=True)
    merged["Financial Score"] = (
        0.65 * profit_rank + 0.20 * margin_rank + 0.15 * growth_rank
    ) * 100
    merged["Recommendation Score"] = (
        merged["Financial Score"] * 0.85 +
        merged["Online Market Score"] * 0.15
    )

    merged["Recommendation"] = np.select(
        [
            (merged["Predicted Profit 2027"] <= 0),
            (merged["Recommendation Score"] >= 70),
            (merged["Recommendation Score"] >= 45),
        ],
        ["🔴 Avoid / Review", "🟢 HIGH — Focus More", "🟡 MEDIUM — Monitor"],
        default="🟠 LOW — Review",
    )

    return merged.sort_values("Recommendation Score", ascending=False).reset_index(drop=True)


# ============================================================
# LIVE ONLINE MARKET INTELLIGENCE
# ============================================================

POSITIVE_WORDS = {
    "growth", "growing", "strong", "surge", "surging", "demand",
    "popular", "sales", "launch", "launched", "success", "record",
    "upgrade", "premium", "trend", "trending", "increase", "increased",
    "rising", "rise", "best", "leader", "leadership", "innovation",
    "expansion", "opportunity", "positive", "boost"
}
NEGATIVE_WORDS = {
    "decline", "declining", "drop", "dropped", "fall", "falling", "weak",
    "slow", "slowing", "loss", "losses", "recall", "issue", "issues",
    "problem", "problems", "lawsuit", "ban", "banned", "delay", "delays",
    "negative", "pressure", "competition", "competitive", "down", "risk"
}


def _sentiment_score(text):
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    return pos, neg


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_news(query, max_items=6):
    """Fetch public Google News RSS headlines for a market/product query."""
    url = (
        "https://news.google.com/rss/search?q=" + quote_plus(query) +
        "&hl=en-IN&gl=IN&ceid=IN:en"
    )
    try:
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception:
        return []

    items = []
    for item in root.findall(".//item")[:max_items]:
        title = item.findtext("title", default="").strip()
        pub_date = item.findtext("pubDate", default="").strip()
        link = item.findtext("link", default="").strip()
        source = item.findtext("source", default="").strip()
        items.append({
            "title": title,
            "date": pub_date,
            "link": link,
            "source": source,
        })
    return items


def build_online_market_signals(product_rows, max_products=8):
    """Create a small web/news signal for the highest-revenue products."""
    rows = product_rows.copy()
    if rows.empty:
        return {}, pd.DataFrame(), []

    top_products = (
        rows.sort_values("2026 Revenue", ascending=False)
        .drop_duplicates("product")
        .head(max_products)
    )

    scores = {}
    summary = []
    headlines = []

    for _, row in top_products.iterrows():
        product = str(row["product"])
        brand = str(row["brand"])
        query = f'"{product}" {brand} market sales demand'
        news = fetch_market_news(query, max_items=6)

        pos = neg = 0
        for article in news:
            p, n = _sentiment_score(article["title"])
            pos += p
            neg += n
            headlines.append({
                "Product": product,
                "Brand": brand,
                "Headline": article["title"],
                "Source": article["source"] or "Google News",
                "Date": article["date"],
                "Link": article["link"],
            })

        # Neutral baseline + headline tone + article volume.
        raw = 50 + (pos - neg) * 6 + min(len(news), 6) * 1.5
        score = float(np.clip(raw, 0, 100))
        scores[product] = score

        if score >= 65:
            signal = "🟢 Positive"
        elif score <= 35:
            signal = "🔴 Negative"
        else:
            signal = "🟡 Neutral / Mixed"

        summary.append({
            "Brand": brand,
            "Product": product,
            "Articles Found": len(news),
            "Online Market Score": round(score, 1),
            "Signal": signal,
        })

    return scores, pd.DataFrame(summary), headlines


# ============================================================
# LOAD DATA
# ============================================================
st.title("📊 AI Market Profit Advisor")
st.subheader("AI-Powered Profit/Loss Prediction, Market Analysis & 2027 Business Recommendation")
st.divider()

if not os.path.exists(DATA_PATH):
    st.error(f"❌ {DATA_PATH} not found. Put your CSV inside the data folder.")
    st.stop()

try:
    df = validate_and_clean(pd.read_csv(DATA_PATH))
    domain_key = infer_domain_label(df)
    domain_emoji, domain_product_label = domain_title(domain_key)
except Exception as e:
    st.error(f"❌ Dataset error: {e}")
    st.stop()

# ============================================================
# CSV UPLOAD — STRICT FIXED SCHEMA
# ============================================================
st.header("📂 Dataset Management")
st.caption("Only CSV files with the exact fixed 16-column format are accepted.")

with st.expander("📋 Required CSV Format", expanded=False):
    st.code(",".join(REQUIRED_COLUMNS))
    st.info("Wrong/missing/extra columns or invalid date/numeric values will be rejected. The file will NOT be merged or used for AI analysis.")


def show_invalid_csv_popup(title, message, missing=None, extra=None):
    @st.dialog("🚨 INVALID CSV FILE")
    def _popup():
        st.error("❌ This CSV cannot be analyzed.")
        st.write(message)
        if missing:
            st.warning("Missing Columns")
            st.code("\n".join(missing))
        if extra:
            st.warning("Extra Columns")
            st.code("\n".join(extra))
        st.info("📌 Upload a CSV using the exact fixed 16-column format shown above.")
    _popup()


uploaded_file = st.file_uploader("Upload CSV for Analysis", type=["csv"], key="csv_upload")
st.caption("✅ A valid upload REPLACES the current dataset. It is not merged with the previous dataset.")

# Streamlit reruns the whole script after UI interactions. The uploaded file can
# therefore remain selected across reruns. Keep a content hash so the same CSV
# is processed/trained only once instead of creating an endless training loop.
if "processed_upload_hash" not in st.session_state:
    st.session_state["processed_upload_hash"] = None

if uploaded_file is not None:
    try:
        upload_bytes = uploaded_file.getvalue()
        upload_hash = hashlib.sha256(upload_bytes).hexdigest()

        if st.session_state["processed_upload_hash"] != upload_hash:
            raw_upload = pd.read_csv(pd.io.common.BytesIO(upload_bytes))
            upload_columns = list(raw_upload.columns)
            missing_upload = [c for c in REQUIRED_COLUMNS if c not in upload_columns]
            extra_upload = [c for c in upload_columns if c not in REQUIRED_COLUMNS]

            # Exact schema: names + order must match.
            if missing_upload or extra_upload or upload_columns != REQUIRED_COLUMNS:
                order_issue = not missing_upload and not extra_upload and upload_columns != REQUIRED_COLUMNS
                msg = "The uploaded file does not match the required fixed 16-column schema."
                if order_issue:
                    msg += " The column names are correct, but their order is incorrect."
                show_invalid_csv_popup(
                    "🚨 INVALID CSV FILE",
                    msg,
                    missing=missing_upload,
                    extra=extra_upload,
                )
                st.stop()

            # Strict validation: do not silently repair/drop bad uploaded rows.
            new_df = validate_and_clean(raw_upload, strict_schema=True)

            if new_df.empty:
                show_invalid_csv_popup(
                    "🚨 INVALID CSV FILE",
                    "The uploaded CSV contains no valid records after validation."
                )
                st.stop()

            # ----------------------------------------------------
            # REPLACE DATASET — DO NOT MERGE WITH OLD DATA
            # ----------------------------------------------------
            new_df = new_df.drop_duplicates().reset_index(drop=True)
            new_df.to_csv(DATA_PATH, index=False)
            df = new_df

            # Automatically identify the uploaded business/product domain.
            domain_key = infer_domain_label(df)
            domain_emoji, domain_product_label = domain_title(domain_key)

            # Remove model generated from a different dataset/product set.
            for model_file in (MODEL_PATH, FEATURE_PATH):
                if os.path.exists(model_file):
                    try:
                        os.remove(model_file)
                    except OSError:
                        pass

            st.success("✅ CSV VALIDATED SUCCESSFULLY")
            st.success(
                f"🔄 Current dataset replaced successfully with {len(df):,} valid records."
            )
            st.info(
                "🤖 Only the newly uploaded CSV will be used for business analysis, "
                "AI training, forecasting and recommendations."
            )

            with st.spinner("🤖 Training AI model on the new dataset..."):
                _, _, mae, rmse, r2 = train_model(df)

            # Mark this exact file as processed BEFORE the next Streamlit rerun.
            # This prevents the uploader from retraining the same CSV repeatedly.
            st.session_state["processed_upload_hash"] = upload_hash

            st.success("✅ AI model trained successfully on the uploaded dataset.")
            a, b, c = st.columns(3)
            a.metric("MAE", money(mae))
            b.metric("RMSE", money(rmse))
            c.metric("R² Score", f"{r2:.3f}")

        else:
            st.success(
                "✅ Uploaded CSV is already loaded. The AI model will NOT retrain on every refresh."
            )

    except pd.errors.EmptyDataError:
        show_invalid_csv_popup("🚨 INVALID CSV FILE", "The uploaded file is empty.")
        st.stop()
    except Exception as e:
        # Important: invalid uploads never reach the save/replace step.
        show_invalid_csv_popup("🚨 INVALID CSV FILE", str(e))
        st.stop()

# ============================================================
# DATASET SUMMARY
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
    model, features, mae, rmse, r2 = load_or_train_model(df)
except Exception as e:
    st.error(f"❌ Model training error: {e}")
    st.stop()

# ============================================================
# 2025 ACTUAL
# ============================================================
st.divider()
st.header("📌 2025 Actual Business Analysis")
df_2025 = df[df["date"].dt.year == 2025].copy()

if df_2025.empty:
    st.warning("No 2025 data found.")
else:
    rev25 = df_2025["revenue"].sum()
    profit25 = df_2025["profit"].sum()
    units25 = df_2025["units_sold"].sum()
    margin25 = profit25 / rev25 * 100 if rev25 else 0

    a, b, c, d = st.columns(4)
    a.metric("2025 Revenue", money(rev25))
    b.metric("2025 Profit", money(profit25))
    c.metric("2025 Units", f"{units25:,.0f}")
    d.metric("2025 Margin", f"{margin25:.2f}%")

    m25 = monthly_actual(df, 2025)
    monthly_chart(m25, "Profit", "📈 2025 Monthly Profit", "Profit (₹)")

# 2025 BRAND-WISE SALES
st.subheader("🏷️ 2025 Brand-wise Actual Sales")
b25 = brand_actual(df, 2025)
if not b25.empty:
    st.dataframe(
        b25.style.format({
            "Units": "{:,.0f}",
            "Revenue": "₹ {:,.0f}",
            "Profit": "₹ {:,.0f}",
        }),
        use_container_width=True, hide_index=True
    )
    st.bar_chart(b25.set_index("brand")["Revenue"], y_label="Revenue (₹)")

# ============================================================
# FOOD / PRODUCT-WISE ACTUAL SALES — 2025
# ============================================================
st.divider()
st.header(f"{domain_emoji} 2025 {domain_product_label} Sales — What Was Actually Sold?")

if not df_2025.empty:
    product25 = product_sales_actual(df, 2025)
    if not product25.empty:
        st.success(
            f"✅ 2025 me {product25['product'].nunique()} {domain_product_label.lower()}s ki actual sales record hui."
        )

        display25 = product25.rename(columns={
            "product": domain_product_label,
            "category": "Category",
            "Brands": "Brands Selling",
            "Units_Sold": "Units Sold",
            "Revenue": "Revenue",
            "Profit": "Profit",
            "Avg_Selling_Price": "Avg Selling Price",
            "Profit_Margin": "Profit Margin %",
        })
        st.dataframe(
            display25.style.format({
                "Units Sold": "{:,.0f}",
                "Revenue": "₹ {:,.0f}",
                "Profit": "₹ {:,.0f}",
                "Avg Selling Price": "₹ {:,.2f}",
                "Profit Margin %": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(f"📦 2025 {domain_product_label}-wise Units Sold")
        st.bar_chart(
            product25.set_index("product")["Units_Sold"],
            y_label="Units Sold",
        )

        selected_product_25 = st.selectbox(
            f"{domain_emoji} Select {domain_product_label.lower()} to see month-wise sales (2025)",
            sorted(product25["product"].tolist()),
            key="selected_product_2025",
        )
        mp25 = monthly_product_sales(df, 2025, selected_product_25)
        st.subheader(f"📅 {selected_product_25} — 2025 Month-wise Sales")
        st.dataframe(
            mp25[["Month", "Units_Sold", "Revenue", "Profit"]].rename(columns={
                "Units_Sold": "Units Sold",
            }).style.format({
                "Units Sold": "{:,.0f}",
                "Revenue": "₹ {:,.0f}",
                "Profit": "₹ {:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# 2026 ACTUAL — NOT EXPECTED
# ============================================================
st.divider()
st.header("📌 2026 Actual Business Analysis")
df_2026 = df[df["date"].dt.year == 2026].copy()

if df_2026.empty:
    st.warning("No 2026 actual data found in the dataset. 2027 prediction cannot use 2026 actuals until 2026 data is uploaded.")
else:
    rev26 = df_2026["revenue"].sum()
    profit26 = df_2026["profit"].sum()
    units26 = df_2026["units_sold"].sum()
    margin26 = profit26 / rev26 * 100 if rev26 else 0
    rev_growth_25_26 = safe_growth(rev26, rev25 if not df_2025.empty else 0)
    profit_growth_25_26 = safe_growth(profit26, profit25 if not df_2025.empty else 0)
    units_growth_25_26 = safe_growth(units26, units25 if not df_2025.empty else 0)

    a, b, c, d = st.columns(4)
    a.metric("2026 Actual Revenue", money(rev26), pct(rev_growth_25_26) + " vs 2025")
    b.metric("2026 Actual Profit", money(profit26), pct(profit_growth_25_26) + " vs 2025")
    c.metric("2026 Actual Units", f"{units26:,.0f}", pct(units_growth_25_26) + " vs 2025")
    d.metric("2026 Actual Margin", f"{margin26:.2f}%")

    m26 = monthly_actual(df, 2026)
    monthly_chart(m26, "Profit", "📈 2026 Monthly Actual Profit", "Profit (₹)")
    monthly_chart(m26, "Revenue", "💰 2026 Monthly Actual Revenue", "Revenue (₹)")

    # BRAND-WISE ACTUAL SALES
    st.subheader("🏷️ 2026 Brand-wise Actual Sales")
    b26 = brand_actual(df, 2026)
    if not b26.empty:
        st.dataframe(
            b26.style.format({
                "Units": "{:,.0f}",
                "Revenue": "₹ {:,.0f}",
                "Profit": "₹ {:,.0f}",
            }),
            use_container_width=True, hide_index=True
        )
        st.bar_chart(b26.set_index("brand")["Revenue"], y_label="Revenue (₹)")

    st.subheader("🏷️ 2026 Brand-wise Profit")
    st.bar_chart(b26.set_index("brand")["Profit"], y_label="Profit (₹)")

# ============================================================
# FOOD / PRODUCT-WISE ACTUAL SALES — 2026
# ============================================================
st.subheader(f"{domain_emoji} 2026 {domain_product_label} Sales — What Was Actually Sold?")

if not df_2026.empty:
    product26 = product_sales_actual(df, 2026)
    if not product26.empty:
        st.success(
            f"✅ 2026 me {product26['product'].nunique()} {domain_product_label.lower()}s ki actual sales record hui."
        )

        display26 = product26.rename(columns={
            "product": domain_product_label,
            "category": "Category",
            "Brands": "Brands Selling",
            "Units_Sold": "Units Sold",
            "Revenue": "Revenue",
            "Profit": "Profit",
            "Avg_Selling_Price": "Avg Selling Price",
            "Profit_Margin": "Profit Margin %",
        })
        st.dataframe(
            display26.style.format({
                "Units Sold": "{:,.0f}",
                "Revenue": "₹ {:,.0f}",
                "Profit": "₹ {:,.0f}",
                "Avg Selling Price": "₹ {:,.2f}",
                "Profit Margin %": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(f"📦 2026 {domain_product_label}-wise Units Sold")
        st.bar_chart(
            product26.set_index("product")["Units_Sold"],
            y_label="Units Sold",
        )

        selected_product_26 = st.selectbox(
            f"{domain_emoji} Select {domain_product_label.lower()} to see month-wise sales (2026)",
            sorted(product26["product"].tolist()),
            key="selected_product_2026",
        )
        mp26 = monthly_product_sales(df, 2026, selected_product_26)
        st.subheader(f"📅 {selected_product_26} — 2026 Month-wise Sales")
        st.dataframe(
            mp26[["Month", "Units_Sold", "Revenue", "Profit"]].rename(columns={
                "Units_Sold": "Units Sold",
            }).style.format({
                "Units Sold": "{:,.0f}",
                "Revenue": "₹ {:,.0f}",
                "Profit": "₹ {:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# 2025 vs 2026
# ============================================================
st.divider()
st.header("📊 2025 vs 2026 Growth Analysis")
if not df_2025.empty and not df_2026.empty:
    comparison = pd.DataFrame({
        "2025 Actual": [rev25, profit25, units25],
        "2026 Actual": [rev26, profit26, units26],
    }, index=["Revenue", "Profit", "Units"])
    st.dataframe(comparison.style.format({
        "2025 Actual": "₹ {:,.0f}",
        "2026 Actual": "₹ {:,.0f}",
    }), use_container_width=True)

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
st.caption("Automatic business summary generated from the uploaded dataset. The wording adapts to the detected product/business type.")

try:
    summary_lines = []
    action_lines = []

    latest_year = int(df["date"].dt.year.max()) if not df.empty else None
    latest_df = df[df["date"].dt.year == latest_year].copy() if latest_year else pd.DataFrame()

    if not latest_df.empty:
        latest_domain_name = domain_product_label
        latest_revenue = latest_df["revenue"].sum()
        latest_profit = latest_df["profit"].sum()
        latest_units = latest_df["units_sold"].sum()
        latest_margin = latest_profit / latest_revenue * 100 if latest_revenue else 0

        prod_latest = latest_df.groupby("product").agg(
            Units=("units_sold", "sum"),
            Revenue=("revenue", "sum"),
            Profit=("profit", "sum")
        )
        prod_latest["Margin"] = np.where(
            prod_latest["Revenue"] != 0,
            prod_latest["Profit"] / prod_latest["Revenue"] * 100,
            0,
        )

        brand_latest = latest_df.groupby("brand").agg(
            Units=("units_sold", "sum"),
            Revenue=("revenue", "sum"),
            Profit=("profit", "sum")
        )

        top_product = prod_latest["Profit"].idxmax()
        top_product_profit = prod_latest.loc[top_product, "Profit"]
        top_product_margin = prod_latest.loc[top_product, "Margin"]
        top_brand = brand_latest["Profit"].idxmax()
        top_brand_profit = brand_latest.loc[top_brand, "Profit"]

        summary_lines.append(
            f"📌 In {latest_year}, total {latest_domain_name.lower()} sales were "
            f"{latest_units:,.0f} units with revenue of {money(latest_revenue)} "
            f"and profit of {money(latest_profit)} (margin {latest_margin:.2f}%)."
        )
        summary_lines.append(
            f"🏆 Best-performing {latest_domain_name.lower()}: **{top_product}** "
            f"with profit {money(top_product_profit)} and margin {top_product_margin:.2f}%."
        )
        summary_lines.append(
            f"🏢 Best brand: **{top_brand}** with profit {money(top_brand_profit)}."
        )

        # Identify a low-margin product as a potential problem area.
        positive_revenue = prod_latest[prod_latest["Revenue"] > 0]
        if not positive_revenue.empty:
            risk_product = positive_revenue["Margin"].idxmin()
            risk_margin = positive_revenue.loc[risk_product, "Margin"]
            if risk_margin < 10:
                action_lines.append(
                    f"⚠️ Review **{risk_product}**: its {latest_year} profit margin is only {risk_margin:.2f}%."
                )

        if latest_margin >= 20:
            action_lines.append("✅ Overall profitability is healthy; protect margin while scaling the strongest products.")
        elif latest_margin >= 10:
            action_lines.append("🟡 Profitability is moderate; optimize pricing, discount and operating costs before aggressive expansion.")
        else:
            action_lines.append("🔴 Overall margin is low; prioritize cost control and pricing review before increasing volume.")

    if not df_2025.empty and not df_2026.empty:
        # 2025→2026 actual change is the primary trend signal.
        summary_lines.append(
            f"📈 2025→2026 actual revenue growth is **{pct(rev_growth_25_26)}**, "
            f"while profit growth is **{pct(profit_growth_25_26)}**."
        )

        p25 = df_2025.groupby("product").agg(
            Units=("units_sold", "sum"), Revenue=("revenue", "sum"), Profit=("profit", "sum")
        )
        p26 = df_2026.groupby("product").agg(
            Units=("units_sold", "sum"), Revenue=("revenue", "sum"), Profit=("profit", "sum")
        )
        merged_growth = p25.join(p26, lsuffix="_25", rsuffix="_26", how="inner")
        if not merged_growth.empty:
            merged_growth["Profit Growth %"] = merged_growth.apply(
                lambda r: safe_growth(r["Profit_26"], r["Profit_25"]), axis=1
            )
            fastest = merged_growth["Profit Growth %"].idxmax()
            fastest_growth = merged_growth.loc[fastest, "Profit Growth %"]
            summary_lines.append(
                f"🚀 Fastest profit-growing {domain_product_label.lower()}: **{fastest}** "
                f"at {pct(fastest_growth)} from 2025 to 2026."
            )
            if fastest_growth > 0:
                action_lines.append(
                    f"🎯 Increase focus on **{fastest}** because its historical profit trend is positive."
                )

    if action_lines:
        st.success("\n\n".join(summary_lines))
        st.info("\n\n".join(action_lines))
    elif summary_lines:
        st.success("\n\n".join(summary_lines))
    else:
        st.info("Upload a dataset containing usable product, sales and profit records to generate the executive summary.")

except Exception as e:
    st.warning(f"⚠️ Executive summary could not be generated: {e}")

# ============================================================
# 2027 FORECAST
# ============================================================
st.divider()
st.header("🔮 2027 Sales & Profit Prediction")
st.info("2027 is predicted from the actual 2025→2026 product and monthly trends. It is a forecast, not a guaranteed result.")

if df_2026.empty:
    st.warning("2027 prediction requires 2026 actual data.")
else:
    m26 = monthly_actual(df, 2026)
    m25 = monthly_actual(df, 2025) if not df_2025.empty else None

    forecast_rows = []
    for month_name in MONTH_NAMES:
        base26 = m26.loc[m26["Month"] == month_name].iloc[0]
        if m25 is not None and not m25.empty:
            base25 = m25.loc[m25["Month"] == month_name].iloc[0]
        else:
            base25 = pd.Series({"Units": 0, "Revenue": 0, "Profit": 0})

        unit_growth = safe_growth(base26["Units"], base25["Units"]) if base25["Units"] else 0
        revenue_growth = safe_growth(base26["Revenue"], base25["Revenue"]) if base25["Revenue"] else 0
        profit_growth = safe_growth(base26["Profit"], base25["Profit"]) if base25["Profit"] else 0

        # Conservative clipping avoids unrealistic one-month spikes.
        unit_growth = float(np.clip(unit_growth, -25, 35))
        revenue_growth = float(np.clip(revenue_growth, -25, 35))
        profit_growth = float(np.clip(profit_growth, -35, 40))

        forecast_rows.append({
            "Month": month_name + " 2027",
            "Expected Units": round(base26["Units"] * (1 + unit_growth / 100)),
            "Expected Revenue": base26["Revenue"] * (1 + revenue_growth / 100),
            "Expected Profit": base26["Profit"] * (1 + profit_growth / 100),
            "Revenue Growth vs 2026 (%)": revenue_growth,
            "Profit Growth vs 2026 (%)": profit_growth,
        })

    forecast_2027 = pd.DataFrame(forecast_rows)
    pred_units27 = forecast_2027["Expected Units"].sum()
    pred_rev27 = forecast_2027["Expected Revenue"].sum()
    pred_profit27 = forecast_2027["Expected Profit"].sum()
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
        forecast_2027.style.format({
            "Expected Units": "{:,.0f}",
            "Expected Revenue": "₹ {:,.0f}",
            "Expected Profit": "₹ {:,.0f}",
            "Revenue Growth vs 2026 (%)": "{:+.2f}%",
            "Profit Growth vs 2026 (%)": "{:+.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    chart_27 = forecast_2027[["Month", "Expected Profit"]]
    monthly_chart(forecast_2027.assign(MonthNo=MONTHS), "Expected Profit", "📈 2027 Predicted Profit Trend", "Profit (₹)")

    # 2025/2026/2027 total comparison
    if not df_2025.empty:
        growth_table = pd.DataFrame({
            "2025 Actual": [rev25, profit25],
            "2026 Actual": [rev26, profit26],
            "2027 Predicted": [pred_rev27, pred_profit27],
        }, index=["Revenue", "Profit"])
        st.subheader("📊 Revenue & Profit: 2025 → 2026 → 2027")
        st.dataframe(
            growth_table.style.format("₹ {:,.0f}"),
            use_container_width=True,
        )

# ============================================================
# ONLINE MARKET ANALYSIS FOR 2027
# ============================================================
st.divider()
st.header("🌐 Online Customer Demand & Market Intelligence")
st.caption(
    "Public web/news signals are used as a customer-demand proxy and supporting factor for 2027 recommendations. "
    "They are not a direct measure of product sales or guaranteed future demand."
)

online_scores = {}
market_signal_table = pd.DataFrame()
market_headlines = []

if not df_2026.empty:
    base_products_2027 = build_2027_product_forecast(df)
    if not base_products_2027.empty:
        refresh = st.button("🔄 Refresh Online Market Analysis", key="refresh_market")
        if refresh:
            st.cache_data.clear()
            st.rerun()

        with st.spinner("🌐 Analyzing current online market/news signals..."):
            online_scores, market_signal_table, market_headlines = build_online_market_signals(
                base_products_2027,
                max_products=8,
            )

        if market_signal_table.empty:
            st.warning(
                "⚠️ Live market sources could not be reached right now. "
                "Recommendations will continue using historical financial data only."
            )
        else:
            avg_online = float(market_signal_table["Online Market Score"].mean())
            st.metric("Overall Online Market Signal", f"{avg_online:.1f}/100")
            st.dataframe(
                market_signal_table.sort_values("Online Market Score", ascending=False),
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
                    st.info("No recent headlines were returned for the selected products.")

# ============================================================
# 2027 PRODUCT RECOMMENDATION
# ============================================================
st.divider()
st.header("🏆 2027 Best Product Recommendation")

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
        st.error(f"🔴 No product has positive predicted 2027 profit. Review {best['product']} before increasing production.")

    display = product_2027[[
        "brand", "product", "category",
        "2026 Units", "2026 Revenue", "2026 Profit",
        "Predicted Units 2027", "Predicted Revenue 2027", "Predicted Profit 2027",
        "Predicted Margin 2027 (%)", "Revenue Growth 2026→2027 (%)",
        "Online Market Score", "Web Growth Adjustment 2027 (%)",
        "Recommendation Score", "Recommendation"
    ]].copy()

    st.subheader("📋 2027 Product Ranking")
    st.dataframe(
        display.style.format({
            "2026 Units": "{:,.0f}",
            "2026 Revenue": "₹ {:,.0f}",
            "2026 Profit": "₹ {:,.0f}",
            "Predicted Units 2027": "{:,.0f}",
            "Predicted Revenue 2027": "₹ {:,.0f}",
            "Predicted Profit 2027": "₹ {:,.0f}",
            "Predicted Margin 2027 (%)": "{:.2f}%",
            "Revenue Growth 2026→2027 (%)": "{:+.2f}%",
            "Online Market Score": "{:.1f}",
            "Web Growth Adjustment 2027 (%)": "{:+.2f}%",
            "Recommendation Score": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🥇 Top 3 Products to Focus on in 2027")
    for i, (_, row) in enumerate(product_2027.head(3).iterrows(), start=1):
        st.markdown(
            f"**{i}. {row['product']} ({row['brand']})** — "
            f"profit {money(row['Predicted Profit 2027'])}, "
            f"margin {row['Predicted Margin 2027 (%)']:.2f}%, "
            f"growth {row['Revenue Growth 2026→2027 (%)']:+.2f}% — "
            f"{row['Recommendation']}"
        )

    # Product comparison chart
    comparison = product_2027.set_index("product")[[
        "2026 Profit", "Predicted Profit 2027"
    ]]
    st.subheader("📈 Product Profit: 2026 Actual vs 2027 Predicted")
    st.bar_chart(comparison)

# ============================================================
# INDIVIDUAL AI PROFIT PREDICTOR
# ============================================================
st.divider()
st.header("🤖 Individual Product Profit/Loss Predictor")

products = sorted(df["product"].dropna().unique().tolist())
brands = sorted(df["brand"].dropna().unique().tolist())
categories = sorted(df["category"].dropna().unique().tolist())
market_conditions = sorted(df["market_condition"].dropna().unique().tolist())
competitor_pressures = sorted(df["competitor_pressure"].dropna().unique().tolist())

if products:
    a, b, c = st.columns(3)
    with a:
        product = st.selectbox("Product", products)
        brand = st.selectbox("Brand", brands)
        category = st.selectbox("Category", categories)
    with b:
        units_sold = st.number_input("Units Sold", min_value=1, value=100, step=1)
        selling_price = st.number_input("Selling Price (₹)", min_value=1.0, value=70000.0, step=500.0)
        discount_percent = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    with c:
        marketing_cost = st.number_input("Marketing Cost (₹)", min_value=0.0, value=50000.0, step=1000.0)
        other_cost = st.number_input("Other Cost (₹)", min_value=0.0, value=20000.0, step=1000.0)
        market_growth_input = st.number_input("Market Growth (%)", min_value=-100.0, max_value=100.0, value=4.0, step=0.5)

    market_condition = st.selectbox("Market Condition", market_conditions)
    competitor_pressure = st.selectbox("Competitor Pressure", competitor_pressures)

    if st.button("🔮 Predict Profit/Loss", use_container_width=True):
        try:
            # Use the selected product's median unit cost.
            cost_map = (
                df.assign(unit_cost=np.where(df["units_sold"] > 0, df["product_cost"] / df["units_sold"], np.nan))
                .groupby("product")["unit_cost"].median().to_dict()
            )
            fallback_cost = (df["product_cost"] / df["units_sold"].replace(0, np.nan)).median()
            unit_cost = cost_map.get(product, fallback_cost)
            unit_cost = float(unit_cost if pd.notna(unit_cost) else 0)

            effective_price = selling_price * (1 - discount_percent / 100)
            revenue = units_sold * effective_price
            product_cost = units_sold * unit_cost
            total_cost = product_cost + marketing_cost + other_cost

            input_data = pd.DataFrame({
                "date": [pd.Timestamp.today()],
                "brand": [brand], "product": [product], "category": [category],
                "units_sold": [units_sold], "selling_price": [effective_price],
                "revenue": [revenue], "product_cost": [product_cost],
                "discount_percent": [discount_percent], "marketing_cost": [marketing_cost],
                "other_cost": [other_cost], "total_cost": [total_cost],
                "profit": [0], "market_growth_percent": [market_growth_input],
                "market_condition": [market_condition], "competitor_pressure": [competitor_pressure],
            })

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
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

st.divider()
st.caption("AI Market Profit Advisor | 2025 Actual + 2026 Actual + 2027 Prediction + Product Recommendation")
