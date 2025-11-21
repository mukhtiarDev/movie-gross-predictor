import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(layout="wide", page_title="Movie Gross Predictor")

# 🌟 CUSTOM CSS INJECTION TO REDUCE VERTICAL SPACING 🌟
# This targets the margins/padding above the st.header (h2) and st.subheader (h3) elements.
st.markdown("""
<style>
    /* Target h2 (st.header) and reduce its top margin */
    h2 {
        margin-top: 1rem !important;
        padding-top: 0 !important;
    }
    /* Target h3 (st.subheader) and significantly reduce its top margin */
    h3 {
        margin-top: 0.5rem !important; /* Reduced from default 1.5rem */
        padding-top: 0 !important;
    }
</style>
""", unsafe_allow_html=True)
# -------------------------------------------------------------

TARGET = "gross"
FEATURES = ['budget','votes','score','runtime','genre','rating','country']

st.title("🎬 Movie Gross Revenue Prediction (Cleaned & Reduced Version)")

# ============================================================
# 1) LOAD & SHOW RAW PROBLEMS
# ============================================================

def load_data(path):
    df = pd.read_csv(path)
    return df

df_raw = load_data("movies.csv")

st.header("1. Raw Data Quality Check")

# Missing values
st.subheader("🔍 Missing Values (Raw Data)")
st.write(df_raw.isnull().sum())

# Duplicate rows
st.subheader("🔂 Duplicate Rows (Raw Data)")
dups = df_raw[df_raw.duplicated()]
st.write(dups if not dups.empty else "No duplicate rows found.")

# Invalid date parsing
df_raw["clean_released"] = df_raw["released"].str.replace(r"\s*\(.*\)", "", regex=True)
df_raw["parsed_released"] = pd.to_datetime(df_raw["clean_released"], errors="coerce")

invalid_dates = df_raw[df_raw["parsed_released"].isna() & df_raw["clean_released"].notna()]
st.subheader("📅 Invalid Date Format (Raw Data)")
st.write(invalid_dates[["released","clean_released"]].head() if not invalid_dates.empty else "No invalid dates found.")

# ============================================================
# 2) CLEAN DATA
# ============================================================

st.header("2. Cleaning Data")

df = df_raw.copy()

# Drop unused columns
df = df.drop(columns=['name','writer','director','star','company','year'], errors='ignore')

# Numeric conversions
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['votes'] = pd.to_numeric(df['votes'], errors='coerce')

# Remove rows missing core numerics
df.dropna(subset=['budget','votes','score','runtime',TARGET], inplace=True)

# Remove zeros
df = df[(df['budget'] > 0) & (df['votes'] > 0)]

# Clean/parse dates
df["clean_released"] = df["released"].str.replace(r"\s*\(.*\)", "", regex=True)
df["parsed_released"] = pd.to_datetime(df["clean_released"], errors="coerce")

# Fill missing categoricals
for col in ['genre','rating','country']:
    df[col] = df[col].fillna("Unknown")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Scale to millions
df['budget'] /= 1_000_000
df['gross'] /= 1_000_000

# Encode categoricals
for col in FEATURES:
    if df[col].dtype == "object":
        df[col] = LabelEncoder().fit_transform(df[col])

# ============================================================
# 3) SHOW ISSUES AFTER CLEANING
# ============================================================

st.header("3. Quality Check After Cleaning")

st.subheader("Missing Values After Clean")
st.write(df.isnull().sum())

st.subheader("Duplicate Rows After Clean")
dups_after = df[df.duplicated()]
st.write(dups_after if not dups_after.empty else "None")

invalid_after = df[df["parsed_released"].isna() & df["clean_released"].notna()]
st.subheader("Invalid Dates After Clean")
st.write(invalid_after if not invalid_after.empty else "None")

# ============================================================
# 4) TRAIN MODEL
# ============================================================

st.header("4. Model Training & Evaluation")

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
n = len(y_test)
k = X_test.shape[1]
adj_r2 = 1 - (1-r2) * (n-1) / (n-k-1)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📊 Model Metrics")
st.write({
    "R-squared (R²)": round(r2,4),
    "Adjusted R²": round(adj_r2,4),
    "MSE": round(mse,4),
    "RMSE": round(rmse,4),
    "MAE (Millions)": round(mae,4)
})

# ============================================================
# 5) PLOTS
# ============================================================

st.header("5. Plots")

# Actual vs predicted
fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, alpha=0.6)
max_val = max(y_test.max(), y_pred.max())
ax1.plot([0,max_val],[0,max_val], linestyle='--')
ax1.set_xlabel("Actual")
ax1.set_ylabel("Predicted")
ax1.set_title("Actual vs Predicted")
st.pyplot(fig1)

# Residuals plot
residuals = y_test - y_pred
fig2, ax2 = plt.subplots()
ax2.scatter(y_pred, residuals, alpha=0.6)
ax2.axhline(0, linestyle='--')
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Residuals")
ax2.set_title("Residual Plot")
st.pyplot(fig2)