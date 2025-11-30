import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------- CONFIG ---------------- #
st.set_page_config(layout="wide", page_title="Movie Gross Predictor")
TARGET = "gross"
FEATURES = ['budget','votes','score','runtime','genre','rating','country','director','writer','star','year']
encoder_maps = {}

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

df_raw = load_data()
st.title("🎬 Movie Gross Revenue Prediction")

# ---------------- DATA QUALITY CHECK ---------------- #
st.header("1️⃣ Data Quality Check (Before Cleaning)")
st.subheader("Missing Values")
st.write(df_raw.isnull().sum())

st.subheader("Duplicate Rows")
st.write(df_raw[df_raw.duplicated()])

# ---------------- CLEAN DATA ---------------- #
st.header("2️⃣ Clean Data")
def clean_data(df):
    df = df.copy()
    # Drop unused columns
    drop_cols = ['name','company']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    # Convert numeric columns
    for col in ["budget","votes","gross","runtime","score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=["budget","votes","gross","runtime","score"], inplace=True)
    df = df[(df["budget"]>0) & (df["votes"]>0)]
    # Fill categorical features
    for col in ["genre","rating","country","director","writer","star"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    # Encode categoricals
    for col in FEATURES:
        if col in df.columns and df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoder_maps[col] = {label: idx for idx, label in enumerate(le.classes_)}
            encoder_maps[f"{col}_options"] = le.classes_.tolist()
    # Scale budget & gross
    df["budget"] /= 1_000_000
    df["gross"]  /= 1_000_000
    # Add year if missing
    if "year" not in df.columns:
        df["year"] = 2025
    return df

df = clean_data(df_raw)
st.subheader("Post-cleaning Missing Values")
st.write(df.isnull().sum())
st.subheader("Duplicate Rows After Cleaning")
st.write(df[df.duplicated()])

# ---------------- MODEL TRAINING ---------------- #
st.header("3️⃣ Linear Regression Model")
test_size = st.slider("Test Size", 0.05, 0.5, 0.2, 0.05)

X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("Model Metrics")
c1,c2,c3,c4 = st.columns(4)
c1.metric("R²", f"{r2:.4f}")
c2.metric("MSE", f"{mse:.4f}")
c3.metric("RMSE", f"{rmse:.4f}")
c4.metric("MAE", f"{mae:.4f}")

# ---------------- PLOTS ---------------- #
st.header("4️⃣ Plots")
fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, alpha=0.7)
m = max(y_test.max(), y_pred.max())
ax1.plot([0,m],[0,m],'r--')
ax1.set_xlabel("Actual")
ax1.set_ylabel("Predicted")
ax1.set_title("Actual vs Predicted")

fig2, ax2 = plt.subplots()
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=0.6)
ax2.axhline(0, color="black", linestyle="--")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Residuals")
ax2.set_title("Residual Plot")

st.pyplot(fig1)
st.pyplot(fig2)

# ---------------- NEW MOVIE PREDICTION ---------------- #
st.header("5️⃣ Predict New Movie Revenue")
with st.form("prediction"):
    c1,c2 = st.columns(2)
    budget = c1.number_input("Budget (Millions)", 0.1, value=30.0)
    votes  = c1.number_input("Votes", 100, value=100000)
    score  = c2.slider("IMDb Score", 1.0, 10.0, 7.0)
    runtime = c2.number_input("Runtime (min)", 40, value=110)
    
    g,r,c = st.columns(3)
    genre   = g.selectbox("Genre", encoder_maps["genre_options"])
    rating  = r.selectbox("Rating", encoder_maps["rating_options"])
    country = c.selectbox("Country", encoder_maps["country_options"])
    
    director = st.text_input("Director", "Unknown")
    writer   = st.text_input("Writer", "Unknown")
    star     = st.text_input("Star", "Unknown")
    year     = st.number_input("Year", 1900, 2030, 2025)

    submit = st.form_submit_button("Predict")
    
    if submit:
        data = {
            "budget": budget,
            "votes": votes,
            "score": score,
            "runtime": runtime,
            "genre": encoder_maps["genre"].get(genre,0),
            "rating": encoder_maps["rating"].get(rating,0),
            "country": encoder_maps["country"].get(country,0),
            "director": encoder_maps["director"].get(director,0),
            "writer": encoder_maps["writer"].get(writer,0),
            "star": encoder_maps["star"].get(star,0),
            "year": year
        }
        input_df = pd.DataFrame([data])[FEATURES]  # ensure correct order
        pred = model.predict(input_df)[0]
        if pred < 0:
            st.warning(f"Predicted Gross: ${pred:.2f}M (Check Inputs)")
        else:
            st.success(f"🎬 Predicted Gross Revenue: **${pred:.2f} Million**")
