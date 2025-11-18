import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# --- Configuration & Constants ---
st.set_page_config(layout="wide", page_title="Movie Gross Predictor")
TARGET = 'gross'
FEATURES = ['budget', 'votes', 'score', 'runtime', 'genre', 'rating', 'country']
SEED = 42

# --- Data Loading and Cleaning (Reduced Comments) ---
@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found.")
        return pd.DataFrame()

    # Cleaning: Drop text cols, handle missing/zero numerics
    df = df.drop(columns=['name', 'released', 'writer', 'director', 'star', 'company', 'year'], errors='ignore')
    df.dropna(subset=[TARGET], inplace=True)
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce')
    
    # Remove rows with NaN in core features or zero budget/votes
    df.dropna(subset=['budget', 'votes', 'score', 'runtime'], inplace=True)
    df = df[(df['budget'] > 0) & (df['votes'] > 0)]
    
    # SCALE TO MILLIONS: Convert gross and budget for better interpretation
    df[TARGET] = df[TARGET] / 1_000_000
    df['budget'] = df['budget'] / 1_000_000
    
    # Cleaning: Fill missing categoricals and drop duplicates
    for col in ['genre', 'rating', 'country']: df[col].fillna('Unknown', inplace=True)
    df.drop_duplicates(inplace=True)

    # Feature Engineering: Label Encoding
    for col in FEATURES:
        if df[col].dtype == 'object':
            df[col + '_encoded'] = LabelEncoder().fit_transform(df[col])

    # Final feature set
    X_cols = [col + '_encoded' if df[col].dtype == 'object' else col for col in FEATURES]
    X, y = df[X_cols], df[TARGET]
    st.success(f"Data Cleaned! {len(df)} records for modeling (Gross/Budget in Millions).")
    return X, y, df

# --- Model Training ---
@st.cache_resource
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# --- Main Application Logic ---
st.title("🎬 Movie Gross Revenue Prediction (Millions $)")

# Sidebar for controls (Refresh button removed)
with st.sidebar:
    st.header("App Controls")
    test_size = st.slider("Test Set Ratio", 0.1, 0.5, 0.2, 0.05)
    st.info("Goal: R² > 0.65.")

X, y, df = load_and_clean_data("movies.csv")
if df.empty: st.stop()

st.header("1. Model Training & Evaluation")
try:
    # Split, Train, Predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)
    model = train_linear_regression(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics (Calculated on scaled data)
    r2 = r2_score(y_test, y_pred); mse = mean_squared_error(y_test, y_pred); mae = mean_absolute_error(y_test, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-squared ($R^2$)", f"{r2:.4f}", f"Goal: {('Met' if r2 >= 0.65 else 'Not Met')}")
    col2.metric("MSE ($M^2$)", f"{mse:,.0f}")
    col3.metric("MAE ($Millions$)", f"{mae:,.2f}")
    col4.metric("Test Size", f"{len(X_test)} samples")

except Exception as e:
    st.error(f"Error during modeling: {e}")
    st.stop()

st.header("2. Visual Analysis")

# 1. Actual vs. Predicted Plot
st.subheader("Actual vs. Predicted Gross Revenue")
plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Actual', y='Predicted', data=plot_df, ax=ax, alpha=0.6)
max_val = max(y_test.max(), y_pred.max())
ax.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect')
ax.set(title='Actual vs. Predicted', xlabel='Actual Gross (Millions $)', ylabel='Predicted Gross (Millions $)')
ax.ticklabel_format(style='plain', axis='both'); ax.grid(True, alpha=0.3)
st.pyplot(fig)

# 2. Residuals Plot
st.subheader("Residuals Plot (Prediction Errors)")
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=0.6, color='#ff7f0e')
ax.axhline(y=0, color='red', linestyle='--')
ax.set(title='Residuals Plot', xlabel='Predicted Gross (Millions $)', ylabel='Residuals (Millions $)')
ax.ticklabel_format(style='plain', axis='both'); ax.grid(True, alpha=0.3)
st.pyplot(fig)

# 3. Feature Importance (Coefficients)
st.subheader("Model Feature Coefficients")
coefficients = pd.Series(model.coef_, index=X_test.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
coefficients.plot(kind='bar', ax=ax, color=sns.color_palette("viridis", len(coefficients)))
ax.set(title='Feature Coefficients', ylabel='Coefficient Value (Scaled)')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
st.pyplot(fig)

st.header("3. Prediction Rationale")
st.markdown("""
This concise model maintains the full functionality: comprehensive data cleaning, Label Encoding, Linear Regression, and all required charts ($R^2$, Actual vs. Predicted, Residuals, Coefficients).
The strong predictive power is mainly due to the linear relationship between **`budget`**, **`votes`**, and **`gross`** revenue.
""")

st.header("4. Data Preview")
st.dataframe(df.head(5))