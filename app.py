import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ================== PAGE CONFIG ==================
st.set_page_config(layout="wide", page_title="Movie Revenue AI", page_icon="🎬")

# ================== STYLE ==================
st.markdown("""
<style>
div.stButton > button:first-child {
    height: 3em;
    width: 100%;
    font-weight: bold;
    border-radius: 10px;
    border: 1px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("movies.csv")
    except:
        np.random.seed(42)
        n_samples = 1000
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance']
        ratings = ['G', 'PG', 'PG-13', 'R']
        countries = ['USA', 'UK', 'France', 'Japan', 'India']

        data = {
            'budget': np.random.randint(1, 200, n_samples) * 1_000_000,
            'votes': np.random.randint(100, 1_000_000, n_samples),
            'score': np.random.uniform(2.0, 9.5, n_samples),
            'runtime': np.random.randint(60, 180, n_samples),
            'year': np.random.randint(1980, 2025, n_samples),
            'genre': np.random.choice(genres, n_samples),
            'rating': np.random.choice(ratings, n_samples),
            'country': np.random.choice(countries, n_samples),
            'director': [f"Director {i}" for i in range(n_samples)],
            'writer': [f"Writer {i}" for i in range(n_samples)],
            'star': [f"Star {i}" for i in range(n_samples)],
        }

        noise = np.random.normal(0, 20_000_000, n_samples)
        gross = (data['budget'] * 1.5) + (data['votes'] * 50) + (data['score'] * 10_000_000) + noise
        data['gross'] = np.abs(gross)

        return pd.DataFrame(data)

# ================== PREPROCESS DATA ==================
def preprocess_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    numeric_cols = ["budget", "votes", "score", "runtime", "gross"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.loc[df["budget"] <= 0, "budget"] = np.nan
    df.loc[df["votes"] < 0, "votes"] = np.nan
    df.loc[df["runtime"] <= 0, "runtime"] = np.nan
    df.loc[df["score"] < 0, "score"] = np.nan
    df.loc[df["gross"] < 0, "gross"] = np.nan

    # Fill missing values
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    cat_cols = ["genre", "rating", "country", "director", "writer", "star"]
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    df["budget_m"] = df["budget"] / 1_000_000
    df["gross_m"] = df["gross"] / 1_000_000

    median_gross = df["gross_m"].median()
    df["is_hit"] = (df["gross_m"] > median_gross).astype(int)

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders, median_gross

# ================== PLOT HELPERS ==================
def plot_confusion(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_roc(model, X_test, y_test, title):
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.2f}")
    ax.plot([0,1],[0,1], '--', label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def specificity(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

# ================== NAVIGATION ==================
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go(page):
    st.session_state.page = page

# ================== LOAD & PREP DATA ==================
df_raw = load_data()
df, encoders, median_threshold = preprocess_data(df_raw)
FEATURES = ['budget_m','votes','score','runtime','genre','rating','country','year']

# ================== HOME ==================
st.title("🎬 Movie Success AI Predictor")
st.markdown("---")

if st.session_state.page == "Home":
    st.subheader("Choose a Model to Train & Evaluate")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("Predict Exact Revenue")
        if st.button("Linear Regression"):
            go("Linear Regression")
    with c2:
        st.success("Hit / Flop")
        if st.button("Random Forest"):
            go("Random Forest")
    with c3:
        st.warning("Hit / Flop")
        if st.button("Logistic Regression"):
            go("Logistic Regression")
    with c4:
        st.error("Compare All")
        if st.button("Comparison"):
            go("Comparison")
    st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba",
             caption="Movie Analytics Dashboard", width='stretch')

# ================== LINEAR REGRESSION ==================
elif st.session_state.page == "Linear Regression":
    st.header("📈 Linear Regression")
    st.button("Back", on_click=go, args=("Home",))

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    X = df[FEATURES]
    y = df["gross_m"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- METRICS & PLOTS ---
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2:.3f}")
    c2.metric("RMSE", f"{rmse:.2f}M")
    c3.metric("MAE", f"{mae:.2f}M")

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, y_pred, alpha=0.7)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax1.set_xlabel("Actual Revenue (M)")
        ax1.set_ylabel("Predicted Revenue (M)")
        ax1.set_title("Actual vs Predicted")
        st.pyplot(fig1)
    with col2:
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(0, color="black", linestyle="--")
        ax2.set_xlabel("Predicted Revenue (M)")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residual Plot")
        st.pyplot(fig2)

    # --- NEW: USER PREDICTION SECTION ---
    st.markdown("---")
    st.subheader("🎬 Predict Revenue for a New Movie")
    st.markdown("Enter the movie details below to get a revenue forecast:")

    with st.form("revenue_pred_form"):
        c_A, c_B, c_C, c_D = st.columns(4)
        with c_A:
            in_budget = st.number_input("Budget ($ Millions)", min_value=0.1, max_value=500.0, value=50.0)
            in_votes = st.number_input("IMDb Votes", min_value=0, value=10000)
        with c_B:
            in_score = st.slider("IMDb Score", 0.0, 10.0, 6.5)
            in_runtime = st.number_input("Runtime (min)", min_value=10, value=100)
        with c_C:
            in_genre = st.selectbox("Genre", encoders['genre'].classes_)
            in_rating = st.selectbox("Rating", encoders['rating'].classes_)
        with c_D:
            in_country = st.selectbox("Country", encoders['country'].classes_)
            in_year = st.number_input("Year", min_value=1900, max_value=2030, value=2024)

        submit_btn = st.form_submit_button("💰 Predict Revenue")

    if submit_btn:
        # Encode categorical inputs
        try:
            val_genre = encoders['genre'].transform([in_genre])[0]
            val_rating = encoders['rating'].transform([in_rating])[0]
            val_country = encoders['country'].transform([in_country])[0]

            # Create input array
            input_data = np.array([[in_budget, in_votes, in_score, in_runtime, 
                                    val_genre, val_rating, val_country, in_year]])
            
            # Predict
            pred_rev = model.predict(input_data)[0]
            
            # Display Result
            st.success(f"### Predicted Revenue: ${pred_rev:,.2f} Million")
            
            # Optional: Visual Context
            if pred_rev > median_threshold:
                 st.balloons()
                 st.write(f"🎉 This exceeds the median industry revenue of ${median_threshold:.2f}M!")
            else:
                 st.write(f"⚠️ This is below the median industry revenue of ${median_threshold:.2f}M.")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# ================== RANDOM FOREST ==================
elif st.session_state.page == "Random Forest":
    st.header("🌲 Random Forest Classifier")
    st.button("Back", on_click=go, args=("Home",))
    
    X = df[FEATURES]
    y = df["is_hit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    c2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
    c3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
    c4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2%}")
    c5.metric("Specificity", f"{specificity(y_test, y_pred):.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_confusion(y_test, y_pred, "Random Forest Confusion Matrix"))
    with col2:
        st.pyplot(plot_roc(model, X_test, y_test, "Random Forest ROC Curve"))

# ================== LOGISTIC REGRESSION ==================
elif st.session_state.page == "Logistic Regression":
    st.header("📊 Logistic Regression")
    st.button("Back", on_click=go, args=("Home",))
    
    X = df[FEATURES]
    y = df["is_hit"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use fitted scaler
    
    # Train on standard scaled data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Test
    y_pred = model.predict(X_test_scaled)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    c2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
    c3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
    c4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2%}")
    c5.metric("Specificity", f"{specificity(y_test, y_pred):.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_confusion(y_test, y_pred, "Logistic Regression Confusion Matrix"))
    with col2:
        st.pyplot(plot_roc(model, X_test_scaled, y_test, "Logistic Regression ROC Curve"))

# ================== COMPARISON ==================
elif st.session_state.page == "Comparison":
    st.header("⚔️ Compare All Models")
    st.button("Back", on_click=go, args=("Home",))

    X = df[FEATURES]
    y_reg = df["gross_m"]
    y_cls = df["is_hit"]

    # Linear Regression (for comparison purposes only)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    lin = LinearRegression().fit(X_train_r, y_train_r)
    pred_reg = (lin.predict(X_test_r) > median_threshold).astype(int)

    # Random Forest
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42
    ).fit(X_train_c, y_train_c)
    pred_rf = rf.predict(X_test_c)

    # Logistic Regression
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_s)
    X_test_scaled = scaler.transform(X_test_s)
    
    log = LogisticRegression(max_iter=1000, random_state=42).fit(
        X_train_scaled, y_train_s
    )
    pred_log = log.predict(X_test_scaled)

    models = ["Linear (Thresholded)", "Random Forest", "Logistic Regression"]
    accs = [
        accuracy_score(y_test_c, pred_reg),
        accuracy_score(y_test_c, pred_rf),
        accuracy_score(y_test_s, pred_log)
    ]
    f1s = [
        f1_score(y_test_c, pred_reg),
        f1_score(y_test_c, pred_rf),
        f1_score(y_test_s, pred_log)
    ]

    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(models))
    ax.bar(x-0.2, accs, width=0.4, label="Accuracy", color='skyblue')
    ax.bar(x+0.2, f1s, width=0.4, label="F1 Score", color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Comparison of Models: Accuracy vs F1 Score")
    ax.legend()
    st.pyplot(fig)

    best_model = models[np.argmax(accs)]
    st.success(f"🏆 Best Model Based on Accuracy: {best_model}")