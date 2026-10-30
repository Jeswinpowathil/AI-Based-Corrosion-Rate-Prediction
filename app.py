
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="CorroPredict AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef5ff, #f7fbff);
}
h1, h2, h3 {
    color: #12345a;
}
div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dce6f1;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🛡️ CorroPredict AI")
st.subheader("AI-Based Corrosion Rate Prediction System")
st.write(
    "Predict corrosion-related values from material composition "
    "and environmental conditions."
)

st.info(
    "Dataset measurement units have not yet been verified. "
    "Predictions are shown in the dataset's recorded units, not confirmed mm/year."
)

# ---------------- LOAD DATA & TRAIN MODEL ----------------
@st.cache_resource
def load_model():
    data_path = Path("corrosion_dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError("corrosion_dataset.csv was not found.")

    df = pd.read_csv(data_path)
    target = "Corrosion rate"

    X = df.drop(columns=[target])
    y = df[target]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X.columns.tolist(), df

try:
    model, features, df = load_model()
except Exception as error:
    st.error(f"Could not load the dataset or train the model: {error}")
    st.stop()

# ---------------- DATASET SUMMARY ----------------
st.markdown("### 📊 Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Samples", len(df))
c2.metric("Input Features", len(features))
c3.metric("Missing Values", int(df.isnull().sum().sum()))

st.divider()

# ---------------- INPUT FORM ----------------
st.markdown("### 🔬 Enter Material & Environmental Data")
st.write("Adjust the values below, then click Predict.")

with st.form("prediction_form"):
    inputs = {}

    st.markdown("#### 🌡️ Environmental Conditions")
    env_cols = st.columns(3)

    environmental_features = [
        "T_MAX", "T_MIN", "T_AVE", "RH_MIN", "RH_AVE",
        "SUNSHINE", "TOW", "PRECIPIT", "WIND_MAX",
        "WIND_AVE", "SOLAR", "UV", "CHLORIDE", "SO2", "TIME"
    ]

    for i, feature in enumerate(environmental_features):
        if feature in features:
            with env_cols[i % 3]:
                inputs[feature] = st.number_input(
                    feature,
                    value=float(df[feature].median()),
                    format="%.4f",
                    help="Default is the dataset median. Check the dataset's documentation for units."
                )

    st.markdown("#### 🧪 Material Composition")
    material_cols = st.columns(3)

    material_features = [
        "Si", "Mn", "P", "S", "Al", "Cu", "Cr", "Ni", "ELEMENTS"
    ]

    for i, feature in enumerate(material_features):
        if feature in features:
            with material_cols[i % 3]:
                inputs[feature] = st.number_input(
                    feature,
                    value=float(df[feature].median()),
                    format="%.4f",
                    help="Default is the dataset median."
                )

    submitted = st.form_submit_button(
        "🔍 Predict Corrosion Rate",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------
if submitted:
    input_df = pd.DataFrame(
        [[inputs[feature] for feature in features]],
        columns=features
    )

    prediction = float(model.predict(input_df)[0])

    st.markdown("### 📈 Prediction Result")
    st.metric("Predicted Corrosion Rate", f"{prediction:.6f}")
    st.caption("Unit not verified — this is in the dataset's recorded units.")

    st.markdown("### 🛠️ Inspection Guidance")
    st.warning(
        "This prototype cannot determine a safe inspection interval "
        "or classify corrosion risk without verified units and engineering thresholds. "
        "Have a qualified engineer assess the result before making maintenance decisions."
    )

# ---------------- DATASET VISUALIZATION ----------------
st.divider()
st.markdown("### 📉 Dataset Corrosion-Rate Distribution")

chart_data = df["Corrosion rate"].dropna()
st.bar_chart(chart_data.value_counts().sort_index())

with st.expander("View sample dataset records"):
    st.dataframe(df.head(10), use_container_width=True)

st.caption("CorroPredict AI | Academic prototype")
