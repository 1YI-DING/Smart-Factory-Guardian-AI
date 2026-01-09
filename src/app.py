import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import joblib  # Used for loading model files
import os

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "sensor_data.csv"
model_dir = BASE_DIR / "models"

# --- Page Configuration ---
st.set_page_config(page_title="Industrial Equipment Predictive Maintenance Dashboard", layout="wide")

# --- 1. Data Loading ---
@st.cache_data
def load_data():
    # Load the "Pro" dataset for visualization
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    else:
        st.error(f"Data file {data_path} not found. Please run generate_realistic_data.py first.")
        st.stop()

# --- 2. Load Pre-trained Models ---
@st.cache_resource
def load_trained_models():
    """Load models and feature lists from local files to avoid redundant training"""
    try:
        rf = joblib.load(model_dir / 'rf_model.pkl')
        iso = joblib.load(model_dir / 'iso_model.pkl')
        # Load the feature order recorded during training to prevent errors
        features = joblib.load(model_dir / 'feature_names.pkl')
        return rf, iso, features
    except FileNotFoundError:
        st.error("Model files not found! Please ensure the 'train.py' script has been run.")
        st.stop()

# Initialize data and models
df = load_data()
rf_model, iso_model, feature_cols = load_trained_models()

# --- 3. Sidebar: Real-time Input Simulator ---
st.sidebar.header("Real-time Sensor Simulator")

def user_input_features():
    # Input fields correspond to training features: 'Vibration', 'Temperature', 'Pressure', 'OperatingHours', 'Vibration_Mean'
    vib = st.sidebar.slider("Current Vibration", 0.0, 1.5, 0.5)
    temp = st.sidebar.slider("Current Temperature", 50.0, 100.0, 70.0)
    pres = st.sidebar.slider("Current Pressure", 50.0, 150.0, 100.0)
    hours = st.sidebar.slider("Total Operating Hours", 0.0, 500.0, 200.0)
    vib_mean = st.sidebar.slider("Rolling Mean Vibration", 0.0, 1.5, 0.5)

    data = {
        'Vibration': vib,
        'Temperature': temp,
        'Pressure': pres,
        'OperatingHours': hours,
        'Vibration_Mean': vib_mean
    }
    # The order here must match feature_cols
    return pd.DataFrame([data])[feature_cols]

input_df = user_input_features()

# --- 4. Main Interface ---
st.title("Smart Factory: Equipment Health Monitoring System (Inference)")
st.markdown("This version loads trained models directly from disk, saving training resources.")

# Row 1: Key Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Samples", len(df))
c2.metric("Total Failures", int(df['Failure'].sum()))
c3.metric("Feature Dimensions", len(feature_cols))

st.divider()

# Row 2: Real-time Prediction Panel
res1, res2 = st.columns(2)
with res1:
    st.subheader("Failure Risk Prediction")
    # Get failure probability
    prob = rf_model.predict_proba(input_df)[0][1]
    if prob > 0.5:
        st.error(f"High Risk! Failure Probability: {prob:.2%}")
        st.button("Create Maintenance Order Now")
    else:
        st.success(f"Healthy Status (Failure Probability: {prob:.2%})")

with res2:
    st.subheader("Operation Pattern Analysis")
    # Isolation Forest prediction: 1 Normal, -1 Anomaly
    is_anomaly = iso_model.predict(input_df)[0]
    if is_anomaly == -1:
        st.warning("Abnormal pattern detected! Check for sensor drift.")
    else:
        st.info("Normal operation pattern")

st.divider()

# Row 3: Historical Trends
st.subheader("📈 Equipment Degradation Curve (History)")
fig = px.line(df.tail(1000), x='OperatingHours', y='Vibration',
              color='Failure',
              color_discrete_map={0: "green", 1: "red"},
              title='Vibration Intensity vs Time (Red indicates failure points)')
st.plotly_chart(fig, use_container_width=True)

if st.checkbox("Show underlying feature list"):
    st.write("Feature order used during model training:", feature_cols)