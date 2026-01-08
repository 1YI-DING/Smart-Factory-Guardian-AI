import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Smart Manufacturing Predictive Maintenance Dashboard", layout="wide")


# --- 1. Data Loading & Preprocessing ---
@st.cache_data  # Streamlit caching mechanism to avoid repeated loading
def load_data():
    # Assuming you have downloaded ai4i2020.csv
    # Simulating UCI dataset column names here
    df = pd.read_csv('ai4i2020.csv')

    # Simple feature engineering
    df['Type_Num'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
    df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']

    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error("Please ensure that ai4i2020.csv exists in the directory.")
    st.stop()


# --- 2. Model Training ---
@st.cache_resource  # Cache model objects
def train_models(data):
    features = ['Type_Num', 'Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Temp_Diff']
    X = data[features]
    y = data['Machine failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest - adding class_weight to handle class imbalance
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)

    # Anomaly Detection
    iso = IsolationForest(contamination=0.04, random_state=42)
    iso.fit(X)

    # Calculate evaluation metrics
    y_pred = rf.predict(X_test)
    metrics = {
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

    return rf, iso, metrics, features


rf_model, iso_model, model_metrics, feature_cols = train_models(df)

# --- 3. Sidebar: Manual Prediction (Digital Twin Simulation) ---
st.sidebar.header("🛠 Real-time Equipment Parameter Input")


def user_input_features():
    type_opt = st.sidebar.selectbox("Product Type (Type)", ('L', 'M', 'H'))
    air_temp = st.sidebar.slider("Air temperature [K]", 295.0, 305.0, 300.0)
    proc_temp = st.sidebar.slider("Process temperature [K]", 305.0, 315.0, 310.0)
    rpm = st.sidebar.number_input("Rotational speed [rpm]", 1200, 2800, 1500)
    torque = st.sidebar.number_input("Torque [Nm]", 3.0, 76.0, 40.0)
    tool_wear = st.sidebar.slider("Tool wear [min]", 0, 250, 50)

    type_map = {'L': 0, 'M': 1, 'H': 2}
    temp_diff = proc_temp - air_temp

    data = {
        'Type_Num': type_map[type_opt],
        'Air temperature [K]': air_temp,
        'Process temperature [K]': proc_temp,
        'Rotational speed [rpm]': rpm,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Temp_Diff': temp_diff
    }
    return pd.DataFrame([data])


input_df = user_input_features()

# --- 4. Main Interface Layout ---
st.title("🏭 Smart Manufacturing: Predictive Maintenance Dashboard")
st.markdown("This system utilizes Random Forest and Isolation Forest algorithms to monitor industrial equipment health in real-time.")

# Row 1: Core Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model Recall", f"{model_metrics['Recall']:.2%}")
col2.metric("F1 Score", f"{model_metrics['F1']:.2f}")
col3.metric("Total Samples", len(df))
col4.metric("Failure Samples", df['Machine failure'].sum())

st.divider()

# Row 2: Prediction Results
res_col1, res_col2 = st.columns(2)

with res_col1:
    st.subheader("🔮 Failure Risk Prediction")
    prediction = rf_model.predict(input_df)[0]
    prob = rf_model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High failure risk detected! (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Equipment operating normally (Failure probability: {prob:.2%})")

with res_col2:
    st.subheader("🔍 Anomaly Detection")
    is_anomaly = iso_model.predict(input_df)[0]
    if is_anomaly == -1:
        st.warning("🚨 Atypical operating pattern detected (Potential new failure type)")
    else:
        st.info("👍 Operating pattern follows historical norms")

st.divider()

# Row 3: Visual Analysis
st.subheader("📊 Sensor Correlation Analysis")
fig = px.scatter(
    df,
    x="Rotational speed [rpm]",
    y="Torque [Nm]",
    color="Machine failure",
    color_continuous_scale="RdBu_r",
    title="Rotational Speed vs Torque (Red represents failure points)",
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# Row 4: Feature Importance
st.subheader("💡 Which factors affect equipment life the most?")
importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

st.bar_chart(importances.set_index('Feature'))

# Display raw data
if st.checkbox("View Raw Dataset Preview"):
    st.write(df.head(50))