import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Get the project root directory (one level up from this script)
BASE_DIR = Path(__file__).resolve().parent.parent

def train_and_save():
    # 1. Load data
    data_path = BASE_DIR / "data" / "sensor_data.csv"

    print("Starting data loading...")
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data from: {data_path}")
    except FileNotFoundError:
        print(f"Error: Could not find file at {data_path}")
        print("Error: sensor_data.csv not found, please run the data generation script first!")
        return

    # 2. Define features and target
    # Ensure these column names match your data generation script exactly
    features = ['Vibration', 'Temperature', 'Pressure', 'OperatingHours', 'Vibration_Mean']
    target = 'Failure'

    X = df[features]
    y = df[target]

    # 3. Split training and testing sets (for model validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Random Forest classifier (RF)
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)

    # Output brief evaluation results
    y_pred = rf_model.predict(X_test)
    print("\nClassification Model Evaluation Report:")
    print(classification_report(y_test, y_pred))

    # 5. Train Anomaly Detection model (Isolation Forest)
    print("Training Anomaly Detection model...")
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    iso_model.fit(X)  # Anomaly detection usually uses the full dataset for feature baseline learning

    # 6. Save models to local files
    print("\nSaving models...")

    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)

    joblib.dump(rf_model, model_dir / 'rf_model.pkl')
    joblib.dump(iso_model, model_dir / 'iso_model.pkl')
    # Recommendation: save the feature list to prevent mismatching order during inference
    joblib.dump(features, model_dir / 'feature_names.pkl')

    print(f"Success! Models saved in: {model_dir}")


if __name__ == "__main__":
    train_and_save()