import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_and_save():
    # 1. Load data
    print("Starting data loading...")
    try:
        df = pd.read_csv('sensor_data.csv')
    except FileNotFoundError:
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
    joblib.dump(rf_model, 'rf_model.pkl')
    joblib.dump(iso_model, 'iso_model.pkl')

    # Recommendation: save the feature list to prevent mismatching order during inference
    joblib.dump(features, 'feature_names.pkl')

    print("Success! Models saved as 'rf_model.pkl' and 'iso_model.pkl'")


if __name__ == "__main__":
    train_and_save()