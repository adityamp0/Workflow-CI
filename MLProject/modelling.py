import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def load_data(data_dir):
    """
    Memuat data hasil preprocessing (NumPy arrays dan CSV).
    """
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
        y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Returning dummy data for testing purposes.")
        X_train = np.random.rand(10, 5)
        X_test = np.random.rand(5, 5)
        y_train = np.random.randint(0, 2, 10)
        y_test = np.random.randint(0, 2, 5)

    return X_train, X_test, y_train, y_test

def train_model():
    # Pastikan tracking URI konsisten (CLI akan override ini jika pakai mlflow run)
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    )

    # Aktifkan autologging (INI SAJA SUDAH CUKUP)
    mlflow.autolog()

    # Path dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'heart_disease_preprocessing')

    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(data_dir)
    print(f"Data loaded. X_train shape: {X_train.shape}")

    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc}")
    print("Model training completed and logged to MLflow.")

if __name__ == "__main__":
    train_model()
