import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def load_data(data_dir):
    """
    Memuat data hasil preprocessing (NumPy arrays dan CSV).
    """
    # Pastikan file ada sebelum memuat
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
        y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        # Return dummy data for testing structure if files are missing
        print("Returning dummy data for testing purposes.")
        X_train = np.random.rand(10, 5)
        X_test = np.random.rand(5, 5)
        y_train = np.random.randint(0, 2, 10)
        y_test = np.random.randint(0, 2, 5)
        
    return X_train, X_test, y_train, y_test

def train_model():
    # Set MLflow tracking URI ke lokal
    # Menggunakan relative path './mlruns' agar tersimpan di dalam folder MLProject
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Basic_Model_Experiment")
    
    # Enable Autologging
    mlflow.autolog()
    
    # Path dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'heart_disease_preprocessing')
    
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(data_dir)
    print(f"Data loaded. X_train shape: {X_train.shape}")

    with mlflow.start_run():
        print("Training model...")
        # Inisialisasi model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Training
        model.fit(X_train, y_train)
        
        # Evaluasi (Autolog akan menangkap ini, tapi kita bisa print juga)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")
        
        print("Model training completed and logged to MLflow.")

if __name__ == "__main__":
    train_model()
