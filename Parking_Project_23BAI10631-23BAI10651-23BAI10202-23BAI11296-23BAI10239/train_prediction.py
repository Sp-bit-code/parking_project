import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(r"C:\Users\LENOVO\Desktop\parking_project")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "prediction.pkl"

# Optional real training CSV
# If you later create a CSV with columns: hour, day, occupied_slots
TRAIN_CSV_PATH = DATA_DIR / "prediction_data.csv"


# ============================================================
# DAY MAPPING
# ============================================================
DAY_TO_NUM = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================
def generate_synthetic_data(n_samples=500, random_state=42):
    """
    Create a small synthetic dataset for training.
    This is useful if you do not yet have a real CSV.
    Features:
        hour: 0-23
        day: 0-6
        occupied_slots: generated from a simple pattern
    Target:
        1 = High Occupancy
        0 = Low Occupancy
    """
    np.random.seed(random_state)

    hours = np.random.randint(0, 24, size=n_samples)
    days = np.random.randint(0, 7, size=n_samples)

    occupied_slots = []
    for h, d in zip(hours, days):
        # Simple realistic pattern:
        # weekdays and office hours = more occupied
        if d in [0, 1, 2, 3, 4]:   # Mon-Fri
            if 8 <= h <= 11 or 16 <= h <= 19:
                base = np.random.randint(25, 45)
            elif 12 <= h <= 15:
                base = np.random.randint(20, 35)
            else:
                base = np.random.randint(5, 20)
        else:  # weekend
            if 10 <= h <= 18:
                base = np.random.randint(10, 30)
            else:
                base = np.random.randint(2, 15)

        occupied_slots.append(base)

    df = pd.DataFrame({
        "hour": hours,
        "day": days,
        "occupied_slots": occupied_slots
    })

    # Convert to binary target
    threshold = df["occupied_slots"].median()
    df["target"] = (df["occupied_slots"] >= threshold).astype(int)

    return df


# ============================================================
# LOAD REAL DATA IF AVAILABLE
# ============================================================
def load_training_data():
    """
    Try to load a real CSV.
    If not found, generate synthetic data.
    """
    if TRAIN_CSV_PATH.exists():
        print(f"Real training CSV found: {TRAIN_CSV_PATH}")
        df = pd.read_csv(TRAIN_CSV_PATH)

        # Clean column names
        df.columns = [c.strip().lower() for c in df.columns]

        required_cols = {"hour", "day", "occupied_slots"}
        missing = required_cols - set(df.columns)

        if missing:
            raise ValueError(
                f"Training CSV is missing columns: {missing}. "
                f"Required columns are: hour, day, occupied_slots"
            )

        # Convert day names to numbers if needed
        if df["day"].dtype == "object":
            df["day"] = df["day"].astype(str).str.strip().str.lower().map(DAY_TO_NUM)

        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df["day"] = pd.to_numeric(df["day"], errors="coerce")
        df["occupied_slots"] = pd.to_numeric(df["occupied_slots"], errors="coerce")

        df = df.dropna(subset=["hour", "day", "occupied_slots"])
        df["hour"] = df["hour"].astype(int)
        df["day"] = df["day"].astype(int)
        df["occupied_slots"] = df["occupied_slots"].astype(int)

        threshold = df["occupied_slots"].median()
        df["target"] = (df["occupied_slots"] >= threshold).astype(int)

        print("Using real dataset for training.")
        return df

    print("No real training CSV found. Using synthetic data.")
    return generate_synthetic_data()


# ============================================================
# TRAIN MODEL
# ============================================================
def train_model():
    """
    Train RandomForest on hour + day features.
    Save model as prediction.pkl.
    """
    df = load_training_data()

    X = df[["hour", "day"]]
    y = df["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=10
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print("Model Training Complete")
    print("==============================")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved successfully at:")
    print(MODEL_PATH)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    train_model()