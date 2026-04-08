import os
from pathlib import Path
import joblib
import numpy as np


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(r"C:\Users\LENOVO\Desktop\parking_project")
MODEL_PATH = BASE_DIR / "model" / "prediction.pkl"


# ============================================================
# DAY MAPPING
# ============================================================
DAY_MAP = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}


# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    """
    Load the trained prediction model from prediction.pkl.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"First train the model and save it as prediction.pkl"
        )

    model = joblib.load(MODEL_PATH)
    return model


# ============================================================
# INPUT VALIDATION
# ============================================================
def convert_day_to_number(day_value):
    """
    Convert weekday name to number.
    Monday = 0, Tuesday = 1, ... Sunday = 6
    """
    if isinstance(day_value, int):
        if 0 <= day_value <= 6:
            return day_value
        raise ValueError("Day number must be between 0 and 6")

    if isinstance(day_value, str):
        day_value = day_value.strip().lower()
        if day_value in DAY_MAP:
            return DAY_MAP[day_value]
        raise ValueError(
            "Invalid day name. Use Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, or Sunday."
        )

    raise TypeError("day must be either a string or an integer")


def validate_hour(hour):
    """
    Check whether hour is valid.
    """
    try:
        hour = int(hour)
    except:
        raise ValueError("Hour must be a number")

    if hour < 0 or hour > 23:
        raise ValueError("Hour must be between 0 and 23")

    return hour


# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_parking(hour, day):
    """
    Predict parking demand based on hour and day.

    Parameters
    ----------
    hour : int
        Hour of the day (0 to 23)
    day : str or int
        Day name or day number

    Returns
    -------
    dict
        {
            "prediction": "High Occupancy" or "Low Occupancy",
            "confidence": percentage,
            "hour": hour,
            "day": day_number
        }
    """
    model = load_model()

    hour = validate_hour(hour)
    day_num = convert_day_to_number(day)

    # Input for model
    X_new = np.array([[hour, day_num]])

    # Predict class
    pred_class = model.predict(X_new)[0]

    # Predict probability if model supports it
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0]
        confidence = float(np.max(proba) * 100)
    else:
        confidence = 0.0

    # Convert numeric label to text
    if pred_class == 1:
        label = "High Occupancy"
    else:
        label = "Low Occupancy"

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "hour": hour,
        "day": day_num
    }


# ============================================================
# TEST RUN
# ============================================================
if __name__ == "__main__":
    try:
        result = predict_parking(10, "Monday")
        print("Prediction Result:")
        print(f"Hour       : {result['hour']}")
        print(f"Day        : {result['day']}")
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']}%")
    except Exception as e:
        print("Error:", e)