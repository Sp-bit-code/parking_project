from pathlib import Path

# Import the prediction function from predict.py
from predict import predict_parking


# ============================================================
# RECOMMENDATION LOGIC
# ============================================================
def recommend_parking(total_slots, occupied_slots, hour, day):
    """
    Recommend whether the user should park now or wait.

    Parameters
    ----------
    total_slots : int
        Total parking slots available in the parking area.

    occupied_slots : int
        Number of slots currently occupied.

    hour : int
        Current or future hour (0 to 23).

    day : str or int
        Day name or day number.

    Returns
    -------
    dict
        {
            "total_slots": ...,
            "occupied_slots": ...,
            "free_slots": ...,
            "prediction": ...,
            "confidence": ...,
            "recommendation": ...
        }
    """
    # Convert input values safely
    total_slots = int(total_slots)
    occupied_slots = int(occupied_slots)

    if total_slots < 0:
        raise ValueError("total_slots cannot be negative")

    if occupied_slots < 0:
        raise ValueError("occupied_slots cannot be negative")

    if occupied_slots > total_slots:
        raise ValueError("occupied_slots cannot be greater than total_slots")

    free_slots = total_slots - occupied_slots

    # Get prediction from predict.py
    pred_result = predict_parking(hour, day)

    predicted_state = pred_result["prediction"]
    confidence = pred_result["confidence"]

    # Recommendation rules
    if free_slots == 0:
        recommendation = "No parking slot available right now."
    elif free_slots <= 3:
        recommendation = "Very few slots are free. Park immediately."
    elif predicted_state == "High Occupancy":
        recommendation = "Parking may get crowded soon. Better park now."
    else:
        recommendation = "Good time to park. Slots are likely available."

    return {
        "total_slots": total_slots,
        "occupied_slots": occupied_slots,
        "free_slots": free_slots,
        "prediction": predicted_state,
        "confidence": confidence,
        "recommendation": recommendation
    }


# ============================================================
# TEST RUN
# ============================================================
if __name__ == "__main__":
    try:
        result = recommend_parking(
            total_slots=50,
            occupied_slots=42,
            hour=10,
            day="Monday"
        )

        print("Recommendation Result")
        print("---------------------")
        print("Total Slots     :", result["total_slots"])
        print("Occupied Slots  :", result["occupied_slots"])
        print("Free Slots      :", result["free_slots"])
        print("Prediction      :", result["prediction"])
        print("Confidence      :", result["confidence"], "%")
        print("Recommendation  :", result["recommendation"])

    except Exception as e:
        print("Error:", e)