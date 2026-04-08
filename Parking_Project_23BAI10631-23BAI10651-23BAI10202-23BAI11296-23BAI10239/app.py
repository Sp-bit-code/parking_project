import streamlit as st
import cv2
import numpy as np
import tempfile
import os

from train_prediction import train_model
from predict import predict_parking
from recommend import recommend_parking

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Parking System", layout="wide")

st.title("🚗 Intelligent Parking System (Single Image Input)")

# =========================
# SESSION STATE
# =========================
if "trained" not in st.session_state:
    st.session_state.trained = False

# =========================
# STEP 1: TRAIN MODEL
# =========================
st.header("Step 1: Train Model")

if not st.session_state.trained:
    if st.button("Train Model"):
        try:
            train_model()
            st.session_state.trained = True
            st.success("Model trained successfully ✅")
        except Exception as e:
            st.error(e)
else:
    st.success("Model already trained ✅")

# =========================
# STEP 2: UPLOAD IMAGE
# =========================
st.header("Step 2: Upload Parking Image")

uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# =========================
# DETECTION FUNCTION (NO MASK)
# =========================
def detect_from_image(img):
    """
    Simple grid-based parking detection
    """
    output = img.copy()

    h, w, _ = img.shape

    rows = 3
    cols = 6

    slot_h = h // rows
    slot_w = w // cols

    occupied = 0
    total = rows * cols

    for i in range(rows):
        for j in range(cols):
            x1 = j * slot_w
            y1 = i * slot_h
            x2 = x1 + slot_w
            y2 = y1 + slot_h

            roi = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            mean_val = np.mean(gray)

            # Occupancy logic
            if mean_val < 100:
                color = (0, 0, 255)
                occupied += 1
            else:
                color = (0, 255, 0)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

    free = total - occupied

    return output, occupied, free, total

# =========================
# STEP 3: PROCESS
# =========================
if uploaded_image and st.session_state.trained:

    if st.button("Analyze Parking"):

        try:
            # Save temp image
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_image.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_image.read())

            img = cv2.imread(temp_path)

            # =========================
            # DETECTION
            # =========================
            output, occupied, free, total = detect_from_image(img)

            # =========================
            # PREDICTION
            # =========================
            pred = predict_parking(10, "Monday")

            # =========================
            # RECOMMENDATION
            # =========================
            rec = recommend_parking(total, occupied, 10, "Monday")

            # =========================
            # DISPLAY
            # =========================
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(temp_path, use_container_width=True)

            with col2:
                st.subheader("Detected Slots")
                st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("---")

            st.subheader("📊 Results")
            st.write(f"Total Slots: {total}")
            st.write(f"Occupied: {occupied}")
            st.write(f"Free: {free}")

            st.markdown("---")

            st.subheader("🔮 Prediction")
            st.write(f"{pred['prediction']} ({pred['confidence']}%)")

            st.markdown("---")

            st.subheader("📍 Recommendation")
            st.write(rec["recommendation"])

        except Exception as e:
            st.error(e)

elif not st.session_state.trained:
    st.warning("⚠️ Train model first")