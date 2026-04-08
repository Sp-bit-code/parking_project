import cv2
import numpy as np
import os


def detect_parking(image_path, mask_path):
    """
    Detect occupied and free parking slots using mask.
    
    Returns:
    output_image, occupied_count, free_count, total_slots
    """

    # =========================
    # LOAD IMAGE
    # =========================
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)

    if img is None:
        raise ValueError("Error loading image")

    if mask is None:
        raise ValueError("Error loading mask")

    # =========================
    # RESIZE MASK
    # =========================
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # =========================
    # BINARY MASK
    # =========================
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # =========================
    # FIND CONTOURS (SLOTS)
    # =========================
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output = img.copy()
    occupied = 0
    total = 0

    # =========================
    # PROCESS EACH SLOT
    # =========================
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore very small noise
        if area < 500:
            continue

        total += 1

        x, y, w, h = cv2.boundingRect(cnt)

        roi = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        mean_val = np.mean(gray)

        # =========================
        # OCCUPANCY LOGIC
        # =========================
        if mean_val < 100:
            # Occupied
            color = (0, 0, 255)  # red
            occupied += 1
        else:
            # Free
            color = (0, 255, 0)  # green

        # Draw box
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

    free = total - occupied

    return output, occupied, free, total


# =========================
# TEST RUN (OPTIONAL)
# =========================
if __name__ == "__main__":
    img_path = r"C:\Users\LENOVO\Desktop\parking_project\data\images\0.png"
    mask_path = r"C:\Users\LENOVO\Desktop\parking_project\data\boxes\0.png"

    output, occ, free, total = detect_parking(img_path, mask_path)

    print("Total:", total)
    print("Occupied:", occ)
    print("Free:", free)

    cv2.imshow("Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()