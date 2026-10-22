# ==========================================
# SIGN LANGUAGE PREDICTION V2
# Part 1 - Import & Load Models
# ==========================================

import cv2
import joblib
import mediapipe as mp
import numpy as np
from collections import Counter
import threading
import pyttsx3

print("=" * 60)
print("SIGN LANGUAGE PREDICTION V2")
print("=" * 60)

# ==========================================
# Model Paths
# ==========================================

MODEL_PATH = "models/hand_landmarker.task"
GESTURE_MODEL_PATH = "models/gesture_model.pkl"
SCALER_PATH = "models/scaler.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# ==========================================
# Load Trained Files
# ==========================================

gesture_model = joblib.load(GESTURE_MODEL_PATH)
print("✓ Gesture Model Loaded")

scaler = joblib.load(SCALER_PATH)
print("✓ Scaler Loaded")

label_encoder = joblib.load(LABEL_ENCODER_PATH)
print("✓ Label Encoder Loaded")

# ==========================================
# MediaPipe Tasks API
# ==========================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode

# ==========================================
# Hand Landmarker Options
# ==========================================

options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=2,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5

)

# ==========================================
# Variables for Stable Prediction
# ==========================================

prediction_history = []

HISTORY_SIZE = 15
# ==========================================
# Text To Speech Engine
# ==========================================

engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

engine.setProperty("rate", 150)

engine.setProperty("volume", 1.0)

last_spoken = ""

print("✓ HandLandmarker Ready")

print("=" * 60)
print("Part 1 Completed Successfully")
print("=" * 60)
# ==========================================
# Part 2 - Webcam & MediaPipe Setup
# ==========================================

# Hand Connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# Open Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✓ Webcam Opened")

# Create HandLandmarker
landmarker = HandLandmarker.create_from_options(options)

print("✓ HandLandmarker Created")

frame_timestamp_ms = 0

print("=" * 60)
print("Part 2 Completed Successfully")
print("=" * 60)
# ==========================================
# Part 3A - Feature Extraction Function
# ==========================================

def extract_features(result):

    # Left Hand Features (63 values)
    left_features = [0.0] * 63

    # Right Hand Features (63 values)
    right_features = [0.0] * 63

    # No hands detected
    if len(result.hand_landmarks) == 0:
        return np.array(left_features + right_features).reshape(1, -1)

    # Process all detected hands
    for hand_landmarks, handedness in zip(
        result.hand_landmarks,
        result.handedness
    ):

        current_hand = []

        # Collect x, y, z values
        for landmark in hand_landmarks:

            current_hand.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # Left or Right
        hand_name = handedness[0].category_name

        if hand_name == "Left":
            left_features = current_hand

        elif hand_name == "Right":
            right_features = current_hand

    # Combine Left + Right = 126 features
    features = left_features + right_features

    # Convert to NumPy
    features = np.array(features).reshape(1, -1)

    return features
# ==========================================
# Part 3B - Prediction Function
# ==========================================

def predict_gesture(features):

    # Scale Features
    scaled_features = scaler.transform(features)

    # Predict Gesture
    prediction = gesture_model.predict(scaled_features)

    # Gesture Name
    gesture = label_encoder.inverse_transform(prediction)[0]

    # Confidence Score
    confidence = 0.0

    if hasattr(gesture_model, "predict_proba"):

        probabilities = gesture_model.predict_proba(
            scaled_features
        )

        confidence = np.max(probabilities) * 100

    return gesture, confidence
# ==========================================
# Part 3C-1 - Main Loop
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read frame")
        break

    # Mirror Image
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Increase Timestamp
    frame_timestamp_ms += 33

    # Detect Hands
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    # Number of Hands
    hand_count = len(result.hand_landmarks)
    if hand_count == 0:
      prediction_history.clear()
    last_spoken = ""

    # Display Hand Count
    cv2.putText(
        frame,
        f"Hands : {hand_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    # ==========================================
    # Draw Landmarks
    # ==========================================

    if hand_count > 0:

        height, width, _ = frame.shape

        # Draw every detected hand
        for hand_landmarks in result.hand_landmarks:

            # Draw landmark points
            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw hand connections
            for start, end in HAND_CONNECTIONS:

                x1 = int(hand_landmarks[start].x * width)
                y1 = int(hand_landmarks[start].y * height)

                x2 = int(hand_landmarks[end].x * width)
                y2 = int(hand_landmarks[end].y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

        # ==========================================
        # Build 126 Feature Vector
        # ==========================================

        features = extract_features(result)

        # ==========================================
        # Predict Gesture
        # ==========================================

        gesture, confidence = predict_gesture(features)
        print("Prediction :", gesture)
        # ==========================================
        # Stable Prediction (Majority Voting)
        # ==========================================
        #if confidence < 85:
        #      gesture = "Unknown"

        prediction_history.append(gesture)

        if len(prediction_history) > HISTORY_SIZE:
            prediction_history.pop(0)

        stable_gesture = Counter(
            prediction_history
        ).most_common(1)[0][0]
        print("Stable :", stable_gesture)
        # ==========================================
        # Speak Only When Gesture Changes
        # ==========================================
        if stable_gesture != last_spoken:

           threading.Thread(
             target=speak,
             args=(stable_gesture,),
             daemon=True
            ).start()

        last_spoken = stable_gesture

        # ==========================================
        # Display Gesture
        # ==========================================

        cv2.putText(
            frame,
            "Gesture : " + stable_gesture,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence : {confidence:.2f}%",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    # ==========================================
    # Show Webcam
    # ==========================================

    cv2.imshow(
        "Sign Language Prediction V2",
        frame
    )

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# Cleanup
# ==========================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()