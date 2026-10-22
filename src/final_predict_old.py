# ==========================================
# SIGN LANGUAGE RECOGNITION SYSTEM
# FINAL PREDICTION WITH SPEECH
# Part 1 - Imports & Model Loading
# ==========================================

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3
import time

print("=" * 60)
print("SIGN LANGUAGE RECOGNITION SYSTEM")
print("FINAL PREDICTION WITH SPEECH")
print("=" * 60)

# ==========================================
# Model Paths
# ==========================================

MODEL_PATH = "models/hand_landmarker.task"
GESTURE_MODEL_PATH = "models/gesture_model.pkl"
SCALER_PATH = "models/scaler.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# ==========================================
# Load Trained Models
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
# HandLandmarker Options
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

print("✓ HandLandmarker Ready")

# ==========================================
# Text To Speech
# ==========================================

engine = pyttsx3.init()

engine.setProperty("rate", 150)

engine.setProperty("volume", 1.0)

# ==========================================
# Speech Variables
# ==========================================

last_prediction = ""

stable_count = 0

last_spoken = ""

STABLE_FRAMES = 8

# ==========================================
# Speech Function
# ==========================================

def speak(text):

    engine.stop()

    engine.say(text)

    engine.runAndWait()

print("✓ Text To Speech Ready")

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
    print("❌ Cannot Open Webcam")
    exit()

print("✓ Webcam Opened")

# Create HandLandmarker
landmarker = HandLandmarker.create_from_options(options)

print("✓ HandLandmarker Created")

# Frame Timestamp
frame_timestamp_ms = 0

print("=" * 60)
print("Part 2 Completed Successfully")
print("=" * 60)
# ==========================================
# Part 3 - Feature Extraction Function
# ==========================================

def extract_features(result):

    # Left Hand (63 values)
    left_features = [0.0] * 63

    # Right Hand (63 values)
    right_features = [0.0] * 63

    # No Hand
    if len(result.hand_landmarks) == 0:
        return np.array(left_features + right_features).reshape(1, -1)

    # Process Detected Hands
    for hand_landmarks, handedness in zip(
        result.hand_landmarks,
        result.handedness
    ):

        current_hand = []

        for landmark in hand_landmarks:

            current_hand.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        hand_name = handedness[0].category_name

        if hand_name == "Left":
            left_features = current_hand

        elif hand_name == "Right":
            right_features = current_hand

    features = left_features + right_features

    return np.array(features).reshape(1, -1)


# ==========================================
# Part 3 - Prediction Function
# ==========================================

def predict_gesture(features):

    # Feature Scaling
    scaled_features = scaler.transform(features)

    # Predict Gesture
    prediction = gesture_model.predict(
        scaled_features
    )

    gesture = label_encoder.inverse_transform(
        prediction
    )[0]

    confidence = 0.0

    # Get Confidence Score
    if hasattr(gesture_model, "predict_proba"):

        probabilities = gesture_model.predict_proba(
            scaled_features
        )

        confidence = float(
            np.max(probabilities) * 100
        )

    return gesture, confidence


print("=" * 60)
print("Part 3 Completed Successfully")
print("=" * 60)
# ==========================================
# Part 4A - Main Loop
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read frame")
        break

    # Mirror View
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    frame_timestamp_ms += 33

    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    hand_count = len(result.hand_landmarks)

    # Display Hand Count
    cv2.putText(
        frame,
        f"Hands : {hand_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # No Hands
    if hand_count == 0:

      last_prediction = ""

      stable_count = 0

      last_spoken = ""

    # Draw Hands
    if hand_count > 0:

        height, width, _ = frame.shape

        for hand_landmarks in result.hand_landmarks:

            # Draw Landmarks
            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x,y),
                    5,
                    (0,255,0),
                    -1
                )

            # Draw Connections
            for start, end in HAND_CONNECTIONS:

                x1 = int(hand_landmarks[start].x * width)
                y1 = int(hand_landmarks[start].y * height)

                x2 = int(hand_landmarks[end].x * width)
                y2 = int(hand_landmarks[end].y * height)

                cv2.line(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (255,0,0),
                    2
                )

        # Extract Features
        features = extract_features(result)

        # Predict
        gesture, confidence = predict_gesture(features)
        # ==========================================
        # Stable Frame Logic
        # ==========================================
        if gesture == last_prediction:
          stable_count += 1
        else:
           last_prediction = gesture
           stable_count = 1

        # ==========================================
        # Speak Only Once
        # ==========================================

        if (
           stable_count >= STABLE_FRAMES
           and gesture != last_spoken
           ):

           speak(gesture)
           last_spoken = gesture

        # ==========================================
        # Display Gesture
        # ==========================================

        cv2.putText(
            frame,
            "Gesture : " + gesture,
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence : {confidence:.2f}%",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2
        )

    # ==========================================
    # Show Webcam
    # ==========================================

    cv2.imshow(
        "Sign Language Recognition",
        frame
    )

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# Cleanup
# ==========================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()