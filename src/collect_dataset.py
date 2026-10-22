import cv2
import mediapipe as mp
import csv
import os
import time

# ==================================================
# FILE PATHS
# ==================================================

MODEL_PATH = "models/hand_landmarker.task"
CSV_PATH = "dataset/landmarks.csv"

# ==================================================
# USER INPUT
# ==================================================

print("=" * 55)
print("     SIGN LANGUAGE DATASET COLLECTION")
print("=" * 55)

# Gesture name
while True:
    GESTURE_LABEL = input(
        "Enter Gesture Name : "
    ).strip().upper()

    if GESTURE_LABEL:
        break

    print("Gesture name cannot be empty.")


# Hand mode
print()
print("Select Hand Mode")
print("1 - One Hand")
print("2 - Two Hands")

while True:

    HAND_MODE = input(
        "Enter Hand Mode (1/2) : "
    ).strip()

    if HAND_MODE in ["1", "2"]:
        break

    print("Please enter only 1 or 2.")


# Number of samples
while True:

    try:

        MAX_SAMPLES = int(
            input("Enter Number of Samples : ")
        )

        if MAX_SAMPLES > 0:
            break

        print(
            "Enter a number greater than 0."
        )

    except ValueError:

        print(
            "Please enter a valid number."
        )


print()
print("=" * 55)
print("Gesture :", GESTURE_LABEL)

if HAND_MODE == "1":
    print("Hand Mode : ONE HAND")
else:
    print("Hand Mode : TWO HANDS")

print("Target Samples :", MAX_SAMPLES)
print("=" * 55)

# ==================================================
# SETTINGS
# ==================================================

# Countdown before saving
COUNTDOWN_SECONDS = 2

# Waiting time after saving one sample
NEXT_SAMPLE_DELAY = 1.0

# ==================================================
# MEDIAPIPE SETUP
# ==================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)

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

# ==================================================
# CREATE DATASET FOLDER
# ==================================================

os.makedirs(
    "dataset",
    exist_ok=True
)

# ==================================================
# CREATE CSV HEADER
# ==================================================

header = []

# Left hand
for i in range(21):

    header.extend([
        f"left_x{i}",
        f"left_y{i}",
        f"left_z{i}"
    ])


# Right hand
for i in range(21):

    header.extend([
        f"right_x{i}",
        f"right_y{i}",
        f"right_z{i}"
    ])


# Label
header.append("label")

# ==================================================
# CREATE CSV FILE
# ==================================================

if not os.path.exists(CSV_PATH):

    with open(
        CSV_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)

# ==================================================
# OPEN CAMERA
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera could not be opened.")

    exit()

# ==================================================
# VARIABLES
# ==================================================

sample_count = 0

frame_timestamp_ms = 0

countdown_active = False

countdown_start_time = 0

waiting_after_save = False

save_time = 0

# ==================================================
# CREATE HAND LANDMARKER
# ==================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # ==========================================
        # READ CAMERA
        # ==========================================

        success, frame = cap.read()

        if not success:

            print("Camera not found.")

            break

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        # ==========================================
        # CONVERT BGR TO RGB
        # ==========================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ==========================================
        # CREATE MEDIAPIPE IMAGE
        # ==========================================

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_frame
        )

        # ==========================================
        # TIMESTAMP
        # ==========================================

        frame_timestamp_ms += 33

        # ==========================================
        # DETECT HANDS
        # ==========================================

        result = landmarker.detect_for_video(

            mp_image,

            frame_timestamp_ms
        )

        # ==========================================
        # EMPTY LANDMARK VALUES
        # ==========================================

        left_features = [0.0] * 63

        right_features = [0.0] * 63

        left_detected = False

        right_detected = False

        # ==========================================
        # EXTRACT LANDMARKS
        # ==========================================

        if result.hand_landmarks:

            for index, hand_landmarks in enumerate(
                result.hand_landmarks
            ):

                features = []

                # 21 landmarks
                for landmark in hand_landmarks:

                    features.extend([

                        landmark.x,

                        landmark.y,

                        landmark.z
                    ])

                # ==================================
                # CHECK LEFT / RIGHT HAND
                # ==================================

                if index < len(
                    result.handedness
                ):

                    hand_name = (

                        result
                        .handedness[index][0]
                        .category_name
                        .lower()
                    )

                    if hand_name == "left":

                        left_features = features

                        left_detected = True

                    elif hand_name == "right":

                        right_features = features

                        right_detected = True

        # ==========================================
        # NUMBER OF HANDS
        # ==========================================

        hands_detected = (
            int(left_detected)
            +
            int(right_detected)
        )

        # ==========================================
        # CHECK REQUIRED HAND MODE
        # ==========================================

        if HAND_MODE == "1":

            required_hands_detected = (
                hands_detected >= 1
            )

        else:

            required_hands_detected = (
                left_detected
                and
                right_detected
            )

        # ==========================================
        # DISPLAY GESTURE
        # ==========================================

        cv2.putText(

            frame,

            f"Gesture : {GESTURE_LABEL}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )

        # ==========================================
        # DISPLAY SAMPLE COUNT
        # ==========================================

        cv2.putText(

            frame,

            f"Samples : {sample_count}/{MAX_SAMPLES}",

            (20, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )

        # ==========================================
        # DISPLAY HAND MODE
        # ==========================================

        if HAND_MODE == "1":

            mode_text = "Mode : One Hand"

        else:

            mode_text = "Mode : Two Hands"

        cv2.putText(

            frame,

            mode_text,

            (20, 120),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 0),

            2
        )

        # ==========================================
        # CURRENT TIME
        # ==========================================

        current_time = time.time()

        # ==========================================
        # WAIT AFTER SAMPLE SAVE
        # ==========================================

        if waiting_after_save:

            cv2.putText(

                frame,

                "SAMPLE SAVED!",

                (20, 170),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (0, 255, 0),

                3
            )

            cv2.putText(

                frame,

                "Change Hand Position",

                (20, 210),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 255),

                2
            )

            if (
                current_time - save_time
                >= NEXT_SAMPLE_DELAY
            ):

                waiting_after_save = False

                countdown_active = False

        # ==========================================
        # REQUIRED HANDS DETECTED
        # ==========================================

        elif required_hands_detected:

            if HAND_MODE == "1":

                detection_text = (
                    "Hand Detected"
                )

            else:

                detection_text = (
                    "Both Hands Detected"
                )

            cv2.putText(

                frame,

                detection_text,

                (20, 170),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )

            # ======================================
            # START COUNTDOWN
            # ======================================

            if not countdown_active:

                countdown_active = True

                countdown_start_time = (
                    current_time
                )

            elapsed = (
                current_time
                -
                countdown_start_time
            )

            countdown = (
                COUNTDOWN_SECONDS
                -
                int(elapsed)
            )

            # ======================================
            # DISPLAY COUNTDOWN
            # ======================================

            if countdown > 0:

                cv2.putText(

                    frame,

                    str(countdown),

                    (
                        frame.shape[1] // 2 - 30,
                        frame.shape[0] // 2
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    3,

                    (0, 0, 255),

                    6
                )

                cv2.putText(

                    frame,

                    "Hold Gesture",

                    (20, 210),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (255, 255, 0),

                    2
                )

            # ======================================
            # SAVE SAMPLE
            # ======================================

            else:

                row = (

                    left_features

                    +

                    right_features

                    +

                    [GESTURE_LABEL]
                )

                with open(

                    CSV_PATH,

                    "a",

                    newline=""

                ) as file:

                    writer = csv.writer(file)

                    writer.writerow(row)

                sample_count += 1

                print(

                    f"Saved {GESTURE_LABEL} "
                    f"sample : "
                    f"{sample_count}/{MAX_SAMPLES}"
                )

                waiting_after_save = True

                save_time = current_time

                countdown_active = False

        # ==========================================
        # REQUIRED HANDS NOT DETECTED
        # ==========================================

        else:

            countdown_active = False

            if HAND_MODE == "1":

                waiting_text = (
                    "Show One Hand"
                )

            else:

                waiting_text = (
                    "Show Both Hands"
                )

            cv2.putText(

                frame,

                waiting_text,

                (20, 170),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 0, 255),

                2
            )

            cv2.putText(

                frame,

                "Waiting...",

                (20, 210),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 255),

                2
            )

        # ==========================================
        # QUIT MESSAGE
        # ==========================================

        cv2.putText(

            frame,

            "Press Q to Quit",

            (20, 250),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 0),

            2
        )

        # ==========================================
        # DISPLAY CAMERA
        # ==========================================

        cv2.imshow(

            "Sign Language Dataset Collection",

            frame
        )

        # ==========================================
        # COLLECTION COMPLETED
        # ==========================================

        if sample_count >= MAX_SAMPLES:

            print()

            print("=" * 55)

            print(
                "DATASET COLLECTION COMPLETED"
            )

            print(
                "Gesture :",
                GESTURE_LABEL
            )

            print(
                "Samples :",
                sample_count
            )

            print(
                "Saved in :",
                CSV_PATH
            )

            print("=" * 55)

            # Display completion
            cv2.putText(

                frame,

                "COLLECTION COMPLETED",

                (
                    30,
                    frame.shape[0] - 40
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (0, 255, 0),

                3
            )

            cv2.imshow(

                "Sign Language Dataset Collection",

                frame
            )

            cv2.waitKey(1500)

            break

        # ==========================================
        # PRESS Q TO QUIT
        # ==========================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print(
                "Dataset collection stopped."
            )

            break

# ==================================================
# RELEASE CAMERA
# ==================================================

cap.release()

cv2.destroyAllWindows()