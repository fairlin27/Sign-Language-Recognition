"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
REAL-TIME ACCURACY TEST
AUTO RECORD VERSION
==========================================================

Features:
    - No SPACE key required
    - Automatic prediction recording
    - One-hand and two-hand support
    - Confidence threshold
    - Stable prediction checking
    - Automatic attempt counting
    - Clear camera overlay
    - Gesture-wise accuracy
    - Overall real-time accuracy

Controls:
    N -> Skip current gesture
    Q -> Quit test

IMPORTANT:
    Show the target sign.
    Keep it stable.
    The system automatically records the attempt.
    After recording, remove/move the hand and show
    the sign again for the next attempt.
==========================================================
"""

import os
import csv
import time

import cv2
import joblib
import numpy as np

from collections import Counter

from camera import CameraManager
from predictor import Predictor


# ==========================================================
# PROJECT PATH
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "gesture_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "label_encoder.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

RESULT_PATH = os.path.join(
    OUTPUT_DIR,
    "realtime_accuracy_results.csv"
)


# ==========================================================
# TEST SETTINGS
# ==========================================================

ATTEMPTS_PER_GESTURE = 10

# Confidence threshold as percentage
MIN_CONFIDENCE_PERCENT = 85.0

# Sign must remain stable for this duration
STABLE_TIME = 0.8

# Delay after recording
COOLDOWN_TIME = 1.5

# Hand/prediction must reset before next attempt
RESET_REQUIRED = True


# ==========================================================
# GESTURES
# ==========================================================

GESTURES = [
    "DOCTOR",
    "EMERGENCY",
    "FAMILY",
    "FOOD",
    "FRIEND",
    "GOOD",
    "HELLO",
    "HELP",
    "LOVE",
    "MONEY",
    "NO",
    "PHONE",
    "PLEASE",
    "SORRY",
    "THANK YOU",
    "TIME",
    "WATER",
    "WELCOME",
    "YES",
    "YOU"
]


# ==========================================================
# HEADER
# ==========================================================

print("=" * 70)

print(
    "       SIGN LANGUAGE RECOGNITION SYSTEM"
)

print(
    "             REAL-TIME ACCURACY TEST"
)

print(
    "                 AUTO RECORD"
)

print("=" * 70)


# ==========================================================
# CONFIDENCE NORMALIZATION
# ==========================================================

def normalize_confidence(value):
    """
    Converts confidence into percentage format.

    Supported input:

        0.9132
        91.32

    Both become:

        91.32

    This prevents:

        9132.00%

    from appearing.
    """

    try:

        confidence = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return 0.0

    # ------------------------------------------------------
    # If predictor returns 0.0 - 1.0
    # ------------------------------------------------------

    if confidence <= 1.0:

        confidence = (
            confidence * 100.0
        )

    # ------------------------------------------------------
    # If predictor already returns 0 - 100
    # ------------------------------------------------------

    else:

        confidence = confidence

    # ------------------------------------------------------
    # Keep inside valid percentage range
    # ------------------------------------------------------

    confidence = max(
        0.0,
        min(
            confidence,
            100.0
        )
    )

    return confidence


# ==========================================================
# CHECK FILES
# ==========================================================

print("\nChecking required files...")

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH
]

for file_path in required_files:

    if not os.path.exists(
        file_path
    ):

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{file_path}"
        )

print(
    "✓ All required files found"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading trained model...")

model = joblib.load(
    MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

print(
    "✓ Model loaded"
)

print(
    "✓ Scaler loaded"
)

print(
    "✓ Label encoder loaded"
)


# ==========================================================
# VERIFY GESTURES
# ==========================================================

available_labels = [
    str(label)
    for label in label_encoder.classes_
]

test_gestures = []

for gesture in GESTURES:

    if gesture in available_labels:

        test_gestures.append(
            gesture
        )

    else:

        print(
            f"Warning: {gesture} "
            f"not available in model"
        )


print(
    f"\nGestures available for testing: "
    f"{len(test_gestures)}"
)


# ==========================================================
# OUTPUT FOLDER
# ==========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# CAMERA
# ==========================================================

print("\nOpening camera...")

camera = CameraManager()

predictor = Predictor()


# ==========================================================
# RESULTS
# ==========================================================

results = []


# ==========================================================
# DRAW BOX TEXT
# ==========================================================

def draw_box_text(
    frame,
    text,
    x,
    y,
    font_scale=0.75,
    thickness=2,
    padding=10
):

    font = cv2.FONT_HERSHEY_SIMPLEX

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - text_height - padding
    )

    x2 = min(
        frame.shape[1],
        x + text_width + padding
    )

    y2 = min(
        frame.shape[0],
        y + baseline + padding
    )

    # ------------------------------------------------------
    # Dark background
    # ------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (20, 20, 20),
        -1
    )

    # ------------------------------------------------------
    # White text
    # ------------------------------------------------------

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ==========================================================
# DRAW STATUS BOX
# ==========================================================

def draw_status_box(
    frame,
    text,
    color
):

    height = 60

    y1 = (
        frame.shape[0]
        -
        height
    )

    cv2.rectangle(
        frame,
        (0, y1),
        (
            frame.shape[1],
            frame.shape[0]
        ),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        text,
        (20, y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA
    )


# ==========================================================
# TEST ONE GESTURE
# ==========================================================

def test_gesture(
    target_gesture,
    gesture_number,
    total_gestures
):

    print("\n")

    print("=" * 70)

    print(
        f"Testing {gesture_number}/"
        f"{total_gestures}: "
        f"{target_gesture}"
    )

    print(
        f"Required attempts: "
        f"{ATTEMPTS_PER_GESTURE}"
    )

    print(
        "Show the sign and hold it steady."
    )

    print(
        "Recording will happen automatically."
    )

    print(
        "N = Skip    Q = Quit"
    )

    print("=" * 70)

    # ------------------------------------------------------
    # Predictions
    # ------------------------------------------------------

    predictions = []

    confidence_values = []

    # ------------------------------------------------------
    # Attempt counter
    # ------------------------------------------------------

    attempt = 0

    # ------------------------------------------------------
    # Stability
    # ------------------------------------------------------

    stable_prediction = None

    stable_start_time = None

    # ------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------

    cooldown_until = 0.0

    # ------------------------------------------------------
    # Reset state
    # ------------------------------------------------------

    waiting_for_reset = False

    # ------------------------------------------------------
    # Camera loop
    # ------------------------------------------------------

    while (
        attempt
        <
        ATTEMPTS_PER_GESTURE
    ):

        success, frame = (
            camera.read()
        )

        if (
            not success
            or
            frame is None
        ):

            continue

        # ==================================================
        # DETECT HANDS
        # ==================================================

        detection_result = (
            camera.detect(
                frame
            )
        )

        # ==================================================
        # HAND COUNT
        # ==================================================

        hand_count = (
            camera.get_hand_count(
                detection_result
            )
        )

        # ==================================================
        # LANDMARK COUNT
        # ==================================================

        landmark_count = (
            camera.get_landmark_count(
                detection_result
            )
        )

        # ==================================================
        # DRAW LANDMARKS
        # ==================================================

        display_frame = (
            camera.draw(
                frame,
                detection_result
            )
        )

        # ==================================================
        # PREDICTION
        # ==================================================

        gesture = "Waiting..."

        raw_confidence = 0.0

        if hand_count > 0:

            try:

                (
                    gesture,
                    raw_confidence
                ) = predictor.predict(
                    detection_result
                )

            except Exception as error:

                print(
                    "Prediction Error:",
                    error
                )

                gesture = "Unknown"

                raw_confidence = 0.0

        # ==================================================
        # CORRECT CONFIDENCE
        # ==================================================

        confidence_percent = (
            normalize_confidence(
                raw_confidence
            )
        )

        # ==================================================
        # CURRENT TIME
        # ==================================================

        current_time = (
            time.perf_counter()
        )

        # ==================================================
        # VALID PREDICTION
        # ==================================================

        valid_prediction = (
            gesture != "Unknown"
            and
            gesture != "Waiting..."
            and
            confidence_percent
            >=
            MIN_CONFIDENCE_PERCENT
            and
            hand_count > 0
        )

        # ==================================================
        # RESET STATE
        # ==================================================

        if waiting_for_reset:

            reset_detected = (
                hand_count == 0
                or
                not valid_prediction
            )

            if reset_detected:

                waiting_for_reset = False

                stable_prediction = None

                stable_start_time = None

        # ==================================================
        # COOLDOWN
        # ==================================================

        in_cooldown = (
            current_time
            <
            cooldown_until
        )

        # ==================================================
        # STABILITY
        # ==================================================

        if (
            valid_prediction
            and
            not waiting_for_reset
            and
            not in_cooldown
        ):

            if (
                gesture
                ==
                stable_prediction
            ):

                if (
                    stable_start_time
                    is None
                ):

                    stable_start_time = (
                        current_time
                    )

            else:

                stable_prediction = (
                    gesture
                )

                stable_start_time = (
                    current_time
                )

        else:

            if (
                not waiting_for_reset
                and
                not in_cooldown
            ):

                stable_prediction = None

                stable_start_time = None

        # ==================================================
        # STABLE DURATION
        # ==================================================

        stable_duration = 0.0

        if (
            stable_start_time
            is not None
        ):

            stable_duration = (
                current_time
                -
                stable_start_time
            )

        # ==================================================
        # AUTO RECORD
        # ==================================================

        if (
            valid_prediction
            and
            not waiting_for_reset
            and
            not in_cooldown
            and
            stable_prediction
            ==
            gesture
            and
            stable_duration
            >=
            STABLE_TIME
        ):

            # --------------------------------------------------
            # Record
            # --------------------------------------------------

            predictions.append(
                gesture
            )

            confidence_values.append(
                confidence_percent
            )

            attempt += 1

            is_correct = (
                gesture
                ==
                target_gesture
            )

            result_text = (
                "CORRECT"
                if is_correct
                else "WRONG"
            )

            print("\n")

            print(
                f"AUTO RECORD "
                f"{attempt}/"
                f"{ATTEMPTS_PER_GESTURE}"
            )

            print(
                f"Target      : "
                f"{target_gesture}"
            )

            print(
                f"Predicted   : "
                f"{gesture}"
            )

            print(
                f"Confidence  : "
                f"{confidence_percent:.2f}%"
            )

            print(
                f"Result      : "
                f"{result_text}"
            )

            # --------------------------------------------------
            # Cooldown
            # --------------------------------------------------

            cooldown_until = (
                current_time
                +
                COOLDOWN_TIME
            )

            # --------------------------------------------------
            # Require reset
            # --------------------------------------------------

            waiting_for_reset = (
                RESET_REQUIRED
            )

            stable_prediction = None

            stable_start_time = None

        # ==================================================
        # UI
        # ==================================================

        draw_box_text(
            display_frame,
            f"TARGET: {target_gesture}",
            20,
            40,
            0.80,
            2
        )

        draw_box_text(
            display_frame,
            f"ATTEMPT: "
            f"{attempt}/"
            f"{ATTEMPTS_PER_GESTURE}",
            20,
            90,
            0.70,
            2
        )

        draw_box_text(
            display_frame,
            f"PREDICTION: "
            f"{gesture}",
            20,
            140,
            0.70,
            2
        )

        draw_box_text(
            display_frame,
            f"CONFIDENCE: "
            f"{confidence_percent:.1f}%",
            20,
            190,
            0.70,
            2
        )

        draw_box_text(
            display_frame,
            f"HANDS: "
            f"{hand_count}/2",
            20,
            240,
            0.70,
            2
        )

        draw_box_text(
            display_frame,
            f"LANDMARKS: "
            f"{landmark_count}/42",
            20,
            290,
            0.70,
            2
        )

        # ==================================================
        # STATUS
        # ==================================================

        if waiting_for_reset:

            status_text = (
                "RECORDED - "
                "SHOW SIGN AGAIN"
            )

            status_color = (
                0,
                255,
                255
            )

        elif in_cooldown:

            remaining = max(
                0.0,
                cooldown_until
                -
                current_time
            )

            status_text = (
                "NEXT ATTEMPT READY "
                f"IN {remaining:.1f}s"
            )

            status_color = (
                0,
                255,
                255
            )

        elif (
            valid_prediction
            and
            stable_prediction
            ==
            gesture
        ):

            remaining = max(
                0.0,
                STABLE_TIME
                -
                stable_duration
            )

            status_text = (
                "HOLD STEADY - "
                f"AUTO RECORD IN "
                f"{remaining:.1f}s"
            )

            status_color = (
                0,
                255,
                0
            )

        elif hand_count > 0:

            status_text = (
                "HAND DETECTED - "
                "RECOGNIZING"
            )

            status_color = (
                0,
                200,
                255
            )

        else:

            status_text = (
                "SHOW TARGET SIGN"
            )

            status_color = (
                255,
                255,
                255
            )

        draw_status_box(
            display_frame,
            status_text,
            status_color
        )

        # ==================================================
        # SHOW WINDOW
        # ==================================================

        cv2.imshow(
            "Real-Time Accuracy Test",
            display_frame
        )

        # ==================================================
        # KEYBOARD
        # ==================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        # --------------------------------------------------
        # QUIT
        # --------------------------------------------------

        if key == ord("q"):

            return "QUIT"

        # --------------------------------------------------
        # SKIP
        # --------------------------------------------------

        if key == ord("n"):

            print(
                f"\nSkipped: "
                f"{target_gesture}"
            )

            return "SKIPPED"

    # ======================================================
    # GESTURE RESULT
    # ======================================================

    correct = sum(
        prediction
        ==
        target_gesture
        for prediction in predictions
    )

    wrong = (
        len(predictions)
        -
        correct
    )

    accuracy = (
        correct
        /
        len(predictions)
        *
        100.0
    ) if predictions else 0.0

    average_confidence = (
        np.mean(
            confidence_values
        )
    ) if confidence_values else 0.0

    # ------------------------------------------------------
    # Most common prediction
    # ------------------------------------------------------

    if predictions:

        most_common = (
            Counter(
                predictions
            )
            .most_common(
                1
            )[0][0]
        )

    else:

        most_common = "N/A"

    # ======================================================
    # PRINT GESTURE RESULT
    # ======================================================

    print("\n")

    print(
        f"{target_gesture} RESULT"
    )

    print(
        f"Correct       : "
        f"{correct}"
    )

    print(
        f"Wrong         : "
        f"{wrong}"
    )

    print(
        f"Accuracy      : "
        f"{accuracy:.2f}%"
    )

    print(
        f"Avg Confidence: "
        f"{average_confidence:.2f}%"
    )

    print(
        f"Most Predicted: "
        f"{most_common}"
    )

    # ======================================================
    # STORE RESULT
    # ======================================================

    results.append(
        [
            target_gesture,
            len(predictions),
            correct,
            wrong,
            accuracy,
            average_confidence,
            most_common
        ]
    )

    return "DONE"


# ==========================================================
# MAIN TEST
# ==========================================================

try:

    total_gestures = len(
        test_gestures
    )

    for index, gesture in enumerate(
        test_gestures,
        start=1
    ):

        # --------------------------------------------------
        # Reset predictor
        # --------------------------------------------------

        predictor.reset()

        # --------------------------------------------------
        # Test gesture
        # --------------------------------------------------

        result = test_gesture(
            gesture,
            index,
            total_gestures
        )

        # --------------------------------------------------
        # Quit
        # --------------------------------------------------

        if result == "QUIT":

            print(
                "\nTest stopped by user."
            )

            break

        # --------------------------------------------------
        # Skip
        # --------------------------------------------------

        if result == "SKIPPED":

            continue

        # --------------------------------------------------
        # Next gesture
        # --------------------------------------------------

        print(
            "\nPreparing next gesture..."
        )

        time.sleep(
            1
        )


finally:

    # ======================================================
    # CLOSE CAMERA
    # ======================================================

    try:

        camera.close()

    except Exception:
        pass

    cv2.destroyAllWindows()


# ==========================================================
# FINAL RESULTS
# ==========================================================

if results:

    total_tests = sum(
        row[1]
        for row in results
    )

    total_correct = sum(
        row[2]
        for row in results
    )

    total_wrong = sum(
        row[3]
        for row in results
    )

    overall_accuracy = (
        total_correct
        /
        total_tests
        *
        100.0
    ) if total_tests else 0.0

    # ======================================================
    # FINAL PRINT
    # ======================================================

    print("\n")

    print("=" * 70)

    print(
        "              REAL-TIME RESULTS"
    )

    print("=" * 70)

    print(
        f"\nTotal Tests      : "
        f"{total_tests}"
    )

    print(
        f"Correct          : "
        f"{total_correct}"
    )

    print(
        f"Wrong            : "
        f"{total_wrong}"
    )

    print(
        f"Overall Accuracy : "
        f"{overall_accuracy:.2f}%"
    )

    print("\n")

    print(
        f"{'GESTURE':<15}"
        f"{'TESTS':<8}"
        f"{'CORRECT':<10}"
        f"{'WRONG':<8}"
        f"{'ACCURACY':<12}"
    )

    print(
        "-" * 65
    )

    for row in results:

        print(
            f"{row[0]:<15}"
            f"{row[1]:<8}"
            f"{row[2]:<10}"
            f"{row[3]:<8}"
            f"{row[4]:>8.2f}%"
        )

    # ======================================================
    # SAVE CSV
    # ======================================================

    with open(
        RESULT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "Gesture",
                "Total Tests",
                "Correct",
                "Wrong",
                "Accuracy (%)",
                "Average Confidence (%)",
                "Most Predicted"
            ]
        )

        for row in results:

            writer.writerow(
                row
            )

        writer.writerow([])

        writer.writerow(
            [
                "OVERALL",
                total_tests,
                total_correct,
                total_wrong,
                overall_accuracy
            ]
        )

    print("\n✓ Results saved:")

    print(
        RESULT_PATH
    )

else:

    print(
        "\nNo test results were recorded."
    )


# ==========================================================
# END
# ==========================================================

print("\n")

print("=" * 70)

print(
    "        REAL-TIME ACCURACY TEST COMPLETED"
)

print("=" * 70)