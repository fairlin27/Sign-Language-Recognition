"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
HELP - REAL-TIME ACCURACY TEST
==========================================================

Tests ONLY:
    HELP

Attempts:
    20 automatic real-time attempts

Controls:
    Q -> Quit

No dataset modification.
No model retraining.
Existing trained model only.
==========================================================
"""

import os
import csv
import time
import cv2
import numpy as np

from collections import Counter

from camera import CameraManager
from predictor import Predictor


# ==========================================================
# SETTINGS
# ==========================================================

TARGET_GESTURE = "HELP"

ATTEMPTS_REQUIRED = 20

MIN_CONFIDENCE_PERCENT = 85.0

STABLE_TIME = 0.8

COOLDOWN_TIME = 1.5


# ==========================================================
# OUTPUT
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

RESULT_PATH = os.path.join(
    OUTPUT_DIR,
    "help_realtime_accuracy_results.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# CONFIDENCE
# ==========================================================

def normalize_confidence(value):

    try:

        confidence = float(value)

    except (
        TypeError,
        ValueError
    ):

        return 0.0

    if confidence <= 1.0:

        confidence *= 100.0

    return max(
        0.0,
        min(
            confidence,
            100.0
        )
    )


# ==========================================================
# DRAW TEXT
# ==========================================================

def draw_text(
    frame,
    text,
    x,
    y,
    scale=0.75,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ==========================================================
# STATUS
# ==========================================================

def draw_status(
    frame,
    text
):

    height = 60

    y = (
        frame.shape[0]
        -
        height
    )

    cv2.rectangle(
        frame,
        (0, y),
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
        (20, y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


# ==========================================================
# MAIN
# ==========================================================

camera = None

predictions = []

confidences = []

attempt = 0

stable_prediction = None

stable_start = None

cooldown_until = 0.0

waiting_for_reset = False


try:

    print("=" * 70)

    print(
        "       SIGN LANGUAGE RECOGNITION SYSTEM"
    )

    print(
        "          HELP REAL-TIME ACCURACY TEST"
    )

    print("=" * 70)

    print("\n")

    print(
        "Target Gesture : HELP"
    )

    print(
        "Required Tests : 20"
    )

    print(
        "Auto Recording : ENABLED"
    )

    print(
        "Q              : Quit"
    )

    print("\nOpening camera...")

    camera = CameraManager()

    predictor = Predictor()

    predictor.reset()

    print("\n✓ Camera Ready")

    print("✓ Prediction Module Ready")

    print("\nShow HELP sign and hold it steady.")

    # ======================================================
    # LOOP
    # ======================================================

    while attempt < ATTEMPTS_REQUIRED:

        success, frame = camera.read()

        if (
            not success
            or
            frame is None
        ):

            continue

        # ==================================================
        # DETECTION
        # ==================================================

        result = camera.detect(
            frame
        )

        hand_count = camera.get_hand_count(
            result
        )

        landmark_count = camera.get_landmark_count(
            result
        )

        frame = camera.draw(
            frame,
            result
        )

        # ==================================================
        # PREDICTION
        # ==================================================

        gesture = "Waiting..."

        raw_confidence = 0.0

        if hand_count > 0:

            try:

                gesture, raw_confidence = (
                    predictor.predict(
                        result
                    )
                )

            except Exception as error:

                print(
                    "Prediction Error:",
                    error
                )

                gesture = "Unknown"

                raw_confidence = 0.0

        confidence = normalize_confidence(
            raw_confidence
        )

        now = time.perf_counter()

        valid = (
            gesture != "Unknown"
            and
            gesture != "Waiting..."
            and
            hand_count > 0
            and
            confidence >= MIN_CONFIDENCE_PERCENT
        )

        # ==================================================
        # WAIT FOR RESET
        # ==================================================

        if waiting_for_reset:

            if (
                hand_count == 0
                or
                not valid
            ):

                waiting_for_reset = False

                stable_prediction = None

                stable_start = None

        # ==================================================
        # COOLDOWN
        # ==================================================

        cooldown = (
            now
            <
            cooldown_until
        )

        # ==================================================
        # STABILITY
        # ==================================================

        if (
            valid
            and
            not waiting_for_reset
            and
            not cooldown
        ):

            if gesture == stable_prediction:

                if stable_start is None:

                    stable_start = now

            else:

                stable_prediction = gesture

                stable_start = now

        else:

            if (
                not waiting_for_reset
                and
                not cooldown
            ):

                stable_prediction = None

                stable_start = None

        # ==================================================
        # STABLE DURATION
        # ==================================================

        stable_duration = 0.0

        if stable_start is not None:

            stable_duration = (
                now
                -
                stable_start
            )

        # ==================================================
        # AUTO RECORD
        # ==================================================

        if (
            valid
            and
            not waiting_for_reset
            and
            not cooldown
            and
            stable_prediction == gesture
            and
            stable_duration >= STABLE_TIME
        ):

            attempt += 1

            predictions.append(
                gesture
            )

            confidences.append(
                confidence
            )

            correct = (
                gesture
                ==
                TARGET_GESTURE
            )

            result_text = (
                "CORRECT"
                if correct
                else "WRONG"
            )

            print("\n")

            print(
                f"AUTO RECORD "
                f"{attempt}/"
                f"{ATTEMPTS_REQUIRED}"
            )

            print(
                f"Target      : "
                f"{TARGET_GESTURE}"
            )

            print(
                f"Predicted   : "
                f"{gesture}"
            )

            print(
                f"Confidence  : "
                f"{confidence:.2f}%"
            )

            print(
                f"Result      : "
                f"{result_text}"
            )

            cooldown_until = (
                now
                +
                COOLDOWN_TIME
            )

            waiting_for_reset = True

            stable_prediction = None

            stable_start = None

        # ==================================================
        # DISPLAY
        # ==================================================

        draw_text(
            frame,
            "TARGET: HELP",
            20,
            40
        )

        draw_text(
            frame,
            f"ATTEMPT: "
            f"{attempt}/"
            f"{ATTEMPTS_REQUIRED}",
            20,
            85
        )

        draw_text(
            frame,
            f"PREDICTION: "
            f"{gesture}",
            20,
            130
        )

        draw_text(
            frame,
            f"CONFIDENCE: "
            f"{confidence:.2f}%",
            20,
            175
        )

        draw_text(
            frame,
            f"HANDS: "
            f"{hand_count}",
            20,
            220
        )

        draw_text(
            frame,
            f"LANDMARKS: "
            f"{landmark_count}",
            20,
            265
        )

        # ==================================================
        # STATUS
        # ==================================================

        if waiting_for_reset:

            status = (
                "RECORDED - "
                "SHOW HELP AGAIN"
            )

        elif cooldown:

            remaining = max(
                0.0,
                cooldown_until
                -
                now
            )

            status = (
                f"NEXT ATTEMPT "
                f"IN {remaining:.1f}s"
            )

        elif (
            valid
            and
            stable_prediction == gesture
        ):

            remaining = max(
                0.0,
                STABLE_TIME
                -
                stable_duration
            )

            status = (
                f"HOLD STEADY - "
                f"RECORD IN "
                f"{remaining:.1f}s"
            )

        elif hand_count > 0:

            status = (
                "HAND DETECTED - "
                "RECOGNIZING"
            )

        else:

            status = (
                "SHOW HELP SIGN"
            )

        draw_status(
            frame,
            status
        )

        # ==================================================
        # DISPLAY WINDOW
        # ==================================================

        cv2.imshow(
            "HELP Real-Time Accuracy Test",
            frame
        )

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        # ==================================================
        # QUIT
        # ==================================================

        if key == ord("q"):

            print(
                "\nTest stopped by user."
            )

            break


finally:

    if camera is not None:

        try:

            camera.close()

        except Exception:
            pass

    cv2.destroyAllWindows()


# ==========================================================
# FINAL RESULT
# ==========================================================

if predictions:

    total_tests = len(
        predictions
    )

    correct_count = sum(
        prediction
        ==
        TARGET_GESTURE
        for prediction
        in predictions
    )

    wrong_count = (
        total_tests
        -
        correct_count
    )

    accuracy = (
        correct_count
        /
        total_tests
        *
        100.0
    )

    average_confidence = (
        np.mean(
            confidences
        )
    )

    most_predicted = (
        Counter(
            predictions
        )
        .most_common(1)[0][0]
    )

    print("\n")

    print("=" * 70)

    print(
        "              HELP RESULT"
    )

    print("=" * 70)

    print(
        f"Total Tests      : "
        f"{total_tests}"
    )

    print(
        f"Correct          : "
        f"{correct_count}"
    )

    print(
        f"Wrong            : "
        f"{wrong_count}"
    )

    print(
        f"Accuracy         : "
        f"{accuracy:.2f}%"
    )

    print(
        f"Avg Confidence   : "
        f"{average_confidence:.2f}%"
    )

    print(
        f"Most Predicted   : "
        f"{most_predicted}"
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

        writer.writerow(
            [
                TARGET_GESTURE,
                total_tests,
                correct_count,
                wrong_count,
                accuracy,
                average_confidence,
                most_predicted
            ]
        )

    print("\n✓ Results saved:")

    print(
        RESULT_PATH
    )

else:

    print(
        "\nNo HELP test results were recorded."
    )


print("\n")

print("=" * 70)

print(
    "       HELP REAL-TIME ACCURACY TEST COMPLETED"
)

print("=" * 70)