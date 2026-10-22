"""
======================================================================
SIGN LANGUAGE RECOGNITION SYSTEM
REAL-TIME ALGORITHM COMPARISON
======================================================================

Algorithms:
    1. SVM
    2. Random Forest
    3. KNN

Purpose:
    Compare the three trained ML algorithms using the same
    real-time hand landmark features.

Important:
    - No new training data is collected.
    - No model is retrained.
    - Only valid attempts are counted.
    - Hand must be detected before countdown starts.
    - Countdown is only 2 -> 1.
    - If hand disappears during countdown, countdown is cancelled.
    - If hand disappears during capture, attempt is cancelled.
    - Cancelled attempts are NOT counted.
    - Results are saved as CSV.
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

import os
import csv
import time
import joblib
import cv2
import numpy as np

from pathlib import Path

from camera import CameraManager


# =====================================================================
# PROJECT DIRECTORIES
# =====================================================================

# realtime_algorithm_comparison.py
# is inside:
#
# C:\Sign_Language_Project\src\
#
# Therefore parent of src = project root.

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================================
# MODEL FILES
# =====================================================================

SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"

RANDOM_FOREST_MODEL_PATH = (
    MODELS_DIR / "random_forest_model.pkl"
)

KNN_MODEL_PATH = MODELS_DIR / "knn_model.pkl"

SCALER_PATH = MODELS_DIR / "scaler.pkl"

LABEL_ENCODER_PATH = (
    MODELS_DIR / "label_encoder.pkl"
)

OUTPUT_CSV_PATH = (
    OUTPUT_DIR / "realtime_algorithm_comparison.csv"
)


# =====================================================================
# TEST SETTINGS
# =====================================================================

TEST_GESTURES = [
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

ATTEMPTS_PER_GESTURE = 10

# Countdown:
# 2 -> 1
COUNTDOWN_SECONDS = 2

# Time given to capture after countdown.
CAPTURE_WAIT_SECONDS = 1.5

# Small delay after displaying prediction.
RESULT_DISPLAY_SECONDS = 1.0


# =====================================================================
# DISPLAY SETTINGS
# =====================================================================

WINDOW_NAME = "Real-Time Algorithm Comparison"

FONT = cv2.FONT_HERSHEY_SIMPLEX


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def model_exists(path):
    """
    Check whether a model file exists.
    """

    return path.exists()


# ---------------------------------------------------------------------

def load_model(path, name):
    """
    Load a joblib model.
    """

    print(f"Loading {name}...")

    if not model_exists(path):

        raise FileNotFoundError(
            f"\nModel file not found:\n{path}\n"
        )

    model = joblib.load(path)

    print(f"✓ {name} Loaded")

    return model


# ---------------------------------------------------------------------

def get_hand_count(result):
    """
    Return number of detected hands.
    """

    if result is None:
        return 0

    if not hasattr(
        result,
        "hand_landmarks"
    ):
        return 0

    try:

        return len(
            result.hand_landmarks
        )

    except Exception:

        return 0


# ---------------------------------------------------------------------

def has_hand(result):
    """
    Return True if at least one hand is detected.
    """

    return get_hand_count(result) > 0


# ---------------------------------------------------------------------

def draw_text(
    frame,
    text,
    position,
    scale=0.8,
    thickness=2
):
    """
    Draw text on frame.
    """

    cv2.putText(
        frame,
        str(text),
        position,
        FONT,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ---------------------------------------------------------------------

def show_frame(
    camera,
    frame,
    result,
    message="",
    countdown=None
):
    """
    Draw landmarks and status text.
    """

    display = frame.copy()

    # ---------------------------------------------------------------
    # Draw hand landmarks
    # ---------------------------------------------------------------

    try:

        display = camera.draw(
            display,
            result
        )

    except Exception:

        pass

    # ---------------------------------------------------------------
    # Main message
    # ---------------------------------------------------------------

    if message:

        draw_text(
            display,
            message,
            (30, 50),
            0.8,
            2
        )

    # ---------------------------------------------------------------
    # Countdown
    # ---------------------------------------------------------------

    if countdown is not None:

        draw_text(
            display,
            f"{countdown}",
            (
                display.shape[1] // 2 - 20,
                display.shape[0] // 2
            ),
            2.0,
            4
        )

    # ---------------------------------------------------------------
    # Controls
    # ---------------------------------------------------------------

    draw_text(
        display,
        "Q = Quit    N = Skip",
        (30, display.shape[0] - 25),
        0.6,
        2
    )

    cv2.imshow(
        WINDOW_NAME,
        display
    )


# =====================================================================
# FEATURE EXTRACTION
# =====================================================================

def extract_features(result):
    """
    Extract 126 features.

    63 features:
        Left hand  = 21 landmarks x 3

    63 features:
        Right hand = 21 landmarks x 3

    Total:
        126 features
    """

    left_features = [0.0] * 63

    right_features = [0.0] * 63

    # ---------------------------------------------------------------
    # Invalid result
    # ---------------------------------------------------------------

    if result is None:

        return np.array(
            left_features + right_features,
            dtype=np.float32
        ).reshape(1, -1)

    # ---------------------------------------------------------------
    # Check MediaPipe result
    # ---------------------------------------------------------------

    if not hasattr(
        result,
        "hand_landmarks"
    ):

        return np.array(
            left_features + right_features,
            dtype=np.float32
        ).reshape(1, -1)

    if not hasattr(
        result,
        "handedness"
    ):

        return np.array(
            left_features + right_features,
            dtype=np.float32
        ).reshape(1, -1)

    # ---------------------------------------------------------------
    # Process hands
    # ---------------------------------------------------------------

    try:

        for hand_landmarks, handedness in zip(
            result.hand_landmarks,
            result.handedness
        ):

            current_hand = []

            # -------------------------------------------------------
            # 21 landmarks
            # -------------------------------------------------------

            for landmark in hand_landmarks:

                current_hand.extend([
                    float(landmark.x),
                    float(landmark.y),
                    float(landmark.z)
                ])

            # -------------------------------------------------------
            # Validate feature length
            # -------------------------------------------------------

            if len(current_hand) != 63:

                continue

            # -------------------------------------------------------
            # Get hand name
            # -------------------------------------------------------

            hand_name = ""

            try:

                if (
                    handedness
                    and
                    len(handedness) > 0
                ):

                    hand_name = (
                        handedness[0].category_name
                    )

            except Exception:

                hand_name = ""

            hand_name = str(
                hand_name
            ).lower().strip()

            # -------------------------------------------------------
            # Store left hand
            # -------------------------------------------------------

            if hand_name == "left":

                left_features = current_hand

            # -------------------------------------------------------
            # Store right hand
            # -------------------------------------------------------

            elif hand_name == "right":

                right_features = current_hand

    except Exception as error:

        print(
            "Feature Extraction Error:",
            error
        )

    # ---------------------------------------------------------------
    # Combine 126 features
    # ---------------------------------------------------------------

    features = (
        left_features
        +
        right_features
    )

    return np.array(
        features,
        dtype=np.float32
    ).reshape(1, -1)


# =====================================================================
# PREDICTION
# =====================================================================

def predict_algorithm(
    model,
    scaled_features,
    label_encoder
):
    """
    Predict gesture and confidence.
    """

    try:

        # -----------------------------------------------------------
        # Prediction
        # -----------------------------------------------------------

        prediction = model.predict(
            scaled_features
        )

        # -----------------------------------------------------------
        # Decode label
        # -----------------------------------------------------------

        gesture = (
            label_encoder
            .inverse_transform(
                prediction
            )[0]
        )

        gesture = str(
            gesture
        ).strip()

        # -----------------------------------------------------------
        # Confidence
        # -----------------------------------------------------------

        confidence = 0.0

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    scaled_features
                )
            )

            confidence = float(
                np.max(
                    probabilities
                ) * 100.0
            )

        return (
            gesture,
            confidence
        )

    except Exception as error:

        print(
            "Prediction Error:",
            error
        )

        return (
            "Unknown",
            0.0
        )


# =====================================================================
# COUNTDOWN
# =====================================================================

def wait_for_hand_and_countdown(
    camera,
    target_gesture
):
    """
    Wait until a hand is detected.

    Once a hand is detected:
        Countdown 2 -> 1

    If hand disappears:
        Countdown is cancelled.

    Returns:
        result, frame

    or

        None, None
    """

    print(
        "Waiting for hand..."
    )

    # ---------------------------------------------------------------
    # STEP 1
    # Wait for hand
    # ---------------------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:

            return None, None

        result = camera.detect(
            frame
        )

        # -----------------------------------------------------------
        # Keyboard
        # -----------------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            return "QUIT", None

        if key == ord("n"):

            return "SKIP", None

        # -----------------------------------------------------------
        # No hand
        # -----------------------------------------------------------

        if not has_hand(result):

            show_frame(
                camera,
                frame,
                result,
                "Waiting for hand..."
            )

            continue

        # -----------------------------------------------------------
        # Hand detected
        # -----------------------------------------------------------

        print(
            "Hand detected."
        )

        break

    # ===============================================================
    # STEP 2
    # COUNTDOWN 2 -> 1
    # ===============================================================

    countdown_start = time.monotonic()

    last_second = None

    while True:

        success, frame = camera.read()

        if not success:

            return None, None

        result = camera.detect(
            frame
        )

        # -----------------------------------------------------------
        # Keyboard
        # -----------------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            return "QUIT", None

        if key == ord("n"):

            return "SKIP", None

        # -----------------------------------------------------------
        # If hand disappears
        # -----------------------------------------------------------

        if not has_hand(result):

            print(
                "Countdown cancelled."
            )

            show_frame(
                camera,
                frame,
                result,
                "Hand lost - try again"
            )

            time.sleep(0.4)

            return None, None

        # -----------------------------------------------------------
        # Calculate elapsed time
        # -----------------------------------------------------------

        elapsed = (
            time.monotonic()
            -
            countdown_start
        )

        remaining = (
            COUNTDOWN_SECONDS
            -
            elapsed
        )

        # -----------------------------------------------------------
        # Countdown finished
        # -----------------------------------------------------------

        if remaining <= 0:

            print(
                "Capturing..."
            )

            show_frame(
                camera,
                frame,
                result,
                "Capturing..."
            )

            return result, frame

        # -----------------------------------------------------------
        # Display 2 -> 1
        # -----------------------------------------------------------

        current_second = int(
            np.ceil(
                remaining
            )
        )

        if current_second != last_second:

            print(
                f"Countdown: {current_second}"
            )

            last_second = (
                current_second
            )

        show_frame(
            camera,
            frame,
            result,
            "Hold your sign",
            current_second
        )


# =====================================================================
# CAPTURE VALID FRAME
# =====================================================================

def capture_prediction_frame(
    camera,
    initial_result,
    initial_frame
):
    """
    After countdown finishes, capture one valid frame.

    If the hand disappears immediately:
        attempt is cancelled.

    Returns:
        result, frame

    or:
        None, None
    """

    # ---------------------------------------------------------------
    # First check the frame from countdown
    # ---------------------------------------------------------------

    if has_hand(initial_result):

        return (
            initial_result,
            initial_frame
        )

    # ---------------------------------------------------------------
    # Otherwise wait briefly for another frame
    # ---------------------------------------------------------------

    start_time = time.monotonic()

    while (
        time.monotonic()
        -
        start_time
        <
        CAPTURE_WAIT_SECONDS
    ):

        success, frame = camera.read()

        if not success:

            return None, None

        result = camera.detect(
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            return "QUIT", None

        if key == ord("n"):

            return "SKIP", None

        # -----------------------------------------------------------
        # Valid hand
        # -----------------------------------------------------------

        if has_hand(result):

            return (
                result,
                frame
            )

        show_frame(
            camera,
            frame,
            result,
            "Waiting for hand..."
        )

    # ---------------------------------------------------------------
    # No hand
    # ---------------------------------------------------------------

    return None, None


# =====================================================================
# TEST ONE ATTEMPT
# =====================================================================

def get_attempt_prediction(
    camera,
    target_gesture,
    models,
    scaler,
    label_encoder
):
    """
    Perform one valid attempt.

    Returns:
        dictionary containing predictions

    Returns None if attempt is cancelled.
    """

    print(
        f"\n{'-' * 70}"
    )

    print(
        f"ATTEMPT"
    )

    print(
        f"Target: {target_gesture}"
    )

    # ---------------------------------------------------------------
    # Wait for hand + countdown
    # ---------------------------------------------------------------

    result, frame = (
        wait_for_hand_and_countdown(
            camera,
            target_gesture
        )
    )

    # ---------------------------------------------------------------
    # Quit
    # ---------------------------------------------------------------

    if result == "QUIT":

        return "QUIT"

    # ---------------------------------------------------------------
    # Skip
    # ---------------------------------------------------------------

    if result == "SKIP":

        return "SKIP"

    # ---------------------------------------------------------------
    # Countdown cancelled
    # ---------------------------------------------------------------

    if result is None:

        return None

    # ---------------------------------------------------------------
    # Capture valid frame
    # ---------------------------------------------------------------

    result, frame = (
        capture_prediction_frame(
            camera,
            result,
            frame
        )
    )

    if result == "QUIT":

        return "QUIT"

    if result == "SKIP":

        return "SKIP"

    if result is None:

        print(
            "Attempt cancelled."
        )

        print(
            "This attempt is NOT counted."
        )

        return None

    # ---------------------------------------------------------------
    # Extract features
    # ---------------------------------------------------------------

    features = extract_features(
        result
    )

    # ---------------------------------------------------------------
    # Scale
    # ---------------------------------------------------------------

    try:

        scaled_features = (
            scaler.transform(
                features
            )
        )

    except Exception as error:

        print(
            "Scaler Error:",
            error
        )

        return None

    # ===============================================================
    # TEST ALL THREE ALGORITHMS
    # ===============================================================

    predictions = {}

    for algorithm_name, model in models.items():

        gesture, confidence = (
            predict_algorithm(
                model,
                scaled_features,
                label_encoder
            )
        )

        correct = (
            gesture.upper()
            ==
            target_gesture.upper()
        )

        predictions[
            algorithm_name
        ] = {

            "prediction": gesture,

            "confidence": confidence,

            "correct": correct

        }

    # ---------------------------------------------------------------
    # Display result
    # ---------------------------------------------------------------

    print(
        "-" * 70
    )

    for algorithm_name in models:

        data = predictions[
            algorithm_name
        ]

        status = (
            "CORRECT"
            if data["correct"]
            else
            "WRONG"
        )

        print(
            f"{algorithm_name:<17}"
            f": {data['prediction']:<17}"
            f"{data['confidence']:>6.2f}% "
            f"{status}"
        )

    print(
        "-" * 70
    )

    # ---------------------------------------------------------------
    # Show result on camera
    # ---------------------------------------------------------------

    result_frame = frame.copy()

    try:

        result_frame = camera.draw(
            result_frame,
            result
        )

    except Exception:

        pass

    y = 40

    draw_text(
        result_frame,
        f"Target: {target_gesture}",
        (20, y),
        0.7,
        2
    )

    y += 35

    for algorithm_name in models:

        data = predictions[
            algorithm_name
        ]

        text = (
            f"{algorithm_name}: "
            f"{data['prediction']} "
            f"{data['confidence']:.1f}%"
        )

        draw_text(
            result_frame,
            text,
            (20, y),
            0.6,
            2
        )

        y += 30

    cv2.imshow(
        WINDOW_NAME,
        result_frame
    )

    cv2.waitKey(
        int(
            RESULT_DISPLAY_SECONDS
            *
            1000
        )
    )

    return predictions


# =====================================================================
# PRINT GESTURE RESULT
# =====================================================================

def print_gesture_results(
    gesture,
    statistics
):
    """
    Print results for one gesture.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{gesture} - REAL-TIME ALGORITHM RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"{'Algorithm':<17}"
        f"{'Correct':<10}"
        f"{'Wrong':<10}"
        f"{'Accuracy':<12}"
        f"{'Avg Conf.':<12}"
    )

    print(
        "-" * 70
    )

    for algorithm_name, data in (
        statistics.items()
    ):

        tests = data["tests"]

        correct = data["correct"]

        wrong = data["wrong"]

        if tests > 0:

            accuracy = (
                correct
                /
                tests
                *
                100
            )

            avg_confidence = (
                sum(
                    data["confidences"]
                )
                /
                len(
                    data["confidences"]
                )
            )

        else:

            accuracy = 0.0

            avg_confidence = 0.0

        print(
            f"{algorithm_name:<17}"
            f"{correct:<10}"
            f"{wrong:<10}"
            f"{accuracy:>7.2f}%    "
            f"{avg_confidence:>7.2f}%"
        )

    print(
        "=" * 70
    )


# =====================================================================
# SAVE CSV
# =====================================================================

def save_csv(
    rows
):
    """
    Save detailed attempt results.
    """

    fieldnames = [
        "Gesture",
        "Algorithm",
        "Attempt",
        "Prediction",
        "Confidence",
        "Correct"
    ]

    with open(
        OUTPUT_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )

    print(
        "\n✓ Results saved:"
    )

    print(
        OUTPUT_CSV_PATH
    )


# =====================================================================
# FINAL RESULTS
# =====================================================================

def print_final_results(
    final_statistics
):
    """
    Print final comparison.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "           FINAL REAL-TIME ALGORITHM COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        f"{'GESTURE':<15}"
        f"{'ALGORITHM':<18}"
        f"{'TESTS':<8}"
        f"{'CORRECT':<10}"
        f"{'WRONG':<8}"
        f"{'ACCURACY':<12}"
    )

    print(
        "-" * 80
    )

    overall = {}

    for gesture in TEST_GESTURES:

        for algorithm_name in (
            final_statistics[gesture]
        ):

            data = (
                final_statistics
                [gesture]
                [algorithm_name]
            )

            tests = data["tests"]

            correct = data["correct"]

            wrong = data["wrong"]

            if tests > 0:

                accuracy = (
                    correct
                    /
                    tests
                    *
                    100
                )

            else:

                accuracy = 0.0

            print(
                f"{gesture:<15}"
                f"{algorithm_name:<18}"
                f"{tests:<8}"
                f"{correct:<10}"
                f"{wrong:<8}"
                f"{accuracy:>7.2f}%"
            )

            if algorithm_name not in overall:

                overall[
                    algorithm_name
                ] = {

                    "tests": 0,

                    "correct": 0

                }

            overall[
                algorithm_name
            ]["tests"] += tests

            overall[
                algorithm_name
            ]["correct"] += correct

    # ===============================================================
    # OVERALL
    # ===============================================================

    print(
        "=" * 80
    )

    print(
        "\n"
        "OVERALL REAL-TIME ACCURACY"
    )

    print(
        "-" * 60
    )

    best_algorithm = None

    best_accuracy = -1.0

    for algorithm_name, data in (
        overall.items()
    ):

        tests = data["tests"]

        correct = data["correct"]

        if tests > 0:

            accuracy = (
                correct
                /
                tests
                *
                100
            )

        else:

            accuracy = 0.0

        print(
            f"{algorithm_name:<18}"
            f": {accuracy:.2f}%"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_algorithm = (
                algorithm_name
            )

    print(
        "\n"
        "BEST REAL-TIME ALGORITHM"
    )

    print(
        "-" * 60
    )

    print(
        f"Algorithm : {best_algorithm}"
    )

    print(
        f"Accuracy  : {best_accuracy:.2f}%"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "       REAL-TIME ALGORITHM COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        "\nAlgorithms:"
    )

    print(
        "1. SVM"
    )

    print(
        "2. Random Forest"
    )

    print(
        "3. KNN"
    )

    print(
        "\nTesting Gestures:"
    )

    for gesture in TEST_GESTURES:

        print(
            f" - {gesture}"
        )

    print(
        "\nEach valid attempt is evaluated by all three algorithms."
    )

    print(
        "No new data is collected."
    )

    print(
        "No model is retrained."
    )

    print(
        "Hand must be detected before countdown starts."
    )

    print(
        "Countdown: 2 -> 1"
    )

    print(
        "=" * 70
    )

    # ===============================================================
    # LOAD MODELS
    # ===============================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "LOADING THREE MACHINE LEARNING MODELS"
    )

    print(
        "=" * 60
    )

    models = {}

    models[
        "SVM"
    ] = load_model(
        SVM_MODEL_PATH,
        "SVM"
    )

    models[
        "Random Forest"
    ] = load_model(
        RANDOM_FOREST_MODEL_PATH,
        "Random Forest"
    )

    models[
        "KNN"
    ] = load_model(
        KNN_MODEL_PATH,
        "KNN"
    )

    # ===============================================================
    # LOAD SCALER
    # ===============================================================

    scaler = load_model(
        SCALER_PATH,
        "Scaler"
    )

    # ===============================================================
    # LOAD LABEL ENCODER
    # ===============================================================

    label_encoder = load_model(
        LABEL_ENCODER_PATH,
        "Label Encoder"
    )

    print(
        "\n✓ Three Algorithm Comparison Module Ready"
    )

    print(
        "=" * 60
    )

    # ===============================================================
    # CAMERA
    # ===============================================================

    camera = None

    csv_rows = []

    final_statistics = {}

    try:

        print(
            "\n✓ REAL-TIME TEST READY"
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Opening camera..."
        )

        print(
            "=" * 70
        )

        camera = CameraManager()

        # ===========================================================
        # GESTURE LOOP
        # ===========================================================

        quit_program = False

        for gesture in TEST_GESTURES:

            if quit_program:

                break

            # -------------------------------------------------------
            # Statistics for current gesture
            # -------------------------------------------------------

            statistics = {

                "SVM": {

                    "tests": 0,
                    "correct": 0,
                    "wrong": 0,
                    "confidences": []

                },

                "Random Forest": {

                    "tests": 0,
                    "correct": 0,
                    "wrong": 0,
                    "confidences": []

                },

                "KNN": {

                    "tests": 0,
                    "correct": 0,
                    "wrong": 0,
                    "confidences": []

                }

            }

            final_statistics[
                gesture
            ] = statistics

            # -------------------------------------------------------
            # Header
            # -------------------------------------------------------

            print(
                "\n"
                + "=" * 60
            )

            print(
                f"Testing Gesture: {gesture}"
            )

            print(
                f"Required Attempts: "
                f"{ATTEMPTS_PER_GESTURE}"
            )

            print(
                "Show the sign and hold it steady."
            )

            print(
                "Hand must be detected before countdown."
            )

            print(
                "Countdown: 2 -> 1"
            )

            print(
                "N = Skip    Q = Quit"
            )

            print(
                "=" * 60
            )

            # -------------------------------------------------------
            # ATTEMPT NUMBER
            #
            # IMPORTANT:
            # attempt_number only increases after a VALID attempt.
            # -------------------------------------------------------

            attempt_number = 0

            while (
                attempt_number
                <
                ATTEMPTS_PER_GESTURE
            ):

                print(
                    "\n"
                    + "-" * 70
                )

                print(
                    f"ATTEMPT "
                    f"{attempt_number + 1}/"
                    f"{ATTEMPTS_PER_GESTURE}"
                )

                print(
                    f"Target: {gesture}"
                )

                # ---------------------------------------------------
                # Run attempt
                # ---------------------------------------------------

                result = (
                    get_attempt_prediction(
                        camera,
                        gesture,
                        models,
                        scaler,
                        label_encoder
                    )
                )

                # ---------------------------------------------------
                # QUIT
                # ---------------------------------------------------

                if result == "QUIT":

                    quit_program = True

                    break

                # ---------------------------------------------------
                # SKIP
                # ---------------------------------------------------

                if result == "SKIP":

                    print(
                        "Gesture skipped."
                    )

                    break

                # ---------------------------------------------------
                # CANCELLED
                #
                # IMPORTANT:
                # Do NOT increase attempt number.
                # ---------------------------------------------------

                if result is None:

                    print(
                        "Waiting for a valid attempt..."
                    )

                    continue

                # ===================================================
                # VALID ATTEMPT
                # ===================================================

                attempt_number += 1

                # ---------------------------------------------------
                # Store statistics
                # ---------------------------------------------------

                for algorithm_name in models:

                    data = result[
                        algorithm_name
                    ]

                    statistics[
                        algorithm_name
                    ]["tests"] += 1

                    statistics[
                        algorithm_name
                    ]["confidences"].append(
                        data["confidence"]
                    )

                    if data["correct"]:

                        statistics[
                            algorithm_name
                        ]["correct"] += 1

                    else:

                        statistics[
                            algorithm_name
                        ]["wrong"] += 1

                    # ------------------------------------------------
                    # CSV
                    # ------------------------------------------------

                    csv_rows.append({

                        "Gesture":
                            gesture,

                        "Algorithm":
                            algorithm_name,

                        "Attempt":
                            attempt_number,

                        "Prediction":
                            data["prediction"],

                        "Confidence":
                            round(
                                data["confidence"],
                                2
                            ),

                        "Correct":
                            (
                                "YES"
                                if data["correct"]
                                else
                                "NO"
                            )

                    })

            # =======================================================
            # GESTURE RESULTS
            # =======================================================

            if not quit_program:

                print_gesture_results(
                    gesture,
                    statistics
                )

                print(
                    "\nPreparing next gesture..."
                )

                time.sleep(1)

        # ===========================================================
        # FINAL RESULTS
        # ===========================================================

        print_final_results(
            final_statistics
        )

        # ===========================================================
        # SAVE CSV
        # ===========================================================

        if csv_rows:

            save_csv(
                csv_rows
            )

        else:

            print(
                "\nNo valid attempts were recorded."
            )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "REAL-TIME ALGORITHM COMPARISON COMPLETED"
        )

        print(
            "=" * 70
        )

    except KeyboardInterrupt:

        print(
            "\n\nTest interrupted by user."
        )

    except Exception as error:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "ERROR"
        )

        print(
            "=" * 70
        )

        print(
            error
        )

    finally:

        # ===========================================================
        # CLOSE CAMERA
        # ===========================================================

        if camera is not None:

            try:

                camera.close()

            except Exception as error:

                print(
                    "Camera Close Error:",
                    error
                )


# =====================================================================
# RUN
# =====================================================================

if __name__ == "__main__":

    main()