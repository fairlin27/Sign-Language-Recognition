# ============================================================
# SIGN LANGUAGE RECOGNITION SYSTEM
# REAL-TIME ALGORITHM COMPARISON
#
# SVM vs Random Forest vs KNN
# ============================================================

import os
import csv
import time
import joblib
import cv2
import numpy as np

from collections import deque

from src.camera import Camera
from src.predictor import Prediction


# ============================================================
# SETTINGS
# ============================================================

GESTURES_TO_TEST = [
    "HELP",
    "YOU",
    "WELCOME"
]

ATTEMPTS_PER_GESTURE = 10

STABLE_FRAMES = 8

MODEL_DIR = "models"
OUTPUT_DIR = "output"


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("       SIGN LANGUAGE RECOGNITION SYSTEM")
print("          REAL-TIME ALGORITHM COMPARISON")
print("=" * 70)

print("\nChecking model files...")

svm_path = os.path.join(MODEL_DIR, "svm_model.pkl")
rf_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
knn_path = os.path.join(MODEL_DIR, "knn_model.pkl")

scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

required_files = [
    svm_path,
    rf_path,
    knn_path,
    scaler_path,
    encoder_path
]

for file_path in required_files:

    if not os.path.exists(file_path):

        print("\nERROR: Required file not found:")
        print(file_path)

        raise SystemExit


print("✓ All required model files found")


print("\nLoading models...")

svm_model = joblib.load(svm_path)

rf_model = joblib.load(rf_path)

knn_model = joblib.load(knn_path)

scaler = joblib.load(scaler_path)

label_encoder = joblib.load(encoder_path)


print("✓ SVM model loaded")
print("✓ Random Forest model loaded")
print("✓ KNN model loaded")
print("✓ Scaler loaded")
print("✓ Label encoder loaded")


# ============================================================
# MODEL DICTIONARY
# ============================================================

models = {

    "SVM": svm_model,

    "Random Forest": rf_model,

    "KNN": knn_model
}


# ============================================================
# RESULTS STORAGE
# ============================================================

results = {

    "SVM": {
        "correct": 0,
        "wrong": 0,
        "tests": 0
    },

    "Random Forest": {
        "correct": 0,
        "wrong": 0,
        "tests": 0
    },

    "KNN": {
        "correct": 0,
        "wrong": 0,
        "tests": 0
    }
}


gesture_results = {

    "SVM": {},

    "Random Forest": {},

    "KNN": {}
}


for model_name in models:

    for gesture in GESTURES_TO_TEST:

        gesture_results[model_name][gesture] = {

            "correct": 0,

            "wrong": 0,

            "tests": 0
        }


# ============================================================
# CAMERA
# ============================================================

print("\n" + "=" * 70)
print("OPENING CAMERA")
print("=" * 70)

camera = Camera()

print("Loading Camera...")

camera.start()

print("✓ Camera Ready")


# ============================================================
# PREDICTION MODULE
# ============================================================

prediction = Prediction(

    camera=camera,

    model=svm_model,

    scaler=scaler,

    label_encoder=label_encoder,

    stable_frames=STABLE_FRAMES
)

print("✓ Prediction Module Ready")


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_landmarks():

    """
    Get the current hand landmark features
    from the prediction module.
    """

    try:

        if hasattr(prediction, "get_features"):

            return prediction.get_features()

        if hasattr(prediction, "extract_features"):

            return prediction.extract_features()

    except Exception:

        pass

    return None


# ============================================================
# MAIN TESTING
# ============================================================

print("\n" + "=" * 70)
print("REAL-TIME TESTING")
print("=" * 70)

print("\nGestures selected:")

for gesture in GESTURES_TO_TEST:

    print("→", gesture)

print("\nAttempts per gesture :", ATTEMPTS_PER_GESTURE)

print("\nIMPORTANT:")
print("Show the gesture clearly and hold it steady.")
print("Press Q to quit.")
print("=" * 70)


try:

    for gesture in GESTURES_TO_TEST:

        print("\n\n" + "=" * 70)

        print("TARGET GESTURE :", gesture)

        print("=" * 70)

        print(
            f"\nTesting {gesture} "
            f"({ATTEMPTS_PER_GESTURE} attempts)"
        )

        print("Show the sign and hold it steady.")

        time.sleep(2)


        attempt = 0

        stable_predictions = deque(
            maxlen=STABLE_FRAMES
        )


        while attempt < ATTEMPTS_PER_GESTURE:

            # ------------------------------------------------
            # READ CAMERA FRAME
            # ------------------------------------------------

            frame = camera.read()

            if frame is None:

                continue


            # ------------------------------------------------
            # DETECT / EXTRACT FEATURES
            # ------------------------------------------------

            try:

                features = prediction.extract_features(frame)

            except Exception:

                features = None


            # ------------------------------------------------
            # IF NO HAND
            # ------------------------------------------------

            if features is None:

                cv2.imshow(
                    "Real-Time Algorithm Comparison",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    raise KeyboardInterrupt

                continue


            features = np.asarray(
                features,
                dtype=np.float32
            ).reshape(1, -1)


            # ------------------------------------------------
            # SCALE FEATURES
            # ------------------------------------------------

            features_scaled = scaler.transform(
                features
            )


            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            predictions = {}

            confidences = {}


            # =================================================
            # SVM
            # =================================================

            svm_prediction = svm_model.predict(
                features_scaled
            )[0]

            svm_label = label_encoder.inverse_transform(
                [svm_prediction]
            )[0]

            predictions["SVM"] = svm_label


            if hasattr(
                svm_model,
                "predict_proba"
            ):

                svm_probability = svm_model.predict_proba(
                    features_scaled
                )[0]

                svm_confidence = (
                    np.max(svm_probability) * 100
                )

            else:

                svm_confidence = 0.0


            confidences["SVM"] = svm_confidence


            # =================================================
            # RANDOM FOREST
            # =================================================

            rf_prediction = rf_model.predict(
                features
            )[0]

            rf_label = label_encoder.inverse_transform(
                [rf_prediction]
            )[0]

            predictions["Random Forest"] = rf_label


            if hasattr(
                rf_model,
                "predict_proba"
            ):

                rf_probability = rf_model.predict_proba(
                    features
                )[0]

                rf_confidence = (
                    np.max(rf_probability) * 100
                )

            else:

                rf_confidence = 0.0


            confidences["Random Forest"] = rf_confidence


            # =================================================
            # KNN
            # =================================================

            knn_prediction = knn_model.predict(
                features_scaled
            )[0]

            knn_label = label_encoder.inverse_transform(
                [knn_prediction]
            )[0]

            predictions["KNN"] = knn_label


            if hasattr(
                knn_model,
                "predict_proba"
            ):

                knn_probability = knn_model.predict_proba(
                    features_scaled
                )[0]

                knn_confidence = (
                    np.max(knn_probability) * 100
                )

            else:

                knn_confidence = 0.0


            confidences["KNN"] = knn_confidence


            # ------------------------------------------------
            # DISPLAY PREDICTIONS
            # ------------------------------------------------

            display_text = [

                f"SVM: {svm_label} "
                f"({svm_confidence:.1f}%)",

                f"Random Forest: {rf_label} "
                f"({rf_confidence:.1f}%)",

                f"KNN: {knn_label} "
                f"({knn_confidence:.1f}%)"
            ]


            y_position = 40


            for text in display_text:

                cv2.putText(

                    frame,

                    text,

                    (20, y_position),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (0, 255, 0),

                    2
                )

                y_position += 35


            cv2.putText(

                frame,

                f"TARGET: {gesture}",

                (20, y_position + 20),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.75,

                (0, 255, 255),

                2
            )


            cv2.imshow(
                "Real-Time Algorithm Comparison",
                frame
            )


            # ------------------------------------------------
            # AUTOMATIC RECORD
            # ------------------------------------------------

            # Use SVM prediction stability
            # to decide when to record.

            stable_predictions.append(
                svm_label
            )


            if len(stable_predictions) < STABLE_FRAMES:

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    raise KeyboardInterrupt

                continue


            # ------------------------------------------------
            # STABLE PREDICTION
            # ------------------------------------------------

            if len(
                set(stable_predictions)
            ) == 1:

                attempt += 1

                print("\n")
                print("=" * 60)

                print(
                    f"AUTO RECORD "
                    f"{attempt}/{ATTEMPTS_PER_GESTURE}"
                )

                print(
                    f"Target : {gesture}"
                )


                # =================================================
                # RECORD EACH ALGORITHM
                # =================================================

                for model_name in models:

                    predicted_label = predictions[
                        model_name
                    ]

                    confidence = confidences[
                        model_name
                    ]


                    correct = (
                        predicted_label == gesture
                    )


                    results[
                        model_name
                    ]["tests"] += 1


                    gesture_results[
                        model_name
                    ][gesture]["tests"] += 1


                    if correct:

                        results[
                            model_name
                        ]["correct"] += 1

                        gesture_results[
                            model_name
                        ][gesture]["correct"] += 1

                    else:

                        results[
                            model_name
                        ]["wrong"] += 1

                        gesture_results[
                            model_name
                        ][gesture]["wrong"] += 1


                    print(
                        f"{model_name:<16} : "
                        f"{predicted_label:<15} "
                        f"{confidence:6.2f}% "
                        f"{'CORRECT' if correct else 'WRONG'}"
                    )


                print("=" * 60)


                # Clear stability buffer
                stable_predictions.clear()


                # Small delay
                time.sleep(0.8)


            # ------------------------------------------------
            # KEYBOARD
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF


            if key == ord("q"):

                raise KeyboardInterrupt


except KeyboardInterrupt:

    print("\n\nTest stopped by user.")


finally:

    # ========================================================
    # CLOSE CAMERA
    # ========================================================

    try:

        camera.stop()

    except Exception:

        pass


    cv2.destroyAllWindows()


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n\n")

print("=" * 70)

print("       REAL-TIME ALGORITHM COMPARISON RESULTS")

print("=" * 70)


# ============================================================
# MODEL RESULTS
# ============================================================

for model_name in models:

    total = results[
        model_name
    ]["tests"]

    correct = results[
        model_name
    ]["correct"]

    wrong = results[
        model_name
    ]["wrong"]


    if total > 0:

        accuracy = (
            correct / total
        ) * 100

    else:

        accuracy = 0


    print("\n" + model_name)

    print("-" * 50)

    print(
        f"Total Tests : {total}"
    )

    print(
        f"Correct     : {correct}"
    )

    print(
        f"Wrong       : {wrong}"
    )

    print(
        f"Accuracy    : {accuracy:.2f}%"
    )


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

print("\n\n")

print("=" * 70)

print("FINAL REAL-TIME ALGORITHM COMPARISON")

print("=" * 70)

print(
    f"{'ALGORITHM':<20}"
    f"{'TESTS':<10}"
    f"{'CORRECT':<10}"
    f"{'WRONG':<10}"
    f"{'ACCURACY':<10}"
)

print("-" * 70)


comparison = {}


for model_name in models:

    total = results[
        model_name
    ]["tests"]

    correct = results[
        model_name
    ]["correct"]

    wrong = results[
        model_name
    ]["wrong"]


    if total > 0:

        accuracy = (
            correct / total
        ) * 100

    else:

        accuracy = 0


    comparison[
        model_name
    ] = accuracy


    print(
        f"{model_name:<20}"
        f"{total:<10}"
        f"{correct:<10}"
        f"{wrong:<10}"
        f"{accuracy:.2f}%"
    )


# ============================================================
# BEST REAL-TIME MODEL
# ============================================================

if comparison:

    best_model = max(
        comparison,
        key=comparison.get
    )

    best_accuracy = comparison[
        best_model
    ]

else:

    best_model = "N/A"

    best_accuracy = 0


print("\n")

print("=" * 70)

print("BEST REAL-TIME MODEL")

print("=" * 70)

print(
    f"Algorithm : {best_model}"
)

print(
    f"Accuracy  : {best_accuracy:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


csv_path = os.path.join(
    OUTPUT_DIR,
    "realtime_algorithm_comparison.csv"
)


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)


    writer.writerow([

        "Algorithm",

        "Gesture",

        "Tests",

        "Correct",

        "Wrong",

        "Accuracy"
    ])


    for model_name in models:

        for gesture in GESTURES_TO_TEST:

            data = gesture_results[
                model_name
            ][gesture]


            tests = data["tests"]

            correct = data["correct"]

            wrong = data["wrong"]


            if tests > 0:

                accuracy = (
                    correct / tests
                ) * 100

            else:

                accuracy = 0


            writer.writerow([

                model_name,

                gesture,

                tests,

                correct,

                wrong,

                f"{accuracy:.2f}"
            ])


print("\n✓ Results saved:")

print(csv_path)


# ============================================================
# COMPLETED
# ============================================================

print("\n")

print("=" * 70)

print(
    "REAL-TIME ALGORITHM COMPARISON COMPLETED"
)

print("=" * 70)