"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
REAL-TIME / DATASET ACCURACY TEST
==========================================================
"""

import os
import csv
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "landmarks.csv"
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

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "accuracy_test_results.csv"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)


# ==========================================================
# HEADER
# ==========================================================

print("=" * 65)

print(
    "       SIGN LANGUAGE RECOGNITION SYSTEM"
)

print(
    "             ACCURACY TEST"
)

print("=" * 65)


# ==========================================================
# CHECK FILES
# ==========================================================

print("\nChecking required files...")

required_files = [
    DATASET_PATH,
    MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH
]

for file_path in required_files:

    if not os.path.exists(file_path):

        print(
            f"\nERROR: File not found:"
        )

        print(
            file_path
        )

        raise SystemExit(1)

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
# LOAD DATASET
# ==========================================================

print("\nLoading dataset...")

rows = []

with open(
    DATASET_PATH,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.reader(file)

    header = next(reader)

    for row in reader:

        if len(row) > 1:

            rows.append(row)


print(
    f"✓ Total samples: {len(rows)}"
)

print(
    f"✓ Total columns: {len(header)}"
)


# ==========================================================
# PREPARE FEATURES
# ==========================================================

X = []

y = []

for row in rows:

    try:

        features = [
            float(value)
            for value in row[:-1]
        ]

        label = row[-1].strip()

        X.append(
            features
        )

        y.append(
            label
        )

    except ValueError:

        continue


X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y
)


print(
    f"✓ Features: {X.shape[1]}"
)

print(
    f"✓ Valid samples: {len(X)}"
)


# ==========================================================
# ENCODE LABELS
# ==========================================================

y_encoded = label_encoder.transform(
    y
)


# ==========================================================
# FEATURE SCALING
# ==========================================================

print("\nApplying feature scaling...")

X_scaled = scaler.transform(
    X
)

print(
    "✓ Scaling completed"
)


# ==========================================================
# PREDICTION
# ==========================================================

print("\nRunning predictions...")

y_pred_encoded = model.predict(
    X_scaled
)

print(
    "✓ Predictions completed"
)


# ==========================================================
# DECODE PREDICTIONS
# ==========================================================

y_pred = label_encoder.inverse_transform(
    y_pred_encoded
)


# ==========================================================
# OVERALL ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y,
    y_pred
)

precision = precision_score(
    y,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    average="weighted",
    zero_division=0
)


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print("\n")

print("=" * 65)

print(
    "              ACCURACY RESULTS"
)

print("=" * 65)

print(
    f"\nOverall Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision        : {precision * 100:.2f}%"
)

print(
    f"Recall           : {recall * 100:.2f}%"
)

print(
    f"F1 Score         : {f1 * 100:.2f}%"
)


# ==========================================================
# CORRECT / WRONG
# ==========================================================

correct = np.sum(
    y == y_pred
)

wrong = np.sum(
    y != y_pred
)

total = len(y)


print("\n")

print(
    f"Total Samples    : {total}"
)

print(
    f"Correct          : {correct}"
)

print(
    f"Wrong            : {wrong}"
)


# ==========================================================
# PER GESTURE ACCURACY
# ==========================================================

print("\n")

print("=" * 65)

print(
    "           PER-GESTURE ACCURACY"
)

print("=" * 65)

gesture_results = []


gestures = label_encoder.classes_


for gesture in gestures:

    actual_mask = (
        y == gesture
    )

    total_gesture = np.sum(
        actual_mask
    )

    correct_gesture = np.sum(
        (
            y == gesture
        )
        &
        (
            y_pred == gesture
        )
    )

    wrong_gesture = (
        total_gesture
        -
        correct_gesture
    )

    if total_gesture > 0:

        gesture_accuracy = (
            correct_gesture
            /
            total_gesture
        ) * 100

    else:

        gesture_accuracy = 0.0

    print(
        f"{gesture:<15}"
        f" Total: {total_gesture:<5}"
        f" Correct: {correct_gesture:<5}"
        f" Wrong: {wrong_gesture:<5}"
        f" Accuracy: {gesture_accuracy:.2f}%"
    )

    gesture_results.append(
        [
            gesture,
            total_gesture,
            correct_gesture,
            wrong_gesture,
            gesture_accuracy
        ]
    )


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    y,
    y_pred,
    labels=gestures,
    zero_division=0
)

print("\n")

print("=" * 65)

print(
    "           CLASSIFICATION REPORT"
)

print("=" * 65)

print()

print(
    report
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y,
    y_pred,
    labels=gestures
)


# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# SAVE ACCURACY RESULTS
# ==========================================================

with open(
    RESULTS_PATH,
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
            "Total Samples",
            "Correct",
            "Wrong",
            "Accuracy (%)"
        ]
    )

    for result in gesture_results:

        writer.writerow(
            result
        )

    writer.writerow([])

    writer.writerow(
        [
            "OVERALL",
            total,
            correct,
            wrong,
            accuracy * 100
        ]
    )


# ==========================================================
# SAVE CLASSIFICATION REPORT
# ==========================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "SIGN LANGUAGE RECOGNITION SYSTEM\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 60
    )

    file.write("\n\n")

    file.write(
        f"Total Samples : {total}\n"
    )

    file.write(
        f"Correct       : {correct}\n"
    )

    file.write(
        f"Wrong         : {wrong}\n"
    )

    file.write(
        f"Accuracy      : {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision     : {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall        : {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score      : {f1 * 100:.2f}%\n"
    )

    file.write("\n\n")

    file.write(
        report
    )


# ==========================================================
# SAVE CONFUSION MATRIX
# ==========================================================

with open(
    CONFUSION_MATRIX_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(
        [
            "Actual / Predicted"
        ]
        +
        list(gestures)
    )

    for index, gesture in enumerate(gestures):

        writer.writerow(
            [
                gesture
            ]
            +
            list(
                cm[index]
            )
        )


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n")

print("=" * 65)

print(
    "              FINAL SUMMARY"
)

print("=" * 65)

print(
    f"\nTotal Samples : {total}"
)

print(
    f"Correct       : {correct}"
)

print(
    f"Wrong         : {wrong}"
)

print(
    f"Accuracy      : {accuracy * 100:.2f}%"
)

print(
    f"Precision     : {precision * 100:.2f}%"
)

print(
    f"Recall        : {recall * 100:.2f}%"
)

print(
    f"F1 Score      : {f1 * 100:.2f}%"
)

print("\n")

print(
    "Results saved:"
)

print(
    f"✓ {RESULTS_PATH}"
)

print(
    f"✓ {REPORT_PATH}"
)

print(
    f"✓ {CONFUSION_MATRIX_PATH}"
)

print("\n")

print("=" * 65)

print(
    "             ACCURACY TEST COMPLETED"
)

print("=" * 65)