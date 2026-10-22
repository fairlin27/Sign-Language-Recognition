# ==========================================
# SIGN LANGUAGE MODEL TRAINING
# Part 1 - Import & Dataset Preparation
# ==========================================

import csv
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("      SIGN LANGUAGE MODEL TRAINING")
print("=" * 60)

# ==========================================
# Load Dataset
# ==========================================

dataset_path = "dataset/landmarks.csv"

rows = []

with open(dataset_path, "r", newline="", encoding="utf-8") as file:

    reader = csv.reader(file)

    header = next(reader)

    for row in reader:
        rows.append(row)

print("\nDataset Loaded Successfully")
print("Total Samples :", len(rows))
print("Total Columns :", len(header))

# ==========================================
# Gesture Count
# ==========================================

gesture_count = {}

for row in rows:

    label = row[-1]

    if label not in gesture_count:
        gesture_count[label] = 0

    gesture_count[label] += 1

print("\nGesture Count\n")

for gesture, count in sorted(gesture_count.items()):
    print(f"{gesture:<15} : {count}")

# ==========================================
# Features and Labels
# ==========================================

X = []
y = []

for row in rows:

    features = [float(value) for value in row[:-1]]

    label = row[-1]

    X.append(features)
    y.append(label)

X = np.array(X, dtype=np.float32)

print("\nTotal Features :", X.shape[1])

# ==========================================
# Label Encoding
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClasses")

for i, gesture in enumerate(label_encoder.classes_):

    print(i, ":", gesture)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded

)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed")

print("=" * 60)
print("Part 1 Completed Successfully")
print("=" * 60)


# ==========================================
# Part 2 - Train Machine Learning Models
# ==========================================

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

print("\n" + "=" * 60)
print("TRAINING MACHINE LEARNING MODELS")
print("=" * 60)

# ==========================================
# SVM
# ==========================================

print("\nTraining SVM...")

svm_model = SVC(

    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42

)

svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)

print("SVM Training Completed")


# ==========================================
# Random Forest
# ==========================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(

    n_estimators=200,
    random_state=42

)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Random Forest Training Completed")


# ==========================================
# KNN
# ==========================================

print("\nTraining KNN...")

knn_model = KNeighborsClassifier(

    n_neighbors=5

)

knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)

print("KNN Training Completed")

print("\nAll Models Trained Successfully")

print("=" * 60)
print("Part 2 Completed Successfully")
print("=" * 60)


# ==========================================
# Part 3 - Model Evaluation
# ==========================================

from sklearn.metrics import (

    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score

)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


# ==========================================
# Accuracy
# ==========================================

svm_acc = accuracy_score(y_test, svm_pred)

rf_acc = accuracy_score(y_test, rf_pred)

knn_acc = accuracy_score(y_test, knn_pred)

print("\nAccuracy")
print("-" * 40)

print(f"SVM           : {svm_acc*100:.2f}%")
print(f"Random Forest : {rf_acc*100:.2f}%")
print(f"KNN           : {knn_acc*100:.2f}%")


# ==========================================
# Precision
# ==========================================

svm_precision = precision_score(

    y_test,
    svm_pred,
    average="weighted",
    zero_division=0

)

rf_precision = precision_score(

    y_test,
    rf_pred,
    average="weighted",
    zero_division=0

)

knn_precision = precision_score(

    y_test,
    knn_pred,
    average="weighted",
    zero_division=0

)

print("\nPrecision")
print("-" * 40)

print(f"SVM           : {svm_precision:.4f}")
print(f"Random Forest : {rf_precision:.4f}")
print(f"KNN           : {knn_precision:.4f}")


# ==========================================
# Recall
# ==========================================

svm_recall = recall_score(

    y_test,
    svm_pred,
    average="weighted",
    zero_division=0

)

rf_recall = recall_score(

    y_test,
    rf_pred,
    average="weighted",
    zero_division=0

)

knn_recall = recall_score(

    y_test,
    knn_pred,
    average="weighted",
    zero_division=0

)

print("\nRecall")
print("-" * 40)

print(f"SVM           : {svm_recall:.4f}")
print(f"Random Forest : {rf_recall:.4f}")
print(f"KNN           : {knn_recall:.4f}")


# ==========================================
# F1 Score
# ==========================================

svm_f1 = f1_score(

    y_test,
    svm_pred,
    average="weighted",
    zero_division=0

)

rf_f1 = f1_score(

    y_test,
    rf_pred,
    average="weighted",
    zero_division=0

)

knn_f1 = f1_score(

    y_test,
    knn_pred,
    average="weighted",
    zero_division=0

)

print("\nF1 Score")
print("-" * 40)

print(f"SVM           : {svm_f1:.4f}")
print(f"Random Forest : {rf_f1:.4f}")
print(f"KNN           : {knn_f1:.4f}")


# ==========================================
# SVM Classification Report
# ==========================================

print("\n" + "=" * 60)
print("SVM CLASSIFICATION REPORT")
print("=" * 60)

print(

    classification_report(

        y_test,
        svm_pred,
        target_names=label_encoder.classes_,
        zero_division=0

    )

)


# ==========================================
# Random Forest Classification Report
# ==========================================

print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 60)

print(

    classification_report(

        y_test,
        rf_pred,
        target_names=label_encoder.classes_,
        zero_division=0

    )

)


# ==========================================
# KNN Classification Report
# ==========================================

print("\n" + "=" * 60)
print("KNN CLASSIFICATION REPORT")
print("=" * 60)

print(

    classification_report(

        y_test,
        knn_pred,
        target_names=label_encoder.classes_,
        zero_division=0

    )

)


# ==========================================
# Confusion Matrices
# ==========================================

print("\n" + "=" * 60)
print("SVM CONFUSION MATRIX")
print("=" * 60)

print(confusion_matrix(y_test, svm_pred))


print("\n" + "=" * 60)
print("RANDOM FOREST CONFUSION MATRIX")
print("=" * 60)

print(confusion_matrix(y_test, rf_pred))


print("\n" + "=" * 60)
print("KNN CONFUSION MATRIX")
print("=" * 60)

print(confusion_matrix(y_test, knn_pred))


print("\n" + "=" * 60)
print("Part 3 Completed Successfully")
print("=" * 60)


# ==========================================
# Part 4 - Best Model Selection
#          & Save ALL Models
# ==========================================

print("\n" + "=" * 60)
print("BEST MODEL SELECTION")
print("=" * 60)


# ==========================================
# Compare Accuracy
# ==========================================

results = {

    "SVM": svm_acc,
    "Random Forest": rf_acc,
    "KNN": knn_acc

}

print("\nAlgorithm Comparison")
print("-" * 40)

for name, acc in results.items():

    print(f"{name:<20} : {acc*100:.2f}%")


# ==========================================
# Best Algorithm
# ==========================================

best_algorithm = max(

    results,
    key=results.get

)

print("\nBest Algorithm :", best_algorithm)

print(

    "Accuracy       : {:.2f}%".format(

        results[best_algorithm] * 100

    )

)


# ==========================================
# Select Best Model
# ==========================================

if best_algorithm == "SVM":

    best_model = svm_model

elif best_algorithm == "Random Forest":

    best_model = rf_model

else:

    best_model = knn_model


# ==========================================
# Create Models Folder
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# SAVE BEST MODEL
# ==========================================

joblib.dump(

    best_model,
    "models/gesture_model.pkl"

)


# ==========================================
# SAVE INDIVIDUAL MODELS
# ==========================================

joblib.dump(

    svm_model,
    "models/svm_model.pkl"

)

joblib.dump(

    rf_model,
    "models/random_forest_model.pkl"

)

joblib.dump(

    knn_model,
    "models/knn_model.pkl"

)


# ==========================================
# SAVE SCALER & LABEL ENCODER
# ==========================================

joblib.dump(

    scaler,
    "models/scaler.pkl"

)

joblib.dump(

    label_encoder,
    "models/label_encoder.pkl"

)


print("\nFiles Saved Successfully")

print("models/gesture_model.pkl")
print("models/svm_model.pkl")
print("models/random_forest_model.pkl")
print("models/knn_model.pkl")
print("models/scaler.pkl")
print("models/label_encoder.pkl")


# ==========================================
# Final Training Summary
# ==========================================

print("\n" + "=" * 60)
print("FINAL TRAINING SUMMARY")
print("=" * 60)

print("Dataset Samples :", len(rows))

print(

    "Total Gestures  :",
    len(label_encoder.classes_)

)

print(

    "Training Samples:",
    len(X_train)

)

print(

    "Testing Samples :",
    len(X_test)

)

print("\nFinal Accuracy")

print(f"SVM           : {svm_acc*100:.2f}%")

print(f"Random Forest : {rf_acc*100:.2f}%")

print(f"KNN           : {knn_acc*100:.2f}%")

print("\nSelected Model :", best_algorithm)

print("\nTraining Completed Successfully")

print("=" * 60)


# ==========================================
# Save Results CSV & Charts
# ==========================================

os.makedirs("output", exist_ok=True)


# ==========================================
# Save CSV
# ==========================================

with open(

    "output/results.csv",
    "w",
    newline="",
    encoding="utf-8"

) as file:

    writer = csv.writer(file)

    writer.writerow([

        "Algorithm",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"

    ])

    writer.writerow([

        "SVM",
        svm_acc,
        svm_precision,
        svm_recall,
        svm_f1

    ])

    writer.writerow([

        "Random Forest",
        rf_acc,
        rf_precision,
        rf_recall,
        rf_f1

    ])

    writer.writerow([

        "KNN",
        knn_acc,
        knn_precision,
        knn_recall,
        knn_f1

    ])


print("\nresults.csv Saved Successfully")


# ==========================================
# Algorithm Names
# ==========================================

algorithms = [

    "SVM",
    "Random Forest",
    "KNN"

]


# ==========================================
# Accuracy Chart
# ==========================================

plt.figure(figsize=(6, 4))

plt.bar(

    algorithms,

    [

        svm_acc * 100,
        rf_acc * 100,
        knn_acc * 100

    ]

)

plt.title("Algorithm Accuracy Comparison")

plt.ylabel("Accuracy (%)")

plt.savefig(

    "output/accuracy_chart.png"

)

plt.close()


# ==========================================
# Precision Chart
# ==========================================

plt.figure(figsize=(6, 4))

plt.bar(

    algorithms,

    [

        svm_precision,
        rf_precision,
        knn_precision

    ]

)

plt.title("Precision Comparison")

plt.savefig(

    "output/precision_chart.png"

)

plt.close()


# ==========================================
# Recall Chart
# ==========================================

plt.figure(figsize=(6, 4))

plt.bar(

    algorithms,

    [

        svm_recall,
        rf_recall,
        knn_recall

    ]

)

plt.title("Recall Comparison")

plt.savefig(

    "output/recall_chart.png"

)

plt.close()


# ==========================================
# F1 Score Chart
# ==========================================

plt.figure(figsize=(6, 4))

plt.bar(

    algorithms,

    [

        svm_f1,
        rf_f1,
        knn_f1

    ]

)

plt.title("F1 Score Comparison")

plt.savefig(

    "output/f1_chart.png"

)

plt.close()


# ==========================================
# Final Message
# ==========================================

print("Charts Saved Successfully")

print("\nSaved Model Files:")

print("✓ models/svm_model.pkl")
print("✓ models/random_forest_model.pkl")
print("✓ models/knn_model.pkl")
print("✓ models/gesture_model.pkl")

print("\nPROJECT READY FOR REAL-TIME PREDICTION")

print("=" * 60)