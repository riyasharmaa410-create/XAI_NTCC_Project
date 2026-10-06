import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "wdbc.data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# 2. FEATURE NAMES
# ==========================================

feature_names = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]


# ==========================================
# 3. LOAD DATASET
# ==========================================

columns = ["id", "diagnosis"] + feature_names

data = pd.read_csv(
    DATA_PATH,
    header=None,
    names=columns
)

print("==========================================")
print("      XAI MODEL TRAINING")
print("==========================================")

print("\nDataset loaded successfully.")
print("Number of samples:", len(data))
print("Number of features:", len(feature_names))


# ==========================================
# 4. PREPARE DATA
# ==========================================

X = data[feature_names]

# B = Benign = 0
# M = Malignant = 1
y = data["diagnosis"].map({
    "B": 0,
    "M": 1
})


# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# ==========================================
# 6. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Model training completed.")


# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. CALCULATE PERFORMANCE
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("        MODEL PERFORMANCE")
print("==========================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 11. SAVE MODEL
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "xai_random_forest_model.pkl"
)

joblib.dump(model, model_path)


# ==========================================
# 12. SAVE TEST DATA
# ==========================================

X_test.to_csv(
    os.path.join(MODEL_DIR, "X_test.csv"),
    index=False
)

y_test.to_csv(
    os.path.join(MODEL_DIR, "y_test.csv"),
    index=False
)

joblib.dump(
    feature_names,
    os.path.join(MODEL_DIR, "feature_names.pkl")
)


print("\n==========================================")
print("        FILES SAVED SUCCESSFULLY")
print("==========================================")

print("Model       :", model_path)
print("Test data   : model/X_test.csv")
print("Test labels : model/y_test.csv")
print("Features    : model/feature_names.pkl")

print("\nTraining completed successfully!")