import os
import numpy as np
from features import extract_advanced_forensic_features
from model import normalize_forensics, train_model, save_model

DATASET_DIR = "dataset"

CLASS_MAP = {
    "real": 0,
    "edited": 1,
    "ai": 2
}

X, y = [], []

for class_name, label in CLASS_MAP.items():
    folder = os.path.join(DATASET_DIR, class_name)

    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)

        forensic = extract_advanced_forensic_features(img_path)
        if forensic is None:
            continue

        feature_vector = normalize_forensics(forensic)

        # -------- DEBUG PRINT --------
        #print(f"\nImage: {img}")
        #for k, v in forensic.items():
         #   print(f"{k:22} : {v:.4f}")
        #print("Normalized Vector:", feature_vector)

        X.append(feature_vector)
        y.append(label)

X = np.array(X)
y = np.array(y)


# -------- DATA SUMMARY --------
print("\n========== TRAINING DATA SUMMARY ==========")
print(f"Total samples      : {len(X)}")
print(f"Feature dimension  : {X.shape[1]} (EXPECTED = 8)")
print("Class distribution :")
print("  Real   :", np.sum(y == 0))
print("  Edited :", np.sum(y == 1))
print("  AI     :", np.sum(y == 2))

# -------- TRAIN MODEL --------
model = train_model(X, y)

np.save("X_train.npy", X)
np.save("y_train.npy", y)

# -------- MODEL INFO --------
print("\n========== MODEL DETAILS ==========")
print("Model type       :", type(model).__name__)
print("Number of trees  :", model.n_estimators)
print("Max depth        :", model.max_depth)
print("Classes learned  :", model.classes_)

# -------- SAVE MODEL --------
save_model(model)

print("\n ML model trained and saved successfully")
