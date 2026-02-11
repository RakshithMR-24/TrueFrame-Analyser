from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

FEATURE_ORDER = [
    "noise", "edge", "sharpness", "jpeg",
    "cfa", "noise_inconsistency",
    "clipping", "entropy"
]

# ---------------- FORENSIC NORMALIZATION ----------------
def normalize_forensics(features: dict) -> list:
    """
    Normalizes the 8 forensic features for ML input.
    """
    for key in FEATURE_ORDER:
        if key not in features:
            raise ValueError(f"Missing feature: {key}")

    return [
        features["noise"] / 30.0,
        features["edge"] * 5.0,
        np.log1p(features["sharpness"]) / 8.0,
        features["jpeg"] / 20.0,
        features["cfa"] * 10.0,
        features["noise_inconsistency"] * 10.0,
        features["clipping"] * 10.0,
        features["entropy"] / 8.0
    ]

# ---------------- ML MODEL ----------------
def train_model(X, y):
    """
    Trains a RandomForestClassifier using forensic-only features.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y)
    return model

def predict_image(model, feature_vector):
    """
    Predicts the class label and confidence for a single feature vector.
    Assumes feature_vector is forensic-only (8 features).
    """
    probs = model.predict_proba([feature_vector])[0]
    labels = ["Real", "Edited", "AI"]
    idx = int(np.argmax(probs))
    return labels[idx], round(float(probs[idx]), 2)

# ---------------- MODEL IO ----------------
def save_model(model, path="trained_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path="trained_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
