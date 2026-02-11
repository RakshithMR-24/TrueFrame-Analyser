from features import extract_advanced_forensic_features
from model import normalize_forensics, load_model, predict_image

def ml_predict(image_path):
    """
    Predicts image authenticity using only forensic features.
    Metadata features are removed for ML prediction.
    """
    model = load_model("trained_model.pkl")

    forensic = extract_advanced_forensic_features(image_path)
    if forensic is None:
        return {
            "label": "Unknown",
            "confidence": 0.0
        }

    # ---------------- FORENSIC-ONLY FEATURE VECTOR ----------------
    feature_vector = normalize_forensics(forensic)

    label, confidence = predict_image(model, feature_vector)

    return {
        "label": label,
        "confidence": confidence
    }
