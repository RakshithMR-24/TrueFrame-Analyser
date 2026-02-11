import numpy as np

# ---------------- SCORE MAPPERS ----------------
def map_metadata_score(meta_label, meta_conf):
    """
    Maps metadata presence to a score.
    Metadata is optional; ML model does not use it.
    """
    if meta_label == "GOOD METADATA":
        return meta_conf
    elif meta_label == "PARTIAL METADATA":
        return 0.5 * meta_conf
    else:
        return 0.0


def map_forensic_score(forensic_label, forensic_conf):
    """
    Maps forensic analysis result to a score.
    """
    if forensic_label == "AUTHENTIC (Camera-consistent)":
        return forensic_conf
    elif forensic_label == "EDITED BUT REAL":
        return 0.6 * forensic_conf
    else:  # AI / suspicious
        return 0.0


def map_ml_score(ml_label, ml_conf):
    """
    Maps ML prediction result to a score.
    ML model is forensic-only.
    """
    if ml_label == "Real":
        return ml_conf
    elif ml_label == "Edited":
        return 0.5 * ml_conf
    else:  # AI
        return 0.0


# ---------------- FUSION LOGIC ----------------
def final_verdict_fusion(metadata_result, forensic_result, ml_result):
    """
    Combines metadata, forensic, and ML scores into a final verdict.
    Metadata is included but has reduced weight.
    """
    meta_label, meta_conf = metadata_result
    forensic_label, forensic_conf = forensic_result
    ml_label = ml_result["label"]
    ml_conf = ml_result["confidence"]

    # Convert to normalized scores
    metadata_score = map_metadata_score(meta_label, meta_conf)
    forensic_score = map_forensic_score(forensic_label, forensic_conf)
    ml_score = map_ml_score(ml_label, ml_conf)

    # Weighted fusion (LOW DEPENDENCY)
    final_score = (
            0.45 * metadata_score +
            0.35 * forensic_score +
            0.20 * ml_score
    )

    final_score = float(np.clip(final_score, 0.0, 1.0))

    # Final verdict thresholds
    if final_score >= 0.65:
        verdict = "REAL IMAGE"
    elif final_score >= 0.45:
        verdict = "REAL BUT EDITED"
    else:
        verdict = "AI-GENERATED"

    return verdict, round(final_score, 2)
