# ---------------- METADATA FEATURES ----------------
from PIL import Image
from PIL.ExifTags import TAGS

# EXIF → feature mapping
EXIF_TO_FEATURE = {
    "Make": "camera_make",
    "Model": "camera_model",
    "LensModel": "lens_model",
    "DateTimeOriginal": "datetime_original",
    "GPSInfo": "gps_info",
    "ExposureTime": "exposure_time",
    "FNumber": "f_number",
    "ISOSpeedRatings": "iso_speed",
    "FocalLength": "focal_length",
    "Flash": "flash",
    "WhiteBalance": "white_balance",
    "MeteringMode": "metering_mode",
    "Orientation": "orientation",
    "ResolutionUnit": "resolution_unit",
    "XResolution": "x_resolution",
    "YResolution": "y_resolution",
    "Compression": "compression",
    "Software": "software",
    "Artist": "artist",
    "Copyright": "copyright",
    "ExifVersion": "exif_version"
}

def extract_metadata_features(image_path):
    """
    Extracts EXIF metadata presence as binary flags.
    Returns: dict(feature_name -> 0/1)
    """

    # Initialize all features to 0
    features = {v: 0 for v in EXIF_TO_FEATURE.values()}

    try:
        img = Image.open(image_path)
        exif = img._getexif()

        if not exif:
            return features

        for tag_id in exif:
            tag = TAGS.get(tag_id, tag_id)

            if tag in EXIF_TO_FEATURE:
                feature_name = EXIF_TO_FEATURE[tag]
                features[feature_name] = 1

    except Exception as e:
        print(f"[WARN] EXIF read failed for {image_path}: {e}")

    return features


def print_metadata_summary(image_path):
    features = extract_metadata_features(image_path)
    present = sum(features.values())
    total = len(features)

    print(f"{image_path} → EXIF Present: {present}/{total} ({present/total:.0%})")

def metadata_presence_report(metadata: dict) -> dict:
    """
    Returns feature-wise metadata presence (True / False)
    """
    return {
        feature_name: bool(value)
        for feature_name, value in metadata.items()
    }



# ---------------- ADVANCED FORENSIC FEATURES ----------------
import cv2
import numpy as np

def _normalize(val, low, high):
    """Maps value to 0–1 based on expected real-image range"""
    return float(np.clip((val - low) / (high - low), 0, 1))


def extract_advanced_forensic_features(image_path):
    """
    Extracts enhanced forensic features for authenticity detection
    Returns a dictionary (for interpretability)
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape

    #  Global noise level
    noise_std = np.std(img)

    # Edge density
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.mean(edges > 0)

    #  Sharpness (sensor vs AI smoothing)
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()

    #  JPEG block artifact strength
    jpeg_blocks = np.mean(np.abs(img[8:, :] - img[:-8, :]))

    #  CFA residual (sensor pattern hint)
    h, w = img.shape

    # Force even dimensions
    h2 = h - (h % 2)
    w2 = w - (w % 2)

    img_even = img[:h2, :w2]

    cfa_residual = np.std(
        img_even[::2, ::2].astype(np.float32) -
        img_even[1::2, 1::2].astype(np.float32)
    )

    # Local noise inconsistency (IMPORTANT)
    blocks = []
    for y in range(0, h - 32, 32):
        for x in range(0, w - 32, 32):
            block = img[y:y+32, x:x+32]
            blocks.append(np.std(block))
    noise_inconsistency = np.std(blocks)

    #  Saturation / clipping ratio
    clipped = np.mean((img <= 2) | (img >= 253))

    #  Texture richness (entropy)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist /= hist.sum() + 1e-8
    entropy = -np.sum(hist * np.log2(hist + 1e-8))

    return {
        "noise": noise_std,
        "edge": edge_density,
        "sharpness": sharpness,
        "jpeg": jpeg_blocks,
        "cfa": cfa_residual,
        "noise_inconsistency": noise_inconsistency,
        "clipping": clipped,
        "entropy": entropy
    }


# ---------------- FORENSIC INTERPRETATION ----------------
def interpret_forensics(f):
    """
    Robust forensic interpretation
    Tuned for modern smartphone images
    """

    # --- NORMALIZATION (UPDATED RANGES) ---
    n = {
        "noise": _normalize(f["noise"], 8, 45),                     # phones are noisy
        "edge": _normalize(f["edge"], 0.01, 0.35),
        "sharpness": _normalize(np.log1p(f["sharpness"]), 2.0, 7.5),
        "jpeg": _normalize(f["jpeg"], 2, 35),                      # recompression tolerant
        "noise_inconsistency": _normalize(f["noise_inconsistency"], 1.0, 18),
        "clipping": _normalize(f["clipping"], 0.001, 0.08),
        "entropy": _normalize(f["entropy"], 5.2, 8.2),
        "cfa": _normalize(f["cfa"], 1.0, 22)                       # ISP breaks CFA
    }

    # --- WEIGHTS ---
    weights = {
        "noise": 0.05,
        "edge": 0.05,
        "sharpness": 0.10,
        "jpeg": 0.20,
        "noise_inconsistency": 0.30,   # primary forensic signal
        "clipping": 0.05,
        "entropy": 0.15,
        "cfa": 0.10
    }

    penalty = 0.0

    # Penalize ONLY extreme synthetic indicators
    for k, w in weights.items():
        if n[k] > 0.85:
            penalty += (n[k] - 0.85) * w * 2.5

    # Optional debug (keep during testing)
    print(" Digital Forensic Extraction")
    for k, v in n.items():
        print(k, round(v, 2))

    # ---- HARD AI RED FLAGS (Extended) ----
    ai_flags = 0

    #  CFA / Sensor inconsistency (VERY STRONG)
    if f["cfa"] > 0.9:
        ai_flags += 2

    #  Extreme clipping (AI tone mapping artifact)
    if f["clipping"] > 0.18:
        ai_flags += 2
    elif f["clipping"] > 0.12:
        ai_flags += 1

    #  Entropy too low (over-smoothing)
    if f["entropy"] < 6.0:
        ai_flags += 2
    elif f["entropy"] < 6.4:
        ai_flags += 1

    #  Noise inconsistency (patch-level instability)
    if f["noise_inconsistency"] > 0.85:
        ai_flags += 2
    elif f["noise_inconsistency"] > 0.70:
        ai_flags += 1

    #  JPEG block abnormality
    if f["jpeg"] > 0.9:
        ai_flags += 1

    #  Edge–texture contradiction (GAN smoothing)
    if f["edge"] < 0.05 and f["entropy"] < 6.3:
        ai_flags += 1

    #  Sharpness–entropy paradox
    if f["sharpness"] > 0.6 and f["entropy"] < 6.4:
        ai_flags += 1

    #  Noise magnitude vs inconsistency mismatch
    if f["noise"] > 0.9 and f["noise_inconsistency"] > 0.75:
        ai_flags += 1

    #  Multi-domain failure rule
    domain_failures = sum([
        f["cfa"] > 0.9,
        f["entropy"] < 6.3,
        f["noise_inconsistency"] > 0.8,
        f["clipping"] > 0.15
    ])

    if domain_failures >= 3:
        ai_flags += 2

    # ---- FINAL AI DECISION ----
    if ai_flags >= 7:
        confidence = round(min(0.25 + ai_flags * 0.03, 0.9), 2)
        return "LIKELY SYNTHETIC / AI-GENERATED", confidence

    # --- BASE REALISM PRIOR ---
    realism = 0.78 - penalty
    confidence = round(float(np.clip(realism, 0, 1)), 2)

    if confidence >= 0.65:
        verdict = "AUTHENTIC (Camera-consistent)"
    elif confidence >= 0.30:
        verdict = "EDITED BUT REAL"
    else:
        verdict = "LIKELY SYNTHETIC / AI-GENERATED"

    return verdict, confidence



