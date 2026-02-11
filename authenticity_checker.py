from features import extract_metadata_features, interpret_forensics

CORE_FEATURES = [
    "camera_make", "camera_model","lens_model", "datetime_original","gps_info","white_balance",
    "exposure_time", "f_number", "iso_speed","orientation","compression",
    "focal_length", "flash", "exif_version"
]
from features import (
    extract_advanced_forensic_features,
    interpret_forensics,
    extract_metadata_features
)


def check_image_authenticity(image_path):
    # Metadata
    metadata_features = extract_metadata_features(image_path)
    meta_conf = sum(metadata_features.values()) / len(metadata_features)

    if meta_conf > 0.6:
        meta_label = "GOOD METADATA"
    elif meta_conf > 0.3:
        meta_label = "PARTIAL METADATA"
    else:
        meta_label = "METADATA MISSING"

    metadata_result = (meta_label, round(meta_conf, 2))

    # Forensics
    forensic_features = extract_advanced_forensic_features(image_path)
    forensic_result = interpret_forensics(forensic_features)

    return metadata_result, forensic_result
