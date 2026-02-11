import streamlit as st
import pandas as pd
import numpy as np
import tempfile
from PIL import Image
from PIL.ExifTags import TAGS

# --------- IMPORT YOUR PIPELINE ---------
from authenticity_checker import check_image_authenticity
from features import (
    extract_metadata_features,
    extract_advanced_forensic_features,
    metadata_presence_report
)
from model import normalize_forensics, load_model, predict_image
from fusion import final_verdict_fusion

st.markdown(
    """
    <style>
    /* -------- APP BACKGROUND IMAGE -------- */
    .stApp {
        background-image: url("background.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }

    /* Optional dark overlay for readability */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.55);
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <script>
    document.addEventListener('contextmenu', event => event.preventDefault());
    document.onkeydown = function(e) {
        if (e.keyCode == 123) return false; // F12
        if (e.ctrlKey && e.shiftKey && e.keyCode == 73) return false; // Ctrl+Shift+I
        if (e.ctrlKey && e.shiftKey && e.keyCode == 74) return false; // Ctrl+Shift+J
        if (e.ctrlKey && e.keyCode == 85) return false; // Ctrl+U
    };
    </script>
    """,
    unsafe_allow_html=True
)

# --------- PAGE CONFIG ---------
st.set_page_config(
    page_title="Image Authenticity Analyzer",
    layout="wide"
)

# ------------------- MODERN UI DESIGN -------------------

st.markdown("""
<style>

/* ---------- Animated Gradient Background ---------- */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #141e30);
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
    color: #e6f1ff;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* ---------- Glass Cards ---------- */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    margin-bottom: 25px;
}

/* ---------- Titles ---------- */
h1 {
    text-align: center;
    font-weight: 700;
    letter-spacing: 1px;
    color: #00f5ff;
}

h2, h3 {
    color: #4dd0e1;
    margin-top: 10px;
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 20px rgba(0, 255, 255, 0.15);
}

/* ---------- File Uploader ---------- */
[data-testid="stFileUploader"] {
    border: 2px dashed #00f5ff;
    border-radius: 14px;
    padding: 25px;
    background: rgba(255,255,255,0.05);
}

/* ---------- DataFrames ---------- */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* ---------- Buttons ---------- */
button {
    background: linear-gradient(90deg, #00f5ff, #00bcd4);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    transition: 0.3s;
}
button:hover {
    transform: scale(1.05);
}

/* ---------- Hide Streamlit Branding ---------- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)



st.markdown(
        """
        <div style="
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 40vh;
            text-align: center;
        ">
            <h1 style="color:#00e5ff; font-size:48px;">
                TrueFrame Analyser
            </h1>
            <h3 style="color:#4dd0e1;">
                A Forensic Based Analysis for Deepfake Identification and Prevention
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown("""

""", unsafe_allow_html=True)

st.markdown("""
<style>

/* -------- GLOBAL -------- */
html, body, [class*="css"] {
    font-family: "Segoe UI", Roboto, sans-serif;
    background-color: #0e1117;
    color: #e6e6e6;
}

/* -------- HEADERS -------- */
h1 {
    color: #00e5ff;
    text-align: center;
    letter-spacing: 1px;
}
h2, h3 {
    color: #4dd0e1;
}

/* -------- METRICS -------- */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #111827, #020617);
    border: 1px solid #1f2933;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 0 12px rgba(0, 229, 255, 0.15);
}

/* -------- DATAFRAMES -------- */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid #1f2933;
    background-color: #020617;
}

/* -------- FILE UPLOADER -------- */
[data-testid="stFileUploader"] {
    border: 2px dashed #00e5ff;
    padding: 20px;
    border-radius: 12px;
}

/* -------- BUTTONS -------- */
button {
    background: linear-gradient(90deg, #00e5ff, #0288d1);
    color: black;
    font-weight: bold;
    border-radius: 10px;
}

/* -------- PROGRESS BAR -------- */
div[role="progressbar"] {
    background-color: #00e5ff;
}

/* -------- REMOVE STREAMLIT BRANDING -------- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}


<style>
...
/* -------- APP BACKGROUND -------- */
.stApp {
    background-image:
        linear-gradient(
            rgba(2, 6, 23, 0.85),
            rgba(2, 6, 23, 0.85)
        ),
        url("./static/background.png");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>


</style>
""", unsafe_allow_html=True)

# --------- IMAGE UPLOAD ---------
uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Save uploaded image temporarily (preserves EXIF)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    image = Image.open(image_path)

    # -------- IMAGE DISPLAY (FIXED SIZE) --------
    img_col, info_col = st.columns([1, 2])

    with img_col:
        st.image(
            image,
            caption="Uploaded Image",
            width=300
        )

    with info_col:
        st.markdown("### Image Information")
        st.write(f"**Resolution:** {image.size[0]} × {image.size[1]}")
        st.write(f"**Format:** {image.format}")

        if image.format not in ["JPEG", "JPG"]:
            st.warning("EXIF metadata is usually available only for JPEG images.")

    st.divider()

    # ============================================================
    # PRIMARY EXPERT-RULE ANALYSIS
    # ============================================================
    st.header("Primary Expert-Rule Analysis")

    metadata_result, forensic_result = check_image_authenticity(image_path)

    # ---------------- METADATA ----------------
    st.subheader("Metadata (EXIF) Analysis")

    metadata = extract_metadata_features(image_path)
    presence = metadata_presence_report(metadata)

    meta_df = pd.DataFrame([
        {"EXIF Feature": k, "Status": "Present" if v else "Missing"}
        for k, v in presence.items()
    ])

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(
            meta_df,
            width= 'stretch',
            hide_index=True
        )

    with col2:
        st.metric("Metadata Verdict", metadata_result[0])
        st.metric("Metadata Confidence", metadata_result[1])


    # ---------------- FORENSICS ----------------
    st.subheader("Forensic Feature Analysis")

    forensic_features = extract_advanced_forensic_features(image_path)

    forensic_df = pd.DataFrame({
        "Forensic Feature": forensic_features.keys(),
        "Value": forensic_features.values()
    })

    col3, col4 = st.columns([2, 1])

    with col3:
        st.dataframe(
            forensic_df,
            use_container_width=True,
            hide_index=True
        )

    with col4:
        st.metric("Forensic Verdict", forensic_result[0])
        st.metric("Forensic Confidence", forensic_result[1])

    # ============================================================
    # ML PREDICTION (FORENSIC-ONLY)
    # ============================================================
    st.divider()
    st.header("ML Prediction (Random Forest – Forensic Only)")

    normalized_vector = normalize_forensics(forensic_features)

    model = load_model("trained_model.pkl")
    ml_label, ml_conf = predict_image(model, normalized_vector)

    ml_result = {
        "label": ml_label,
        "confidence": ml_conf
    }

    col5, col6 = st.columns(2)
    with col5:
        st.metric("ML Prediction", ml_label)
    with col6:
        st.metric("ML Confidence", ml_conf)



    # ============================================================
    # CONFIDENCE VISUALIZATION
    # ============================================================
    st.divider()
    st.header("Confidence Visualization")

    conf_df = pd.DataFrame({
        "Module": ["Metadata", "Forensics", "ML"],
        "Confidence": [
            metadata_result[1],
            forensic_result[1],
            ml_conf
        ]
    })

    # Create margins using columns to center the chart
    left_spacer, center_col, right_spacer = st.columns([1, 2, 1])

    with center_col:
        st.subheader("Module-wise Confidence Scores")
        st.bar_chart(conf_df.set_index("Module"))

    # ============================================================
    # FINAL FUSION
    # ============================================================
    st.divider()
    st.header("Final Fused Verdict")

    final_verdict, final_score = final_verdict_fusion(
        metadata_result,
        forensic_result,
        ml_result
    )

    col7, col8 = st.columns(2)
    with col7:
        st.metric("Final Verdict", final_verdict)
    with col8:
        st.metric("Final Confidence Score", final_score)

    st.progress(final_score)

    st.markdown(
        """
        **Fusion Strategy**
        - Metadata + Forensic rules → Primary reliability
        - ML (Random Forest) → Statistical validation
        - Weighted fusion → Robust final decision
        """
    )

else:
    st.info("Upload an image to begin analysis")