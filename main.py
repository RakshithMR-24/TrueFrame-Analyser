from authenticity_checker import check_image_authenticity
from predict import ml_predict
from fusion import final_verdict_fusion
from features import extract_metadata_features, metadata_presence_report
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from analyzer import analyze_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
   allow_origins=["*"],
   allow_methods=["*"],
   allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    result = analyze_image(temp_path)
    return result


# ---------------- IMAGE PATH ----------------
img_path = r"C:\Users\Rakshith\PycharmProjects\MiniProject2\test_folder\IMG_20241123_212959.jpg"

# ---------------- METADATA EXTRACTION ----------------
metadata = extract_metadata_features(img_path)
metadata_presence = metadata_presence_report(metadata)

print("\nEXTRACTED METADATA FEATURES:")
for feature, present in metadata_presence.items():
    status = "Present" if present else "Missing"
    print(f"  {feature}: {status}")

# ---------------- EXPERT-RULE ANALYSIS ----------------
metadata_result, forensic_result = check_image_authenticity(img_path)

# ---------------- FORENSIC-ONLY ML PREDICTION ----------------
ml_result = ml_predict(img_path)

# ---------------- FINAL FUSION ----------------
verdict, score = final_verdict_fusion(
    metadata_result,
    forensic_result,
    ml_result
)

# ---------------- DISPLAY RESULTS ----------------
print(f"""
-----------------------------------
IMAGE: {img_path}

Metadata Verdict:
  Label: {metadata_result[0]}
  Confidence: {metadata_result[1]}

Forensic Verdict:
  Label: {forensic_result[0]}
  Confidence: {forensic_result[1]}

ML Prediction:
  Label: {ml_result['label']}
  Confidence: {ml_result['confidence']}

-----------------------------------
FINAL VERDICT: {verdict}
FINAL CONFIDENCE SCORE: {score}
-----------------------------------
""")
