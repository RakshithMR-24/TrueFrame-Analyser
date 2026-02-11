TrueFrame Analyser
A Forensic-Based Deepfake Media Identification System using Machine Learning

--> Overview

**TrueFrame Analyser** is a digital forensic tool designed to determine
whether an image is:

-    Real\
-    Real but Edited\
-    AI Generated

The system combines **EXIF metadata analysis**, **advanced forensic
digital fingerprinting**, and a **Random Forest Machine Learning model**
to produce an authenticity confidence score.

>  Core Modules

--> Metadata Analysis

Extracts and verifies: - Camera Make & Model\
- Date & Time Information\
- GPS Metadata\
- Software Used\
- Resolution & Format\
- EXIF Consistency Check

Forensic Digital Fingerprint Analysis

Analyzes: - Noise Patterns\
- Edge Consistency\
- Image Sharpness\
- JPEG Compression Artifacts\
- CFA (Color Filter Array) Pattern\
- Noise Inconsistency\
- Clipping Detection\
- Entropy Measurement

 Machine Learning Module

-   Algorithm: Random Forest Classifier\
-   Input: Normalized forensic feature vector\
-   Output: Authenticity probability score

Fusion Engine

Weighted fusion of: - Metadata verdict\
- Forensic rule-based verdict\
- ML statistical prediction

Produces a final categorized verdict with confidence score.

 Project Structure

TrueFrame-Analyser/ │ ├── app.py \# Main Application (Streamlit / Flask)
├── authenticity_checker.py \# Rule-based authenticity logic ├──
features.py \# Metadata & forensic feature extraction ├── model.py \#
Model loading & prediction ├── fusion.py \# Final verdict fusion logic
├── trained_model.pkl \# Trained Random Forest model ├── static/ \# UI
assets & background └── README.md

 Installation

--> Clone Repository

``` bash
git clone https://github.com/YOUR_USERNAME/TrueFrame-Analyser.git
cd TrueFrame-Analyser
```

--> Install Dependencies

``` bash
pip install -r requirements.txt
```

--> Run Application

``` bash
streamlit run app.py
```

Output

The system provides: - Metadata presence report - Forensic feature
values - ML prediction & confidence - Module-wise confidence
visualization - Final fused authenticity verdict

 Research Recognition

Presented at **ICAIDS Conference -- Loyola College (Feb 9--10)**\
 Best Paper Award
 Best Presentation Award

Paper Title:\
**TrueFrame Analyser: A Forensic Based Analysis For Deepfake Media
Identification Using Machine Learning**


Author
Rakshith Subrmanya\
Cyber Security and Digital Forensics\
NFSU Chennai

 License

Developed for academic and research purposes.
