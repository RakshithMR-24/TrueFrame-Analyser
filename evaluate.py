import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import load_model

# Load trained data
X = np.load("X_train.npy")
y = np.load("y_train.npy")

# Load trained model
model = load_model("trained_model.pkl")

# Predict on training data (or validation data if you add split)
y_pred = model.predict(X)

# Confusion Matrix
labels = ["Real", "Edited", "AI"]
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")

plt.title("ML Model Confusion Matrix")
plt.show()

importances = model.feature_importances_
features = [
    "Noise", "Edge Density", "Sharpness", "JPEG Artifacts",
    "CFA", "Noise Inconsistency", "Clipping", "Entropy"
]

plt.figure(figsize=(8, 4))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Forensic Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()
