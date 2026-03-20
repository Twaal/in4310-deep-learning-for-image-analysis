import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataset import stratified_split, CLASSES

SEED = 42
np.random.seed(SEED)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")

# Load test split to get true labels
_, _, test_split = stratified_split(DATA_ROOT, seed=SEED)
true_labels = test_split[1]

# Load saved scores for model3_transfer
scores = np.load(os.path.join(MODELS_DIR, "model3_transfer_scores.npy"))
pred_labels = np.argmax(scores, axis=1)

# Compute confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot
fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix model3_transfer on test set")
plt.tight_layout()

out_path = os.path.join(REPORT_DIR, "confusion_matrix_model3_transfer.png")
plt.savefig(out_path, dpi=150)
