from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent
FEAT_DIR = BASE / "features"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

train = np.load(FEAT_DIR / "train_features.npz")
test = np.load(FEAT_DIR / "test_features.npz")
X_train, y_train = train["X"], train["y"]
X_test, y_test = test["X"], test["y"]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", verbose=True)
print("Starting SVM training (one-vs-one over 7 classes -> 21 binary sub-problems)...")
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=EMOTIONS, digits=3))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(EMOTIONS)))
ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
ax.set_yticks(range(len(EMOTIONS)))
ax.set_yticklabels(EMOTIONS)
for i in range(len(EMOTIONS)):
    for j in range(len(EMOTIONS)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (acc={acc * 100:.1f}%)")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(BASE / "confusion_matrix.png", dpi=150)
print(f"Saved confusion matrix to {BASE / 'confusion_matrix.png'}")

joblib.dump({"model": clf, "scaler": scaler}, BASE / "svm_model.pkl")
print(f"Saved model to {BASE / 'svm_model.pkl'}")
