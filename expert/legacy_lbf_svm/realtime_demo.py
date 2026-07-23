import time
from pathlib import Path

import cv2
import joblib
import numpy as np

BASE = Path(__file__).resolve().parent
CASCADE_PATH = str(BASE.parent / "beginner" / "haarcascade_frontalface_default.xml")
LBF_PATH = str(BASE.parent / "beginner" / "lbfmodel.yaml")
MODEL_PATH = BASE / "svm_model.pkl"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
UPSCALE_SIZE = 200  # must match extract_features.py

detector = cv2.CascadeClassifier(CASCADE_PATH)
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(LBF_PATH)

bundle = joblib.load(MODEL_PATH)
clf, scaler = bundle["model"], bundle["scaler"]


def landmarks_to_features(gray_roi):
    upscaled = cv2.resize(gray_roi, (UPSCALE_SIZE, UPSCALE_SIZE), interpolation=cv2.INTER_CUBIC)
    face_box = np.array([[0, 0, UPSCALE_SIZE, UPSCALE_SIZE]], dtype=np.int32)
    ok, landmarks = facemark.fit(upscaled, face_box)
    if not ok or landmarks is None or len(landmarks) == 0:
        return None
    lm = landmarks[0].reshape(-1, 2)
    centroid = lm.mean(axis=0)
    centered = lm - centroid
    scale = np.sqrt((centered ** 2).sum(axis=1)).mean()
    if scale < 1e-6:
        return None
    return (centered / scale).flatten()


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Emotion Recognition", 800, 600)

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            feat = landmarks_to_features(roi)
            label = "?"
            if feat is not None:
                feat_s = scaler.transform(feat.reshape(1, -1))
                pred = clf.predict(feat_s)[0]
                label = EMOTIONS[pred]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                         0.9, (0, 255, 0), 2)

        elapsed_ms = (time.time() - t0) * 1000
        cv2.putText(frame, f"{elapsed_ms:.1f} ms/frame", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
