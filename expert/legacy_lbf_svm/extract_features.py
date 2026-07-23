import argparse
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DATASET = BASE / "facial_expression_dataset"
LBF_PATH = str(BASE.parent / "beginner" / "lbfmodel.yaml")
OUT_DIR = BASE / "features"
OUT_DIR.mkdir(exist_ok=True)

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
UPSCALE_SIZE = 200  # FER-2013 crops are 48x48; upscale before landmark regression

facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(LBF_PATH)


def extract_landmarks(gray_48):
    """Treat the whole (already-cropped) face image as the face box and
    regress 68 landmarks directly, skipping Haar/MediaPipe detection."""
    upscaled = cv2.resize(gray_48, (UPSCALE_SIZE, UPSCALE_SIZE), interpolation=cv2.INTER_CUBIC)
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
    return (centered / scale).flatten()  # 136-dim feature vector


def process_split(split, limit=None):
    X, y, failed = [], [], 0
    for label_idx, emotion in enumerate(EMOTIONS):
        folder = DATASET / split / emotion
        files = sorted(folder.glob("*.jpg"))
        if limit:
            files = files[:limit]
        print(f"[{split}] {emotion}: {len(files)} images")
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                failed += 1
                continue
            feat = extract_landmarks(img)
            if feat is None:
                failed += 1
                continue
            X.append(feat)
            y.append(label_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    total = len(X) + failed
    rate = (failed / total * 100) if total else 0.0
    print(f"[{split}] done: {len(X)} succeeded, {failed} failed ({rate:.1f}%)")
    np.savez(OUT_DIR / f"{split}_features.npz", X=X, y=y)
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="max images per class (for smoke testing)")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    t0 = time.time()
    if args.split in ("train", "both"):
        process_split("train", limit=args.limit)
    if args.split in ("test", "both"):
        process_split("test", limit=args.limit)
    print(f"Total time: {time.time() - t0:.1f}s")
