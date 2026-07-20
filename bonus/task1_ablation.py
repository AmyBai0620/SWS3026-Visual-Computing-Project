"""Ablation: what does each post-processing step actually buy us?

Runs the same frame range of the reference video under three configurations
and reports identity switches and keypoint jitter for each.

  A  naive      : nearest-to-naive selection (no temporal association) and no
                  smoothing -- closest to the provided danceapp.py behaviour
                  once a single person has to be picked
  B  +continuity: adds the previous-frame association bonus
  C  +smoothing : adds EMA on top (this is what danceapp_v2.py uses)
"""

import os

import cv2
import numpy as np

from pose_pipeline import PoseTracker

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "dance_example_1.mp4")
MODEL = os.path.join(HERE, "yolov8n-pose.pt")

N_FRAMES = 550
SWITCH_PX = 200  # bbox-center jump treated as "we are now tracking someone else"

CONFIGS = [
    ("A naive       ", dict(use_continuity=False, smooth_alpha=1.0)),
    ("B +continuity ", dict(use_continuity=True, smooth_alpha=1.0)),
    ("C +smoothing  ", dict(use_continuity=True, smooth_alpha=0.6)),
]


def run(**kwargs):
    tracker = PoseTracker(MODEL, **kwargs)
    cap = cv2.VideoCapture(VIDEO)
    centers, track = [], []

    for _ in range(N_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        res = tracker.update(frame)
        centers.append(None if res.bbox is None else
                       ((res.bbox[0] + res.bbox[2]) / 2, (res.bbox[1] + res.bbox[3]) / 2))
        track.append((res.xy.copy(), res.valid.copy()) if res.found else None)

    cap.release()

    switches = sum(
        1 for a, b in zip(centers, centers[1:])
        if a is not None and b is not None
        and np.hypot(b[0] - a[0], b[1] - a[1]) > SWITCH_PX
    )

    steps = []
    for prev, cur in zip(track, track[1:]):
        if prev is None or cur is None:
            continue
        both = prev[1] & cur[1]
        if both.any():
            steps.append(np.median(np.linalg.norm(cur[0][both] - prev[0][both], axis=1)))

    return switches, (float(np.median(steps)) if steps else float("nan"))


def main():
    lines = [f"{'config':<15} {'id switches':>12} {'median step px':>15}"]
    for name, kwargs in CONFIGS:
        sw, jit = run(**kwargs)
        lines.append(f"{name:<15} {sw:>12} {jit:>15.2f}")
        print(lines[-1])

    text = "\n".join(lines)
    with open(os.path.join(HERE, "task1_ablation.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
