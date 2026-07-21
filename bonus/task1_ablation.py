"""Ablation: what does each post-processing step actually buy us?

Runs a frame range of a reference video under three configurations and reports
identity switches and keypoint jitter for each.

  A  naive      : nearest-to-naive selection (no temporal association) and no
                  smoothing -- closest to the provided danceapp.py behaviour
                  once a single person has to be picked
  B  +continuity: adds the previous-frame association bonus
  C  +smoothing : adds EMA on top (this is what danceapp_v2.py uses)

Usage
  python task1_ablation.py [name] [n_frames]
    name  video stem (default dance_example_1), looked up in ./ then ./video/
"""

import os
import sys

import cv2
import numpy as np

from pose_pipeline import PoseTracker

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "yolov8n-pose.pt")

SWITCH_FRAC = 0.15  # bbox-center jump (fraction of frame diagonal) = "different person"

CONFIGS = [
    ("A naive       ", dict(use_continuity=False, smooth_alpha=1.0)),
    ("B +continuity ", dict(use_continuity=True, smooth_alpha=1.0)),
    ("C +smoothing  ", dict(use_continuity=True, smooth_alpha=0.6)),
]


def resolve_video(name):
    for cand in (os.path.join(HERE, name + ".mp4"),
                 os.path.join(HERE, "video", name + ".mp4")):
        if os.path.exists(cand):
            return cand
    raise SystemExit(f"video not found for '{name}'")


def run(video, switch_px, n_frames, **kwargs):
    tracker = PoseTracker(MODEL, **kwargs)
    cap = cv2.VideoCapture(video)
    centers, track = [], []

    for _ in range(n_frames):
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
        and np.hypot(b[0] - a[0], b[1] - a[1]) > switch_px
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
    name = sys.argv[1] if len(sys.argv) > 1 else "dance_example_1"
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 550
    video = resolve_video(name)

    cap = cv2.VideoCapture(video)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    switch_px = SWITCH_FRAC * np.hypot(fw, fh)

    lines = [
        f"video: {name} ({fw}x{fh})  frames<= {n_frames}  switch threshold {switch_px:.0f}px",
        f"{'config':<15} {'id switches':>12} {'median step px':>15}",
    ]
    print(lines[0])
    print(lines[1])
    for cfg_name, kwargs in CONFIGS:
        sw, jit = run(video, switch_px, n_frames, **kwargs)
        lines.append(f"{cfg_name:<15} {sw:>12} {jit:>15.2f}")
        print(lines[-1])

    text = "\n".join(lines)
    with open(os.path.join(HERE, f"task1_ablation_{name}.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
