"""Build the before/after figures and the numbers quoted in FINDINGS.md.

Runs the reference video twice over the same frame range:
  * "before" = the original danceapp.py drawing (all people, no filtering)
  * "after"  = PoseTracker (main dancer, confidence filtering, EMA smoothing)

Outputs
  task1_compare.png        side-by-side panels at a few representative frames
  task1_stats.txt          people counts, selection stability, jitter, timing
"""

import os

import cv2
import numpy as np
from ultralytics import YOLO

from pose_pipeline import PoseTracker, draw_pose, draw_all_poses_raw

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "dance_example_1.mp4")
MODEL = os.path.join(HERE, "yolov8n-pose.pt")

# Frames chosen from a prior sweep: one clean single-dancer frame, then three
# frames where the detector also fires on the curtain/plant on the right.
SHOT_FRAMES = [274, 412, 481, 549]
LAST_FRAME = max(SHOT_FRAMES)

PANEL_W = 480


def label(img, text, color=(255, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def resize_panel(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (PANEL_W, int(h * PANEL_W / w)))


def main():
    model = YOLO(MODEL)
    tracker = PoseTracker(MODEL)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO}")

    shots = {}
    people_per_frame = []
    centers = []           # selected dancer center, to measure identity switches
    inf_times = []
    smoothed_track = []    # selected keypoints per frame, for jitter stats

    for fi in range(LAST_FRAME + 1):
        ret, frame = cap.read()
        if not ret:
            break

        res = tracker.update(frame)
        people_per_frame.append(res.n_people)
        inf_times.append(res.inference_ms)
        centers.append(None if res.bbox is None else
                       ((res.bbox[0] + res.bbox[2]) / 2, (res.bbox[1] + res.bbox[3]) / 2))
        smoothed_track.append((res.xy.copy(), res.valid.copy()) if res.found else None)

        if fi in SHOT_FRAMES:
            raw_result = model(frame, conf=0.3, imgsz=640, verbose=False)[0]
            before = draw_all_poses_raw(frame, raw_result)
            after = draw_pose(frame, res, show_bbox=True)
            shots[fi] = (
                label(resize_panel(before), f"BEFORE  f{fi}  people={res.n_people}"),
                label(resize_panel(after),
                      f"AFTER  f{fi}  {'no dancer' if not res.found else 'main dancer'}"),
            )

    cap.release()

    # ---- figure -------------------------------------------------------
    rows = []
    for fi in SHOT_FRAMES:
        if fi in shots:
            rows.append(np.hstack(shots[fi]))
    if rows:
        grid = np.vstack(rows)
        out_png = os.path.join(HERE, "task1_compare.png")
        cv2.imwrite(out_png, grid)
        print(f"wrote {out_png}  ({grid.shape[1]}x{grid.shape[0]})")

    # ---- statistics ---------------------------------------------------
    people = np.array(people_per_frame)
    inf = np.array(inf_times)

    # identity switches: large jumps of the selected bbox center
    jumps = 0
    for a, b in zip(centers, centers[1:]):
        if a is not None and b is not None:
            if np.hypot(b[0] - a[0], b[1] - a[1]) > 200:
                jumps += 1

    # jitter: median per-frame displacement of jointly-valid keypoints
    def median_step(track):
        steps = []
        for prev, cur in zip(track, track[1:]):
            if prev is None or cur is None:
                continue
            both = prev[1] & cur[1]
            if both.any():
                steps.append(np.median(np.linalg.norm(cur[0][both] - prev[0][both], axis=1)))
        return float(np.median(steps)) if steps else float("nan")

    smooth_jitter = median_step(smoothed_track)

    lines = [
        f"frames processed          : {len(people)}",
        f"frames with 0 people      : {int((people == 0).sum())}",
        f"frames with >1 person     : {int((people > 1).sum())}",
        f"max people in a frame     : {int(people.max())}",
        f"identity switches (>200px): {jumps}",
        f"median keypoint step (px) : {smooth_jitter:.2f}   [EMA alpha=0.6]",
        f"inference mean / p95 (ms) : {inf.mean():.1f} / {np.percentile(inf, 95):.1f}",
        f"achievable fps            : {1000 / inf.mean():.1f}   (video is 30 fps)",
    ]
    text = "\n".join(lines)
    print(text)

    with open(os.path.join(HERE, "task1_stats.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
