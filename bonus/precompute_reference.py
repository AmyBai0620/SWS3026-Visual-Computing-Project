"""Precompute the reference dancer's keypoints for Task 2 (Just Dance).

The reference video never changes, so we run the pose pipeline over it once,
offline, and cache the per-frame keypoints. At runtime just_dance.py then only
has to run inference on the webcam, halving the live GPU/CPU load -- which is
what makes two panels playable at the same time on a CPU.

Usage
  python precompute_reference.py [name]     # default dance_example_5
      looked up in ./ then ./video/, writes ref_<name>.npz next to it.

The .npz holds
  xy    (N,17,2) float32  main-dancer keypoints in pixels
  valid (N,17)   bool     per-keypoint validity (conf >= threshold)
  t     (N,)     float32  timestamp of each frame in seconds
  fps   scalar            source frame rate
  size  (2,)              (width, height) of the reference video
"""

import os
import sys

import cv2
import numpy as np

from pose_pipeline import PoseTracker, NUM_KEYPOINTS

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "yolov8n-pose.pt")


def resolve_video(name):
    for cand in (os.path.join(HERE, name + ".mp4"),
                 os.path.join(HERE, "video", name + ".mp4")):
        if os.path.exists(cand):
            return cand
    raise SystemExit(f"video not found for '{name}'")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "dance_example_5"
    video = resolve_video(name)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = PoseTracker(MODEL)
    xy_list, valid_list, t_list = [], [], []
    missing = 0
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = tracker.update(frame)
        if res.found:
            xy_list.append(res.xy.astype(np.float32))
            valid_list.append(res.valid.copy())
        else:
            # keep frame alignment: store zeros marked invalid
            xy_list.append(np.zeros((NUM_KEYPOINTS, 2), np.float32))
            valid_list.append(np.zeros(NUM_KEYPOINTS, bool))
            missing += 1
        t_list.append(fi / fps)
        fi += 1
    cap.release()

    xy = np.stack(xy_list)
    valid = np.stack(valid_list)
    t = np.asarray(t_list, np.float32)

    out = os.path.join(os.path.dirname(video), f"ref_{name}.npz")
    np.savez_compressed(out, xy=xy, valid=valid, t=t,
                        fps=np.float32(fps), size=np.array([w, h], np.int32))

    print(f"video     : {name}  {w}x{h}  {fps:.1f} fps")
    print(f"frames    : {len(xy)}  ({len(xy) / fps:.1f} s)")
    print(f"no-dancer : {missing} frames ({100 * missing / len(xy):.1f}%)")
    print(f"mean valid keypoints/frame : {valid.sum(1).mean():.1f} / {NUM_KEYPOINTS}")
    print(f"wrote     : {out}  ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
