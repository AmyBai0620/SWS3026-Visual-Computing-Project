"""Body keypoint detection pipeline for the Bonus Level (Task 1).

Wraps the raw YOLOv8-pose output with the post-processing that the plain
example in danceapp.py is missing:

  1. per-keypoint confidence filtering  -> no phantom points at (0, 0)
  2. main-dancer selection              -> only one skeleton when a bystander
                                           walks into frame
  3. temporal association               -> the selection does not flip between
                                           two people from frame to frame
  4. EMA smoothing                      -> reduces per-frame jitter
  5. short detection dropouts are held  -> the skeleton does not blink out

The module is GUI-free on purpose so the same tracker can be used for the
reference video and for the webcam in Task 2.
"""

import time

import cv2
import numpy as np
from ultralytics import YOLO

# COCO-17 skeleton, same pairs as the provided example.
SKELETON = [
    (0, 5), (0, 6),      # nose to shoulders
    (5, 6),              # shoulders
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # torso sides
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

NUM_KEYPOINTS = 17


class PoseResult:
    """Keypoints of the selected dancer for a single frame."""

    def __init__(self, xy, valid, bbox, n_people, inference_ms, held):
        self.xy = xy                      # (17, 2) float32, pixel coordinates
        self.valid = valid                # (17,) bool, False = filtered out
        self.bbox = bbox                  # (x1, y1, x2, y2) or None
        self.n_people = n_people          # people detected before selection
        self.inference_ms = inference_ms  # model time for this frame
        self.held = held                  # True = reused from a previous frame

    @property
    def found(self):
        return self.bbox is not None

    def torso_size(self):
        """Shoulder-to-hip distance, used to normalise scale in Task 2."""
        needed = (5, 6, 11, 12)
        if not all(self.valid[i] for i in needed):
            return None
        shoulder = (self.xy[5] + self.xy[6]) / 2.0
        hip = (self.xy[11] + self.xy[12]) / 2.0
        return float(np.linalg.norm(shoulder - hip))


class PoseTracker:
    def __init__(
        self,
        model_path="yolov8n-pose.pt",
        det_conf=0.3,
        kpt_conf=0.5,
        smooth_alpha=0.6,
        hold_frames=5,
        imgsz=640,
        use_continuity=True,
    ):
        self.model = YOLO(model_path)
        self.det_conf = det_conf
        self.kpt_conf = kpt_conf
        self.smooth_alpha = smooth_alpha
        self.hold_frames = hold_frames
        self.imgsz = imgsz
        self.use_continuity = use_continuity  # off = ablation baseline
        self.reset()

    def reset(self):
        self._prev_xy = None       # smoothed keypoints of the last good frame
        self._prev_valid = None
        self._prev_center = None   # bbox center, used for temporal association
        self._prev_bbox = None
        self._misses = 0

    # ------------------------------------------------------------------
    # main-dancer selection
    # ------------------------------------------------------------------
    def _score_candidates(self, boxes_xyxy, kpt_conf, frame_w, frame_h):
        """Rank detected people; the dancer is large, confident and central.

        A separate bonus rewards the candidate closest to the person picked in
        the previous frame, which is what stops the skeleton from jumping
        between two dancers of similar size.
        """
        frame_area = float(frame_w * frame_h)
        diag = float(np.hypot(frame_w, frame_h))
        scores = []

        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            area_term = np.sqrt(area / frame_area)  # sqrt keeps it from dominating

            mean_conf = float(np.mean(kpt_conf[i]))

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            off_center = np.hypot(cx - frame_w / 2.0, cy - frame_h / 2.0) / (diag / 2.0)
            center_term = 1.0 - 0.5 * min(1.0, off_center)  # mild preference only

            score = area_term * mean_conf * center_term

            if self.use_continuity and self._prev_center is not None:
                dist = np.hypot(cx - self._prev_center[0], cy - self._prev_center[1])
                # full bonus when the candidate barely moved, fading to none
                # once it is a third of the frame away
                continuity = max(0.0, 1.0 - dist / (diag / 3.0))
                score *= 1.0 + 0.8 * continuity

            scores.append(score)

        return int(np.argmax(scores))

    # ------------------------------------------------------------------
    def update(self, frame):
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        result = self.model(frame, conf=self.det_conf, imgsz=self.imgsz, verbose=False)[0]
        inference_ms = (time.perf_counter() - t0) * 1000.0

        boxes = result.boxes
        kpts = result.keypoints
        has_person = (
            boxes is not None
            and len(boxes) > 0
            and kpts is not None
            and kpts.conf is not None
        )

        if not has_person:
            return self._miss(inference_ms, n_people=0)

        boxes_xyxy = boxes.xyxy.cpu().numpy()
        kpt_xy = kpts.xy.cpu().numpy()      # (P, 17, 2) in pixels
        kpt_conf = kpts.conf.cpu().numpy()  # (P, 17)
        n_people = int(kpt_xy.shape[0])

        idx = self._score_candidates(boxes_xyxy, kpt_conf, w, h)
        xy = kpt_xy[idx].astype(np.float32).copy()
        valid = kpt_conf[idx] >= self.kpt_conf

        # Guard against the model reporting a point at the origin with a high
        # score, which would otherwise be drawn in the top-left corner.
        valid &= ~np.all(xy == 0, axis=1)

        if not valid.any():
            return self._miss(inference_ms, n_people=n_people)

        xy = self._smooth(xy, valid)

        x1, y1, x2, y2 = boxes_xyxy[idx]
        self._prev_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self._prev_bbox = (float(x1), float(y1), float(x2), float(y2))
        self._prev_xy = xy.copy()
        self._prev_valid = valid.copy()
        self._misses = 0

        return PoseResult(xy, valid, self._prev_bbox, n_people, inference_ms, held=False)

    def _smooth(self, xy, valid):
        """EMA on keypoints that were valid in both this and the last frame."""
        if self._prev_xy is None or self._prev_valid is None:
            return xy
        a = self.smooth_alpha
        both = valid & self._prev_valid
        xy[both] = a * xy[both] + (1.0 - a) * self._prev_xy[both]
        return xy

    def _miss(self, inference_ms, n_people):
        """No usable detection: briefly reuse the last pose, then give up."""
        self._misses += 1
        if self._prev_xy is not None and self._misses <= self.hold_frames:
            return PoseResult(
                self._prev_xy.copy(),
                self._prev_valid.copy(),
                self._prev_bbox,
                n_people,
                inference_ms,
                held=True,
            )
        self._prev_xy = None
        self._prev_valid = None
        self._prev_center = None
        self._prev_bbox = None
        return PoseResult(
            np.zeros((NUM_KEYPOINTS, 2), np.float32),
            np.zeros(NUM_KEYPOINTS, bool),
            None,
            n_people,
            inference_ms,
            held=False,
        )


# ----------------------------------------------------------------------
# display helpers
# ----------------------------------------------------------------------
def fit_letterbox(frame, box_w, box_h, pad=(0, 0, 0)):
    """Resize a BGR frame into box_w x box_h *preserving aspect ratio*.

    A plain cv2.resize / PIL.resize stretches the frame to the target box, so a
    4:3 webcam shown in a 9:16 panel comes out squashed (too thin) and a
    portrait clip in a landscape panel comes out too wide. This scales by the
    smaller factor and pads the leftover with `pad`, so nobody is distorted.
    """
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return np.full((box_h, box_w, 3), pad, np.uint8)
    scale = min(box_w / w, box_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((box_h, box_w, 3), pad, np.uint8)
    x0, y0 = (box_w - nw) // 2, (box_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------
def draw_skeleton(canvas, xy, valid, color_line=(0, 255, 0),
                  color_point=(0, 0, 255), radius=5):
    """Draw a skeleton in place from raw (17,2) keypoints + (17,) validity.

    Shared by the live pipeline (draw_pose) and by just_dance.py, which draws
    the precomputed reference skeleton straight from cached arrays.
    """
    pts = np.asarray(xy).astype(int)
    for a, b in SKELETON:
        if valid[a] and valid[b]:
            cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), color_line, 2)
    for i in range(NUM_KEYPOINTS):
        if valid[i]:
            cv2.circle(canvas, tuple(pts[i]), radius, color_point, -1)
    return canvas


def draw_pose(frame, res, show_frame=True, color_line=(0, 255, 0),
              color_point=(0, 0, 255), show_bbox=False):
    """Draw one PoseResult. Invalid keypoints and their bones are skipped."""
    canvas = frame.copy() if show_frame else np.ones_like(frame) * 255

    if not res.found:
        return canvas

    draw_skeleton(canvas, res.xy, res.valid, color_line, color_point)

    if show_bbox and res.bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in res.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 200, 0), 2)

    return canvas


def draw_all_poses_raw(frame, result, show_frame=True):
    """Reproduce the original danceapp.py behaviour, for before/after figures.

    Every detected person is drawn and no confidence filtering is applied, so
    undetected joints land at (0, 0) in the top-left corner.
    """
    canvas = frame.copy() if show_frame else np.ones_like(frame) * 255
    h, w = frame.shape[:2]

    if result.keypoints is None:
        return canvas

    for person in result.keypoints.xyn.cpu().numpy():
        pts = [(int(x * w), int(y * h)) for x, y in person]
        for a, b in SKELETON:
            if a < len(pts) and b < len(pts):
                cv2.line(canvas, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(canvas, p, 5, (0, 0, 255), -1)

    return canvas
