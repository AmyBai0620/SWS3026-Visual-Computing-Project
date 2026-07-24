"""Bonus Level Task 2 -- Just Dance (double-panel GUI with live scoring).

Left panel  : the reference dancer, drawn from the precomputed skeleton
              (precompute_reference.py), played on the video's own clock.
Right panel : your webcam, live pose from PoseTracker, with a big Just-Dance
              style feedback overlay (PERFECT / SUPER / GOOD / X) and a numeric
              score that updates every frame.

Rubric coverage:
  Q13  program in action                    -> both panels running
  Q14  align + similarity metric            -> pose_score.PoseScorer
  Q15  numeric + textual, over time, total  -> overlay + end-of-dance summary

Prerequisite:
  python precompute_reference.py dance_example_5     # makes ref_*.npz

Controls:
  Start Webcam   turn the camera on (see yourself + skeleton)
  Start Dance    reference video plays from the top; scoring begins
  Stop Dance     stop early
"""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from pose_pipeline import PoseTracker, draw_skeleton, fit_letterbox
from pose_score import PoseScorer, TIER_POINTS, scorable

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "yolov8n-pose.pt")
REF_NAME = "dance_example_1"

DISPLAY_W, DISPLAY_H = 360, 640    # starting panel size; both panels then follow
MIN_DISPLAY = 160                  # never render a panel smaller than this
HUD_TOP = 0.13                     # panel fraction reserved above the video (tier word)
HUD_BOT = 0.10                     # ...and below it (score / running average)

# Inference is the pipeline's bottleneck. Measured on this CPU (yolov8n-pose):
#   imgsz 640 -> 89 ms/frame (11 fps) | 512 -> 83 | 416 -> 72 | 320 -> 56 (18 fps)
# The camera pushes 30 fps, so the consumer is always the slow side; drop imgsz
# if you need more headroom, at some cost in keypoint accuracy.
CAM_IMGSZ = 640
# Keypoint EMA. Lower = smoother but laggier: the delay is about (1-a)/a frames,
# so 0.6 costs ~0.7 frames (~60 ms here) and 0.85 costs ~0.2 (~15 ms).
CAM_SMOOTH = 0.85

TIER_COLORS = {   # BGR
    "PERFECT": (60, 215, 255),     # gold
    "SUPER": (90, 230, 90),        # green
    "GOOD": (240, 180, 60),        # blue
    "X": (150, 150, 150),          # grey
}


class CameraFeed:
    """Grabber thread that keeps only the newest frame, with its capture time.

    The webcam produces 30 fps but pose inference consumes ~11 fps, so a plain
    cap.read() loop hands back whatever the driver queued while we were busy:
    the backlog grows and the frame being scored drifts further and further
    behind reality (the "half a second" of lag). Grabbing continuously in a
    thread and keeping just the last frame caps staleness at one camera period.
    """

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.ok = self.cap.isOpened()
        try:                                   # honoured by some backends only
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = -1.0
        self.running = self.ok
        if self.ok:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                break
            with self._lock:                   # cap.read() returns a fresh array
                self._frame, self._stamp = f, time.perf_counter()
        self.running = False

    def latest(self):
        """Newest frame and the wall-clock time it was captured."""
        with self._lock:
            return self._frame, self._stamp

    def close(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


def find_ref():
    for base in (HERE, os.path.join(HERE, "video")):
        npz = os.path.join(base, f"ref_{REF_NAME}.npz")
        vid = os.path.join(base, f"{REF_NAME}.mp4")
        if os.path.exists(npz) and os.path.exists(vid):
            return npz, vid
    return None, None


def draw_hud(canvas, y_top, y_bot, hud):
    """Draw the feedback into the reserved bands, never over the dancer.

    Painting the tier word onto the video meant it landed on the dancer's face,
    because that is exactly where the head sits in a well-framed shot. The bands
    above (rows < y_top) and below (rows >= y_bot) the video are ours to use.
    Font sizes follow the panel width so the HUD stays proportional as the
    window is resized.
    """
    h, w = canvas.shape[:2]
    hint, tier = hud.get("hint"), hud.get("tier")

    if hint:                                     # a prompt replaces the tier word
        fs = max(0.45, min(1.0, w / 620.0))
        (tw, th), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.putText(canvas, hint, ((w - tw) // 2, (y_top + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (60, 215, 255), 2, cv2.LINE_AA)
    elif tier:
        fs = max(0.8, min(2.4, w / 300.0))
        (tw, th), _ = cv2.getTextSize(tier, cv2.FONT_HERSHEY_DUPLEX, fs, 3)
        x, by = (w - tw) // 2, (y_top + th) // 2
        cv2.putText(canvas, tier, (x + 2, by + 2), cv2.FONT_HERSHEY_DUPLEX,
                    fs, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(canvas, tier, (x, by), cv2.FONT_HERSHEY_DUPLEX,
                    fs, TIER_COLORS[tier], 3, cv2.LINE_AA)

    band = h - y_bot
    score, running = hud.get("score"), hud.get("running")
    if score is not None:
        fs = max(0.6, min(1.8, w / 400.0))
        s = f"{score:.0f}"
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
        cv2.putText(canvas, s, (w - tw - 14, y_bot + (band + th) // 2),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    if running is not None:
        fs = max(0.4, min(0.9, w / 700.0))
        s = f"avg {running:.0f}"
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.putText(canvas, s, (14, y_bot + (band + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 235, 235), 1, cv2.LINE_AA)


class JustDanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Just Dance")
        self.root.geometry("820x760")

        npz, vid = find_ref()
        if npz is None:
            messagebox.showerror(
                "Missing reference",
                f"Run:  python precompute_reference.py {REF_NAME}\n"
                "to generate the reference skeleton first.")
            root.after(100, root.destroy)
            return
        d = np.load(npz)
        self.ref_xy, self.ref_valid, self.ref_t = d["xy"], d["valid"], d["t"]
        self.ref_fps = float(d["fps"])
        self.ref_video = vid
        self.scorer = PoseScorer(self.ref_xy, self.ref_valid, self.ref_t)

        self.tracker_cam = PoseTracker(MODEL, imgsz=CAM_IMGSZ, smooth_alpha=CAM_SMOOTH)
        self.running_cam = False
        self.dance_active = False
        self.t_play = 0.0            # current reference time (display only)
        self.speed = 1.0            # reference playback speed (set at Start Dance)
        self.play_t0 = None          # wall clock at which the dance started
        self.play_speed = 1.0
        self.speed_var = tk.StringVar(value="1.0x")

        self.q = queue.Queue(maxsize=4)

        # Panel size follows the window. The worker threads read these two ints
        # when they letterbox a frame, so resizing needs no restart.
        self.disp_w, self.disp_h = DISPLAY_W, DISPLAY_H

        # layout: controls pinned to the bottom, two equal panels filling the rest
        ctrl = tk.Frame(root)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, pady=6)
        inner = tk.Frame(ctrl); inner.pack()
        tk.Button(inner, text="Start Webcam", command=self.start_cam).pack(side=tk.LEFT, padx=4)
        tk.Button(inner, text="Start Dance", command=self.start_dance).pack(side=tk.LEFT, padx=4)
        tk.Button(inner, text="Stop Dance", command=self.stop_dance).pack(side=tk.LEFT, padx=4)
        tk.Label(inner, text="Speed").pack(side=tk.LEFT, padx=(10, 2))
        tk.OptionMenu(inner, self.speed_var, "1.0x", "0.75x", "0.5x").pack(side=tk.LEFT)

        body = tk.Frame(root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # grid_propagate off breaks the feedback loop "bigger image -> label asks
        # for more room -> window grows -> bigger image"; the cells are sized by
        # the window alone, and uniform= keeps the two panels identical.
        body.grid_propagate(False)
        body.columnconfigure(0, weight=1, uniform="panel")
        body.columnconfigure(1, weight=1, uniform="panel")
        body.rowconfigure(1, weight=1)
        tk.Label(body, text="Reference").grid(row=0, column=0)
        tk.Label(body, text="You").grid(row=0, column=1)
        self.label_ref = tk.Label(body, bg="black")
        self.label_ref.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 4))
        self.label_cam = tk.Label(body, bg="black")
        self.label_cam.grid(row=1, column=1, sticky="nsew", padx=6, pady=(0, 4))
        # one panel is enough to watch: uniform columns make them the same size
        self.label_ref.bind("<Configure>", self._on_panel_resize)

        self.root.minsize(480, 380)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._pump()

    def _on_panel_resize(self, ev):
        self.disp_w = max(MIN_DISPLAY, ev.width)
        self.disp_h = max(MIN_DISPLAY, ev.height)

    # ---------------- controls ----------------
    def start_cam(self):
        if not self.running_cam:
            self.running_cam = True
            threading.Thread(target=self.webcam_loop, daemon=True).start()

    def start_dance(self):
        if self.dance_active:
            return
        if not self.running_cam:
            self.start_cam()
        self.speed = float(self.speed_var.get().rstrip("x"))
        self.scorer.reset()
        self.dance_active = True
        threading.Thread(target=self.reference_loop, daemon=True).start()

    def stop_dance(self):
        if self.dance_active:
            self.dance_active = False

    def on_close(self):
        self.running_cam = False
        self.dance_active = False
        self.root.after(150, self.root.destroy)

    # ---------------- reference playback (the clock) ----------------
    def reference_loop(self):
        cap = cv2.VideoCapture(self.ref_video)
        n = len(self.ref_xy)
        fps = self.ref_fps
        speed = max(0.1, self.speed)   # <1 = slow motion, so a learner can keep up
        start = time.perf_counter()
        self.play_t0, self.play_speed = start, speed   # the clock scoring aligns to
        pos = 0                        # index of the next frame to show
        while self.dance_active and pos < n:
            # Wall-clock time this frame is due. Without this wait the loop --
            # which does NO inference, it uses precomputed skeletons -- would
            # spin far faster than the video's fps and fast-forward the dance.
            due = start + pos / (fps * speed)
            now = time.perf_counter()
            if now < due:
                time.sleep(due - now)

            ret, frame = cap.read()
            if not ret:
                break
            self.t_play = float(self.ref_t[pos])   # reference seconds; scorer aligns here
            draw_skeleton(frame, self.ref_xy[pos], self.ref_valid[pos],
                          color_line=(0, 220, 0), color_point=(0, 0, 255))
            self._emit(self.label_ref, frame)
            pos += 1

            # If decoding/drawing fell behind, skip frames to catch back up so
            # playback stays on the clock instead of drifting slow.
            while (pos < n and
                   time.perf_counter() - (start + pos / (fps * speed)) > 1.0 / fps):
                if not cap.grab():
                    break
                pos += 1
        cap.release()
        self.dance_active = False
        self.play_t0 = None
        self._show_summary()

    def ref_time_at(self, t_cap):
        """Reference time matching a webcam frame *captured* at t_cap.

        Scoring used to compare against self.t_play -- where the reference is
        right now -- even though the frame had already spent an inference time
        in the pipeline. That pits "you, 150 ms ago" against "the reference,
        now", and the scorer quietly absorbs the difference into its human-lag
        estimate, biasing every frame. Deriving the target from the capture
        stamp takes the pipeline delay out of the alignment completely.
        """
        t0 = self.play_t0
        if t0 is None:
            return self.t_play
        return float(np.clip((t_cap - t0) * self.play_speed, 0.0, float(self.ref_t[-1])))

    # ---------------- webcam + scoring ----------------
    def webcam_loop(self):
        feed = CameraFeed(0)
        if not feed.ok:
            self.running_cam = False
            messagebox.showwarning("No webcam", "Could not open the camera.")
            return
        last_stamp = -1.0
        while self.running_cam:
            frame, stamp = feed.latest()
            if frame is None or stamp == last_stamp:
                if not feed.running:
                    break
                time.sleep(0.002)          # nothing new yet; never re-score a frame
                continue
            last_stamp = stamp
            frame = cv2.flip(frame, 1)
            res = self.tracker_cam.update(frame)
            draw_skeleton(frame, res.xy, res.valid,
                          color_line=(0, 255, 255), color_point=(0, 0, 255)) \
                if res.found else None

            # The feedback is no longer painted onto the frame: it is handed to
            # _emit, which draws it in the bands around the video (see draw_hud).
            body_ok = res.found and scorable(res.valid)
            hud = None
            if self.dance_active:
                t_ref = self.ref_time_at(stamp)   # align to when this frame was shot
                if body_ok:
                    r = self.scorer.update(res.xy, res.valid, t_ref)
                    hud = dict(tier=r["tier"], score=r["score"],
                               running=self.scorer.summary()["mean"])
                else:
                    self.scorer.update(None, None, t_ref)
                    hud = dict(hint="STEP BACK - show your whole body")
            elif not body_ok:
                hud = dict(hint="Step back so your whole body is visible")
            self._emit(self.label_cam, frame, hud)
        feed.close()

    def _show_summary(self):
        s = self.scorer.summary()
        if s["frames"] == 0:
            return
        pts = s["tiers"]
        msg = (f"Overall: {s['mean']:.0f} / 100   ({s['grade']})\n\n"
               f"PERFECT : {pts['PERFECT']}\n"
               f"SUPER   : {pts['SUPER']}\n"
               f"GOOD    : {pts['GOOD']}\n"
               f"X       : {pts['X']}\n"
               f"scored frames: {s['frames']}")
        self.root.after(0, lambda: messagebox.showinfo("Dance finished", msg))

    # ---------------- Tk plumbing ----------------
    def _emit(self, label, frame_bgr, hud=None):
        # Reserve a band above and below the video for the HUD instead of
        # relying on whatever letterbox bars the aspect ratio happens to leave:
        # a panel that matches the video aspect would leave none at all. Both
        # panels reserve the same bands, so the two dancers stay the same size
        # and line up vertically.
        w, h = self.disp_w, self.disp_h
        top = int(min(96, max(34, h * HUD_TOP)))
        bot = int(min(76, max(28, h * HUD_BOT)))
        inner = max(40, h - top - bot)
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[top:top + inner] = fit_letterbox(frame_bgr, w, inner)
        if hud:
            draw_hud(canvas, top, top + inner, hud)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        try:
            self.q.put_nowait((label, img))
        except queue.Full:
            pass

    def _pump(self):
        try:
            while True:
                label, img = self.q.get_nowait()
                tkimg = ImageTk.PhotoImage(image=img)
                label.imgtk = tkimg
                label.configure(image=tkimg)
        except queue.Empty:
            pass
        self.root.after(15, self._pump)


if __name__ == "__main__":
    root = tk.Tk()
    app = JustDanceApp(root)
    root.mainloop()
