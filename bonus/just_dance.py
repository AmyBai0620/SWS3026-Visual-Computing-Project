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

from pose_pipeline import PoseTracker, draw_skeleton
from pose_score import PoseScorer, TIER_POINTS, scorable

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "yolov8n-pose.pt")
REF_NAME = "dance_example_5"

DISPLAY_W, DISPLAY_H = 360, 640    # portrait reference video

TIER_COLORS = {   # BGR
    "PERFECT": (60, 215, 255),     # gold
    "SUPER": (90, 230, 90),        # green
    "GOOD": (240, 180, 60),        # blue
    "X": (150, 150, 150),          # grey
}


def find_ref():
    for base in (HERE, os.path.join(HERE, "video")):
        npz = os.path.join(base, f"ref_{REF_NAME}.npz")
        vid = os.path.join(base, f"{REF_NAME}.mp4")
        if os.path.exists(npz) and os.path.exists(vid):
            return npz, vid
    return None, None


def draw_hint(frame, text):
    """Centered prompt, e.g. when the dancer is not fully in frame."""
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x, y = (w - tw) // 2, h // 2
    cv2.rectangle(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (60, 215, 255), 2, cv2.LINE_AA)
    return frame


def draw_score_overlay(frame, tier, score, running):
    """Big feedback text + numeric score, Just-Dance style."""
    h, w = frame.shape[:2]
    if tier:
        color = TIER_COLORS[tier]
        (tw, th), _ = cv2.getTextSize(tier, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)
        x = (w - tw) // 2
        cv2.putText(frame, tier, (x + 2, 70 + 2), cv2.FONT_HERSHEY_DUPLEX,
                    1.6, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, tier, (x, 70), cv2.FONT_HERSHEY_DUPLEX,
                    1.6, color, 3, cv2.LINE_AA)
    if score is not None:
        cv2.putText(frame, f"{score:4.0f}", (w - 120, h - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    if running is not None:
        cv2.putText(frame, f"avg {running:4.0f}", (12, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


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

        self.tracker_cam = PoseTracker(MODEL)
        self.running_cam = False
        self.dance_active = False
        self.t_play = 0.0            # current reference time, drives scoring

        self.q = queue.Queue(maxsize=4)

        # layout
        left = tk.Frame(root); left.pack(side=tk.LEFT, padx=8, pady=6)
        right = tk.Frame(root); right.pack(side=tk.RIGHT, padx=8, pady=6)
        tk.Label(left, text="Reference").pack()
        self.label_ref = tk.Label(left); self.label_ref.pack()
        tk.Label(right, text="You").pack()
        self.label_cam = tk.Label(right); self.label_cam.pack()

        ctrl = tk.Frame(root); ctrl.place(relx=0.5, rely=0.97, anchor="s")
        tk.Button(ctrl, text="Start Webcam", command=self.start_cam).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="Start Dance", command=self.start_dance).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="Stop Dance", command=self.stop_dance).pack(side=tk.LEFT, padx=4)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._pump()

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
        start = time.perf_counter()
        pos = 0                       # next frame index held by the decoder
        while self.dance_active:
            target = int((time.perf_counter() - start) * self.ref_fps)
            if target >= n:
                break
            # advance sequentially to the wall-clock target (skip, don't seek)
            while pos < target:
                if not cap.grab():
                    target = pos
                    break
                pos += 1
            ret, frame = cap.read()
            if not ret:
                break
            idx = min(pos, n - 1)
            pos += 1
            self.t_play = float(self.ref_t[idx])
            draw_skeleton(frame, self.ref_xy[idx], self.ref_valid[idx],
                          color_line=(0, 220, 0), color_point=(0, 0, 255))
            self._emit(self.label_ref, frame)
            time.sleep(0.003)
        cap.release()
        self.dance_active = False
        self._show_summary()

    # ---------------- webcam + scoring ----------------
    def webcam_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.running_cam = False
            messagebox.showwarning("No webcam", "Could not open the camera.")
            return
        while self.running_cam:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            res = self.tracker_cam.update(frame)
            draw_skeleton(frame, res.xy, res.valid,
                          color_line=(0, 255, 255), color_point=(0, 0, 255)) \
                if res.found else None

            body_ok = res.found and scorable(res.valid)
            if self.dance_active:
                if body_ok:
                    r = self.scorer.update(res.xy, res.valid, self.t_play)
                    run = self.scorer.summary()["mean"]
                    draw_score_overlay(frame, r["tier"], r["score"], run)
                else:
                    self.scorer.update(None, None, self.t_play)
                    draw_hint(frame, "STEP BACK - show your whole body")
            elif not body_ok:
                draw_hint(frame, "Step back so your whole body is visible")
            self._emit(self.label_cam, frame)
        cap.release()

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
    def _emit(self, label, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((DISPLAY_W, DISPLAY_H))
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
