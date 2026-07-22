"""Bonus Level Task 1 -- reference-video pose pipeline with a Tk GUI.

Differences from the provided danceapp.py:

  * uses PoseTracker (confidence filtering, main-dancer selection, smoothing)
  * plays the video at its own frame rate instead of at inference speed, by
    dropping frames the CPU cannot keep up with
  * worker threads never touch Tk widgets; frames are handed to the main loop
    through a queue, which is the supported way to use Tkinter
  * a small HUD reports people detected, inference time and effective fps

The original danceapp.py is kept unchanged for the before/after demo.
"""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from pose_pipeline import PoseTracker, draw_pose, fit_letterbox

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "yolov8n-pose.pt")

DISPLAY_W, DISPLAY_H = 480, 270


def hud(frame, res, eff_fps):
    """Bottom status strip: what the pipeline is doing right now."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
    state = "no dancer" if not res.found else ("held" if res.held else "tracking")
    n_valid = int(res.valid.sum())
    text = (f"{state} | people {res.n_people} | kpts {n_valid}/17 | "
            f"{res.inference_ms:.0f} ms | {eff_fps:.1f} fps")
    cv2.putText(frame, text, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stickman Dance GUI (v2)")
        self.root.geometry("1060x420")

        self.running_file = False
        self.running_cam = False
        # Prefer the video/ folder; fall back to the bundled sample.
        default_video = os.path.join(HERE, "video", "dance_example_2.mp4")
        if not os.path.exists(default_video):
            default_video = os.path.join(HERE, "dance_example_1.mp4")
        self.video_path = default_video

        # One tracker per source: they hold independent temporal state.
        self.tracker_file = PoseTracker(MODEL)
        self.tracker_cam = PoseTracker(MODEL)

        # Worker threads push (label, image) here; only the Tk main loop pops.
        self.q = queue.Queue(maxsize=4)

        # Both panels expand with the window so enlarging it fills the space.
        left = tk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=6)
        right = tk.Frame(root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=6)

        tk.Label(left, text="Reference video").pack(side=tk.TOP)
        cf = tk.Frame(left)
        cf.pack(side=tk.BOTTOM, pady=4)
        tk.Button(cf, text="Open Video", command=self.load_video).pack(side=tk.LEFT, padx=3)
        tk.Button(cf, text="Start", command=self.start_video).pack(side=tk.LEFT, padx=3)
        tk.Button(cf, text="Stop", command=self.stop_video).pack(side=tk.LEFT, padx=3)
        self.label_file = tk.Label(left, bg="black")
        self.label_file.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        tk.Label(right, text="Webcam").pack(side=tk.TOP)
        cc = tk.Frame(right)
        cc.pack(side=tk.BOTTOM, pady=4)
        tk.Button(cc, text="Start Webcam", command=self.start_cam).pack(side=tk.LEFT, padx=3)
        tk.Button(cc, text="Stop Webcam", command=self.stop_cam).pack(side=tk.LEFT, padx=3)
        self.label_cam = tk.Label(right, bg="black")
        self.label_cam.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._pump()

    # ---------------- controls ----------------
    def load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if path:
            self.video_path = path
            messagebox.showinfo("Video Selected", os.path.basename(path))

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video first.")
            return
        if not self.running_file:
            self.running_file = True
            self.tracker_file.reset()
            threading.Thread(target=self.process_video_file, daemon=True).start()

    def stop_video(self):
        self.running_file = False

    def start_cam(self):
        if not self.running_cam:
            self.running_cam = True
            self.tracker_cam.reset()
            threading.Thread(target=self.process_webcam, daemon=True).start()

    def stop_cam(self):
        self.running_cam = False

    def on_close(self):
        self.running_file = self.running_cam = False
        self.root.after(150, self.root.destroy)

    # ---------------- workers ----------------
    def process_video_file(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.running_file = False
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start = time.perf_counter()
        frame_idx = 0
        shown = 0

        while cap.isOpened() and self.running_file:
            # Keep playback on the video's own clock: if inference fell behind,
            # skip the frames we should already be past instead of playing the
            # whole clip in slow motion.
            target_idx = int((time.perf_counter() - start) * src_fps)
            while frame_idx < target_idx:
                if not cap.grab():          # grab() decodes nothing, so skipping is cheap
                    break
                frame_idx += 1

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            shown += 1

            res = self.tracker_file.update(frame)
            out = draw_pose(frame, res, show_frame=True, show_bbox=True)
            eff = shown / max(1e-6, time.perf_counter() - start)
            self._emit(self.label_file, hud(out, res, eff))

        cap.release()
        self.running_file = False

    def process_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.running_cam = False
            return

        start = time.perf_counter()
        shown = 0
        while cap.isOpened() and self.running_cam:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)      # mirror, so the user can follow along
            shown += 1

            res = self.tracker_cam.update(frame)
            out = draw_pose(frame, res, show_frame=True, show_bbox=True)
            eff = shown / max(1e-6, time.perf_counter() - start)
            self._emit(self.label_cam, hud(out, res, eff))

        cap.release()
        self.running_cam = False

    # ---------------- Tk plumbing ----------------
    def _emit(self, label, frame_bgr):
        # Hand the raw frame to the main loop; letterboxing to the *current*
        # panel size happens there, where the widget dimensions are known.
        try:
            self.q.put_nowait((label, frame_bgr))
        except queue.Full:
            pass  # display is behind; dropping this frame is the right call

    def _pump(self):
        """Runs on the Tk main loop: the only place widgets are touched."""
        try:
            while True:
                label, frame_bgr = self.q.get_nowait()
                bw, bh = label.winfo_width(), label.winfo_height()
                if bw <= 1 or bh <= 1:            # not laid out yet
                    bw, bh = DISPLAY_W, DISPLAY_H
                fitted = fit_letterbox(frame_bgr, bw, bh)
                rgb = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
                tkimg = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                label.imgtk = tkimg               # keep a reference alive
                label.configure(image=tkimg)
        except queue.Empty:
            pass
        self.root.after(15, self._pump)


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseApp(root)
    root.mainloop()
