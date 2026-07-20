"""Build a single overview image of all Haar-vs-MediaPipe robustness scenarios.

Each tile = one scenario screenshot (Haar left / MediaPipe right) with a banner
showing the scenario name and both detection rates. Haar rate turns red when the
detector effectively failed (<50%). Output: robustness_overview.png
"""
import csv
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent

# Nicer English labels for the presentation
NICE = {
    "baseline_frontal": "Baseline (frontal)",
    "turn_left_30": "Turn left ~30",
    "turn_left_60": "Turn left ~60",
    "look_up_down": "Look up / down",
    "cover_mouth": "Cover mouth",
    "cover_one_eye": "Cover one eye",
    "dim_light": "Dim light",
    "backlight": "Backlight",
    "close_up": "Close-up",
    "far_away": "Far away",
}

TILE_W = 620            # scaled screenshot width per tile
BANNER_H = 58           # height of the text banner above each screenshot
COLS = 2
PAD = 12                # padding between/around tiles
BG = (245, 245, 245)    # light gray canvas


def load_rates():
    rates = {}
    with open(BASE / "results.csv", newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            rates[row["scenario"]] = (
                float(row["haar_hit_rate_%"]),
                float(row["mp_hit_rate_%"]),
            )
    return rates


def make_tile(name, haar, mp):
    shot = cv2.imread(str(BASE / f"shot_{name}.png"))
    if shot is None:
        raise FileNotFoundError(f"shot_{name}.png not found")
    h, w = shot.shape[:2]
    scaled = cv2.resize(shot, (TILE_W, int(h * TILE_W / w)))
    sh = scaled.shape[0]

    tile = np.full((BANNER_H + sh, TILE_W, 3), 255, np.uint8)
    tile[BANNER_H:] = scaled

    # banner
    cv2.rectangle(tile, (0, 0), (TILE_W, BANNER_H), (40, 40, 40), -1)
    cv2.putText(tile, NICE.get(name, name), (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    haar_color = (60, 60, 220) if haar < 50 else (90, 200, 90)   # BGR: red vs green
    cv2.putText(tile, f"Haar: {haar:.0f}%", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, haar_color, 2, cv2.LINE_AA)
    cv2.putText(tile, f"MediaPipe: {mp:.0f}%", (200, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 200, 90), 2, cv2.LINE_AA)
    return tile


def main():
    rates = load_rates()
    order = list(NICE.keys())
    tiles = [make_tile(n, *rates[n]) for n in order if n in rates]

    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + COLS - 1) // COLS
    canvas_w = COLS * tw + (COLS + 1) * PAD
    canvas_h = rows * th + (rows + 1) * PAD
    canvas = np.full((canvas_h, canvas_w, 3), BG, np.uint8)

    for i, tile in enumerate(tiles):
        r, c = divmod(i, COLS)
        y = PAD + r * (th + PAD)
        x = PAD + c * (tw + PAD)
        canvas[y:y + th, x:x + tw] = tile

    out = BASE / "robustness_overview.png"
    cv2.imwrite(str(out), canvas)
    print(f"Saved {out}  ({canvas_w}x{canvas_h})")


if __name__ == "__main__":
    main()
