"""Scrolling perspective background for Temple Run.

The forward-motion illusion in an endless runner comes from the ground and walls
scrolling toward the camera, NOT from a background video. So we:

  * generate seamless, tileable stone/sand textures procedurally (no assets), and
  * sample them through a perspective remap whose depth coordinate scrolls every
    frame -> tiles rush toward the player and shrink toward the vanishing point.

A real photo, if dropped in as assets/backdrop.* (or sky.*), is used ONLY for the
region above the horizon -- distant scenery barely moves, so a static image there
looks right while the motion is carried by the scrolling floor/walls.

cv2.remap with BORDER_WRAP does the tiling and the seamless scroll for free.
"""

import os

import cv2
import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ----------------------------------------------------------------------
# procedural seamless tiles
# ----------------------------------------------------------------------
def _tileable_noise(size, freqs, rng):
    """Sum of full-period sines -> value noise that wraps seamlessly. Diagonal
    terms break the obvious grid look."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    n = np.zeros((size, size), np.float32)
    for f, a in freqs:
        p = rng.uniform(0, 2 * np.pi, 3)
        n += a * np.sin(2 * np.pi * f * xx / size + p[0]) * \
                 np.sin(2 * np.pi * f * yy / size + p[1])
        n += a * 0.5 * np.sin(2 * np.pi * f * (xx + yy) / size + p[2])
    return n


def make_stone_tile(size=192, base=(120, 138, 138), mortar=(60, 95, 72),
                    cols=4, rows=4, seed=1, moss=(60, 115, 80), moss_amt=0.55):
    """A seamless, weathered, mossy stone tile. Everything is built from
    modulo-grids and full-period sines, so it tiles cleanly for BORDER_WRAP."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    img = np.empty((size, size, 3), np.float32)
    img[:] = base

    # weathering: multi-octave tonal variation across the stone
    weather = _tileable_noise(size, [(1, 0.5), (2, 0.3), (4, 0.18),
                                     (8, 0.10), (16, 0.06)], rng)
    img *= (1.0 + 0.20 * weather)[..., None]

    # brick grid with alternating half-brick row offset (even rows -> tiles)
    bw, bh = size // cols, size // rows
    row = (yy // bh).astype(np.int32)
    offset = (row % 2) * (bw // 2)
    gx = ((xx + offset) % bw)
    gy = (yy % bh)
    edge = np.minimum(np.minimum(gx, bw - gx), np.minimum(gy, bh - gy))  # dist to seam
    line = edge < 2

    # per-brick tint, modulo brick index so it survives wrapping
    bidx = (row % rows) * cols + (((xx + offset) // bw).astype(np.int32) % cols)
    img *= (1.0 + 0.07 * np.sin(bidx * 2.399))[..., None]

    # moss: patchy field, thicker in the grooves between stones
    mn = _tileable_noise(size, [(3, 0.6), (5, 0.35), (7, 0.22), (11, 0.12)], rng)
    mn = (mn - mn.min()) / (mn.max() - mn.min() + 1e-6)
    crack = np.clip(1.0 - edge / 6.0, 0, 1)                 # near seams -> moss
    field = np.clip(mn * 0.85 + crack * 0.5, 0, 1)
    m = np.clip((field - 0.45) / 0.28, 0, 1)
    m = (m * m * (3 - 2 * m) * moss_amt)[..., None]         # smoothstep
    img = img * (1 - m) + np.array(moss, np.float32) * m

    img[line] = mortar
    return np.clip(img, 0, 255).astype(np.uint8)


def load_backdrop(width, height):
    """Real photo for the far distance, resized to (height, width). None if absent."""
    for name in ("backdrop.png", "backdrop.jpg", "backdrop.jpeg", "sky.png", "sky.jpg"):
        p = os.path.join(ASSET_DIR, name)
        if not os.path.exists(p):
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.resize(img, (width, height))
    return None


# ----------------------------------------------------------------------
# perspective scene
# ----------------------------------------------------------------------
class SceneRenderer:
    def __init__(self, W, H, CX, HORIZON_Y, BOTTOM_Y, CORRIDOR_HALF, K, palette):
        self.W, self.H = W, H
        self.CX, self.HZ, self.BY = CX, HORIZON_Y, BOTTOM_Y
        self.CH, self.K = CORRIDOR_HALF, K
        self.P = palette

        # mossy grey flagstones for the path, greener mossier stone for the sides
        self.floor_tile = make_stone_tile(base=(125, 140, 140), mortar=(62, 98, 74),
                                          cols=4, rows=4, seed=3,
                                          moss=(60, 120, 82), moss_amt=0.5)
        self.wall_tile = make_stone_tile(base=(74, 104, 86), mortar=(42, 60, 48),
                                         cols=3, rows=6, seed=7,
                                         moss=(55, 105, 72), moss_amt=0.7)
        self.backdrop = load_backdrop(W, HORIZON_Y)
        self._precompute()

    def _precompute(self):
        # everything the runner sees lives in the band below the horizon; only
        # compute/remap there each frame (rows [HZ, BY)) to keep the cost low.
        self.b0, self.b1 = self.HZ, self.BY           # band row range
        BH = self.b1 - self.b0
        W = self.W
        ys, xs = np.mgrid[self.b0:self.b1, 0:W].astype(np.float32)
        t = np.clip((ys - self.BY) / (self.HZ - self.BY), 0.0, 0.999)
        hw = np.maximum(self.CH * (1.0 - t), 1e-3)
        u = (xs - self.CX) / hw
        z = self.K * t / (1.0 - t)

        FT = self.floor_tile.shape[0]
        WT = self.wall_tile.shape[0]
        ACROSS, DENS = 3.0, 5.5
        fogcol = np.array(self.P["FOG"], np.float32)

        # floor and wall together tile the whole band, so we composite with one
        # masked copy and fog the band once -> two remaps, one blend.
        self.floor_mask8 = ((np.abs(u) <= 1.0) * 255).astype(np.uint8)
        self.floor_mx = ((u * 0.5 + 0.5) * FT * ACROSS).astype(np.float32)
        self.floor_bv = (z * DENS).astype(np.float32)

        left_edge, right_edge = self.CX - hw, self.CX + hw
        dist_edge = np.where(xs < self.CX, left_edge - xs, xs - right_edge)
        self.wall_mx = ((dist_edge / self.CH) * WT * 1.5).astype(np.float32)
        self.wall_bv = (z * DENS).astype(np.float32)

        # 3-channel and contiguous so the per-frame fog is two cv2 calls; the
        # equivalent numpy expression measured ~3.5x slower on the same data.
        fog = np.clip(t * 1.15, 0, 0.82)[..., None]
        self.band_inv = np.ascontiguousarray(
            np.repeat((1.0 - fog).astype(np.float32), 3, axis=2))
        self.band_fogadd = np.ascontiguousarray((fogcol * fog).astype(np.float32))

        # static sky / backdrop above the horizon, built once
        self._sky = np.empty((self.HZ, W, 3), np.uint8)
        if self.backdrop is not None:
            self._sky[:] = self.backdrop
        else:
            for y in range(self.HZ):
                self._sky[y] = _lerp(self.P["SKY_TOP"], self.P["SKY_HORIZON"], y / self.HZ)

    def background(self, scroll):
        c = np.empty((self.H, self.W, 3), np.uint8)
        c[:self.HZ] = self._sky
        floor = cv2.remap(self.floor_tile, self.floor_mx, self.floor_bv + scroll,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        wall = cv2.remap(self.wall_tile, self.wall_mx, self.wall_bv + scroll,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        band = cv2.copyTo(floor, self.floor_mask8, wall)     # floor over wall
        band = cv2.multiply(band, self.band_inv, dtype=cv2.CV_32F)
        c[self.b0:self.b1] = cv2.add(band, self.band_fogadd, dtype=cv2.CV_8U)
        return c
