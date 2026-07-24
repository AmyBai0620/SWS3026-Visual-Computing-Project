"""Bonus Level Task 2, extra activity -- Temple-Run-style body-controlled game.

Answers rubric Q16: "besides dancing, what other actions can you detect and
score? How do you render new objects, detect the new action, and compute score?"

WHAT WE DETECT (three gestures, all from the shoulder keypoints so it still
works when only your upper body is in frame):

  * LEAN left / right  -> move between three lanes
  * JUMP  (shoulders rise above the calibrated baseline)   -> clear a low hurdle
  * DUCK  (shoulders drop below the baseline)               -> pass an overhang

HOW WE DETECT IT: at startup we calibrate a standing baseline (median shoulder
center + shoulder width) over ~1 s of "stand still". Every frame the shoulder
center is measured relative to that baseline, in units of shoulder width so it
is distance-invariant:
    dx = (sx - base_x) / sw   -> lane   (|dx| > LANE_TH)
    up = (base_y - sy) / sw   -> jump    (up > MOVE_TH)
    dn = (sy - base_y) / sw   -> duck    (dn > MOVE_TH)

HOW WE RENDER: a first-person perspective corridor. The world is modelled in
depth z (0 = at the player, large = far). Everything is projected through a
vanishing point so lanes converge, the floor scrolls, and obstacles grow as
they rush toward you -- the Temple-Run look.

GAME FEEL (what makes it read as a real game rather than a demo):
  * the runner slides between lanes with an eased, banking motion instead of
    teleporting, and the camera pans/bobs with him;
  * hits are forgiving -- an obstacle is not judged on the single frame it
    crosses the player plane but over a window around it (EARLY_GRACE before,
    LATE_GRACE after), because a webcam gesture is never frame-accurate;
  * every event has feedback: coin sparks and a rising "+10", clear/near-miss
    call-outs, a combo multiplier, impact debris, camera shake, slow motion and
    a red vignette on a hit, speed streaks as the run accelerates.

HOW WE SCORE: +distance every frame, +COIN_PTS per coin, +CLEAR_PTS per dodged
obstacle, all multiplied by a combo multiplier that grows while you stay clean
and resets when you are hit; a hit costs a life (3 total). The results screen
counts the breakdown up on game over.

Run:  python temple_run.py     (SPACE start/pause, R restart, Q quit)
"""

import math
import os
import random
import time
from collections import deque

import cv2
import numpy as np

from pose_pipeline import PoseTracker, draw_skeleton, fit_letterbox
from temple_bg import SceneRenderer

MODEL = "yolov8n-pose.pt"
HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(HERE, "assets")

# gesture thresholds, in units of shoulder width
LANE_TH = 0.45
MOVE_TH = 0.34            # position threshold for jump/duck (lowered for responsiveness)
VEL_TH = 1.1             # shoulder-widths per second: catches the takeoff instantly
HOLD_S = 0.35            # latch a jump/duck this long so a brief peak still counts
RECOVER_S = 0.15         # extra dead time for the OPPOSITE velocity trigger

# scoring
COIN_PTS = 10
CLEAR_PTS = 5
NEAR_PTS = 3              # bonus for a last-moment lane dodge
START_LIVES = 3
COMBO_STEP = 4            # clears per multiplier step
COMBO_MAX = 4             # multiplier cap
COMBO_TIME = 4.0          # combo drops if nothing is cleared for this long

# judging windows (seconds) -- a webcam gesture is never frame-accurate
EARLY_GRACE = 0.30        # a jump/duck this long BEFORE the hit still counts
LATE_GRACE = 0.16         # ...and this long after; the verdict waits that long
LANE_BACK = 0.12          # lane history window looked back at the hit instant

GAME_W, GAME_H = 640, 720

# ---- perspective / world ----
CX = GAME_W // 2
HORIZON_Y = 210                 # vanishing height
BOTTOM_Y = GAME_H
CORRIDOR_HALF = 300             # half floor width at the near plane
LANE_OFF = 190                  # lane center x-offset at the near plane
K = 46.0                        # perspective falloff constant
MAX_DEPTH = 115.0               # obstacles spawn here
HIT_DEPTH = 6.0                 # resolved when they reach the player plane
PLAYER_Z = 4.0                  # the player sits just in front of HIT_DEPTH

# ---- runner feel ----
LANE_TAU = 0.075          # lane-change easing time constant (s)
TILT_GAIN = 11.0          # how hard he banks into a lane change
MAX_TILT = 17.0           # degrees
JUMP_H = 132.0            # apex height in px at the near plane
RISE_TAU = 0.055          # how fast he leaves the ground
FALL_SPEED = 900.0        # px/s coming down
DUCK_SQUASH = 0.60
SQUASH_TAU = 0.05

# ---- camera ----
CAM_FOLLOW = 0.20         # fraction of the runner's offset the camera pans with
CAM_BOB = 2.6             # vertical bob amplitude (px)
SHAKE_TIME = 0.42
SHAKE_AMP = 15.0
SLOWMO_TIME = 0.30
SLOWMO_FACTOR = 0.42

# ---- temple palette (BGR) -- "ancient path over a sea of clouds" ----
SKY_TOP = (175, 140, 95)        # soft blue high up
SKY_HORIZON = (150, 190, 235)   # warm sunlit glow near the horizon
FLOOR_NEAR = (110, 130, 130)    # mossy grey stone
FLOOR_FAR = (200, 205, 205)     # bright hazy distance (clouds)
WALL_COLOR = (70, 95, 80)       # green mossy ruins
LANE_LINE = (150, 180, 190)
FOG = (205, 210, 210)           # distance fades into bright cloud haze
COIN_GOLD = (60, 200, 250)
BARRIER = (60, 60, 205)         # stone block in one lane (switch lane)
HURDLE = (40, 130, 200)         # low log (JUMP)
OVERHANG = (150, 90, 175)       # high beam (DUCK)
DUST = (150, 165, 170)
BLOOD = (50, 50, 235)


TEX_SCROLL = 2.6                # texture rows per depth-unit of travel

_PALETTE = dict(FOG=FOG, SKY_TOP=SKY_TOP, SKY_HORIZON=SKY_HORIZON, FLOOR_FAR=FLOOR_FAR)
SCENE = SceneRenderer(GAME_W, GAME_H, CX, HORIZON_Y, BOTTOM_Y, CORRIDOR_HALF, K, _PALETTE)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def project(lane_off, z):
    """World (lane offset in [-1,1], depth z>=0) -> (screen x, y, scale, t).

    t is the 0(near)..1(far) depth fraction; scale = 1-t shrinks far things.
    """
    t = z / (z + K)
    scale = 1.0 - t
    sx = CX + lane_off * LANE_OFF * scale
    sy = BOTTOM_Y + (HORIZON_Y - BOTTOM_Y) * t
    return sx, sy, scale, t


def _make_vignette(color):
    """Black over the middle, `color` at the rim -- added on top for a flash.

    The falloff starts late and rises steeply on purpose: a soft, wide vignette
    just bleaches the whole frame instead of reading as an impact at the edges.
    """
    yy, xx = np.mgrid[0:GAME_H, 0:GAME_W].astype(np.float32)
    dx = (xx - GAME_W / 2.0) / (GAME_W / 2.0)
    dy = (yy - GAME_H / 2.0) / (GAME_H / 2.0)
    r = np.sqrt(dx * dx + dy * dy) / 1.4142
    m = np.clip((r - 0.55) / 0.45, 0, 1) ** 2.0
    return (np.array(color, np.float32)[None, None, :] * m[..., None]).astype(np.uint8)


VIG_RED = _make_vignette(BLOOD)
VIG_GOLD = _make_vignette(COIN_GOLD)


# ----------------------------------------------------------------------
# sprite assets (real PNGs in assets/, with graceful fallback to drawing)
# ----------------------------------------------------------------------
class Sprite:
    """A tightly-cropped RGBA image, pre-downscaled once for cheap per-frame use.

    `alpha` is kept as a 3-channel float32 so the per-frame composite can run
    entirely through OpenCV's SIMD paths (see blit)."""

    def __init__(self, rgb, alpha):
        self.rgb = rgb
        self.alpha = cv2.merge([alpha, alpha, alpha])
        self.h, self.w = rgb.shape[:2]
        self.aspect = self.w / self.h


# a slab of solid fog colour, sliced to size so the distance haze is one cv2 call
_FOG_SLAB = np.empty((900, 1100, 3), np.uint8)
_FOG_SLAB[:] = FOG


def _load_sprite(filename, maxdim=380):
    path = os.path.join(ASSET_DIR, filename)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    a = img[:, :, 3]
    ys, xs = np.where(a > 10)
    if len(xs) == 0:
        return None
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # crop to content
    h, w = img.shape[:2]
    sc = min(1.0, maxdim / max(h, w))
    if sc < 1.0:
        img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    return Sprite(img[:, :, :3].copy(), img[:, :, 3].astype(np.float32) / 255.0)


# key -> possible filenames (first that exists wins)
_ASSET_FILES = {
    "player_run": ["player_run.png"],
    "player_jump": ["player_jump.png"],
    "player_slide": ["player_slide(duck).png", "player_slide.png"],
    "barrier": ["obstacle_wall(pillar).png", "obstacle_wall.png", "obstacle_barrier.png"],
    "hurdle": ["obstacle_low.png", "obstacle_hurdle.png"],
    "overhang": ["obstacle_high.png", "obstacle_overhang.png"],
    "coin": ["coin.png"],
    "life": ["life.png"],
    "hit": ["icon_hit.png"],
}


def _load_all():
    spr = {}
    for key, names in _ASSET_FILES.items():
        s = None
        for nm in names:
            s = _load_sprite(nm)
            if s is not None:
                break
        spr[key] = s
    return spr


SPR = _load_all()


def blit(canvas, spr, cx, cy, target_h=None, target_w=None,
         anchor="bottom", fog=0.0, rot=0.0, alpha=1.0):
    """Alpha-composite a sprite onto canvas, scaled, rotated, fogged and clipped.

    anchor: 'bottom' (cy is the base line), 'center', or 'top'.
    rot:    degrees, positive = counter-clockwise (used for the lane-change bank).
    """
    if spr is None:
        return False
    if target_h and not target_w:
        target_w = max(1, int(round(target_h * spr.aspect)))
    elif target_w and not target_h:
        target_h = max(1, int(round(target_w / spr.aspect)))
    if not target_h or target_w < 2 or target_h < 2:
        return False

    rgb = cv2.resize(spr.rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    al = cv2.resize(spr.alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if abs(rot) > 0.5:
        # pad first so the corners survive the rotation, then keep the anchor
        # on the ORIGINAL content by shifting cy down by the pad we added.
        # (|rot| stays under MAX_TILT, so a 16% margin is enough -- every extra
        # pixel of padding is paid for again in the composite below.)
        pad = int(0.16 * max(target_w, target_h))
        M = cv2.getRotationMatrix2D((target_w / 2.0, target_h * 0.85), rot, 1.0)
        M[0, 2] += pad
        M[1, 2] += pad
        size = (target_w + 2 * pad, target_h + 2 * pad)
        rgb = cv2.warpAffine(rgb, M, size, flags=cv2.INTER_LINEAR)
        al = cv2.warpAffine(al, M, size, flags=cv2.INTER_LINEAR)
        target_w, target_h = size
        if anchor == "bottom":
            cy += pad

    if fog > 0 and target_h <= _FOG_SLAB.shape[0] and target_w <= _FOG_SLAB.shape[1]:
        cv2.addWeighted(rgb, 1.0 - fog, _FOG_SLAB[:target_h, :target_w], fog, 0, dst=rgb)
        al *= (1.0 - 0.6 * fog)
    if alpha < 1.0:
        al *= max(0.0, alpha)

    x0 = int(round(cx - target_w / 2))
    if anchor == "bottom":
        y0 = int(round(cy - target_h))
    elif anchor == "center":
        y0 = int(round(cy - target_h / 2))
    else:
        y0 = int(round(cy))

    H, W = canvas.shape[:2]
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + target_w), min(H, y0 + target_h)
    if xs1 <= xs0 or ys1 <= ys0:
        return True
    sx0, sy0 = xs0 - x0, ys0 - y0
    sub = canvas[ys0:ys1, xs0:xs1]
    a = al[sy0:sy0 + (ys1 - ys0), sx0:sx0 + (xs1 - xs0)]
    fg = rgb[sy0:sy0 + (ys1 - ys0), sx0:sx0 + (xs1 - xs0)]
    # sub + (fg - sub) * a, through OpenCV rather than numpy: the equivalent
    # numpy expression allocates four full-size float temporaries and measured
    # ~6x slower, which at this sprite size is milliseconds per frame.
    d = cv2.subtract(fg, sub, dtype=cv2.CV_32F)
    cv2.multiply(d, a, dst=d)
    sub[:] = cv2.add(d, sub, dtype=cv2.CV_8U)
    return True


def _alpha_text(c, text, org, scale, color, thick, alpha, font=cv2.FONT_HERSHEY_DUPLEX):
    """putText with a real alpha, by blending only the text's bounding box."""
    if alpha <= 0.02:
        return
    alpha = min(1.0, alpha)
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = int(org[0]), int(org[1])
    x0, y0 = max(0, x - 3), max(0, y - th - 6)
    x1, y1 = min(c.shape[1], x + tw + 3), min(c.shape[0], y + bl + 4)
    if x1 <= x0 or y1 <= y0:
        return
    roi = c[y0:y1, x0:x1]
    ov = roi.copy()
    cv2.putText(ov, text, (x - x0, y - y0), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(ov, text, (x - x0, y - y0), font, scale, color, thick, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _center_text(c, text, y, scale, color, thick, alpha=1.0):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    _alpha_text(c, text, ((c.shape[1] - tw) // 2, y), scale, color, thick, alpha)


# ----------------------------------------------------------------------
# effects: particles, floating score text, shockwave rings
# ----------------------------------------------------------------------
class FX:
    """Tiny particle/text system. Everything fades by shrinking or by alpha, so
    it costs a handful of cv2 primitives per frame -- no full-frame blends."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.parts = []     # [x, y, vx, vy, g, life, life0, size, color]
        self.texts = []     # [x, y, vy, life, life0, text, color, scale]
        self.rings = []     # [x, y, r0, r1, life, life0, color, thick]

    def burst(self, x, y, n, color, spd=(90, 300), size=(2, 6),
              life=(0.30, 0.65), g=760.0, spread=math.pi, r0=(4, 22)):
        # particles start on a small ring, not on a single point: a burst that
        # spawns all-coincident reads as one dot on the frame it appears.
        for _ in range(n):
            a = random.uniform(-math.pi / 2 - spread / 2, -math.pi / 2 + spread / 2)
            v = random.uniform(*spd)
            lf = random.uniform(*life)
            d = random.uniform(*r0)
            self.parts.append([x + math.cos(a) * d, y + math.sin(a) * d,
                               math.cos(a) * v, math.sin(a) * v,
                               g, lf, lf, random.uniform(*size), color])

    def puff(self, x, y, n=3):
        self.burst(x, y, n, DUST, spd=(30, 90), size=(2, 4),
                   life=(0.20, 0.40), g=-40.0, spread=math.pi * 0.7, r0=(2, 10))

    def text(self, x, y, s, color, scale=0.85, life=0.9, vy=-70.0):
        self.texts.append([float(x), float(y), vy, life, life, s, color, scale])

    def ring(self, x, y, r0, r1, color, life=0.35, thick=3):
        self.rings.append([float(x), float(y), r0, r1, life, life, color, thick])

    def update(self, dt):
        for p in self.parts:
            p[5] -= dt
            p[3] += p[4] * dt
            p[0] += p[2] * dt
            p[1] += p[3] * dt
        self.parts = [p for p in self.parts if p[5] > 0]
        for t in self.texts:
            t[3] -= dt
            t[1] += t[2] * dt
            t[2] *= (1.0 - 1.6 * dt)
        self.texts = [t for t in self.texts if t[3] > 0]
        for r in self.rings:
            r[4] -= dt
        self.rings = [r for r in self.rings if r[4] > 0]

    def draw(self, c):
        for x, y, r0, r1, life, life0, color, thick in self.rings:
            f = 1.0 - life / life0
            r = int(r0 + (r1 - r0) * f)
            if r > 0:
                cv2.circle(c, (int(x), int(y)), r,
                           lerp(color, FOG, min(0.85, f)), max(1, int(thick * (1 - f) + 1)),
                           cv2.LINE_AA)
        for x, y, _vx, _vy, _g, life, life0, size, color in self.parts:
            f = life / life0
            r = max(1, int(size * f))
            cv2.circle(c, (int(x), int(y)), r, color, -1, cv2.LINE_AA)
        for x, y, _vy, life, life0, s, color, scale in self.texts:
            f = life / life0
            a = min(1.0, f * 1.8)                      # hold, then fade out
            pop = 1.0 + 0.25 * max(0.0, 1.0 - (1 - f) * 6.0)
            (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, scale * pop, 2)
            _alpha_text(c, s, (x - tw / 2, y), scale * pop, color, 2, a)


class Camera:
    """Screen-space camera: pans with the runner, bobs with speed, shakes on hit."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.dx = 0.0
        self.dy = 0.0
        self.shake = 0.0
        self.bob = 0.0

    def hit(self):
        self.shake = SHAKE_TIME

    def update(self, dt, player_x, speed):
        self.shake = max(0.0, self.shake - dt)
        self.bob = (self.bob + dt * speed * 0.32) % (2 * math.pi)
        # follow: the camera pans a fraction of the way toward the runner's lane
        self.dx = -CAM_FOLLOW * player_x * LANE_OFF
        self.dy = CAM_BOB * math.sin(self.bob * 2.0)
        if self.shake > 0:
            f = (self.shake / SHAKE_TIME) ** 1.5
            self.dx += random.uniform(-1, 1) * SHAKE_AMP * f
            self.dy += random.uniform(-1, 1) * SHAKE_AMP * f

    def apply(self, c):
        """Shift the world by whole pixels (a strided copy + edge replicate).

        cv2.warpAffine would give sub-pixel smoothness for ~2.5 ms/frame; at
        30 fps and these amplitudes nobody can see the difference."""
        dx, dy = int(round(self.dx)), int(round(self.dy))
        H, W = c.shape[:2]
        if dx == 0 and dy == 0:
            return c
        if abs(dx) >= W or abs(dy) >= H:
            return c
        out = np.empty_like(c)
        w, h = W - abs(dx), H - abs(dy)
        out[max(dy, 0):max(dy, 0) + h, max(dx, 0):max(dx, 0) + w] = \
            c[max(-dy, 0):max(-dy, 0) + h, max(-dx, 0):max(-dx, 0) + w]
        if dx > 0:
            out[:, :dx] = out[:, dx:dx + 1]
        elif dx < 0:
            out[:, W + dx:] = out[:, W + dx - 1:W + dx]
        if dy > 0:                       # rows last, so the corners fill in too
            out[:dy] = out[dy:dy + 1]
        elif dy < 0:
            out[H + dy:] = out[H + dy - 1:H + dy]
        return out


class GestureDetector:
    """Turns shoulder geometry into (lane, action) after a standing calibration."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._cal = []
        self.base = None
        self.ready = False
        self._prev_yn = None        # previous normalized shoulder-y
        self._prev_t = None
        self._latch = "run"         # currently latched jump/duck
        self._latch_until = 0.0

    def _shoulders(self, res):
        if not res.found or not (res.valid[5] and res.valid[6]):
            return None
        c = (res.xy[5] + res.xy[6]) / 2.0
        w = float(np.linalg.norm(res.xy[5] - res.xy[6]))
        return c[0], c[1], w

    def update(self, res):
        idle = dict(lane=1, action="run", dx=0.0, dy=0.0, vy=0.0, seen=False)
        s = self._shoulders(res)
        if s is None:
            idle.update(ready=self.ready, calibrating=not self.ready)
            return idle

        if not self.ready:
            self._cal.append(s)
            if len(self._cal) >= 25:
                arr = np.array(self._cal)
                self.base = (np.median(arr[:, 0]), np.median(arr[:, 1]),
                             max(1.0, np.median(arr[:, 2])))
                self.ready = True
            idle.update(ready=self.ready, calibrating=True, seen=True,
                        progress=len(self._cal) / 25.0)
            return idle

        now = time.perf_counter()
        bx, by, _ = self.base
        sw = max(1.0, s[2])
        dx = (s[0] - bx) / sw
        up = (by - s[1]) / sw            # >0 means shoulders above baseline (jumped)
        dn = (s[1] - by) / sw

        # vertical velocity in shoulder-widths/sec (up positive) -> instant takeoff detect
        yn = s[1] / sw
        vy = 0.0
        if self._prev_yn is not None and self._prev_t is not None:
            dt = max(1e-3, now - self._prev_t)
            vy = (self._prev_yn - yn) / dt
        self._prev_yn, self._prev_t = yn, now

        # Detect a fresh jump/duck from either position or velocity, then latch.
        #
        # The velocity trigger only counts when the POSITION does not contradict
        # it. Without that gate, coming down from a jump is a fast downward
        # shoulder move and fires a DUCK (and rising out of a duck fires a
        # JUMP): the recovery half of every gesture produced its opposite.
        # ...and, for the same reason, a velocity trigger is ignored while the
        # OPPOSITE gesture is still unwinding (the latch outlives the pose by
        # HOLD_S, which is exactly the window the recovery falls in).
        gate = 0.5 * MOVE_TH
        recovering = now < self._latch_until + RECOVER_S
        raw = "run"
        if up > MOVE_TH or (vy > VEL_TH and dn < gate
                            and not (recovering and self._latch == "duck")):
            raw = "jump"
        elif dn > MOVE_TH or (vy < -VEL_TH and up < gate
                              and not (recovering and self._latch == "jump")):
            raw = "duck"
        if raw != "run":
            self._latch = raw
            self._latch_until = now + HOLD_S
        action = self._latch if now < self._latch_until else "run"

        lane = 0 if dx < -LANE_TH else (2 if dx > LANE_TH else 1)
        # dx / dy / vy are returned so the camera panel can *show* the rule that
        # produced the decision -- the demo explains itself on screen.
        return dict(lane=lane, action=action, ready=True, calibrating=False,
                    dx=dx, dy=up, vy=vy, seen=True)


class Obstacle:
    __slots__ = ("kind", "lane", "z", "phase", "t_hit", "resolved", "passed")

    def __init__(self, kind, lane, z):
        self.kind = kind        # 'barrier' | 'wall2' | 'hurdle' | 'overhang' | 'coin'
        self.lane = lane        # 0,1,2 (lane offset -1,0,1)
        self.z = z
        self.phase = random.uniform(0, 6.28)
        self.t_hit = None       # time it crossed the player plane
        self.resolved = False
        self.passed = False     # True once judged, whatever the verdict


class Player:
    """Screen-space state of the runner: eased lane x, bank, jump arc, squash."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = 0.0            # lane offset in [-1, 1], continuous
        self.vx = 0.0
        self.tilt = 0.0
        self.lift = 0.0         # px above the ground plane
        self.squash = 1.0
        self.state = "run"
        self.landed = 0.0       # landing-squash timer
        self._dust_t = 0.0

    def update(self, dt, lane, action, fx, ground_xy):
        target = float((-1.0, 0.0, 1.0)[lane])
        k = 1.0 - math.exp(-dt / LANE_TAU)
        prev = self.x
        self.x += (target - self.x) * k
        self.vx = (self.x - prev) / max(dt, 1e-3)
        self.tilt = max(-MAX_TILT, min(MAX_TILT, -self.vx * TILT_GAIN))

        was_air = self.lift > 1.0
        if action == "jump":
            self.lift += (JUMP_H - self.lift) * (1.0 - math.exp(-dt / RISE_TAU))
        else:
            self.lift = max(0.0, self.lift - FALL_SPEED * dt)
        if was_air and self.lift <= 1.0:                    # touchdown
            self.landed = 0.16
            fx.puff(ground_xy[0], ground_xy[1], 6)
        self.landed = max(0.0, self.landed - dt)

        sq_target = DUCK_SQUASH if action == "duck" else 1.0
        if self.landed > 0:
            sq_target *= 0.86
        self.squash += (sq_target - self.squash) * (1.0 - math.exp(-dt / SQUASH_TAU))
        self.state = action

        # footfall dust while running on the ground
        self._dust_t -= dt
        if action == "run" and self.lift <= 1.0 and self._dust_t <= 0:
            fx.puff(ground_xy[0] + random.uniform(-14, 14), ground_xy[1], 2)
            self._dust_t = 0.09


class Game:
    def __init__(self):
        self.best = 0
        self.fx = FX()
        self.cam = Camera()
        self.player = Player()
        self.reset()

    def reset(self):
        self.obstacles = []
        self.spawn_t = 0.9
        self.speed = 30.0           # depth units / s
        self.distance = 0.0
        self.coins = 0
        self.cleared = 0
        self.bonus = 0              # extra points earned through the combo multiplier
        self.lives = START_LIVES
        self.over = False
        self.new_best = False
        self.flash = 0.0
        self.gold_flash = 0.0
        self.slowmo = 0.0
        self.combo = 0
        self.combo_t = 0.0
        self.best_combo = 0
        self.scroll = 0.0           # 0..8 phase for runner leg-swing / coin spin
        self.tex_scroll = 0.0       # continuous phase for the scrolling ground texture
        self.disp_score = 0.0       # HUD score, eased toward the real one
        self.result_t = 0.0         # results-screen reveal timer
        self.last_jump_t = -9.0
        self.last_duck_t = -9.0
        self.last_lane_t = -9.0
        self.last_kind = None
        self._popup_slot = 0
        self._popup_t = -9.0
        self.lane_hist = deque()
        self.prev_action = "run"
        self.prev_lane = 1
        self.fx.clear()
        self.cam.reset()
        self.player.reset()

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    @property
    def multiplier(self):
        return min(COMBO_MAX, 1 + self.combo // COMBO_STEP)

    @property
    def score(self):
        return (int(self.distance / 10) + self.coins * COIN_PTS
                + self.cleared * CLEAR_PTS + self.bonus)

    def _spawn(self):
        """One spawn event; keeps a minimum spatial gap so the run stays fair."""
        kind = random.choices(["barrier", "wall2", "hurdle", "overhang", "coin"],
                              weights=[3, 2, 2, 2, 3])[0]
        # never chain two reaction-heavy obstacles too tightly
        gap = random.uniform(30.0, 48.0)
        if kind in ("hurdle", "overhang") and self.last_kind in ("hurdle", "overhang"):
            gap = max(gap, 46.0)
        lane = random.randint(0, 2)

        if kind == "coin":
            n = random.randint(3, 5)              # a coin RUN reads far better than one coin
            for i in range(n):
                self.obstacles.append(Obstacle("coin", lane, MAX_DEPTH + i * 7.0))
            gap += n * 7.0
        else:
            # hurdles and overhangs span the whole corridor -- they are cleared by
            # a jump/duck, never by a lane change, so they live in the middle lane.
            if kind in ("hurdle", "overhang"):
                lane = 1
            self.obstacles.append(Obstacle(kind, lane, MAX_DEPTH))

        self.last_kind = kind
        self.spawn_t = gap / max(1.0, self.speed)

    def update(self, dt, lane, action, now):
        if self.over:
            self.result_t += dt
            return
        self.fx.update(dt)

        if self.slowmo > 0:                       # brief impact slow motion
            self.slowmo = max(0.0, self.slowmo - dt)
            dt *= SLOWMO_FACTOR

        # event edges the judging windows are built on
        if action == "jump" and self.prev_action != "jump":
            self.last_jump_t = now
        if action == "duck" and self.prev_action != "duck":
            self.last_duck_t = now
        if lane != self.prev_lane:
            self.last_lane_t = now
        self.prev_action, self.prev_lane = action, lane

        self.lane_hist.append((now, lane))
        while self.lane_hist and now - self.lane_hist[0][0] > 0.6:
            self.lane_hist.popleft()

        self.distance += self.speed * dt * 0.6
        self.scroll = (self.scroll + self.speed * dt) % 8.0
        self.tex_scroll += self.speed * dt * TEX_SCROLL
        self.flash = max(0.0, self.flash - dt)
        self.gold_flash = max(0.0, self.gold_flash - dt)
        self.speed = min(62.0, self.speed + 0.75 * dt)

        if self.combo > 0:
            self.combo_t -= dt
            if self.combo_t <= 0:
                self.combo = 0

        self.spawn_t -= dt
        if self.spawn_t <= 0:
            self._spawn()

        psx, pground, pscale = self.player_screen()
        self.player.update(dt, lane, action, self.fx, (psx, pground))
        self.cam.update(dt, self.player.x, self.speed)
        self.disp_score += (self.score - self.disp_score) * (1.0 - math.exp(-dt / 0.12))

        for ob in self.obstacles:
            ob.z -= self.speed * dt
            if ob.t_hit is None and ob.z <= HIT_DEPTH:
                ob.t_hit = now
            if ob.t_hit is not None and not ob.passed:
                self._judge(ob, lane, action, now)
        self.obstacles = [o for o in self.obstacles if o.z > -6.0]

    # ------------------------------------------------------------------
    # judging: a window around the crossing instant, not a single frame
    # ------------------------------------------------------------------
    def _lane_safe(self, ob, lane):
        if ob.kind == "coin":
            return lane == ob.lane
        if ob.kind == "barrier":
            return lane != ob.lane          # ob.lane is blocked
        if ob.kind == "wall2":
            return lane == ob.lane          # ob.lane is the only SAFE lane
        return False

    def _avoided_now(self, ob, lane, action, now):
        if ob.kind in ("coin", "barrier", "wall2"):
            if self._lane_safe(ob, lane):
                return True
            # ...or we were already safe just before it hit us
            return any(t >= ob.t_hit - LANE_BACK and self._lane_safe(ob, ln)
                       for t, ln in self.lane_hist)
        if ob.kind == "hurdle":
            return action == "jump" or self.last_jump_t >= ob.t_hit - EARLY_GRACE
        return action == "duck" or self.last_duck_t >= ob.t_hit - EARLY_GRACE

    def _judge(self, ob, lane, action, now):
        if self._avoided_now(ob, lane, action, now):
            self._award(ob, now)
        elif now >= ob.t_hit + LATE_GRACE:
            self._fail(ob)

    def _obstacle_screen(self, ob):
        sx, sy, sc, _ = project((-1, 0, 1)[ob.lane], max(ob.z, 0.1))
        return int(sx), int(sy), sc

    def _popup(self, text, color, scale=0.72, life=0.85, x=None):
        """Floating score text, placed ABOVE the runner's head and stacked so a
        coin run does not pile three labels on top of each other."""
        psx, pground, psc = self.player_screen()
        now = time.perf_counter()
        if now - self._popup_t > 0.55:
            self._popup_slot = 0
        self._popup_slot = min(3, self._popup_slot + 1)
        self._popup_t = now
        y = pground - 215 * psc - 32 * (self._popup_slot - 1)
        self.fx.text(psx if x is None else x, y, text, color, scale, life)

    def _bump_combo(self):
        self.combo += 1
        self.best_combo = max(self.best_combo, self.combo)
        self.combo_t = COMBO_TIME
        if self.combo % COMBO_STEP == 0 and self.multiplier > 1:
            self.gold_flash = 0.30
            _, gy, sc = self.player_screen()
            self.fx.text(CX, gy - 330 * sc, f"COMBO x{self.multiplier}", COIN_GOLD,
                         scale=1.0, life=1.0)

    def _award(self, ob, now):
        ob.passed = ob.resolved = True
        sx, sy, sc = self._obstacle_screen(ob)
        mult = self.multiplier

        if ob.kind == "coin":
            self.coins += 1
            self.bonus += COIN_PTS * (mult - 1)
            cy = sy - int(46 * sc)
            self.fx.burst(sx, cy, 10, COIN_GOLD, spd=(80, 240), size=(2, 5), life=(0.25, 0.5))
            self.fx.ring(sx, cy, 6, 42, COIN_GOLD, life=0.28, thick=3)
            self._popup(f"+{COIN_PTS * mult}", COIN_GOLD, scale=0.72, life=0.7)
            self._bump_combo()
            return

        self.cleared += 1
        self.bonus += CLEAR_PTS * (mult - 1)
        near = (ob.kind in ("barrier", "wall2")
                and now - self.last_lane_t < 0.30)
        pts = CLEAR_PTS * mult
        if near:
            self.bonus += NEAR_PTS
            pts += NEAR_PTS
        self._popup(f"{'CLOSE!' if near else 'CLEAR'}  +{pts}",
                    (170, 255, 190) if near else (210, 240, 240), scale=0.7, life=0.75)
        self._bump_combo()

    def _fail(self, ob):
        ob.passed = True
        if ob.kind == "coin":
            return                      # a missed coin is not a mistake, just a miss
        self.lives -= 1
        self.combo = 0
        self.flash = 0.45
        self.slowmo = SLOWMO_TIME
        self.cam.hit()
        psx, pground, _ = self.player_screen()
        self.fx.burst(psx, pground - 80, 24, (70, 80, 210), spd=(140, 420),
                      size=(3, 8), life=(0.35, 0.75), r0=(10, 46))
        self.fx.burst(psx, pground - 80, 12, (200, 210, 215), spd=(90, 300),
                      size=(2, 6), life=(0.30, 0.60), r0=(6, 34))
        self.fx.ring(psx, pground - 80, 14, 140, (90, 110, 255), life=0.32, thick=6)
        if self.lives <= 0:
            self.over = True
            self.new_best = self.score > self.best
            self.best = max(self.best, self.score)
            self.result_t = 0.0

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def player_screen(self):
        """Runner's (x, ground y, scale) on screen, following the eased lane x."""
        sx, sy, sc, _ = project(self.player.x, PLAYER_Z)
        return sx, sy + 8 * sc, sc

    def render(self, lane, action):
        c = SCENE.background(self.tex_scroll)
        self._lane_lines(c)
        for ob in sorted(self.obstacles, key=lambda o: -o.z):    # far first
            self._draw_obstacle(c, ob)
        if self.speed > 38:
            self._speed_streaks(c)
        self._draw_player(c)
        self.fx.draw(c)
        c = self.cam.apply(c)                                    # shake/pan the world

        if self.flash > 0:
            f = self.flash / 0.45
            cv2.addWeighted(VIG_RED, 1.0 * f, c, 1.0 - 0.25 * f, 0, dst=c)   # rim + dim
        elif self.lives == 1 and not self.over:                  # last-life heartbeat
            pulse = 0.30 + 0.22 * math.sin(time.perf_counter() * 7.0)
            cv2.addWeighted(VIG_RED, pulse, c, 1.0, 0, dst=c)
        if self.gold_flash > 0:
            cv2.addWeighted(VIG_GOLD, 0.7 * (self.gold_flash / 0.30), c, 1.0, 0, dst=c)

        self._action_prompt(c)
        self._hud(c)
        return c

    def _speed_streaks(self, c):
        """Radial motion streaks from the vanishing point -- the faster, the more."""
        f = min(1.0, (self.speed - 38.0) / 24.0)
        n = int(6 + 10 * f)
        ov = c.copy()
        rng = random.Random(int(self.tex_scroll * 0.7) * 7919)
        for _ in range(n):
            # only the downward fan: streaks in the sky read as lens scratches
            a = rng.uniform(0.10 * math.pi, 0.90 * math.pi)
            r0 = rng.uniform(150, 330)
            ln = rng.uniform(90, 260) * (0.4 + f)
            x0 = CX + math.cos(a) * r0
            y0 = HORIZON_Y + math.sin(a) * r0 * 0.75
            x1 = CX + math.cos(a) * (r0 + ln)
            y1 = HORIZON_Y + math.sin(a) * (r0 + ln) * 0.75
            cv2.line(ov, (int(x0), int(y0)), (int(x1), int(y1)),
                     (245, 248, 250), 1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.22 * (0.4 + f), c, 1.0 - 0.22 * (0.4 + f), 0, dst=c)

    def _action_prompt(self, c):
        """Big pulsing JUMP/DUCK call-out for the obstacle about to arrive."""
        soon = [o for o in self.obstacles
                if o.kind in ("hurdle", "overhang") and not o.passed and 0 < o.z < 26]
        if not soon:
            return
        ob = min(soon, key=lambda o: o.z)
        urgency = 1.0 - ob.z / 26.0
        txt = "JUMP!" if ob.kind == "hurdle" else "DUCK!"
        col = (90, 210, 255) if ob.kind == "hurdle" else (225, 160, 245)
        scale = 1.0 + 0.35 * urgency + 0.06 * math.sin(time.perf_counter() * 18)
        _center_text(c, txt, GAME_H - 140, scale, col, 3, 0.45 + 0.55 * urgency)

    def _lane_lines(self, c):
        # faint lane dividers over the textured floor, for gameplay clarity
        for off in (-1 / 3, 1 / 3):
            x_near = CX + int(off * 2 * CORRIDOR_HALF)
            cv2.line(c, (x_near, BOTTOM_Y), (CX, HORIZON_Y), LANE_LINE, 1, cv2.LINE_AA)

    def _lane_offset(self, lane):
        return (-1, 0, 1)[lane]

    def _blit_pillar(self, c, off, sc, fog):
        """A tall stone pillar at a given lane offset and depth-scale."""
        t = 1.0 - sc
        sx = int(CX + off * LANE_OFF * sc)
        sy = int(BOTTOM_Y + (HORIZON_Y - BOTTOM_Y) * t)
        if blit(c, SPR["barrier"], sx, sy, target_h=max(8, int(290 * sc)),
                anchor="bottom", fog=fog):
            return
        w = int(66 * sc); h = int(150 * sc); top = sy - h
        col = lerp(BARRIER, FOG, fog)
        cv2.rectangle(c, (sx - w, top), (sx + w, sy), col, -1)
        cv2.rectangle(c, (sx - w, top), (sx + w, sy), lerp(col, (0, 0, 0), 0.3), max(1, int(2 * sc)))

    def _draw_obstacle(self, c, ob):
        if ob.resolved and ob.kind == "coin":
            return                                   # collected: the sparks replace it
        off = self._lane_offset(ob.lane)
        sx, sy, sc, t = project(off, max(ob.z, 0.1))
        fog = min(0.85, t)
        sx = int(sx); sy = int(sy)
        hw = int(CORRIDOR_HALF * sc)

        if ob.kind == "coin":
            bob = int(8 * sc * math.sin(ob.phase + self.scroll * 0.9))
            cy = sy - int(46 * sc) + bob
            if blit(c, SPR["coin"], sx, cy, target_h=max(3, int(56 * sc)),
                    anchor="center", fog=fog):
                return
            r = max(2, int(20 * sc))
            spin = abs(np.cos(ob.phase + self.scroll))
            col = lerp(COIN_GOLD, FOG, fog)
            cv2.ellipse(c, (sx, cy), (max(1, int(r * spin)), r), 0, 0, 360, col, -1)
            cv2.ellipse(c, (sx, cy), (max(1, int(r * spin)), r), 0, 0, 360, (30, 90, 130), 1)
            return

        if ob.kind == "barrier":       # tall pillar in one lane -> switch lane
            self._blit_pillar(c, off, sc, fog)
            return

        if ob.kind == "wall2":         # two lanes blocked -> move to the safe one
            for bl in (0, 1, 2):
                if bl != ob.lane:      # ob.lane is the safe lane
                    self._blit_pillar(c, self._lane_offset(bl), sc, fog)
            return

        # hurdle / overhang span the WHOLE corridor: they are centred on the
        # vanishing point, not on a lane, because no lane is safe from them.
        # The close-range call-out is drawn once by _action_prompt, so the small
        # per-object tag is only for the ones still far away.
        far = ob.z > 26
        if ob.kind == "hurdle":        # low log spanning the corridor -> JUMP
            if blit(c, SPR["hurdle"], CX, sy + int(6 * sc),
                    target_w=int(2.05 * hw), anchor="bottom", fog=fog):
                if far:
                    _tag(c, "JUMP", CX, sy - int(30 * sc), sc, (90, 200, 250))
                return
            col = lerp(HURDLE, FOG, fog); hh = max(2, int(16 * sc))
            cv2.rectangle(c, (CX - hw, sy - hh), (CX + hw, sy), col, -1)
            if far:
                _tag(c, "JUMP", CX, sy - hh - int(6 * sc), sc, col)
            return

        # overhang arch spanning the corridor -> DUCK under the top beam
        if blit(c, SPR["overhang"], CX, sy,
                target_w=int(2.1 * hw), anchor="bottom", fog=fog):
            if far:
                _tag(c, "DUCK", CX, sy - int(150 * sc), sc, (210, 150, 235))
            return
        col = lerp(OVERHANG, FOG, fog); yb = sy - int(120 * sc); hh = max(2, int(18 * sc))
        cv2.rectangle(c, (CX - hw, yb - hh), (CX + hw, yb), col, -1)
        if far:
            _tag(c, "DUCK", CX, yb - hh - int(6 * sc), sc, col)

    def _draw_player(self, c):
        p = self.player
        sx, ground, s = self.player_screen()
        sx = int(sx); ground = int(ground)
        lift = int(p.lift * s)

        # ground shadow: shrinks and fades as he leaves the ground
        air = min(1.0, p.lift / JUMP_H)
        shw = int(46 * s * (1.0 - 0.45 * air))
        shh = max(3, int(11 * s * (1.0 - 0.4 * air)))
        cv2.ellipse(c, (sx, ground), (shw, shh), 0, 0, 360,
                    lerp((25, 25, 35), FLOOR_NEAR, 0.55 * air), -1, cv2.LINE_AA)

        key = {"jump": "player_jump", "duck": "player_slide"}.get(p.state, "player_run")
        spr = SPR.get(key)
        if spr is not None:
            base_h = 150.0 if p.state == "duck" else 178.0
            h = int(base_h * s * (p.squash if p.state != "duck" else 1.0))
            blit(c, spr, sx, ground - lift, target_h=max(4, h),
                 anchor="bottom", rot=p.tilt)
            return

        cy = ground - lift
        _draw_runner(c, sx, cy, s, p.squash * (DUCK_SQUASH if p.state == "duck" else 1.0),
                     self.scroll, p.state)

    def _hud(self, c):
        c[:52] = cv2.multiply(c[:52], (0.5, 0.5, 0.5, 0.0))   # darken the strip only

        txt = str(int(self.disp_score))
        cv2.putText(c, txt, (12, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        if self.best:
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.95, 2)
            cv2.putText(c, f"BEST {self.best}", (18 + tw, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (165, 195, 205), 1, cv2.LINE_AA)

        # combo multiplier, center, pulsing while alive
        if self.multiplier > 1:
            f = max(0.0, self.combo_t / COMBO_TIME)
            sc = 0.72 + 0.10 * math.sin(time.perf_counter() * 8)
            (tw, _), _ = cv2.getTextSize(f"x{self.multiplier}", cv2.FONT_HERSHEY_DUPLEX, sc, 2)
            cv2.putText(c, f"x{self.multiplier}", (CX - tw // 2, 36),
                        cv2.FONT_HERSHEY_DUPLEX, sc, COIN_GOLD, 2, cv2.LINE_AA)
            cv2.rectangle(c, (CX - 40, 44), (CX - 40 + int(80 * f), 47), COIN_GOLD, -1)

        # coins + lives, right
        cv2.putText(c, f"{self.coins}", (GAME_W - 60, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, COIN_GOLD, 2, cv2.LINE_AA)
        if not blit(c, SPR.get("coin"), GAME_W - 82, 28, target_h=26, anchor="center"):
            cv2.circle(c, (GAME_W - 82, 28), 10, COIN_GOLD, -1)

        heart = SPR.get("life")
        for i in range(START_LIVES):
            x = GAME_W - 250 + i * 30
            alive = i < self.lives
            pulse = 1.0
            if alive and self.lives == 1:
                pulse = 1.0 + 0.18 * math.sin(time.perf_counter() * 7.0)
            if heart is not None:
                blit(c, heart, x, 28, target_h=int(28 * pulse), anchor="center",
                     fog=0.0 if alive else 0.75)
                continue
            col = (80, 90, 235) if alive else (70, 70, 70)
            cv2.circle(c, (x, 26), 9, col, -1)

        # speed bar along the bottom of the HUD strip
        f = min(1.0, (self.speed - 30.0) / 32.0)
        cv2.rectangle(c, (0, 50), (int(GAME_W * f), 52), lerp((120, 200, 120), (60, 90, 245), f), -1)

    def draw_results(self, c):
        """Coin-settlement / score summary screen shown on game over (Q16).

        Rows count up one after another so the recording has a beat to it."""
        ov = c.copy()
        cv2.rectangle(ov, (0, 0), (GAME_W, GAME_H), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.62, c, 0.38, 0, dst=c)

        px0, py0, px1, py1 = 70, 140, GAME_W - 70, GAME_H - 140
        cv2.rectangle(c, (px0, py0), (px1, py1), (35, 30, 28), -1)
        cv2.rectangle(c, (px0, py0), (px1, py1), (90, 150, 200), 3)

        _center_text(c, "RUN COMPLETE", py0 + 58, 1.15, (90, 210, 250), 3)

        rows = [
            ("Coins", f"{self.coins}  x{COIN_PTS}", self.coins * COIN_PTS, COIN_GOLD),
            ("Obstacles", f"{self.cleared}  x{CLEAR_PTS}", self.cleared * CLEAR_PTS, (200, 220, 220)),
            ("Distance", f"{int(self.distance)}m", int(self.distance / 10), (200, 220, 220)),
            ("Combo bonus", f"best x{min(COMBO_MAX, 1 + self.best_combo // COMBO_STEP)}",
             self.bonus, (150, 235, 255)),
        ]
        y = py0 + 120
        for i, (name, mid, pts, col) in enumerate(rows):
            rev = min(1.0, max(0.0, (self.result_t - 0.25 * i) / 0.35))   # staggered reveal
            if rev <= 0:
                y += 46
                continue
            _alpha_text(c, name, (px0 + 40, y), 0.7, col, 2, rev, cv2.FONT_HERSHEY_SIMPLEX)
            _alpha_text(c, mid, (px0 + 210, y), 0.7, (170, 170, 170), 2, rev, cv2.FONT_HERSHEY_SIMPLEX)
            pt = f"+{int(pts * rev)}"
            (tw, _), _ = cv2.getTextSize(pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            _alpha_text(c, pt, (px1 - 40 - tw, y), 0.7, col, 2, rev, cv2.FONT_HERSHEY_SIMPLEX)
            y += 46

        cv2.line(c, (px0 + 40, y), (px1 - 40, y), (90, 90, 90), 1)
        y += 52
        tot = min(1.0, max(0.0, (self.result_t - 1.0) / 0.5))
        _center_text(c, f"TOTAL  {int(self.score * tot)}", y, 1.25, (255, 255, 255), 3, tot)
        if self.result_t > 1.6 and self.new_best:
            _center_text(c, "NEW BEST!", y + 42, 0.8, COIN_GOLD, 2,
                         0.55 + 0.45 * math.sin(time.perf_counter() * 6))
        _center_text(c, "SPACE  play again      Q  quit", py1 - 22, 0.6, (200, 200, 200), 2)
        return c


def _tag(c, text, cx, y, sc, col):
    if sc < 0.45:
        return
    fs = 0.5 * sc + 0.25
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
    cv2.putText(c, text, (int(cx - tw / 2), int(y)), cv2.FONT_HERSHEY_DUPLEX,
                fs, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_runner(c, x, y, s, squash, scroll, action):
    """A little explorer seen from behind."""
    skin = (120, 165, 205)
    shirt = (40, 110, 225)        # orange-red
    pants = (70, 60, 55)
    pack = (60, 140, 90)
    bh = int(70 * s * squash)     # body height
    bw = int(26 * s)
    # legs (swinging while running)
    swing = int(10 * s * np.sin(scroll * 2)) if action == "run" else 0
    hipy = y
    cv2.line(c, (x - int(9 * s), hipy), (x - int(9 * s) + swing, hipy + int(34 * s * squash)), pants, max(2, int(9 * s)))
    cv2.line(c, (x + int(9 * s), hipy), (x + int(9 * s) - swing, hipy + int(34 * s * squash)), pants, max(2, int(9 * s)))
    # torso
    ty = hipy - bh
    cv2.rectangle(c, (x - bw, ty), (x + bw, hipy), shirt, -1)
    cv2.rectangle(c, (x - bw, ty), (x + bw, hipy), lerp(shirt, (0, 0, 0), 0.3), 1)
    # backpack
    cv2.rectangle(c, (x - int(16 * s), ty + int(6 * s)), (x + int(16 * s), ty + int(38 * s * squash)), pack, -1)
    # arms
    aswing = int(12 * s * np.cos(scroll * 2)) if action == "run" else 0
    ay = ty + int(10 * s)
    cv2.line(c, (x - bw, ay), (x - bw - int(6 * s), ay + int(26 * s) + aswing), shirt, max(2, int(7 * s)))
    cv2.line(c, (x + bw, ay), (x + bw + int(6 * s), ay + int(26 * s) - aswing), shirt, max(2, int(7 * s)))
    # head
    cv2.circle(c, (x, ty - int(15 * s)), int(16 * s), skin, -1)
    cv2.circle(c, (x, ty - int(22 * s)), int(17 * s), pants, -1)  # hat/hair top


def _banner(canvas, text, sub=None):
    h, w = canvas.shape[:2]
    ov = canvas.copy()
    cv2.rectangle(ov, (0, h // 2 - 70), (w, h // 2 + 60), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, canvas, 0.45, 0, dst=canvas)
    _center_text(canvas, text, h // 2, 1.2, (60, 215, 255), 3)
    if sub:
        (sw2, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(canvas, sub, ((w - sw2) // 2, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)


def _countdown(canvas, elapsed):
    """3 - 2 - 1 - GO!, each number punching in. Returns True when finished."""
    step = int(elapsed / 0.7)
    if step >= 4:
        return True
    word = ["3", "2", "1", "GO!"][step]
    f = (elapsed - step * 0.7) / 0.7
    scale = (3.4 if step < 3 else 2.6) * (1.35 - 0.35 * min(1.0, f * 3))
    col = (255, 255, 255) if step < 3 else (90, 240, 140)
    _center_text(canvas, word, GAME_H // 2 + 30, scale, col, 5, min(1.0, 2.2 * (1 - f)))
    return False


CAM_W = 480             # side panel width; wider than a strip so the player is big
CAM_VIEW_H = 360        # 480x360 == 4:3, so a landscape webcam fills it with no bars
DASH_BG = (34, 30, 27)


def _gauge(panel, y, label, value, th, neg, pos, active, span=1.3, second=None):
    """Horizontal gauge: a normalised measurement against its threshold ticks.

    Showing the number the rule actually thresholds is what makes the gesture
    detection legible to an audience watching the recording."""
    x0, x1 = 20, CAM_W - 20
    mid = (x0 + x1) // 2
    half = (x1 - x0) / 2.0
    cv2.putText(panel, label, (x0, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (170, 190, 200), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (x0, y), (x1, y + 12), (55, 58, 62), -1)
    for sgn, (nm, col) in ((-1, neg), (1, pos)):
        px = int(mid + sgn * th / span * half)
        cv2.line(panel, (px, y - 4), (px, y + 16), (120, 130, 140), 1)
        cv2.putText(panel, nm, (px - 22 if sgn < 0 else px + 4, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
    if second is not None:
        # the velocity trigger, rescaled so that VEL_TH lands on the SAME tick:
        # both rules share one axis, so a jump latched by speed alone still
        # shows a marker crossing the line.
        sx = int(mid + max(-1.0, min(1.0, second / span)) * half)
        cv2.rectangle(panel, (sx - 2, y + 1), (sx + 2, y + 11), (120, 200, 255), 1)
    px = int(mid + max(-1.0, min(1.0, value / span)) * half)
    cv2.rectangle(panel, (px - 3, y - 4), (px + 3, y + 16),
                  (90, 240, 140) if active else (235, 235, 235), -1)


def _cam_panel(frame, res, g, state):
    """Webcam (native aspect, large) on top; a live gesture dashboard below.

    The game board is portrait while a webcam is landscape, so letterboxing the
    camera into a tall strip drops the player into a band of black bars and
    shrinks them. Instead we show the webcam at 4:3 in the upper block -- filled,
    not cropped, not stretched -- and spend the leftover height on a dashboard
    that visualises the gesture rule that produced the decision (Q16 detection).
    """
    panel = np.empty((GAME_H, CAM_W, 3), np.uint8)
    cam = frame.copy()
    if res.found:
        draw_skeleton(cam, res.xy, res.valid,
                      color_line=(0, 255, 255), color_point=(0, 0, 255))
    panel[:CAM_VIEW_H] = fit_letterbox(cam, CAM_W, CAM_VIEW_H, pad=(24, 22, 20))
    panel[CAM_VIEW_H:] = DASH_BG
    cv2.line(panel, (0, CAM_VIEW_H), (CAM_W, CAM_VIEW_H), (90, 150, 200), 2)

    # state chip over the top-left of the webcam
    panel[:40] = cv2.multiply(panel[:40], (0.45, 0.45, 0.45, 0.0))
    cv2.putText(panel, state, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (60, 215, 255), 2, cv2.LINE_AA)

    act = g["action"].upper()
    acol = {"JUMP": (90, 220, 255), "DUCK": (215, 150, 240)}.get(act, (235, 235, 235))
    y0 = CAM_VIEW_H

    if not g.get("seen"):
        cv2.putText(panel, "no shoulders detected", (18, y0 + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 235), 2, cv2.LINE_AA)
        cv2.putText(panel, "step back so your shoulders are in frame",
                    (18, y0 + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (150, 165, 175), 1, cv2.LINE_AA)
        return panel
    if g.get("calibrating"):
        p = g.get("progress", 0.0)
        cv2.putText(panel, "CALIBRATING", (18, y0 + 58), cv2.FONT_HERSHEY_DUPLEX,
                    0.9, (90, 220, 150), 2, cv2.LINE_AA)
        cv2.rectangle(panel, (18, y0 + 84), (CAM_W - 18, y0 + 108), (55, 58, 62), -1)
        cv2.rectangle(panel, (18, y0 + 84), (18 + int((CAM_W - 36) * p), y0 + 108),
                      (90, 220, 150), -1)
        cv2.putText(panel, "stand straight and still", (18, y0 + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 185, 195), 1, cv2.LINE_AA)
        return panel

    # big current-action word, then the two rule gauges
    (tw, _), _ = cv2.getTextSize(act, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)
    cv2.putText(panel, act, ((CAM_W - tw) // 2, y0 + 60), cv2.FONT_HERSHEY_DUPLEX,
                1.3, acol, 3, cv2.LINE_AA)
    _gauge(panel, y0 + 158, "LEAN   dx / shoulder-width", g["dx"], LANE_TH,
           ("LEFT", (250, 210, 120)), ("RIGHT", (250, 210, 120)), g["lane"] != 1)
    _gauge(panel, y0 + 262, "RISE   dy (solid) + speed (hollow)", g["dy"], MOVE_TH,
           ("DUCK", (215, 150, 240)), ("JUMP", (90, 220, 255)),
           act in ("JUMP", "DUCK"), second=g["vy"] / VEL_TH * MOVE_TH)
    return panel


def main():
    # No temporal smoothing on the game tracker: a jump must register the moment
    # it happens, not a few frames later (smoothing is for the dance scorer).
    tracker = PoseTracker(MODEL, smooth_alpha=1.0)
    gesture = GestureDetector()
    game = Game()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("could not open webcam")

    win = "Temple Run (body controlled)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    prev = time.perf_counter()
    state = "menu"        # menu -> calibrating -> countdown -> playing <-> paused -> over
    count_t0 = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        res = tracker.update(frame)
        g = gesture.update(res)

        now = time.perf_counter()
        dt = min(0.1, now - prev)
        prev = now

        if state == "calibrating" and g["ready"]:
            state, count_t0 = "countdown", now
        if state == "playing":
            game.update(dt, g["lane"], g["action"], now)
            if game.over:
                state = "over"
        elif state == "over":
            game.update(dt, g["lane"], g["action"], now)   # only advances the reveal

        board = game.render(g["lane"], g["action"])
        if state == "menu":
            _banner(board, "TEMPLE RUN", "SPACE to start   |   Q quit")
        elif state == "calibrating":
            _banner(board, "CALIBRATING", "stand straight and still")
        elif state == "countdown":
            if _countdown(board, now - count_t0):
                state = "playing"
        elif state == "paused":
            _banner(board, "PAUSED", "SPACE resume   |   R restart   |   Q quit")
        elif state == "over":
            board = game.draw_results(board)

        cam = _cam_panel(frame, res, g, state.upper())
        cv2.imshow(win, np.hstack([cam, board]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            if state in ("menu", "over"):
                gesture.reset(); game.reset(); state = "calibrating"
            elif state == "playing":
                state = "paused"
            elif state == "paused":
                state = "playing"
        if key in (ord('r'), ord('p')) and state in ("playing", "paused", "over"):
            if key == ord('p'):
                state = "paused" if state == "playing" else "playing"
            else:
                gesture.reset(); game.reset(); state = "calibrating"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
