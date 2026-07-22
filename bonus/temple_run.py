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

HOW WE SCORE: +distance every frame, +COIN_PTS per coin, +CLEAR_PTS per dodged
obstacle; a hit costs a life (3 total). Final score shown on game over.

Run:  python temple_run.py     (Q quit, R restart/recalibrate)
"""

import os
import time
import random

import cv2
import numpy as np

from pose_pipeline import PoseTracker, draw_skeleton
from temple_bg import SceneRenderer

MODEL = "yolov8n-pose.pt"
HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(HERE, "assets")

# gesture thresholds, in units of shoulder width
LANE_TH = 0.45
MOVE_TH = 0.34            # position threshold for jump/duck (lowered for responsiveness)
VEL_TH = 1.1             # shoulder-widths per second: catches the takeoff instantly
HOLD_S = 0.35            # latch a jump/duck this long so a brief peak still counts

# scoring
COIN_PTS = 10
CLEAR_PTS = 5
START_LIVES = 3

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


# ----------------------------------------------------------------------
# sprite assets (real PNGs in assets/, with graceful fallback to drawing)
# ----------------------------------------------------------------------
class Sprite:
    """A tightly-cropped RGBA image, pre-downscaled once for cheap per-frame use."""

    def __init__(self, rgb, alpha):
        self.rgb = rgb
        self.alpha = alpha
        self.h, self.w = rgb.shape[:2]
        self.aspect = self.w / self.h


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
         anchor="bottom", fog=0.0):
    """Alpha-composite a sprite onto canvas, scaled, fog-faded, and clipped.

    anchor: 'bottom' (cy is the base line), 'center', or 'top'.
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
    if fog > 0:
        rgb = (rgb * (1.0 - fog) + np.array(FOG, np.float32) * fog).astype(np.uint8)
        al = al * (1.0 - 0.6 * fog)

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
    a = al[sy0:sy0 + (ys1 - ys0), sx0:sx0 + (xs1 - xs0), None]
    fg = rgb[sy0:sy0 + (ys1 - ys0), sx0:sx0 + (xs1 - xs0)]
    canvas[ys0:ys1, xs0:xs1] = (fg * a + sub * (1.0 - a)).astype(np.uint8)
    return True


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
        s = self._shoulders(res)
        if s is None:
            return dict(lane=1, action="run", ready=self.ready, calibrating=not self.ready)

        if not self.ready:
            self._cal.append(s)
            if len(self._cal) >= 25:
                arr = np.array(self._cal)
                self.base = (np.median(arr[:, 0]), np.median(arr[:, 1]),
                             max(1.0, np.median(arr[:, 2])))
                self.ready = True
            return dict(lane=1, action="run", ready=self.ready, calibrating=True)

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

        # detect a fresh jump/duck from either position or velocity, then latch it
        raw = "run"
        if up > MOVE_TH or vy > VEL_TH:
            raw = "jump"
        elif dn > MOVE_TH or vy < -VEL_TH:
            raw = "duck"
        if raw != "run":
            self._latch = raw
            self._latch_until = now + HOLD_S
        action = self._latch if now < self._latch_until else "run"

        lane = 0 if dx < -LANE_TH else (2 if dx > LANE_TH else 1)
        return dict(lane=lane, action=action, ready=True, calibrating=False)


class Obstacle:
    __slots__ = ("kind", "lane", "z", "scored", "phase")

    def __init__(self, kind, lane, z):
        self.kind = kind        # 'barrier' | 'hurdle' | 'overhang' | 'coin'
        self.lane = lane        # 0,1,2 (lane offset -1,0,1)
        self.z = z
        self.scored = False
        self.phase = random.uniform(0, 6.28)


class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obstacles = []
        self.spawn_t = 0.0
        self.speed = 30.0           # depth units / s
        self.distance = 0.0
        self.coins = 0
        self.cleared = 0
        self.lives = START_LIVES
        self.over = False
        self.flash = 0.0
        self.scroll = 0.0           # 0..8 phase for runner leg-swing / coin spin
        self.tex_scroll = 0.0       # continuous phase for the scrolling ground texture

    @property
    def score(self):
        return int(self.distance / 10) + self.coins * COIN_PTS + self.cleared * CLEAR_PTS

    def _spawn(self):
        kind = random.choices(["barrier", "wall2", "hurdle", "overhang", "coin"],
                              weights=[3, 2, 2, 2, 3])[0]
        if kind == "wall2":
            # two lanes blocked; ob.lane stores the single SAFE lane to move into
            lane = random.randint(0, 2)
        else:
            lane = random.randint(0, 2)
        self.obstacles.append(Obstacle(kind, lane, MAX_DEPTH))

    def update(self, dt, lane, action):
        if self.over:
            return
        self.distance += self.speed * dt * 0.6
        self.scroll = (self.scroll + self.speed * dt) % 8.0
        self.tex_scroll += self.speed * dt * TEX_SCROLL
        self.flash = max(0.0, self.flash - dt)
        self.spawn_t -= dt
        if self.spawn_t <= 0:
            self._spawn()
            self.spawn_t = random.uniform(0.9, 1.5)
        self.speed = min(58.0, self.speed + 0.7 * dt)

        for ob in self.obstacles:
            ob.z -= self.speed * dt
            if not ob.scored and ob.z <= HIT_DEPTH:
                ob.scored = True
                self._resolve(ob, lane, action)
        self.obstacles = [o for o in self.obstacles if o.z > -4.0]

    def _resolve(self, ob, lane, action):
        if ob.kind == "coin":
            if lane == ob.lane:
                self.coins += 1
            return
        if ob.kind == "barrier":
            avoided = lane != ob.lane          # ob.lane is blocked
        elif ob.kind == "wall2":
            avoided = lane == ob.lane          # ob.lane is the only SAFE lane
        elif ob.kind == "hurdle":
            avoided = action == "jump"
        else:  # overhang
            avoided = action == "duck"
        if avoided:
            self.cleared += 1
        else:
            self.lives -= 1
            self.flash = 0.4
            if self.lives <= 0:
                self.over = True

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def render(self, lane, action):
        c = SCENE.background(self.tex_scroll)
        self._lane_lines(c)
        # far-to-near so nearer objects overdraw
        for ob in sorted(self.obstacles, key=lambda o: -o.z):
            self._draw_obstacle(c, ob)
        self._draw_player(c, lane, action)
        if self.flash > 0:
            f = self.flash / 0.4
            red = np.zeros_like(c); red[:] = (0, 0, 220)
            c = cv2.addWeighted(red, 0.30 * f, c, 1, 0)
            # impact burst on the player
            psx, psy, psc, _ = project(self._lane_offset(lane), PLAYER_Z)
            blit(c, SPR.get("hit"), int(psx), int(psy) - int(40 * psc),
                 target_h=int((150 + 60 * (1 - f)) * psc), anchor="center")
        self._hud(c)
        return c

    def draw_results(self, c):
        """Coin-settlement / score summary screen shown on game over (Q16)."""
        ov = c.copy()
        cv2.rectangle(ov, (0, 0), (GAME_W, GAME_H), (0, 0, 0), -1)
        c[:] = cv2.addWeighted(ov, 0.62, c, 0.38, 0)

        px0, py0, px1, py1 = 70, 150, GAME_W - 70, GAME_H - 150
        cv2.rectangle(c, (px0, py0), (px1, py1), (35, 30, 28), -1)
        cv2.rectangle(c, (px0, py0), (px1, py1), (90, 150, 200), 3)

        cx = GAME_W // 2
        _center(c, "RUN COMPLETE", py0 + 58, 1.15, (90, 210, 250), 3)

        rows = [
            ("Coins", f"{self.coins}  x{COIN_PTS}", self.coins * COIN_PTS, COIN_GOLD),
            ("Obstacles", f"{self.cleared}  x{CLEAR_PTS}", self.cleared * CLEAR_PTS, (200, 220, 220)),
            ("Distance", f"{int(self.distance)}m", int(self.distance / 10), (200, 220, 220)),
        ]
        y = py0 + 120
        for name, mid, pts, col in rows:
            cv2.putText(c, name, (px0 + 40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
            cv2.putText(c, mid, (px0 + 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 2, cv2.LINE_AA)
            pt = f"+{pts}"
            (tw, _), _ = cv2.getTextSize(pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(c, pt, (px1 - 40 - tw, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
            y += 46
        # coin icon by the Coins row
        blit(c, SPR.get("coin"), px0 + 165, py0 + 114, target_h=30, anchor="center")

        cv2.line(c, (px0 + 40, y), (px1 - 40, y), (90, 90, 90), 1)
        y += 56
        _center(c, f"TOTAL  {self.score}", y, 1.25, (255, 255, 255), 3)
        _center(c, "SPACE  play again      Q  quit", py1 - 30, 0.6, (200, 200, 200), 2)
        return c

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
        off = self._lane_offset(ob.lane)
        sx, sy, sc, t = project(off, max(ob.z, 0.1))
        fog = min(0.85, t)
        sx = int(sx); sy = int(sy)
        hw = int(CORRIDOR_HALF * sc)

        if ob.kind == "coin":
            if blit(c, SPR["coin"], sx, sy - int(46 * sc),
                    target_h=max(3, int(56 * sc)), anchor="center", fog=fog):
                return
            r = max(2, int(20 * sc))
            spin = abs(np.cos(ob.phase + self.scroll))
            col = lerp(COIN_GOLD, FOG, fog)
            cv2.ellipse(c, (sx, sy - int(34 * sc)), (max(1, int(r * spin)), r), 0, 0, 360, col, -1)
            cv2.ellipse(c, (sx, sy - int(34 * sc)), (max(1, int(r * spin)), r), 0, 0, 360, (30, 90, 130), 1)
            return

        if ob.kind == "barrier":       # tall pillar in one lane -> switch lane
            self._blit_pillar(c, off, sc, fog)
            return

        if ob.kind == "wall2":         # two lanes blocked -> move to the safe one
            for bl in (0, 1, 2):
                if bl != ob.lane:      # ob.lane is the safe lane
                    self._blit_pillar(c, self._lane_offset(bl), sc, fog)
            return

        if ob.kind == "hurdle":        # low log spanning the corridor -> JUMP
            if blit(c, SPR["hurdle"], sx, sy + int(6 * sc),
                    target_w=int(2.05 * hw), anchor="bottom", fog=fog):
                _tag(c, "JUMP", sx, sy - int(30 * sc), sc, (90, 200, 250))
                return
            col = lerp(HURDLE, FOG, fog); hh = max(2, int(16 * sc))
            cv2.rectangle(c, (sx - hw, sy - hh), (sx + hw, sy), col, -1)
            _tag(c, "JUMP", sx, sy - hh - int(6 * sc), sc, col)
            return

        # overhang arch spanning the corridor -> DUCK under the top beam
        if blit(c, SPR["overhang"], sx, sy,
                target_w=int(2.1 * hw), anchor="bottom", fog=fog):
            _tag(c, "DUCK", sx, sy - int(150 * sc), sc, (210, 150, 235))
            return
        col = lerp(OVERHANG, FOG, fog); yb = sy - int(120 * sc); hh = max(2, int(18 * sc))
        cv2.rectangle(c, (sx - hw, yb - hh), (sx + hw, yb), col, -1)
        _tag(c, "DUCK", sx, yb - hh - int(6 * sc), sc, col)

    def _draw_player(self, c, lane, action):
        off = self._lane_offset(lane)
        sx, sy, sc, _ = project(off, PLAYER_Z)
        sx = int(sx); base = int(sy)
        s = sc                    # near-plane scale (~1)
        ground = base + int(8 * s)    # feet line, kept clear of the frame edge

        # ground shadow first, so the runner sits on top of it
        lift = int(46 * s) if action == "jump" else 0
        shw = int(46 * s * (0.65 if action == "jump" else 1.0))
        cv2.ellipse(c, (sx, ground), (shw, int(11 * s)), 0, 0, 360, (25, 25, 35), -1)

        key = {"jump": "player_jump", "duck": "player_slide"}.get(action, "player_run")
        spr = SPR.get(key)
        if spr is not None:
            # all three anchored feet-on-ground; jump is lifted off it. Heights
            # tuned so the character stays the same apparent size across poses.
            if action == "duck":
                blit(c, spr, sx, ground, target_h=int(150 * s), anchor="bottom")
            elif action == "jump":
                blit(c, spr, sx, ground - lift, target_h=int(178 * s), anchor="bottom")
            else:
                blit(c, spr, sx, ground, target_h=int(178 * s), anchor="bottom")
            return

        cy = base - lift if action == "jump" else (base + int(6 * s) if action == "duck" else base)
        _draw_runner(c, sx, cy, s, 0.6 if action == "duck" else 1.0, self.scroll, action)

    def _hud(self, c):
        ov = c.copy()
        cv2.rectangle(ov, (0, 0), (GAME_W, 46), (0, 0, 0), -1)
        c[:] = cv2.addWeighted(ov, 0.5, c, 0.5, 0)
        cv2.putText(c, f"SCORE {self.score}", (12, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(c, f"COINS {self.coins}", (GAME_W - 165, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, COIN_GOLD, 2, cv2.LINE_AA)
        heart = SPR.get("life")
        for i in range(START_LIVES):
            x = GAME_W // 2 - 34 + i * 30
            if i < self.lives and blit(c, heart, x, 34, target_h=30, anchor="center"):
                continue
            if i >= self.lives and heart is not None:
                blit(c, heart, x, 34, target_h=30, anchor="center", fog=0.75)
                continue
            col = (80, 90, 235) if i < self.lives else (70, 70, 70)
            cv2.circle(c, (x, 22), 9, col, -1)


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


def _center(canvas, text, y, scale, color, thick):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    cv2.putText(canvas, text, ((canvas.shape[1] - tw) // 2, y),
                cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


def _banner(canvas, text, sub=None):
    h, w = canvas.shape[:2]
    ov = canvas.copy()
    cv2.rectangle(ov, (0, h // 2 - 70), (w, h // 2 + 60), (0, 0, 0), -1)
    canvas[:] = cv2.addWeighted(ov, 0.55, canvas, 0.45, 0)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
    cv2.putText(canvas, text, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_DUPLEX,
                1.2, (60, 215, 255), 3, cv2.LINE_AA)
    if sub:
        (sw2, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(canvas, sub, ((w - sw2) // 2, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)


def _cam_panel(frame, res, g, state):
    cam = frame.copy()
    if res.found:
        draw_skeleton(cam, res.xy, res.valid,
                      color_line=(0, 255, 255), color_point=(0, 0, 255))
    cam = cv2.resize(cam, (360, GAME_H))
    cv2.putText(cam, f"{state} | {g['action'].upper()} | lane {g['lane']}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 215, 255), 2)
    return cam


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
    state = "menu"        # menu -> calibrating -> playing <-> paused -> over

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
            state = "playing"
        if state == "playing":
            game.update(dt, g["lane"], g["action"])
            if game.over:
                state = "over"

        board = game.render(g["lane"], g["action"])
        if state == "menu":
            _banner(board, "TEMPLE RUN", "SPACE to start   |   Q quit")
        elif state == "calibrating":
            _banner(board, "CALIBRATING", "stand straight and still")
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
        if key == ord('r') and state in ("playing", "paused", "over"):
            gesture.reset(); game.reset(); state = "calibrating"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
