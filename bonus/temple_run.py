"""Bonus Level Task 2, extra activity -- Temple-Run-style body-controlled game.

Answers rubric Q16: "besides dancing, what other actions can you detect and
score? How do you render new objects, detect the new action, and compute score?"

WHAT WE DETECT (three full-body gestures, all from the shoulder keypoints so it
still works when only your upper body is in frame):

  * LEAN left / right  -> move between three lanes
  * JUMP  (shoulders rise above the calibrated baseline)   -> clear a low bar
  * DUCK  (shoulders drop below the baseline)               -> pass a high bar

HOW WE DETECT IT: at startup we calibrate a standing baseline (median shoulder
center + shoulder width) over a second of "stand still". Every frame we measure
the shoulder center relative to that baseline, in units of shoulder width so it
is distance-invariant:
    dx = (sx - base_x) / sw   -> lane        (|dx| > LANE_TH)
    up = (base_y - sy) / sw   -> jump         (up > MOVE_TH)
    dn = (sy - base_y) / sw   -> duck         (dn > MOVE_TH)

HOW WE RENDER: an OpenCV canvas with three lanes; obstacles and coins scroll
toward the player; a webcam thumbnail (with skeleton + current action) sits in
the corner.

HOW WE SCORE: +distance every frame, +COIN_PTS per coin collected, +CLEAR_PTS
for each obstacle dodged; a hit costs a life (3 total). Final score shown on
game over.

Run:  python temple_run.py     (press Q to quit, R to recalibrate)
"""

import time
import random

import cv2
import numpy as np

from pose_pipeline import PoseTracker

MODEL = "yolov8n-pose.pt"

# gesture thresholds, in units of shoulder width
LANE_TH = 0.45
MOVE_TH = 0.40

# scoring
COIN_PTS = 10
CLEAR_PTS = 5
START_LIVES = 3

GAME_W, GAME_H = 640, 720
LANE_X = [GAME_W // 6, GAME_W // 2, 5 * GAME_W // 6]   # 3 lane centers
HIT_Y = GAME_H - 130          # obstacles are resolved when they cross this line
PLAYER_R = 34

LANE_COLOR = (60, 60, 60)
COIN_COLOR = (60, 215, 255)
WALL_COLOR = (80, 80, 230)
LOW_COLOR = (230, 170, 60)     # jump bar
HIGH_COLOR = (200, 90, 210)    # duck bar


class GestureDetector:
    """Turns shoulder geometry into (lane, action), after a standing calibration."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._cal = []          # shoulder samples during calibration
        self.base = None        # (x, y, width)
        self.ready = False

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

        bx, by, _ = self.base
        sw = max(1.0, s[2])
        dx = (s[0] - bx) / sw
        up = (by - s[1]) / sw
        dn = (s[1] - by) / sw

        lane = 1
        if dx < -LANE_TH:
            lane = 0
        elif dx > LANE_TH:
            lane = 2

        action = "run"
        if up > MOVE_TH:
            action = "jump"
        elif dn > MOVE_TH:
            action = "duck"

        return dict(lane=lane, action=action, ready=True, calibrating=False)


class Obstacle:
    __slots__ = ("kind", "lane", "y", "scored")

    def __init__(self, kind, lane, y):
        self.kind = kind        # 'wall' | 'low' | 'high' | 'coin'
        self.lane = lane
        self.y = y
        self.scored = False


class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obstacles = []
        self.spawn_t = 0.0
        self.speed = 260.0          # px/s obstacles fall
        self.distance = 0.0
        self.coins = 0
        self.cleared = 0
        self.lives = START_LIVES
        self.over = False
        self.flash = 0.0            # red flash timer on hit

    @property
    def score(self):
        return int(self.distance / 10) + self.coins * COIN_PTS + self.cleared * CLEAR_PTS

    def _spawn(self):
        kind = random.choices(["wall", "low", "high", "coin"],
                              weights=[3, 2, 2, 3])[0]
        lane = random.randint(0, 2)
        self.obstacles.append(Obstacle(kind, lane, -60))

    def update(self, dt, lane, action):
        if self.over:
            return
        self.distance += self.speed * dt
        self.flash = max(0.0, self.flash - dt)
        self.spawn_t -= dt
        if self.spawn_t <= 0:
            self._spawn()
            self.spawn_t = random.uniform(1.1, 1.8)
        self.speed = min(460.0, self.speed + 4.0 * dt)   # gently accelerate

        for ob in self.obstacles:
            ob.y += self.speed * dt
            if not ob.scored and ob.y >= HIT_Y:
                ob.scored = True
                self._resolve(ob, lane, action)
        self.obstacles = [o for o in self.obstacles if o.y < GAME_H + 40]

    def _resolve(self, ob, lane, action):
        if ob.kind == "coin":
            if lane == ob.lane:
                self.coins += 1
            return
        # obstacles: determine whether the player avoided it
        if ob.kind == "wall":
            avoided = lane != ob.lane
        elif ob.kind == "low":
            avoided = action == "jump"
        else:  # high
            avoided = action == "duck"
        if avoided:
            self.cleared += 1
        else:
            self.lives -= 1
            self.flash = 0.35
            if self.lives <= 0:
                self.over = True

    def render(self, lane, action, ready):
        c = np.full((GAME_H, GAME_W, 3), 30, np.uint8)
        for lx in LANE_X:
            cv2.line(c, (lx, 0), (lx, GAME_H), LANE_COLOR, 2)
        cv2.line(c, (0, HIT_Y), (GAME_W, HIT_Y), (50, 50, 50), 1)

        for ob in self.obstacles:
            y = int(ob.y)
            if ob.kind == "coin":
                cv2.circle(c, (LANE_X[ob.lane], y), 16, COIN_COLOR, -1)
                cv2.circle(c, (LANE_X[ob.lane], y), 16, (0, 0, 0), 1)
            elif ob.kind == "wall":
                x = LANE_X[ob.lane]
                cv2.rectangle(c, (x - 55, y - 22), (x + 55, y + 22), WALL_COLOR, -1)
            elif ob.kind == "low":
                cv2.rectangle(c, (0, y - 14), (GAME_W, y + 14), LOW_COLOR, -1)
                cv2.putText(c, "JUMP", (GAME_W // 2 - 40, y + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            else:  # high
                cv2.rectangle(c, (0, y - 14), (GAME_W, y + 14), HIGH_COLOR, -1)
                cv2.putText(c, "DUCK", (GAME_W // 2 - 40, y + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # player
        px = LANE_X[lane]
        py = HIT_Y
        col = (80, 220, 80)
        if action == "jump":
            py -= 46
        elif action == "duck":
            py += 26
        cv2.circle(c, (px, py), PLAYER_R, col, -1)
        cv2.circle(c, (px, py), PLAYER_R, (0, 0, 0), 2)
        if action != "run":
            cv2.putText(c, action.upper(), (px - 40, py - PLAYER_R - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        if self.flash > 0:
            overlay = c.copy()
            cv2.rectangle(overlay, (0, 0), (GAME_W, GAME_H), (0, 0, 200), -1)
            c = cv2.addWeighted(overlay, 0.25, c, 0.75, 0)

        # HUD
        cv2.putText(c, f"Score {self.score}", (12, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(c, f"Lives {'#' * max(0, self.lives)}", (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 120, 255), 2)
        cv2.putText(c, f"Coins {self.coins}", (GAME_W - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COIN_COLOR, 2)

        if not ready:
            _banner(c, "CALIBRATING - stand still")
        if self.over:
            _banner(c, f"GAME OVER  score {self.score}", sub="press R to play again")
        return c


def _banner(canvas, text, sub=None):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h // 2 - 50), (w, h // 2 + (60 if sub else 30)),
                  (0, 0, 0), -1)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cv2.putText(canvas, text, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_DUPLEX,
                1.0, (60, 215, 255), 2, cv2.LINE_AA)
    if sub:
        (sw2, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(canvas, sub, ((w - sw2) // 2, h // 2 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)


def main():
    tracker = PoseTracker(MODEL)
    gesture = GestureDetector()
    game = Game()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("could not open webcam")

    from pose_pipeline import draw_skeleton
    win = "Temple Run (body controlled)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    prev = time.perf_counter()

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
        game.update(dt, g["lane"], g["action"])

        board = game.render(g["lane"], g["action"], g["ready"])

        # webcam thumbnail with skeleton + action label
        cam = frame.copy()
        if res.found:
            draw_skeleton(cam, res.xy, res.valid,
                          color_line=(0, 255, 255), color_point=(0, 0, 255))
        cam = cv2.resize(cam, (360, GAME_H))
        label = "READY" if g["ready"] else "CALIBRATING"
        cv2.putText(cam, f"{label}: {g['action'].upper()}  lane {g['lane']}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 215, 255), 2)

        cv2.imshow(win, np.hstack([cam, board]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            gesture.reset()
            game.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
