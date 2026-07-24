"""Temple Run without a webcam: self-checks + presentation stills/video.

Stage 1 forbids a live demo, and a webcam recording cannot be replayed
deterministically, so everything about the game that can be verified offline is
verified here -- and the same synthetic run doubles as material for the slides.

  python temple_run_demo.py             # self-checks only (fast, no files)
  python temple_run_demo.py --stills    # + task2_temple_run_preview.png & stills
  python temple_run_demo.py --video     # + video/temple_run_demo.mp4 (~20 MB)

Two synthetic drivers stand in for the missing hardware:

  * a SCRIPTED PLAYER reacts to the obstacle stream, so collision judging,
    scoring, the combo multiplier and the results screen all run for real;
  * a SYNTHETIC SKELETON (two moving shoulder keypoints) is fed straight into
    GestureDetector, so lean/jump/duck detection is exercised without YOLO.
"""

import argparse
import math
import os
import random
import sys
import time

import cv2
import numpy as np

import temple_run as T

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30.0
DT = 1.0 / FPS


# ----------------------------------------------------------------------
# synthetic inputs
# ----------------------------------------------------------------------
class FakePose:
    """Minimal stand-in for PoseResult: only the shoulders are ever read."""

    def __init__(self, cx, cy, sw, full_body=True):
        self.xy = np.zeros((17, 2), np.float32)
        self.valid = np.zeros(17, bool)
        pts = {5: (cx - sw / 2, cy), 6: (cx + sw / 2, cy)}
        if full_body:
            pts.update({0: (cx, cy - sw * 0.9),
                        7: (cx - sw * .75, cy + sw * .5), 8: (cx + sw * .75, cy + sw * .5),
                        9: (cx - sw * .85, cy + sw * 1.), 10: (cx + sw * .85, cy + sw * 1.),
                        11: (cx - sw * .35, cy + sw * 1.2), 12: (cx + sw * .35, cy + sw * 1.2)})
        for k, (x, y) in pts.items():
            self.xy[k] = (x, y)
            self.valid[k] = True
        self.bbox = (cx - sw, cy - sw, cx + sw, cy + sw * 1.5)
        self.n_people, self.inference_ms, self.held = 1, 0.0, False

    @property
    def found(self):
        return True


def scripted_reaction(game, lane, action, act_until, t, memo, botch_every=5):
    """A player that dodges the nearest obstacle, botching every Nth one."""
    if t >= act_until:
        action = "run"
    cand = [o for o in game.obstacles if not o.passed and o.z > 2]
    if cand:
        ob = min(cand, key=lambda o: o.z)
        if ob.z / max(1.0, game.speed) < 0.30:
            # the object itself is memoised: a freed Obstacle's id() is reused
            if id(ob) not in memo or memo[id(ob)][0] is not ob:
                memo["n"] = memo.get("n", 0) + 1
                memo[id(ob)] = (ob, memo["n"] % botch_every == 0)
            if not memo[id(ob)][1]:
                if ob.kind == "coin":
                    lane = ob.lane
                elif ob.kind == "barrier":
                    lane = (ob.lane + 1) % 3 if lane == ob.lane else lane
                elif ob.kind == "wall2":
                    lane = ob.lane
                elif ob.kind == "hurdle":
                    action, act_until = "jump", t + T.HOLD_S
                elif ob.kind == "overhang":
                    action, act_until = "duck", t + T.HOLD_S
    return lane, action, act_until


# ----------------------------------------------------------------------
# 1. game-logic checks
# ----------------------------------------------------------------------
def _fresh():
    g = T.Game()
    g.obstacles.clear()
    return g


def _play(g, script, dur=1.2, dt=1 / 60):
    """script: [(t, lane, action), ...], each held until the next entry."""
    t, i, lane, act = 0.0, 0, 1, "run"
    while t < dur:
        while i < len(script) and script[i][0] <= t:
            _, lane, act = script[i]
            i += 1
        g.update(dt, lane, act, t)
        t += dt
    return g


def logic_checks():
    out = []

    def check(name, cond):
        out.append((name, bool(cond)))

    L = T.START_LIVES
    # judging window: a gesture is never frame-accurate, so EARLY_GRACE before
    # and LATE_GRACE after the crossing both count -- but stale input does not.
    g = _fresh(); g.speed = 30; g.obstacles.append(T.Obstacle("hurdle", 1, 12.0))
    _play(g, [(0.0, 1, "jump"), (0.10, 1, "run")], dur=1.0)
    check("jump just before the hurdle clears it", g.lives == L and g.cleared == 1)

    g = _fresh(); g.speed = 30; g.obstacles.append(T.Obstacle("hurdle", 1, 6.05))
    _play(g, [(0.0, 1, "run"), (0.10, 1, "jump")], dur=1.0)
    check("jump just after it crossed still clears", g.lives == L and g.cleared == 1)

    g = _fresh(); g.speed = 30; g.obstacles.append(T.Obstacle("hurdle", 1, 30.0))
    _play(g, [(0.0, 1, "jump"), (0.10, 1, "run")], dur=1.5)
    check("a stale jump does NOT clear", g.lives == L - 1)

    g = _fresh(); g.speed = 30; g.obstacles.append(T.Obstacle("overhang", 1, 8.0))
    _play(g, [(0.0, 1, "jump")], dur=1.0)
    check("jumping under an overhang costs a life", g.lives == L - 1)

    # lanes
    g = _fresh(); g.speed = 30; g.obstacles.append(T.Obstacle("barrier", 0, 8.0))
    _play(g, [(0.0, 0, "run")], dur=1.0)
    check("standing in the blocked lane costs a life", g.lives == L - 1)

    g = _fresh(); g.speed = 30; g.obstacles.append(T.Obstacle("barrier", 0, 8.0))
    _play(g, [(0.0, 2, "run")], dur=1.0)
    check("another lane is safe", g.lives == L and g.cleared == 1)

    g = _fresh(); g.speed = 30; g.obstacles.append(T.Obstacle("wall2", 2, 8.0))
    _play(g, [(0.0, 2, "run")], dur=1.0)
    check("wall2: only its lane is safe", g.lives == L and g.cleared == 1)

    # coins
    g = _fresh(); g.speed = 30; g.obstacles.append(T.Obstacle("coin", 1, 8.0))
    _play(g, [(0.0, 1, "run")], dur=1.0)
    check("coin collected in its lane", g.coins == 1 and g.lives == L)

    g = _fresh(); g.speed = 30; g.obstacles.append(T.Obstacle("coin", 0, 8.0))
    _play(g, [(0.0, 2, "run")], dur=1.0)
    check("a missed coin is not a penalty", g.coins == 0 and g.lives == L)

    # combo
    g = _fresh(); g.speed = 30
    for i in range(T.COMBO_STEP * 2):
        g.obstacles.append(T.Obstacle("barrier", 0, 8.0 + i * 9.0))
    _play(g, [(0.0, 2, "run")], dur=4.0)
    check("clean play raises the multiplier", g.multiplier > 1 and g.bonus > 0)
    g.obstacles.append(T.Obstacle("barrier", 2, 8.0))
    _play(g, [(0.0, 2, "run")], dur=1.0)
    check("a hit resets the combo", g.combo == 0 and g.multiplier == 1)

    # lives / spawning
    g = _fresh(); g.speed = 30
    for i in range(3):
        g.obstacles.append(T.Obstacle("barrier", 1, 8.0 + i * 10.0))
    _play(g, [(0.0, 1, "run")], dur=3.0)
    check("three hits end the run", g.over and g.lives == 0)

    g = _fresh()
    for _ in range(400):
        g.spawn_t = 0
        g.update(1 / 60, 1, "run", 0.0)
    kinds = [(o.kind, o.lane) for o in g.obstacles]
    check("hurdles/overhangs always span the corridor",
          all(l == 1 for k, l in kinds if k in ("hurdle", "overhang")))
    check("every obstacle kind is reachable", len({k for k, _ in kinds}) >= 4)
    return out


# ----------------------------------------------------------------------
# 2. gesture checks
# ----------------------------------------------------------------------
def _gesture_sequence(kind, dur=1.0, amp=62.0, fps=60):
    """A complete gesture -- out AND back, like a real jump or squat."""
    g = T.GestureDetector()
    for _ in range(30):
        g.update(FakePose(320, 210, 120))          # calibrate: stand still
    acts, n = [], int(dur * fps)
    for i in range(n):
        off = amp * math.sin(math.pi * i / n)
        cy = 210 - off if kind == "jump" else 210 + off
        time.sleep(1 / fps / 6)                    # let the velocity term advance
        acts.append(g.update(FakePose(320, cy, 120))["action"])
    return acts


def gesture_checks():
    out = []
    jumps = _gesture_sequence("jump")
    ducks = _gesture_sequence("duck")
    # The recovery half of a gesture is a fast move in the OPPOSITE direction;
    # without the position gate + refractory it fired the opposite action.
    out.append(("a jump is detected", jumps.count("jump") > 0.5 * len(jumps)))
    out.append(("a jump never reads as a duck", jumps.count("duck") == 0))
    out.append(("a duck is detected", ducks.count("duck") > 0.5 * len(ducks)))
    out.append(("a duck never reads as a jump", ducks.count("jump") == 0))

    # lanes are scale invariant: the same lean at half the distance to the
    # camera (half the shoulder width) must give the same lane.
    lanes = []
    for sw in (80, 120, 200):
        g = T.GestureDetector()
        for _ in range(30):
            g.update(FakePose(320, 210, sw))
        lanes.append(g.update(FakePose(320 + 0.6 * sw, 210, sw))["lane"])
    out.append(("lane threshold is distance invariant", set(lanes) == {2}))

    g = T.GestureDetector()
    for _ in range(30):
        g.update(FakePose(320, 210, 120))
    out.append(("standing still stays in the middle lane",
                g.update(FakePose(320, 210, 120))["lane"] == 1))

    class Gone:
        found = False
        xy = np.zeros((17, 2), np.float32)
        valid = np.zeros(17, bool)
    r = g.update(Gone())
    out.append(("losing the body is not an input", r["action"] == "run" and not r["seen"]))
    return out


# ----------------------------------------------------------------------
# 3. stills / video from a scripted run
# ----------------------------------------------------------------------
def synthetic_run(write_video=False, want_stills=True, seconds=75):
    random.seed(7)
    game, gest = T.Game(), T.GestureDetector()
    lane, action, act_until, memo = 1, "run", -1.0, {}
    stills, t, frames = {}, 0.0, 0
    vw = None
    if write_video:
        os.makedirs(os.path.join(HERE, "video"), exist_ok=True)
        vw = cv2.VideoWriter(os.path.join(HERE, "video", "temple_run_demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (T.CAM_W + T.GAME_W, T.GAME_H))

    while frames < seconds * FPS and not game.over:
        lane, action, act_until = scripted_reaction(game, lane, action, act_until, t, memo)
        game.update(DT, lane, action, t)
        board = game.render(lane, action)
        if vw is not None or want_stills:
            # a synthetic body posed to match whatever the script just did,
            # placed in the top 4:3 region of the fake webcam frame
            cy = 220 - 60 if action == "jump" else (220 + 55 if action == "duck" else 220)
            cx = 320 + (lane - 1) * 78
            pose = FakePose(cx, cy, 130)
            g = dict(lane=lane, action=action, ready=True, calibrating=False, seen=True,
                     dx=(lane - 1) * 0.6, dy=(0.45 if action == "jump" else
                                              (-0.45 if action == "duck" else 0.0)),
                     vy=0.0)
            cam = np.full((480, 640, 3), (40, 45, 55), np.uint8)
            cv2.rectangle(cam, (0, 360), (640, 480), (60, 70, 85), -1)
            out = np.hstack([T._cam_panel(cam, pose, g, "PLAYING"), board])
        if vw is not None:
            vw.write(out)
        if want_stills:
            if game.flash > 0.30:
                stills.setdefault("hit", out.copy())
            if game.multiplier >= 3 and game.coins > 6:
                stills.setdefault("combo", out.copy())
            if game.player.lift > 90:
                stills.setdefault("jump", out.copy())
            if action == "duck":
                stills.setdefault("duck", out.copy())
            if game.speed > 52:
                stills.setdefault("fast", out.copy())
        t += DT
        frames += 1

    for i in range(70):                                   # results screen reveal
        game.update(DT, lane, "run", t)
        b = game.draw_results(game.render(lane, "run"))
        if vw is not None or want_stills:
            out = np.hstack([out[:, :T.CAM_W], b])
        if vw is not None:
            vw.write(out)
        if want_stills and i == 60:
            stills["results"] = out.copy()
        t += DT
    if vw is not None:
        vw.release()

    if want_stills:
        shots = os.path.join(HERE, "temple_run_shots")
        os.makedirs(shots, exist_ok=True)
        for k, im in stills.items():
            cv2.imwrite(os.path.join(shots, f"{k}.png"), im)
        hero = stills.get("combo", stills.get("fast"))
        if hero is not None:      # the one the write-up links to
            cv2.imwrite(os.path.join(HERE, "task2_temple_run_preview.png"), hero)
    return game, frames, sorted(stills)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true", help="write preview PNGs")
    ap.add_argument("--video", action="store_true", help="write video/temple_run_demo.mp4")
    a = ap.parse_args()

    rows = [("GAME LOGIC", logic_checks()), ("GESTURES", gesture_checks())]
    if a.stills or a.video:
        game, frames, shots = synthetic_run(write_video=a.video, want_stills=True)
        print(f"\nscripted run: {frames / FPS:.0f}s, score {game.score}, "
              f"{game.coins} coins, {game.cleared} cleared, best combo {game.best_combo}, "
              f"top speed {game.speed:.0f}")
        print("stills:", ", ".join(shots))

    bad = 0
    for title, checks in rows:
        print(f"\n{title}")
        for name, passed in checks:
            print(("  PASS  " if passed else "  FAIL  ") + name)
            bad += not passed
    print("\nALL PASS" if not bad else f"\n{bad} CHECK(S) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
