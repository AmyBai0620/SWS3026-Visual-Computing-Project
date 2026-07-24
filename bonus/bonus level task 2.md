---
title: Bonus Task 2 — Just Dance Scoring and a Motion Game
tags:
  - visual-computing
  - pose-estimation
  - bonus-level
---

# Bonus Level — Task 2
## Just Dance: Scoring the Dance, and a Motion-Controlled Temple Run

> [!abstract] Summary
> Task 2 reuses the Task 1 `PoseTracker` for **both** panels — reference video (left) and webcam (right). Poses are compared as **12 limb-direction vectors**, which are invariant to where you stand and how big you are; the user's lag behind the reference is tracked with a **monotonic, bounded window**; each frame yields a **0–100 number and a PERFECT/SUPER/GOOD/X tier**, averaged into a final score. The design was hardened after we found a cheat: standing still could score SUPER. The extra game (Q16) is a full-body **Temple Run** driven by shoulder gestures.

> [!info] Rubric coverage
> **Q13 — "screenshots or a video of your Dance program in action"** → §1.
> **Q14 — "How did you score? How did you align spatially and temporally? What similarity/distance metrics?"** → §2.
> **Q15 — "Numeric or textual or both? How does it change over time? Overall score, and how?"** → §3.
> **Q16 — "Besides dancing, what other actions can you detect and score? How do you render new objects, detect the action, compute the score?"** → §5.
> §4 is the headline story: how we found and fixed the "stand still and score SUPER" exploit.

---

## 1. The Dance program in action (Q13)

`just_dance.py` — two panels:

- **Left (Reference):** the reference video with its precomputed skeleton overlaid.
- **Right (You):** live webcam + `PoseTracker` skeleton + a large Just-Dance-style tier word (PERFECT / SUPER / GOOD / X) + the running number.

The reference thread is the **clock**: it advances `t_play`; each webcam frame is scored against whatever `t_play` currently is. Threads only push images to a `queue`; the Tk main loop pops them and updates widgets (thread-safe).

> [!important] The design decision that makes it run on CPU
> The reference video never changes, so `precompute_reference.py` computes its per-frame keypoints **once, offline**, into `video/ref_<name>.npz` (e.g. `dance_example_7`: 3022 frames, 352 KB). At run time the reference panel is a table lookup and **only the webcam does live inference** — halving the load, which is what lets two panels stay smooth on CPU. The cost: to change reference video you must precompute it first, then set `REF_NAME` in [just_dance.py:39](just_dance.py#L39) (currently `dance_example_7`; caches exist for `dance_example_5/6/7`).

**Screenshots/recording:** the webcam path must be captured live on the presenting machine (rubric requires screenshots or a pre-recorded video — no live demo in Stage 1).

---

## 2. Scoring, alignment, and metric (Q14)

Three sub-questions; each answered directly.

### 2.1 Spatial alignment — limb directions, invariant by construction

You and the reference dancer differ in body size, position, and distance to the camera, so **raw pixel coordinates cannot be compared**. We represent each pose as **12 limb direction vectors** (upper/lower arms, upper/lower legs, the two torso sides, shoulder line, hip line):

```python
v = xy[b] - xy[a]          # limb a -> b
units[i] = v / norm(v)     # unit direction: length (body size) dropped, only heading kept
```

The angle between two unit directions is **naturally invariant to translation and scale** — a small person and a tall person striking the same pose produce identical angles, with no manual registration. (`normalize_pose()` also provides hip-centred, torso-scaled coordinates for visualisation and as a fallback distance metric.)

### 2.2 Similarity metric — angle tolerance through a smoothstep

Per limb, take the angle $\theta$ between the user's direction and the reference's. It scores **full credit at 0° and zero at `ANGLE_TOL = 50°`**, mapped through a **smoothstep** rather than a straight line:

```python
def _angle_credit(angle):
    x = np.clip(1.0 - angle / ANGLE_TOL, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)     # flat near the ends, steep in the middle
```

The frame score is a **weighted average** of limb credits × 100, weighting **arms 1.5, legs 1.2, torso 0.6–0.7** — hands and feet are what an audience judges a dance on.

> [!insight] Why not cosine similarity?
> The naive `(cos θ + 1)/2` compresses all human poses into the high band: two *random* directions are 90° apart, cosine 0, and still score 50. Wrong moves would read as "decent". The 50°-tolerance smoothstep separates right from wrong far more sharply — see §4 for the quantitative difference.

### 2.3 Temporal alignment — a monotonic, bounded lag window

You are always slightly behind the reference (the brief calls this out explicitly). We first **search a wide window once** to acquire your lag, then every frame search only a narrow `t_play − lag ± 0.13 s` window, clamping `lag` to **≤ 0.6 s** and adapting it slowly:

```python
if self._lag is None:                       # first: wide window [t-0.5, t+0.2] to lock the lag
    lo, hi = search(t - acq_back), search(t + acq_fwd)
else:                                        # after: only +/-0.13s around t - lag
    c = t_play - self._lag
    lo, hi = search(c - track_half), search(c + track_half)
implied  = clip(t_play - ref_t[best_j], 0, max_lag)   # update lag: clamp + slow EMA
self._lag = clip((1 - lag_ema) * self._lag + lag_ema * implied, 0, max_lag)
```

The comparison target moves **forward with playback and can never sit still**. Section 4 explains why this replaced a simpler "widest-window maximum" — that older design was the actual hole behind the cheat.

---

## 3. Score over time and overall (Q15)

Four sub-questions; each answered directly.

- **Numeric or textual? Both.** Every frame emits a 0–100 number *and* a tier: **PERFECT ≥ 85, SUPER ≥ 70, GOOD ≥ 50, else X**.
- **How does it change over time?** Each frame's score is EMA-smoothed (α = 0.5) so the tier word does not flicker, and is shown live. `task2_score_timeline_dance_example_5.png` plots three dancers (good / speed-varying / wrong moves) whose curves land in visibly different bands.
- **Overall score after the dance? Yes.** The final score is the **mean of every per-frame score**, plus a tier histogram (how many frames were PERFECT/SUPER/GOOD/X). `just_dance.py` pops this up when the reference video ends.
- **The key rule: unscorable frames count as 0, they are not dropped.** If you leave frame or your body is not fully visible, that frame scores **0 and is included** in the average — otherwise "step out during the hard part" becomes a free pass (see §4).

> [!warning] Numbers need a rerun before the presentation
> `task2_score_timeline_dance_example_5.png` and any quoted "measured total" were produced by the pre-§4 scoring system and no longer reflect the current (stricter) one. Rerun `python score_timeline.py <name>` and re-measure once. Note the timeline figure uses `dance_example_5` while `just_dance.py` now points at `dance_example_7` — align both to whatever you actually demo.

---

## 4. The story worth telling: finding and killing "stand still → SUPER" (Q14/Q15)

> [!note] Why lead with this
> It is the clearest "found a problem → diagnosed the root cause → designed a fix → verified it" chain in Task 2, and every step is on-rubric for how scoring and alignment work.

### 4.1 The exploit

Testing found a cheat pose: **stand completely still** and the system still handed out SUPER frequently, with an overall GOOD. Someone not dancing at all should not score well — the scoring had a hole.

### 4.2 Root cause — not one bug, four stacked

| # | Cause | Why it let you cheat |
|---|---|---|
| 1 | Temporal alignment took the **max over a wide window** | picking the best of ~21 noisy comparisons per frame is biased high; hold still and the reference eventually *sweeps past* a near-match |
| 2 | Normalised only over **visible** limbs, **dropped** unscorable frames | show only your torso, get the arms right, no penalty for legs; step out during hard parts and those frames vanish — **the more you hide, the higher you score** |
| 3 | Tolerance **70° + linear** falloff | an arm 60° off still scored; "just stand up straight" reached 50–60, everything crushed into the high band |
| 4 | 2D limb directions unstable under **foreshortening** | a limb pointing at the camera projects to near-zero length, so noise dominates its direction |

### 4.3 The four fixes, one per cause

**Fix 1 — monotonic bounded lag window (§2.3) + a liveness gate.** The window now moves forward with playback, so freezing gets you compared against a *moving* reference. On top of that, a liveness gate tracks the smoothed motion rate of both user and reference; when the reference is moving but you are not, it removes up to 90% of the score. Two people **correctly holding the same pose are not penalised** (the reference is not moving either) — only "frozen when you should be moving":

```python
if self._rmot > MOTION_THRESH:                 # only when the reference actually moves
    deficit = clip(1 - self._umot / self._rmot, 0, 1)
    factor  = 1 - LIVENESS_STRENGTH * deficit  # the stiller you are, the harder the cut
score *= factor
```

**Fix 2 — coverage penalty + missing frames score 0.** Multiply the score by the fraction of the arm/leg limbs the reference is using that you actually show; unscorable frames record 0 and count toward the average:

```python
ref_key = ok2 & KEY_LIMBS                       # arm/leg limbs the reference uses this frame
shown   = LIMB_WEIGHTS[both & KEY_LIMBS].sum()  # the part of those you also show
score  *= shown / LIMB_WEIGHTS[ref_key].sum()   # hide your legs -> scaled down
```

**Fix 3 — tolerance 70° → 50° + smoothstep** (§2.2): wrong limbs lose credit faster and the tiers mean something again.

**Fix 4 — down-weight foreshortened limbs:** a limb whose projected length is under 15% of the torso is weighted down — "if the direction is unreliable, trust it less".

### 4.4 Verification, no webcam needed

`validate_scoring.py` builds fake users with known ground truth from the reference sequence. All three self-checks **PASS**:

| Test | Setup | Result | Verdict |
|---|---|---|---|
| Spatial invariance | same pose, 1.8× body size + translation, no noise | similarity **100**, normalised diff **0.0000** | ✅ position/size-independent |
| Temporal alignment | matched user, 5-frame (167 ms) lag injected | recovered lag median **167 ms** | ✅ exact |
| Matched scoring | 1.4× size + translation + noise + lag | total **98.1 / PERFECT** | ✅ high when it should be |
| Discrimination | reference played **in reverse** (wrong moves) | total **56.3 / GOOD**, 40+ points lower | ✅ clearly lower |

**Anti-cheat test, before vs after the fixes:**

| Scenario | Before | After |
|---|---|---|
| **Stand still** (the strongest cheat — freezing on a pose from the video) | **68.8 GOOD** (30 PERFECT + 166 SUPER) | **50.4 GOOD**, 208 of 389 frames judged **X** |
| Actually dancing | 97 | **98.2 PERFECT** |
| Only upper body, legs hidden | ~98 | **67 GOOD** |

> [!success] Honest reading of the result
> ① Our "stand still" is the *adversarial worst case* — frozen on a pose that literally appears in the video; a real person standing with arms down would differ more and score lower still. After the fix it is only GOOD overall and mostly X per frame, no longer SUPER. ② Reverse playback still scores 56, not 0, because a continuous dance played backwards still passes through many similar poses — it is not random flailing. The 98-vs-56 gap is what proves the discrimination is real. ③ We stopped tolerance at **50°, not tighter**, deliberately: real webcam noise is larger, and over-tightening to win an offline test would punish genuine dancers.

---

## 5. Extra action: motion-controlled Temple Run (Q16)

`temple_run.py` — a standalone OpenCV window controlled by full-body motion. Preview: `task2_temple_run_preview.png`.

### 5.1 What it detects, and the features (Q16: detection)

Three gestures, all built on the **most reliable keypoints — the shoulders**, so it plays even when only the upper body is in frame (a lesson from close-range webcam framing):

| Action | Feature | Effect |
|---|---|---|
| Lean left / right | shoulder-centre offset `dx = (sx − base_x)/shoulder_width`, past ±0.45 | switch between three lanes |
| Jump | shoulders rise `up > 0.34` **or** rise-speed `vy > 1.1 shoulder-widths/s` | clear a low hurdle |
| Duck | shoulders drop `dn > 0.34` **or** drop-speed | pass under an overhang |

Every threshold is **normalised in shoulder-widths**, so it is immune to how far you stand from the camera. At startup a ~1 s "stand still" calibration fixes the baseline (median shoulder-centre + shoulder-width). No training set is needed — the gestures are geometric rules, not a learned classifier.

> [!insight] Cutting jump latency (a real playtest fix)
> A jump is instantaneous; "position crossed a threshold" alone reacts half a beat late. Three changes: (1) the game's `PoseTracker` runs with **smoothing off** (`smooth_alpha=1.0`) — responsiveness over smoothness; (2) a **velocity trigger** fires the instant shoulder rise-speed exceeds threshold, catching the takeoff rather than waiting for the apex; (3) the action is **latched for 0.35 s** so a brief peak reliably covers the frame where the obstacle is resolved.

**The camera panel now shows the rule, not just the skeleton.** A webcam is landscape but the game board is portrait, so letterboxing the camera into a tall side-strip dropped the player into a band of black bars and made them tiny. The panel is now split: the **webcam fills a 4:3 block up top at native aspect** (no crop, no stretch, the player is large), and the **leftover height becomes a dashboard** with two gauges — `dx` and `dy` in shoulder-widths, the threshold ticks, and a hollow marker for the *velocity* term rescaled onto the same axis — plus the big current-action word and the calibration progress. An audience watching the recording sees the exact number that produced each decision.

> [!warning] The bug the gauge exposed: every gesture fired its own opposite
> Putting the measurement on screen immediately showed the flaw. A gesture has two halves, and **the recovery half is a fast move in the opposite direction**: coming down from a jump is a large negative shoulder velocity, which tripped the *velocity* duck trigger — so the runner dropped into a slide the instant he landed (and rising out of a duck fired a jump).
> **Fix:** a velocity trigger only counts when the *position* does not contradict it (`vy < −VEL_TH` **and** `up < 0.5·MOVE_TH`), plus a `RECOVER_S = 0.15 s` dead time for the opposite trigger while the previous latch unwinds. A *position* trigger is always trusted — a deep squat is unambiguous. Measured on a synthetic out-and-back gesture: **6 contaminated frames per gesture → 0**, and the correct action is held for more frames than before (53 → 59).

### 5.2 How new objects are rendered (Q16: rendering)

**First-person perspective corridor + real sprite art + a scrolling texture floor.** The world is modelled in depth `z` (0 at the player, larger = farther); every object is projected through a vanishing point: `t = z/(z+K)`, near-large-far-small `scale = 1 − t`, lanes converging to the vanishing point.

- **Background (`temple_bg.py`):** the runner's motion comes from a scrolling *floor*, not a background video. Seamless stone/sand textures are **generated procedurally**, sampled in perspective with `cv2.remap`; the sampled depth offset advances each frame with speed, and `BORDER_WRAP` tiles it seamlessly → tiles rush toward the player. Above the horizon is a near-static backdrop (a real photo if `assets/backdrop.*` exists, else a gradient). Optimised to **~21 ms/frame** by computing only the strip below the horizon.
- **Objects and character:** transparent PNG sprites under `assets/`, composited with alpha + depth scaling + fog by `blit()` — the player in three states (run/jump/slide, all rear view), stone pillars (barrier), hurdles, overhangs, coins, hearts (life HUD), explosion (on hit).
- **Graceful degradation:** any missing sprite or texture falls back to built-in drawing; nothing crashes.

The alpha compositing is the same technique as the Expert-task sticker rendering.

**Game feel — the layer that separates a demo from a game.** Everything below exists because a recording is judged on how it *moves*:

| Effect | What it does | Why |
|---|---|---|
| Eased lane change + bank | the runner slides between lanes (`τ = 75 ms`) and tilts into the turn | he used to teleport between three x positions |
| Chase camera | pans 20 % toward the runner, bobs with speed, shakes on a hit | sells the turn and the impact |
| Jump arc, landing squash, footfall dust | fast rise, gravity-paced fall, a small squash and a dust puff on touchdown | the jump had no weight |
| Coin sparks, rings, floating `+points` | stacked above the runner's head so a coin run does not pile labels on one spot | every event now has feedback |
| Impact: debris, red rim vignette, 0.3 s slow motion | on losing a life | a life used to vanish with a red flash |
| Speed streaks, last-life heartbeat vignette | scale with `speed` / trigger at 1 life | shows the difficulty ramp |
| `3 · 2 · 1 · GO!` countdown, staggered results reveal | between calibration and play, and on game over | gives the recording a beat |

> [!important] It got faster, not slower
> All of that is **cheaper than the version before it**: `render()` went **29.3 ms → 25.4 ms** per frame (busy scene, same machine). The win came from two hot loops that were written in numpy: the per-sprite alpha composite `fg*a + sub*(1−a)` and the background's fog blend allocate several full-size float temporaries each. Rewritten through OpenCV (`cv2.subtract/multiply/add`, `cv2.copyTo` for the floor/wall mask) they are **6× and 3.5× faster** — 5.2 ms → 0.8 ms for one large sprite. The camera shift is a strided copy instead of `warpAffine` (2.5 ms → 0.4 ms), and the HUD dims only its own 52-row strip.

### 5.3 How the score is computed (Q16: scoring)

Score = **distance / 10 + coins × 10 + obstacles avoided × 5**, each event multiplied by a **combo multiplier** (×1 → ×4, one step per 4 clean events, lost on a hit or after 4 s of nothing) plus a **+3 near-miss bonus** for a lane dodge inside 0.3 s of impact. A hit costs one of 3 lives, zero lives ends the run, and obstacle speed ramps from 30 to 62 depth-units/s.

> [!insight] Judging a webcam gesture on a single frame is unfair
> An obstacle used to be resolved on the **one frame** it crossed the player plane, against the lane and action of that instant. But the input is a 30 fps pose estimate of a human body — nobody is frame-accurate. Obstacles are now judged over a **window**: a jump/duck counts if it started up to `EARLY_GRACE = 0.30 s` before the crossing, the verdict is **deferred by `LATE_GRACE = 0.16 s`** so a slightly late reaction still saves you, and lane obstacles also look back `0.12 s` through a lane history. Spawning enforces a minimum spatial gap (and a wider one between two consecutive reaction obstacles) so no combination is unavoidable, and hurdles/overhangs — which no lane is safe from — are rendered **spanning the whole corridor** instead of centred on a lane, which is what they actually mean.

A full **state machine** wraps it (a playtest fix — the original started instantly with no buffer):
`menu (SPACE) → calibrate (stand still) → 3·2·1 countdown → playing (P/SPACE pause) → results`.
The results screen counts up coins × 10, obstacles × 5, distance / 10, combo bonus and TOTAL, and flags a NEW BEST; SPACE replays, Q quits.

### 5.4 Verification, no webcam needed

`temple_run_demo.py` runs the whole game **without a webcam or YOLO**: a scripted player reacts to the real obstacle stream (so judging, scoring, combo and the results screen all execute), and a synthetic two-keypoint skeleton is fed straight into `GestureDetector`. **21 checks, all passing:**

- **Judging window** — a jump just before *and* just after the crossing clears the hurdle; a stale jump does not; the wrong action costs a life.
- **Lanes and coins** — blocked lane hits, other lanes are safe, `wall2` has exactly one safe lane, coins score only in their lane, a missed coin is not a penalty.
- **Scoring** — the multiplier grows on clean play and pays a bonus, a hit resets it, three hits end the run.
- **Spawning** — hurdles/overhangs always span the corridor; every obstacle kind occurs.
- **Gestures** — a jump is detected and **never reads as a duck** (and vice versa); the lane threshold gives the same lane at 80/120/200 px shoulder width (distance invariance); standing still stays in the middle lane; losing the body produces no input.

The same run writes `task2_temple_run_preview.png` and `temple_run_shots/` (`--stills`), and a full mp4 with `--video` — a deterministic backup if the live recording goes wrong.

---

## Appendix. Files and reproduction

```text
bonus/
├── pose_pipeline.py          # Task 1 tracker (reused directly)
├── precompute_reference.py   # reference keypoints -> video/ref_<name>.npz
├── pose_score.py             # alignment + similarity + feedback (Q14/Q15)
├── just_dance.py             # two-panel GUI, live scoring (Q13)
├── validate_scoring.py       # offline scoring self-checks (no webcam)
├── score_timeline.py         # score-vs-time figure (Q15)
├── temple_run.py             # motion-controlled Temple Run (Q16)
├── temple_run_demo.py        # offline self-checks + preview stills/video (Q16)
└── temple_bg.py              # procedural perspective floor
```

```bash
conda activate vcwork                       # torch 2.7.1 CPU, opencv, ultralytics

cd bonus
python precompute_reference.py dance_example_7   # once per reference video
python just_dance.py                             # two-panel Just Dance (set REF_NAME first)
python validate_scoring.py                       # offline scoring checks
python temple_run.py                             # motion game (Q quit, R recalibrate)
python temple_run_demo.py                        # 21 offline checks, no webcam
python temple_run_demo.py --stills               # + refresh the preview PNGs
```

Shipped scoring parameters: `ANGLE_TOL=50°`, arm/leg/torso weights `1.5 / 1.2 / 0.6–0.7`, score EMA `α=0.5`, `max_lag=0.6 s`, track window `±0.13 s`, `LIVENESS_STRENGTH=0.9`, foreshorten cutoff `0.15·torso`. Temple Run: `LANE_TH=0.45`, `MOVE_TH=0.34`, `VEL_TH=1.1` shoulder-widths/s.
