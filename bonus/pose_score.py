"""Dance scoring for Task 2 (Just Dance) -- the answer to rubric Q14 & Q15.

Q14  How do we score, align, and which similarity metric?
  * Spatial alignment  : we compare *limb direction vectors*, whose cosine is
    invariant to translation and scale by construction. So a small user and a
    large reference dancer, standing in different parts of the frame, still
    match perfectly when they strike the same pose -- no manual registration
    needed. (normalize_pose() is also provided for display / distance fallback.)
  * Similarity metric  : per limb we take the angle between the user's and the
    reference's segment direction and map it through a smoothstep that gives
    full credit near 0 and falls to zero at ANGLE_TOL. The pose score is the
    weighted mean over matched limbs, on a 0-100 scale.
  * Temporal alignment : the user lags the reference, so for each webcam frame
    we search a short window of reference frames and take the best match. A
    small penalty on the temporal offset keeps a frame right under the current
    playback time preferred over one cherry-picked from the far edge of the
    window.

Guards that stop the score from being gamed (previously you could freeze in a
generic pose and still be graded SUPER):
  * liveness gate  : if the reference is clearly moving but the user is frozen,
    the score is knocked down in proportion to how still the user is. Two
    people both *holding* the same pose are not penalised -- only a static user
    during a moving passage is.
  * coverage       : the score is scaled by the fraction of the arm/leg limbs
    the reference is using that the user actually shows, so you cannot get 100
    by only presenting your upper body while your legs are wrong or hidden.
  * frames where the user is not visible enough to score count as 0, they are
    not silently dropped, so stepping out of frame during a hard passage costs
    you instead of being free.
  * foreshortening : a limb pointing at the camera projects to almost a point,
    so its 2D direction is dominated by noise; such limbs are down-weighted.

Q15  Numeric or textual, over time, and an overall score?
  * Both: a 0-100 number every frame AND a textual tier (Perfect/Super/Good/X).
  * Over time: the per-frame number is EMA-smoothed so the tier does not
    flicker; a running mean is kept.
  * Overall: at the end we report the mean per-frame score and a tier
    breakdown (how many Perfect/Super/Good/X frames).
"""

import numpy as np

# Directed limbs (parent -> child). Cosine of these vectors captures "which way
# each body segment points", which is what a dance pose really is.
LIMBS = [
    (5, 7), (7, 9),      # left upper arm, forearm
    (6, 8), (8, 10),     # right upper arm, forearm
    (11, 13), (13, 15),  # left thigh, shin
    (12, 14), (14, 16),  # right thigh, shin
    (5, 11), (6, 12),    # left / right torso side
    (5, 6), (11, 12),    # shoulders, hips
]

# Limbs weighted higher: arms and legs are what a viewer judges in a dance.
LIMB_WEIGHTS = np.array([
    1.5, 1.5, 1.5, 1.5,   # arms
    1.2, 1.2, 1.2, 1.2,   # legs
    0.7, 0.7,             # torso sides
    0.6, 0.6,             # shoulders, hips
], np.float32)

# The limbs a dancer is actually judged on (arms + legs). Coverage and the
# liveness gate are computed over these, not the near-static torso segments.
KEY_LIMBS = LIMB_WEIGHTS >= 1.2

# A limb pointing more than ANGLE_TOL away from the target scores zero; within
# it, credit follows a smoothstep. Tightened from 70 to 50 deg: 70 handed out
# partial credit to limbs that were visibly wrong, which compressed every score
# into the high band and let sloppy poses read as SUPER.
ANGLE_TOL = np.deg2rad(50.0)

# A limb whose projected length is below this fraction of the torso is treated
# as pointing toward/away from the camera; its 2D direction is unreliable.
FORESHORTEN_FRAC = 0.15

# Liveness gate. We track a *smoothed* per-step rotation rate for the user and
# for the matched reference (a single frame step is only a few degrees, so it
# has to be accumulated). The reference counts as "moving" once its smoothed
# rate exceeds MOTION_THRESH; STRENGTH is how hard a fully frozen user is
# punished during such a passage (0.85 -> keeps 15% of the score).
MOTION_THRESH = np.deg2rad(2.0)
LIVENESS_STRENGTH = 0.9

# Feedback tiers (Q15: textual score). Thresholds on the 0-100 similarity.
TIERS = [(85, "PERFECT"), (70, "SUPER"), (50, "GOOD"), (0, "X")]
TIER_POINTS = {"PERFECT": 100, "SUPER": 75, "GOOD": 50, "X": 0}


def normalize_pose(xy, valid):
    """Hip-centered, torso-scaled keypoints. For display / distance fallback.

    Returns (normalized_xy, valid) or (None, None) if the torso is not visible.
    """
    if not (valid[5] and valid[6] and valid[11] and valid[12]):
        return None, None
    hip = (xy[11] + xy[12]) / 2.0
    shoulder = (xy[5] + xy[6]) / 2.0
    scale = np.linalg.norm(shoulder - hip)
    if scale < 1e-3:
        return None, None
    return (xy - hip) / scale, valid


def _torso_scale(xy, valid):
    """Shoulder-to-hip distance, the yardstick for the foreshortening test."""
    if not (valid[5] and valid[6] and valid[11] and valid[12]):
        return None
    d = float(np.linalg.norm((xy[5] + xy[6]) / 2.0 - (xy[11] + xy[12]) / 2.0))
    return d if d > 1e-3 else None


def _limb_units(xy, valid):
    """Unit vector and pixel length for each limb, plus a usable-limb mask."""
    units = np.zeros((len(LIMBS), 2), np.float32)
    lens = np.zeros(len(LIMBS), np.float32)
    ok = np.zeros(len(LIMBS), bool)
    for i, (a, b) in enumerate(LIMBS):
        if valid[a] and valid[b]:
            v = xy[b] - xy[a]
            n = np.linalg.norm(v)
            if n > 1e-3:
                units[i] = v / n
                lens[i] = n
                ok[i] = True
    return units, lens, ok


def _angle_credit(angle):
    """Per-limb credit: smoothstep from 1 at 0 error to 0 at ANGLE_TOL.

    Smoothstep (vs the old linear ramp) is flat near 0 -- small webcam wobble is
    forgiven -- and drops fast in the middle, so a genuinely wrong limb is
    punished instead of keeping half credit."""
    x = np.clip(1.0 - angle / ANGLE_TOL, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _mean_rotation(u_a, ok_a, u_b, ok_b):
    """Mean angular change (radians) of the limbs valid in both poses."""
    m = ok_a & ok_b
    if not m.any():
        return None
    cos = np.clip(np.sum(u_a[m] * u_b[m], axis=1), -1.0, 1.0)
    return float(np.mean(np.arccos(cos)))


def pose_similarity(xy1, valid1, xy2, valid2):
    """Weighted limb-direction similarity of two poses, on a 0-100 scale.

    xy2/valid2 is treated as the reference (it sets the torso scale and which
    limbs coverage expects). Returns (score, n_matched_limbs); score is None if
    fewer than four limbs are shared.
    """
    u1, l1, ok1 = _limb_units(xy1, valid1)
    u2, l2, ok2 = _limb_units(xy2, valid2)
    both = ok1 & ok2
    if both.sum() < 4:
        return None, int(both.sum())

    cos = np.clip(np.sum(u1[both] * u2[both], axis=1), -1.0, 1.0)
    credit = _angle_credit(np.arccos(cos))

    w = LIMB_WEIGHTS[both].astype(np.float32).copy()
    torso = _torso_scale(xy2, valid2)
    if torso is not None:
        # trust a near-camera (short-projection) limb less
        short = np.clip(np.minimum(l1[both], l2[both]) / (FORESHORTEN_FRAC * torso),
                        0.0, 1.0)
        w = w * short
    if w.sum() < 1e-6:
        return None, int(both.sum())

    score = float(np.sum(w * credit) / np.sum(w)) * 100.0

    # coverage: scale down if the user is not showing the arm/leg limbs the
    # reference is actually using this frame.
    ref_key = ok2 & KEY_LIMBS
    denom = float(LIMB_WEIGHTS[ref_key].sum())
    if denom > 1e-6:
        shown = float(LIMB_WEIGHTS[both & KEY_LIMBS].sum())
        score *= shown / denom

    return score, int(both.sum())


def tier_of(score):
    for thresh, name in TIERS:
        if score >= thresh:
            return name
    return "X"


def scorable(valid):
    """True if enough limbs are visible to score a pose (matches the >=4-limb
    rule in pose_similarity). Used by the GUI to prompt the user to step back."""
    if valid is None:
        return False
    return sum(1 for a, b in LIMBS if valid[a] and valid[b]) >= 4


class PoseScorer:
    """Scores a live webcam pose against a precomputed reference sequence.

    ref_xy    (N,17,2), ref_valid (N,17), ref_t (N,) seconds -- from
    precompute_reference.py.
    """

    def __init__(self, ref_xy, ref_valid, ref_t,
                 acq_back=0.5, acq_fwd=0.2, ema=0.5,
                 track_half=0.13, max_lag=0.6, lag_ema=0.1, mot_ema=0.12):
        self.ref_xy = ref_xy
        self.ref_valid = ref_valid
        self.ref_t = ref_t
        self.acq_back = acq_back      # wide search to first *acquire* the lag
        self.acq_fwd = acq_fwd
        self.ema = ema
        self.track_half = track_half  # once locked, only search +/- this window
        self.max_lag = max_lag        # the lag can never exceed this (anti-freeze)
        self.lag_ema = lag_ema        # how fast the tracked lag adapts
        self.mot_ema = mot_ema        # smoothing for the motion-rate estimates
        self.reset()

    def reset(self):
        self._smooth = None
        self._scores = []             # every per-frame score (0 included)
        self._tiers = {"PERFECT": 0, "SUPER": 0, "GOOD": 0, "X": 0}
        self._lag = None              # tracked user->reference lag in seconds
        self._reset_motion()

    def _reset_motion(self):
        self._pu = self._pu_ok = None   # previous user limb units / mask
        self._pr = self._pr_ok = None   # previous *matched* reference units / mask
        self._umot = None               # smoothed user motion rate (rad/step)
        self._rmot = None               # smoothed reference motion rate

    def _ref_window(self, t_play):
        """Reference frames to compare against this webcam frame.

        Before a lag is known we search a wide window to acquire it. After that
        we only look within track_half of `t_play - lag`. Because the lag is
        clamped to max_lag and adapts slowly, this target *marches forward with
        playback* -- a user who freezes is then compared against a moving
        reference and can no longer keep matching one lucky frame, which is what
        used to let 'stand still' score SUPER.
        """
        if self._lag is None:
            lo = np.searchsorted(self.ref_t, t_play - self.acq_back, "left")
            hi = np.searchsorted(self.ref_t, t_play + self.acq_fwd, "right")
        else:
            c = t_play - self._lag
            lo = np.searchsorted(self.ref_t, c - self.track_half, "left")
            hi = np.searchsorted(self.ref_t, c + self.track_half, "right")
        return lo, max(lo + 1, hi)

    def update(self, user_xy, user_valid, t_play):
        """Score one webcam frame at reference playback time t_play (seconds).

        Returns a dict:
          score   float 0-100 (smoothed)
          raw     float 0-100 (this frame only) or None if unscorable
          tier    PERFECT/SUPER/GOOD/X
          offset  best-matching reference lag in seconds (user - ref) or None
        """
        best_raw, best_j = None, None
        if user_xy is not None:
            lo, hi = self._ref_window(t_play)
            for j in range(lo, min(hi, len(self.ref_xy))):
                s, _ = pose_similarity(user_xy, user_valid,
                                       self.ref_xy[j], self.ref_valid[j])
                if s is None:
                    continue
                if best_raw is None or s > best_raw:
                    best_raw, best_j = s, j

        # Unscorable (user absent or no match): counts as 0, not skipped.
        if best_j is None:
            self._reset_motion()
            return self._record(0.0, raw=None, offset=None)

        # Track the lag, clamped so it can never latch onto a frozen pose.
        implied = float(np.clip(t_play - self.ref_t[best_j], 0.0, self.max_lag))
        self._lag = implied if self._lag is None else \
            float(np.clip((1 - self.lag_ema) * self._lag + self.lag_ema * implied,
                          0.0, self.max_lag))

        score = best_raw * self._liveness(user_xy, user_valid, best_j)
        offset = float(t_play - self.ref_t[best_j])
        return self._record(score, raw=best_raw, offset=offset)

    def _liveness(self, user_xy, user_valid, j):
        """Factor <=1 that punishes a frozen user while the reference moves."""
        u_units, _, u_ok = _limb_units(user_xy, user_valid)
        r_units, _, r_ok = _limb_units(self.ref_xy[j], self.ref_valid[j])
        factor = 1.0
        if self._pu is not None:
            a = self.mot_ema
            ur = _mean_rotation(self._pu, self._pu_ok, u_units, u_ok)
            rr = _mean_rotation(self._pr, self._pr_ok, r_units, r_ok)
            if ur is not None:
                self._umot = ur if self._umot is None else (1 - a) * self._umot + a * ur
            if rr is not None:
                self._rmot = rr if self._rmot is None else (1 - a) * self._rmot + a * rr
            if (self._umot is not None and self._rmot is not None
                    and self._rmot > MOTION_THRESH):
                deficit = float(np.clip(1.0 - self._umot / self._rmot, 0.0, 1.0))
                factor = 1.0 - LIVENESS_STRENGTH * deficit
        self._pu, self._pu_ok = u_units, u_ok
        self._pr, self._pr_ok = r_units, r_ok
        return factor

    def _record(self, score, raw, offset):
        score = float(np.clip(score, 0.0, 100.0))
        self._smooth = score if self._smooth is None else \
            self.ema * score + (1 - self.ema) * self._smooth
        tier = tier_of(self._smooth)
        self._scores.append(self._smooth)
        self._tiers[tier] += 1
        return dict(score=self._smooth, raw=raw, tier=tier, offset=offset)

    def summary(self):
        """Overall score after the dance (Q15)."""
        if not self._scores:
            return dict(mean=0.0, grade="X", tiers=dict(self._tiers), frames=0)
        mean = float(np.mean(self._scores))
        return dict(mean=mean, grade=tier_of(mean),
                    tiers=dict(self._tiers), frames=len(self._scores))
