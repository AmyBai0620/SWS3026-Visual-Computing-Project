"""Dance scoring for Task 2 (Just Dance) -- the answer to rubric Q14 & Q15.

Q14  How do we score, align, and which similarity metric?
  * Spatial alignment  : we compare *limb direction vectors*, whose cosine is
    invariant to translation and scale by construction. So a small user and a
    large reference dancer, standing in different parts of the frame, still
    match perfectly when they strike the same pose -- no manual registration
    needed. (normalize_pose() is also provided for display / distance fallback.)
  * Similarity metric  : per limb we take the angle between the user's and the
    reference's segment direction, and give full credit at 0 degrees fading to
    zero at ANGLE_TOL. The pose score is the weighted mean over matched limbs,
    on a 0-100 scale. This angular-tolerance mapping is far more discriminative
    than (cos+1)/2, which floors every roughly-human pose near 50.
  * Temporal alignment : the user lags the reference. For each webcam frame we
    search a short window of reference frames around the current playback time
    and take the best match, which tolerates a variable lag.

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

# A limb pointing more than ANGLE_TOL away from the target scores zero; within
# it, credit falls off linearly with the angular error. 70 deg is forgiving
# enough for webcam noise while still punishing a genuinely wrong limb.
ANGLE_TOL = np.deg2rad(70.0)

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


def _limb_units(xy, valid):
    """Unit vector for each limb, plus a mask of which limbs are usable."""
    units = np.zeros((len(LIMBS), 2), np.float32)
    ok = np.zeros(len(LIMBS), bool)
    for i, (a, b) in enumerate(LIMBS):
        if valid[a] and valid[b]:
            v = xy[b] - xy[a]
            n = np.linalg.norm(v)
            if n > 1e-3:
                units[i] = v / n
                ok[i] = True
    return units, ok


def pose_similarity(xy1, valid1, xy2, valid2):
    """Mean weighted cosine similarity of matched limbs, on a 0-100 scale.

    Returns (score, n_limbs). score is None if too few limbs are shared.
    """
    u1, ok1 = _limb_units(xy1, valid1)
    u2, ok2 = _limb_units(xy2, valid2)
    both = ok1 & ok2
    if both.sum() < 4:
        return None, int(both.sum())
    cos = np.clip(np.sum(u1[both] * u2[both], axis=1), -1.0, 1.0)
    angle = np.arccos(cos)                              # 0..pi per limb
    sim01 = np.clip(1.0 - angle / ANGLE_TOL, 0.0, 1.0)  # full credit -> 0 at TOL
    w = LIMB_WEIGHTS[both]
    score = float(np.sum(w * sim01) / np.sum(w)) * 100.0
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
                 win_back=0.5, win_fwd=0.2, ema=0.5):
        self.ref_xy = ref_xy
        self.ref_valid = ref_valid
        self.ref_t = ref_t
        self.win_back = win_back    # search this many seconds *before* now (lag)
        self.win_fwd = win_fwd      # and this many *after*
        self.ema = ema
        self.reset()

    def reset(self):
        self._smooth = None
        self._scores = []           # every accepted per-frame score
        self._tiers = {"PERFECT": 0, "SUPER": 0, "GOOD": 0, "X": 0}

    def _ref_window(self, t_play):
        lo = np.searchsorted(self.ref_t, t_play - self.win_back, "left")
        hi = np.searchsorted(self.ref_t, t_play + self.win_fwd, "right")
        return lo, max(lo + 1, hi)

    def update(self, user_xy, user_valid, t_play):
        """Score one webcam frame at reference playback time t_play (seconds).

        Returns a dict:
          score      float 0-100 (smoothed) or None if unscorable this frame
          raw        float 0-100 (this frame only) or None
          tier       PERFECT/SUPER/GOOD/X or None
          offset     best-matching reference lag in seconds (user - ref)
        """
        if user_xy is None:
            return dict(score=self._smooth, raw=None, tier=None, offset=None)

        lo, hi = self._ref_window(t_play)
        best, best_idx = None, None
        for j in range(lo, min(hi, len(self.ref_xy))):
            s, _ = pose_similarity(user_xy, user_valid,
                                   self.ref_xy[j], self.ref_valid[j])
            if s is not None and (best is None or s > best):
                best, best_idx = s, j

        if best is None:
            return dict(score=self._smooth, raw=None, tier=None, offset=None)

        # temporal smoothing so the tier does not flicker frame to frame
        self._smooth = best if self._smooth is None else \
            self.ema * best + (1 - self.ema) * self._smooth
        tier = tier_of(self._smooth)

        self._scores.append(self._smooth)
        self._tiers[tier] += 1
        offset = float(t_play - self.ref_t[best_idx])

        return dict(score=self._smooth, raw=best, tier=tier, offset=offset)

    def summary(self):
        """Overall score after the dance (Q15)."""
        if not self._scores:
            return dict(mean=0.0, grade="X", tiers=dict(self._tiers), frames=0)
        mean = float(np.mean(self._scores))
        return dict(mean=mean, grade=tier_of(mean),
                    tiers=dict(self._tiers), frames=len(self._scores))
