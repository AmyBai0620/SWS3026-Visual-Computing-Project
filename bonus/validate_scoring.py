"""Offline validation of the Task 2 scoring math -- no webcam needed.

We fabricate a "user" from the reference itself so we know the ground truth,
then check the three claims the rubric (Q14) asks us to back up:

  1. Spatial invariance  : a user who is a *different size and in a different
     part of the frame* but in the SAME pose scores ~100.
  2. Temporal alignment  : a user who lags the reference by a known number of
     frames is still matched, and the recovered offset equals that lag.
  3. Discrimination      : a user doing the WRONG movement (reference played
     backwards) scores much lower than a matching user.

Run:  python validate_scoring.py [name]
"""

import os
import sys

import numpy as np

from pose_score import PoseScorer, pose_similarity, normalize_pose

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def load_ref(name):
    for cand in (os.path.join(HERE, f"ref_{name}.npz"),
                 os.path.join(HERE, "video", f"ref_{name}.npz")):
        if os.path.exists(cand):
            d = np.load(cand)
            return d["xy"], d["valid"], d["t"], float(d["fps"])
    raise SystemExit(f"run precompute_reference.py {name} first")


def transform(xy, scale, tx, ty, noise_px):
    """Mimic a different-sized dancer elsewhere in frame, with jitter."""
    out = xy * scale + np.array([tx, ty], np.float32)
    out = out + rng.normal(0, noise_px, out.shape).astype(np.float32)
    return out


def simulate(ref_xy, ref_valid, ref_t, fps, lag_frames, reverse=False,
             scale=1.4, tx=250.0, ty=120.0, noise_px=3.0):
    scorer = PoseScorer(ref_xy, ref_valid, ref_t)
    n = len(ref_xy)
    offsets = []
    for i in range(n):
        src = (n - 1 - (i - lag_frames)) if reverse else (i - lag_frames)
        if src < 0 or src >= n or not ref_valid[src].any():
            scorer.update(None, None, float(ref_t[i]))
            continue
        user_xy = transform(ref_xy[src], scale, tx, ty, noise_px)
        r = scorer.update(user_xy, ref_valid[src], float(ref_t[i]))
        if r["offset"] is not None:
            offsets.append(r["offset"])
    return scorer.summary(), np.array(offsets)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "dance_example_5"
    ref_xy, ref_valid, ref_t, fps = load_ref(name)
    lag_frames = 5
    lag_s = lag_frames / fps

    print(f"reference: {name}  {len(ref_xy)} frames @ {fps:.0f} fps")
    print(f"injected lag: {lag_frames} frames = {lag_s * 1000:.0f} ms\n")

    # --- 1 & 2: matching user, scaled+translated+lagged --------------------
    summ, offs = simulate(ref_xy, ref_valid, ref_t, fps, lag_frames)
    print("MATCHING user (1.4x size, +250,+120 px, 3px noise, 5-frame lag):")
    print(f"  overall score : {summ['mean']:.1f} / 100   grade {summ['grade']}")
    print(f"  tier counts   : {summ['tiers']}")
    print(f"  recovered lag : median {np.median(offs) * 1000:.0f} ms "
          f"(injected {lag_s * 1000:.0f} ms)")

    # --- 3: wrong movement (reference reversed) ---------------------------
    summ_w, _ = simulate(ref_xy, ref_valid, ref_t, fps, lag_frames, reverse=True)
    print("\nWRONG user (reference played backwards):")
    print(f"  overall score : {summ_w['mean']:.1f} / 100   grade {summ_w['grade']}")
    print(f"  tier counts   : {summ_w['tiers']}")

    # --- spatial-invariance spot check ------------------------------------
    i = len(ref_xy) // 2
    a_n, _ = normalize_pose(ref_xy[i], ref_valid[i])
    b_xy = transform(ref_xy[i], 1.8, 400, 300, 0.0)
    b_n, _ = normalize_pose(b_xy, ref_valid[i])
    same, _ = pose_similarity(ref_xy[i], ref_valid[i], b_xy, ref_valid[i])
    print("\nspatial invariance (same pose, 1.8x + shifted, no noise):")
    print(f"  limb-cosine score      : {same:.1f} / 100  (expect ~100)")
    print(f"  normalized max abs diff : {np.nanmax(np.abs(a_n - b_n)):.4f}  (expect ~0)")

    ok = (summ["mean"] > 80 and summ_w["mean"] < summ["mean"] - 15
          and abs(np.median(offs) - lag_s) < 1.5 / fps and same > 98)
    print("\nRESULT:", "PASS" if ok else "CHECK — thresholds not all met")


if __name__ == "__main__":
    main()
