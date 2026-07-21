"""Score-over-time figure for Task 2 -- the visual answer to rubric Q15.

Q15 asks "how does the score change over time as the user is dancing?".
We simulate a user (the reference itself, resized / shifted / lagged / noised,
so ground truth is known) and plot the smoothed per-frame score across the whole
dance, alongside a wrong-movement run for contrast. Also draws the feedback-tier
bands (PERFECT / SUPER / GOOD / X).

Run:  python score_timeline.py [name]      (needs ref_<name>.npz)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pose_score import PoseScorer, TIERS

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)


def load_ref(name):
    for cand in (os.path.join(HERE, f"ref_{name}.npz"),
                 os.path.join(HERE, "video", f"ref_{name}.npz")):
        if os.path.exists(cand):
            d = np.load(cand)
            return d["xy"], d["valid"], d["t"], float(d["fps"])
    raise SystemExit(f"run precompute_reference.py {name} first")


def _torso(xy, valid):
    if not (valid[5] and valid[6] and valid[11] and valid[12]):
        return 100.0
    return float(np.linalg.norm((xy[5] + xy[6]) / 2 - (xy[11] + xy[12]) / 2)) or 100.0


def simulate(ref_xy, ref_valid, ref_t, lag_frames=5, reverse=False,
             noise_frac=0.05, vary_lag=False):
    """Noise is a fraction of torso length (so it perturbs limb ANGLES, not
    just pixels). vary_lag makes the user drift in and out of time."""
    scorer = PoseScorer(ref_xy, ref_valid, ref_t)
    n = len(ref_xy)
    ts, sc = [], []
    for i in range(n):
        lag = lag_frames
        if vary_lag:                       # timing drifts 3..22 frames over the song
            lag = int(12 + 9 * np.sin(2 * np.pi * i / 90))
        src = (n - 1 - (i - lag)) if reverse else (i - lag)
        if 0 <= src < n and ref_valid[src].any():
            sigma = noise_frac * _torso(ref_xy[src], ref_valid[src])
            user = ref_xy[src] * 1.35 + np.array([200, 90], np.float32)
            user = user + rng.normal(0, sigma, user.shape).astype(np.float32)
            r = scorer.update(user, ref_valid[src], float(ref_t[i]))
        else:
            r = scorer.update(None, None, float(ref_t[i]))
        ts.append(ref_t[i])
        sc.append(r["score"] if r["score"] is not None else np.nan)
    return np.array(ts), np.array(sc), scorer.summary()


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "dance_example_5"
    ref_xy, ref_valid, ref_t, fps = load_ref(name)

    # a good dancer; a mistimed+imprecise dancer; someone doing the wrong moves
    t1, s1, sum1 = simulate(ref_xy, ref_valid, ref_t, lag_frames=5, noise_frac=0.05)
    t2, s2, sum2 = simulate(ref_xy, ref_valid, ref_t, noise_frac=0.14, vary_lag=True)
    t3, s3, sum3 = simulate(ref_xy, ref_valid, ref_t, lag_frames=5, reverse=True, noise_frac=0.05)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    band_colors = {"PERFECT": "#fff4cc", "SUPER": "#d9f2d9", "GOOD": "#dce8fb", "X": "#eeeeee"}
    edges = [100] + [t for t, _ in TIERS]      # 100,85,70,50,0
    names = [n for _, n in TIERS]
    for top, bot, nm in zip(edges[:-1], edges[1:], names):
        ax.axhspan(bot, top, color=band_colors[nm], alpha=0.7, zorder=0)
        ax.text(ref_t[-1] * 1.005, (top + bot) / 2, nm, va="center",
                fontsize=8, color="#555")

    ax.plot(t1, s1, color="#1a7f37", lw=2, label=f"good dancer (overall {sum1['mean']:.0f})")
    ax.plot(t2, s2, color="#d97706", lw=1.6, label=f"mistimed + imprecise (overall {sum2['mean']:.0f})")
    ax.plot(t3, s3, color="#b91c1c", lw=1.4, ls="--", label=f"wrong moves (overall {sum3['mean']:.0f})")

    ax.set_xlim(0, ref_t[-1] * 1.06)
    ax.set_ylim(0, 100)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("smoothed score")
    ax.set_title(f"Task 2 score over time — {name}")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()

    out = os.path.join(HERE, f"task2_score_timeline_{name}.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    for tag, s in (("good", sum1), ("sloppy", sum2), ("wrong", sum3)):
        print(f"  {tag:7s} overall {s['mean']:5.1f}  grade {s['grade']:8s} tiers {s['tiers']}")


if __name__ == "__main__":
    main()
