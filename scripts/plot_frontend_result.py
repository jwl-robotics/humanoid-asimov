"""Summary figure for the Stage-2 front-end sharpening (docs/STAGE2_FRONTEND_DESIGN.md).

Numbers are the measured results — left panel from scripts/run_frontend.py (the money gate, before vs
after the IRLS refit + textured far field), right panel from scripts/run_vio.py over seeds 0-4 at the
honest sig_vis=2 mrad. Re-measure with those scripts if the front-end changes; this only draws them.

    .venv/bin/python scripts/plot_frontend_result.py
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RENDERS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))

# --- measured ---
GATE = {"median": (7.69, 2.06), "bias": (4.03, 0.36)}          # mrad, (before, after)
SEEDS = [0, 1, 2, 3, 4]
VIO = [0.20, 0.02, 0.26, 0.15, 0.14]                            # % drift, honest sig_vis=2 mrad
ESKF = [0.23, 0.73, 0.49, 0.45, 0.67]

AMBER, CYAN, RED = "#d9a23f", "#3fd0c9", "#e05050"


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(2)
    a1.bar(x - 0.2, [GATE["median"][0], GATE["bias"][0]], 0.38, color=AMBER, label="5-point minimal model (before)")
    a1.bar(x + 0.2, [GATE["median"][1], GATE["bias"][1]], 0.38, color=CYAN, label="IRLS refit on all inliers (after)")
    a1.axhline(2.0, color="k", ls="--", lw=0.8, alpha=0.5)
    a1.text(1.35, 2.15, "honest σ_vis = 2 mrad", fontsize=9, color="k", alpha=0.7)
    a1.set_xticks(x); a1.set_xticklabels(["median |rotation error|", "systematic bias"])
    a1.set_ylabel("front-end error (mrad)")
    a1.set_title("Money gate: the estimator was the bottleneck, not the tracks")
    a1.legend(fontsize=9); a1.grid(axis="y", alpha=.3)
    for xi, (b, a) in zip(x, (GATE["median"], GATE["bias"])):
        a1.annotate(f"{b:.1f}", (xi - 0.2, b), ha="center", va="bottom", fontsize=9, color=AMBER)
        a1.annotate(f"{a:.2f}", (xi + 0.2, a), ha="center", va="bottom", fontsize=9, color="#1a8f88")

    xs = np.arange(len(SEEDS))
    a2.bar(xs - 0.2, ESKF, 0.38, color=RED, label="ESKF (IMU + contact)")
    a2.bar(xs + 0.2, VIO, 0.38, color=CYAN, label="VIO (+ vision)")
    a2.set_xticks(xs); a2.set_xticklabels([f"seed {s}" for s in SEEDS])
    a2.set_ylabel("loop drift (% of path)")
    a2.set_title("VIO beats the ESKF on every seed — at the honest 2 mrad noise")
    a2.legend(fontsize=9); a2.grid(axis="y", alpha=.3)
    a2.text(0.5, 0.62, f"median  VIO {np.median(VIO):.2f}%   vs   ESKF {np.median(ESKF):.2f}%",
            transform=a2.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="#eef6f5", ec=CYAN))

    fig.suptitle("Stage 2 front-end sharpening — an IRLS refit recovers the √N averaging the minimal solver throws away",
                 fontsize=12)
    os.makedirs(RENDERS, exist_ok=True); fig.tight_layout()
    fig.savefig(os.path.join(RENDERS, "stage2_frontend.png"), dpi=110)
    print(f"saved {os.path.join(RENDERS, 'stage2_frontend.png')}")


if __name__ == "__main__":
    main()
