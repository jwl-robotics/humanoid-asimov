"""Stage 4 — embedded-noise robustness: run the estimator on the clean dataset vs with the embedded
pipeline noise and ablate which effect hurts. Finding: TIMING breaks the estimator, amplitude does not —
latency is the dominant driver (even a uniform latency, contact delayed too, costs ~5x the clean drift at
5 ms), inter-sensor skew adds on top, and quantization/jitter are negligible.

    .venv/bin/python scripts/run_embedded.py
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from humanoid_asimov.embedded import apply_embedded_noise
from humanoid_asimov.estimator import ESKF, RobotKinematics, quat2mat

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "walk_dataset.npz"))
RENDERS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))
OFF = dict(enc_res=0, gyro_lsb=0, accel_lsb=0)          # disable quantization (amplitude) for timing ablations


def run_eskf(log, kin):
    t = log["t"]; dt = float(t[1] - t[0]); jn = list(log["joint_names"])
    e = ESKF(kin, log["gt_pos"][0], quat2mat(log["gt_quat"][0]))
    p = np.empty((len(t), 3))
    for k in range(len(t)):
        e.step(log["gyro"][k], log["accel"][k], log["enc_q"][k], log["enc_qd"][k],
               jn, log["contact"][k], dt)
        p[k] = e.p
    gt = log["gt_pos"]
    ferr = float(np.linalg.norm(p[-1, :2] - gt[-1, :2]))
    path = float(np.sum(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1)))
    bg_y_err = float((e.bg[1] - log["gt_bg"][-1, 1]) * 1e3)     # b_g,y error (mrad/s) — the sensitive axis
    return ferr, ferr / path * 100, bg_y_err


def main():
    d = np.load(DATA, allow_pickle=True)
    log = {k: d[k] for k in d.files}
    log["joint_names"] = list(d["joint_names"])
    kin = RobotKinematics()

    print(f"{'config':40} {'final err':>10} {'drift':>8} {'b_g,y err':>11}")
    R = {}
    def row(name, corrupt):
        lg = corrupt(log) if corrupt else log
        ferr, drift, bge = run_eskf(lg, kin)
        print(f"{name:40} {ferr*100:8.1f}cm {drift:7.2f}% {bge:9.2f}")
        R[name] = drift
        return drift

    base = row("clean (ImuNoise only)", None)
    row("+ embedded (all effects)", lambda L: apply_embedded_noise(L, seed=0))
    print("  -- ablation: each effect alone --")
    q = row("+ quantization only", lambda L: apply_embedded_noise(L, seed=0, stagger_ms=0, latency_ms=0, jitter_ms=0))
    jt = row("+ jitter only (0.8 ms)", lambda L: apply_embedded_noise(L, seed=0, stagger_ms=0, latency_ms=0, **OFF))
    sg = row("+ stagger only (2 ms skew)", lambda L: apply_embedded_noise(L, seed=0, latency_ms=0, jitter_ms=0, **OFF))
    lt = row("+ latency only (5 ms, vs contact)", lambda L: apply_embedded_noise(L, seed=0, stagger_ms=0, jitter_ms=0, **OFF))
    print("  -- latency variant: relative skew (contact not delayed) vs uniform (contact delayed too) --")
    row("+ latency 5 ms, UNIFORM (incl contact)", lambda L: apply_embedded_noise(L, seed=0, stagger_ms=0, jitter_ms=0, delay_contact=True, **OFF))

    lats = [0, 2, 5, 8, 12, 16, 20]
    rel = [run_eskf(apply_embedded_noise(log, seed=0, stagger_ms=0, jitter_ms=0, latency_ms=l, **OFF), kin)[1] for l in lats]
    uni = [run_eskf(apply_embedded_noise(log, seed=0, stagger_ms=0, jitter_ms=0, latency_ms=l, delay_contact=True, **OFF), kin)[1] for l in lats]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.bar(["clean", "quant", "jitter", "stagger\n2ms", "latency\n5ms"], [base, q, jt, sg, lt],
           color=["#888888", "#3aa76a", "#3aa76a", "#e0a030", "#e05050"])
    a1.axhline(base, color="k", ls="--", lw=0.7, alpha=0.4)
    a1.set_ylabel("drift (%)"); a1.set_title("which embedded effect hurts (amplitude vs timing)")
    a2.plot(lats, rel, "-o", color="#e05050", label="relative skew (IMU/enc vs contact)")
    a2.plot(lats, uni, "-o", color="#3fd0c9", label="uniform (all sensors incl contact)")
    a2.set_xlabel("latency (ms)"); a2.set_ylabel("drift (%)"); a2.legend(); a2.grid(alpha=.3)
    a2.set_title("latency is the dominant drift driver (skew adds to it)")
    fig.suptitle("Stage 4 — embedded-pipeline noise: timing dominates amplitude", fontsize=13)
    os.makedirs(RENDERS, exist_ok=True); fig.tight_layout()
    fig.savefig(os.path.join(RENDERS, "stage4_embedded.png"), dpi=110)
    print(f"saved {os.path.join(RENDERS, 'stage4_embedded.png')}")


if __name__ == "__main__":
    main()
