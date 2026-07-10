"""Stage 1 consistency validation: per-block NEES + contact NIS for the ESKF.

Scores the *covariance's honesty*, not just the drift. Reports **per-block** NEES
(pos / vel / tilt / yaw / b_g / b_a) — the diverging unobservable modes (yaw, absolute position) would
mask the observable ones in a lumped 15-dof NEES — plus an M-run ANEES and the contact-update NIS +
innovation whiteness. Compares two tunings:
  BEFORE — 15-state filter, inflated process noise (the initial guess; slip-bias frozen off)
  AFTER  — 18-state: Q matched to the injected LSM6DSOX noise + the slip-velocity bias b_c estimated online
Consistent ≈ ANEES within the chi-square band; above = optimistic (P too small), below = conservative.

    .venv/bin/python scripts/run_nees.py
"""
import os

import numpy as np
from scipy.stats import chi2

from humanoid_asimov.estimator import ESKF, RobotKinematics, log_so3, quat2mat
from humanoid_asimov.sensors import ImuNoise
from humanoid_asimov.walk import simulate

RENDERS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))
M = 10                                                   # seeds for the M-run ANEES
BEFORE = dict(sig_a=0.08, sig_g=0.006, sig_bg=5e-5, sig_ba=1e-3, sig_contact=0.10,
              sig_bc=0.0, p0_bc=0.0)                              # old 15-state (slip-bias frozen off)
AFTER = dict(sig_a=6.7e-4, sig_g=4.5e-5, sig_bg=2e-4, sig_ba=4e-3, sig_contact=0.05,
             sig_bc=1e-3, p0_bc=1e-3)                            # shared slip-bias, matched-noise Q (best estimate)
BLOCKS = {"pos": slice(0, 3), "vel": slice(3, 6), "bg": slice(9, 12), "ba": slice(12, 15)}
DOF = {"pos": 3, "vel": 3, "tilt": 2, "yaw": 1, "bg": 3, "ba": 3, "all": 15}


def _theta_basis(a):
    a = a / np.linalg.norm(a)
    tmp = np.array([1., 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1., 0])
    b1 = np.cross(a, tmp); b1 /= np.linalg.norm(b1)
    return a, np.stack([b1, np.cross(a, b1)], axis=1)     # yaw axis (3,), tilt basis (3x2)


def run_filter(kin, data, cfg):
    """Run the ESKF over one dataset; return per-step per-block NEES + contact NIS + innovations."""
    t, gyro, accel = data["t"], data["gyro"], data["accel"]
    enc_q, enc_qd, contact = data["enc_q"], data["enc_qd"], data["contact"]
    gt_pos, gt_quat, gt_v = data["gt_pos"], data["gt_quat"], data["gt_linvel"]
    gt_bg, gt_ba = data["gt_bg"], data["gt_ba"]
    jn = list(data["joint_names"]); dt = float(t[1] - t[0])
    e = ESKF(kin, gt_pos[0], quat2mat(gt_quat[0]), **cfg)
    nees = {k: [] for k in DOF}
    nis, innov = [], []
    for k in range(len(t)):
        e.step(gyro[k], accel[k], enc_q[k], enc_qd[k], jn, contact[k], dt)
        dth = log_so3(e.R.T @ quat2mat(gt_quat[k]))       # body-frame orientation error
        err = np.concatenate([gt_pos[k] - e.p, gt_v[k] - e.v, dth, gt_bg[k] - e.bg, gt_ba[k] - e.ba])
        P = 0.5 * (e.P + e.P.T)
        for name, sl in BLOCKS.items():
            nees[name].append(float(err[sl] @ np.linalg.solve(P[sl, sl], err[sl])))
        Pth = P[6:9, 6:9]
        a, B = _theta_basis(e.R.T @ np.array([0, 0, 1.]))  # gravity axis in body = yaw dof
        nees["yaw"].append(float((a @ dth) ** 2 / (a @ Pth @ a)))
        e2 = B.T @ dth
        nees["tilt"].append(float(e2 @ np.linalg.solve(B.T @ Pth @ B, e2)))
        nees["all"].append(float(err @ np.linalg.solve(P[:15, :15], err)))   # marginal over the GT-scored 15
        if e.last_nis is not None:
            nis.append(e.last_nis); innov.append(e.last_innov)
    return {k: np.array(v) for k, v in nees.items()}, np.array(nis), np.array(innov), e


def main():
    kin = RobotKinematics()
    acc = {"BEFORE": {k: [] for k in DOF}, "AFTER": {k: [] for k in DOF}}
    nis_all = {"BEFORE": [], "AFTER": []}
    innov_all = {"BEFORE": [], "AFTER": []}
    seed0, est0, gt_bg0 = {}, {}, None
    for s in range(M):
        _, _, data, _ = simulate(seconds=8.0, imu_noise=ImuNoise(), seed=s)
        if s == 0:
            gt_bg0 = data["gt_bg"][-1]
        for tag, cfg in (("BEFORE", BEFORE), ("AFTER", AFTER)):
            nees, nis, innov, est = run_filter(kin, data, cfg)
            for k in DOF:
                acc[tag][k].append(nees[k])
            nis_all[tag].append(nis); innov_all[tag].append(innov)
            if s == 0:
                seed0[tag] = nees; est0[tag] = est
        print(f"  seed {s+1}/{M} done", flush=True)

    print(f"\n=== per-block ANEES (M={M} seeds) — consistent if inside the band ===")
    print(f"{'block':6}{'dof':>4}   {'BEFORE':>18}   {'AFTER':>18}   band")
    for k in ("pos", "vel", "tilt", "yaw", "bg", "ba", "all"):
        d = DOF[k]; lo, hi = chi2.ppf(0.025, M * d) / M, chi2.ppf(0.975, M * d) / M
        b = float(np.mean([x.mean() for x in acc["BEFORE"][k]]))
        a = float(np.mean([x.mean() for x in acc["AFTER"][k]]))
        tag = lambda v: "ok" if lo <= v <= hi else ("OPTIMISTIC" if v > hi else "conservative")
        print(f"{k:6}{d:>4}   {b:8.2f} {tag(b):>9}   {a:8.2f} {tag(a):>9}   [{lo:.2f},{hi:.2f}]")

    print("\n=== contact NIS (dof 3, mean≈3) + innovation whiteness ===")
    for tag in ("BEFORE", "AFTER"):
        nis = np.concatenate(nis_all[tag]); iv = np.concatenate(innov_all[tag])
        r1 = float(np.mean([np.corrcoef(iv[:-1, i], iv[1:, i])[0, 1] for i in range(3)]))
        inband = float(np.mean((nis > chi2.ppf(0.025, 3)) & (nis < chi2.ppf(0.975, 3))) * 100)
        print(f"{tag:6}: NIS mean {nis.mean():5.2f} median {np.median(nis):5.2f} | {inband:4.0f}% in band | lag-1 autocorr {r1:+.2f}")

    print("\n=== bias honesty (seed 0, final) — b_g error should shrink; b_c = learned slip ===")
    for tag in ("BEFORE", "AFTER"):
        e = est0[tag]
        be = (e.bg - gt_bg0) * 1e3
        print(f"{tag:6}: b_g err (mrad/s) [{be[0]:+.2f},{be[1]:+.2f},{be[2]:+.2f}]"
              f" | learned b_c (mm/s) [{e.bc[0]*1e3:+.1f},{e.bc[1]*1e3:+.1f},{e.bc[2]*1e3:+.1f}]")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(len(seed0["AFTER"]["pos"])) * 0.002
        fig, ax = plt.subplots(2, 3, figsize=(16, 8))
        for i, k in enumerate(("pos", "vel", "tilt", "yaw", "bg", "ba")):
            a = ax[i // 3, i % 3]; d = DOF[k]
            a.axhspan(chi2.ppf(0.025, d), chi2.ppf(0.975, d), color="#3fd0c9", alpha=0.12)
            a.plot(t, seed0["BEFORE"][k], color="#e0a030", lw=0.7, alpha=0.8, label="before")
            a.plot(t, seed0["AFTER"][k], color="#3fd0c9", lw=0.9, label="after")
            a.axhline(d, color="w", lw=0.7, ls="--", alpha=0.5)
            a.set_title(f"NEES · {k} (dof {d})"); a.set_yscale("log"); a.grid(alpha=0.2)
            a.legend(fontsize=7); a.set_facecolor("#0a0e12")
        fig.patch.set_facecolor("#0a0e12")
        os.makedirs(RENDERS, exist_ok=True)
        out = os.path.join(RENDERS, "stage1_nees.png")
        fig.tight_layout(); fig.savefig(out, dpi=110, facecolor="#0a0e12")
        print(f"\nsaved {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
