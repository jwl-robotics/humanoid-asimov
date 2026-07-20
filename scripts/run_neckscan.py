"""Stage 3 follow-up — active calibration via a neck "look-around" scan.

A pure yaw-loop observes only the camera-mount axis perpendicular to the turn; the axis near the turn
and the camera–IMU time offset td stay weakly observable (run_calib freezes td for exactly that
reason). A neck-pitch scan adds a second rotation axis, which should buy the missing observability —
but it was tried during Stage 3 and HURT: the then-~6 mrad front-end lost KLT tracks under camera
pitch. The front-end has since been sharpened (IRLS refit on all inliers, ~2 mrad), so this script
retries the idea properly: sweep gentle (amplitude, frequency) scans, estimate the full calibration
(mount AND td, both unfrozen, known injected errors), and measure both sides of the trade —
observability (final ±σ funnels on the weak axes and td, recovery of the injected errors) vs
track quality (front-end success rate, filter gate acceptance, inlier counts, re-anchor churn).

    .venv/bin/python scripts/run_neckscan.py
"""
import os

import cv2
import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from humanoid_asimov.calib import CalibVioESKF
from humanoid_asimov.estimator import RobotKinematics, exp_so3, log_so3, quat2mat
from humanoid_asimov.scene import HEAD_CAM_QUAT
from humanoid_asimov.sensors import ImuNoise
from humanoid_asimov.vision import AnchorRotationEstimator, CameraModel, FeatureTracker, HeadCamExtrinsics
from humanoid_asimov.walk import loop_heading, simulate

RENDERS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))
_GL2CV = np.diag([1.0, -1.0, -1.0])
PERIOD, KF_EVERY, ANCHOR_MAX_AGE, ANCHOR_MIN_SHARED = 24.0, 15, 4.0, 30
DPHI_TRUE = np.radians([1.5, -2.5, 1.0])      # same ~3.1° injected mount error as run_calib
TD_TRUE = 0.010                                # + a 10 ms camera clock offset, unfrozen and estimated

# (amplitude rad, freq Hz) scan grid — "gentle" by design, plus one deliberately aggressive point to
# find where track quality breaks. None = the Stage-3 baseline (no scan).
SCANS = [None, (0.10, 0.15), (0.20, 0.15), (0.35, 0.15),
         (0.10, 0.30), (0.20, 0.30), (0.35, 0.30), (0.35, 0.60)]


def _quat_from_rotvec(v):
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, v / ang, ang); return q


def _quat_mul(a, b):
    q = np.zeros(4); mujoco.mju_mulQuat(q, a, b); return q


def frontend(frames, fidx, ft_t):
    """run_calib's front-end (no prior gate — the filter's chi-square gate is the only rejector), plus
    per-keyframe quality diagnostics."""
    cam = CameraModel()
    tracker, anchor = FeatureTracker(fb_thresh=1.5), AnchorRotationEstimator(cam)
    keyframes, anchor_set = [], False
    stats = dict(attempts=0, ok=0, inliers=[], shared=[], reanchors=0)
    for k in range(len(frames)):
        kf = (k % KF_EVERY == 0)
        tracks = tracker.process(cv2.cvtColor(frames[k], cv2.COLOR_RGB2GRAY), keyframe=kf)
        if not kf:
            continue
        i, tt = int(fidx[k]), float(ft_t[k])
        if not anchor_set:
            anchor.set_anchor(tt, tracks); anchor_set = True
            keyframes.append((i, None, True)); continue
        stats["attempts"] += 1
        m = anchor.estimate(tt, tracks)
        if m is not None:
            stats["ok"] += 1
            stats["inliers"].append(m["n_inliers"]); stats["shared"].append(m["n_shared"])
        n_shared = sum(1 for x in tracks if x in anchor.anchor_uv)
        reanchor = (m is None) or (tt - anchor.anchor_t > ANCHOR_MAX_AGE) or (n_shared < ANCHOR_MIN_SHARED)
        stats["reanchors"] += bool(reanchor)
        keyframes.append((i, m["R"] if m is not None else None, reanchor))
        if reanchor:
            anchor.set_anchor(tt, tracks)
    return keyframes, stats


def run_one(neck_scan, seed=0):
    q_true = _quat_mul(np.array(HEAD_CAM_QUAT, float), _quat_from_rotvec(_GL2CV @ DPHI_TRUE))
    _, _, log, frames = simulate(seconds=PERIOD, heading=loop_heading(PERIOD), environment=True,
                                 render_camera="head_cam", fps=30, imu_noise=ImuNoise(), seed=seed,
                                 head_cam_quat=q_true, neck_scan=neck_scan)
    frames = np.array(frames)
    t, gyro, accel = log["t"], log["gyro"], log["accel"]
    enc_q, enc_qd, contact = log["enc_q"], log["enc_qd"], log["contact"]
    jn = list(log["joint_names"]); dt = float(t[1] - t[0])
    ft_shift = log["frame_t"] - TD_TRUE                    # relabel stamps ⇒ content at stamp+td
    fidx = np.array([int(np.argmin(np.abs(t - tt))) for tt in ft_shift])

    kin = RobotKinematics(); ext = HeadCamExtrinsics(kin)
    R_HC_nom = ext.R_HC.copy()
    R_HC_true = R_HC_nom @ exp_so3(DPHI_TRUE)
    keyframes, fe = frontend(frames, fidx, ft_shift)
    kf_by_idx = {i: (Rm, ca) for (i, Rm, ca) in keyframes}

    vio = CalibVioESKF(kin, log["gt_pos"][0], quat2mat(log["gt_quat"][0]), R_HC0=R_HC_nom,
                       td0=0.0, est_extrinsic=True, est_td=True)
    n_delivered = n_fused = 0
    sphi_t, std_t, tl = [], [], []
    for k in range(len(t)):
        vio.step(gyro[k], accel[k], enc_q[k], enc_qd[k], jn, contact[k], dt)
        if k in kf_by_idx:
            R_meas, clone_after = kf_by_idx[k]
            R_BH_j = ext.head_pose(enc_q[k], jn)[0]
            w_neck = ext.head_angvel_base(enc_q[k], enc_qd[k], jn)
            if R_meas is not None:
                n_delivered += 1
                n_fused += bool(vio.update_rel_rot(R_meas, R_BH_j, gyro[k], w_neck))
            if clone_after:
                vio.clone(R_BH_j, gyro[k], w_neck)
        tl.append(t[k])
        sphi_t.append(np.sqrt(np.maximum(np.diag(vio.P[18:21, 18:21]), 0.0)))
        std_t.append(np.sqrt(max(vio.P[21, 21], 0.0)))

    ephi = log_so3(vio.R_HC.T @ R_HC_true)                 # remaining mount error (rad)
    ni = int(jn.index("neck_pitch_joint"))
    amp_achieved = float(np.percentile(np.abs(enc_q[:, ni]), 99))
    return dict(neck_scan=neck_scan, ephi=ephi, sphi=np.array(sphi_t), std=np.array(std_t), tl=np.array(tl),
                etd=float(vio.td - TD_TRUE), fe=fe, n_delivered=n_delivered, n_fused=n_fused,
                amp_achieved=amp_achieved)


def main():
    results = []
    for scan in SCANS:
        name = "no scan (baseline)" if scan is None else f"amp {scan[0]:.2f} rad · {scan[1]:.2f} Hz"
        r = run_one(scan)
        results.append(r)
        fe = r["fe"]
        peak = 0.0 if scan is None else scan[0] * 2 * np.pi * scan[1]
        print(f"{name:26} | achieved amp {np.degrees(r['amp_achieved']):4.1f}° peak {peak:4.2f} rad/s | "
              f"fe {fe['ok']}/{fe['attempts']} ({100*fe['ok']/max(fe['attempts'],1):3.0f}%) "
              f"inl {np.mean(fe['inliers']):5.1f} reanc {fe['reanchors']:2d} | "
              f"fused {r['n_fused']}/{r['n_delivered']} | "
              f"phi err {np.round(np.abs(r['ephi'])*1e3,1)} mrad "
              f"sig [{r['sphi'][-1][0]*1e3:.1f} {r['sphi'][-1][1]*1e3:.1f} {r['sphi'][-1][2]*1e3:.1f}] | "
              f"td err {r['etd']*1e3:+5.1f} ms sig {r['std'][-1]*1e3:4.1f}", flush=True)

    # ---- figure: observability gain vs track-quality cost --------------------
    labels = ["none" if r["neck_scan"] is None else f"{r['neck_scan'][0]:.2f}r\n{r['neck_scan'][1]:.2f}Hz"
              for r in results]
    x = np.arange(len(results))
    sy = [r["sphi"][-1][1] * 1e3 for r in results]         # weak mount axis (≈ turn axis)
    st = [r["std"][-1] * 1e3 for r in results]
    fer = [100 * r["fe"]["ok"] / max(r["fe"]["attempts"], 1) for r in results]
    fur = [100 * r["n_fused"] / max(r["n_delivered"], 1) for r in results]
    ete = [abs(r["etd"]) * 1e3 for r in results]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))
    w = 0.27
    a1.bar(x - w, sy, w, color="#3fd0c9", label="σ φ_y — weak mount axis (mrad)")
    a1.bar(x, st, w, color="#e0a030", label="σ td (ms)")
    a1.bar(x + w, ete, w, color="#e05050", label="|td error| (ms)")
    a1.set_xticks(x, labels, fontsize=8); a1.legend(fontsize=8); a1.grid(alpha=.25, axis="y")
    a1.set_title("what the scan buys: weak-axis + time-offset funnels")
    a2.plot(x, fer, "-o", color="#3aa76a", label="front-end success (%)")
    a2.plot(x, fur, "-s", color="#6ea8fe", label="filter gate acceptance (%)")
    a2.set_xticks(x, labels, fontsize=8); a2.set_ylim(0, 105); a2.legend(fontsize=8); a2.grid(alpha=.25)
    a2.set_title("what the scan costs: track quality under camera pitch")
    fig.suptitle("Stage 3 — active calibration: neck-scan sweep (mount + td estimated, errors injected)",
                 fontsize=12)
    os.makedirs(RENDERS, exist_ok=True); fig.tight_layout()
    fig.savefig(os.path.join(RENDERS, "stage3_neckscan.png"), dpi=110)
    print(f"saved {os.path.join(RENDERS, 'stage3_neckscan.png')}")


if __name__ == "__main__":
    main()
