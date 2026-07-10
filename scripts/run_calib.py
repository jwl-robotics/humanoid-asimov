"""Stage 3 — online self-calibration demo: inject a known camera-mount error and recover it while walking
(no rig, no ground truth fed to the filter).

The sim renders a ~3.1° perturbed camera mount; the estimator starts nominal and self-calibrates online from
the loop's vision. Prints injected vs recovered and plots the convergence. Shows the observability structure
the design predicts: the mount axis ⊥ the turn converges fast (tight covariance) while components with a
projection onto the turn axis stay weakly observable (wide covariance). (The camera–IMU time offset is also
weakly observable on a near-constant-rate loop, see STAGE3_CALIB_DESIGN.md §4.2, so it is frozen here.)

    .venv/bin/python scripts/run_calib.py
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
# ANCHOR_MAX_AGE 4.0 s (vs 2.0 in run_vio): calibration benefits from longer spans — the δφ_HC information
# grows with anchor age and the refit's long-span tail is gated, so the extra span buys observability here.
PERIOD, KF_EVERY, ANCHOR_MAX_AGE, ANCHOR_MIN_SHARED = 24.0, 15, 4.0, 30
DPHI_TRUE = np.radians([1.5, -2.5, 1.0])                  # ~3.1° camera-frame mount error to recover
TD_TRUE = 0.0                                            # (time offset off — weakly observable on a yaw-loop)


def _quat_from_rotvec(v):
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, v / ang, ang); return q


def _quat_mul(a, b):
    q = np.zeros(4); mujoco.mju_mulQuat(q, a, b); return q


def frontend(frames, fidx, ft_t):
    cam = CameraModel()
    tracker, anchor = FeatureTracker(fb_thresh=1.5), AnchorRotationEstimator(cam)
    keyframes, anchor_set = [], False
    for k in range(len(frames)):
        kf = (k % KF_EVERY == 0)
        tracks = tracker.process(cv2.cvtColor(frames[k], cv2.COLOR_RGB2GRAY), keyframe=kf)
        if not kf:
            continue
        i, tt = int(fidx[k]), float(ft_t[k])
        if not anchor_set:
            anchor.set_anchor(tt, tracks); anchor_set = True
            keyframes.append((i, None, True)); continue
        m = anchor.estimate(tt, tracks)
        n_shared = sum(1 for x in tracks if x in anchor.anchor_uv)
        reanchor = (m is None) or (tt - anchor.anchor_t > ANCHOR_MAX_AGE) or (n_shared < ANCHOR_MIN_SHARED)
        keyframes.append((i, m["R"] if m is not None else None, reanchor))
        if reanchor:
            anchor.set_anchor(tt, tracks)
    return keyframes


def main(seed=0):
    q_true = _quat_mul(np.array(HEAD_CAM_QUAT, float), _quat_from_rotvec(_GL2CV @ DPHI_TRUE))
    _, _, log, frames = simulate(seconds=PERIOD, heading=loop_heading(PERIOD), environment=True,
                                 render_camera="head_cam", fps=30, imu_noise=ImuNoise(), seed=seed,
                                 head_cam_quat=q_true)                 # rendered mount = perturbed truth
    frames = np.array(frames)
    t, gyro, accel = log["t"], log["gyro"], log["accel"]
    enc_q, enc_qd, contact = log["enc_q"], log["enc_qd"], log["contact"]
    jn = list(log["joint_names"]); dt = float(t[1] - t[0])
    ft_shift = log["frame_t"] - TD_TRUE                               # relabel stamps ⇒ content at stamp+td
    fidx = np.array([int(np.argmin(np.abs(t - tt))) for tt in ft_shift])

    kin = RobotKinematics(); ext = HeadCamExtrinsics(kin)
    R_HC_nom = ext.R_HC.copy()
    R_HC_true = R_HC_nom @ exp_so3(DPHI_TRUE)                          # truth to recover (never fed in)
    kf_by_idx = {i: (Rm, ca) for (i, Rm, ca) in frontend(frames, fidx, ft_shift)}

    R0, p0 = quat2mat(log["gt_quat"][0]), log["gt_pos"][0]
    vio = CalibVioESKF(kin, p0, R0, R_HC0=R_HC_nom, td0=0.0, est_extrinsic=True, est_td=False)
    tl, ephi, sphi, etd, std = [], [], [], [], []
    for k in range(len(t)):
        vio.step(gyro[k], accel[k], enc_q[k], enc_qd[k], jn, contact[k], dt)
        if k in kf_by_idx:
            R_meas, clone_after = kf_by_idx[k]
            R_BH_j = ext.head_pose(enc_q[k], jn)[0]
            w_neck = ext.head_angvel_base(enc_q[k], enc_qd[k], jn)
            if R_meas is not None:
                vio.update_rel_rot(R_meas, R_BH_j, gyro[k], w_neck)
            if clone_after:
                vio.clone(R_BH_j, gyro[k], w_neck)
        tl.append(t[k])
        ephi.append(log_so3(vio.R_HC.T @ R_HC_true))
        sphi.append(np.sqrt(np.maximum(np.diag(vio.P[18:21, 18:21]), 0.0)))
        etd.append(vio.td - TD_TRUE); std.append(np.sqrt(max(vio.P[21, 21], 0.0)))
    ephi, sphi = np.array(ephi), np.array(sphi)
    etd, std = np.array(etd), np.array(std)

    print(f"MOUNT  injected {np.degrees(np.linalg.norm(DPHI_TRUE)):.2f} deg (axes {np.degrees(DPHI_TRUE).round(2)} deg)")
    print(f"       remaining per-axis (mrad): {(ephi[-1]*1e3).round(1)}")
    print(f"       φ_x (⊥ the yaw-loop, observable): {DPHI_TRUE[0]*1e3:+.0f} -> {ephi[-1,0]*1e3:+.1f} mrad "
          f"({100*(1-abs(ephi[-1,0])/abs(DPHI_TRUE[0])):.0f}% recovered), funnel σ {sphi[-1,0]*1e3:.1f} mrad")
    print("       φ_y (≈ the turn axis n_C) + φ_z (partly along it): weakly observable — funnels stay wide,")
    print(f"       σ φ_y {sphi[-1,1]*1e3:.0f} mrad (the filter correctly reports what a yaw-loop cannot pin).")

    fig, ax = plt.subplots(figsize=(9, 5))
    for a, (nm, c) in enumerate((("φ_x  (⊥ turn — observable)", "#e05050"),
                                 ("φ_y  (≈ turn axis n_C — weak)", "#3fd0c9"),
                                 ("φ_z", "#e0a030"))):
        ax.plot(tl, ephi[:, a] * 1e3, color=c, lw=1.5, label=nm)
        ax.fill_between(tl, -3 * sphi[:, a] * 1e3, 3 * sphi[:, a] * 1e3, color=c, alpha=0.12)
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_title("Stage 3 — camera-mount extrinsic self-calibration while walking (error → 0, ±3σ)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("remaining mount error (mrad)"); ax.legend(); ax.grid(alpha=.3)
    os.makedirs(RENDERS, exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(RENDERS, "stage3_calib.png"), dpi=110)
    print(f"saved {os.path.join(RENDERS, 'stage3_calib.png')}")


if __name__ == "__main__":
    main()
