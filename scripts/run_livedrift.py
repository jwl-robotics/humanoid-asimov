"""Continuous-lap live-drift monitor — does VIO keep the drift bounded over many laps?

Runs the closed loop for several laps and tracks the horizontal position error vs time for leg-odometry,
the Stage-1 ESKF, and VIO. Over repeated laps the yaw drift of the IMU-only estimators compounds (position
diverges), while VIO — anchoring the yaw-drift rate + the gyro-z bias every lap — grows far slower. The
payoff view: watch the gap open up.

    .venv/bin/python scripts/run_livedrift.py
"""
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from humanoid_asimov.estimator import ESKF, LegOdometry, RobotKinematics, exp_so3, quat2mat
from humanoid_asimov.sensors import ImuNoise
from humanoid_asimov.vio import VioESKF
from humanoid_asimov.vision import AnchorRotationEstimator, CameraModel, FeatureTracker, HeadCamExtrinsics
from humanoid_asimov.walk import loop_heading, simulate

RENDERS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))
# ANCHOR_MAX_AGE 4.0 s (vs 2.0 in run_vio): the multi-lap monitor favours longer-lived anchors so the
# yaw-drift information accumulates across laps; the refit's long-span tail is gated as usual.
PERIOD, KF_EVERY, ANCHOR_MAX_AGE, ANCHOR_MIN_SHARED = 24.0, 15, 4.0, 30
LAPS = 3


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
    # a realistic uncalibrated IMU bias (esp. the z/yaw axis, ~0.23°/s) — the estimators must learn it online;
    # leg-odometry (no bias estimation) drifts on it, and only vision pins the yaw-axis component.
    imu = ImuNoise(gyro_bias=np.array([0.0015, 0.0015, 0.004]))
    _, _, log, frames = simulate(seconds=LAPS * PERIOD, heading=loop_heading(PERIOD), environment=True,
                                 render_camera="head_cam", fps=30, imu_noise=imu, seed=seed)
    frames = np.array(frames)
    t, gyro, accel = log["t"], log["gyro"], log["accel"]
    enc_q, enc_qd, contact = log["enc_q"], log["enc_qd"], log["contact"]
    gt_pos, gt_quat = log["gt_pos"], log["gt_quat"]
    jn = list(log["joint_names"]); dt = float(t[1] - t[0])
    fidx = np.array([int(np.argmin(np.abs(t - tt))) for tt in log["frame_t"]])

    kin = RobotKinematics(); ext = HeadCamExtrinsics(kin)
    kf_by_idx = {i: (Rm, ca) for (i, Rm, ca) in frontend(frames, fidx, log["frame_t"])}

    def _yaw(R):
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    gt_yaw = np.array([_yaw(quat2mat(q)) for q in gt_quat])
    R0, p0 = quat2mat(gt_quat[0]), gt_pos[0]
    eskf = ESKF(kin, p0, R0, sig_contact=0.15)
    vio = VioESKF(kin, p0, R0, sig_contact=0.15)
    lo, R_gyro = LegOdometry(kin, p0), R0.copy()
    yerr = {"leg-odom": [], "ESKF": [], "VIO": []}         # yaw error (deg) — the monotonic drift signal
    perr = {"leg-odom": [], "ESKF": [], "VIO": []}         # position error (cm) — oscillates over a closed loop
    for k in range(len(t)):
        eskf.step(gyro[k], accel[k], enc_q[k], enc_qd[k], jn, contact[k], dt)
        vio.step(gyro[k], accel[k], enc_q[k], enc_qd[k], jn, contact[k], dt)
        if k in kf_by_idx:
            R_meas, clone_after = kf_by_idx[k]
            R_BC_j = ext.compute(enc_q[k], jn)[0]
            if R_meas is not None:
                vio.update_rel_rot(R_meas, R_BC_j)
            if clone_after:
                vio.clone(R_BC_j)
        R_gyro = R_gyro @ exp_so3(gyro[k] * dt)
        lo.step(R_gyro, enc_q[k], jn, contact[k])
        g = gt_pos[k, :2]
        for name, R_est, p_est in (("leg-odom", R_gyro, lo.p), ("ESKF", eskf.R, eskf.p), ("VIO", vio.R, vio.p)):
            yerr[name].append(abs(np.degrees(_wrap(_yaw(R_est) - gt_yaw[k]))))
            perr[name].append(float(np.linalg.norm(p_est[:2] - g)))
    yerr = {k: np.array(v) for k, v in yerr.items()}
    perr = {k: np.array(v) for k, v in perr.items()}

    lap_idx = [min(len(t) - 1, int(round((lap + 1) * PERIOD / dt)) - 1) for lap in range(LAPS)]
    print(f"=== continuous {LAPS}-lap drift (seed {seed}) — |yaw error| (deg) at each lap end ===")
    print(f"{'estimator':10} " + " ".join(f"lap{i+1:>2}" for i in range(LAPS)))
    for name in ("leg-odom", "ESKF", "VIO"):
        print(f"{name:10} " + " ".join(f"{yerr[name][i]:5.1f}" for i in lap_idx))
    print(f"=> yaw error after {LAPS} laps: VIO {yerr['VIO'][-1]:.1f}deg stays bounded vs "
          f"ESKF {yerr['ESKF'][-1]:.1f}deg / leg-odom {yerr['leg-odom'][-1]:.1f}deg (unbounded gyro-drift).")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, c in (("leg-odom", "#e0a030"), ("ESKF", "#e05050"), ("VIO", "#3fd0c9")):
        a1.plot(t, yerr[name], color=c, lw=1.5, label=name)
        a2.plot(t, perr[name] * 100, color=c, lw=1.5, label=name)
    for a in (a1, a2):
        for lap in range(1, LAPS):
            a.axvline(lap * PERIOD, color="k", ls=":", alpha=0.3)
        a.set_xlabel("t (s)"); a.legend(); a.grid(alpha=.3)
    a1.set_ylabel("|yaw error| (deg)"); a1.set_title("yaw error grows for IMU-only, bounded for VIO")
    a2.set_ylabel("position error (cm)"); a2.set_title("position error (oscillates over a closed loop)")
    fig.suptitle(f"Continuous {LAPS}-lap live drift — vision anchors the heading the IMU cannot observe", fontsize=13)
    os.makedirs(RENDERS, exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(RENDERS, "livedrift.png"), dpi=110)
    print(f"saved {os.path.join(RENDERS, 'livedrift.png')}")


if __name__ == "__main__":
    main()
