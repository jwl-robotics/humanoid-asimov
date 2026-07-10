"""Stage 1: run the estimator on the logged dataset and score it against ground truth.

    .venv/bin/python scripts/run_estimator.py

Compares three estimates of the base ground track:
  - leg-odom · GT attitude   (the translational drift floor, given perfect attitude)
  - leg-odom · gyro attitude  (realistic dead reckoning, no filter)
  - ESKF (fused)              (IMU predict + contact-velocity update + online bias estimation)
Plots the tracks, the horizontal error vs time, and the ESKF's estimated gyro bias.
"""
import os

import numpy as np

from humanoid_asimov.estimator import ESKF, LegOdometry, RobotKinematics, exp_so3, quat2mat

DATASET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "walk_dataset.npz"))
RENDERS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))


def metrics(est, gt):
    err = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)       # horizontal
    err3 = np.linalg.norm(est - gt, axis=1)                     # full 3D
    path = float(np.sum(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1)))
    return dict(ate=float(np.sqrt((err ** 2).mean())), final=float(err[-1]),
                drift=float(err[-1] / max(path, 1e-6) * 100.0), path=path, err=err,
                ate3d=float(np.sqrt((err3 ** 2).mean())), zerr=float(abs(est[-1, 2] - gt[-1, 2])))


def main():
    d = np.load(DATASET, allow_pickle=True)
    t, gyro, accel = d["t"], d["gyro"], d["accel"]
    enc_q, enc_qd, contact = d["enc_q"], d["enc_qd"], d["contact"]
    gt_pos, gt_quat = d["gt_pos"], d["gt_quat"]
    joint_names = list(d["joint_names"])
    dt = float(t[1] - t[0])
    kin = RobotKinematics()
    R0 = quat2mat(gt_quat[0])

    # --- leg-odometry baselines ---
    R_gt = [quat2mat(q) for q in gt_quat]
    R = R0.copy()
    R_gyro = []
    for k in range(len(t)):
        R = R @ exp_so3(gyro[k] * dt)
        R_gyro.append(R.copy())
    est = {}
    for label, R_seq in (("leg-odom · GT att", R_gt), ("leg-odom · gyro att", R_gyro)):
        lo = LegOdometry(kin, gt_pos[0])
        est[label] = np.array([lo.step(R_seq[k], enc_q[k], joint_names, contact[k]) for k in range(len(t))])

    # --- ESKF (fused) ---
    eskf = ESKF(kin, gt_pos[0], R0)
    traj, bias = [], []
    for k in range(len(t)):
        eskf.step(gyro[k], accel[k], enc_q[k], enc_qd[k], joint_names, contact[k], dt)
        traj.append(eskf.p.copy())
        bias.append(eskf.bg.copy())
    est["ESKF (fused)"] = np.array(traj)
    bias = np.array(bias)

    for label in est:
        m = metrics(est[label], gt_pos)
        print(f"{label:22}: ATE {m['ate']*100:5.1f} cm | final {m['final']*100:5.1f} cm | drift {m['drift']:5.2f}% | 3D-ATE {m['ate3d']*100:5.1f} cm | z-err {m['zerr']*100:4.1f} cm")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"leg-odom · GT att": "#6ad0ff", "leg-odom · gyro att": "#e0a030", "ESKF (fused)": "#3fd0c9"}
        fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
        ax[0].plot(gt_pos[:, 0], gt_pos[:, 1], color="w", lw=2.4, label="ground truth")
        for label, e in est.items():
            ax[0].plot(e[:, 0], e[:, 1], color=colors[label], lw=1.5, label=label)
        ax[0].set_title("ground track (x-y)"); ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
        ax[0].axis("equal"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
        for label in ("leg-odom · gyro att", "ESKF (fused)"):
            ax[1].plot(t, metrics(est[label], gt_pos)["err"] * 100, color=colors[label], label=label)
        ax[1].set_title("horizontal error vs time"); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("error [cm]")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
        for i, axis in enumerate("xyz"):
            ax[2].plot(t, bias[:, i] * 1e3, label=f"b_g {axis}")
        ax[2].set_title("ESKF estimated gyro bias"); ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("bias [mrad/s]")
        ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25)
        for a in ax:
            a.set_facecolor("#0a0e12")
        fig.patch.set_facecolor("#0a0e12")
        os.makedirs(RENDERS, exist_ok=True)
        out = os.path.join(RENDERS, "stage1_eskf.png")
        fig.tight_layout(); fig.savefig(out, dpi=110, facecolor="#0a0e12")
        print(f"saved {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
