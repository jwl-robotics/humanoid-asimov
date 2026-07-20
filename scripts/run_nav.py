"""Stage 5 — closed-loop navigation: the robot walks to a sequence of waypoints using ONLY its own
onboard state estimate (ESKF). Full loop each control step: sensors → estimator → waypoint planner →
VelocityCommand → gait steering. Estimation error surfaces directly as navigation error.

    .venv/bin/python scripts/run_nav.py
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from humanoid_asimov.estimator import ESKF, RobotKinematics, quat2mat
from humanoid_asimov.nav import WaypointNavigator
from humanoid_asimov.scene import build_model
from humanoid_asimov.sensors import ImuNoise, Sensors
from humanoid_asimov.walk import simulate

RENDERS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))
WAYPOINTS = [(3.0, 0.0), (3.0, 2.5), (0.3, 2.5), (0.3, -0.2)]
SECONDS = 34.0


def main(seed=0):
    kin = RobotKinematics()
    jn = list(Sensors(build_model(add_environment=True)).joint_names)
    nav = WaypointNavigator(WAYPOINTS)
    st = {"e": None}
    est, cmd = [], []

    def hook(t, gyro, accel, enc_q, enc_qd, contact, dt, ctrl, gt_pos, gt_quat, frame=None):
        e = st["e"]
        if e is None:                                          # init once at the known start pose
            # ESKF default sig_contact=0.05: verified best for this SHORT path (7.1 vs 9.6 cm at 0.15). The
            # b_g-runaway-on-turns that motivates a looser 0.15 on the long continuous loop (Stage-2 finding)
            # needs sustained turning to accumulate; it does not bite over ~14 s of intermittent turns here.
            e = ESKF(kin, gt_pos.copy(), quat2mat(gt_quat)); st["e"] = e
        e.step(gyro, accel, enc_q, enc_qd, jn, contact, dt)
        x, y = float(e.p[0]), float(e.p[1])
        yaw = float(np.arctan2(e.R[1, 0], e.R[0, 0]))
        c = nav.command(x, y, yaw)
        ctrl.heading = (lambda tt, h=nav.desired_heading: h)   # steer the gait toward the goal heading
        est.append((t, x, y, yaw)); cmd.append((c.vx, c.vyaw, nav.i))

    _, _, log, _ = simulate(seconds=SECONDS, environment=True, imu_noise=ImuNoise(), seed=seed,
                            heading=lambda t: 0.0, control_hook=hook)
    est = np.array(est); cmd = np.array(cmd); gt = log["gt_pos"]
    # trim at the step the final waypoint is reached (the gait has no "stop", so ignore the walk-off tail)
    done = cmd[:, 2] >= len(WAYPOINTS)
    end = int(np.argmax(done)) if done.any() else len(est) - 1
    est, gt = est[:end + 1], gt[:end + 1]

    reached = int(cmd[:end + 1, 2].max())
    drift = float(np.linalg.norm(est[-1, 1:3] - gt[-1, :2]))
    misses = [float(np.min(np.linalg.norm(gt[:, :2] - np.array(w), axis=1))) for w in WAYPOINTS]
    path = float(np.sum(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1)))
    print(f"waypoints reached: {reached}/{len(WAYPOINTS)} in {est[-1, 0]:.1f} s, {path:.1f} m path")
    print(f"true closest-approach to each waypoint (cm): {[round(m * 100) for m in misses]}")
    print(f"estimator-vs-truth drift at completion: {drift * 100:.1f} cm ({drift / path * 100:.2f}% of path)")
    print("  -> the nav error the ESKF yaw-drift injects; VIO (which fixes yaw) would cut it.")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot(est[:, 1], est[:, 2], color="#3fd0c9", lw=2.0, label="estimated path (what nav sees)")
    ax.plot(gt[:, 0], gt[:, 1], color="#e0a030", lw=1.5, ls="--", label="true path (where it really went)")
    wp = np.array(WAYPOINTS)
    ax.plot(wp[:, 0], wp[:, 1], "s", color="#e05050", ms=13, label="waypoints", zorder=5)
    for k, (gx, gy) in enumerate(WAYPOINTS):
        ax.annotate(str(k + 1), (gx, gy), color="w", ha="center", va="center", fontsize=9, zorder=6)
    ax.plot(est[0, 1], est[0, 2], "o", color="w", ms=9, label="start")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="upper right")
    ax.set_title("Stage 5 — closed-loop waypoint navigation on the onboard estimate")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    os.makedirs(RENDERS, exist_ok=True); fig.tight_layout()
    fig.savefig(os.path.join(RENDERS, "stage5_nav.png"), dpi=110)
    print(f"saved {os.path.join(RENDERS, 'stage5_nav.png')}")


if __name__ == "__main__":
    main()
