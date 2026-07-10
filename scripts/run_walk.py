"""Drive Asimov with the PD-tracked walking gait; save an animation + an estimator dataset.

    .venv/bin/python scripts/run_walk.py

Writes renders/walk.gif + renders/walk_strip.png, persists data/walk_dataset.npz (the logged
onboard sensors + ground truth for the estimator), and prints a summary. This is the development
locomotion black box — the estimator sees only the onboard sensors, never the gait targets, the
support wrench, or the ground truth.
"""
import os

import numpy as np
from PIL import Image

from humanoid_asimov.sensors import ImuNoise
from humanoid_asimov.walk import save_dataset, simulate

RENDERS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def main():
    model, data, log, frames = simulate(
        seconds=8.0, render_camera="side_camera", fps=25, imu_noise=ImuNoise(), seed=0)

    base = log["gt_pos"]
    travel = base[-1, :2] - base[0, :2]
    print(f"steps={len(log['t'])}  duration={log['t'][-1]:.1f}s  frames={len(frames)}")
    print(f"base height: mean={base[:,2].mean():.3f} m  min={base[:,2].min():.3f} m (stayed upright)")
    print(f"travel: dx={travel[0]:+.2f}  dy={travel[1]:+.2f} m")
    print(f"IMU (noisy): |gyro| mean={np.linalg.norm(log['gyro'], axis=1).mean():.2f} rad/s  "
          f"|accel| mean={np.linalg.norm(log['accel'], axis=1).mean():.2f} m/s^2")
    print(f"contact duty: L={log['contact'][:,0].mean():.0%}  R={log['contact'][:,1].mean():.0%}")

    os.makedirs(RENDERS, exist_ok=True)
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    Image.fromarray(np.concatenate([frames[i] for i in idx], axis=1)).save(
        os.path.join(RENDERS, "walk_strip.png"))
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(os.path.join(RENDERS, "walk.gif"), save_all=True,
                 append_images=imgs[1:], duration=40, loop=0)

    os.makedirs(DATA, exist_ok=True)
    path = save_dataset(log, os.path.join(DATA, "walk_dataset.npz"))
    print(f"saved {RENDERS}/walk.gif + walk_strip.png  |  dataset -> {path}")


if __name__ == "__main__":
    main()
