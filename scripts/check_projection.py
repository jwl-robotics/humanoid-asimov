"""Stage 2.1 convention canary — verify camera intrinsics + head-cam extrinsics + frame conventions.

Renders a head-cam frame in the feature room, then projects the environment landmark centers through our
own CameraModel + FK HeadCamExtrinsics (+ the GT base pose, which is legal in a *verification* — it scores,
never feeds the estimator). If K, the GL→CV flip, the quaternion order, and R-vs-Rᵀ are all correct, the red
rings land on the landmark centers. This one image catches the four classic VIO convention bugs at once.

    .venv/bin/python scripts/check_projection.py
"""
import os

import cv2
import mujoco
import numpy as np

from humanoid_asimov.estimator import quat2mat
from humanoid_asimov.scene import build_model
from humanoid_asimov.sensors import Sensors
from humanoid_asimov.vision import CameraModel, HeadCamExtrinsics

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders", "stage2_projection.png"))
W, H = 640, 480


def main():
    model = build_model(add_head_camera=True, add_environment=True)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    yaw = 0.7                                            # face into a corner with several landmarks
    data.qpos[0:2] = [0.0, 0.0]
    data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    renderer.update_scene(data, cam)
    rgb = renderer.render().copy()

    # our own projection chain: encoders → FK extrinsics, composed with the (eval-only) GT base pose
    sensors = Sensors(model)
    enc_q, _ = sensors.encoders(data)
    ext = HeadCamExtrinsics()
    R_BC, p_BC = ext.compute(enc_q, list(sensors.joint_names))
    R_WB = quat2mat(data.qpos[3:7])
    p_WB = data.qpos[0:3].copy()
    R_WC = R_WB @ R_BC
    p_WC = p_WB + R_WB @ p_BC
    camm = CameraModel(W, H, 70.0)

    drawn = 0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != 0 or model.geom_contype[g] != 0:
            continue
        if model.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
            continue
        p_cam = R_WC.T @ (data.geom_xpos[g] - p_WC)
        if p_cam[2] < 0.2:                               # behind camera or too close
            continue
        u, v = camm.project(p_cam)[0]
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(rgb, (int(u), int(v)), 9, (255, 0, 0), 2)
            drawn += 1
    cv2.putText(rgb, f"{drawn} landmark centers projected (rings should sit on landmarks)", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    cv2.imwrite(OUT, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"projected {drawn} landmark centers -> {OUT}")


if __name__ == "__main__":
    main()
