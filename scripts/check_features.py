"""Stage 2.0 verification — does the feature room give the head camera trackable structure?

Renders the monocular head camera from several base poses in the feature room and runs the
Shi-Tomasi corner detector (``cv2.goodFeaturesToTrack``) — the same front-end the VIO tracker will
use. Saves an annotated 2x2 montage and prints per-view feature counts. The bare checker scene gives
features that are *repetitive* (ambiguous to match); the feature room should give a few hundred
well-spread corners on distinct 3-D structure (walls, panels, landmarks) that are unique to match.

    .venv/bin/python scripts/check_features.py
"""
import os

import cv2
import mujoco
import numpy as np

from humanoid_asimov.scene import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
RENDERS = os.path.abspath(os.path.join(HERE, "..", "renders"))
W, H = 640, 480


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _feature_count(model, data, renderer, cam, qstand, x, y, yaw):
    data.qpos[:] = qstand
    data.qpos[0:2] = [x, y]
    data.qpos[3:7] = _yaw_quat(yaw)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.01, minDistance=6)
    return rgb, (corners.reshape(-1, 2) if corners is not None else np.empty((0, 2)))


def _standing_qpos(model, data):
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    return data.qpos.copy()


def main():
    poses = [(0.0, 0.0, 0.0), (2.5, 0.5, 0.6), (0.0, 0.0, -1.4), (-2.0, 1.5, 2.4)]

    # feature room: annotated montage + counts
    model = build_model(add_head_camera=True, add_environment=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    qstand = _standing_qpos(model, data)

    tiles, counts = [], []
    for (x, y, yaw) in poses:
        rgb, corners = _feature_count(model, data, renderer, cam, qstand, x, y, yaw)
        counts.append(len(corners))
        vis = rgb.copy()
        for c in corners.astype(int):
            cv2.circle(vis, (int(c[0]), int(c[1])), 3, (0, 255, 0), -1)
        cv2.putText(vis, f"{len(corners)} features", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        tiles.append(vis)

    # bare scene, same poses, for the contrast line
    bare_m = build_model(add_head_camera=True, add_environment=False)
    bare_d = mujoco.MjData(bare_m)
    bare_r = mujoco.Renderer(bare_m, H, W)
    bare_cam = mujoco.mj_name2id(bare_m, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    bare_q = _standing_qpos(bare_m, bare_d)
    bare_counts = [len(_feature_count(bare_m, bare_d, bare_r, bare_cam, bare_q, x, y, yaw)[1])
                   for (x, y, yaw) in poses]

    montage = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4])])
    os.makedirs(RENDERS, exist_ok=True)
    out = os.path.join(RENDERS, "stage2_features.png")
    cv2.imwrite(out, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

    print(f"feature room : counts {counts}  mean {np.mean(counts):.0f}")
    print(f"bare scene   : counts {bare_counts}  mean {np.mean(bare_counts):.0f}"
          f"   (checker corners — repetitive, ambiguous to match)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
