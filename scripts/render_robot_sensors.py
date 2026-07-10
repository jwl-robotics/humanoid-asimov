"""Render the actual Asimov robot with its sensors annotated (for the Stage-0 dashboard page).

Outputs:
  renders/robot_sensors.png        — labelled iso view (IMU / head-cam / feet / encoders located on the real mesh)
  renders/turntable/frame_XX.png   — 24-frame 360° turntable for a drag-to-rotate viewer

    .venv/bin/python scripts/render_robot_sensors.py
"""
import os

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from humanoid_asimov.scene import build_spec

RENDERS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))
W = H = 680
FOVY = 42.0
TARGET = np.array([0.0, 0.0, 0.55])


def look_at(cam_pos, target, up=(0.0, 0.0, 1.0)):
    cam_pos = np.asarray(cam_pos, float)
    z = cam_pos - np.asarray(target, float); z /= np.linalg.norm(z)      # cam +z points away (GL looks down −z)
    x = np.cross(np.asarray(up, float), z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten())
    return R, q


def project(P, cam_pos, R, fovy, w, h):
    p = R.T @ (np.asarray(P, float) - cam_pos)
    if p[2] >= -1e-6:
        return None
    f = (h / 2) / np.tan(np.radians(fovy) / 2)
    return (w / 2 + f * p[0] / (-p[2]), h / 2 - f * p[1] / (-p[2]))


def main():
    spec = build_spec(add_head_camera=True, add_environment=False)
    cam_pos = np.array([1.65, -1.75, 1.15])
    R, q = look_at(cam_pos, TARGET)
    cam = spec.worldbody.add_camera()
    cam.name = "iso_annot"; cam.pos = cam_pos; cam.quat = q; cam.fovy = FOVY
    model = spec.compile()
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        data.qpos[2] = 0.60
    mujoco.mj_forward(model, data)

    ren = mujoco.Renderer(model, H, W)
    ren.update_scene(data, "iso_annot")
    img = ren.render().copy()

    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
    cid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n)
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    knee = next((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in range(model.njnt)
                 if "knee" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")), None)

    # (label, world-point, colour, label-anchor corner)
    marks = [
        ("HEAD CAMERA\n(we added this)", data.cam_xpos[cid("head_cam")].copy(), "#3fd0c9", (0.82, 0.13)),
        ("IMU — gyro + accel\n(pelvis · original)", data.site_xpos[sid("imu_in_pelvis")].copy(), "#e05050", (0.02, 0.40)),
        ("JOINT ENCODERS\n(every joint · original)", data.xanchor[jid(knee)].copy() if knee else TARGET, "#6ea8fe", (0.82, 0.58)),
        ("FOOT CONTACT\n(derived, not a sensor)", data.site_xpos[sid("left_foot")].copy(), "#d9a23f", (0.02, 0.84)),
        ("", data.site_xpos[sid("right_foot")].copy(), "#d9a23f", None),
    ]

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(img); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    for label, P, c, corner in marks:
        uv = project(P, cam_pos, R, FOVY, W, H)
        if uv is None:
            continue
        u, v = uv
        ax.plot(u, v, "o", ms=13, mfc="none", mec=c, mew=2.6, zorder=6)
        if corner:
            lx, ly = corner[0] * W, corner[1] * H
            ax.annotate(label, xy=(u, v), xytext=(lx, ly), color=c, fontsize=11.5, fontweight="bold",
                        ha="left" if corner[0] < 0.5 else "right", va="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.35", fc="#0a0e12", ec=c, lw=1.2, alpha=0.92),
                        arrowprops=dict(arrowstyle="-", color=c, lw=1.5, alpha=0.9,
                                        connectionstyle="arc3,rad=0.12"))
    ax.set_title("Asimov — onboard sensors (red/blue = original · cyan = added · amber = derived)",
                 color="#d7e2ee", fontsize=12)
    fig.patch.set_facecolor("#0a0e12")
    os.makedirs(RENDERS, exist_ok=True)
    fig.savefig(os.path.join(RENDERS, "robot_sensors.png"), dpi=100, bbox_inches="tight", facecolor="#0a0e12")
    print("saved renders/robot_sensors.png")

    # --- turntable: orbit a camera we control (so we can project the sensors onto every frame) ---
    import cv2
    tdir = os.path.join(RENDERS, "turntable"); os.makedirs(tdir, exist_ok=True)
    for old in os.listdir(tdir):
        os.remove(os.path.join(tdir, old))
    TW = 470
    tren = mujoco.Renderer(model, TW, TW)
    camid = cid("iso_annot")
    RED, CYAN, AMBER, BLUE = (80, 80, 224), (201, 208, 63), (63, 162, 217), (254, 168, 110)   # BGR
    key = [(data.cam_xpos[cid("head_cam")].copy(), CYAN, 9),
           (data.site_xpos[sid("imu_in_pelvis")].copy(), RED, 9),
           (data.site_xpos[sid("left_foot")].copy(), AMBER, 8),
           (data.site_xpos[sid("right_foot")].copy(), AMBER, 8)]
    enc = [data.xanchor[j].copy() for j in range(model.njnt)
           if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]          # every joint encoder
    dist, el, N = 2.7, np.radians(15.0), 20
    for k in range(N):
        az = 2 * np.pi * k / N
        cpos = TARGET + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        Rk, qk = look_at(cpos, TARGET)
        model.cam_pos[camid] = cpos; model.cam_quat[camid] = qk
        mujoco.mj_forward(model, data)
        tren.update_scene(data, "iso_annot")
        f = cv2.cvtColor(tren.render(), cv2.COLOR_RGB2BGR)
        for P in enc:
            uv = project(P, cpos, Rk, FOVY, TW, TW)
            if uv:
                cv2.circle(f, (int(uv[0]), int(uv[1])), 4, BLUE, 2, cv2.LINE_AA)
        for P, c, r in key:
            uv = project(P, cpos, Rk, FOVY, TW, TW)
            if uv:
                cv2.circle(f, (int(uv[0]), int(uv[1])), r, c, 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(tdir, f"frame_{k:02d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"saved {N} annotated turntable frames → renders/turntable/")


if __name__ == "__main__":
    main()
