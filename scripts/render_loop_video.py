"""Render a side-by-side video of the loop walk: external view | head-cam (VIO) with live KLT tracks.

    .venv/bin/python scripts/render_loop_video.py
Saves renders/stage2_loop_video.mp4 + a preview frame renders/stage2_loop_preview.png.
"""
import os

import cv2
import mujoco
import numpy as np

from humanoid_asimov.scene import build_model
from humanoid_asimov.vision import FeatureTracker
from humanoid_asimov.walk import WalkController, loop_heading

RD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))
PERIOD, FPS, VW, VH = 24.0, 15, 420, 320


def main():
    model = build_model(add_head_camera=True, add_environment=True)
    data = mujoco.MjData(model)
    ctrl = WalkController(model, heading=loop_heading(PERIOD))
    ctrl.reset_to_gait(model, data, 0.0)
    dt = model.opt.timestep
    for _ in range(int(1.5 / dt)):                       # settle (unshown)
        ctrl.apply(model, data)
        mujoco.mj_step(model, data)

    rend = mujoco.Renderer(model, VH, VW)
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    excam = mujoco.MjvCamera()
    excam.azimuth, excam.elevation, excam.distance = 110.0, -28.0, 13.0
    excam.lookat[:] = [-0.4, 3.3, 0.5]                   # loop centre
    tracker = FeatureTracker(VW, VH, budget=120, fb_thresh=1.5)
    trails: dict = {}

    os.makedirs(RD, exist_ok=True)
    out = os.path.join(RD, "stage2_loop_video.mp4")
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (VW * 2, VH))
    steps = int(round((1.0 / FPS) / dt))
    n = int(PERIOD * FPS)
    for f in range(n):
        for _ in range(steps):
            ctrl.apply(model, data)
            mujoco.mj_step(model, data)
        rend.update_scene(data, excam); ext = rend.render().copy()
        rend.update_scene(data, head_id); head = rend.render().copy()

        tracks = tracker.process(cv2.cvtColor(head, cv2.COLOR_RGB2GRAY), keyframe=(f % 8 == 0))
        for i, uv in tracks.items():
            trails.setdefault(i, []).append(uv)
            trails[i] = trails[i][-8:]
        for i, uv in tracks.items():
            pts = np.array(trails[i], int)
            for j in range(1, len(pts)):
                cv2.line(head, tuple(pts[j - 1]), tuple(pts[j]), (0, 255, 0), 1)
            cv2.circle(head, (int(uv[0]), int(uv[1])), 2, (0, 255, 0), -1)
        trails = {i: trails[i] for i in tracks}

        cv2.putText(ext, "external", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(head, "head cam (VIO) + tracks", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        combined = np.hstack([ext, head])
        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        if f == n // 3:
            cv2.imwrite(os.path.join(RD, "stage2_loop_preview.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"saved {out}  ({n} frames, {PERIOD:.0f}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
