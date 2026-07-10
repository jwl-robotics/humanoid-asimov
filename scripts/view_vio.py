"""Live VIO front-end viewer — watch the head camera track features as the robot walks the loop.

    .venv/bin/python scripts/view_vio.py

Opens an OpenCV window: LEFT = external view of the robot looping the feature room; RIGHT = the head
camera with the **live KLT feature tracks** (green dots + short trails) the VIO front-end is following.
The robot keeps looping. Press **q** or **Esc** to quit. (Plain python — this is a cv2 window, not the
MuJoCo viewer.)
"""
import cv2
import mujoco
import numpy as np

from humanoid_asimov.scene import build_model
from humanoid_asimov.vision import FeatureTracker
from humanoid_asimov.walk import WalkController, loop_heading

PERIOD, FPS, VW, VH = 24.0, 20, 480, 360
WIN = "VIO  —  external  |  head-cam + live feature tracks"


def main():
    model = build_model(add_head_camera=True, add_environment=True)
    data = mujoco.MjData(model)
    ctrl = WalkController(model, heading=loop_heading(PERIOD))
    ctrl.reset_to_gait(model, data, 0.0)
    dt = model.opt.timestep
    for _ in range(int(1.5 / dt)):                           # settle (unshown)
        ctrl.apply(model, data); mujoco.mj_step(model, data)

    rend = mujoco.Renderer(model, VH, VW)
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    excam = mujoco.MjvCamera()
    excam.azimuth, excam.elevation, excam.distance = 110.0, -28.0, 13.0
    excam.lookat[:] = [-0.4, 3.3, 0.5]
    tracker = FeatureTracker(VW, VH, budget=120, fb_thresh=1.5)
    trails: dict = {}
    steps = int(round((1.0 / FPS) / dt))

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, VW * 2, VH)
    print("Live VIO front-end. Left = external, right = head-cam with live tracks. Press q / Esc to quit.")
    f = 0
    while True:
        for _ in range(steps):
            ctrl.apply(model, data); mujoco.mj_step(model, data)
        rend.update_scene(data, excam); ext = rend.render().copy()
        rend.update_scene(data, head_id); head = rend.render().copy()

        tracks = tracker.process(cv2.cvtColor(head, cv2.COLOR_RGB2GRAY), keyframe=(f % 8 == 0))
        for i, uv in tracks.items():
            trails.setdefault(i, []).append(uv); trails[i] = trails[i][-8:]
        for i, uv in tracks.items():
            pts = np.array(trails[i], int)
            for j in range(1, len(pts)):
                cv2.line(head, tuple(pts[j - 1]), tuple(pts[j]), (0, 255, 0), 1)
            cv2.circle(head, (int(uv[0]), int(uv[1])), 2, (0, 255, 0), -1)
        trails = {i: trails[i] for i in tracks}

        cv2.putText(ext, "external", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(head, f"head cam  +  {len(tracks)} live tracks", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        frame = cv2.cvtColor(np.hstack([ext, head]), cv2.COLOR_RGB2BGR)
        cv2.imshow(WIN, frame)
        f += 1
        if (cv2.waitKey(max(1, int(1000 / FPS))) & 0xFF) in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
