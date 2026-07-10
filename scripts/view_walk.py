"""Watch Asimov walk LIVE in the interactive viewer (real-time, orbitable).

    .venv/bin/mjpython scripts/view_walk.py     # macOS
    .venv/bin/python   scripts/view_walk.py     # Linux / Windows

Runs the PD-tracked walking gait (+ torso virtual support) inside the interactive window and
paces it to wall-clock time. The gait loops, so it walks indefinitely. Mouse: left-drag = orbit,
right-drag = pan, scroll = zoom. Close the window to quit.
"""
import time

import mujoco
import mujoco.viewer

from humanoid_asimov.scene import build_model
from humanoid_asimov.walk import WalkController


def main():
    model = build_model(add_head_camera=True)
    data = mujoco.MjData(model)
    ctrl = WalkController(model, stabilize=True)
    ctrl.reset_to_gait(model, data, 0.0)

    steps_per_sync = max(1, int((1 / 60) / model.opt.timestep))  # ~60 fps refresh
    print("Walking live. Orbit: left-drag | Pan: right-drag | Zoom: scroll | Quit: close window")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # camera follows the robot as it walks (you can still orbit/zoom around it)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = ctrl.pelvis_id
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.5, 90.0, -15.0
        while viewer.is_running():
            t0 = time.time()
            for _ in range(steps_per_sync):
                ctrl.apply(model, data)
                mujoco.mj_step(model, data)
            viewer.sync()
            target = steps_per_sync * model.opt.timestep
            slack = target - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)


if __name__ == "__main__":
    main()
