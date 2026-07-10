"""Interactive viewer — Asimov walking the closed loop in the VIO feature room.

Run on macOS:   .venv/bin/mjpython scripts/view_loop.py

Mouse: left-drag = orbit, right-drag = pan, scroll = zoom.
Press **C** (also Tab / [ / ]) to cycle cameras:  free (orbit) → head_cam (the VIO first-person view)
→ external studio cams. The current camera is printed to the terminal. Close the window to quit.
"""
import time

import mujoco
import mujoco.viewer

from humanoid_asimov.scene import build_model
from humanoid_asimov.walk import WalkController, loop_heading

PERIOD = 24.0
CYCLE_KEYS = {ord("C"), ord("c"), ord("["), ord("]"), 258}   # C / [ / ] / Tab(GLFW 258)


def main():
    model = build_model(add_head_camera=True, add_environment=True)
    data = mujoco.MjData(model)
    ctrl = WalkController(model, heading=loop_heading(PERIOD))
    ctrl.reset_to_gait(model, data, 0.0)
    dt = model.opt.timestep
    steps_per_sync = max(1, round((1 / 60.0) / dt))

    cams, names = [-1], ["free (orbit)"]                      # -1 = free camera, then named fixed cams
    for nm in ("head_cam", "front_camera", "side_camera", "back_camera"):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, nm)
        if cid >= 0:
            cams.append(cid); names.append(nm)
    state = {"i": 0, "viewer": None}

    def apply_cam():
        v = state["viewer"]
        if v is None:
            return
        cid = cams[state["i"]]
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE if cid < 0 else mujoco.mjtCamera.mjCAMERA_FIXED
        if cid >= 0:
            v.cam.fixedcamid = cid
        print(f"camera -> {names[state['i']]}")

    def key_callback(keycode):
        if keycode in CYCLE_KEYS:
            state["i"] = (state["i"] + 1) % len(cams)
            apply_cam()

    print("Walking the loop. Orbit / pan / zoom with the mouse.")
    print(f"Press C to cycle cameras:  {'  ->  '.join(names)}")
    print("Close the window to quit.")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        state["viewer"] = viewer
        while viewer.is_running():
            t0 = time.time()
            for _ in range(steps_per_sync):
                ctrl.apply(model, data)
                mujoco.mj_step(model, data)
            viewer.sync()
            dead = 1 / 60.0 - (time.time() - t0)
            if dead > 0:
                time.sleep(dead)


if __name__ == "__main__":
    main()
