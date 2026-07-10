"""Sanity tests for the Stage-0 sim harness.  Run: .venv/bin/python -m pytest -q"""
import mujoco
import numpy as np

from humanoid_asimov.scene import HEAD_CAMERA, build_model
from humanoid_asimov.walk import simulate


def test_head_camera_present():
    m = build_model(add_head_camera=True)
    cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
    assert HEAD_CAMERA in cams


def test_walk_stays_upright_and_moves():
    _, _, log, _ = simulate(seconds=4.0, settle=1.0)
    assert log["gt_pos"][:, 2].min() > 0.45  # upright the whole time
    assert np.linalg.norm(log["gt_pos"][-1, :2] - log["gt_pos"][0, :2]) > 1.5  # actually walked
    assert not np.isnan(log["accel"]).any()
    assert not np.isnan(log["gyro"]).any()


def test_log_schema():
    _, _, log, _ = simulate(seconds=1.0, settle=0.5)
    n = len(log["t"])
    for key, dim in (("gyro", 3), ("accel", 3), ("enc_q", 15), ("enc_qd", 15),
                     ("gt_pos", 3), ("gt_quat", 4), ("gt_linvel", 3), ("gt_angvel", 3)):
        assert log[key].shape == (n, dim), key
    assert log["contact"].shape == (n, 2) and log["contact"].dtype == bool
    assert len(log["joint_names"]) == 15
