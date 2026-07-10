"""Drive Asimov by PD-tracking its open walking trajectory (development locomotion black box).

The upstream model has no leg actuators (the RL policy supplies torques on hardware). To generate
motion for estimator development we track the reference joint trajectory (asimov-mjlab's imitation
gait — 12 leg joints @ 50 Hz) with a per-joint PD law applied as generalized forces, plus a torso
"virtual support" so the open-loop gait stays on its feet.

Honesty note: the support is **load-bearing** (~44% body weight), so this is NOT true dynamic
walking — it is a stand-in for the real balance controller, used only to produce self-consistent
IMU/encoder/contact data + ground truth. The estimator sees only the onboard sensors (never the
gait targets, the support wrench, or the ground truth). Swapping in the real RL policy is the
designated next upgrade.
"""
from __future__ import annotations

import csv
import os

import mujoco
import numpy as np

from .scene import build_model
from .sensors import Sensors

DATA_CSV = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "walking_1.25Hz_50Hz.csv"))
GAIT_HZ = 50.0
GAIT_RAMP = 10        # first rows are a stand->walk ramp; skip them when looping
GAIT_PERIOD = 40      # steady gait period in rows (1.25 Hz at 50 Hz)

# Per-joint PD gains (N·m/rad, N·m·s/rad) and a torque clip (safety, not normally binding).
KP = {"hip_pitch": 200.0, "hip_roll": 200.0, "hip_yaw": 120.0, "knee": 200.0,
      "ankle_pitch": 60.0, "ankle_roll": 40.0}
KD = {"hip_pitch": 5.0, "hip_roll": 5.0, "hip_yaw": 4.0, "knee": 5.0,
      "ankle_pitch": 3.0, "ankle_roll": 2.0}
TORQUE_LIMIT = 150.0

# Torso "virtual support" (dev stand-in for the balance controller; estimator never sees it).
BASE_KP_TILT, BASE_KD_TILT = 500.0, 50.0
BASE_KP_Z, BASE_KD_Z = 1200.0, 120.0
BASE_KP_YAW, BASE_KD_YAW = 250.0, 35.0        # optional yaw steering (loop trajectories; see heading=)
BASE_GRAVITY_FF = 0.5
NOMINAL_Z = 0.60


def loop_heading(period=24.0, laps=1.0):
    """A constant-yaw-rate heading ψ(t) = 2π·laps·t/period — the robot traces a closed loop and
    returns to its start heading after `period` s (one lap). Pass to simulate(heading=...)."""
    omega = 2.0 * np.pi * laps / period
    return lambda t: omega * t


def _gain(joint_name, table, default):
    for key, val in table.items():
        if key in joint_name:
            return val
    return default


def load_gait(path=DATA_CSV):
    """Return (joint_names[12], targets[T, 12]) — the leg-joint angle reference (radians)."""
    with open(path) as f:
        rows = list(csv.reader(f))
    names = rows[0][:12]
    traj = np.array([[float(x) for x in r[:12]] for r in rows[1:]])
    return names, traj


class WalkController:
    """Per-joint PD tracking of the reference gait (as generalized forces) + torso support."""

    def __init__(self, model, stabilize=True, heading=None, neck_scan=None):
        names, self.traj = load_gait()
        self.stabilize = stabilize
        self.heading = heading            # callable t -> desired yaw (rad); None = free yaw (straight)
        self.neck_scan = neck_scan        # (amplitude_rad, freq_hz) neck-pitch "look-around" scan, or None
        if neck_scan is not None:
            nj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch_joint")
            self.neck_qadr = model.jnt_qposadr[nj]
            self.neck_dadr = model.jnt_dofadr[nj]
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        self.weight = float(np.sum(model.body_mass) * abs(model.opt.gravity[2]))
        qadr, dadr, kp, kd = [], [], [], []
        for name in names:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise KeyError(f"gait joint {name!r} not in model")
            qadr.append(model.jnt_qposadr[j])
            dadr.append(model.jnt_dofadr[j])
            kp.append(_gain(name, KP, 60.0))
            kd.append(_gain(name, KD, 3.0))
        self.names = names
        self.qadr, self.dadr = np.array(qadr), np.array(dadr)
        self.kp, self.kd = np.array(kp), np.array(kd)
        # steady periodic window (integer cycles from the end of the ramp) — loops seamlessly
        ncycles = (len(self.traj) - GAIT_RAMP) // GAIT_PERIOD
        self.loop0 = GAIT_RAMP
        self.loop_len = ncycles * GAIT_PERIOD

    def target(self, t):
        """Interpolated joint targets at time t, looping the steady gait window (no wrap kick)."""
        s = (t * GAIT_HZ) % self.loop_len
        i = int(s)
        a = s - i
        r0 = self.loop0 + i
        r1 = self.loop0 + (i + 1) % self.loop_len
        return (1.0 - a) * self.traj[r0] + a * self.traj[r1]

    def reset_to_gait(self, model, data, t=0.0):
        data.qpos[self.qadr] = self.target(t)
        data.qpos[2] = NOMINAL_Z              # start near support height to cut the drop transient
        mujoco.mj_forward(model, data)

    def apply(self, model, data):
        data.qfrc_applied[:] = 0.0
        q = data.qpos[self.qadr]
        qd = data.qvel[self.dadr]
        tau = self.kp * (self.target(data.time) - q) - self.kd * qd
        data.qfrc_applied[self.dadr] = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
        if self.neck_scan is not None:                    # optional "look-around" neck-pitch scan
            amp, freq = self.neck_scan
            tgt = amp * np.sin(2.0 * np.pi * freq * data.time)
            data.qfrc_applied[self.neck_dadr] += (30.0 * (tgt - data.qpos[self.neck_qadr])
                                                  - 3.0 * data.qvel[self.neck_dadr])
        if self.stabilize:
            self._base_assist(model, data)

    def _base_assist(self, model, data):
        """World-frame virtual-support wrench on the pelvis (upright + near nominal height)."""
        pid = self.pelvis_id
        R = data.xmat[pid].reshape(3, 3)
        tilt = np.cross(R[:, 2], np.array([0.0, 0.0, 1.0]))   # world axis ~ sin(tilt); no yaw component
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pid, vel6, 0)
        ang_w = vel6[0:3].copy()
        wz = ang_w[2]
        ang_w[2] = 0.0                                        # tilt damping uses roll/pitch rate only
        torque = BASE_KP_TILT * tilt - BASE_KD_TILT * ang_w
        if self.heading is not None:                          # steer yaw toward the commanded heading
            psi = float(np.arctan2(R[1, 0], R[0, 0]))
            err = (self.heading(data.time) - psi + np.pi) % (2 * np.pi) - np.pi
            torque[2] = BASE_KP_YAW * err - BASE_KD_YAW * wz
        z, vz = data.qpos[2], data.qvel[2]
        fz = max(0.0, BASE_GRAVITY_FF * self.weight + BASE_KP_Z * (NOMINAL_Z - z) - BASE_KD_Z * vz)
        mujoco.mj_applyFT(model, data, np.array([0.0, 0.0, fz]), torque, data.xpos[pid], pid,
                          data.qfrc_applied)


# Log schema (per physics step). Estimator-visible vs. ground-truth (gt_*) is documented in sensors.py.
LOG_KEYS = ("t", "gyro", "accel", "enc_q", "enc_qd", "contact", "contact_force",
            "gt_pos", "gt_quat", "gt_linvel", "gt_angvel", "gt_bg", "gt_ba")


def simulate(seconds=8.0, render_camera=None, fps=30, width=640, height=480,
             stabilize=True, fix_foot_contacts=True, settle=1.5, imu_noise=None, seed=0,
             heading=None, environment=False, env_seed=0, head_cam_quat=None, head_cam_pos=None,
             neck_scan=None, control_hook=None):
    """Run the PD-tracked gait and log a self-consistent dataset. Returns (model, data, log, frames).

    Fixes the earlier one-step sensor/GT misalignment by refreshing sensors for the post-step state
    (``mj_forward`` after ``mj_step``) and logging sensors + ground truth at that same instant.
    ``settle`` seconds are simulated but not logged (sheds the startup transient). ``heading`` steers a
    loop trajectory (see ``loop_heading``); ``environment`` adds the visual-only feature room for VIO.
    """
    model = build_model(add_head_camera=True, fix_foot_contacts=fix_foot_contacts,
                        add_environment=environment, env_seed=env_seed,
                        head_cam_quat=head_cam_quat, head_cam_pos=head_cam_pos)
    data = mujoco.MjData(model)
    ctrl = WalkController(model, stabilize=stabilize, heading=heading, neck_scan=neck_scan)
    ctrl.reset_to_gait(model, data, 0.0)
    sensors = Sensors(model)
    rng = np.random.default_rng(seed)
    dt = model.opt.timestep

    for _ in range(int(settle / dt)):        # shed the startup transient (unlogged)
        ctrl.apply(model, data)
        mujoco.mj_step(model, data)
    t0 = data.time

    log = {k: [] for k in LOG_KEYS}
    frames, frame_t = [], []
    renderer = mujoco.Renderer(model, height, width) if render_camera is not None else None
    next_frame = data.time

    for _ in range(int(seconds / dt)):
        ctrl.apply(model, data)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)        # sensors now consistent with the post-step GT state

        imu = sensors.imu(data)
        if imu_noise is not None:
            imu = imu_noise.corrupt(imu, dt, rng)
        q, qd = sensors.encoders(data)
        con = sensors.contacts(data)
        pos, quat = sensors.base_pose_gt(data)
        linv, angv = sensors.base_velocity_gt(data)

        log["t"].append(data.time - t0)
        log["gyro"].append(imu.gyro)
        log["accel"].append(imu.accel)
        log["enc_q"].append(q)
        log["enc_qd"].append(qd)
        log["contact"].append([con["left"]["in_contact"], con["right"]["in_contact"]])
        log["contact_force"].append([con["left"]["normal_force"], con["right"]["normal_force"]])
        log["gt_pos"].append(pos)
        log["gt_quat"].append(quat)
        log["gt_linvel"].append(linv)
        log["gt_angvel"].append(angv)
        log["gt_bg"].append(imu_noise.gyro_bias.copy() if imu_noise is not None else np.zeros(3))
        log["gt_ba"].append(imu_noise.accel_bias.copy() if imu_noise is not None else np.zeros(3))

        if control_hook is not None:          # closed-loop navigation: estimator + planner steer ctrl live
            control_hook(data.time - t0, imu.gyro, imu.accel, q, qd,
                         [con["left"]["in_contact"], con["right"]["in_contact"]], dt, ctrl,
                         pos, quat)           # gt pose passed for estimator INIT only (known start reference)

        if renderer is not None and data.time >= next_frame:
            renderer.update_scene(data, render_camera)
            frames.append(renderer.render().copy())
            frame_t.append(data.time - t0)
            next_frame += 1.0 / fps

    out = {k: np.array(v) for k, v in log.items()}
    out["joint_names"] = list(sensors.joint_names)
    out["frame_t"] = np.array(frame_t)
    return model, data, out, frames


def save_dataset(log, path):
    """Persist a simulate() log to a compressed .npz (joint_names stored as an array of str)."""
    arrs = {k: v for k, v in log.items() if k != "joint_names"}
    arrs["joint_names"] = np.array(log["joint_names"])
    np.savez_compressed(path, **arrs)
    return path
