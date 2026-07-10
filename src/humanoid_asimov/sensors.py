"""Onboard sensing, split into what the estimator may use vs. ground truth for evaluation.

The real Asimov exposes a 6-DoF IMU (raw gyro + accel) and joint encoders — and crucially
**no external motion capture and no attitude/velocity oracle**. There is no foot-force sensor on the
platform: the per-foot contact flag here is derived from the engine's collision list (a stand-in for a
real contact estimator). So:

- **Estimator-visible** (`Sensors.imu / .encoders / .contacts`): raw gyro + accelerometer, encoder
  angles/rates for the sensable joints, and per-foot contact flags + normal force.
- **Ground truth** (`Sensors.*_gt`): base pose/velocity and the IMU orientation quaternion — the
  sim provides these; the real robot does not, so the estimator must NOT consume them. They exist
  only to score the estimator.

Indices are precomputed once (no per-call model rescans). An optional `ImuNoise` adds LSM6DSOX-style
white noise + bias random walk so consistency metrics are meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

# Model sensor names.
S_GYRO = "imu_ang_vel"   # gyroscope: angular velocity (rad/s), IMU/pelvis frame
S_ACC = "imu_lin_acc"    # accelerometer: specific force (m/s^2), IMU frame (includes gravity)
S_QUAT = "imu_quat"      # orientation quaternion (w, x, y, z), world — GROUND TRUTH only

FOOT_BODIES = {
    "left": ("left_ankle_roll_link", "left_toe_link"),
    "right": ("right_ankle_roll_link", "right_toe_link"),
}
FOOT_SITES = {"left": "left_foot", "right": "right_foot"}

# Joints the estimator is allowed to read encoders from: legs (12) + waist + neck (15).
# Excludes passive toes (no encoder on the real robot) and the spring-passive arms.
ESTIMATOR_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint",
]


@dataclass
class Imu:
    gyro: np.ndarray    # (3,) rad/s
    accel: np.ndarray   # (3,) m/s^2 (specific force, includes gravity)


def _sensor_slice(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise KeyError(f"sensor {name!r} not in model")
    adr = model.sensor_adr[sid]
    return slice(int(adr), int(adr + model.sensor_dim[sid]))


class Sensors:
    """Precomputed sensor read-out for one model. Estimator-visible + `*_gt` (eval only)."""

    def __init__(self, model):
        self.model = model
        self._gyro = _sensor_slice(model, S_GYRO)
        self._acc = _sensor_slice(model, S_ACC)
        self._quat = _sensor_slice(model, S_QUAT)

        self.joint_names = [n for n in ESTIMATOR_JOINTS
                            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self._qadr = np.array([model.jnt_qposadr[j] for j in jids], dtype=int)
        self._dadr = np.array([model.jnt_dofadr[j] for j in jids], dtype=int)

        self._foot_geoms = {s: self._body_geoms(b) for s, b in FOOT_BODIES.items()}
        self._foot_sites = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, FOOT_SITES[s])
                            for s in FOOT_BODIES}

    def _body_geoms(self, body_names):
        geoms = set()
        for bn in body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bn)
            if bid >= 0:
                geoms |= {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid}
        return frozenset(geoms)

    # -------- estimator-visible --------
    def imu(self, data) -> Imu:
        return Imu(np.array(data.sensordata[self._gyro]), np.array(data.sensordata[self._acc]))

    def encoders(self, data):
        """(q, qd) for the sensable joints, in `self.joint_names` order."""
        return np.array(data.qpos[self._qadr]), np.array(data.qvel[self._dadr])

    def contacts(self, data):
        """Per-foot {in_contact, normal_force} from summed contact normals."""
        forces = {s: 0.0 for s in self._foot_geoms}
        f6 = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            for side, gs in self._foot_geoms.items():
                if c.geom1 in gs or c.geom2 in gs:
                    mujoco.mj_contactForce(self.model, data, i, f6)
                    forces[side] += abs(f6[0])
        return {s: {"in_contact": forces[s] > 1.0, "normal_force": forces[s]} for s in self._foot_geoms}

    # -------- ground truth (eval only — never feed to the estimator) --------
    def orientation_gt(self, data):
        return np.array(data.sensordata[self._quat])              # (4,) wxyz, world

    def base_pose_gt(self, data):
        return np.array(data.qpos[0:3]), np.array(data.qpos[3:7])  # world pos, wxyz quat

    def base_velocity_gt(self, data):
        return np.array(data.qvel[0:3]), np.array(data.qvel[3:6])  # world linear, BODY-frame angular

    def foot_positions_gt(self, data):
        return {s: (np.array(data.site_xpos[i]) if i >= 0 else None)
                for s, i in self._foot_sites.items()}


@dataclass
class ImuNoise:
    """Additive IMU noise + bias random walk (LSM6DSOX-scale defaults).

    White noise per sample + a slowly-drifting bias the estimator must track. Seeded for
    reproducibility. `corrupt(imu, dt, rng)` returns a noisy copy; call once per logged sample.
    """
    gyro_noise: float = 1.0e-3      # rad/s (per sample)
    accel_noise: float = 1.5e-2     # m/s^2 (per sample)
    gyro_bias_rw: float = 2.0e-4    # rad/s / sqrt(s)
    accel_bias_rw: float = 4.0e-3   # m/s^2 / sqrt(s)
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def corrupt(self, imu: Imu, dt: float, rng) -> Imu:
        self.gyro_bias = self.gyro_bias + rng.normal(0.0, self.gyro_bias_rw * np.sqrt(dt), 3)
        self.accel_bias = self.accel_bias + rng.normal(0.0, self.accel_bias_rw * np.sqrt(dt), 3)
        gyro = imu.gyro + self.gyro_bias + rng.normal(0.0, self.gyro_noise, 3)
        accel = imu.accel + self.accel_bias + rng.normal(0.0, self.accel_noise, 3)
        return Imu(gyro, accel)
