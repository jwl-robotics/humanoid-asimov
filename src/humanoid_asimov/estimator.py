"""State estimation from onboard sensors (Stage 1: IMU + leg-odometry).

The estimator recovers the base pose from what the robot can actually sense — gyro, accelerometer,
joint encoders, foot contact — never the sim ground truth. Pieces:

- ``RobotKinematics`` — the robot's own forward kinematics from encoder angles, via a MuJoCo model
  used purely as an FK engine (identity base ⇒ foot positions + Jacobians in the base frame). The
  robot's *known* kinematics (its URDF), not sim state.
- ``LegOdometry`` — dead reckoning from "planted foot = fixed world anchor": p_base = anchor − R·p_f.
- ``ESKF`` — the contact-aided **error-state EKF**. IMU strapdown drives the predict; the
  no-slip contact-foot velocity (world velocity of a planted foot ≈ 0) drives the update; the filter
  estimates the gyro/accel biases online. 18-dim error state {δp, δv, δθ, δb_g, δb_a, δb_c}, body-frame
  orientation error (R_true = R·Exp(δθ)).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import build_model
from .sensors import FOOT_SITES

GRAVITY = np.array([0.0, 0.0, -9.81])


# ---------------------------------------------------------------- SO(3) helpers
def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_so3(w):
    """Rotation matrix from an axis-angle vector (Rodrigues)."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        return np.eye(3) + skew(w)
    k = w / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def log_so3(R):
    """Axis-angle vector from a rotation matrix (inverse of exp_so3)."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(c))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-9:
        return 0.5 * v
    return (theta / (2.0 * np.sin(theta))) * v


def quat2mat(q):
    """MuJoCo (w, x, y, z) quaternion → 3x3 rotation matrix (world←body)."""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


# ---------------------------------------------------------------- kinematics
class RobotKinematics:
    """Forward kinematics (+ foot Jacobians) from encoder angles — MuJoCo as an FK engine."""

    def __init__(self):
        self.model = build_model(add_head_camera=False)
        self.data = mujoco.MjData(self.model)
        self.qadr, self.dadr = {}, {}
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.qadr[name] = self.model.jnt_qposadr[j]
            self.dadr[name] = self.model.jnt_dofadr[j]
        self.foot_sites = {
            s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for s, name in FOOT_SITES.items()
        }

    def _set_state(self, enc_q, joint_names, enc_qd=None):
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0                         # base at origin, identity orientation
        self.data.qvel[:] = 0.0
        for i, name in enumerate(joint_names):
            self.data.qpos[self.qadr[name]] = enc_q[i]
            if enc_qd is not None:
                self.data.qvel[self.dadr[name]] = enc_qd[i]

    def foot_pos_base(self, enc_q, joint_names):
        """Foot-site positions in the base frame. {'left':(3,), 'right':(3,)}."""
        self._set_state(enc_q, joint_names)
        mujoco.mj_kinematics(self.model, self.data)     # base at origin ⇒ site_xpos == base-frame position
        return {s: self.data.site_xpos[i].copy() for s, i in self.foot_sites.items()}

    def foot_pos_vel_base(self, enc_q, enc_qd, joint_names):
        """Per foot: (position in base frame, joint-induced velocity in base frame).

        The joint velocity is J(q)·q̇ with the base DOFs zeroed — i.e. the foot's velocity relative to
        the base due to leg-joint motion only (the base-motion term is handled separately in the ESKF).
        """
        self._set_state(enc_q, joint_names, enc_qd)
        mujoco.mj_forward(self.model, self.data)         # need cdof for the site Jacobian
        jacp = np.zeros((3, self.model.nv))
        out = {}
        for s, sid in self.foot_sites.items():
            mujoco.mj_jacSite(self.model, self.data, jacp, None, sid)
            p_f = self.data.site_xpos[sid].copy()
            v_f = jacp @ self.data.qvel                  # base DOFs are 0 ⇒ joint contribution only
            out[s] = (p_f, v_f.copy())
        return out


# ---------------------------------------------------------------- leg odometry
class LegOdometry:
    """Base-position dead reckoning from the planted-foot anchor constraint (attitude supplied)."""

    def __init__(self, kin: RobotKinematics, p0):
        self.kin = kin
        self.p = np.array(p0, dtype=float)
        self.anchor = None
        self.foot = None

    @staticmethod
    def _choose(contact, current):
        left, right = bool(contact[0]), bool(contact[1])
        if current == "left" and left:
            return "left"
        if current == "right" and right:
            return "right"
        if left:
            return "left"
        if right:
            return "right"
        return None

    def step(self, R, enc_q, joint_names, contact):
        fp = self.kin.foot_pos_base(enc_q, joint_names)
        foot = self._choose(contact, self.foot)
        if foot is None:
            return self.p.copy()
        if foot != self.foot or self.anchor is None:
            self.anchor = self.p + R @ fp[foot]
            self.foot = foot
        self.p = self.anchor - R @ fp[foot]
        return self.p.copy()


# ---------------------------------------------------------------- error-state EKF
class ESKF:
    """Contact-aided error-state EKF: IMU predict + contact-velocity update + online bias estimation.

    State: p, v (world), R (world←body), b_g, b_a, b_c. Error δx = [δp, δv, δθ, δb_g, δb_a, δb_c] (18),
    body-frame orientation error R_true = R·Exp(δθ). b_c is a slip-velocity bias: the planted foot's world
    velocity is modelled as R·b_c (not exactly 0), because the ankle site rolls about a *moving* contact
    point — estimating it online keeps that systematic out of b_g (which it otherwise corrupts, +2.5 mrad/s)
    and off the vertical channel. (b_c is shared across feet on purpose: it is observed every step so it
    stays bounded; a per-foot b_c was tried but its fore-aft component aliases with base velocity and
    diverges. The residual it leaves — the lateral slip flips sign between feet — is the known limit here,
    addressable later by an InEKF with foot-position states.) Yaw is unobservable from IMU+leg-odometry
    alone (drifts slowly) — expected; vision (Stage 2) anchors it.

    Process-noise defaults ≈ the injected LSM6DSOX sensor noise. This yields an honest *point estimate*
    (b_g de-biased, low drift) but a mildly over-confident covariance: the within-stance contact error is
    *colored* (the NIS stays correlated), which a diagonal-Q ESKF cannot fully model — the principled cure
    is an InEKF with foot-position states, not more Q tuning. See docs/STAGE1_ESKF_REVIEW.md.
    """

    def __init__(self, kin: RobotKinematics, p0, R0, v0=None,
                 sig_a=6.7e-4, sig_g=4.5e-5, sig_bg=2.0e-4, sig_ba=4.0e-3, sig_contact=0.05,
                 sig_bc=1.0e-3, p0_bc=1.0e-3, sig_grav=0.5, grav_band=0.8, gate=1.0e6):
        self.kin = kin
        self.p = np.array(p0, float)
        self.v = np.zeros(3) if v0 is None else np.array(v0, float)
        self.R = np.array(R0, float)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.bc = np.zeros(3)       # shared slip-velocity bias (body frame): planted foot's world vel ≈ R·b_c
        self.P = np.diag(np.concatenate([[1e-4] * 3, [1e-2] * 3, [1e-4] * 3,
                                         [1e-6] * 3, [1e-4] * 3, [p0_bc] * 3]))
        self.sig_a, self.sig_g, self.sig_bg, self.sig_ba, self.sig_bc = sig_a, sig_g, sig_bg, sig_ba, sig_bc
        self.Rc = (sig_contact ** 2) * np.eye(3)
        self.Rg = (sig_grav ** 2) * np.eye(3)
        self.grav_band = grav_band  # gravity update only when |accel| ≈ g (quasi-static)
        self.gate = gate            # χ² gate effectively OFF (1e6): per-update gating of a colored residual
        #                             mis-gates; structured Rc first (STAGE1 review #3), then re-enable at 16.27
        self.anchor_foot = None     # single stance foot (persistence avoids double-support conflict)
        self.last_nis = None        # contact-update NIS + innovation (for consistency validation)
        self.last_innov = None

    def _predict_step(self, gyro_m, accel_m, dt):
        """Advance the nominal state; return the (F, Q) error-state transition (18-dim). Split out so
        VioESKF can wrap the same propagation for its augmented (cloned) covariance — the Stage-1 math
        below is unchanged."""
        w = gyro_m - self.bg
        a = accel_m - self.ba
        Rn = self.R
        a_w = Rn @ a + GRAVITY
        # nominal integration
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        self.R = Rn @ exp_so3(w * dt)
        # error-state transition (18-dim: adds the slip-velocity bias b_c as a random walk)
        F = np.eye(18)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -Rn @ skew(a) * dt
        F[3:6, 12:15] = -Rn * dt
        F[6:9, 6:9] = exp_so3(-w * dt)
        F[6:9, 9:12] = -np.eye(3) * dt
        Q = np.zeros((18, 18))
        Q[3:6, 3:6] = (self.sig_a ** 2) * dt * np.eye(3)
        Q[6:9, 6:9] = (self.sig_g ** 2) * dt * np.eye(3)
        Q[9:12, 9:12] = (self.sig_bg ** 2) * dt * np.eye(3)
        Q[12:15, 12:15] = (self.sig_ba ** 2) * dt * np.eye(3)
        Q[15:18, 15:18] = (self.sig_bc ** 2) * dt * np.eye(3)
        return F, Q

    def predict(self, gyro_m, accel_m, dt):
        F, Q = self._predict_step(gyro_m, accel_m, dt)
        self.P = F @ self.P @ F.T + Q

    def _inject(self, dx):
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.R = self.R @ exp_so3(dx[6:9])
        self.bg = self.bg + dx[9:12]
        self.ba = self.ba + dx[12:15]
        self.bc = self.bc + dx[15:18]

    def _stance_foot(self, contact):
        left, right = bool(contact[0]), bool(contact[1])
        if self.anchor_foot == "left" and left:
            return "left"
        if self.anchor_foot == "right" and right:
            return "right"
        if left:
            return "left"
        if right:
            return "right"
        return None

    def _contact_rH(self, p_f, v_f_joint, gyro_m):
        """Residual (innov) + Jacobian H (3×dim) for the no-slip contact-velocity update. FD-checkable —
        see scripts/check_eskf_jacobian.py. Measurement h(x) = v + R·u, u = [ω]·p_f + v_f_joint − b_c,
        ω = gyro − b_g; innov = −h (a planted foot's world velocity ≈ 0)."""
        w = gyro_m - self.bg
        u = skew(w) @ p_f + v_f_joint - self.bc          # foot velocity in body frame, minus the slip bias
        innov = -(self.v + self.R @ u)                   # planted foot: world velocity ≈ 0
        H = np.zeros((3, self.P.shape[0]))               # 18, or 21/25 when the VIO clone / calib states exist
        H[:, 3:6] = np.eye(3)
        H[:, 6:9] = -self.R @ skew(u)
        H[:, 9:12] = self.R @ skew(p_f)                  # b_g enters via ω = gyro − b_g (the lever term R·[p_f]×)
        H[:, 15:18] = -self.R                            # slip-velocity bias (planted foot ≈ R·b_c)
        return innov, H

    def update_contact(self, enc_q, enc_qd, joint_names, contact, gyro_m):
        """No-slip update on a single stance foot: its world velocity ≈ 0 (slip/impact gated out)."""
        self.last_nis, self.last_innov = None, None
        foot = self._stance_foot(contact)
        self.anchor_foot = foot
        if foot is None:
            return
        p_f, v_f_joint = self.kin.foot_pos_vel_base(enc_q, enc_qd, joint_names)[foot]
        innov, H = self._contact_rH(p_f, v_f_joint, gyro_m)
        S = H @ self.P @ H.T + self.Rc
        self.last_nis = float(innov @ np.linalg.solve(S, innov))   # NIS (3-dof) for consistency checks
        self.last_innov = innov.copy()
        if self.last_nis > self.gate:                    # chi-square gate: reject slip/impact
            return
        K = self.P @ H.T @ np.linalg.inv(S)
        self._inject(K @ innov)
        I_KH = np.eye(self.P.shape[0]) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.Rc @ K.T        # Joseph form

    def update_gravity(self, accel_m):
        """Quasi-static accelerometer ≈ gravity direction → observes roll/pitch + accel bias."""
        if abs(float(np.linalg.norm(accel_m)) - 9.81) > self.grav_band:
            return                                        # only when nearly static (|accel| ≈ g)
        g_body = self.R.T @ np.array([0.0, 0.0, 9.81])    # predicted specific force if static
        innov = accel_m - (g_body + self.ba)
        H = np.zeros((3, self.P.shape[0]))                # dimension-aware (18, or 21/25 when cloned/calib)
        H[:, 6:9] = skew(g_body)
        H[:, 12:15] = np.eye(3)
        S = H @ self.P @ H.T + self.Rg
        K = self.P @ H.T @ np.linalg.inv(S)
        self._inject(K @ innov)
        I_KH = np.eye(self.P.shape[0]) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.Rg @ K.T

    def step(self, gyro_m, accel_m, enc_q, enc_qd, joint_names, contact, dt):
        """One full cycle: IMU predict → contact-velocity update.

        The gravity/roll-pitch update is available via ``update_gravity`` but is left OUT of the
        walking loop: during continuous gait ``|accel| ≈ g`` also occurs mid-stride when the accel is
        not pure gravity, so it injects bad tilt corrections. It is useful for static init / standing.
        """
        self.predict(gyro_m, accel_m, dt)
        self.update_contact(enc_q, enc_qd, joint_names, contact, gyro_m)
        return self.p.copy()
