"""Stage 3 — online camera–IMU self-calibration (see docs/STAGE3_CALIB_DESIGN.md).

`CalibVioESKF` augments the Stage-2 `VioESKF` with a camera-mount **rotation extrinsic** `δφ_HC` (3) and a
camera **time offset** `td` (1), both estimated online from the same relative-rotation vision measurement.
Error state: 18 (base) + 4 (calib) [+ 3 clone] = **22 uncloned / 25 cloned**, with the calib block placed
*before* the clone so cloning stays a clean truncate-then-append. `VioESKF`/`ESKF` are subclassed, not
modified (Stage-1/2 stay reproducible). Every Jacobian block is FD-verified by scripts/check_calib_jacobian.py.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag

from .estimator import ESKF, exp_so3, log_so3
from .vio import VioESKF


class CalibVioESKF(VioESKF):
    def __init__(self, kin, p0, R0, *, R_HC0, td0=0.0, est_extrinsic=True, est_td=True,
                 sig_phi0=np.radians(5.0), sig_td0=0.030, q_phi=0.0, q_td=0.0, **kw):
        super().__init__(kin, p0, R0, **kw)
        self.R_HC = np.asarray(R_HC0, float)          # online mount estimate (head←camera, CV)
        self.td = float(td0)                          # online time-offset estimate (s)
        self.q_phi = q_phi if est_extrinsic else 0.0
        self.q_td = q_td if est_td else 0.0
        pcal = [sig_phi0 ** 2 if est_extrinsic else 0.0] * 3 + [sig_td0 ** 2 if est_td else 0.0]
        P18 = self.P                                  # 18×18 from ESKF.__init__ (calib block appended)
        self.P = np.zeros((22, 22))
        self.P[:18, :18] = P18
        self.P[18:22, 18:22] = np.diag(pcal)
        self.R_BH_A = None                            # anchor head pose (recompose R_BC,A with LIVE mount)
        self.w_CA = None                              # anchor camera angular rate (captured at clone)

    # ---- stochastic cloning (22 → 25) --------------------------------------
    def clone(self, R_BH_A, gyro_A, w_neck_A):
        if self.cloned:
            self.P = self.P[:22, :22].copy()          # drop the previous clone
        self.R_A = self.R.copy()
        self.R_BH_A = np.asarray(R_BH_A, float)
        R_BC_A = self.R_BH_A @ self.R_HC
        self.w_CA = R_BC_A.T @ ((np.asarray(gyro_A, float) - self.bg) + np.asarray(w_neck_A, float))
        P = self.P
        self.P = np.block([[P, P[:, 6:9]], [P[6:9, :], P[6:9, 6:9]]])   # copy δθ block + cross-covariances
        self.cloned = True

    re_anchor = clone

    def predict(self, gyro_m, accel_m, dt):
        F, Q = self._predict_step(gyro_m, accel_m, dt)                  # 18-dim base
        Qcal = np.diag([self.q_phi ** 2 * dt] * 3 + [self.q_td ** 2 * dt])
        F = block_diag(F, np.eye(4)); Q = block_diag(Q, Qcal)          # calib states are constants
        if self.cloned:
            F = block_diag(F, np.eye(3)); Q = block_diag(Q, np.zeros((3, 3)))
        self.P = F @ self.P @ F.T + Q

    # ---- injection ----------------------------------------------------------
    def _inject(self, dx):
        ESKF._inject(self, dx)                        # base p,v,R,b_g,b_a,b_c (reads dx[:18]) — grandparent
        self.R_HC = self.R_HC @ exp_so3(dx[18:21])    # mount extrinsic (camera-frame right perturbation)
        self.td = self.td + dx[21]                    # time offset
        if self.cloned and len(dx) >= 25:
            self.R_A = self.R_A @ exp_so3(dx[22:25])  # clone nominal

    # ---- vision update ------------------------------------------------------
    def _cam_rate(self, R_BC, gyro, w_neck):
        """Camera angular rate in the camera frame: ω_C = R_BCᵀ·((gyro − b_g) + ω_neck^B)."""
        return R_BC.T @ ((np.asarray(gyro, float) - self.bg) + np.asarray(w_neck, float))

    def _calib_rH(self, R_meas, R_BH_j, gyro_j, w_neck_j):
        """Residual r + Jacobian H (3×25) for the relative-rotation measurement with mount + td states."""
        R_BC_A = self.R_BH_A @ self.R_HC              # recompose BOTH sides with the LIVE mount estimate
        R_BC_j = R_BH_j @ self.R_HC
        w_Cj, w_CA = self._cam_rate(R_BC_j, gyro_j, w_neck_j), self.w_CA
        A_hat = self.R.T @ self.R_A
        h_hat = R_BC_A.T @ (self.R_A.T @ self.R) @ R_BC_j
        C = exp_so3(-w_Cj * self.td)
        h_td = exp_so3(-w_CA * self.td) @ h_hat @ exp_so3(w_Cj * self.td)
        r = log_so3(h_td.T @ R_meas)
        R_CB_j = R_BC_j.T
        H = np.zeros((3, 25))
        H[:, 6:9] = C @ R_CB_j                        # δθ
        H[:, 18:21] = C @ (np.eye(3) - h_hat.T)       # δφ_HC
        H[:, 21] = w_Cj - C @ h_hat.T @ w_CA          # δtd (single column)
        H[:, 22:25] = -C @ R_CB_j @ A_hat             # δθ_A
        return r, H

    def update_rel_rot(self, R_meas, R_BH_j, gyro_j, w_neck_j):
        """Fuse a vision relative-rotation R_meas, estimating the mount + time offset. True if applied."""
        if not self.cloned:
            return False
        r, H = self._calib_rH(np.asarray(R_meas, float), np.asarray(R_BH_j, float), gyro_j, w_neck_j)
        S = H @ self.P @ H.T + self.Rvis
        self.last_vis_nis = float(r @ np.linalg.solve(S, r))
        if self.last_vis_nis > self.vis_gate:
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self._inject(K @ r)
        I_KH = np.eye(25) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.Rvis @ K.T          # Joseph form
        return True
