"""Stage 2.2 — VIO fusion: the contact-aided ESKF augmented with a cloned anchor-rotation state, updated
by the front-end's relative-rotation measurements (see docs/STAGE2_VIO_DESIGN.md).

Vision measures R_{C_A→C_j} (camera rotation from an anchor keyframe A to the current keyframe j). We fuse
it as a **3-dof SO(3) update** against a **stochastically-cloned** anchor rotation state (error dims 18→21).
The update is gauge-safe: it observes the yaw *drift rate* + b_g (incl. b_g,z), not the absolute yaw.

The Stage-1 ``ESKF`` is subclassed, never modified — Stage-1 results stay reproducible.
"""
from __future__ import annotations

import numpy as np

from .estimator import ESKF, exp_so3, log_so3


class VioESKF(ESKF):
    """ESKF + a cloned anchor-rotation state for anchored relative-rotation vision updates."""

    def __init__(self, *args, sig_vis=2.0e-3, vis_gate=11.34, **kwargs):
        super().__init__(*args, **kwargs)
        # sig_vis=2 mrad is HONEST: the sharpened front-end (IRLS refit on all inliers + textured far field,
        # docs/STAGE2_FRONTEND_DESIGN.md) measures median |rotation error| ~2 mrad, |bias| ~0.4, so R_vis
        # bounds the real measurement. (Before that refit the front-end was ~5-6 mrad and this same 2 mrad was
        # optimistic — the front-end accuracy, not the fusion, was the limiter; that is now fixed, so VIO beats
        # the ESKF across seeds at this honest noise level and pins b_g,z.)
        self.Rvis = (sig_vis ** 2) * np.eye(3)     # vision rotation-measurement noise (rad^2)
        self.vis_gate = vis_gate                   # chi-square(3) gate
        self.cloned = False
        self.R_A = None                            # anchor nominal rotation R_WB(t_A)
        self.R_BC_A = None                         # anchor camera extrinsic (camera→base) at t_A
        self.last_vis_nis = None

    # ---- stochastic cloning -------------------------------------------------
    def clone(self, R_BC_A):
        """Clone the current rotation error as the anchor state (18→21), storing the anchor nominals."""
        if self.cloned:
            self.P = self.P[:18, :18].copy()       # drop the previous clone first
        self.R_A = self.R.copy()
        self.R_BC_A = np.asarray(R_BC_A, float)
        P = self.P
        self.P = np.block([[P, P[:, 6:9]], [P[6:9, :], P[6:9, 6:9]]])   # copy δθ block + cross-covariances
        self.cloned = True

    re_anchor = clone

    def predict(self, gyro_m, accel_m, dt):
        F, Q = self._predict_step(gyro_m, accel_m, dt)
        if self.cloned:                            # the clone is static: F_aug = blkdiag(F, I), Q_aug = (Q, 0)
            F = np.block([[F, np.zeros((18, 3))], [np.zeros((3, 18)), np.eye(3)]])
            Q = np.block([[Q, np.zeros((18, 3))], [np.zeros((3, 18)), np.zeros((3, 3))]])
        self.P = F @ self.P @ F.T + Q

    # ---- vision update ------------------------------------------------------
    def _vis_rH(self, R_meas, R_BC_j):
        """Residual r and Jacobian H (3x21) for a relative-rotation measurement R_meas = R_{C_A→C_j}.
        h(x) = R_BC,Aᵀ · (R_Aᵀ R̂_WB,j) · R_BC,j ;  r = Log(h(x)ᵀ · R_meas). Body-frame δθ (R_true=R̂·Exp(δθ))."""
        A_hat = self.R.T @ self.R_A                        # R̂_WB,jᵀ · R_A
        h_hat = self.R_BC_A.T @ (self.R_A.T @ self.R) @ R_BC_j
        r = log_so3(h_hat.T @ R_meas)
        H = np.zeros((3, 21))
        H[:, 6:9] = R_BC_j.T                               # ∂r/∂δθ   = R_CB,j
        H[:, 18:21] = -R_BC_j.T @ A_hat                    # ∂r/∂δθ_A = −R_CB,j · Â
        return r, H

    def _inject(self, dx):
        super()._inject(dx)                                # p, v, R, b_g, b_a, b_c (reads dx[:18])
        if self.cloned and len(dx) >= 21:                  # clone nominal (contact update also nudges it
            self.R_A = self.R_A @ exp_so3(dx[18:21])       # via cross-covariance; vision update directly)

    def update_rel_rot(self, R_meas, R_BC_j):
        """Fuse a vision relative-rotation R_meas (= R_{C_A→C_j}). Returns True if applied (else gated)."""
        if not self.cloned:
            return False
        r, H = self._vis_rH(np.asarray(R_meas, float), np.asarray(R_BC_j, float))
        S = H @ self.P @ H.T + self.Rvis
        self.last_vis_nis = float(r @ np.linalg.solve(S, r))
        if self.last_vis_nis > self.vis_gate:
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self._inject(K @ r)
        I_KH = np.eye(21) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.Rvis @ K.T          # Joseph form
        return True
