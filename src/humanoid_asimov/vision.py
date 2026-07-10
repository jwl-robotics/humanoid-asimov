"""Stage 2 VIO front-end — camera model + head-camera extrinsics (see docs/STAGE2_VIO_DESIGN.md).

Uses only onboard data: rendered head-cam frames + joint encoders + the known camera mount — never sim
ground truth (that is eval-only, sensors.py). Conventions: base/body B (estimator state R_WB), head
H = neck_pitch_link, camera C in the **CV** convention (z forward, x right, y down). MuJoCo cameras are GL
(−z forward, +y up), so a fixed flip F = diag(1, −1, −1) maps head→camera(CV).
"""
from __future__ import annotations

import cv2
import mujoco
import numpy as np

from .estimator import RobotKinematics, exp_so3, log_so3, quat2mat, skew
from .scene import HEAD_CAM_POS, HEAD_CAM_QUAT

_GL2CV = np.diag([1.0, -1.0, -1.0])          # GL camera axes → CV camera axes


class CameraModel:
    """Pinhole intrinsics from MuJoCo's vertical FOV; pixel ↔ bearing (CV frame, z forward)."""

    def __init__(self, width: int = 640, height: int = 480, fovy_deg: float = 70.0):
        self.width, self.height = width, height
        f = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
        self.fx = self.fy = float(f)
        self.cx, self.cy = width / 2.0, height / 2.0
        self.K = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fy, self.cy],
                           [0.0, 0.0, 1.0]])

    def bearings(self, uv: np.ndarray) -> np.ndarray:
        """(N,2) pixels → (N,3) unit bearings in the CV camera frame."""
        uv = np.atleast_2d(np.asarray(uv, float))
        x = (uv[:, 0] - self.cx) / self.fx
        y = (uv[:, 1] - self.cy) / self.fy
        b = np.stack([x, y, np.ones_like(x)], axis=1)
        return b / np.linalg.norm(b, axis=1, keepdims=True)

    def project(self, p_cam: np.ndarray) -> np.ndarray:
        """(N,3) points in the CV camera frame → (N,2) pixels (no cheirality/bounds check)."""
        p_cam = np.atleast_2d(np.asarray(p_cam, float))
        u = self.fx * p_cam[:, 0] / p_cam[:, 2] + self.cx
        v = self.fy * p_cam[:, 1] / p_cam[:, 2] + self.cy
        return np.stack([u, v], axis=1)


class HeadCamExtrinsics:
    """Camera(CV)→base rotation R_BC(enc_q) and camera-in-base position p_BC(enc_q), from the robot's own
    neck FK + the known mount. A base-at-origin RobotKinematics makes neck_pitch_link's xpos/xmat base-frame,
    so no sim state leaks in — only encoder angles + the fixed mount."""

    def __init__(self, kin: RobotKinematics | None = None):
        self.kin = kin or RobotKinematics()
        self.head_bid = mujoco.mj_name2id(self.kin.model, mujoco.mjtObj.mjOBJ_BODY, "neck_pitch_link")
        self.R_HC = quat2mat(np.array(HEAD_CAM_QUAT, float)) @ _GL2CV     # head → camera(CV)
        self.p_HC = np.array(HEAD_CAM_POS, float)

    def head_pose(self, enc_q, joint_names):
        """Head (neck_pitch_link) pose in the base frame, from the robot's own encoder FK: (R_BH, p_BH).
        The mount R_HC/p_HC is applied separately, so an *online* mount estimate (Stage 3) can compose here."""
        self.kin._set_state(enc_q, joint_names)
        mujoco.mj_kinematics(self.kin.model, self.kin.data)
        R_BH = self.kin.data.xmat[self.head_bid].reshape(3, 3).copy()     # base ← head
        p_BH = self.kin.data.xpos[self.head_bid].copy()                   # head origin in base
        return R_BH, p_BH

    def compute(self, enc_q, joint_names):
        """Return (R_BC, p_BC) with the *nominal* mount: camera(CV)→base rotation + camera pos in base."""
        R_BH, p_BH = self.head_pose(enc_q, joint_names)
        return R_BH @ self.R_HC, p_BH + R_BH @ self.p_HC

    def head_angvel_base(self, enc_q, enc_qd, joint_names):
        """Head angular velocity relative to the base, in the base frame, from encoders only. _set_state
        zeroes the base DOFs, so mj_jacBody's rotational part × qvel is the neck-joint contribution ω_neck^B."""
        self.kin._set_state(enc_q, joint_names, enc_qd)
        mujoco.mj_forward(self.kin.model, self.kin.data)                  # need cdof for the body Jacobian
        jacr = np.zeros((3, self.kin.model.nv))
        mujoco.mj_jacBody(self.kin.model, self.kin.data, None, jacr, self.head_bid)
        return jacr @ self.kin.data.qvel


# ---------------------------------------------------------------- feature tracking (KLT + FB check)
_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class FeatureTracker:
    """Pyramidal Lucas-Kanade tracking with forward-backward validation + keyframe refill. Keeps
    persistent integer track IDs so the anchor estimator can match features across keyframes."""

    def __init__(self, width=640, height=480, budget=150, min_dist=20.0, fb_thresh=0.7, border=8):
        self.w, self.h = width, height
        self.budget, self.min_dist = budget, min_dist
        self.fb_thresh, self.border = fb_thresh, border
        self.prev = None
        self.pts = np.empty((0, 2), np.float32)
        self.ids = np.empty(0, np.int64)
        self._next = 0

    def _refill(self, gray):
        need = self.budget - len(self.pts)
        if need <= 0:
            return
        mask = np.full(gray.shape, 255, np.uint8)
        for p in self.pts:                                # keep new corners clear of live tracks
            cv2.circle(mask, (int(p[0]), int(p[1])), int(self.min_dist), 0, -1)
        new = cv2.goodFeaturesToTrack(gray, maxCorners=int(need), qualityLevel=0.01,
                                      minDistance=self.min_dist, blockSize=3, mask=mask)
        if new is None:
            return
        new = new.reshape(-1, 2).astype(np.float32)
        cv2.cornerSubPix(gray, new, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        ids = np.arange(self._next, self._next + len(new))
        self._next += len(new)
        self.pts = np.vstack([self.pts, new]).astype(np.float32)
        self.ids = np.concatenate([self.ids, ids])

    def process(self, gray, keyframe=False):
        """Track into `gray`; refill on keyframes. Returns {track_id: (u,v)} of live features."""
        if self.prev is not None and len(self.pts):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts, None, **_LK)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, nxt, None, **_LK)
            fb = np.linalg.norm(self.pts - back, axis=1)
            b = self.border
            good = ((st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_thresh)
                    & (nxt[:, 0] > b) & (nxt[:, 0] < self.w - b)
                    & (nxt[:, 1] > b) & (nxt[:, 1] < self.h - b))
            self.pts = nxt[good].astype(np.float32)
            self.ids = self.ids[good]
        if keyframe:
            self._refill(gray)
        self.prev = gray
        return {int(i): p.copy() for i, p in zip(self.ids, self.pts)}


# ---------------------------------------------------------------- robust two-view refinement
def _t_basis(t):
    """3x2 orthonormal basis of the tangent plane of the unit sphere at t (for the 2-dof t-direction)."""
    a = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(t, a)
    b1 /= np.linalg.norm(b1)
    return np.stack([b1, np.cross(t, b1)], axis=1)


def _sampson_px(R_cv, t, xa, xj, f):
    """Sampson (first-order geometric) epipolar residual per correspondence, scaled to ~pixels.
    xa, xj: (N,3) normalized homogeneous coords; convention x_jᵀ E x_a = 0 with E = [t]× R_cv."""
    Fa = xa @ (skew(t) @ R_cv).T
    Fb = xj @ (skew(t) @ R_cv)
    num = np.sum(xj * Fa, axis=1)
    den = np.sqrt(Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Fb[:, 0] ** 2 + Fb[:, 1] ** 2) + 1e-12
    return f * num / den


def _refine_rt(R0_cv, t0, xa, xj, f, iters=25, huber_px=1.0):
    """IRLS Gauss-Newton on the 5-dof (R ∈ SO(3), t ∈ S²) manifold, minimizing Huber-weighted Sampson
    error over ALL inlier correspondences. This is the accuracy step the minimal solver lacks: RANSAC's
    winning essential matrix is an exact fit to ONE 5-point sample, so its rotation error is set by 5
    noisy pixels no matter how many inliers vote for it (~5-7 mrad here); refitting on the full inlier
    set recovers the √N averaging (~1-2 mrad). Translation stays a modeled nuisance (never output):
    at 0.4-0.9 m walking baseline the parallax is first-order, so rotation-only fits are badly biased,
    while freezing t at truth is near-optimal — the 5-dof joint refinement is the honest middle."""
    R, t = R0_cv.copy(), np.asarray(t0, float) / (np.linalg.norm(t0) + 1e-12)
    lam = 1e-4
    r = _sampson_px(R, t, xa, xj, f)
    huber = lambda e: np.sum(np.where(np.abs(e) < huber_px, 0.5 * e ** 2,
                                      huber_px * (np.abs(e) - 0.5 * huber_px)))
    cost, eps = huber(r), 1e-6
    for _ in range(iters):
        B = _t_basis(t)
        w = np.ones_like(r)
        big = np.abs(r) > huber_px
        w[big] = huber_px / np.abs(r[big])
        J = np.zeros((len(r), 5))
        for k in range(3):                                 # FD Jacobian: 5 params, N≈50 — negligible cost
            dw = np.zeros(3); dw[k] = eps
            J[:, k] = (_sampson_px(R @ exp_so3(dw), t, xa, xj, f) - r) / eps
        for k in range(2):
            tp = t + eps * B[:, k]
            J[:, 3 + k] = (_sampson_px(R, tp / np.linalg.norm(tp), xa, xj, f) - r) / eps
        A = J.T @ (w[:, None] * J) + lam * np.eye(5)       # LM damping: t is unobservable at low parallax
        try:
            d = -np.linalg.solve(A, J.T @ (w * r))
        except np.linalg.LinAlgError:
            break
        R_n = R @ exp_so3(d[:3])
        t_n = t + B @ d[3:5]
        t_n /= np.linalg.norm(t_n)
        r_n = _sampson_px(R_n, t_n, xa, xj, f)
        c_n = huber(r_n)
        if c_n < cost:
            R, t, r, cost, lam = R_n, t_n, r_n, c_n, max(lam * 0.5, 1e-7)
            if np.linalg.norm(d) < 1e-9:
                break
        else:
            lam *= 10.0
            if lam > 1e3:
                break
    return R, t


# ---------------------------------------------------------------- anchored relative rotation
class AnchorRotationEstimator:
    """Camera rotation R_{C_A→C_j} from anchor keyframe A to current keyframe j: 5-pt RANSAC for
    init + outlier gating ONLY, then a robust IRLS refit of (R, t̂) on all inliers (translation is
    a modeled nuisance, discarded — we only want rotation). Re-anchor when shared tracks run out.

    Optional R_prior (gyro+neck-FK relative rotation) is a LOOSE accept/reject gate (default 30 mrad
    ≈ 20σ of the refined estimate): it kills the planar twisted-pair branch flips the coplanar floor
    induces, but never enters the cost — the accepted value is a stationary point of the purely
    visual objective, so the measurement stays independent of the IMU the filter integrates."""

    def __init__(self, cam: CameraModel, min_inliers=25, min_ratio=0.5, refine=True,
                 prior_gate=0.030, sampson_gate_px=1.5):
        self.cam, self.K = cam, cam.K
        self.min_inliers, self.min_ratio = min_inliers, min_ratio
        self.refine, self.prior_gate, self.sampson_gate_px = refine, prior_gate, sampson_gate_px
        self.anchor_t = None
        self.anchor_uv: dict = {}

    def set_anchor(self, t, tracks):
        self.anchor_t = t
        self.anchor_uv = {i: np.asarray(uv, np.float32) for i, uv in tracks.items()}

    def _norm(self, uv):
        c = self.cam
        return np.column_stack([(uv[:, 0] - c.cx) / c.fx, (uv[:, 1] - c.cy) / c.fy, np.ones(len(uv))])

    def estimate(self, t, tracks, R_prior=None):
        """Measurement dict for anchor→t (R = R_{C_A C_j}), or None if too few shared / degenerate /
        outside the (optional) prior gate."""
        shared = [i for i in tracks if i in self.anchor_uv]
        if len(shared) < self.min_inliers:
            return None
        pa = np.array([self.anchor_uv[i] for i in shared], np.float32)
        pj = np.array([tracks[i] for i in shared], np.float32)
        E, mask = cv2.findEssentialMat(pa, pj, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            return None
        _, R_cv, t_cv, mask = cv2.recoverPose(E, pa, pj, self.K, mask=mask)
        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.min_inliers or inliers / len(shared) < self.min_ratio:
            return None
        m = mask.ravel().astype(bool)
        if self.refine:                                    # two-pass IRLS: refit → re-gate inliers → refit
            xa, xj = self._norm(pa), self._norm(pj)
            R_cv, t_dir = _refine_rt(R_cv, t_cv.ravel(), xa[m], xj[m], self.cam.fx)
            keep = np.abs(_sampson_px(R_cv, t_dir, xa, xj, self.cam.fx)) < self.sampson_gate_px
            if keep.sum() >= self.min_inliers:
                R_cv, t_dir = _refine_rt(R_cv, t_dir, xa[keep], xj[keep], self.cam.fx)
                m, inliers = keep, int(keep.sum())
        R = R_cv.T
        if R_prior is not None and self.prior_gate is not None:
            if np.linalg.norm(log_so3(R.T @ R_prior)) > self.prior_gate:
                return None                                # branch flip / gross failure: coast on IMU+contact
        parallax = float(np.median(np.linalg.norm(pj[m] - pa[m], axis=1)))
        return dict(R=R, t_anchor=self.anchor_t, t=t, n_shared=len(shared),
                    n_inliers=inliers, ratio=inliers / len(shared), parallax=parallax)

    @property
    def n_shared_live(self):
        return len(self.anchor_uv)
