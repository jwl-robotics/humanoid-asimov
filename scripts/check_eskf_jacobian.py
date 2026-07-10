"""FD-check of the base ESKF Jacobians — the contact-update H (3×18) and the propagation F (18×18).

Mirrors check_vio_jacobian.py / check_calib_jacobian.py: the analytic Jacobian comes from the real code
path (`ESKF._contact_rH`, `ESKF._predict_step`); it is compared to a finite difference of the actual
nonlinear map (states composed/differenced on the manifold, `R` via Exp/Log). The contact H exercises the
b_g lever term R·[p_f]× and the slip-velocity-bias block; F is the full error-state propagation. The
contact-H zero blocks (δp, δb_a) are checked too.

    .venv/bin/python scripts/check_eskf_jacobian.py
"""
import os

import numpy as np

from humanoid_asimov.estimator import ESKF, RobotKinematics, exp_so3, log_so3

KIN = RobotKinematics()
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "walk_dataset.npz"))


def _state(e):
    return [e.p.copy(), e.v.copy(), e.R.copy(), e.bg.copy(), e.ba.copy(), e.bc.copy()]


def _set(e, s):
    e.p, e.v, e.R, e.bg, e.ba, e.bc = [np.array(x, float) for x in s]


def _boxminus(a, b):                                        # a ⊖ b : two state tuples → 18-vec error
    return np.concatenate([a[0] - b[0], a[1] - b[1], log_so3(b[2].T @ a[2]),
                           a[3] - b[3], a[4] - b[4], a[5] - b[5]])


def _oplus(s, dx):                                          # s ⊕ dx (error-state injection)
    return [s[0] + dx[0:3], s[1] + dx[3:6], s[2] @ exp_so3(dx[6:9]),
            s[3] + dx[9:12], s[4] + dx[12:15], s[5] + dx[15:18]]


def _rand_state(rng):
    return [rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3), exp_so3(rng.uniform(-0.6, 0.6, 3)),
            rng.uniform(-0.01, 0.01, 3), rng.uniform(-0.05, 0.05, 3), rng.uniform(-0.02, 0.02, 3)]


def check_F():
    rng = np.random.default_rng(1)
    s0 = _rand_state(rng)
    gyro = rng.uniform(-0.6, 0.6, 3)
    accel = rng.uniform(-2, 2, 3) + np.array([0.0, 0.0, 9.81])
    dt = 0.002
    e = ESKF(KIN, s0[0], s0[2]); _set(e, s0)
    F, _ = e._predict_step(gyro, accel, dt)                 # analytic F at the nominal state
    s_next = _state(e)
    eps, Ffd = 1e-6, np.zeros((18, 18))
    for i in range(18):
        dx = np.zeros(18); dx[i] = eps
        ep = ESKF(KIN, s0[0], s0[2]); _set(ep, _oplus(s0, dx))
        ep._predict_step(gyro, accel, dt)
        Ffd[:, i] = _boxminus(_state(ep), s_next) / eps
    D = Ffd - F
    dropped = max(float(np.abs(D[0:3, 6:9]).max()), float(np.abs(D[0:3, 12:15]).max()))
    D[0:3, 6:9] = 0.0; D[0:3, 12:15] = 0.0          # F omits the two O(dt²) δp cross-terms by design (½·a·dt²)
    err = float(np.abs(D).max())
    print(f"  F (propagation, 18×18)   modeled terms max err {err:.2e} · "
          f"dropped O(dt²) δp cross-terms {dropped:.1e} (= ½|a|dt², a standard first-order omission)")
    return err < 1e-6


def check_contactH():
    d = np.load(DATA, allow_pickle=True)
    jn = list(d["joint_names"])
    k = next(i for i in range(len(d["contact"])) if d["contact"][i][0] or d["contact"][i][1])
    foot = "left" if d["contact"][k][0] else "right"
    p_f, v_f_joint = KIN.foot_pos_vel_base(d["enc_q"][k], d["enc_qd"][k], jn)[foot]
    rng = np.random.default_rng(2)
    s0 = _rand_state(rng); gyro = rng.uniform(-0.6, 0.6, 3)
    e = ESKF(KIN, s0[0], s0[2]); _set(e, s0)
    innov0, H = e._contact_rH(p_f, v_f_joint, gyro)
    h0 = -innov0                                            # measurement h(x) = v + R·u = −innov
    eps, Hfd = 1e-6, np.zeros((3, 18))
    for i in range(18):
        dx = np.zeros(18); dx[i] = eps
        ep = ESKF(KIN, s0[0], s0[2]); _set(ep, _oplus(s0, dx))
        innov_p, _ = ep._contact_rH(p_f, v_f_joint, gyro)
        Hfd[:, i] = (-innov_p - h0) / eps
    err = float(np.abs(Hfd - H).max())
    zero = float(np.abs(H[:, list(range(0, 3)) + list(range(12, 15))]).max())   # δp, δb_a blocks
    print(f"  contact H (3×18)         analytic-vs-FD max err {err:.2e} | zero blocks (δp, δb_a) {zero:.2e}")
    return err < 1e-6 and zero < 1e-12


def main():
    print("ESKF Jacobian FD-check:")
    ok = check_F() and check_contactH()
    print("\n" + ("[OK] ESKF JACOBIANS VERIFIED (propagation F + contact H)"
                  if ok else "[X] MISMATCH — fix before trusting the filter"))


if __name__ == "__main__":
    main()
