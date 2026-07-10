"""Stage 2.2 — finite-difference check of the VIO relative-rotation Jacobian H (3x21).

Sets the measurement equal to the prediction (so residual r0 = 0 → the operating point of the update),
then perturbs each error dimension by ε and compares (r(ε)-r0)/ε to the analytic H. The δθ and δθ_A
blocks must match to ~1e-6; all other blocks (δp, δv, δb_g, δb_a, δb_c) must be identically zero.

    .venv/bin/python scripts/check_vio_jacobian.py
"""
import numpy as np

from humanoid_asimov.estimator import RobotKinematics, exp_so3, log_so3
from humanoid_asimov.vio import VioESKF


def _rand_rot(rng, scale=1.0):
    return exp_so3(rng.uniform(-scale, scale, 3))


def main():
    rng = np.random.default_rng(1)
    kin = RobotKinematics()
    e = VioESKF(kin, np.zeros(3), np.eye(3))
    e.R = _rand_rot(rng)
    e.R_A = _rand_rot(rng)
    e.R_BC_A = _rand_rot(rng)
    e.cloned = True
    R_BC_j = _rand_rot(rng)

    # analytic H (independent of the measurement) from the actual code path
    _, H = e._vis_rH(np.eye(3), R_BC_j)

    # FD reference: perturb the TRUE state (the measurement z = h(x_true) side), holding the prediction
    # ĥ fixed — this is the EKF's H = ∂r/∂δx with δx = x_true − x̂ (right error x_true = x̂·Exp(δx)).
    R_hat, RA_hat = e.R.copy(), e.R_A.copy()
    h_hat = e.R_BC_A.T @ (RA_hat.T @ R_hat) @ R_BC_j

    def resid(dth, dthA):
        z = e.R_BC_A.T @ ((RA_hat @ exp_so3(dthA)).T @ (R_hat @ exp_so3(dth))) @ R_BC_j
        return log_so3(h_hat.T @ z)

    r0 = resid(np.zeros(3), np.zeros(3))
    assert np.linalg.norm(r0) < 1e-9, f"r0 not zero: {r0}"
    eps = 1e-6
    Hfd = np.zeros((3, 21))
    for k in range(3):
        d = np.zeros(3); d[k] = eps
        Hfd[:, 6 + k] = (resid(d, np.zeros(3)) - r0) / eps
        Hfd[:, 18 + k] = (resid(np.zeros(3), d) - r0) / eps

    err_th = np.abs(Hfd[:, 6:9] - H[:, 6:9]).max()
    err_thA = np.abs(Hfd[:, 18:21] - H[:, 18:21]).max()
    zero_cols = np.abs(H[:, list(range(0, 6)) + list(range(9, 18))]).max()
    print(f"H δθ   block  (cols 6:9)   analytic-vs-FD max err = {err_th:.2e}")
    print(f"H δθ_A block  (cols 18:21) analytic-vs-FD max err = {err_thA:.2e}")
    print(f"H zero blocks (δp,δv,δb_g,δb_a,δb_c) max |H| = {zero_cols:.2e}")
    ok = err_th < 1e-5 and err_thA < 1e-5 and zero_cols < 1e-12
    print("\n" + ("✓ JACOBIAN VERIFIED — safe to fuse" if ok else "✗ JACOBIAN MISMATCH — fix before fusing"))


if __name__ == "__main__":
    main()
