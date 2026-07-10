"""Stage 3 — finite-difference check of the calibration Jacobian H (3x25).

Analytic H comes from the real code path (`CalibVioESKF._calib_rH`); the FD reference perturbs the TRUE
state (the measurement side) with the prediction held fixed. The δθ / δφ_HC / δtd / δθ_A blocks must match
to ~1e-6, the other blocks must be zero, and a global-yaw gauge shift must give H·δx = 0 — checked at both
t̂d = 0 and t̂d = 20 ms (the C-factored form).

    .venv/bin/python scripts/check_calib_jacobian.py
"""
import numpy as np

from humanoid_asimov.calib import CalibVioESKF
from humanoid_asimov.estimator import RobotKinematics, exp_so3, log_so3

KIN = RobotKinematics()


def _rr(rng, s=1.0):
    return exp_so3(rng.uniform(-s, s, 3))


def check(td_hat):
    rng = np.random.default_rng(3)
    e = CalibVioESKF(KIN, np.zeros(3), np.eye(3), R_HC0=_rr(rng), td0=td_hat)
    e.R, e.R_A = _rr(rng), _rr(rng)
    e.bg = rng.uniform(-0.01, 0.01, 3)
    R_BH_j, R_BH_A = _rr(rng), _rr(rng)
    gyro_j = rng.uniform(-0.5, 0.5, 3)
    w_neck_j, w_neck_A = rng.uniform(-0.2, 0.2, 3), rng.uniform(-0.2, 0.2, 3)
    gyro_A = rng.uniform(-0.5, 0.5, 3)
    e.R_BH_A = R_BH_A
    e.w_CA = (R_BH_A @ e.R_HC).T @ ((gyro_A - e.bg) + w_neck_A)
    e.cloned = True

    R_BC_j = R_BH_j @ e.R_HC
    w_Cj, w_CA = R_BC_j.T @ ((gyro_j - e.bg) + w_neck_j), e.w_CA
    _, H = e._calib_rH(np.eye(3), R_BH_j, gyro_j, w_neck_j)         # analytic (measurement-independent)

    h_hat = (R_BH_A @ e.R_HC).T @ (e.R_A.T @ e.R) @ R_BC_j
    h_td = exp_so3(-w_CA * td_hat) @ h_hat @ exp_so3(w_Cj * td_hat)

    def resid(dth, dthA, dphi, dtd):                               # z = h(x_true), prediction ĥ_td fixed
        RHC = e.R_HC @ exp_so3(dphi)
        RWCj = (e.R @ exp_so3(dth)) @ R_BH_j @ RHC @ exp_so3(w_Cj * (td_hat + dtd))
        RWCA = (e.R_A @ exp_so3(dthA)) @ R_BH_A @ RHC @ exp_so3(w_CA * (td_hat + dtd))
        return log_so3(h_td.T @ (RWCA.T @ RWCj))

    z3 = np.zeros(3)
    r0 = resid(z3, z3, z3, 0.0)
    assert np.linalg.norm(r0) < 1e-9, f"r0 {r0}"
    eps, Hfd = 1e-6, np.zeros((3, 25))
    for k in range(3):
        d = np.zeros(3); d[k] = eps
        Hfd[:, 6 + k] = (resid(d, z3, z3, 0.0) - r0) / eps
        Hfd[:, 22 + k] = (resid(z3, d, z3, 0.0) - r0) / eps
        Hfd[:, 18 + k] = (resid(z3, z3, d, 0.0) - r0) / eps
    Hfd[:, 21] = (resid(z3, z3, z3, eps) - r0) / eps

    print(f"-- t̂d = {td_hat*1e3:.0f} ms --")
    ok = True
    for nm, sl in (("δθ", slice(6, 9)), ("δφ_HC", slice(18, 21)),
                   ("δtd", slice(21, 22)), ("δθ_A", slice(22, 25))):
        err = float(np.abs(Hfd[:, sl] - H[:, sl]).max())
        ok = ok and err < 1e-5
        print(f"   {nm:6} analytic-vs-FD max err {err:.2e}")
    zero = float(np.abs(H[:, list(range(0, 6)) + list(range(9, 18))]).max())
    dg = np.zeros(25); ez = np.array([0.0, 0.0, 1.0])
    dg[6:9], dg[22:25] = e.R.T @ ez, e.R_A.T @ ez
    gauge = float(np.abs(H @ dg).max())
    print(f"   zero blocks max |H| {zero:.2e} | global-yaw gauge |H·δx| {gauge:.2e}")
    return ok and zero < 1e-12 and gauge < 1e-10


def main():
    ok = all(check(td) for td in (0.0, 0.020))
    print("\n" + ("[OK] CALIBRATION JACOBIAN VERIFIED — safe to build on"
                  if ok else "[X] MISMATCH — fix before proceeding"))


if __name__ == "__main__":
    main()
