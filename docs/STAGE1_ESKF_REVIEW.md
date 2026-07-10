# Stage 1 ESKF — review + NEES findings

> A rigorous review of the contact-aided ESKF + the NEES/NIS harness results. The math is
> correct; the point estimate is the deliverable and it is good; the *consistency* is the
> remaining work, and its cause is diagnosed here.

## ✅ Slip-bias fix — DONE (2026-07-08)
Added a shared slip-velocity bias `b_c` (→ 18-dim ESKF): the planted foot's world velocity is modelled as
`R·b_c` (not exactly 0) and estimated online. Result (`run_nees.py`, M=10):
- **`b_g` de-biased → `[+0.08, +0.11, -0.15]` mrad/s** (was `[-0.6, +2.5, -0.6]`): the contact-point
  systematic no longer leaks into the gyro bias. Learned `b_c ≈ [-5, +4, +13]` mm/s (matches the predicted
  slip). **Drift 0.73% → 0.32%** — now *beats* gyro dead-reckoning (0.54%). z-err 4.1→3.4 cm.
- **Covariance is still over-confident** (per-block NEES optimistic; contact NIS runs hot (mean 6.3, median 1.2) with lag-1 autocorr
  ≈ +0.85 — the residual is *colored*, not white). Root cause: the within-stance contact error is
  correlated (heel→toe CoP motion) and the lateral slip flips sign per foot. A diagonal-Q ESKF cannot model
  colored noise, and a **per-foot `b_c` was tried but its fore-aft component aliases with base velocity and
  diverges** (each foot is observed only ~half the time). No Q/Rc tuning whitens colored noise.
- **The estimate is the deliverable and it's good** (b_g honest, drift 0.32%); the remaining *consistency*
  work is a right-InEKF with foot-position contact states (position-level contact + trajectory-independent
  Jacobians → whitens the residual + honest covariance) plus stand→walk→stand segments so `b_a` becomes
  observable. Parked as Stage 1.5; then → Stage 2 (monocular VIO).

## Verdict
- **ESKF math verified correct** — every Jacobian (F, contact-H, gravity-H), the injection, and the
  Joseph form checked by finite differences (~1e-7) against the code's own propagation. No math bugs.
- **Drift is yaw-limited** — quantified: attitude explains 85% of the ESKF error (corr 0.998); yaw error
  6.5 mrad ≈ raw-gyro 6.8 mrad; `b_g z` is *provably* unlearnable in 8 s by any filter (InEKF included).
  Yaw-about-gravity + absolute position are the classic 4-dim unobservable subspace of IMU+contact.
- **Honest framing:** the ESKF *doesn't beat* dead reckoning on drift (0.73% vs 0.54%) — it fuses at the
  velocity level (position = ∫v̂, so slip random-walks it) while leg-odom's anchor is a position-level
  constraint. Its value = velocity + covariance + online bias, and the gap to the 0.03% kinematic floor
  is exactly the Stage-2 (VIO) motivation. Say this, don't oversell.

## Three defects (found + partly addressed)
1. **Q was mis-tuned** — filter white-noise ×134/×119 too high, bias-RW ×4 too low vs the injected
   LSM6DSOX noise. Correct values (dt=2 ms): sig_a≈6.7e-4, sig_g≈4.5e-5, sig_bg=2e-4, sig_ba=4e-3.
   Adopt these *together with* the model fix (alone, the matched Q over-confidences — proven below).
2. **Contact error is systematic, not white** — the stance **ankle site is not at zero velocity**
   (sits 9.4 mm above ground; the foot rolls about a *moving* contact point → ~[-3.9, ±3.2, -6.9] mm/s
   body-frame). Via the 0.61 m z-lever this pulls **`b_g y` the wrong way (+2.53 mrad/s, sign-flipped)**
   and drives a hidden **~4 cm z drift**. NIS lag-1 autocorr +0.8 confirms non-white residual. Fixed by the
   shared `b_c` state above (the faithful physical alternative — a force-weighted CoP from an ankle F/T
   sensor — avoids a GT leak but was not needed once `b_c` landed).
3. **Metrics hid the z drift** — fixed (`run_estimator.py` now reports 3D-ATE + z-err).

## NEES/NIS harness results (`scripts/run_nees.py`, M=10 seeds)
Reproduces the diagnosis. Two configs ship: **BEFORE** (15-state, inflated Q, slip-bias off) and
**AFTER** (the delivered 18-state — Q matched to the injected LSM6DSOX noise + the shared slip-velocity
bias `b_c`). Per-block ANEES (consistent ≈ dof; 3-dof band [1.68, 4.70], all-15 band [11.80, 18.58]):

| block (dof) | BEFORE | AFTER |
|---|---|---|
| pos (3)  | 3.19 ok            | 2.75 ok |
| vel (3)  | 6.83 optimistic    | 61.35 optimistic |
| tilt (2) | 0.64 conservative  | 67.61 optimistic |
| yaw (1)  | 0.19 conservative  | 0.29 conservative |
| b_g (3)  | 2.44 ok            | 12.71 optimistic |
| b_a (3)  | 1.32 conservative  | 111.05 optimistic |
| all (15) | 16.60 (false pass) | 614.58 optimistic |

Contact NIS (dof 3): BEFORE mean 1.18 / median 0.15 / 38% in band / lag-1 +0.80; AFTER mean 6.31 /
median 1.22 / 69% in band / lag-1 +0.85.

- **BEFORE's all-15 16.6 is a false pass** — it lands in-band only because the optimistic velocity block
  and the conservative tilt/yaw/`b_a` blocks cancel in the lumped statistic; the per-block split exposes it.
- **The slip-bias fix (AFTER) buys the point estimate, not the covariance.** It de-biases `b_g`
  (`[-0.63, +2.53, -0.58]` → `[+0.08, +0.11, -0.15]` mrad/s) and cuts drift 0.73 → 0.32% — but its
  covariance is openly *optimistic*. Once Q is matched to the injected noise the colored contact residual
  (lag-1 +0.85; NIS mean 6.3 ≫ the ~3 a white dof-3 residual gives) can no longer hide, so vel / tilt /
  `b_g` / `b_a` and the lumped all-15 (~615) all read optimistic. Tightening Q does **not** rescue
  consistency; a diagonal-Q EKF cannot represent colored noise.
- **Lesson: consistency here is a modeling problem, not a tuning problem.** The measurement model is the
  culprit; the structural fix is a right-invariant EKF with foot-position contact states (position-level
  contact + trajectory-independent Jacobians → whitened residual, honest covariance), plus
  stand→walk→stand segments so `b_a` becomes observable. Parked as Stage 1.5 — Stage 2 went first (the
  drift was yaw-limited), and its VIO update is gauge-safe, carrying the contact residual untouched.
