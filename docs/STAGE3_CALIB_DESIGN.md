# Stage 3 — Online camera–IMU self-calibration (build spec)

> Design, with **every new Jacobian block pre-verified by finite differences during
> the design** (results in §3). In-situ self-calibration: while walking the loop, the robot estimates its own
> camera-mount **rotation extrinsic** `δφ_HC` and camera–IMU **time offset** `td` online — no rig, only the
> onboard sensors Stage 2 already uses. Validation = inject a *known* mount error + time offset into the sim
> and recover them. Convention-correct for `R_true = R̂·Exp(δθ)` (body-frame error) + the Stage-2 anchored
> relative-rotation update.

> **Outcome (post-build):** 3.0 (rotation extrinsic) landed as designed — the observable axis recovers
> online (~87%, funnel σ → 1.1 mrad) and the anisotropy is honestly reported. 3.1 (td): the gait's rate
> modulation proved too weak against the constant-rate-loop degeneracy in practice, so td ships **frozen**
> (machinery FD-verified at t̂d = 20 ms); a rate-varying trajectory (neck_scan / speed steps) is the pending
> excitation. See SELF_REVIEW.md §1.4.

## 0. Scope
**Core (calibrate online):** `δφ_HC` (3) camera-mount rotation error + `td` (1) camera time offset → error
state 21 → **25**. Both enter the *existing* rotation-only vision measurement with non-zero FD-checkable
Jacobians, are observable on the 24 s/360° loop, recoverable from injected truth, and visibly damage the
Stage-2 estimator when wrong (real before/after). **Order:** 3.0 rotation → 3.1 time-offset → joint.
**Defer:** camera translation `p_HC` (∂r/∂p_HC ≡ 0 in a rotation-only measurement — needs a reprojection
residual; front-end already persists bearings for it); IMU scale/misalignment (excitation-poor here);
rolling-shutter/latency → Stage 4.

## 1. State augmentation — `CalibVioESKF(VioESKF)` in new `src/humanoid_asimov/calib.py`
Subclass; **never modify** `ESKF`/`VioESKF` (stay byte-identical + reproducible). Error layout:
```
δx(25) = [δp 0:3 | δv 3:6 | δθ 6:9 | δb_g 9:12 | δb_a 12:15 | δb_c 15:18 |
          δφ_HC 18:21 | δtd 21 | δθ_A 22:25]   ← calib BEFORE the clone, so clone stays the tail
```
Nominal calib in the filter: `R_HC` (mount est., head←cam CV), `td` (offset est., s). `clone()` keeps its
truncate-then-append pattern (`P[:22,:22]` then block-append rows/cols 6:9 — this auto-carries δθ↔δφ, δθ↔δtd
cross-covariances into the clone). **Contact update needs NO change** (it sizes H by `self.P.shape[0]`).
Propagation: calib states are **constants** → `F_aug = blkdiag(F₁₈, I₄, I₃)`, `Q_aug = blkdiag(Q₁₈, Q_cal, 0)`,
**`Q_cal = 0`** for the demo (injected truth is constant; covariance collapse onto it = the success signal).
Frozen mode (`est_extrinsic/est_td=False`) = zero that P₀/Q → reproduces `VioESKF` to ~1e-9 (regression test).
Priors: **σ_φ0 = 5°/axis, σ_td0 = 30 ms** (must *cover* the injected error, else the χ² gate starves the
calibration). Inject: `R_HC ← R_HC·Exp(dx[18:21])`, `td ← td + dx[21]`, `R_A ← R_A·Exp(dx[22:25])`.

## 2. Measurement model + Jacobian (all FD-verified, §3)
Camera-frame right mount error `R_HC,true = R̂_HC·Exp(δφ_HC)` (enters *both* keyframes identically:
`R_BC,k = R̂_BC,k·Exp(δφ)`). Time offset: image stamped `t` is captured at `t+td`; `R_WC(t+τ)=R_WC(t)·Exp(ω_C τ)`.
```
ĥ    = R̂_BC,Aᵀ (R_Aᵀ R̂) R̂_BC,j ,  R̂_BC,k = R_BH,k·R̂_HC        (R_BH from neck FK)
ĥ_td = Exp(−ω̂_CA·t̂d) · ĥ · Exp(ω̂_Cj·t̂d)                       (first-order capture-time compensation)
r    = Log(ĥ_tdᵀ · R̃)                                          (R̃ = R_cvᵀ from the front-end)
```
With `Â = R̂ᵀR_A`, `R_CB,j = R̂_BC,jᵀ`, **`C = Exp(−ω̂_Cj·t̂d)`** (≈ I once converged):

| block | cols | value |
|---|---|---|
| δθ | 6:9 | `C · R_CB,j` |
| **δφ_HC** | 18:21 | **`C · (I − ĥᵀ)`** |
| **δtd** | 21 | **`ω̂_Cj − C·ĥᵀ·ω̂_CA`** |
| δθ_A | 22:25 | `−C · R_CB,j · Â` |
| δp,δv,δb_g,δb_a,δb_c | rest | `0` |

**Camera angular rate** `ω̂_C = R̂_BCᵀ·((gyro_m − b̂_g) + ω_neck^B)` — base rate (bias-corrected) + head-vs-base
rate from FK. New helpers on `HeadCamExtrinsics` (owns `kin`): `head_pose(enc_q,jn)→(R_BH,p_BH)` (factor out
of `compute`) and `head_angvel_base(enc_q,enc_qd,jn)` via `mj_jacBody` (rotational, base DOFs zero). Keep C:
omitting it at `t̂d=20 ms` leaves 2.7e-2 Jacobian error (harmless to the filter, fatal to a 1e-5 FD gate).

## 3. FD checks — `scripts/check_calib_jacobian.py` (mirror `check_vio_jacobian.py`)
Analytic H from the real code path; perturb the **true-state** side; `r₀=0` assert; per-block max err.
Pre-verified in the design pass (reproduce these):
- δθ/δθ_A/δφ/td vs FD, random state, `t̂d=0`: 6e-10 / 1e-9 / 4e-8 / 5e-8 ✓; same at `t̂d=20 ms`: ≤5e-8 ✓
- negative controls: transposed δφ `(I−ĥ)` → 1.47 (loud fail); C omitted at 20 ms → 2.7e-2
- degenerates: `ĥ=I` ⇒ δφ block=0 (2e-16); constant-ω ⇒ td block=0 (1.5e-16) — the td-observability proof
- zero blocks ≤1e-12; global-yaw gauge `H·δx=0` (3e-16)

## 4. Observability on the loop
- **δφ_HC: 2 axes fast, 1 slow (honest anisotropy).** `(I−ĥᵀ)` kills the component of δφ along the anchor-
  relative rotation *axis* `n_C` (≈ world-up in cam frame, mostly −y with −0.23 z from the 15° pitch; <4°
  spread). Loop content: ~309 mrad/keyframe about `n_C` (commanded yaw) vs ~68 mrad ⊥ (gait ripple) → **⊥ axes
  ≈1 mrad/lap, `n_C` axis ≈5 mrad/lap** (slower, still converges). Report δφ in the `(n_C,⊥)` basis; the largest
  eigvec of final `P_φφ` should align with `n_C` — the filter *knowing which axis it can't learn fast* is the
  consistency exhibit.
- **td: needs rate *variation*, not sustained rate** (corrects the old Stage-2 hook — a smooth turntable makes
  td unobservable, proven by the constant-ω degenerate). The **gait** modulates the camera rate at step
  frequency → measured gain median **1.12 rad/s** → **≈0.3 ms/lap**: td is the *best*-observed state.
- Global yaw gauge preserved (H·δx_gauge=0). td vs b_g separable (td = gait-freq oscillation, b_g = secular).

## 5. Injection-for-validation (the demo) — `scripts/run_calib.py`
Principle: **the sim renders the truth; the estimator keeps the nominal.** Truth only scores, never feeds.
- **Rotation:** parameterize the camera mount in `scene.build_spec/build_model` + `walk.simulate` (kwargs
  `head_cam_quat/head_cam_pos`, default to the constants). Driver computes `q_true = HEAD_CAM_QUAT ⊗
  quat_from_rotvec(F·δφ_true)`, `F=diag(1,−1,−1)` ⇔ `R_HC,true = R_HC·Exp(δφ_true)` (machine-exact; scoring
  metric `Log(R̂_HCᵀ R_HC,true)`). Inject `δφ_true ≈ (1.5°,−2.5°,1.0°)`, ‖·‖≈3.1° (≤5°). `RobotKinematics`
  carries no camera + `HeadCamExtrinsics` reads the untouched constants → nominal side stays nominal.
- **Time offset:** zero sim change — `log["frame_t"] -= TD_TRUE` in the driver (that *is* td). `TD_TRUE=+20 ms`
  (10 steps, exact on the 2 ms grid; ±1 ms recovery floor). Existing `fidx=argmin|t−frame_t|` feeds frames at
  their wrong stamps.
- **Driver configs (same seed/loop, `ImuNoise()`):** A clean+`VioESKF` (Stage-2 baseline), B injected+plain
  `VioESKF` (damage — most vision gated → drift decays to ESKF level), C injected+`CalibVioESKF` (recover +
  restore). Fusion-loop diff: `update_rel_rot(R_meas, R_BH_j, gyro[k], w_neck)` + `clone(R_BH_j, gyro[k], w_neck)`.
  Log: `Log(R̂_HCᵀ R_HC,true)`, `t̂d−TD_TRUE`, `diag(P)[18:22]`, vision NIS + accept flag. Plots: per-axis δφ̂
  convergence (cam + `(n_C,⊥)` basis) w/ ±3σ funnels + truth; t̂d trace; gate-accept timeline; A/B/C tracks.

## 6. Plan + pitfalls
**3.0** (rotation, `est_td=False`, C=I): calib.py + vision.py helpers + scene/walk kwargs → **Gate 1** FD-check
(all §3) → **Gate 2** frozen-regression = VioESKF → **Gate 3** recover injected δφ → **Gate 4** A/B/C table.
**3.1** (time-offset then joint): enable td + ĥ_td/C, stamp-shift injection, FD at `t̂d≠0`, td-only then
**joint** (inject both, estimate both — the headline). Check `b_g,z` err stays ≤0.3 mrad/s; td-sign unit test.
**Pitfalls:** (1) GL/CV `F·δφ` conversion (the one flip — round-trip guard). (2) **Store `R_BH_A`, recompose
both extrinsic sides with the live `R̂_HC` every update** — freezing `R_BC,A` at clone breaks the common-
perturbation structure (δφ would enter one-sided, faking 3-axis observability). (3) Capture `ω̂_CA` in `clone()`
(anchor instant). (4) Gate starvation — priors must cover the injected error; log acceptance; first anchor at
99.9% gate. (5) One td sign convention everywhere (`t_capture=t_stamp+td`) + a directional unit test.
(6) Keep C (FD gate). (7) `Q_cal=0` (a random walk masks non-convergence). (8) td quantization floor 1 ms.

## 7. Success metrics (M=10 seeds, medians; `δφ_true`≈3.1°, `td_true`=20 ms, 24 s lap)
FD gates ≤1e-5 (zero-blocks ≤1e-12, degenerates exact, negative controls loud); frozen-regression ~1e-9;
`e_φ` ⊥-axes ≤2 mrad / `n_C` axis ≤10 mrad at lap end (from 52 mrad); `e_td` ≤2 ms; error inside ±3σ ≥95% for
t>2 s; σ contraction (φ⊥ 87→≤3, φ∥ 87→≤12, td 30→≤1.5 ms); **A/B/C: C close-gap ≤1.3× A + ≥2× better than B,
`b_g,z` ≤0.3 mrad/s**; contact NIS unchanged from Stage 1/2. **Headline: a ~3° mount error + 20 ms time offset
recovered to ~0.1° (observable plane) and ~1 ms, online, while walking one loop — restoring the estimator to
within ~30% of its perfectly-calibrated accuracy** from a "before" where vision was gated into uselessness.
