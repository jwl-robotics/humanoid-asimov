# Stage 2 — Monocular VIO design (build spec)

> Design. Add the head camera to the 18-state ESKF to observe the one thing
> IMU + leg-odometry cannot — **heading (yaw)** — and cut the loop close-gap. Convention-correct for our
> `R_true = R̂·Exp(δθ)` (body-frame δθ) ESKF.

## Architecture — loosely-coupled, anchored **relative-rotation** VIO
A KLT front-end tracks features at 30 Hz; every ~0.5 s (keyframe) a robust **2-view relative rotation**
`R̃_{C_A→C_j}` is estimated between the current keyframe and a persistent **anchor** keyframe (re-anchored
every ~2–4 s). That SO(3) measurement enters the ESKF as a **3-dof update against a cloned rotation state**
(stochastic cloning, 18 → 21 error dims). **No landmarks in the state, no triangulation, no reprojection.**

**Why this, not MSCKF/tightly-coupled:**
1. **The missing information is rotation-only.** Scale + metric velocity/position already come from IMU +
   contact (Stage-1 drift 0.32%); Stage 1 proved position error is 85% attitude-driven + yaw-limited.
   MSCKF's extra product is *translation* info — the one thing we don't need.
2. **Consistency by construction.** The relative-rotation `H` (below) annihilates the global-yaw gauge
   **exactly at any linearization point** → no spurious yaw information, **no FEJ/OC-EKF needed**. Our
   honest-covariance (Stage-1) story survives Stage 2. MSCKF/EKF-SLAM would fight this.
3. **Degeneracy-proof:** translation is discarded entirely → no low-parallax translation failure.
4. **Cheap:** ~150 KLT tracks @30 Hz + one 3-dof update @2 Hz + a 21×21 filter. Whole lap of vision ≈ 3–5 s.
5. **Upgrade path kept:** front-end persists per-keyframe bearings → a Stage-3 MSCKF-lite can bolt on.

**Trade-off (honest):** absolute yaw stays gauge-unobservable (needs a map / loop-closure → deferred);
yaw becomes a slow **anchor-hop random walk** (~3–4 mrad/lap) instead of the IMU bias RW (σ≈13.6 mrad@24 s)
→ a **3–5× drift cut** for ~400 lines.

## The math
**State clone (21-dim error):** `δx = [δp, δv, δθ, δb_g, δb_a, δb_c, δθ_A]`. On clone (keyframe `t_A`): store
nominal `R_A = R̂_WB(t_A)` + anchor FK `R_BC,A`; augment `P` copying rows/cols `6:9` (with cross-covariances)
into `18:21`. Between keyframes `F_aug = blkdiag(F_18, I_3)`, `Q_aug = blkdiag(Q, 0)`. Contact update just
gains 3 zero columns. Re-anchor = drop `18:21`, then clone again.

**Measurement:** `h(x) = R_{C_A C_j} = (R_WB,A·R_BC,A)ᵀ·(R_WB,j·R_BC,j)`; `z = R̃` from the front-end.
OpenCV `recoverPose` returns `R_cv = R_{C_j C_A}` → **`R̃_{C_A C_j} = R_cvᵀ`**.

**Residual + Jacobian** (with `Â = R̂_WB,jᵀ·R_A`, `R_CB,j = R_BC,jᵀ`):
```
ĥ = R_BC,Aᵀ · (R_Aᵀ R̂_WB,j) · R_BC,j ;   r = log_so3(ĥᵀ z)  ∈ R³
H (3×21) = [ 0 | 0 | R_CB,j | 0 | 0 | 0 | −R_CB,j·Â ]
             δp  δv   δθ     bg   ba  bc    δθ_A
```
Sanity: `t_j=t_A ⇒ Â=I ⇒ r=0`. **δp/δv/b_g have ZERO H-blocks** — they're corrected only via `P`
cross-covariances (that is how vision observes `b_g`, incl. the Stage-1-impossible `b_g,z`).

**Update** (mirror `update_contact`): `S=HPHᵀ+R_vis`; `NIS=rᵀS⁻¹r` (log it); χ²(3) gate 11.34; `K=PHᵀS⁻¹`;
inject; Joseph. **Injection must also update the clone nominal:** `R_A ← R_A·exp_so3(δx[18:21])`.
`R_vis = σ_r²·I`, start **σ_r = 2 mrad** (1 px / 342.76 px focal ≈ 2.9 mrad/bearing, ÷√inliers, ×√2–2 for
correlation); let NIS mean + lag-1 autocorr arbitrate.

**Gauge-safety proof:** a global yaw shift γ about `e_z` gives `H·δx ≡ 0` identically (both δθ_j, δθ_A map
through the same γ and cancel) → yaw stays honestly unobservable, no fake information.

**Geometry/conventions (bug-traps):** `K`: `fx=fy=240/tan(35°)=342.76`, `cx=320, cy=240`. MuJoCo cam looks
along **−z**, +y up (GL) → CV flip `F=diag(1,−1,−1)`, `R_HC = quat2mat(HEAD_CAM_QUAT)·F`, `p_HC=(0.10,0,0.13)`.
FK: `R_BC(t)=R_BH(enc_q)·R_HC` via a base-at-origin `RobotKinematics` (neck_pitch_link xpos/xmat are already
base-frame). **Pass all 15 estimator joints to FK** (waist/neck sag under the leg-only gait PD; ignoring them
injects that motion as fake base yaw).

## Front-end spec (OpenCV; KLT not ORB — no relocalization in Stage 2)
- **Detect (keyframes):** `goodFeaturesToTrack(qualityLevel=0.01, minDistance=20, blockSize=3)` into an
  **8×6 bucket grid** (~4/cell), refill only sparse cells; budget 120–150 live tracks; `cornerSubPix`.
- **Track (30 Hz):** `calcOpticalFlowPyrLK(winSize=(21,21), maxLevel=3)` + **forward-backward check** (reject
  FB err > 0.7 px); cull within 8 px of border.
- **Keyframe:** every 15 frames (0.5 s) or mean-flow > 40 px.
- **Anchor (the drift-killer):** re-anchor when shared live tracks < 30 **or** anchor age > 4 s (~5–7/lap).
- **Rotation:** `findEssentialMat(pts_A,pts_j,K,RANSAC,prob=0.999,threshold=1.0)` → `recoverPose` →
  `R̃=R_cvᵀ`; **discard t**; require ≥25 inliers, inlier ratio ≥0.5, else **no update** (coast on IMU+contact).
- **Low-parallax fallback** (median rot-compensated flow < 2 px): rotation-only **Kabsch** on bearings.
- **Persist tracks, not frames** (720 frames ≈ 660 MB): `data/loop_tracks.npz` per-keyframe {t, ids, uv,
  bearings} + per-pair {anchor_id, R̃, n_inliers, parallax}. Regenerate frames from the seed if needed.

## Implementation plan
**2.1 front-end** (`src/humanoid_asimov/vision.py` + `scripts/run_frontend.py`): `CameraModel` (K, GL→CV,
pixel↔bearing), head-cam FK extrinsics, `FeatureTracker`, `AnchorRotationEstimator`; write `loop_tracks.npz`
+ diagnostics. **Verify vs GT (GT scores, never feeds):** track-length histogram (median ≥1.5 s, ≥60 tracks/kf,
grid filled); FB survival >85%; RANSAC inliers >80% median; **rotation-vs-GT ≤2 mrad median, |bias| ≤0.5 mrad**
(the money gate — sets `R_vis`); **convention canary** — reproject wall-panel corners with GT pose+K (pixel-exact
overlay catches GL/CV flip, quat order, row-order, R vs Rᵀ); timing <5 ms/frame.

**2.2 fusion** (`src/humanoid_asimov/vio.py`: `VioESKF(ESKF)` with `clone()`, `update_rel_rot()`, `re_anchor()`;
`scripts/run_vio.py`; extend `run_nees.py`). Keep the Stage-1 `ESKF` class byte-identical. Per keyframe:
predict+contact to `t_kf` → vision update → maybe re-anchor. Delay first update ~1 s (settle transients).
**Verify (M=10 seeds):** loop close-gap VIO ≤0.4×ESKF and ≤5 cm; yaw error ≤4 mrad (staircase vs t^{3/2});
`b_g,z` error ≤0.3 mrad/s; per-block ANEES in-band incl. **yaw**; vision NIS≈3, lag-1 |ρ|≤0.3; **contact NIS
unchanged from Stage 1**; causality ablation (kill vision at t=12 s → slope reverts; `R_vis`×100 → VIO≈ESKF).

## Success metrics (24 s lap, ImuNoise, M=10, medians)
| metric | leg-odom | ESKF (S1) | **VIO pass** |
|---|---|---|---|
| loop close-gap / final pos err | ~25–40 cm | ~8–15 cm | **≤5 cm and ≤0.4× ESKF** |
| yaw error @24 s | ~10–15 mrad | ~10–15 mrad | **≤4 mrad median** |
| yaw growth | t^{3/2} | t^{3/2} | **staircase ~1.5 mrad/hop** |
| `b_g,z` error @end | — | ≥~1 mrad/s (unobs.) | **≤0.3 mrad/s** |
| consistency | — | per-block ANEES | **yaw+tilt+vel in-band; vision NIS≈3; contact NIS unchanged** |

## Deferred (explicit)
Tightly-coupled MSCKF/reprojection (Stage 3, iff extrinsic-translation/td calib or velocity-floor demands it);
persistent landmarks / SLAM / loop-closure relocalization (only route to *absolute*-yaw bound); extrinsic +
time-offset auto-calibration → **Stage 3** (rotation-extrinsic δφ_HC enters as `(I−Adj(R̃))δφ`, observable
under sustained yaw; `p_HC` does NOT appear in a rotation-only measurement — the reason Stage 3 adds
reprojection); rolling shutter / latency / encoder-noise in FK → Stage 4.
