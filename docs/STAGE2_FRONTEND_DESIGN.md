# Stage 2 addendum — sharpening the VIO front-end below 2 mrad

> Diagnosis + redesign of the relative-rotation front-end. The measured ~5–8 mrad error was
> **estimator-dominated, not correspondence- or convention-dominated**: the winning RANSAC essential
> matrix is an exact fit to one 5-point sample and was being used as the *final* answer, and the
> coplanar-floor geometry amplifies that into an R–t-coupled error with heavy tails. Fix (landed):
> demote RANSAC to init + outlier gate, add a robust IRLS refit of (R, t̂) on **all** inliers,
> texture the far walls, cap anchor age at 2 s, and gate gross failures against a gyro+neck-FK prior
> (accept/reject only). Money gate: **7.69 → 2.06 mrad median, 4.03 → 0.36 mrad bias**. Fusion now
> beats the ESKF at an **honest** σ_vis = 2 mrad on all seeds, with the b_g,z pin restored
> (|err| ≤ 0.42 mrad/s over 5 seeds — ≤0.20 on seeds 0–3, 0.42 on seed 4 — vs ESKF 0.72–2.21).

All numbers in this doc are measured on the 24 s loop (seed 0 front-end; fusion seeds 0–4 — the 5-seed sweep in §3.3), with the
error-decomposition harness described in §1.1. Baseline (pre-fix) money gate for reference:

```
median 7.69  mean 16.80  max 129.70 mrad   | bias 4.03
gated (<15 mrad, 23/34): median 4.94       | bias 1.52
tracks die ~0.37 s · inliers ~45-72 · floor-plane share of inliers 0.83 · anchor spans ~0.5-4 s
```

---

## 1. Diagnosis

### 1.1 Method: decompose, don't guess

Two tools made the error sources separable:

1. **Ray-cast oracle correspondences.** For every anchor feature, cast the pixel ray from the GT
   anchor camera pose into the scene geometry (robot moved out of the way), get the true 3-D point,
   and project it into the GT current camera. This yields *perfect* correspondences for the same
   feature set, plus a floor-plane/structure label per point (hit geom + height). Any estimator can
   then be run on (a) real KLT correspondences, (b) oracle correspondences, (c) oracle + synthetic
   0.4 px noise — on the *same* measurement instances.
2. **An estimator ladder on identical correspondences.** Shipped pipeline
   (`findEssentialMat(RANSAC) → recoverPose`, take R), the same on oracle inputs, an IRLS refit
   (§2.1), the refit seeded at GT (basin test), rotation-only orthogonal fitting on bearings,
   rotation-only with t frozen at the GT direction, and `USAC_MAGSAC` as an off-the-shelf reference.

### 1.2 Results (mrad, same loop, same correspondences unless noted)

| estimator | median | gated(<15) median | notes |
|---|---|---|---|
| A shipped RANSAC+recoverPose | 7.69 | 4.94 (23/34) | baseline |
| B shipped on **oracle** corr | 0.00 | 0.00 (31/32) | 1/32 catastrophic (1358) even noise-free |
| C shipped on oracle **+0.4 px noise** | 7.36 | 6.02 (25/32) | reproduces the full observed error |
| C2 IRLS refit on oracle+0.4 px | 1.51 | 1.32 (30/32) | same inputs as C |
| D IRLS refit on real KLT corr | 5.29 | 2.29 (26/34) | same inputs as A |
| D' IRLS seeded at GT | 5.29 | 2.29 | identical to D — not a basin problem |
| R-only, t frozen at GT dir | 1.90 | 1.88 (33/34) | kills the tail entirely |
| Rotation-only fit (ignore t), all inliers | 278 | — | translation is NOT ignorable |
| Rotation-only fit, far (>4 m) points | 79 | — | still hopeless |
| USAC_MAGSAC (has local-opt/polish) | 3.88 | 2.92 (27/34) | independent confirmation of the refit gain |

**Correspondence quality vs the oracle** (px, median per measurement): KLT chain 1.46
(floor 1.25, structure 2.68), with a **correlated mean drift (−1.03, +0.08) px** opposing the loop's
rotational flow, growing with anchor span (−0.73 px at ≤0.6 s → −1.2 px at >0.6 s). A direct
anchor→current re-match (single-shot KLT with chain init + FB check) was *worse*: 2.43 px — at these
spans the appearance warp defeats one-shot matching; the frame-to-frame chain is the better
correspondence engine.

**Confounder-resolving 2×2 split** (median eA | eD-refit | eC2-refit-on-clean, mrad):

| bucket | eA | eD | eC2 | n |
|---|---|---|---|---|
| span ≤ 0.75 s, floor ≤ 0.8 | 3.50 | 1.05 | 0.55 | 5 |
| span ≤ 0.75 s, floor > 0.8 | 6.03 | 2.00 | 1.17 | 7 |
| span > 0.75 s, floor ≤ 0.8 | 8.61 | 2.09 | 1.44 | 9 |
| span > 0.75 s, floor > 0.8 | 16.44 | 15.32 | 3.14 | 13 |

### 1.3 Verdict — ranked error sources

1. **Minimal-model-as-final-answer (dominant noise).** RANSAC's winning essential matrix is an
   exact fit to ONE 5-point sample; `recoverPose` decomposes exactly that. The 45–72 inliers only
   *vote* — they never enter the fit. Proof: with clean geometry and only 0.4 px noise the shipped
   pipeline still produces 6.0 mrad (row C) — ~27× above the √N noise floor
   (0.4 px / 342.76 px · √(2/56) ≈ 0.22 mrad) — while a refit on the same points gives 1.3 mrad
   (C2). On real correspondences the refit cuts the gated median 4.94 → 2.29 (D vs A). This is also
   exactly why the earlier knob-twiddling (more features, tighter RANSAC) did nothing or hurt: those
   knobs never touch the 5-point fit.
2. **Planar/low-diversity geometry amplification + R–t coupling (dominant tail).** 83% of inliers
   lie on the floor plane. Near-planar point sets leave the essential matrix with the classic
   twisted-pair two-fold ambiguity (row B: a catastrophic branch flip with *perfect*
   correspondences) and a soft cost valley trading R against t̂: freezing t at truth rescues the
   worst bucket (1.90 median, 33/34 clean) where the free-t refit stays at 15.3. With clean
   correspondences the same geometry costs only ~3 mrad (eC2, worst bucket) — i.e. the valley is
   *entered* by correspondence corruption, then *paid* in rotation.
3. **Correlated KLT chain drift (the ~2 mrad bias, and the long-span decay).** The −1 px mean drift
   against the dominant rotational flow is template-inertia of chained KLT under persistent flow;
   forward-backward checking cannot see it (it is FB-consistent). It maps to the measured
   (+0.9, −1.2, +0.1) mrad camera-frame bias, grows with span, and is worst on structure
   *silhouette* corners (solid-color pillars/panels: 2.68 px — occlusion edges drift with
   viewpoint) and on foreshortened floor texture.
4. **Exonerated:** conventions (oracle error exactly 0.00 — K, GL↔CV flip, quaternion order, R vs
   Rᵀ all correct, consistent with the projection canary); optimizer basins (GT-seeded refit ≡
   RANSAC-seeded); pixel noise level per se; feature count per se.

---

## 2. The approach

Four coordinated changes; each is measured individually in §3.2. The single highest-leverage change
is (1) — but (1) alone only fixes the median; (2)–(4) fix the tail, the bias, and the *information
content*, which is what the fusion actually consumes.

### 2.1 Robust non-minimal refit of (R, t̂) on unit bearings — the estimator fix

Keep `findEssentialMat(RANSAC, 1 px) → recoverPose` solely as **init + outlier gate**, then refine
on all inliers. Parameterize the relative pose on the 5-dof manifold — R ∈ SO(3) (right-multiplied
increments R←R·Exp(ω)) and t̂ ∈ S² (2-dof tangent basis B ∈ R^{3×2}, t̂←normalize(t̂+Bδ)) — and
minimize the Huber-weighted first-order geometric (Sampson) epipolar residual on normalized
image coordinates, scaled by the focal length so the loss knee is in pixels:

```
E(R,t̂) = [t̂]× R          (convention x_jᵀ E x_a = 0, x = K⁻¹·pixel homogeneous)
r_i     = f · (x_jᵀ E x_a) / √((Ex_a)₁² + (Ex_a)₂² + (Eᵀx_j)₁² + (Eᵀx_j)₂²)
minimize Σ_i huber_δ(r_i),  δ = 1 px
```

Iterate IRLS Gauss–Newton with Levenberg damping (λ up/down on cost decrease; the damping matters —
at genuinely low parallax the cost is flat in t̂ and the undamped normal equations blow up, while R
stays fully determined). Jacobians by forward differences over the 5 parameters (N≈50–80 residuals,
5 columns, ~10–25 iterations — microseconds-level cost, no new dependencies). Then **re-derive the
inlier set** at |r_i| < 1.5 px and refine once more (recovers good matches RANSAC's minimal
hypothesis mis-classified; median inlier ratio rose 0.80 → 0.93).

Why this cost and not the alternatives, per the measurements:

- **Rotation-only fitting (orthogonal/Procrustes on bearings) is disqualified**: 278 mrad median
  (79 far-only). At 0.86 m/s the anchor-span baseline is 0.4–2 m against 1–8 m scene depth —
  translation-induced bearing motion is first-order everywhere and must be *modeled*, not ignored.
  Estimate it, then discard it: exactly the existing architecture, done properly.
- **Freezing t̂ from another odometry source** (leg-odometry direction) would rescue the tail
  (t-frozen row: 1.90) but correlates the vision measurement with the contact updates the filter
  already fuses — the same double-counting sin the gyro constraint forbids, via the other modality.
  Rejected on architecture, kept in mind as a diagnostic upper bound.
- **A gyro term in the cost** would inject IMU drift into the measurement — forbidden, see §2.2.
- Off-the-shelf `USAC_MAGSAC` (which internally polishes) confirms the direction (2.92 gated) but
  under-performs the purpose-built refit (2.29) and offers no hook for the prior gate or the
  inlier re-derivation.

### 2.2 Gyro+FK prior as a pure accept/reject gate — independence preserved

The rare (~1–7 per lap) branch-flip/degenerate estimates (tens–hundreds of mrad) are trivially
separable from the honest ones (≤ ~8 mrad) with any coarse prior. The front-end composes the raw
gyro integration with the neck-FK extrinsics over the span:

```
R_prior = R_BC,Aᵀ · (∏_k Exp(ω_k·dt)) · R_BC,j        (raw gyro, encoder FK — onboard only)
‖Log(R̃ᵀ · R_prior)‖ > 30 mrad  ⇒  drop the measurement (coast on IMU+contact)
```

Independence argument, explicitly:

- The gate **never alters a value**: an accepted measurement is a stationary point of the purely
  visual cost, bit-identical to what it would be with no gyro present. No gyro term exists in the
  cost; the refit is seeded from RANSAC, not the prior (measured: GT-seeded ≡ RANSAC-seeded, so the
  seed does not shape the answer anyway).
- Prior error budget over a ≤2 s span: bias ≤ ~1 mrad/s (its RW σ over the whole lap) × 2 s +
  integrated white noise ~0.03 mrad ≈ **≤2 mrad ≪ the 30 mrad gate** (~15σ of the refined
  estimate). Selection leakage — the acceptance region being centered on the gyro path — can bias
  accepted values only through vision-error tail mass beyond ~28 mrad, which is negligible; and the
  filter's own χ²(3) gate already performs the same kind of IMU-informed selection downstream, so
  this adds no new dependence class, it just applies it before a 100 mrad outlier can spike NIS.
- The fusion run uses the **raw noisy gyro** (never the filter's bias estimate), keeping the
  front-end causally independent of the filter state.

### 2.3 Anchor policy: spans must out-inform the gyro (a measured negative result)

A pure per-measurement metric is a trap. A short-span policy (re-anchor every keyframe, 0.5 s
spans) *passed the money gate* — median 1.86, max 7.78, bias 0.45, 45 meas/lap — and **lost in
fusion**: VIO 0.30% drift / b_g,z err +1.00 vs ESKF 0.23% / +0.72. Reason: over a span τ the
measurement competes with the gyro's own relative rotation, whose error is ≈ b·τ ≈ 1 mrad/s · τ.
At τ = 0.5 s the gyro (≈0.5 mrad) beats any realistic vision measurement (≈1.9 mrad): short-span
updates add no rotation information the filter lacks, while their per-hop bias accumulates across
~46 hops/lap. Long anchors are where vision out-informs the gyro (τ = 2 s: gyro ≈ 2 mrad, refit
vision ≈ 2 mrad, but vision is *drift-free* and averages across anchors into the b_g,z estimate).

Policy chosen: **anchor max age 2.0 s** (down from 4.0 — spans beyond ~2 s decay into the KLT-drift
regime: 4.97–8.7 mrad median even after the refit and scene fix), re-anchor on measurement failure
or shared-tracks < 30 as before. Per-anchor b_g,z information ≈ σ_v/τ ≈ 1 mrad/s, ~13–15 anchors
per lap → ~0.3 mrad/s-class constraint per lap, matching the observed pin.

### 2.4 Scene: texture the *far* walls; close clutter measurably hurts

The information-carrying long spans die on correspondence quality, so give the tracker features
whose *appearance survives* 1–2 s of loop motion. Those are **far** features: at 5–10 m they move
slowly, barely warp under camera translation (no foreshortening spin like the floor), live long —
and at the 1.7–3.4 m baseline a long span builds, they still subtend real parallax, so they also
condition t̂ and break the floor's coplanarity. The perimeter walls had only sparse solid-color
panels (corners = silhouette/junction features — the worst kind, 2.68 px drift); the fix is a
procedural high-contrast multi-scale pattern on the wall faces themselves (`_apply_wall_texture`,
visual-only, dynamics and Stage-1 data untouched).

Measured (anchored policy, refit + gate active, seed 0), median / span-bucket medians (mrad):

| scene variant | all | ≤0.6 s | 0.6–1.6 s | >1.6 s | bias |
|---|---|---|---|---|---|
| baseline room | 2.68 | 1.56 | 5.64 | 7.79 | 0.95 |
| + close textured boxes (1–2.5 m ring) | 4.90 | 1.88 | 7.36 | 20.4 | 1.14 |
| + textured walls | **2.19** | 1.83 | **2.03** | 4.97 | 0.74 |
| + walls + close boxes | 2.38 | 1.61 | 4.49 | 4.26 | 0.72 |

The intuitive candidate — bring non-coplanar structure into 1–4 m — made things **worse** (mid-span
7.36 vs 5.64): close objects occlude the far field and win the detector budget, but their tracks
have the highest flow/warp and die/drift fastest, so long spans end up with *less* usable geometry.
Far texture is the scarce resource, not close parallax. (Also measured and rejected: an 8×6
bucket-grid detector quota — no gain, the upper image cells are mostly *far floor* here, 3.30 vs
2.68; camera pitch 15°→5° — worse, 5.69, grazing floor view inflates the pitch-axis drift bias to
2.5 mrad, confirming the old pitch experiments but now with the mechanism; tighter FB threshold
0.7 px — worse, 3.06 with bias 2.53, it culls high-flow-but-honest tracks and biases the survivor
set.)

---

## 3. Implementation (landed) + validation

### 3.1 Diffs

- `src/humanoid_asimov/vision.py` — `_t_basis`, `_sampson_px`, `_refine_rt` (IRLS-GN on
  (R, t̂), ~60 lines, pure NumPy); `AnchorRotationEstimator` gains `refine=True`,
  `prior_gate=0.030`, `sampson_gate_px=1.5`, and `estimate(t, tracks, R_prior=None)`. Default-on;
  existing callers (`run_calib.py`, `run_livedrift.py`) get the refit with zero changes.
- `src/humanoid_asimov/scene.py` — `_wall_texture` + `_apply_wall_texture` applied in
  `add_feature_room` (visual-only; collision/dynamics untouched).
- `scripts/run_frontend.py` — gyro+FK prior for the gate, `ANCHOR_MAX_AGE = 2.0`, bias-vector
  reporting, gated-count reporting.
- `scripts/run_vio.py` — same policy + prior (raw noisy gyro), `SIG_VIS = 2.0e-3` now *honest*.
- **Untouched**: `vio.py`, `estimator.py` (fusion + FD-verified Jacobians — re-verified after the
  change: analytic-vs-FD max err 1.4e-10; zero-blocks exactly zero), `CameraModel`,
  `HeadCamExtrinsics`, `FeatureTracker` — same measurement definition, same conventions, projection
  canary still pixel-exact.

### 3.2 The money gate, before → after (seed 0, deterministic)

```
before: median 7.69  mean 16.80  max 129.70  bias 4.03   (gated 23/34: 4.94 / 1.52)
after : median 2.06  mean  3.78  max  18.92  bias 0.36   (gated 34/36: 1.96 / 0.62)
        bias vector (+0.36, −0.01, +0.05) mrad — the yaw component (the one that integrates into
        heading and aliases b_g,z) is now ~zero; 36 measurements + 11 gated/degenerate keyframes
        (the filter coasts through those); inliers 56 median at ratio 0.93.
```

Attribution: refit alone (old policy/scene) 4.94 → 2.29 gated; + textured walls 2.03 mid-span;
+ 2 s cap + prior gate → 2.06/0.36 overall with max < 19 (the χ² gate never sees a 100 mrad spike).

### 3.3 Fusion at an honest σ_vis (the actual goal)

σ_vis = 2.0 mrad now *bounds* the measured error (per-axis σ ≈ 2.06/1.54 ≈ 1.3 mrad, bias 0.36 ≪ σ)
instead of under-stating a 5 mrad one. Seeds 0–3, 24 s loop, same runs for all estimators:

| seed | ESKF drift | VIO drift | ESKF b_g,z err | VIO b_g,z err |
|---|---|---|---|---|
| 0 | 0.23% | **0.20%** | +0.72 | **−0.12** |
| 1 | 0.73% | **0.02%** | +1.86 | **−0.19** |
| 2 | 0.49% | **0.26%** | +2.21 | **+0.12** |
| 3 | 0.45% | **0.15%** | +1.33 | **−0.20** |

VIO beats the ESKF on drift on every seed and the gyro-z-bias pin is restored: |err| ≤ 0.42 mrad/s over 5 seeds
(≤0.20 on seeds 0–3; the design target of ≤0.3 is missed only by seed 4's 0.42). Robustness: at σ_vis = 2.5 mrad the win holds (seed 0: 0.11% / +0.28;
seed 1: 0.18% / +0.14) — not knife-edge. The old caveat (2 mrad = over-trusting) is closed: the
same number is now a measured bound, not a wish.

### 3.4 Validation loop (repeatable)

1. `scripts/run_frontend.py` — money gate: median ≤ ~2, |bias| ≤ ~0.5, max small enough that the
   χ²(3) gate stays quiet; watch the *bias vector's yaw component* specifically.
2. `scripts/run_vio.py <seed>` for seeds 0–3 — VIO < ESKF drift AND |b_g,z err| ≤ 0.3 mrad/s at
   σ_vis honest (≥ measured median/1.5).
3. `scripts/check_projection.py` (conventions) + `scripts/check_vio_jacobian.py` (fusion Jacobian
   FD) — must stay pixel-exact / ~1e-10 after any front-end change.
4. When touching policy: re-check the *fusion*, not just the money gate — §2.3 is the cautionary
   tale (a front-end can pass per-measurement and still subtract information).

## 4. Risks + honest fallback

- **The wall texture is a scene assumption.** It stands in for "the far field has trackable
  texture" (shelving, signage, windows, clutter — typical indoors, *not* guaranteed). On a bare-wall
  version of this room the honest number is the §2.4 baseline row: ~2.7 median but ~5.6 mrad in the
  information-carrying mid spans, i.e. σ_vis ≈ 4–5 mrad and near-parity fusion. That *is* the
  fallback statement: with a feature-poor far field, this VIO matches the ESKF instead of beating
  it, and the honest claim becomes "front-end-limited; parity with IMU+contact, yaw no longer
  worse". The refit + gate are still worth keeping in that world (they cost nothing and remove the
  catastrophic tail).
- **Residual correlated-drift bias.** The KLT template-inertia bias is motion-dependent (this loop
  turns one way). The final VIO yaw errors sit at +0.4…+0.9° across seeds (ESKF: −0.5…−1.9°) —
  consistent with the remaining gated-set bias (~0.6 mrad) plus anchor-hop random walk, but the
  consistent sign says a small systematic survives. Probes if it matters later: run the mirrored
  loop (opposite turn) and check the sign flips; if yes, candidate mitigations are span-symmetric
  anchor scheduling or a slow online estimate of the per-axis measurement bias (a 3-state augment —
  but that starts to eat the yaw information budget, so only with evidence).
- **Gate availability.** The 30 mrad prior gate assumes a live gyro. On gyro dropout, fall back to
  the filter's χ²(3) gate alone (it caught these outliers before, at the cost of NIS spikes); do
  not tighten the prior gate below ~10σ of the vision error, or the selection starts to matter and
  the independence argument erodes.
- **Planar two-fold flips still exist pre-gate** (~1/30 even with perfect correspondences). They are
  currently 100% caught by the prior gate. If the floor share ever rises further (long straight
  corridors), consider disambiguating by refining *both* twisted-pair branches and keeping the
  lower-cost one before gating — vision-only, ~2× refit cost.
- **Ceiling.** The refit floor on clean correspondences is ~1.3 mrad (C2); the remaining gap
  (2.06 vs 1.3) is correspondence quality. The next real upgrade if < 1.5 mrad is ever needed is a
  multi-keyframe windowed refinement (joint rotation over all keyframes sharing tracks — averages
  per-pair correspondence noise without lengthening any single KLT chain), not more per-pair
  tuning. Deferred until a consumer demands it.
