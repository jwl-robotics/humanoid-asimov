# Self-review — humanoid state estimation + navigation

> Adversarial end-of-project review of the full six-stage build: contact-aided ESKF → anchored
> relative-rotation VIO → online camera-mount/time-offset self-calibration → embedded-pipeline
> noise study → closed-loop navigation slice. Everything below is grounded in the code as it
> exists at review time (file:line references) — the point is to know, before anyone else finds
> them, where the soft spots are.
>
> **Verdict.** The filter math is correct (independently re-derived here on top of the in-repo FD
> checks), the observability story is real and honestly told, and the staged discipline (each
> stage byte-preserves the last, truth only ever scores) is the project's strongest property. The
> genuine weaknesses are concentrated in one theme: *covariance honesty in the vision path was
> asserted by design but never re-validated by harness*, plus a handful of modeling gaps (IMU
> lever arm, encoder-velocity idealization), one doc/code contradiction (the time-offset demo),
> and cross-stage tuning drift. None overturn the headline results; several would sting if
> surfaced by someone else first.

> **Post-review status (2026-07-09).** Findings addressed since this review was written:
> **1.2** the front-end was sharpened instead of R_vis raised — IRLS refit on all inliers, 7.69 → 2.06 mrad
> median (bias 0.36), so σ_vis = 2 mrad is now measured-honest (STAGE2_FRONTEND_DESIGN.md); the vision-path
> NEES harness remains the open half. **1.4** run_calib.py's docstring now states the td freeze honestly.
> **1.9** update_gravity is dimension-aware. **G3 (partially)** run_vio.py sweep makes the headline a 5-seed
> median: VIO 0.15% vs ESKF 0.49%, wins every seed; b_g,z |err| median 0.19 (≤0.42 every seed) vs 1.66 blind.
> ESKF F/H now FD-checked in-repo (check_eskf_jacobian.py). Point numbers below (0.14%, +0.01 mrad/s, "~6 mrad
> front-end") are the pre-refit values current at review time — superseded by the medians above. Still open:
> vision-path NEES (1.2/G4), IMU lever arm (1.1), shared-anchor correlation (1.3), enc_qd differencing
> ablation (1.7), χ² gate re-enable (1.8), td on a rate-varying trajectory (1.4), nav Rc (1.6).

---

## 1. Correctness audit

The Jacobians (ESKF F/H, VIO H 3×21, calibration H 3×25) are FD-verified in-repo
(`scripts/check_*_jacobian.py` — with zero-block checks, negative controls, degenerate cases, and
the gauge identity at 1e-16). Re-derived by hand and confirmed:

- **Gauge annihilation is exact, not linearized-lucky.** For a global yaw shift γ about e_z,
  H·δx = R_CB,j(γR̂ᵀe_z) − R_CB,j(R̂ᵀR_A)(γR_Aᵀe_z) ≡ 0 identically, **for any** R̂, R_A
  (`vio.py:57-58`). That is why no FEJ/OC machinery is needed — the single most defensible
  theoretical property in the repo.
- **Clone/injection bookkeeping across the subclass chain is right.** `CalibVioESKF._inject`
  deliberately grandparent-calls `ESKF._inject` and re-maps `dx[18:21]`→mount, `dx[21]`→td,
  `dx[22:25]`→clone (`calib.py:57-62`) — exactly the trap "calib before clone" ordering creates,
  handled. Cloning copies rows/cols 6:9 *with* cross-covariances, so the clone inherits δθ↔δb_g
  and δθ↔δφ_HC correlations.
- **Anchor-extrinsic recomposition is done right.** `_calib_rH` stores `R_BH_A` and recomposes
  **both** keyframes' `R_BC` with the *live* `R_HC` every update (`calib.py:71-72`); freezing
  `R_BC,A` at clone would fake 3-axis mount observability — respected in code.
- **Injected-truth plumbing is machine-exact.** `q_true = HEAD_CAM_QUAT ⊗ quat(F·δφ)` with
  F = diag(1,−1,−1) satisfies `R_HC,true = R_HC·Exp(δφ)` exactly (F is a rotation, det +1). The FK
  model carries no camera and reads only nominal constants — perturbed truth renders the images
  but never leaks into the estimator.

Findings, ranked by how much they matter:

### 1.1 Unmodeled IMU lever arm (modeling gap, medium — the one nobody wrote down)
The IMU site sits at r = (−0.05, 0, 0.08) m from the freejoint origin (`asimov.xml:94`), but
`_predict_step` treats the accelerometer as the specific force *at that origin*
(`estimator.py:188-194`). The true output additionally contains ω̇×r + ω×(ω×r); gait-frequency
ω ripple and foot-strike ω̇ spikes give lever-arm accelerations of order 0.1–2 m/s² — one to two
orders above the injected white noise. Consequences: (1) a gait-synchronous colored δv error — a
second contributor to the optimistic-velocity-NEES symptom `STAGE1_ESKF_REVIEW.md` attributes
entirely to the moving contact point; (2) the nonzero mean of ω×(ω×r) biases the weakly-observable
b_a. Contact updates at 500 Hz keep v pinned so it doesn't invalidate results, but it is a real
unmodeled systematic on a platform whose thesis is "model the systematics." Fix (cheap, standard):
express the state at the IMU point, or add the ω̇×r + ω×(ω×r) terms. Do this before any hardware claim.

### 1.2 Vision noise is set to the *target*, not the *measured* front-end error (consistency, medium-high)
`sig_vis` defaults to 2 mrad (`vio.py:20`), but the front-end came in at ~5–6 mrad median at review
time — so R_vis under-weighted the true variance by ~×6–9, making the yaw/b_g,z covariance
overconfident by about the same factor and sporadically gating healthy measurements against a
too-small S. The Stage-2 design's own acceptance criteria (vision NIS ≈ 3, lag-1 |ρ| ≤ 0.3,
per-block ANEES incl. yaw) were **never executed** — `run_nees.py` is ESKF-only. *(Half-addressed:
the front-end was sharpened to 2.06 mrad median so σ_vis = 2 mrad is now honest; the vision-path
NEES harness remains the open half.)*

### 1.3 Correlated vision residuals against a shared anchor (modeling assumption, medium)
Every keyframe measured against the same anchor shares that anchor's feature-localization error
(`vision.py:232-259`), but `update_rel_rot` treats each as independent white noise — double-counting
the anchor-error information by ~updates-per-anchor, compounding 1.2. Re-anchoring every ≤2–4 s
bounds it (a legitimate defense). Principled fixes: a per-anchor measurement-bias state, R_vis
inflation by updates-per-anchor, or a Schmidt treatment. Corollary: the front-end's *bias* is
indistinguishable from gyro bias — a constant b mrad error per interval T injects a false rate b/T
into b_g, so the achieved b_g,z error implies the gated front-end bias was ≲0.05 mrad (the
quantitative reason the bias gate exists).

### 1.4 The time-offset demo contradicts its own docstring (results integrity, medium)
The design predicted "td is the *best*-observed state" (from a measured 1.12 rad/s gait modulation),
but on the constant-rate yaw loop reality was the near-degeneracy the FD check itself proves
(constant-ω ⇒ td block = 0). So the shipped demo estimates the mount only and freezes td. The honest
outcome — "td sits near the constant-rate degeneracy on this trajectory; the filter reports it via a
wide funnel" — is fine and even interesting; the *docstring claiming it was nailed* was the liability.
*(Addressed: docstring now states the freeze honestly.)* Better still: run the td recovery on a
rate-varying trajectory (stop–start, speed steps, or the already-plumbed `neck_scan`,
`walk.py:75-77`); the injection plumbing is already FD-checked at t̂d = 20 ms, so it is mostly a
driver change. Related nit: reporting φ in the (n_C, ⊥) basis makes the observability exhibit
*stronger* — φ_z is ~93% ⊥ to n_C and should converge nearly as fast as φ_x.

### 1.5 The Stage-5 velocity-command loop is partially cosmetic (framing, medium)
`nav.py` emits `VelocityCommand(vx, vy, vyaw)`, but the only actuated channel is heading: `vx` is
fixed by the gait CSV cadence, `vy` is always zero, `c.vyaw` is never used, and the steering
authority is the virtual-support world torque — the same non-physical wrench that carries ~44% of
body weight (`walk.py:138-141`). So Stage 5 validates estimate → waypoint geometry → heading command
→ (crutch) yaw servo: a genuine closed loop *through the estimator*, but not an exercised
velocity-command interface. The honest sentence: "the planner speaks the velocity-command interface;
the dev locomotion consumes only its heading projection." (What *is* clean: "reached" fires on the
estimate while true closest-approach and final truth-gap are reported separately,
`run_nav.py:52-58`.)

### 1.6 Turn-induced b_g-runaway mode — checked on the nav path (tuning drift, low-medium)
Stage 2.2 found that tight `sig_contact=0.05` can overfit turn-induced slip into b_g and run the bias
away on *sustained* turning — loosened to 0.15 for the 360° loop. The nav path has three ~90° turns, so
the concern applied and was **checked** (`run_nav.py:36`): on this short, intermittently-turning path
0.05 is in fact the better setting — 7.1 cm final vs 9.6 cm at 0.15 — because the turns are too brief for
the runaway to accumulate, and the looser Rc gives up more than it saves. So the default is justified
here, not merely inherited. The deeper point still stands: a hand-picked per-trajectory Rc means the
filter lacks a slip-adaptive measurement noise (scale Rc with |ω_z| / stance force), which is the
production answer.

### 1.7 "Amplitude is negligible" rests on an undeclared encoder-velocity assumption (Stage-4, medium)
`apply_embedded_noise` quantizes `enc_q` but passes `enc_qd` through clean (`embedded.py:52-58`). If
firmware instead differenced the quantized 14-bit angle at 500 Hz, induced velocity noise would be
±ENC_RES/dt ≈ 0.19 rad/s — through the leg Jacobian, several cm/s of foot-velocity noise, *comparable
to sig_contact itself*. Real drives usually report observer-filtered velocity, so the assumption is
defensible — but it is load-bearing for the headline "timing dominates amplitude" and is stated
nowhere. Add the one-line ablation (`enc_qd = diff(quantized q)/dt`) to bound it. The timing
conclusion likely survives; surviving-after-ablation is a much stronger sentence.

### 1.8 The contact χ² gate is disabled in practice, and Stage 4 is measured that way (low-medium)
`gate=1.0e6` (`estimator.py:165`) against a χ²(3) whose 99.9% point is 16.27 — the "loose gate"
comment undersells that this is *off*. Deliberate (per-update NIS with a correlated residual
mis-gates), but the Stage-4 latency result (5 ms skew → 2.04%) is measured with outlier rejection
off; a working gate would reject some swing-phase "no-slip" updates latency creates and plausibly
shrink the ratio. Cheap experiment; pre-empts an obvious "what if you just gated?" question.

### 1.9 Latent dimension bug: `update_gravity` was not clone-aware (code defect, low)
`update_gravity` hardcoded H(3,18) / `np.eye(18)` (`estimator.py:268-275`); called while a VIO clone
(21) or calib state (25) is active it raised a shape error. No pipeline calls it (excluded from the
walking loop by design), but its documented use — static init — is exactly when a Stage-2/3 filter
would want it. *(Addressed: now dimension-aware.)*

### 1.10 Correct-but-subtle things that survived the audit (state these proactively)
- **The measured gyro appears in both predict and the contact measurement** (`estimator.py:243-244`),
  correlating process and measurement noise within a step — deliberately ignored, negligible against
  σ_contact = 50 mm/s.
- **b_c vs b_g separability**: b_c enters H directly as −R, b_g through the lever R[p_f]× — the z-lever
  makes b_g,x/y observable and b_g,z weak, exactly the observed structure. The *per-foot* variant
  aliases with v at ~50% duty — tried, diverged, documented.
- **Quantization is applied after the injected sensor noise** — the physically correct ADC ordering.
- **Frame conventions** (GL→CV flip, quat order, R vs Rᵀ, `R̃ = R_cvᵀ`) were nailed once and *pinned*
  by a reprojection canary + every FD gate's r₀ = 0 assertion — why a 25-state filter over three
  subclasses has no frame bugs.
- **Init transient**: filters init v = 0, σ_v = 0.1 while the robot is mid-gait at ~0.3–0.5 m/s — a
  3–5σ inconsistency the 500 Hz contact stream absorbs in ~0.1 s. Harmless here; on hardware, init
  standing (when `update_gravity` earns its keep — see 1.9).

---

## 2. Gaps and risks — and how to speak to each

**G1. The ~6 mrad front-end, and its reach into everything.** It bounds VIO drift, the calibration
funnels (σ_φ⊥ ≈ 1.5 mrad), and b_g,z via its bias; there is no low-parallax fallback and no E/H model
selection, so turn-in-place = no vision. *Speak to it:* it was measured continuously (money-gate
median + bias), R_vis should match it (§1.2), and the upgrade list is concrete and cheap (rotation
averaging over the anchor window first — it attacks noise *and* the §1.3 correlation at once). The
loosely-coupled design means the front-end is a swappable module behind a scalar quality gate.

**G2. The locomotion crutch (~44% weight offload, torque-steered).** Not dynamic walking; slip
statistics, impact spectra, contact duty, and steering authority are all crutch-shaped, so every
tuned constant is regime-specific, and Stage 5's actuation path runs through the crutch itself
(§1.5). *Speak to it:* declared and quantified in-repo (0.74 s topple without it; 7 mm/stance slip;
2°/10 s yaw wander) — the estimator never consumes it, so the *interfaces and observability results*
stand while the *constants* are expected casualties of the swap to a real balance policy. That swap
is the designated next milestone.

**G3. Observability validated on one trajectory family.** The yaw loop makes the mount's n_C axis and
td *structurally* weak; the wide funnels are honest, but "would converge with richer excitation" is
currently an argument, not a plot. *(Partially addressed: the VIO headline is now a 5-seed median;
calibration/nav remain single-seed.)* *Speak to it:* the degeneracy directions were *predicted* from
H's structure and confirmed by FD degenerates — the strongest form of understanding your filter — and
the excitation fix is one flag away (`neck_scan`, heading-rate steps). Then run it plus a 10-seed
sweep of run_calib/run_nav.

**G4. Covariance consistency is proven only for Stage 1.** The ESKF's colored-contact optimism is
measured and root-caused (and §1.1 adds the IMU lever arm to that root cause); the vision path adds
an under-set R_vis (§1.2) and shared-anchor correlation (§1.3) with **no NEES/NIS harness ever run on
VioESKF/CalibVioESKF**. *Speak to it:* consistency-as-methodology is already this project's
differentiator (per-block NEES, false-pass exposure, NIS autocorr) — extending the harness to Stage
2/3 is the single highest-credibility next task, and the expected outcome is understood (yaw block
optimistic ×~3 until R_vis is corrected, then in-band). The principled endgame for the contact side
remains the right-invariant EKF with foot states.

**G5. No loop closure, no map, no global frame.** Everything is odometric; absolute yaw and position
drift without bound by construction — tolerable over 12 m (7.1 cm), not over 500 m. *Speak to it:*
deliberate scope — the anchored architecture is loop-closure-*shaped* (anchors are proto-keyframes;
promote, match, and you have a pose graph), and gauge honesty means the filter never *pretends* to
know absolute heading, which is precisely the property a map layer wants underneath it.

**G6. Idealizations inherited from sim.** Perfect kinematic model (FK = sim model), clean binary
contact at 1 N, no rolling shutter/blur/exposure, encoder velocity assumed firmware-filtered (§1.7),
IMU scale/misalignment unmodeled. *Speak to it:* each is named, each has a known instrument
(kinematic calibration; force-derived contact probability; RS-aware front-end; the one-line enc_qd
ablation; scale states under sufficient excitation) — and Stage 4 exists precisely to rank which
matter first, with a defensible answer: timestamps before noise models.

**G7. Software-engineering debt at the edges.** `update_gravity` clone-awareness (§1.9, fixed);
magic-slice state layout across three files; the td scaffolding/docstring (§1.4, fixed);
`sig_contact` divergence between runs (§1.6); the gate=1e6 comment (§1.8). None affect shipped
results; all are small. The next state addition should force a state-layout registry (named slices,
one source of truth) or composition — the `update_gravity` miss is exactly the class of bug the
magic-slice layout invites.

**G8. Doc/number drift.** Point-in-time numbers fossilize across the stage docs and dashboard as
later stages land (0.14% vs the 0.15% median; the 3.1° mount value; pre-refit calibration tiles).
*Speak to it:* the post-review status header above is the ledger; headline numbers are single-sourced
from `run_vio.py sweep` / `run_frontend.py`, and the generated pages rebuild from `build_dashboard.py`
after any re-measure.

---

### Bottom line

The project earns its central claims: a correctly-implemented, observability-literate estimator
family whose every stage measures its own limit and hands it to the next stage as a spec. The math
holds up under adversarial re-derivation; the honest-reporting culture (published failures, per-block
consistency, FD negative controls) is genuinely rare and is the thing to lead with. The work
remaining to make it airtight is narrow and known: model the IMU lever arm, set R_vis to the measured
front-end error and run the Stage-2/3 consistency harness, land the td recovery on a rate-varying
trajectory, reconcile the nav run's contact noise with the Stage-2 lesson, and multi-seed the
remaining headline numbers. Every one of those is days, not weeks.
