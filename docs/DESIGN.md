# Humanoid state-estimation + navigation — DESIGN

A monocular-camera + IMU + leg-odometry **state estimator** with **online self-calibration** for a bipedal
humanoid in MuJoCo simulation — validated against embedded-systems sensor noise and closed around a minimal
navigation slice. This is the technical design; per-stage detail lives in the `STAGE*` docs, the
end-of-project `SELF_REVIEW.md`, and the living dashboard (`docs/dashboard/`).

## Problem
A humanoid that senses with **one monocular camera + a 6-DOF IMU + joint encoders** (no depth, no lidar) needs
its base pose and velocity to navigate. Leg odometry + IMU alone leave **heading (yaw) unobservable**, so the
estimate drifts; a single camera supplies the missing constraint but brings its own unknowns (camera-mount
extrinsics, camera–IMU time offset). This project builds that estimator, calibrates its sensors online,
characterises how the embedded sensor pipeline degrades it, and closes the loop with a navigation slice.
Locomotion is out of scope — a black box that accepts `VelocityCommand(vx, vy, vyaw)`.

## Harness + honesty caveat (Stage 0)
The robot loads from the open **Asimov** MJCF (`asimovinc/asimov-1`); a **head camera** is added via MjSpec
(forward, 15° down — absent upstream, the prerequisite for monocular VIO). Onboard sensing is split into
**estimator-visible** {raw gyro, accel, 15 leg+waist+neck encoders, foot contact} vs **eval-only** ground truth
{orientation, base pose/velocity, world foot positions}, with LSM6DSOX-scale IMU noise + bias.

**Locomotion is a load-bearing development crutch — stated plainly.** The upstream model has no leg actuators
(a learned policy supplies torques on hardware); to generate motion for estimator development we PD-track the
open imitation gait with a torso **"virtual support"** that offloads ~44% of body weight and holds the torso
upright. This is **not true dynamic walking** — it is a stand-in for the balance controller, used only to
produce self-consistent IMU/encoder/contact data + ground truth. The estimator only ever sees onboard sensors.
Quantified (topple at 0.74 s without the support; ~7 mm/stance slip) on the Stage-0 dashboard page.

## Architecture
```
            ┌───────────── MuJoCo / MJX (Asimov MJCF + added head cam) ─────────────┐
            │  IMU (gyro,accel) · joint encoders · foot contact · monocular camera   │
            └───────────────────────────────┬───────────────────────────────────────┘
                       raw sensors (+ Stage-4 embedded-noise layer)
                                             │
                ┌────────────────────────────▼────────────────────────────┐
                │  STATE ESTIMATOR  (the core deliverable)                  │
                │   • IMU prediction + leg/contact kinematic odometry       │
                │   • monocular visual factor (anchored relative rotation)  │
                │   • ONLINE auto-calibration: camera-mount extrinsic,      │
                │     camera–IMU time-offset td, IMU biases                 │
                │   → 6-DoF base pose + velocity + covariance               │
                └────────────────────────────┬────────────────────────────┘
                              estimated state │
                ┌────────────────────────────▼────────────────────────────┐
                │  MINIMAL NAV SLICE (Stage 5)                              │
                │   goal + waypoints → planner → VelocityCommand(vx,vy,vyaw)│
                └────────────────────────────┬────────────────────────────┘
                                             │  VelocityCommand
                ┌────────────────────────────▼────────────────────────────┐
                │  LOCOMOTION  =  BLACK BOX  (velocity-following gait)      │
                └──────────────────────────────────────────────────────────┘
```
The command interface is a body-frame `VelocityCommand{vx, vy, vyaw}` — the estimator + planner *emit* it; a
locomotion layer would consume it. No walking controller is built here.

## Staged plan (each stage nailed + validated before the next)
- **Stage 0 — Sim harness.** Load the MJCF, add the head camera, expose the onboard/eval sensor split, stand up
  the locomotion black box, log ground truth. *Done: self-consistent walking dataset.*
- **Stage 1 — Contact-aided ESKF (no vision).** 18-state error-state EKF: IMU prediction + a no-slip
  contact-velocity update + online gyro/accel/slip biases. *Result: 0.32% drift — but heading is unobservable
  from IMU + contact alone, which motivates vision.*
- **Stage 2 — Monocular VIO.** A loosely-coupled **anchored relative-rotation** visual update (essential-matrix
  rotation, translation discarded) on a stochastically-cloned rotation state; gauge-safe by construction.
  *Result: beats the ESKF on every seed (~0.15% vs ~0.49% median drift over 5 seeds) at a genuinely honest
  ~2 mrad vision noise — the front-end's essential-matrix rotation is refit (IRLS on all inliers), not left at
  the 5-point minimal model; pins the yaw drift + the gyro-z bias the ESKF cannot see.*
- **Stage 3 — Online self-calibration (headline).** Jointly estimate the **camera-mount rotation extrinsic** +
  **camera–IMU time offset** online from the existing vision stream (error state 21→25). Start mis-calibrated,
  recover from injected error. *Result: recovers an injected 3.1° mount error on the observable axis; the
  covariance honestly reports what the trajectory leaves unobservable.*
- **Stage 4 — Sim-to-real noise.** Inject the embedded pipeline — encoder/IMU quantization, sequential-CAN
  stagger, transport latency, sample jitter — and ablate. *Result: timing effects dominate; amplitude effects
  are negligible.*
- **Stage 5 — Minimal nav slice.** A waypoint planner turns the estimated pose into `VelocityCommand`. *Result:
  reaches its waypoints on the estimate; the estimate-vs-truth gap is the yaw drift VIO removes.*

## Estimation choices
- **Filter vs smoother:** an error-state EKF (real-time / embedded-friendly) is the baseline; a factor-graph
  smoother (IMU preintegration + visual + leg-odometry factors) is the natural extension for loop closure.
- **VIO coupling:** loosely-coupled anchored rotation was chosen over a tightly-coupled MSCKF to stay gauge-safe
  and robust on a low-parallax walking trajectory; the rotation-only measurement is what makes yaw + the gyro-z
  bias observable without a global map.
- **Leg odometry:** contact-aided error-state filtering (the canonical legged-robot result), with a shared
  slip-velocity bias to keep the gyro bias honest.

## Evaluation
Reproducible repo + a metrics table (not slides): drift / final error vs ground truth, covariance consistency
(NEES/NIS), auto-calibration convergence curves with ±3σ funnels, and an accuracy-vs-noise ablation. Every
filter Jacobian is finite-difference-verified in-repo (`scripts/check_*_jacobian.py`).
