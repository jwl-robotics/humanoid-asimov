"""Stage 4 — embedded sensor noise (the sim-to-real pipeline).

The sim-to-real gap for a low-cost humanoid is dominated by the **embedded pipeline**, not physics
fidelity: finite encoder/IMU resolution (quantization), the CAN bus reading joints *sequentially* (a
staggered per-joint sampling skew), transport **latency**, and sample-interval **jitter**. This module
corrupts a logged dataset with those effects — by re-sampling each stream at its true (perturbed) capture
time and quantizing — so the estimator can be stress-tested the way it would face real hardware.

Applied on top of the Gaussian/bias `ImuNoise` already in the dataset. Physics is untouched; only the
*measurement pipeline* is made realistic.
"""
from __future__ import annotations

import numpy as np

# Representative LSM6DSOX-class resolutions + a 14-bit absolute joint encoder.
GYRO_LSB = np.radians(0.070)          # ~0.07 deg/s per LSB (±2000 dps, 16-bit)  → 1.2e-3 rad/s
ACCEL_LSB = 4.0 * 9.81 / 32768.0      # ±4 g, 16-bit                              → 1.2e-3 m/s^2
ENC_RES = 2 * np.pi / (1 << 14)       # 14-bit absolute encoder                   → 3.8e-4 rad


def _quantize(x, res):
    return x if res <= 0 else np.round(x / res) * res    # res<=0 disables (for ablations)


def _interp_cols(t, x, tq):
    """Linear-interpolate each column of x(t) at query times tq (edge-clamped)."""
    if x.ndim == 1:
        return np.interp(tq, t, x)
    return np.stack([np.interp(tq, t, x[:, c]) for c in range(x.shape[1])], axis=1)


def apply_embedded_noise(log, seed=0, stagger_ms=2.0, latency_ms=5.0, jitter_ms=0.8,
                         enc_res=ENC_RES, gyro_lsb=GYRO_LSB, accel_lsb=ACCEL_LSB, delay_contact=False):
    """Return a copy of `log` with the embedded-pipeline effects applied to the estimator-visible streams
    (gyro, accel, enc_q, enc_qd). Ground-truth channels (gt_*) are left exact — they only score.
    `delay_contact` also lags the contact flags by `latency_ms` (uniform latency). Uniform latency already
    degrades the estimate badly (~5x clean drift at 5 ms); leaving contact undelayed adds IMU/encoder-vs-
    contact skew on top of it (see run_embedded.py).

    The sample delivered at arrival time t[k] was *captured* earlier — that capture time (master sim
    clock) is exactly the query time each stream is re-sampled at, and a real embedded pipeline stamps
    it at the source. It is exposed per stream as ``stamp_imu`` (T,), ``stamp_enc`` (T, nj) and
    ``stamp_contact`` (T,) so a delay-compensating estimator can consume it (run_timing.py); the naive
    estimator ignores these keys and fuses every sample as if captured at t[k]."""
    rng = np.random.default_rng(seed)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in log.items()}
    t = np.asarray(log["t"], float)
    nj = log["enc_q"].shape[1]
    lat = latency_ms * 1e-3

    # IMU (rigid on the base): transport latency + per-sample interval jitter, then quantize.
    tq_imu = t - lat + rng.uniform(-jitter_ms * 1e-3, jitter_ms * 1e-3, len(t))
    out["gyro"] = _quantize(_interp_cols(t, log["gyro"], tq_imu), gyro_lsb)
    out["accel"] = _quantize(_interp_cols(t, log["accel"], tq_imu), accel_lsb)
    out["stamp_imu"] = tq_imu

    # Encoders: latency + a fixed per-joint stagger (CAN reads joints sequentially), then quantize angle.
    stagger = rng.uniform(0.0, stagger_ms * 1e-3, nj)
    encq, encqd = np.empty_like(log["enc_q"]), np.empty_like(log["enc_qd"])
    tq_enc = np.empty_like(log["enc_q"])
    for j in range(nj):
        tqj = t - lat - stagger[j]
        encq[:, j] = np.interp(tqj, t, log["enc_q"][:, j])
        encqd[:, j] = np.interp(tqj, t, log["enc_qd"][:, j])
        tq_enc[:, j] = tqj
    out["enc_q"] = _quantize(encq, enc_res)
    out["enc_qd"] = encqd
    out["stamp_enc"] = tq_enc

    out["stamp_contact"] = t.copy()
    if delay_contact and lat > 0 and "contact" in out:            # lag contact flags by the same latency
        shift = int(round(lat / float(t[1] - t[0])))
        src = np.maximum(np.arange(len(t)) - shift, 0)
        out["contact"] = log["contact"][src]
        out["stamp_contact"] = t[src]                             # exact capture time of the delivered flag
    return out
