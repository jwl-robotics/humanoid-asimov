"""Correctness tests for the buffer-and-replay delayed-measurement fusion (timing.py).

The contract under test: a measurement delivered LATE but fused through the replay path must produce
the same posterior as an oracle filter that received it on time — and with zero delay the whole
machinery must reduce to the plain predict-then-update loop. Run: .venv/bin/python -m pytest -q
"""
import numpy as np
import pytest

from humanoid_asimov.estimator import ESKF, RobotKinematics, quat2mat
from humanoid_asimov.sensors import ImuNoise
from humanoid_asimov.timing import ReplayFilter, StampedStream
from humanoid_asimov.vio import VioESKF
from humanoid_asimov.walk import simulate

ATOL, RTOL = 1e-13, 1e-9      # "identical up to replay bookkeeping" — far below any physical scale


@pytest.fixture(scope="module")
def log():
    _, _, log, _ = simulate(seconds=2.0, settle=0.5, imu_noise=ImuNoise(), seed=0)
    return log


@pytest.fixture(scope="module")
def kin():
    return RobotKinematics()


def _new_streams():
    return {k: StampedStream() for k in ("gyro", "enc_q", "enc_qd")}


def _feed_streams(st, log, k):
    tk = float(log["t"][k])
    st["gyro"].append(tk, log["gyro"][k])
    st["enc_q"].append(tk, log["enc_q"][k])
    st["enc_qd"].append(tk, log["enc_qd"][k])


def _contact_fn(st, jn, flags, stamp):
    def ready():
        return all(st[s].covers(stamp) for s in ("gyro", "enc_q", "enc_qd"))

    def apply(f):
        f.update_contact(st["enc_q"].at(stamp), st["enc_qd"].at(stamp), jn, flags, st["gyro"].at(stamp))

    return apply, ready


def _assert_same(f_replay, f_oracle):
    for name in ("p", "v", "R", "bg", "ba", "bc", "P"):
        np.testing.assert_allclose(getattr(f_replay, name), getattr(f_oracle, name),
                                   rtol=RTOL, atol=ATOL, err_msg=name)


def test_stamped_stream_exact_at_samples():
    s = StampedStream(horizon=10.0)
    for i in range(5):
        s.append(0.25 * i, np.array([float(i), -float(i)]))          # exactly-representable stamps
    assert np.array_equal(s.at(0.75), np.array([3.0, -3.0]))         # bit-exact at a sample stamp
    np.testing.assert_allclose(s.at(0.625), [2.5, -2.5], rtol=0, atol=1e-15)
    assert s.covers(0.0) and s.covers(1.0) and not s.covers(1.01)
    p = StampedStream(horizon=0.5)                                   # rolling window prunes the tail...
    for i in range(5):
        p.append(0.25 * i, np.array([float(i)]))
    assert not p.covers(0.0) and p.covers(0.5) and p.covers(1.0)     # ...but keeps everything in-horizon


def test_zero_delay_reduces_to_naive(log, kin):
    """With every stamp equal to its arrival time, the replay filter must match the plain
    predict-then-update loop (current behavior) — no rollbacks, no pending, same posterior."""
    t, jn = log["t"], list(log["joint_names"])
    dt = float(t[1] - t[0])
    p0, R0 = log["gt_pos"][0], quat2mat(log["gt_quat"][0])

    naive = ESKF(kin, p0, R0)
    for k in range(len(t)):
        naive.step(log["gyro"][k], log["accel"][k], log["enc_q"][k], log["enc_qd"][k],
                   jn, log["contact"][k], dt)

    rep = ReplayFilter(ESKF(kin, p0, R0), float(t[0]) - dt)
    st = _new_streams()
    for k in range(len(t)):
        _feed_streams(st, log, k)
        rep.process_imu(float(t[k]), log["gyro"][k], log["accel"][k])
        fn, ready = _contact_fn(st, jn, log["contact"][k], float(t[k]))
        rep.process_meas(float(t[k]), fn, ready)

    assert rep.n_rollbacks == 0 and rep.n_clamped == 0 and not rep.pending
    # sole numerical difference: per-stamp dt (t[k]-t[k-1]) vs the constant dt — identical to ~1e-15
    for name in ("p", "v", "R", "bg", "ba", "bc", "P"):
        np.testing.assert_allclose(getattr(rep.f, name), getattr(naive, name),
                                   rtol=1e-7, atol=1e-9, err_msg=name)


def _oracle_per_stamp(kin, log, jn, head0):
    """Plain loop, but propagating over true inter-stamp intervals (the replay filter's convention)."""
    f = ESKF(kin, log["gt_pos"][0], quat2mat(log["gt_quat"][0]))
    prev = head0
    for k in range(len(log["t"])):
        tk = float(log["t"][k])
        f.predict(log["gyro"][k], log["accel"][k], tk - prev)
        f.update_contact(log["enc_q"][k], log["enc_qd"][k], jn, log["contact"][k], log["gyro"][k])
        prev = tk
    return f


def test_late_contact_matches_oracle(log, kin):
    """Every contact measurement delivered 7 samples late (stamps true) must reproduce the on-time
    oracle's posterior exactly — the replay re-derives the same predict/update sequence."""
    t, jn = log["t"], list(log["joint_names"])
    dt = float(t[1] - t[0])
    head0 = float(t[0]) - dt
    DELAY = 7

    oracle = _oracle_per_stamp(kin, log, jn, head0)

    rep = ReplayFilter(ESKF(kin, log["gt_pos"][0], quat2mat(log["gt_quat"][0])), head0)
    st = _new_streams()

    def deliver(j):
        fn, ready = _contact_fn(st, jn, log["contact"][j], float(t[j]))
        rep.process_meas(float(t[j]), fn, ready)

    for k in range(len(t)):
        _feed_streams(st, log, k)
        rep.process_imu(float(t[k]), log["gyro"][k], log["accel"][k])
        if k >= DELAY:
            deliver(k - DELAY)
    for j in range(len(t) - DELAY, len(t)):
        deliver(j)

    assert rep.n_rollbacks > 0                      # the late path was actually exercised
    assert not rep.pending
    _assert_same(rep.f, oracle)


def test_mid_interval_stamp_matches_split_oracle(log, kin):
    """A late measurement stamped BETWEEN two IMU samples must fuse via a split of the covering
    predict — identical to an oracle that hand-executes predict(α·dt) → update → predict((1−α)·dt)."""
    t, jn = log["t"], list(log["joint_names"])
    dt = float(t[1] - t[0])
    head0 = float(t[0]) - dt
    J = 400
    tau = float(t[J]) + 0.3 * dt                    # mid-interval capture time

    # oracle inputs at tau: same interpolation, but an unpruned horizon (it is fed the whole log up
    # front, so the default rolling window would have discarded tau's bracket before the query)
    st_o = {k: StampedStream(horizon=1e9) for k in ("gyro", "enc_q", "enc_qd")}
    for k in range(len(t)):
        _feed_streams(st_o, log, k)

    oracle = ESKF(kin, log["gt_pos"][0], quat2mat(log["gt_quat"][0]))
    prev = head0
    for k in range(len(t)):
        tk = float(t[k])
        if k == J + 1:                               # split the covering predict at tau
            oracle.predict(log["gyro"][k], log["accel"][k], tau - prev)
            oracle.update_contact(st_o["enc_q"].at(tau), st_o["enc_qd"].at(tau), jn,
                                  log["contact"][J], st_o["gyro"].at(tau))
            prev = tau
        oracle.predict(log["gyro"][k], log["accel"][k], tk - prev)
        oracle.update_contact(log["enc_q"][k], log["enc_qd"][k], jn, log["contact"][k], log["gyro"][k])
        prev = tk

    rep = ReplayFilter(ESKF(kin, log["gt_pos"][0], quat2mat(log["gt_quat"][0])), head0)
    st = _new_streams()
    for k in range(len(t)):
        _feed_streams(st, log, k)
        rep.process_imu(float(t[k]), log["gyro"][k], log["accel"][k])
        fn, ready = _contact_fn(st, jn, log["contact"][k], float(t[k]))
        rep.process_meas(float(t[k]), fn, ready)
        if k == J + 5:                               # deliver the extra mid-interval measurement late
            fn, ready = _contact_fn(st, jn, log["contact"][J], tau)
            rep.process_meas(tau, fn, ready)

    assert rep.n_rollbacks > 0
    _assert_same(rep.f, oracle)


def test_late_vision_matches_oracle(log, kin):
    """Vision path: a stochastic clone and a relative-rotation update (18→21-dim covariance) both
    delivered late must match the on-time oracle — replay across a dimension-changing event."""
    t, jn = log["t"], list(log["joint_names"])
    dt = float(t[1] - t[0])
    head0 = float(t[0]) - dt
    C, U, LATE = 300, 900, 10
    R_BC = np.eye(3)
    R_meas = (quat2mat(log["gt_quat"][C]).T @ quat2mat(log["gt_quat"][U]))   # R_BC = I ⇒ camera Δ-rotation

    def make(): return VioESKF(kin, log["gt_pos"][0], quat2mat(log["gt_quat"][0]), vis_gate=1e9)

    applied_o, applied_r = [], []
    oracle = make()
    prev = head0
    for k in range(len(t)):
        tk = float(t[k])
        oracle.predict(log["gyro"][k], log["accel"][k], tk - prev)
        oracle.update_contact(log["enc_q"][k], log["enc_qd"][k], jn, log["contact"][k], log["gyro"][k])
        if k == C:
            oracle.clone(R_BC)
        if k == U:
            applied_o.append(oracle.update_rel_rot(R_meas, R_BC))
        prev = tk

    rep = ReplayFilter(make(), head0)
    st = _new_streams()
    for k in range(len(t)):
        _feed_streams(st, log, k)
        rep.process_imu(float(t[k]), log["gyro"][k], log["accel"][k])
        fn, ready = _contact_fn(st, jn, log["contact"][k], float(t[k]))
        rep.process_meas(float(t[k]), fn, ready)
        if k == C + LATE:                            # clone delivered late, stamped at its capture time
            rep.process_meas(float(t[C]), lambda f: f.clone(R_BC))
        if k == U + LATE:                            # vision update delivered late as well
            rep.process_meas(float(t[U]), lambda f: applied_r.append(f.update_rel_rot(R_meas, R_BC)))

    assert applied_o == [True] and True in applied_r  # the update genuinely fused in both filters
    assert rep.n_rollbacks >= 2
    for name in ("p", "v", "R", "bg", "R_A"):
        np.testing.assert_allclose(getattr(rep.f, name), getattr(oracle, name),
                                   rtol=RTOL, atol=ATOL, err_msg=name)
    assert rep.f.P.shape == oracle.P.shape == (21, 21)
    np.testing.assert_allclose(rep.f.P, oracle.P, rtol=RTOL, atol=ATOL)


def test_nowcast_moves_state_forward(log, kin):
    """state_at() must extrapolate the mean across the no-data gap without touching the filter."""
    t, jn = log["t"], list(log["joint_names"])
    dt = float(t[1] - t[0])
    rep = ReplayFilter(ESKF(kin, log["gt_pos"][0], quat2mat(log["gt_quat"][0])), float(t[0]) - dt)
    st = _new_streams()
    for k in range(200):
        _feed_streams(st, log, k)
        rep.process_imu(float(t[k]), log["gyro"][k], log["accel"][k])
        fn, ready = _contact_fn(st, jn, log["contact"][k], float(t[k]))
        rep.process_meas(float(t[k]), fn, ready)
    p_head = rep.f.p.copy()
    P_head = rep.f.P.copy()
    p_now, _, _, P_now = rep.state_at(float(t[199]) + 5e-3)
    assert np.array_equal(rep.f.p, p_head) and np.array_equal(rep.f.P, P_head)  # untouched
    assert np.linalg.norm(p_now - p_head) > 0            # mean extrapolated across the gap
    assert np.trace(P_now[:3, :3]) >= np.trace(P_head[:3, :3])                  # uncertainty grew
