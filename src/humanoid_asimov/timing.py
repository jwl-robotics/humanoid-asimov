"""Measurement-delay compensation: buffer-and-replay fusion of stamped, late-arriving measurements.

Stage 4 measured that *timing* — transport latency and inter-stream skew — is what actually breaks the
estimator (0.32% clean → ~2% at 5 ms), because the naive loop fuses every sample as if captured "now".
The principled fix, given source capture timestamps (one master clock, stamped at the sensor): keep a
short rolling history of filter states, covariances and IMU inputs; when a measurement stamped in the
past arrives, roll the filter back to its stamp, apply it there, and re-propagate the buffered IMU
inputs (and every already-applied measurement in between, in order) forward to the head. With zero
delay every path degenerates to plain predict-then-update — the naive loop — exactly.

Pieces:

- ``StampedStream`` — a causal per-stream sample buffer (stamp, value) with linear interpolation.
  Reconstructs any input (encoder angle, gyro rate) *at a measurement's capture time*: this alone
  undoes the per-joint CAN stagger, since each joint's series is queried at one common instant.
- ``ReplayFilter`` — the buffer-and-replay wrapper around an ESKF-family filter (ESKF / VioESKF /
  CalibVioESKF — anything with ``predict(gyro, accel, dt)`` and rebinding state updates). Feed it IMU
  samples at their capture stamps and measurements as ``(stamp, apply_fn)`` closures; it keeps events
  time-ordered, snapshots after every event, and replays on out-of-order arrival. Measurements stamped
  *ahead* of the propagated head (e.g. an undelayed contact flag racing a delayed IMU) wait in a
  pending queue until the IMU stream covers them, then fuse at the correct instant via a partial
  predict split of the covering IMU interval.

The filter object is snapshotted generically (every attribute except the shared kinematics engine), so
cloned / calibration-augmented states with changing covariance dimensions replay correctly.
"""
from __future__ import annotations

import bisect
import copy

import numpy as np


class StampedStream:
    """Causal buffer of (stamp, value) samples; linear interpolation at query times (edge-clamped).

    Querying exactly at a sample stamp returns that sample's value bit-exactly — the zero-delay path
    reduces to the raw sample, not a numerically-perturbed copy.
    """

    def __init__(self, horizon=0.5):
        self.horizon = float(horizon)
        self.t = []                     # stamps, strictly increasing
        self.v = []

    def append(self, stamp, value):
        stamp = float(stamp)
        if self.t and stamp <= self.t[-1]:
            raise ValueError(f"non-monotonic stamp {stamp} after {self.t[-1]}")
        self.t.append(stamp)
        self.v.append(np.array(value, float))
        floor = stamp - self.horizon    # prune, always keeping two samples to bracket old queries
        k = 0
        while k < len(self.t) - 2 and self.t[k + 1] <= floor:
            k += 1
        if k:
            del self.t[:k]
            del self.v[:k]

    def covers(self, stamp):
        return bool(self.t) and self.t[0] <= stamp <= self.t[-1]

    def at(self, stamp):
        i = bisect.bisect_left(self.t, stamp)
        if i >= len(self.t):
            return self.v[-1].copy()
        if self.t[i] == stamp or i == 0:
            return self.v[i].copy()
        t0, t1 = self.t[i - 1], self.t[i]
        a = (stamp - t0) / (t1 - t0)
        return (1.0 - a) * self.v[i - 1] + a * self.v[i]


_SHARED_ATTRS = ("kin",)                # shared engines, excluded from snapshots (stateless between calls)


def _snapshot(filt):
    return {k: copy.deepcopy(v) for k, v in filt.__dict__.items() if k not in _SHARED_ATTRS}


def _restore(filt, snap):
    for k in list(filt.__dict__):
        if k not in _SHARED_ATTRS:
            del filt.__dict__[k]
    for k, v in snap.items():
        filt.__dict__[k] = copy.deepcopy(v)


class ReplayFilter:
    """Buffer-and-replay delayed-measurement fusion around an ESKF-family filter.

    Events (IMU samples, measurement closures) are kept time-ordered over a rolling ``horizon``; each
    processed event stores a full post-event snapshot. A measurement whose stamp lands before the head
    restores the snapshot preceding its slot and re-executes everything after it — IMU predicts *and*
    the measurements that had already been applied — in stamp order. IMU samples propagate over their
    true inter-stamp intervals (the sample stamped ``s`` covers the interval ending at ``s``, matching
    the naive loop's convention); a measurement stamped mid-interval splits the covering predict.

    ``process_meas`` takes ``apply_fn(filt)`` — e.g. ``lambda f: f.update_contact(...)`` with inputs
    reconstructed at the stamp via ``StampedStream.at`` — plus an optional ``ready_fn`` that defers
    application until every input stream covers the stamp.
    """

    def __init__(self, filt, t0, horizon=0.2):
        self.f = filt
        self.horizon = float(horizon)
        self.head = float(t0)           # capture time the filter state is propagated to
        self.events = []                # processed, stamp-ordered
        self.pending = []               # measurements waiting on the head / their input streams
        self._seq = 0                   # tie-break: stable order for equal stamps
        self.base = (self.head, _snapshot(filt))     # rollback floor (oldest restorable state)
        self.n_rollbacks = 0
        self.n_replayed = 0             # events re-executed by rollbacks
        self.n_clamped = 0              # measurements older than the buffer floor (applied there)

    # ---- feeding -----------------------------------------------------------
    def process_imu(self, stamp, gyro, accel):
        """One IMU sample at its capture stamp: predict from the head to the stamp, then let any
        pending measurement the head now covers fuse in (replaying as needed)."""
        stamp = float(stamp)
        if stamp <= self.head:
            raise ValueError(f"IMU stamp {stamp} not ahead of head {self.head}")
        ev = {"stamp": stamp, "prio": 0, "seq": self._next_seq(), "kind": "imu",
              "gyro": np.array(gyro, float), "accel": np.array(accel, float)}
        self._exec(ev, len(self.events))
        self.events.append(ev)
        self._prune()
        self.flush()

    def process_meas(self, stamp, apply_fn, ready_fn=None, label=None):
        """One measurement closure at its capture stamp. Fused immediately if the head covers the
        stamp (rolling back if the stamp is in the past), else queued until it does."""
        ev = {"stamp": float(stamp), "prio": 1, "seq": self._next_seq(), "kind": "meas",
              "apply": apply_fn, "ready": ready_fn, "label": label}
        if ev["stamp"] > self.head or (ready_fn is not None and not ready_fn()):
            bisect.insort(self.pending, ev, key=lambda e: (e["stamp"], e["seq"]))
            return
        self._insert(ev)

    def flush(self):
        """Fuse every pending measurement whose stamp the head has passed and whose inputs are ready."""
        rest = []
        for ev in self.pending:
            if ev["stamp"] <= self.head and (ev["ready"] is None or ev["ready"]()):
                self._insert(ev)
            else:
                rest.append(ev)
        self.pending = rest

    # ---- output ------------------------------------------------------------
    def state_at(self, t_query):
        """Now-cast (p, v, R, P) at ``t_query`` ≥ head: mean *and* covariance predicted across the
        no-data gap with the newest IMU sample held (the filter itself is left untouched)."""
        last = next((e for e in reversed(self.events) if e["kind"] == "imu"), None)
        if last is None or t_query <= self.head:
            return self.f.p.copy(), self.f.v.copy(), self.f.R.copy(), self.f.P.copy()
        snap = _snapshot(self.f)
        self.f.predict(last["gyro"], last["accel"], float(t_query) - self.head)
        out = (self.f.p.copy(), self.f.v.copy(), self.f.R.copy(), self.f.P.copy())
        _restore(self.f, snap)
        return out

    # ---- internals ---------------------------------------------------------
    def _next_seq(self):
        self._seq += 1
        return self._seq

    @staticmethod
    def _key(ev):
        return (ev["stamp"], ev["prio"], ev["seq"])

    def _insert(self, ev):
        """Place a measurement at its stamp-ordered slot; if that slot is in the past, roll back to
        the snapshot before it and re-execute everything after it in order."""
        if ev["stamp"] < self.base[0]:
            self.n_clamped += 1         # older than the buffer floor: fuses at the floor state
        keys = [self._key(e) for e in self.events]
        idx = bisect.bisect_left(keys, self._key(ev))
        self.events.insert(idx, ev)
        if idx == len(self.events) - 1:              # newest slot: no history to rewind
            self._exec(ev, idx)
            return
        self.n_rollbacks += 1
        head, snap = (self.events[idx - 1]["head_after"], self.events[idx - 1]["snap"]) if idx else self.base
        self.head = head
        _restore(self.f, snap)
        for i in range(idx, len(self.events)):
            self._exec(self.events[i], i)
            self.n_replayed += 1
        self.n_replayed -= 1                          # the inserted event itself is not a re-execution

    def _exec(self, ev, idx):
        """Execute one event at list position ``idx`` from the current (restored) filter state."""
        if ev["kind"] == "imu":
            self.f.predict(ev["gyro"], ev["accel"], ev["stamp"] - self.head)
            self.head = ev["stamp"]
        else:
            gap = ev["stamp"] - self.head
            if gap > 0.0:               # mid-interval stamp: split the covering IMU predict at it
                nxt = next((e for e in self.events[idx + 1:] if e["kind"] == "imu"), None)
                if nxt is not None:     # (no later IMU can only mean gap == 0 by the pending gate)
                    self.f.predict(nxt["gyro"], nxt["accel"], gap)
                    self.head = ev["stamp"]
            ev["apply"](self.f)
        ev["head_after"] = self.head
        ev["snap"] = _snapshot(self.f)

    def _prune(self):
        floor = self.head - self.horizon
        while len(self.events) > 1 and self.events[0]["stamp"] < floor:
            ev = self.events.pop(0)
            self.base = (ev["head_after"], ev["snap"])
