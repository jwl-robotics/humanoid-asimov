"""Stage 4 follow-up — measurement-delay compensation: re-run the embedded-noise ablation grid with
the buffer-and-replay filter (timing.py) consuming the capture-time stamps, against the naive filter
that fuses every sample as if current. Same dataset, same seeds, same corruptions as run_embedded.py.

Expected: the timing rows (stagger, latency, skew) collapse toward the clean 0.32% baseline, because
given source timestamps the damage is *recoverable* — the naive filter's error was mis-ordering
information, not losing it. Quantization stays where it is (amplitude, irreversible). Consistency
(position NEES / contact NIS, both 3-dof, consistent ≈ 3) is reported per row: the naive filter under
latency is not just wrong but *confidently* wrong; compensation restores calibrated uncertainty.

    .venv/bin/python scripts/run_timing.py          # the seed-0 grid + latency sweep + figure
    .venv/bin/python scripts/run_timing.py seeds    # 3-seed corruption draws on the timing rows ->
                                                    #   medians (firms the single-seed 0.23-0.24%)
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from humanoid_asimov.embedded import apply_embedded_noise
from humanoid_asimov.estimator import ESKF, RobotKinematics, quat2mat
from humanoid_asimov.timing import ReplayFilter, StampedStream

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "walk_dataset.npz"))
RENDERS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))
OFF = dict(enc_res=0, gyro_lsb=0, accel_lsb=0)          # disable quantization for the timing ablations


def _metrics(p, log, nees, nis):
    gt = log["gt_pos"]
    ferr = float(np.linalg.norm(p[-1, :2] - gt[-1, :2]))
    path = float(np.sum(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1)))
    return dict(ferr=ferr, drift=ferr / path * 100.0,
                nees=float(np.mean(nees)), nis=float(np.mean(nis)) if len(nis) else float("nan"))


def run_naive(log, kin):
    """The Stage-4 baseline loop: every sample fused as if captured at its arrival time."""
    t = log["t"]; dt = float(t[1] - t[0]); jn = list(log["joint_names"])
    e = ESKF(kin, log["gt_pos"][0], quat2mat(log["gt_quat"][0]))
    p = np.empty((len(t), 3)); nees = np.empty(len(t)); nis = []
    for k in range(len(t)):
        e.step(log["gyro"][k], log["accel"][k], log["enc_q"][k], log["enc_qd"][k],
               jn, log["contact"][k], dt)
        p[k] = e.p
        err = log["gt_pos"][k] - e.p
        nees[k] = float(err @ np.linalg.solve(e.P[:3, :3], err))
        if e.last_nis is not None:
            nis.append(e.last_nis)
    return _metrics(p, log, nees, nis)


def run_compensated(log, kin, horizon=0.2):
    """Buffer-and-replay loop: IMU predicts span true inter-stamp intervals; the contact update is
    applied at the contact flag's capture time with encoder/gyro inputs reconstructed at that same
    instant (per-joint, undoing the stagger); the scored estimate is the now-cast at arrival time."""
    t = np.asarray(log["t"], float); dt = float(t[1] - t[0]); jn = list(log["joint_names"])
    nj = log["enc_q"].shape[1]
    s_imu = np.asarray(log.get("stamp_imu", t), float)
    s_enc = np.asarray(log["stamp_enc"], float) if "stamp_enc" in log else np.repeat(t[:, None], nj, 1)
    s_con = np.asarray(log.get("stamp_contact", t), float)

    gyro_s = StampedStream()
    encq_s = [StampedStream() for _ in range(nj)]
    encqd_s = [StampedStream() for _ in range(nj)]
    # Init on the same terms as the naive filter: the granted pose is the truth at CAPTURE time t[0],
    # so the state is labeled t[0]-dt exactly like the naive loop's (whose first predict ends at t[0]).
    # IMU samples stamped at or before that label carry pre-init information the grant already covers —
    # feeding them would integrate an extra `latency` of IMU on top of an already-current pose and plant
    # a permanent ~omega*latency attitude error (unobservable in yaw), penalizing compensation for
    # honoring its stamps. Skip them; streams still buffer every sample for input reconstruction.
    t_init = float(t[0]) - dt
    rf = ReplayFilter(ESKF(kin, log["gt_pos"][0], quat2mat(log["gt_quat"][0])), t_init, horizon=horizon)
    nis_by_stamp = {}                                    # final (most-informed) application wins

    def contact_fn(tau, flags):
        def ready():
            return (gyro_s.covers(tau) and all(s.covers(tau) for s in encq_s)
                    and all(s.covers(tau) for s in encqd_s))

        def apply(f):
            q = np.array([float(encq_s[j].at(tau)) for j in range(nj)])
            qd = np.array([float(encqd_s[j].at(tau)) for j in range(nj)])
            f.update_contact(q, qd, jn, flags, gyro_s.at(tau))
            if f.last_nis is not None:
                nis_by_stamp[tau] = f.last_nis

        return apply, ready

    p = np.empty((len(t), 3)); nees = np.empty(len(t))
    for k in range(len(t)):
        gyro_s.append(s_imu[k], log["gyro"][k])
        for j in range(nj):
            encq_s[j].append(s_enc[k, j], log["enc_q"][k, j])
            encqd_s[j].append(s_enc[k, j], log["enc_qd"][k, j])
        if s_imu[k] > t_init:                            # pre-init stamps: covered by the init grant
            rf.process_imu(s_imu[k], log["gyro"][k], log["accel"][k])
        if k == 0 or s_con[k] != s_con[k - 1]:           # a re-delivered (clamped) stamp is a duplicate
            fn, ready = contact_fn(float(s_con[k]), log["contact"][k])
            rf.process_meas(float(s_con[k]), fn, ready)
        pk, _, _, Pk = rf.state_at(float(t[k]))          # now-cast at arrival time (what control sees)
        p[k] = pk
        err = log["gt_pos"][k] - pk
        nees[k] = float(err @ np.linalg.solve(Pk[:3, :3], err))
    return _metrics(p, log, nees, list(nis_by_stamp.values())), rf


def main():
    d = np.load(DATA, allow_pickle=True)
    log = {k: d[k] for k in d.files}
    log["joint_names"] = list(d["joint_names"])
    kin = RobotKinematics()

    rows = [
        ("clean (ImuNoise only)", None),
        ("+ embedded (all effects)", dict()),
        ("+ quantization only", dict(stagger_ms=0, latency_ms=0, jitter_ms=0)),
        ("+ jitter only (0.8 ms)", dict(stagger_ms=0, latency_ms=0, **OFF)),
        ("+ stagger only (2 ms skew)", dict(latency_ms=0, jitter_ms=0, **OFF)),
        ("+ latency 5 ms (vs contact skew)", dict(stagger_ms=0, jitter_ms=0, **OFF)),
        ("+ latency 5 ms uniform (incl contact)", dict(stagger_ms=0, jitter_ms=0, delay_contact=True, **OFF)),
        ("+ latency 15 ms (vs contact skew)", dict(stagger_ms=0, jitter_ms=0, latency_ms=15, **OFF)),
        ("+ latency 15 ms uniform (incl contact)", dict(stagger_ms=0, jitter_ms=0, latency_ms=15, delay_contact=True, **OFF)),
    ]

    print(f"{'config':40} {'naive':>7} {'comp':>7}   {'NEES n/c':>13}   {'NIS n/c':>13}")
    R = {}
    for name, kw in rows:
        lg = apply_embedded_noise(log, seed=0, **kw) if kw is not None else log
        mn = run_naive(lg, kin)
        mc, rf = run_compensated(lg, kin)
        R[name] = (mn, mc)
        print(f"{name:40} {mn['drift']:6.2f}% {mc['drift']:6.2f}%   "
              f"{mn['nees']:6.1f}/{mc['nees']:5.1f}   {mn['nis']:6.1f}/{mc['nis']:5.1f}"
              f"   [{rf.n_rollbacks} rollbacks, {rf.n_clamped} clamped]")

    lats = [0, 2, 5, 8, 12, 16, 20]
    sweep = {}
    for tag, extra in (("skew", dict()), ("uniform", dict(delay_contact=True))):
        for f_tag, runner in (("naive", lambda L: run_naive(L, kin)),
                              ("comp", lambda L: run_compensated(L, kin)[0])):
            sweep[(tag, f_tag)] = [
                runner(apply_embedded_noise(log, seed=0, stagger_ms=0, jitter_ms=0,
                                            latency_ms=l, **extra, **OFF))["drift"]
                for l in lats]
            print(f"  sweep {tag}/{f_tag}: " + " ".join(f"{x:.2f}" for x in sweep[(tag, f_tag)]))

    # ---- figure ----
    labels = ["clean", "all\neffects", "quant", "jitter", "stagger\n2ms", "lat 5ms\nskew", "lat 5ms\nuniform"]
    keys = [r[0] for r in rows[:7]]
    x = np.arange(len(labels)); w = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))
    a1.bar(x - w / 2, [R[k][0]["drift"] for k in keys], w, color="#e05050", label="naive (fuse as current)")
    a1.bar(x + w / 2, [R[k][1]["drift"] for k in keys], w, color="#3aa76a", label="compensated (buffer-and-replay)")
    a1.axhline(R[keys[0]][0]["drift"], color="k", ls="--", lw=0.7, alpha=0.4)
    a1.set_xticks(x, labels); a1.set_ylabel("drift (%)"); a1.legend()
    a1.set_title("stamped replay collapses the timing rows to baseline")
    for tag, c in (("skew", "#e05050"), ("uniform", "#3fd0c9")):
        a2.plot(lats, sweep[(tag, "naive")], "-o", color=c, label=f"naive · {tag}")
        a2.plot(lats, sweep[(tag, "comp")], "--s", color="#3aa76a" if tag == "skew" else "#2a7c5f",
                label=f"compensated · {tag}")
    a2.set_xlabel("latency (ms)"); a2.set_ylabel("drift (%)"); a2.legend(); a2.grid(alpha=.3)
    a2.set_title("drift vs latency: flat once stamps are honored")
    fig.suptitle("Stage 4 — the same corruptions, naive vs delay-compensated fusion", fontsize=13)
    os.makedirs(RENDERS, exist_ok=True); fig.tight_layout()
    fig.savefig(os.path.join(RENDERS, "stage4_timing.png"), dpi=110)
    print(f"saved {os.path.join(RENDERS, 'stage4_timing.png')}")


def seeds_sweep(n=3):
    """Re-draw the corruption randomness (stagger offsets, jitter) n times on the rows where it
    matters and report naive/compensated medians — the walk dataset itself is fixed, so this firms
    the corruption-draw sensitivity of the single-seed grid numbers, not the sim's."""
    d = np.load(DATA, allow_pickle=True)
    log = {k: d[k] for k in d.files}
    log["joint_names"] = list(d["joint_names"])
    kin = RobotKinematics()
    rows = [
        ("+ embedded (all effects)", dict()),
        ("+ jitter only (0.8 ms)", dict(stagger_ms=0, latency_ms=0, **OFF)),
        ("+ stagger only (2 ms skew)", dict(latency_ms=0, jitter_ms=0, **OFF)),
        ("+ latency 5 ms (vs contact skew)", dict(stagger_ms=0, jitter_ms=0, **OFF)),
        ("+ latency 5 ms uniform (incl contact)", dict(stagger_ms=0, jitter_ms=0, delay_contact=True, **OFF)),
        ("+ latency 15 ms (vs contact skew)", dict(stagger_ms=0, jitter_ms=0, latency_ms=15, **OFF)),
        ("+ latency 15 ms uniform (incl contact)", dict(stagger_ms=0, jitter_ms=0, latency_ms=15, delay_contact=True, **OFF)),
    ]
    print(f"{'config':40} {'naive med (per-seed)':>32} {'comp med (per-seed)':>32}")
    for name, kw in rows:
        nv, cp = [], []
        for s in range(n):
            lg = apply_embedded_noise(log, seed=s, **kw)
            nv.append(run_naive(lg, kin)["drift"])
            cp.append(run_compensated(lg, kin)[0]["drift"])
        fmt = lambda v: f"{float(np.median(v)):5.2f}% ({'/'.join(f'{x:.2f}' for x in v)})"
        print(f"{name:40} {fmt(nv):>32} {fmt(cp):>32}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "seeds":
        seeds_sweep()
    else:
        main()
