"""Stage 5b — the closed-loop showdown: the same 3-circuit waypoint course driven by four estimator
configurations, all navigating on their OWN estimate (estimation error feeds back into the path):

    eskf    ESKF, clean timing                  — the Stage-5 baseline, now over 3 circuits
    vio     VIO in the loop                     — head-cam frames rendered INSIDE the control loop at
                                                  keyframe cadence (2 Hz) and fused as in Stage 2
    naive   ESKF + 5 ms latency / 2 ms stagger  — embedded.py-style corruption applied ONLINE, fused
                                                  as if current (the Stage-4 failure mode, closed-loop)
    replay  same corruption, compensated        — stamped samples through the buffer-and-replay
                                                  ReplayFilter (timing.py); nav steers on the now-cast

Honest metrics: per-circuit TRUE closest approach to each waypoint, and the est-vs-truth gap over
time. Final-position error is NOT the headline — the course returns home, which flatters it.

    .venv/bin/python scripts/run_nav5b.py eskf|vio|naive|replay   # one variant -> data/nav5b_<v>.npz
    .venv/bin/python scripts/run_nav5b.py all                     # any missing variants, then aggregate
    .venv/bin/python scripts/run_nav5b.py report                  # aggregate only (figures + matrix +
                                                                  #   data/nav5b_tracks.json for nav.html)
"""
import json
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from humanoid_asimov.estimator import ESKF, RobotKinematics, exp_so3, quat2mat
from humanoid_asimov.nav import WaypointNavigator
from humanoid_asimov.sensors import ImuNoise
from humanoid_asimov.timing import ReplayFilter, StampedStream
from humanoid_asimov.vio import VioESKF
from humanoid_asimov.vision import (AnchorRotationEstimator, CameraModel, FeatureTracker,
                                    HeadCamExtrinsics, _LK)
from humanoid_asimov.walk import simulate

RENDERS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

WAYPOINTS = [(3.0, 0.0), (3.0, 2.5), (0.3, 2.5), (0.3, -0.2)]
CIRCUITS = 3
SECONDS = 75.0                # 3 circuits at ~15 s each + margin; trimmed at completion
SEED = 0
# sig_contact=0.15 for ALL variants. run_nav.py's 0.05 is best for its SHORT single circuit, but a
# measured check here (data/nav5b_eskf005_check.npz) showed the Stage-2 turn-slip b_g runaway DOES
# accumulate over 3 circuits of repeated 90-degree turns: at 0.05 the ESKF reaches only 10/12
# waypoints with ~98 degrees of terminal yaw error; at 0.15 it completes 12/12 in 44 s at 3.6.
SIG_CONTACT = 0.15
# Corruption for naive/replay: transport latency + per-joint CAN stagger, contact undelayed (the
# worst-case skew row of the Stage-4 grid), no quantization/jitter — the pure-timing ablation.
LATENCY = 0.005
STAGGER_MS = 2.0
# VIO front-end in the loop: every rendered frame IS a keyframe (2 Hz), Stage-2 anchor rules.
KF_HZ = 2.0
ANCHOR_MAX_AGE, ANCHOR_MIN_SHARED = 2.0, 30

VARIANTS = ("eskf", "vio", "naive", "replay")
LABELS = {"eskf": "ESKF · clean", "vio": "VIO · clean",
          "naive": "ESKF · 5 ms latency, naive", "replay": "ESKF · 5 ms latency, replay-compensated"}
COLORS = {"eskf": "#e05050", "vio": "#3fd0c9", "naive": "#d9a23f", "replay": "#3aa76a"}


class SeededTracker(FeatureTracker):
    """FeatureTracker with a rotation-seeded LK initial guess. At 2 Hz frame spacing the inter-frame
    pixel motion during a turn exceeds the LK pyramid's capture range; predicting each feature through
    the gyro-integrated camera rotation (onboard data only) and refining from there keeps the tracks
    alive. The forward-backward check is unchanged — a bad seed is culled, never trusted."""

    def __init__(self, cam, **kw):
        super().__init__(**kw)
        self.cam = cam

    def process(self, gray, keyframe=False, R_pn=None):
        if self.prev is not None and len(self.pts):
            seed = self.pts
            if R_pn is not None:               # b_prev = R_pn · b_now  =>  b_now rows = b_prev @ R_pn
                bp = self.cam.bearings(self.pts) @ R_pn
                uv = self.cam.project(bp)
                ok = ((bp[:, 2] > 0.05) & (uv[:, 0] > 0) & (uv[:, 0] < self.w)
                      & (uv[:, 1] > 0) & (uv[:, 1] < self.h))
                seed = np.where(ok[:, None], uv, self.pts).astype(np.float32)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts, seed.copy(),
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **_LK)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, nxt, None, **_LK)
            fb = np.linalg.norm(self.pts - back, axis=1)
            b = self.border
            good = ((st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_thresh)
                    & (nxt[:, 0] > b) & (nxt[:, 0] < self.w - b)
                    & (nxt[:, 1] > b) & (nxt[:, 1] < self.h - b))
            self.pts = nxt[good].astype(np.float32)
            self.ids = self.ids[good]
        if keyframe:
            self._refill(gray)
        self.prev = gray
        return {int(i): p.copy() for i, p in zip(self.ids, self.pts)}


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def run_variant(variant, seed=SEED, sig_contact=SIG_CONTACT, tag=None):
    kin = RobotKinematics()
    ext = HeadCamExtrinsics(kin)
    cam = CameraModel()
    nav = WaypointNavigator(WAYPOINTS * CIRCUITS)
    corrupt = variant in ("naive", "replay")
    stagger = np.random.default_rng(seed).uniform(0.0, STAGGER_MS * 1e-3, 15)

    st = {"e": None, "rf": None, "jn": None, "nj": None,
          # VIO front-end state
          "tracker": None, "anchor": None, "R_cum": np.eye(3), "R_WC_prev": None,
          "anchor_Rcum": None, "anchor_RBC": None, "n_kf": 0, "n_meas": 0, "n_fused": 0,
          # delayed-delivery buffers (true samples in, delayed reads out)
          "imu_s": None, "enc_s": None, "encd_s": None,
          # replay-side reconstruction streams (delivered samples at their capture stamps)
          "gyro_rs": None, "encq_rs": None, "encqd_rs": None}
    est, navi = [], []

    def hook(t, gyro, accel, enc_q, enc_qd, contact, dt, ctrl, gt_pos, gt_quat, frame=None):
        nj = len(enc_q)
        if st["e"] is None:                                    # init once at the known start pose
            st["jn"], st["nj"] = jn_holder[0], nj
            p0, R0 = gt_pos.copy(), quat2mat(gt_quat)
            if variant == "vio":
                st["e"] = VioESKF(kin, p0, R0, sig_contact=sig_contact)
                st["tracker"] = SeededTracker(cam, fb_thresh=1.5)
                st["anchor"] = AnchorRotationEstimator(cam)
            else:
                st["e"] = ESKF(kin, p0, R0, sig_contact=sig_contact)
            if corrupt:
                st["imu_s"] = StampedStream(horizon=0.5)
                st["enc_s"] = [StampedStream(horizon=0.5) for _ in range(nj)]
                st["encd_s"] = [StampedStream(horizon=0.5) for _ in range(nj)]
            if variant == "replay":
                st["rf"] = ReplayFilter(st["e"], float(t) - dt, horizon=0.2)
                st["gyro_rs"] = StampedStream(horizon=0.5)
                st["encq_rs"] = [StampedStream(horizon=0.5) for _ in range(nj)]
                st["encqd_rs"] = [StampedStream(horizon=0.5) for _ in range(nj)]
        e, jn = st["e"], st["jn"]

        if corrupt:                                            # what the pipeline DELIVERS at arrival t
            st["imu_s"].append(t, np.concatenate([gyro, accel]))
            for j in range(nj):
                st["enc_s"][j].append(t, enc_q[j])
                st["encd_s"][j].append(t, enc_qd[j])
            d_imu = st["imu_s"].at(t - LATENCY)
            d_gyro, d_accel = d_imu[:3], d_imu[3:]
            d_q = np.array([float(st["enc_s"][j].at(t - LATENCY - stagger[j])) for j in range(nj)])
            d_qd = np.array([float(st["encd_s"][j].at(t - LATENCY - stagger[j])) for j in range(nj)])
        else:
            d_gyro, d_accel, d_q, d_qd = gyro, accel, enc_q, enc_qd

        if variant == "replay":
            rf = st["rf"]
            s_imu = t - LATENCY                                # source capture stamps (master clock)
            st["gyro_rs"].append(s_imu, d_gyro)
            for j in range(nj):
                st["encq_rs"][j].append(t - LATENCY - stagger[j], d_q[j])
                st["encqd_rs"][j].append(t - LATENCY - stagger[j], d_qd[j])
            if s_imu > rf.base[0]:                             # pre-init stamps: covered by the init grant
                rf.process_imu(s_imu, d_gyro, d_accel)
            tau, flags = float(t), list(contact)               # contact flag is undelayed (worst-case skew)
            gyro_rs, encq_rs, encqd_rs = st["gyro_rs"], st["encq_rs"], st["encqd_rs"]

            def ready():
                return (gyro_rs.covers(tau) and all(s.covers(tau) for s in encq_rs)
                        and all(s.covers(tau) for s in encqd_rs))

            def apply(f):
                q = np.array([float(encq_rs[j].at(tau)) for j in range(nj)])
                qd = np.array([float(encqd_rs[j].at(tau)) for j in range(nj)])
                f.update_contact(q, qd, jn, flags, gyro_rs.at(tau)[:3])

            rf.process_meas(tau, apply, ready)
            px, _, Rx, _ = rf.state_at(float(t))               # now-cast: what control acts on
        else:
            e.step(d_gyro, d_accel, d_q, d_qd, jn, contact, dt)
            if variant == "vio":
                st["R_cum"] = st["R_cum"] @ exp_so3(np.asarray(gyro) * dt)   # raw-gyro attitude (prior/seed)
                if frame is not None:
                    _vio_keyframe(t, frame, enc_q)
            px, Rx = e.p, e.R

        x, y, yaw = float(px[0]), float(px[1]), _yaw(Rx)
        nav.command(x, y, yaw)
        ctrl.heading = (lambda tt, h=nav.desired_heading: h)
        est.append((t, x, y, yaw)); navi.append(nav.i)

    def _vio_keyframe(t, frame, enc_q):
        e, jn, tracker, anchor = st["e"], st["jn"], st["tracker"], st["anchor"]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        R_BC_j = ext.compute(enc_q, jn)[0]
        R_WC_j = st["R_cum"] @ R_BC_j
        R_pn = st["R_WC_prev"].T @ R_WC_j if st["R_WC_prev"] is not None else None
        tracks = tracker.process(gray, keyframe=True, R_pn=R_pn)
        st["R_WC_prev"] = R_WC_j
        st["n_kf"] += 1
        m, n_shared = None, 0
        if st["anchor_Rcum"] is not None:
            R_prior = st["anchor_RBC"].T @ (st["anchor_Rcum"].T @ st["R_cum"]) @ R_BC_j
            m = anchor.estimate(float(t), tracks, R_prior=R_prior)
            n_shared = sum(1 for x in tracks if x in anchor.anchor_uv)
        if m is not None:
            st["n_meas"] += 1
            st["n_fused"] += bool(e.update_rel_rot(m["R"], R_BC_j))
        if (st["anchor_Rcum"] is None or m is None
                or (float(t) - anchor.anchor_t) > ANCHOR_MAX_AGE or n_shared < ANCHOR_MIN_SHARED):
            e.clone(R_BC_j)
            anchor.set_anchor(float(t), tracks)
            st["anchor_Rcum"], st["anchor_RBC"] = st["R_cum"].copy(), R_BC_j

    jn_holder = [None]
    render = dict(render_camera="head_cam", fps=KF_HZ) if variant == "vio" else {}

    # simulate() resolves joint names after build; grab them via a tiny pre-run of Sensors
    from humanoid_asimov.scene import build_model
    from humanoid_asimov.sensors import Sensors
    jn_holder[0] = list(Sensors(build_model(add_environment=True)).joint_names)

    _, _, log, _ = simulate(seconds=SECONDS, environment=True, imu_noise=ImuNoise(), seed=seed,
                            heading=lambda t: 0.0, control_hook=hook, **render)

    est = np.array(est); navi = np.array(navi); gt = log["gt_pos"][:, :2]
    gt_yaw = np.array([_yaw(quat2mat(q)) for q in log["gt_quat"]])
    done = navi >= len(WAYPOINTS) * CIRCUITS
    end = int(np.argmax(done)) if done.any() else len(est) - 1
    est, gt, gt_yaw, navi = est[:end + 1], gt[:end + 1], gt_yaw[:end + 1], navi[:end + 1]

    extra = {}
    if variant == "vio":
        extra = dict(n_kf=st["n_kf"], n_meas=st["n_meas"], n_fused=st["n_fused"])
        print(f"  vio front-end: {st['n_kf']} keyframes, {st['n_meas']} measurements, "
              f"{st['n_fused']} fused")
    if variant == "replay":
        rf = st["rf"]
        extra = dict(rollbacks=rf.n_rollbacks, replayed=rf.n_replayed, clamped=rf.n_clamped)
        print(f"  replay: {rf.n_rollbacks} rollbacks, {rf.n_replayed} events re-executed, "
              f"{rf.n_clamped} clamped")

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, f"nav5b_{tag or variant}.npz")
    np.savez_compressed(out, t=est[:, 0], est=est[:, 1:], gt=gt, gt_yaw=gt_yaw, navi=navi,
                        reached=int(navi.max()), extra=json.dumps(extra))
    print(f"saved {out}")
    return out


# ---------------------------------------------------------------- metrics + report
def _metrics(d):
    """Per-circuit true closest approach per waypoint + est-vs-truth gap stats."""
    t, est, gt, navi = d["t"], d["est"], d["gt"], d["navi"]
    nwp = len(WAYPOINTS)
    approach = np.full((CIRCUITS, nwp), np.nan)
    for gi in range(nwp * CIRCUITS):
        w = navi == gi
        if w.any():
            dist = np.linalg.norm(gt[w] - np.array(WAYPOINTS[gi % nwp]), axis=1)
            approach[gi // nwp, gi % nwp] = float(dist.min())
    gap = np.linalg.norm(est[:, :2] - gt, axis=1)
    yaw_err = np.abs(np.degrees((d["est"][:, 2] - d["gt_yaw"] + np.pi) % (2 * np.pi) - np.pi))
    return dict(reached=int(d["reached"]), total=nwp * CIRCUITS, t_end=float(t[-1]),
                approach_cm=approach * 100,
                worst_cm=[float(np.nanmax(approach[c]) * 100) for c in range(CIRCUITS)],
                mean_cm=float(np.nanmean(approach) * 100),
                gap_med_cm=float(np.median(gap) * 100), gap_max_cm=float(gap.max() * 100),
                gap_final_cm=float(gap[-1] * 100), yaw_final_deg=float(yaw_err[-1]),
                extra=json.loads(str(d["extra"])))


def _plot_tracks(ax, runs):
    wp = np.array(WAYPOINTS)
    ax.plot(wp[:, 0], wp[:, 1], "s", color="#888", ms=13, zorder=5, label="waypoints")
    for k, (gx, gy) in enumerate(WAYPOINTS):
        ax.annotate(str(k + 1), (gx, gy), color="w", ha="center", va="center", fontsize=9, zorder=6)
    for v in VARIANTS:
        g = runs[v]["gt"]
        ax.plot(g[:, 0], g[:, 1], color=COLORS[v], lw=1.3, label=LABELS[v], alpha=0.9)
    ax.plot(*runs["eskf"]["gt"][0], "o", color="w", ms=8, label="start")
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.legend(loc="center", fontsize=8, framealpha=0.9)      # course interior is empty
    ax.set_title(f"true paths — {CIRCUITS} circuits on the robot's own estimate")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


def _plot_gap(ax, runs):
    for v in VARIANTS:
        d = runs[v]
        gap = np.linalg.norm(d["est"][:, :2] - d["gt"], axis=1) * 100
        ax.plot(d["t"], gap, color=COLORS[v], lw=1.4, label=LABELS[v])
    ax.set_xlabel("t (s)"); ax.set_ylabel("est-vs-truth gap (cm)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("estimate-vs-truth gap — what the planner does not know")


def _write_tracks(runs):
    """Decimate the 500 Hz per-run tracks to the 10 Hz grid the interactive nav.html chart uses
    (the vio.html Win chart's cadence) + always keep each run's true final sample. Full-rate data
    stays in the npz artifacts; build_dashboard.py inlines this JSON into the page."""
    step = 50
    out = dict(dt=0.002 * step, t0=round(float(runs[VARIANTS[0]]["t"][0]), 3),
               waypoints=[list(w) for w in WAYPOINTS], wp_per=len(WAYPOINTS), circuits=CIRCUITS,
               seed=SEED, labels=LABELS, colors=COLORS, runs={})
    for v in VARIANTS:
        d = runs[v]
        idx = np.arange(0, len(d["t"]), step)
        if idx[-1] != len(d["t"]) - 1:
            idx = np.append(idx, len(d["t"]) - 1)
        out["runs"][v] = dict(
            t_end=round(float(d["t"][-1]), 3), reached=int(d["reached"]),
            gt=np.round(d["gt"][idx], 3).tolist(),
            est=np.round(d["est"][idx, :2], 3).tolist(),
            navi=d["navi"][idx].astype(int).tolist())
    p = os.path.join(DATA, "nav5b_tracks.json")
    with open(p, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"saved {p}")


def report():
    runs = {}
    for v in VARIANTS:
        p = os.path.join(DATA, f"nav5b_{v}.npz")
        if not os.path.exists(p):
            print(f"missing {p} — run `run_nav5b.py {v}` first"); return
        runs[v] = dict(np.load(p, allow_pickle=False))

    M = {v: _metrics(runs[v]) for v in VARIANTS}
    print(f"\n=== Stage 5b — closed-loop showdown ({CIRCUITS} circuits, seed {SEED}) ===")
    print(f"{'variant':38} {'wp':>5} {'time':>7} {'worst approach/circuit (cm)':>28} "
          f"{'mean':>6} {'gap med':>8} {'gap max':>8} {'yaw end':>8}")
    for v in VARIANTS:
        m = M[v]
        worst = " / ".join(f"{w:5.1f}" for w in m["worst_cm"])
        print(f"{LABELS[v]:38} {m['reached']:>2}/{m['total']} {m['t_end']:6.1f}s "
              f"{worst:>28} {m['mean_cm']:5.1f} {m['gap_med_cm']:7.1f} {m['gap_max_cm']:7.1f} "
              f"{m['yaw_final_deg']:6.1f}°")

    os.makedirs(RENDERS, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), width_ratios=[1, 1.15])
    _plot_tracks(a1, runs); _plot_gap(a2, runs)
    fig.suptitle("Stage 5b — closed-loop showdown: ESKF vs VIO vs latency (naive vs compensated)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RENDERS, "stage5b_nav.png"), dpi=110)
    print(f"saved {os.path.join(RENDERS, 'stage5b_nav.png')}")

    # the two panels standalone for nav.html: tracks = the interactive chart's no-JS/print
    # fallback, gap = its own static card
    for name, plot, size in (("stage5b_tracks.png", _plot_tracks, (7.4, 6.6)),
                             ("stage5b_gap.png", _plot_gap, (8.6, 5.2))):
        f1, ax = plt.subplots(figsize=size)
        plot(ax, runs); f1.tight_layout()
        f1.savefig(os.path.join(RENDERS, name), dpi=110)
        print(f"saved {os.path.join(RENDERS, name)}")

    _write_tracks(runs)
    with open(os.path.join(DATA, "nav5b_metrics.json"), "w") as f:
        json.dump({v: {k: val for k, val in M[v].items() if k != "approach_cm"} for v in VARIANTS},
                  f, indent=1)
    print(f"saved {os.path.join(DATA, 'nav5b_metrics.json')}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg in VARIANTS:
        run_variant(arg)
    elif arg == "report":
        report()
    elif arg == "all":
        for v in VARIANTS:
            if not os.path.exists(os.path.join(DATA, f"nav5b_{v}.npz")):
                print(f"--- running {v} ---")
                run_variant(v)
        report()
    else:
        raise SystemExit(f"unknown arg {arg!r} — use one of {VARIANTS + ('all', 'report')}")
