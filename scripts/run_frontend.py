"""Stage 2.1 — run the VIO front-end over the loop + verify it against ground truth.

Tracks features (KLT) along the loop walk, forms keyframes, and estimates the camera rotation from each
anchor keyframe to the current one (RANSAC init + IRLS refit on all inliers, translation a discarded
nuisance — see docs/STAGE2_FRONTEND_DESIGN.md). Anchors persist up to 2 s (long spans carry the yaw-drift
information — a 0.5 s hop is one the gyro already predicts better than vision); a gyro+neck-FK relative
rotation gates gross failures (loose 30 mrad, accept/reject only — the measurement value stays
vision-only). The **money gate**: compare each estimated R̃_{C_A→C_j} to the GT camera rotation (from
gt_quat + FK extrinsics — GT scores, never feeds). Median error ≤ ~2 mrad and |bias| ≤ ~0.5 mrad ⇒ the
measurements are trustworthy for fusion. Saves the keyframe/anchor/rotation schedule to
data/loop_tracks.npz.

    .venv/bin/python scripts/run_frontend.py
"""
import os

import cv2
import numpy as np

from humanoid_asimov.estimator import exp_so3, log_so3, quat2mat
from humanoid_asimov.vision import (AnchorRotationEstimator, CameraModel, FeatureTracker, HeadCamExtrinsics)
from humanoid_asimov.walk import loop_heading, simulate

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "loop_tracks.npz"))
PERIOD, KF_EVERY, ANCHOR_MAX_AGE, ANCHOR_MIN_SHARED = 24.0, 15, 2.0, 30


def main(seed=0):
    _, _, log, frames = simulate(seconds=PERIOD, heading=loop_heading(PERIOD), environment=True,
                                 render_camera="head_cam", fps=30, imu_noise=None, seed=seed)
    frames = np.array(frames)
    ft_t, t_log = log["frame_t"], log["t"]
    gt_quat, enc_q, jn = log["gt_quat"], log["enc_q"], list(log["joint_names"])
    fidx = np.array([int(np.argmin(np.abs(t_log - tt))) for tt in ft_t])   # frame → log index
    print(f"frames {len(frames)} @ {frames.shape[2]}x{frames.shape[1]}")

    cam, ext = CameraModel(640, 480, 70.0), HeadCamExtrinsics()
    tracker, anchor = FeatureTracker(640, 480, fb_thresh=1.5), AnchorRotationEstimator(cam)

    # gyro-integrated body rotation per log step — the coarse prior for the front-end's outlier gate.
    # Onboard-only (gyro + encoder FK); over a 0.5 s span its error is ≪ the 30 mrad gate.
    dtl = float(t_log[1] - t_log[0])
    R_cum = np.empty((len(t_log), 3, 3))
    R_cum[0] = np.eye(3)
    for i in range(1, len(t_log)):
        R_cum[i] = R_cum[i - 1] @ exp_so3(log["gyro"][i] * dtl)

    def R_WC_gt(i):
        R_BC, _ = ext.compute(enc_q[i], jn)
        return quat2mat(gt_quat[i]) @ R_BC

    meas, track_len, anchor_idx, n_gated = [], {}, None, 0
    for k in range(len(frames)):
        kf = (k % KF_EVERY == 0)
        tracks = tracker.process(cv2.cvtColor(frames[k], cv2.COLOR_RGB2GRAY), keyframe=kf)
        for i in tracks:
            track_len[i] = track_len.get(i, 0) + 1
        if not kf:
            continue
        i, t = int(fidx[k]), float(ft_t[k])
        m = None
        if anchor.anchor_t is not None:
            R_BC_A, R_BC_j = ext.compute(enc_q[anchor_idx], jn)[0], ext.compute(enc_q[i], jn)[0]
            R_prior = R_BC_A.T @ (R_cum[anchor_idx].T @ R_cum[i]) @ R_BC_j   # gyro+FK R_{C_A C_j}
            m = anchor.estimate(t, tracks, R_prior=R_prior)
            if m is not None:
                R_gt = R_WC_gt(anchor_idx).T @ R_WC_gt(i)             # GT R_{C_A C_j}
                e = log_so3(m["R"].T @ R_gt)                          # residual axis-angle (rad)
                m.update(err_mrad=float(np.linalg.norm(e)) * 1e3, err_vec=e * 1e3,
                         anchor_idx=anchor_idx, cur_idx=i)
                meas.append(m)
            else:
                n_gated += 1
        n_shared = sum(1 for x in tracks if x in anchor.anchor_uv) if anchor.anchor_t is not None else 0
        if anchor.anchor_t is None or m is None or (t - anchor.anchor_t) > ANCHOR_MAX_AGE \
                or n_shared < ANCHOR_MIN_SHARED:
            anchor.set_anchor(t, tracks); anchor_idx = i               # re-anchor

    errs = np.array([m["err_mrad"] for m in meas])
    bvec = np.mean([m["err_vec"] for m in meas], axis=0)
    bias = np.linalg.norm(bvec)
    lengths = np.array(list(track_len.values()))
    inl = np.array([m["n_inliers"] for m in meas])
    print(f"\n=== front-end over the loop (seed {seed}) ===")
    print(f"keyframes with a rotation measurement: {len(meas)}  (+{n_gated} gated/degenerate → no update)")
    print(f"MONEY GATE  rotation-vs-GT (mrad): median {np.median(errs):.2f}  mean {errs.mean():.2f}"
          f"  max {errs.max():.2f}  | bias {bias:.2f}   [pass: median≲2, bias≲0.5]")
    print(f"  bias vec cam-frame ({bvec[0]:+.2f}, {bvec[1]:+.2f}, {bvec[2]:+.2f}) mrad (x~pitch, y~yaw, z~roll)")
    ok = errs < 15.0                                              # outliers the fusion χ² gate would reject
    gbias = np.linalg.norm(np.mean([m["err_vec"] for m, g in zip(meas, ok) if g], axis=0))
    print(f"  gated (err<15, {ok.sum()}/{len(errs)}): median {np.median(errs[ok]):.2f}  bias {gbias:.2f}")
    print(f"track length: median {np.median(lengths):.0f} f ({np.median(lengths)/30:.2f} s)"
          f" | ≥45 f: {(lengths >= 45).mean()*100:.0f}%  | live/kf≈{np.median([m['n_shared'] for m in meas]):.0f}")
    print(f"inliers: median {np.median(inl):.0f}  ratio median {np.median([m['ratio'] for m in meas]):.2f}"
          f" | parallax median {np.median([m['parallax'] for m in meas]):.1f} px")

    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, R_meas=np.array([m["R"] for m in meas]),
                        anchor_idx=np.array([m["anchor_idx"] for m in meas]),
                        cur_idx=np.array([m["cur_idx"] for m in meas]),
                        n_inliers=inl, parallax=np.array([m["parallax"] for m in meas]),
                        err_mrad=errs, period=PERIOD, seed=seed)
    print(f"saved {DATA}")


if __name__ == "__main__":
    main()
