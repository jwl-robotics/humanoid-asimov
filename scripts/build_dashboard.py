"""Generate the script-built dashboard pages (stage0 / imu101 / frontend / calib / embedded / nav).

The index + Stage-1/2 pages are hand-authored; these six are script-generated because they embed the
result plots as base64 (self-contained, opens offline). Regenerate the plots first, then rebuild:

    .venv/bin/python scripts/run_walk.py             # data/walk_dataset.npz (embedded needs it)
    .venv/bin/python scripts/run_calib.py            # renders/stage3_calib.png
    .venv/bin/python scripts/run_embedded.py         # renders/stage4_embedded.png
    .venv/bin/python scripts/run_nav.py              # renders/stage5_nav.png
    .venv/bin/python scripts/run_livedrift.py        # renders/livedrift.png
    .venv/bin/python scripts/plot_frontend_result.py # renders/stage2_frontend.png
    .venv/bin/python scripts/render_robot_sensors.py # renders/robot_sensors.png + turntable/
    .venv/bin/python scripts/build_dashboard.py
"""
import base64
import os

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DASH = os.path.join(ROOT, "docs", "dashboard")
RENDERS = os.path.join(ROOT, "renders")
UPDATED = "2026-07-09"

CSS = """
:root{--bg:#0a0e12;--panel:#0e141b;--panel2:#0b1016;--inset:#090d11;--border:#1c2530;--border2:#26313f;
--ink:#d7e2ee;--mut:#8a95a5;--dim:#5c6773;--faint:#3d4652;--cyan:#3fd0c9;--blue:#6ea8fe;--amber:#d9a23f; --coral:#fc4526;
--green:#4ade80;--warn:#e0705f;--mono:ui-monospace,"SF Mono","JetBrains Mono",Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--mut);font-family:var(--mono);font-size:13px;line-height:1.6;padding:0 0 70px;
background-image:radial-gradient(1200px 600px at 70% -10%,rgba(63,208,201,.045),transparent 60%),
linear-gradient(rgba(63,208,201,.02) 1px,transparent 1px),linear-gradient(90deg,rgba(63,208,201,.02) 1px,transparent 1px);
background-size:auto,44px 44px,44px 44px}
.wrap{max-width:1080px;margin:0 auto;padding:0 22px}
a{color:var(--cyan);text-decoration:none}a:hover{text-shadow:0 0 8px rgba(63,208,201,.5)}
header{border-bottom:1px solid var(--border);background:linear-gradient(180deg,rgba(14,20,27,.9),rgba(10,14,18,0))}
.topline{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:10px 0;font-size:10px;
letter-spacing:.16em;color:var(--dim);text-transform:uppercase;border-bottom:1px dashed var(--border)}
.back{font-size:10px;letter-spacing:.14em;text-transform:uppercase;opacity:.8}
h1{margin:24px 0 8px;font-size:clamp(19px,3vw,28px);font-weight:700;color:var(--ink);letter-spacing:.1em;text-transform:uppercase}
h1 .tick{color:var(--cyan)}
.sub{max-width:820px;color:var(--mut);font-size:13px;margin-bottom:18px}
.sub em{color:var(--ink);font-style:normal}
.badges{display:flex;flex-wrap:wrap;gap:7px;margin:14px 0 26px}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;
padding:3px 9px;border:1px solid var(--border2);border-radius:3px;color:var(--mut);background:rgba(255,255,255,.015)}
.chip.ours{color:var(--cyan);border-color:rgba(63,208,201,.45)}.chip.warn{color:var(--warn);border-color:rgba(224,112,95,.5)}
.chip.green{color:var(--green);border-color:rgba(74,222,128,.4)}.chip.amber{color:var(--amber);border-color:rgba(217,162,63,.45)}
.sec{margin-top:32px}
.seclabel{font-size:10px;letter-spacing:.22em;color:var(--dim);text-transform:uppercase;display:flex;align-items:center;gap:10px;margin-bottom:14px}
.seclabel b{color:var(--cyan);font-weight:600}.seclabel .rule{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}
p{margin:11px 0;max-width:860px}p b,li b{color:var(--ink);font-weight:600}
.hi{color:var(--cyan)}.hw{color:var(--warn)}.hg{color:var(--green)}.ha{color:var(--amber)}
ul{margin:10px 0 10px 4px;list-style:none}li{margin:7px 0;padding-left:18px;position:relative;max-width:860px}
li:before{content:"▸";position:absolute;left:0;color:var(--faint)}
.panel{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--border);border-radius:8px;padding:16px 18px;margin:14px 0}
.fig{background:#f7f8fa;border:1px solid var(--border2);border-radius:8px;padding:12px;margin:16px 0}
.fig img{width:100%;display:block;border-radius:3px}
.cap{font-size:11px;color:var(--dim);margin-top:9px;letter-spacing:.02em;text-align:center}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:12px}
th,td{text-align:left;padding:7px 11px;border-bottom:1px solid var(--border)}
th{color:var(--dim);font-size:10px;letter-spacing:.12em;text-transform:uppercase;font-weight:600}
td.n{color:var(--ink);text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:rgba(255,255,255,.014)}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--border);border:1px solid var(--border);border-radius:8px;overflow:hidden;margin:16px 0}
.kv>div{background:var(--panel);padding:13px 15px}
.kv .val{color:var(--cyan);font-size:19px;font-weight:700;font-variant-numeric:tabular-nums}
.kv .val.warn{color:var(--warn)}.kv .val.green{color:var(--green)}
.kv .lab{font-size:10px;color:var(--dim);letter-spacing:.1em;text-transform:uppercase;margin-top:3px}
.kv .src{font-size:10px;color:var(--faint);margin-top:5px}
.take{border-left:2px solid var(--cyan);padding:4px 0 4px 15px;margin:16px 0;color:var(--ink)}
.take.warn{border-color:var(--warn)}.take .t{font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--dim);display:block;margin-bottom:3px}
code{background:var(--inset);border:1px solid var(--border);border-radius:3px;padding:1px 5px;font-size:12px;color:var(--ink)}
footer{margin-top:40px;padding-top:16px;border-top:1px dashed var(--border);font-size:10px;color:var(--faint);letter-spacing:.1em;text-transform:uppercase;display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px}
.flow{display:flex;flex-wrap:wrap;gap:11px;align-items:stretch;margin:16px 0}
.fbox{flex:1;min-width:190px;background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--border2);border-radius:8px;padding:13px 15px}
.fbox .ft{font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--cyan);margin-bottom:5px;font-weight:600}
.fbox .fb{color:var(--mut);font-size:12px;line-height:1.5}
.fbox .eq{font-family:var(--mono);color:var(--ink);font-size:12px;margin-top:8px;background:var(--inset);border:1px solid var(--border);border-radius:4px;padding:5px 9px;display:inline-block}
.fbox.warn{border-color:rgba(224,112,95,.45)}.fbox.warn .ft{color:var(--warn)}
.fbox.good{border-color:rgba(74,222,128,.42)}.fbox.good .ft{color:var(--green)}
.farrow{display:flex;align-items:center;color:var(--faint);font-size:22px;flex:none;padding:0 2px}
.turntable{background:#0a0e12;border:1px solid var(--border2);border-radius:8px;padding:12px 10px 10px;margin:16px 0;text-align:center}
.turntable img{width:min(440px,92%);cursor:grab;border-radius:4px;touch-action:none;user-select:none;-webkit-user-drag:none}
.turntable img:active{cursor:grabbing}
.turntable input{width:min(440px,92%);margin-top:9px;accent-color:var(--cyan);cursor:pointer}
.tthint{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-top:5px}
.ttlegend{display:flex;flex-wrap:wrap;justify-content:center;gap:15px;margin-top:8px;font-size:11px;font-weight:600}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}
@media(max-width:640px){.two{grid-template-columns:1fr}}
.two .col{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--border);border-radius:8px;padding:13px 15px}
.two .col h5{font-size:10px;letter-spacing:.13em;text-transform:uppercase;margin-bottom:8px}
.two .col.a h5{color:var(--cyan)}.two .col.b h5{color:var(--amber)}
.two .col ul{margin:0}.two .col li{font-size:12px}
"""


def img_uri(name):
    with open(os.path.join(RENDERS, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def shell(title, tab, badges, sub, body):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{tab}</title>
<style>{CSS}</style></head><body>
<header><div class="wrap"><div class="topline"><span>Humanoid State Estimation</span>
<a class="back" href="index.html">&larr; back to dashboard</a></div>
<h1><span style="color:var(--coral)">▍</span> {title}</h1><div class="badges">{badges}</div><p class="sub">{sub}</p></div></header>
<div class="wrap">{body}
<footer><span>Personal robotics project — state estimation &amp; navigation</span><span>updated {UPDATED}</span></footer>
</div>
<script>if(window.self!==window.top){{document.querySelectorAll('.back,a[href="index.html"]').forEach(function(e){{e.style.display='none';}});}}</script>
</body></html>"""


def sec(label, body):
    return f'<div class="sec"><div class="seclabel"><b>{label}</b><span class="rule"></span></div>{body}</div>'


def fig(name, cap):
    return f'<div class="fig"><img src="{img_uri(name)}" alt="{cap}"><div class="cap">{cap}</div></div>'


def turntable(n=20):
    uris = []
    for k in range(n):
        with open(os.path.join(RENDERS, "turntable", f"frame_{k:02d}.jpg"), "rb") as f:
            uris.append("data:image/jpeg;base64," + base64.b64encode(f.read()).decode())
    arr = "[" + ",".join('"' + u + '"' for u in uris) + "]"
    return ('<div class="turntable"><img id="ttImg" src="' + uris[0] + '" alt="rotatable robot" draggable="false">'
            '<input type="range" id="ttRange" min="0" max="' + str(n - 1) + '" value="0" aria-label="rotate robot">'
            '<div class="tthint">drag the robot to rotate &#8635; &middot; or use the slider</div>'
            '<div class="ttlegend"><span style="color:#e05050">&#9679; IMU</span>'
            '<span style="color:#3fd0c9">&#9679; head camera</span>'
            '<span style="color:#6ea8fe">&#9679; joint encoders</span>'
            '<span style="color:#d9a23f">&#9679; foot contact</span></div></div>'
            '<script>(function(){var TT=' + arr + ',im=document.getElementById("ttImg"),'
            'rg=document.getElementById("ttRange"),N=' + str(n) + ';'
            'rg.addEventListener("input",function(){im.src=TT[+rg.value];});'
            'var dg=false,sx=0,sv=0;'
            'im.addEventListener("pointerdown",function(e){dg=true;sx=e.clientX;sv=+rg.value;im.setPointerCapture(e.pointerId);});'
            'im.addEventListener("pointermove",function(e){if(!dg)return;var d=Math.round((e.clientX-sx)/11),'
            'v=((sv-d)%N+N)%N;rg.value=v;im.src=TT[v];});'
            'im.addEventListener("pointerup",function(){dg=false;});})();</script>')


# ============================================================ STAGE 3 — calibration
calib = shell(
    'Stage 3 &mdash; <span class="tick">Online Self-Calibration</span>',
    "Stage 3 — Self-Calibration",
    '<span class="chip ours">camera&ndash;IMU extrinsic</span><span class="chip ours">time offset</span>'
    '<span class="chip green">Jacobians FD-verified</span><span class="chip amber">observability-honest</span>',
    "While walking its loop, the robot estimates its <em>own</em> camera-mount rotation online &mdash; no "
    "calibration rig, only the onboard sensors the VIO already uses. It also carries an FD-verified "
    "camera&ndash;IMU time-offset state (<code>td</code>), but this constant-rate loop sits at that state's "
    "observability degeneracy, so <code>td</code> is held <em>frozen</em> here while the mount calibrates "
    "online. Validation: inject a known mount error and recover it.",
    sec("The idea",
        "<p>A real robot's head camera is never mounted exactly where the CAD says, and its frame timestamps are "
        "a few ms out of sync with the IMU clock. The classic fix is a <b>calibration rig</b> (wave a checkerboard "
        "before deploying). <b>In-situ self-calibration</b> instead has the robot recover those errors itself, "
        "while just walking &mdash; robust to lifetime drift, no downtime.</p>"
        "<p>The Stage-2 VIO filter is augmented with a camera-mount rotation extrinsic <code>&delta;&phi;_HC</code> (3) "
        "and a time offset <code>td</code> (1); the error state grows <b>21&nbsp;&rarr;&nbsp;25</b>, with the "
        "calibration states placed <em>before</em> the stochastic clone so cloning stays a clean truncate-then-append. "
        "Both enter the <em>existing</em> relative-rotation vision measurement, so no new sensor is needed.</p>")
    + sec("Recovering an injected 3.1&deg; mount error",
        fig("stage3_calib.png",
            "Inject a 3.1&deg; camera-mount error; the estimator starts from the nominal mount and self-calibrates "
            "online. φ_x (⊥ the turn) drives to ~0 with a tight ±3σ funnel; φ_y/φ_z "
            "(≈ the turn axis) keep wide funnels &mdash; the filter correctly reports what a yaw-loop cannot pin.")
        + '<div class="kv"><div><div class="val green">87%</div><div class="lab">observable axis recovered</div>'
        '<div class="src">φ_x: 26 &rarr; −3.4 mrad</div></div>'
        '<div><div class="val">1.1 mrad</div><div class="lab">φ_x funnel σ (confident)</div><div class="src">tight, consistent</div></div>'
        '<div><div class="val warn">7 mrad</div><div class="lab">φ_y funnel σ (uncertain)</div><div class="src">honestly wide</div></div>'
        '<div><div class="val">~1e-7</div><div class="lab">Jacobian FD error</div><div class="src">all blocks, td=0 &amp; 20ms</div></div></div>')
    + sec("The observability structure (the real lesson)",
        "<p>A <b>pure yaw-loop observes only the mount component perpendicular to the turn axis.</b> The component "
        "<em>along</em> the yaw axis is nearly invisible per frame (only the gait's ripple excites it), and the "
        "time offset is weakly observable too &mdash; the loop sits near the constant-angular-rate degeneracy where "
        "a common time shift cancels. This isn't a bug: the finite-difference check passes to ~1e-7, and the filter's "
        "covariance <em>reports</em> the anisotropy (the φ_y funnel stays ~6× wider than φ_x).</p>"
        '<div class="take"><span class="t">Takeaway</span>A consistent filter recovers what the trajectory makes '
        'observable and honestly flags what it cannot &mdash; a stronger result than a lucky full recovery. Full '
        '3-axis + time-offset calibration needs richer rotational excitation than a single yaw-loop.</div>'))

# ============================================================ STAGE 4 — embedded noise
embedded = shell(
    'Stage 4 &mdash; <span class="tick">The Sim-to-Real Gap</span>',
    "Stage 4 — Embedded Noise",
    '<span class="chip ours">embedded pipeline</span><span class="chip warn">timing &gt; amplitude</span>'
    '<span class="chip green">ablation study</span>',
    "The sim-to-real gap for a low-cost humanoid is dominated by the <em>embedded pipeline</em> &mdash; finite "
    "sensor resolution, sequential CAN reads, transport latency, sample jitter &mdash; not physics fidelity. "
    "This stage injects those effects and stress-tests the estimator to find which ones actually matter.",
    sec("The model",
        "<p><code>apply_embedded_noise</code> corrupts a logged dataset by re-sampling each sensor stream at its "
        "<em>true</em> (perturbed) capture time and quantizing: encoder/IMU <b>quantization</b> (14-bit / 16-bit LSBs), "
        "per-joint CAN <b>stagger</b> (the bus reads joints sequentially), transport <b>latency</b>, and sample-interval "
        "<b>jitter</b>. Physics is untouched &mdash; only the measurement pipeline is made realistic.</p>")
    + sec("The finding: timing dominates, amplitude is negligible",
        fig("stage4_embedded.png",
            "Left: ESKF drift under each effect alone. Amplitude effects (quantization, jitter) are no worse than "
            "clean; timing effects (inter-joint stagger, latency) blow drift up several-fold. Right: drift climbs "
            "steeply with latency.")
        + "<table><tr><th>effect</th><th>drift</th><th>verdict</th></tr>"
        "<tr><td>clean (Gaussian IMU noise only)</td><td class='n'>0.32%</td><td>baseline</td></tr>"
        "<tr><td>+ all embedded effects together</td><td class='n'>0.92%</td><td>realistic full stack &mdash; jitter dithers the coherent latency skew</td></tr>"
        "<tr><td>+ encoder/IMU quantization</td><td class='n'>0.32%</td><td class='hg'>negligible</td></tr>"
        "<tr><td>+ sample jitter (0.8 ms)</td><td class='n'>0.07%</td><td class='hg'>negligible</td></tr>"
        "<tr><td>+ inter-joint stagger (2 ms)</td><td class='n'>1.51%</td><td class='hw'>hurts</td></tr>"
        "<tr><td>+ transport latency (5 ms)</td><td class='n'>2.04%</td><td class='hw'>hurts most</td></tr>"
        "<tr><td>+ latency (5 ms, uniform — contact delayed too)</td><td class='n'>1.74%</td><td class='hw'>still ~5&times; clean</td></tr>"
        "<tr><td>+ latency (15 ms)</td><td class='n'>6.5%</td><td class='hw'>catastrophic</td></tr></table>")
    + sec("Why it ties back to Stage 3",
        "<p>The dominant killer is <b>timing</b> &mdash; latency and the relative skew between sensor streams &mdash; "
        "while sensor <em>precision</em> barely registers &mdash; to close the sim-to-real gap you model the "
        "pipeline's timing structure, not its amplitude.</p>"
        '<div class="take"><span class="t">Takeaway</span>Latency is the dominant drift driver &mdash; and Stage 3&rsquo;s '
        'camera&ndash;IMU time-offset state (<code>td</code>) is the template for the defense: model per-stream '
        'time offsets as states and calibrate them online &mdash; given a trajectory that excites them &mdash; instead '
        'of trusting timestamps. Stage 4 says why; Stage 3 shows how (and where the excitation has to come from).</div>'))

# ============================================================ STAGE 5 — navigation
nav = shell(
    'Stage 5 &mdash; <span class="tick">Closed-Loop Navigation</span>',
    "Stage 5 — Navigation",
    '<span class="chip ours">estimator &rarr; planner &rarr; command</span>'
    '<span class="chip green">4/4 waypoints</span><span class="chip amber">on the estimate</span>',
    "The capstone slice: the robot walks to a sequence of waypoints using <em>only its own onboard state "
    "estimate</em>, closing the loop sensors &rarr; estimator &rarr; waypoint planner &rarr; "
    "<code>VelocityCommand</code> &rarr; gait steering &mdash; the same command interface the real control stack accepts.",
    sec("The loop",
        "<p>Every control step the ESKF ingests the onboard sensors and reports a pose; a <code>WaypointNavigator</code> "
        "turns that pose into a body-frame <code>VelocityCommand(vx, vy, vyaw)</code> toward the active waypoint, which "
        "steers the walking gait. Because the planner sees only the <em>estimate</em>, estimation error surfaces "
        "directly as navigation error &mdash; which is exactly why an accurate estimator matters for navigation.</p>")
    + sec("Navigating on the onboard estimate",
        fig("stage5_nav.png",
            "The robot reaches all four waypoints from its own ESKF estimate (cyan). The true path (orange) bulges "
            "outward: that gap is the ESKF's unobservable yaw drift accumulating over the turns.")
        + '<div class="kv"><div><div class="val green">4/4</div><div class="lab">waypoints reached</div>'
        '<div class="src">14.6 s, 12 m path</div></div>'
        '<div><div class="val">7.1 cm</div><div class="lab">est-vs-truth at finish</div><div class="src">0.59% of path</div></div>'
        '<div><div class="val warn">yaw drift</div><div class="lab">the error source</div><div class="src">ESKF cannot observe it</div></div></div>')
    + sec("The through-line",
        "<p>The estimated and true paths diverge because heading is unobservable from IMU + contact alone &mdash; the "
        "same limitation Stage 1 diagnosed and Stage 2's VIO fixes. Navigation makes the consequence concrete and "
        "visible: <b>a few degrees of yaw drift becomes tens of centimeters of position error at the goal.</b></p>"
        + fig("livedrift.png",
              "Continuous 3-lap drift with a realistic uncalibrated IMU bias: leg-odometry and the ESKF accumulate "
              "unbounded yaw drift (17&deg; / 7&deg;), while VIO stays bounded at ~1&deg; across all three laps "
              "as vision keeps pinning the yaw-axis bias. Position error (right) oscillates over the closed loop.")
        + "<p>And the VIO-over-ESKF win is <b>robust, not a lucky seed</b>: across 5 seeds VIO beats the ESKF on "
        "drift and gyro-z bias on every one (median 0.15% vs 0.49%; b<sub>g,z</sub> 0.19 vs 1.66 mrad/s).</p>"
        '<div class="take"><span class="t">Takeaway</span>The whole project reads as one arc &mdash; leg-odometry and '
        'IMU drift in yaw, VIO anchors it, self-calibration keeps the sensors honest, the embedded-noise study says '
        'timing is what breaks on hardware, and navigation closes the loop and shows why every earlier stage mattered.</div>'))

# ============================================================ STAGE 2 — front-end sharpening
frontend = shell(
    'Stage 2 &mdash; <span class="tick">Front-End Sharpening</span>',
    "Stage 2 — Front-End Sharpening",
    '<span class="chip ours">essential-matrix refit</span><span class="chip ours">IRLS on all inliers</span>'
    '<span class="chip green">honest 2 mrad</span><span class="chip amber">gyro-gated, IMU-independent</span>',
    "The VIO's vision measurement is a camera relative-rotation. It sat at ~8 mrad (7.69 median) &mdash; the recurring limiter "
    "of the whole project (the fusion only won when it over-trusted vision). A diagnosis found the <em>estimator</em>, "
    "not the feature tracks, was the bottleneck; a refit sharpened it to ~2 mrad and made the VIO win <em>honest</em>.",
    sec("The diagnosis &mdash; the tracks were fine, the solver wasn't",
        "<p><code>findEssentialMat + recoverPose</code> returns the <b>5-point minimal model</b>: RANSAC picks the "
        "best 5-point sample, and the many inliers <em>vote</em> for it but <em>never enter the fit</em>. So the "
        "rotation error is set by 5 noisy pixels no matter how many features agree. Proof: on clean geometry with "
        "0.4 px synthetic noise the shipped pipeline still produced <b>6.0 mrad</b> &mdash; ~27&times; the &radic;N "
        "correspondence floor of 0.22 mrad; an IRLS refit on the same points gave <b>1.3 mrad</b>.</p>"
        "<ul><li>The long tail was <b>R&ndash;t coupling on the coplanar floor</b> (83% of inliers) &mdash; "
        "twisted-pair branch flips even with perfect correspondences.</li>"
        "<li>The ~2 mrad bias was <b>correlated KLT chain drift</b> (template inertia against the loop's rotational "
        "flow, invisible to the forward&ndash;backward check).</li>"
        "<li>Exonerated: the conventions (oracle error 0.00) and rotation-only estimators (Kabsch: <b>278 mrad</b> "
        "&mdash; at 0.4&ndash;0.9 m walking baselines translation must be modeled as a nuisance, not ignored).</li></ul>")
    + sec("The fix &mdash; refit on all inliers, gyro-gated",
        fig("stage2_frontend.png",
            "Left: the minimal-model error (7.69 mrad median, 4.03 bias) vs the IRLS refit (2.06, 0.36) &mdash; the "
            "refit recovers the &radic;N averaging. Right: at the honest &sigma;_vis = 2 mrad, VIO beats the ESKF on "
            "every seed.")
        + "<ul><li><b>Huber-IRLS Gauss&ndash;Newton refit</b> of (R, t) on <em>all</em> inliers, minimizing the "
        "Sampson epipolar error on the SO(3)&times;S&sup2; manifold (translation a modeled nuisance, then discarded).</li>"
        "<li><b>Gyro + neck-FK prior as an accept/reject-only gate</b> (loose 30 mrad): it kills the coplanar "
        "branch-flips but <em>never enters the cost</em> &mdash; the accepted value is a stationary point of the "
        "purely-visual objective, so the measurement stays independent of the IMU the filter already integrates.</li>"
        "<li><b>Anchor age capped at 2 s</b>: a measured negative result &mdash; 0.5 s spans pass the money gate but "
        "<em>lose</em> in fusion, because below ~1 s the gyro out-informs vision.</li>"
        "<li><b>Textured far walls</b>: the long-anchor workhorse (slow-moving, low-warp, real parallax at 2&ndash;3 m "
        "baselines). A close-structure variant measurably backfired &mdash; occlusion + feature-budget theft.</li></ul>")
    + sec("The result &mdash; honest, and robust",
        '<div class="kv"><div><div class="val green">2.06 mrad</div><div class="lab">money-gate median</div>'
        '<div class="src">was 7.69 &middot; bias 4.03 &rarr; 0.36</div></div>'
        '<div><div class="val">0.15%</div><div class="lab">VIO drift (5-seed median)</div><div class="src">vs 0.49% ESKF</div></div>'
        '<div><div class="val green">every seed</div><div class="lab">VIO beats ESKF</div><div class="src">drift + b_g,z</div></div>'
        '<div><div class="val">1.4e-10</div><div class="lab">fusion FD-check</div><div class="src">untouched, re-verified</div></div></div>'
        '<div class="take"><span class="t">Takeaway</span>The recurring limiter is gone. Because the front-end is '
        'now genuinely ~2 mrad, the fusion&rsquo;s &sigma;_vis = 2 mrad is <b>honest</b> &mdash; it bounds the real '
        'measurement rather than under-stating it &mdash; so VIO wins outright with a calibrated covariance. The fusion '
        'filter and its finite-difference-verified Jacobians were never touched: a front-end-only change feeding the '
        'same measurement interface.</div>'))

# ============================================================ STAGE 0 — the harness (robot + sensors)
stage0 = shell(
    'Stage 0 &mdash; <span class="tick">The Harness</span>',
    "Stage 0 — The Harness",
    '<span class="chip">real Asimov model</span><span class="chip ours">head camera added</span>'
    '<span class="chip amber">walking = black box</span><span class="chip green">ground truth logged</span>',
    "Before any estimation, Stage 0 stands up the foundation: load the real robot, expose exactly the sensors it "
    "carries onboard, let it walk, and log the ground truth to grade every later stage against. Everything "
    "downstream only ever sees the onboard sensors &mdash; never the ground truth.",
    sec("The robot, and where its sensors live",
        fig("robot_sensors.png", "The actual Asimov mesh. Red/blue = original sensors, cyan = what we added, "
            "amber = derived. Drag the robot below to look around it.")
        + turntable()
        + '<div class="two"><div class="col a"><h5>original &mdash; ships with the model</h5><ul>'
        '<li><b>IMU</b> on the pelvis: a <b>gyroscope</b> (angular velocity) + an <b>accelerometer</b> (specific force)</li>'
        '<li><b>Joint encoders</b>: an angle at every joint (legs, waist, neck)</li></ul></div>'
        '<div class="col b"><h5>added / derived by us</h5><ul>'
        '<li><b>Head camera</b> (cyan): the robot&rsquo;s eye &mdash; absent upstream; we bolted it on for the VIO</li>'
        '<li><b>Foot contact</b> (amber): <em>not a sensor</em> &mdash; there is no foot-force sensor on this model, so '
        'we derive &ldquo;is this foot planted?&rdquo; from the physics engine&rsquo;s collision list</li></ul></div></div>')
    + sec("Onboard vs. eval-only &mdash; the honest split",
        "<p>The estimator is only allowed the <b>estimator-visible</b> stream. The <b>ground truth</b> is logged "
        "purely to score the estimate afterward and is <em>never fed in</em> &mdash; that is what keeps the "
        "evaluation honest.</p>"
        '<div class="two"><div class="col a"><h5>estimator sees</h5><ul><li>raw gyro + accelerometer</li>'
        '<li>15 joint-encoder angles</li><li>foot-contact flags</li><li>head-camera frames</li></ul></div>'
        '<div class="col b"><h5>eval-only (ground truth)</h5><ul><li>true orientation</li>'
        '<li>true base position + velocity</li><li>true world foot positions</li></ul></div></div>')
    + sec("Walking is a black box &mdash; and we say so",
        "<p>The model has <b>zero actuators</b> &mdash; on real hardware a learned policy supplies the joint torques. "
        "To produce motion for estimator development, we PD-track an open reference gait plus a torso &ldquo;virtual "
        "support&rdquo; that offloads ~44% of body weight and keeps it upright. <b>This is a development stand-in, not "
        "true dynamic walking</b> &mdash; but that is fine: estimation only needs the onboard sensor stream to be "
        "self-consistent with the motion, which it is.</p>"
        '<div class="take"><span class="t">Takeaway</span>Stage 0 delivers a clean, honest dataset: the exact onboard '
        'sensors a real Asimov carries, plus ground truth to grade against. What those raw signals actually <em>are</em>, '
        'and how they become a pose, is the <a href="imu101.html">IMU-101 primer</a>.</div>'))

# ============================================================ IMU 101 — the primer
imu101 = shell(
    'IMU 101 &mdash; <span class="tick">From Raw Signals to a Pose</span>',
    "IMU 101 — Primer",
    '<span class="chip">inertial navigation</span><span class="chip warn">why yaw drifts</span>'
    '<span class="chip ours">the fusion idea</span>',
    "The conceptual bedrock of the whole project: what an IMU actually measures, how you turn it into orientation "
    "and position, and why <em>heading</em> is the one thing it can never pin down on its own. Read this first and "
    "stages 0&ndash;5 all click into place.",
    sec("What the IMU actually measures &mdash; not orientation, not position",
        '<div class="flow"><div class="fbox"><div class="ft">Gyroscope</div><div class="fb">how fast you are '
        '<b>rotating</b>, about each body axis. A rate, not an angle.</div><div class="eq">&omega;&nbsp;&nbsp;(rad/s)</div></div>'
        '<div class="fbox"><div class="ft">Accelerometer</div><div class="fb">your <b>proper acceleration</b> &mdash; your '
        'motion <em>minus gravity</em>, in the body frame. At rest it reads the 9.81 push of gravity, pointing '
        '&ldquo;up&rdquo;.</div><div class="eq">f = a &minus; g&nbsp;&nbsp;(m/s²)</div></div></div>'
        "<p>Neither is orientation or position. You get those only by <b>integrating</b> &mdash; and integration is "
        "where the trouble starts.</p>")
    + sec("Turning it into a pose &mdash; and where it drifts",
        '<div class="flow"><div class="fbox"><div class="ft">gyro &rarr; orientation</div><div class="fb">integrate the '
        'angular rate over time.</div><div class="eq">R &larr; R &middot; Exp(&omega;&middot;dt)</div></div>'
        '<div class="farrow">&rarr;</div><div class="fbox warn"><div class="ft">but it DRIFTS</div><div class="fb">bias '
        '+ noise accumulate every step; orientation slowly wanders with nothing to check it.</div></div></div>'
        '<div class="flow"><div class="fbox good"><div class="ft">accel &rarr; gravity &rarr; tilt anchor</div><div class="fb">'
        'when you are not accelerating hard, the accelerometer points along gravity &mdash; an <b>absolute '
        '&ldquo;down&rdquo;</b> that snaps <b>roll &amp; pitch</b> back. &check;</div></div><div class="farrow">&rarr;</div>'
        '<div class="fbox warn"><div class="ft">&hellip;but nothing for yaw</div><div class="fb">gravity is vertical, so '
        'it says <em>nothing</em> about rotation <em>about</em> vertical.</div></div></div>'
        '<div class="flow"><div class="fbox"><div class="ft">accel &rarr; position</div><div class="fb">rotate to world, '
        'subtract gravity, double-integrate.</div><div class="eq">p &larr; &int;&int; (R&middot;f + g) dt²</div></div>'
        '<div class="farrow">&rarr;</div><div class="fbox warn"><div class="ft">drifts FAST</div><div class="fb">'
        'double-integrating noise grows the error like t^1.5&ndash;t² &mdash; useless within seconds without help.</div></div></div>')
    + sec("Why yaw is the hard one",
        "<p><b>It is the gravity reference.</b> Tilt the robot and gravity shows up in the &ldquo;wrong&rdquo; "
        "accelerometer axes, so the filter can always correct <b>roll and pitch</b> &mdash; they are <em>observable</em>, "
        "they don&rsquo;t drift. But spin about the <em>vertical</em> axis (<b>yaw / heading</b>) and gravity doesn&rsquo;t "
        "move at all &mdash; the accelerometer gives <b>zero</b> information about it. Yaw can only come from the drifting "
        "gyro. No gravity-equivalent heading reference exists <em>in the IMU</em> &mdash; until a camera watches the world turn.</p>"
        '<div class="kv"><div><div class="val green">roll / pitch</div><div class="lab">observable</div>'
        '<div class="src">gravity anchors them</div></div><div><div class="val warn">yaw</div>'
        '<div class="lab">unobservable from IMU</div><div class="src">no reference &rarr; drifts</div></div>'
        '<div><div class="val">position</div><div class="lab">drifts fastest</div><div class="src">needs legs + vision</div></div></div>')
    + sec("Where legs and the camera come in &mdash; the whole project in one arrow",
        '<div class="flow"><div class="fbox"><div class="ft">Legs (Stage 1)</div><div class="fb">a planted foot is a '
        'fixed anchor &rarr; gives <b>velocity</b>. But leg motion is in the <em>body</em> frame; placing it in the world '
        'needs your heading. Legs <b>depend on</b> yaw &mdash; they can&rsquo;t fix it.</div></div><div class="farrow">&rarr;</div>'
        '<div class="fbox good"><div class="ft">Camera (Stage 2)</div><div class="fb">watching fixed features rotate '
        'gives an <b>independent rotation</b> the IMU can&rsquo;t &mdash; it finally pins the <b>yaw drift + the gyro-z '
        'bias</b>.</div></div></div>'
        '<div class="take"><span class="t">The one idea</span>Every stage is the same move: <b>integration drifts, so '
        'add a reference that bounds it</b> &mdash; gravity for tilt, a planted foot for velocity, the camera for heading, '
        'and (Stage 3) the camera calibrating itself. That is the entire arc.</div>'))

os.makedirs(DASH, exist_ok=True)
for name, html in (("stage0.html", stage0), ("imu101.html", imu101), ("frontend.html", frontend),
                   ("calib.html", calib), ("embedded.html", embedded), ("nav.html", nav)):
    with open(os.path.join(DASH, name), "w") as f:
        f.write(html)
    print(f"wrote docs/dashboard/{name}  ({len(html)//1024} KB)")
