"""Build the Asimov MuJoCo model with an added first-person head camera.

The upstream model (``external/asimov-1/sim-model/xmls/asimov.xml``) ships only external
"studio" tracking cameras — there is **no onboard camera**. Monocular VIO needs an egocentric
camera on the head (``neck_pitch_link``). We add it programmatically with ``MjSpec`` so the
vendored model file stays untouched and our modification lives in version-controlled code.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

_HERE = os.path.dirname(__file__)
MODEL_XML = os.path.abspath(os.path.join(
    _HERE, "..", "..", "external", "asimov-1", "sim-model", "xmls", "asimov.xml"))

HEAD_BODY = "neck_pitch_link"
HEAD_CAMERA = "head_cam"

# First-person camera pose, expressed in the head (neck_pitch) frame.
# Standing, the head frame aligns with the world: +x = forward, +z = up.


def _head_cam_quat(pitch_down_deg: float):
    """Quaternion (w, x, y, z) for a camera looking forward (+x) with +z up, pitched
    ``pitch_down_deg`` below horizontal (so it favours the terrain/ground ahead).

    The base orientation aims the camera's -z (view axis) along +x and its +y (up) along
    +z; we then rotate about the camera's local x (right) axis to pitch it downward.
    """
    base = np.array([0.5, 0.5, -0.5, -0.5])               # forward (+x), level, +z up
    h = np.radians(pitch_down_deg) / 2.0
    pitch = np.array([np.cos(h), -np.sin(h), 0.0, 0.0])   # rotate about camera local x, downward
    w0, x0, y0, z0 = base
    wp, xp, yp, zp = pitch
    return (
        w0 * wp - x0 * xp - y0 * yp - z0 * zp,
        w0 * xp + x0 * wp + y0 * zp - z0 * yp,
        w0 * yp - x0 * zp + y0 * wp + z0 * xp,
        w0 * zp + x0 * yp - y0 * xp + z0 * wp,
    )


HEAD_CAM_PITCH_DOWN_DEG = 15.0          # tilt below horizontal: close floor gives parallax + the mix of
#                                         near-floor and far-wall/pillar features keeps the essential
#                                         matrix well-conditioned (levelling to 5° hurt — low parallax).
HEAD_CAM_POS = (0.10, 0.0, 0.13)        # forehead height, just forward of the face (clears the chest)
HEAD_CAM_QUAT = _head_cam_quat(HEAD_CAM_PITCH_DOWN_DEG)
HEAD_CAM_FOVY = 70.0                    # vertical field of view (deg), approximate monocular lens


def build_spec(add_head_camera: bool = True, add_environment: bool = False, env_seed: int = 0,
               head_cam_quat=None, head_cam_pos=None) -> mujoco.MjSpec:
    """Load the Asimov model as an editable MjSpec (head camera + optional feature room). head_cam_quat/pos
    override the nominal mount — used by Stage-3 calibration to render a *perturbed* mount as ground truth
    while the estimator keeps believing the nominal."""
    spec = mujoco.MjSpec.from_file(MODEL_XML)
    if add_head_camera:
        head = spec.body(HEAD_BODY)
        if head is None:
            raise RuntimeError(f"body {HEAD_BODY!r} not found in {MODEL_XML}")
        cam = head.add_camera()
        cam.name = HEAD_CAMERA
        cam.pos = HEAD_CAM_POS if head_cam_pos is None else head_cam_pos
        cam.quat = HEAD_CAM_QUAT if head_cam_quat is None else head_cam_quat
        cam.fovy = HEAD_CAM_FOVY
    if add_environment:
        add_feature_room(spec, seed=env_seed)
    return spec


# Bodies whose collision geoms are "the foot".
FOOT_CONTACT_BODIES = ("left_ankle_roll_link", "left_toe_link",
                       "right_ankle_roll_link", "right_toe_link")


def _fix_foot_contacts(model):
    """Give the feet torsional friction + firmer contact so the stance foot stays planted.

    MuJoCo's default contacts (condim=3) spin freely, and the soft default solref lets the foot
    capsules sink ~12 mm under strike loads — surfacing as ~11 deg/stance yaw twist (a large fake
    gyro-z + a ~50 deg/10 s veer) and corrupted anchor geometry. condim=4 + torsional friction +
    a firmer solref keeps the planted foot planted, translationally and rotationally.
    """
    foot_bids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in FOOT_CONTACT_BODIES}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if (model.geom_bodyid[g] in foot_bids and model.geom_contype[g] != 0) or name == "floor":
            model.geom_condim[g] = 4                       # enable torsional friction
            model.geom_friction[g] = (1.0, 0.1, 0.01)      # [tangential, torsional, rolling]
            model.geom_solref[g] = (0.005, 1.0)            # firmer contact (less penetration)


def build_model(add_head_camera: bool = True, fix_foot_contacts: bool = True,
                add_environment: bool = False, env_seed: int = 0,
                head_cam_quat=None, head_cam_pos=None) -> mujoco.MjModel:
    """Compile the model (head camera + firmer foot contacts + optional feature room)."""
    model = build_spec(add_head_camera, add_environment, env_seed, head_cam_quat, head_cam_pos).compile()
    if fix_foot_contacts:
        _fix_foot_contacts(model)
    return model


# ---------------------------------------------------------------- VIO feature room
# A visual-only environment so the monocular head camera has non-repeating, 3-D structure to track.
# Everything is collision-free (contype=conaffinity=0): the robot dynamics — and the Stage-1 dataset —
# are unchanged; only the camera view differs. The vendored floor gives repetitive/coplanar features
# (VIO-hostile); this adds textured walls + scattered landmarks the robot walks through and turns against.
ENV_ROOM = 8.0          # room half-extent (m): perimeter walls at ±8
ENV_WALL_H = 3.0        # wall height (m)

_PALETTE = np.array([
    [0.86, 0.32, 0.29], [0.29, 0.55, 0.85], [0.95, 0.76, 0.24], [0.34, 0.74, 0.46],
    [0.70, 0.40, 0.80], [0.92, 0.56, 0.24], [0.24, 0.70, 0.72], [0.82, 0.44, 0.56],
    [0.55, 0.62, 0.30], [0.46, 0.50, 0.86], [0.86, 0.62, 0.72], [0.40, 0.76, 0.66],
])


def _vgeom(wb, gtype, size, pos, rgba, name=None):
    """Add a collision-free (visual-only) geom to the worldbody."""
    g = wb.add_geom()
    g.type = gtype
    g.size = list(size)
    g.pos = list(pos)
    g.rgba = list(rgba)
    g.contype = 0
    g.conaffinity = 0
    if name:
        g.name = name
    return g


def _concrete_texture(size: int = 1024, seed: int = 7) -> np.ndarray:
    """Procedural concrete: medium grey with dense high-contrast grain + aggregate specks, so the
    down-tilted camera tracks unique floor corners everywhere (unlike the repetitive checker)."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 0.34, np.float64)
    for scale in (16, 32, 64, 128):                                # gentle large-scale mottle only
        coarse = (rng.random((scale, scale)) * 255).astype(np.uint8)
        up = np.asarray(Image.fromarray(coarse).resize((size, size), Image.BILINEAR), np.float64) / 255.0
        img += (up - 0.5) * 0.09
    img += (rng.random((size, size)) - 0.5) * 0.15                 # dense fine grain → corners
    n = size * size // 90                                          # aggregate specks (sharp)
    ys, xs = rng.integers(0, size, n), rng.integers(0, size, n)
    img[ys, xs] = rng.choice([0.13, 0.52], n)
    img = np.clip(img, 0.10, 0.50)                                 # medium grey, never blows to white
    rgb = np.stack([img, img, img * 0.99], -1)                     # neutral-warm grey
    return (rgb * 255).astype(np.uint8)


def _apply_concrete_floor(spec: mujoco.MjSpec, seed: int = 7, texrepeat: float = 4.0):
    """Repoint the vendored 'floor' geom to a procedural concrete material (visual only)."""
    img = _concrete_texture(seed=seed)
    tex = spec.add_texture()
    tex.name = "concrete_tex"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.height, tex.width, tex.nchannel = img.shape[0], img.shape[1], 3
    tex.data = img.tobytes()
    mat = spec.add_material()
    mat.name = "concrete_mat"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "concrete_tex"
    mat.texrepeat = [texrepeat, texrepeat]
    mat.texuniform = True
    mat.reflectance = 0.0
    mat.specular = 0.0                                 # matte — kill specular blowout
    mat.shininess = 0.0
    for g in spec.geoms:
        if g.name == "floor":
            g.material = "concrete_mat"


def _wall_texture(size: int = 256, seed: int = 3) -> np.ndarray:
    """Procedural wall pattern: random colored rectangles at many scales + fine grain + specks —
    dense, high-contrast, non-repeating corners across the whole wall face (think posters/brick/
    signage on an industrial wall, not a bare painted one)."""
    rng = np.random.default_rng(seed)
    pal = _PALETTE
    img = np.zeros((size, size, 3))
    img[:] = pal[rng.integers(len(pal))]
    for _ in range(140):                               # rectangles at many scales → multi-scale corners
        w, h = rng.integers(8, 64, 2)
        x, y = rng.integers(0, size - w), rng.integers(0, size - h)
        img[y:y + h, x:x + w] = pal[rng.integers(len(pal))] * (0.55 + 0.6 * rng.random())
    img += (rng.random((size, size, 1)) - 0.5) * 0.24  # fine luminance grain
    n = size * size // 60
    ys, xs = rng.integers(0, size, n), rng.integers(0, size, n)
    img[ys, xs] = rng.choice([0.08, 0.95], (n, 1))     # sharp dark/bright specks
    return (np.clip(img, 0.03, 0.97) * 255).astype(np.uint8)


def _apply_wall_texture(spec: mujoco.MjSpec, seed: int = 3):
    """Texture the perimeter walls (visual only). Far features are the front-end's long-anchor
    workhorse: they move slowly and barely warp under the walk (KLT holds them with ~no drift,
    unlike the foreshortening floor), and at the 2-3 m baselines a multi-second anchor span builds,
    even 5-10 m walls carry real parallax — so they both stabilize the correspondences and condition
    the essential-matrix refit that the coplanar floor alone leaves degenerate. Bare walls starved
    the tracker of exactly these features (measured: mid-span rotation error ~2.7x worse without)."""
    img = _wall_texture(seed=seed)
    tex = spec.add_texture()
    tex.name = "wall_tex"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.height, tex.width, tex.nchannel = img.shape[0], img.shape[1], 3
    tex.data = img.tobytes()
    mat = spec.add_material()
    mat.name = "wall_mat"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "wall_tex"
    mat.texrepeat = [1.5, 1.5]
    mat.texuniform = True
    mat.reflectance = 0.0
    mat.specular = 0.0
    mat.shininess = 0.0
    for g in spec.geoms:
        if g.name and g.name.startswith("wall_"):
            g.material = "wall_mat"


def add_feature_room(spec: mujoco.MjSpec, seed: int = 0, room: float = ENV_ROOM,
                     wall_h: float = ENV_WALL_H):
    """Add a visual-only 'feature room': perimeter walls + wall panels + free-standing landmarks +
    floor patches, all at distinct positions/colors (non-repeating, unlike the checker floor)."""
    rng = np.random.default_rng(seed)
    wb = spec.worldbody
    pal = _PALETTE
    _apply_concrete_floor(spec, seed=seed + 100)      # concrete floor (non-repeating, replaces checker)

    def col(i=None):
        return list(pal[rng.integers(len(pal)) if i is None else i % len(pal)]) + [1.0]

    t = 0.05
    # (center, halfsize, runs_along_y) for the 4 perimeter walls
    walls = [((room, 0, wall_h / 2), (t, room, wall_h / 2), True),
             ((-room, 0, wall_h / 2), (t, room, wall_h / 2), True),
             ((0, room, wall_h / 2), (room, t, wall_h / 2), False),
             ((0, -room, wall_h / 2), (room, t, wall_h / 2), False)]
    for i, (pos, size, along_y) in enumerate(walls):
        _vgeom(wb, mujoco.mjtGeom.mjGEOM_BOX, size, pos,
               [0.26 + 0.03 * i, 0.30, 0.38 - 0.03 * i, 1.0], f"wall_{i}")
    _apply_wall_texture(spec, seed=seed + 21)         # dense far corners (long-anchor stability)
    for i, (pos, size, along_y) in enumerate(walls):
        # colored panels on the wall face → corner features across the wall
        for _ in range(9):
            pw, ph = 0.4 + 0.5 * rng.random(), 0.4 + 0.6 * rng.random()
            z = 0.4 + rng.random() * (wall_h - 1.0)
            if along_y:
                off = -0.06 if pos[0] > 0 else 0.06
                _vgeom(wb, mujoco.mjtGeom.mjGEOM_BOX, [0.02, pw, ph],
                       [pos[0] + off, (rng.random() * 2 - 1) * (room - 1), z], col())
            else:
                off = -0.06 if pos[1] > 0 else 0.06
                _vgeom(wb, mujoco.mjtGeom.mjGEOM_BOX, [pw, 0.02, ph],
                       [(rng.random() * 2 - 1) * (room - 1), pos[1] + off, z], col())

    # free-standing landmarks (pillars + boxes) at varied depth → parallax + a yaw reference
    placed = 0
    while placed < 18:
        ang, r = rng.random() * 2 * np.pi, 2.5 + rng.random() * (room - 3.5)
        x, y = r * np.cos(ang), r * np.sin(ang)
        if -1.5 < x < 7.5 and abs(y) < 1.1:              # keep the forward walking lane (+x) clear
            continue
        if rng.random() < 0.5:
            h, rad = 0.5 + rng.random() * 1.7, 0.08 + rng.random() * 0.18
            _vgeom(wb, mujoco.mjtGeom.mjGEOM_CYLINDER, [rad, h, 0], [x, y, h], col())
        else:
            s = 0.15 + rng.random(3) * 0.4
            s[2] = 0.2 + rng.random() * 0.7
            _vgeom(wb, mujoco.mjtGeom.mjGEOM_BOX, s, [x, y, s[2]], col())
        placed += 1

    # floor patches → features in the lower (downward-tilted) part of the frame
    for _ in range(14):
        s = [0.2 + rng.random() * 0.5, 0.2 + rng.random() * 0.5, 0.004]
        _vgeom(wb, mujoco.mjtGeom.mjGEOM_BOX, s,
               [(rng.random() * 2 - 1) * (room - 1), (rng.random() * 2 - 1) * (room - 1), 0.006], col())
    return spec
