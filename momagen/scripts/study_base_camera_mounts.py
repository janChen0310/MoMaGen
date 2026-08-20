"""TidyBot third-person camera MOUNT STUDY (headless renderer).

WHY: pi0.5 requires an exterior/third-person view alongside the wrist view (openpi's
image contract is base_0_rgb + left/right_wrist_0_rgb, and PI's own mobile manipulators
fed the wrist cameras plus a FORWARD-facing camera mounted between the arms to the
low-level policy). TidyBot's stock `base_camera_link` sits at world z~0.246 m pitched
45 deg DOWN (USD: base->base_camera_link fixed joint, localPos0=(0.2525,0,0.315),
localRot0=45deg about +Y), so it stares at the floor just ahead of the wheels and never
sees a 0.888 m counter top. This script mounts candidate masts on the base, renders the
robot WITH the candidate positions marked, and renders the view FROM each candidate, so
a mount can be chosen from pictures rather than from arithmetic.

Frames: the `base` link is x-forward, y-left, z-up. A mount is an Xform parented to the
base link, oriented R_z(yaw)*R_y(pitch) -- rotation about +Y pitches the optical axis
DOWN -- with a USD Camera inside carrying the standard (0.5,0.5,-0.5,-0.5) camera-
convention quat, exactly mirroring how the stock base camera is built.

Run (on the box):  CAM_OUT=/tmp/jc_cammounts python momagen/scripts/study_base_camera_mounts.py
"""
import math, os
os.environ["OMNIGIBSON_HEADLESS"] = "1"; os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
import numpy as np
import torch as th
import omnigibson as og
from omnigibson.macros import gm
from momagen.scripts.collect_tidybot_source_demo import load_tidybot_env_config

REPO = os.environ.get("MC_REPO", "/home/ubuntu/DATA4/backup_root_home/yhu/MoMaGen")
TEMPLATE = REPO + "/momagen/datasets/source_og/tidybot_picking_up_trash.hdf5"
OUT = os.environ.get("CAM_OUT", "/tmp/jc_cammounts")
SCENE_INSTANCE = os.environ.get("CAM_SCENE", "")
ACTIVITY = os.environ.get("CAM_ACTIVITY", "")
RES = int(os.environ.get("CAM_RES", "512"))

def log(m): print(m, flush=True)
def arr(x): return np.array(x.cpu() if hasattr(x, "cpu") else x, float)

os.makedirs(OUT, exist_ok=True)

# ---------------- env ----------------
cfg = load_tidybot_env_config(TEMPLATE)
cfg["scene"]["scene_model"] = "house_single_floor"
if SCENE_INSTANCE:
    cfg["scene"]["scene_instance"] = SCENE_INSTANCE
    cfg["scene"]["scene_file"] = None
if ACTIVITY:
    cfg["task"]["activity_name"] = ACTIVITY
cfg["robots"][0]["self_collisions"] = False
gm.ENABLE_TRANSITION_RULES = False
log("scene cfg: %s" % {k: cfg["scene"].get(k) for k in ("scene_model", "scene_instance")})
log("task: %s" % cfg.get("task", {}).get("activity_name"))
log("building env...")
env = og.Environment(configs=cfg)
robot = env.robots[0]
env.reset()
for _ in range(30):
    og.sim.step()

# ---------------- measure the base ----------------
base_link = robot.links["base"]
bp, bq = (arr(v) for v in robot.get_position_orientation())
blo, bhi = (arr(v) for v in base_link.aabb)
log("ROBOT pos=%s quat=%s" % (np.round(bp, 3).tolist(), np.round(bq, 4).tolist()))
log("BASE aabb lo=%s hi=%s  footprint=%s mm" % (
    np.round(blo, 3).tolist(), np.round(bhi, 3).tolist(),
    np.round((bhi - blo) * 1000, 1).tolist()))
# stock camera, for reference in the report
stock = robot.links.get("base_camera_link")
if stock is not None:
    sp = arr(stock.get_position_orientation()[0])
    log("STOCK base_camera_link world=%s (z=%.3f)" % (np.round(sp, 3).tolist(), sp[2]))

yaw = math.atan2(2 * (bq[3] * bq[2] + bq[0] * bq[1]), 1 - 2 * (bq[1] ** 2 + bq[2] ** 2))
half_x = float((bhi[0] - blo[0]) / 2.0)
half_y = float((bhi[1] - blo[1]) / 2.0)
# half extents are world-axis-aligned; for a yawed base take the smaller as the true half-width
half = min(half_x, half_y)
log("base half-extent used for corner mounts: %.3f m (aabb halves %.3f/%.3f)" % (half, half_x, half_y))

# ---------------- candidate mounts (base-link local frame) ----------------
# world_z = local_z - 0.0688 (the `base` Xform sits 68.8 mm below the robot root)
Z_OFF = 0.0688
def L(world_z): return world_z + Z_OFF

CANDIDATES = [
    # name                     local (x, y, z)                 pitch_dn  yaw   note
    ("A_rear_mid_h130",  (-half + 0.02, 0.00, L(1.30)), 30.0,  0.0, "rear edge midpoint, 1.30 m mast"),
    ("B_rear_mid_h110",  (-half + 0.02, 0.00, L(1.10)), 25.0,  0.0, "rear edge midpoint, 1.10 m mast"),
    ("C_rear_left_h120", (-half + 0.04,  half - 0.04, L(1.20)), 30.0, -14.0, "rear-LEFT corner, 1.20 m"),
    ("D_rear_right_h120",(-half + 0.04, -half + 0.04, L(1.20)), 30.0,  14.0, "rear-RIGHT corner, 1.20 m"),
    ("E_rear_mid_h150",  (-half + 0.02, 0.00, L(1.50)), 38.0,  0.0, "rear edge midpoint, 1.50 m mast"),
    ("F_side_left_h120", ( 0.00,  half - 0.04, L(1.20)), 32.0, -10.0, "mid-LEFT edge, 1.20 m"),
]

def quat_mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw)

from pxr import UsdGeom, Gf, Vt
stage = og.sim.stage
base_path = base_link.prim_path
log("base prim path: %s" % base_path)

cam_paths = {}
for name, loc, pitch, yaw_deg, note in CANDIDATES:
    p = math.radians(pitch); y = math.radians(yaw_deg)
    q_pitch = (math.cos(p / 2), 0.0, math.sin(p / 2), 0.0)   # +Y  -> pitch DOWN
    q_yaw = (math.cos(y / 2), 0.0, 0.0, math.sin(y / 2))     # +Z  -> yaw LEFT
    q = quat_mul(q_yaw, q_pitch)
    mount = UsdGeom.Xform.Define(stage, f"{base_path}/jc_mount_{name}")
    mount.AddTranslateOp().Set(Gf.Vec3d(float(loc[0]), float(loc[1]), float(loc[2])))
    mount.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
    cpath = f"{base_path}/jc_mount_{name}/Camera"
    cam = UsdGeom.Camera.Define(stage, cpath)
    cam.GetFocalLengthAttr().Set(14.0)          # ~ the stock base cam (13) / arm cam (11)
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetVerticalApertureAttr().Set(15.2908)
    cam.GetClippingRangeAttr().Set((0.01, 1000000))
    UsdGeom.Xformable(cam.GetPrim()).AddOrientOp().Set(
        Gf.Quatf(0.5, Gf.Vec3f(0.5, -0.5, -0.5)))   # USD camera convention -> looks along +X
    cam_paths[name] = cpath
    log("MOUNT %-20s local=%s pitch=%.0f yaw=%.0f  (%s)" % (
        name, np.round(np.array(loc), 3).tolist(), pitch, yaw_deg, note))

og.sim.step()
og.sim.render()

import omni.replicator.core as rep
import imageio

def render_batch(paths, res_map, tag):
    log("[%s] attaching %d cameras..." % (tag, len(paths)))
    anns = {}
    for name, path in paths.items():
        rp = rep.create.render_product(path, res_map(name))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        anns[name] = ann
    for _ in range(30):
        og.sim.render()
    for name, ann in anns.items():
        try:
            fr = np.array(ann.get_data())
            if fr.ndim == 3 and fr.shape[-1] >= 3:
                img = fr[:, :, :3].astype(np.uint8)
                imageio.imwrite(os.path.join(OUT, f"{name}.png"), img)
                log("  wrote %-22s %s  mean=%.1f std=%.1f" % (
                    name + ".png", img.shape, float(img.mean()), float(img.std())))
            else:
                log("  FAILED %s (got %s)" % (name, getattr(fr, "shape", None)))
        except Exception as e:
            log("  ERROR %s: %s" % (name, e))

# PASS 1 -- what each candidate SEES (no markers in the scene yet)
cam_paths["STOCK_base_camera"] = f"{robot.links['base_camera_link'].prim_path}/Camera"
render_batch(cam_paths, lambda n: (RES, RES), "views")

# ---------------- markers at each candidate (world space, visual only) ----------------
COLORS = {
    "A_rear_mid_h130":  (1.0, 0.15, 0.15),
    "B_rear_mid_h110":  (1.0, 0.65, 0.0),
    "C_rear_left_h120": (0.15, 0.8, 0.15),
    "D_rear_right_h120":(0.15, 0.45, 1.0),
    "E_rear_mid_h150":  (0.8, 0.2, 1.0),
    "F_side_left_h120": (0.1, 0.9, 0.9),
}
world_pos = {}
for name, loc, pitch, yaw_deg, note in CANDIDATES:
    # base-local -> world (base link pose)
    lp, lq = (arr(v) for v in base_link.get_position_orientation())
    lw, lx, ly, lz = lq[3], lq[0], lq[1], lq[2]
    R = np.array([
        [1 - 2*(ly*ly + lz*lz), 2*(lx*ly - lz*lw),     2*(lx*lz + ly*lw)],
        [2*(lx*ly + lz*lw),     1 - 2*(lx*lx + lz*lz), 2*(ly*lz - lx*lw)],
        [2*(lx*lz - ly*lw),     2*(ly*lz + lx*lw),     1 - 2*(lx*lx + ly*ly)],
    ])
    wp = lp + R @ np.array(loc, float)
    world_pos[name] = wp
    sph = UsdGeom.Sphere.Define(stage, f"/World/jc_marker_{name}")
    sph.GetRadiusAttr().Set(0.045)
    UsdGeom.Xformable(sph.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in wp]))
    sph.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*COLORS[name])]))
    log("MARKER %-20s world=%s (height %.3f m)" % (name, np.round(wp, 3).tolist(), wp[2]))

og.sim.step(); og.sim.render()

# ---------------- external cameras that SHOW the robot + markers ----------------
def lookat_quat(eye, target):
    f = np.array(target, float) - np.array(eye, float)
    f /= (np.linalg.norm(f) + 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up); r /= (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)
    M = np.stack([r, u, -f], axis=1)   # camera looks down -Z
    t = M[0, 0] + M[1, 1] + M[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s; x = (M[2, 1] - M[1, 2]) / s; y = (M[0, 2] - M[2, 0]) / s; z = (M[1, 0] - M[0, 1]) / s
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = math.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        w = (M[2, 1] - M[1, 2]) / s; x = 0.25 * s; y = (M[0, 1] + M[1, 0]) / s; z = (M[0, 2] + M[2, 0]) / s
    elif M[1, 1] > M[2, 2]:
        s = math.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        w = (M[0, 2] - M[2, 0]) / s; x = (M[0, 1] + M[1, 0]) / s; y = 0.25 * s; z = (M[1, 2] + M[2, 1]) / s
    else:
        s = math.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        w = (M[1, 0] - M[0, 1]) / s; x = (M[0, 2] + M[2, 0]) / s; y = (M[1, 2] + M[2, 1]) / s; z = 0.25 * s
    return (w, x, y, z)

ext_specs = []
for i, (az_off, dist, el, tag) in enumerate([
        (2.2, 2.0, 0.62, "sideL"), (-2.2, 2.0, 0.62, "sideR"),
        (math.pi, 1.8, 0.75, "rear"), (0.6, 2.2, 0.55, "frontL"),
        (-0.6, 2.2, 0.55, "frontR"), (math.pi, 2.6, 1.05, "rear_high")]):
    az = yaw + az_off
    eye = bp + np.array([dist * math.cos(az) * math.cos(el),
                         dist * math.sin(az) * math.cos(el),
                         0.75 + dist * math.sin(el)])
    ext_specs.append((f"ext{i}_{tag}", eye))

for name, eye in ext_specs:
    q = lookat_quat(eye, bp + np.array([0.0, 0.0, 0.80]))
    cam = UsdGeom.Camera.Define(stage, f"/World/jc_extcam_{name}")
    cam.GetFocalLengthAttr().Set(16.0)
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetVerticalApertureAttr().Set(15.2908)
    cam.GetClippingRangeAttr().Set((0.01, 1000000))
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in eye]))
    xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
    cam_paths[name] = f"/World/jc_extcam_{name}"
    log("EXTCAM %-16s eye=%s" % (name, np.round(eye, 2).tolist()))

# the stock low base camera, for the before/after comparison
cam_paths["STOCK_base_camera"] = f"{robot.links['base_camera_link'].prim_path}/Camera"

og.sim.step(); og.sim.render()

# ---------------- render every camera ----------------

# PASS 2 -- where each candidate IS (markers now exist)
ext_paths = {n: p for n, p in cam_paths.items() if n.startswith("ext")}
render_batch(ext_paths, lambda n: (720, 720), "mounts")

log("CAMSTUDY_DONE out=%s" % OUT)
og.shutdown()
os._exit(0)
