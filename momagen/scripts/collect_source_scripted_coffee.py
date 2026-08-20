# NOTE: TidyBot make-coffee scripted source demo (see docs/tutorials/tidybot-task-pipelines.md).
# Paths default to the old-box workspace; override with MC_REPO / MC_VIDEO env vars.
"""SCRIPTED TidyBot make-coffee source demo (headless), for MoMaGen datagen.

Task (BDDL datagen_make_coffee in house_single_floor): TRANSFER the white sugar cube
("milk") from the open teacup and the brown die ("coffee") from the open toy box into
the wide coffee cup on the kelker kitchen counter -- PICK-AND-DROP mechanics (grasp
the token itself, carry, release above the cup), replacing the tilt-pour design: the
MoMaGen replay executor reproduces pick/carry/drop with high fidelity (validated at
83% on the trash task) but cannot reproduce in-hand pour rotations (~120 failed
generation trials across 5 fixed root causes).

4 phases: [1] grasp sugar cube (from inside the 1.6x teacup -- scaled so the Hand-E
finger envelope fits the bore) [2] drop into cup [3] grasp die (from the open box)
[4] drop into cup. Vessels never move. Base drives between per-phase standoffs in
free-space segments only (generation mirrors this via the JC_REACH_MAX_DIST nav gate).

Records via DataCollectionWrapper (+ mask/use), logs 4-phase boundaries, prints BDDL
success, writes an mp4. In-cup checks + EARLY_ABORT make it retry-harness friendly.
"""
import math, os
os.environ["OMNIGIBSON_HEADLESS"] = "1"; os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
# NB: do NOT set OMNIGIBSON_GPU_ID here — on multi-GPU boxes the worker GPU is picked
# with CUDA_VISIBLE_DEVICES alone; combining the two breaks Vulkan device enumeration
# (Isaac Kit segfaults at boot in the XR viewport extension).
import h5py
import numpy as np
import torch as th
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataCollectionWrapper
from omnigibson.macros import gm
from momagen.scripts.collect_tidybot_source_demo import load_tidybot_env_config

REPO = os.environ.get("MC_REPO", "/home/ubuntu/DATA4/backup_root_home/yhu/MoMaGen")
TEMPLATE = REPO + "/momagen/datasets/source_og/tidybot_picking_up_trash.hdf5"  # robot kwargs template
OUTPUT = REPO + "/momagen/datasets/source_og/tidybot_make_coffee.hdf5"
SCENE_INSTANCE = "house_single_floor_task_datagen_make_coffee_0_0_template"
VIDEO = os.environ.get("MC_VIDEO", REPO + "/jc_script_coffee_demo.mp4")

# ---- object registry names (from the sampled instance's inst_to_name) ----
CUP = os.environ.get("MC_CUP", "coffee_cup_599")
TEACUP = os.environ.get("MC_TEACUP", "teacup_598")
BOX = os.environ.get("MC_BOX", "toy_box_597")
SUGAR = os.environ.get("MC_SUGAR", "sugar_cube_596")
DIE = os.environ.get("MC_DIE", "dice_595")

# ---- grasp/pour tunables ----
TILT_DEG = 12.0          # near-top-down rim approach (proven recipe family from the trash task)
PAD_OFF = 0.05           # approach-direction backoff of the grasp target
TEA_RIM_R = 0.0345       # teacup rim radius (bb 73mm/2, minus wall)
TEA_TOP = 0.052          # teacup height
BOX_HALF_Y = 0.060       # toy box half-extent along pour axis (bb 126mm/2, minus wall)
BOX_TOP = 0.063          # toy box height
GRIP_BELOW_TOP = 0.0     # target the very top edge (achieved fingers land ~+35mm above command)
# Measured DLS-IK saturation residuals (v4 run, per vessel, world frame at this base
# pose): fingers plateau short of the commanded grip. Pre-bias the target by the
# NEGATED residual (trash-task SAT_BIAS recipe) so the fingers land ON the rim/wall.
SAT_BIAS = {
    # teacup: v12-measured residual, negated in xy -- validated by the v13 catch.
    # toy_box: xy reset to 0 (the v12 measurement came from the north-wall grip; it
    # does not transfer to the east grip and overshot v13 by ~10 cm).
    "teacup":  np.array([+0.021, +0.002, -0.030]),
    "toy_box": np.array([+0.017, 0.0, -0.030]),   # v14/v15 east-grip residual, negated
    # token grasps (v27b measurement): the descent to the sugar cube sagged [-11,-62]mm
    # and landed the fingers ON the teacup's south rim -- pre-compensate xy so the
    # descent enters the bore; z keeps the standard -0.030 free-space plateau bias
    # v27c: true free-space descent sag is only ~7mm (the v27b "62mm" was rim contact)
    # fingertip depth: full-depth descents wedge the hand body under the teacup rim
    # xy: feedforward the consistent +16..+23mm east landing residual (v59-63).
    # z: aim at the tall token's shaft; the counter-edge fulcrum caps the fingers at
    # ~0.937-0.947, which is the token's upper half by design.
    "sugar":   np.array([-0.017, 0.0, +0.015]),
    "die":     np.array([-0.017, 0.0, +0.015]),   # the sugar's exact proven biases
}
# pour-time xy sag while the wrist rotates (v13: sugar landed [-76,-74]mm from cup):
# bias the pour station up-range; half-measured, since the closer workspace also cuts sag
# drop-station sag: the hover (z~1.07-1.12, 0.36m reach) sits in the arm's saturated
# region; release lands [-119,+70]mm from command, DETERMINISTICALLY (v27e, twice
# identical). Negated-bias so the release point centers on the cup.
# MEASURED (probe, base parked at the drop station, cmd_z=1.10): the achieved eef
# undershoots the commanded x by 0.085-0.11 (~0.09, flat across reach 0.25-0.41) and
# lands 0.02 south. Command cup + this and the hand arrives OVER the cup centre.
# x: LIVE-CONFIRMED (v38: cmd 4.475 -> achieved 4.381 vs cup 4.380). y: the probe's
# -0.027 sag did NOT reproduce in the live drop (achieved y == cmd y to 1mm), so the
# comp is zero -- carrying it just released the cube 27mm north of centre.
DROP_BIAS = np.array([+0.095, 0.0])
POUR_BIAS = {
    "teacup":  np.array([+0.04, 0.0, 0.0]),
    "toy_box": np.array([+0.01, -0.05, 0.0]),   # center the observed NW landing mode
}
def token_in_cup(token_name):
    tp = arr(objs[token_name].get_position_orientation()[0])
    cp = arr(objs[CUP].get_position_orientation()[0])
    rim = float(arr(objs[CUP].aabb[1])[2])
    # 2.2x cup interior: rim radius ~6.8-7.2cm tapering inward; max in-cup rest
    # dxy ~0.05-0.06. dxy 0.07-0.14 at z~0.92-0.95 = lodged on the HANDLE (v30
    # pattern B) -- keep the gate strict. Floor bound derived from the live cup
    # aabb bottom (robust to settle/displacement): counter rest 0.895-0.904 stays
    # excluded (cup base+wall adds ~0.03).
    cup_bot = float(arr(objs[CUP].aabb[0])[2])
    dxy = float(np.linalg.norm(tp[:2] - cp[:2]))
    ok = dxy < 0.065 and (cup_bot + 0.02) < tp[2] < rim - 0.05
    log("CHECK %s in cup: %s (dxy=%.3f z=%.3f rim=%.3f)" % (
        token_name, ok, float(np.linalg.norm(tp[:2] - cp[:2])), tp[2], rim))
    return ok

def early_abort(reason):
    log("EARLY_ABORT %s" % reason)
    try:
        _v = os.environ.get("MC_VIDEO")
        if _v and frames:
            import imageio
            _av = _v.replace(".mp4", "_abort.mp4")
            imageio.mimsave(_av, frames[::2], fps=30)
            log("ABORT_VIDEO %s (%d frames)" % (_av, len(frames)))
    except Exception as _e:
        log("ABORT_VIDEO failed: %s" % _e)
    og.shutdown()
    os._exit(3)
# place-time sag: descending WITH a held vessel undershoots ~12 cm toward the base
# (east); west errors land safely ON the counter, east ones fall off the edge.
# The box place is accurate unbiased (v14+v15) -- teacup only.
PLACE_BIAS = {
    "teacup":  np.array([-0.13, 0.0, 0.0]),
    "toy_box": np.array([0.0, 0.0, 0.0]),
}
# CENTER-PIVOT pours (v25): the eef orbits so the VESSEL CENTER stays fixed over the
# basin while rotating -- grip-pivot pours swung the mouth through a big arc, making
# the exit point scatter run-to-run and the swing volume collide with the cup.
# POUR_CLEAR = vessel-center height above the cup rim.
POUR_CLEAR = {"teacup": 0.09, "toy_box": 0.11}
# commanded tilt overshoots the achieved tilt (orientation tracking sags with the
# held-object lever arm): teacup 130 deg validated (v14 sugar landed IN the cup);
# the 13cm box only achieved ~90 deg at 130 commanded -> command 150 + harder shake
# teacup at 130 commanded is at the token's friction margin (v14 in-cup, v15 flung
# during untilt -- chaotic): 145 forces a reliable early exit during the tilt ramp
# achieved tilt (telemetry): teacup 140@155cmd, box 117@160cmd. The token must exit
# EARLY over the bottom lip (achieved ~105-115 deg); at >130 achieved it pools at the
# grip-side lip and STICKS TO THE FINGER PADS (sticky mode) -- v14's success exited
# during the ramp. Command for early exit, not maximum tilt.
# TWO-STAGE pour: stage-1 = the proven moderate tilt that ejects cleanly at source
# time (deeper pools the token at the grip lip / sticks it to the pads -- 8/8 deep-tilt
# source attempts failed); after the token is VERIFIED in the cup, continue to the
# DEEP tilt with the vessel empty -- pure recording margin so the generation replay
# (which under-achieves recorded tilt by ~20-40 deg) still crosses the ejection
# threshold (JCMC: die never ejected in 40+ replayed trials of a moderate recording).
POUR_TILT = {"teacup": math.radians(120.0), "toy_box": math.radians(140.0)}
POUR_TILT_DEEP = {"teacup": math.radians(160.0), "toy_box": math.radians(175.0)}
POUR_SHAKE = {"teacup": math.radians(15.0), "toy_box": math.radians(12.0)}
POUR_HOLD = {"teacup": 100, "toy_box": 140}
POUR_SHAKE_CYCLES = {"teacup": 4, "toy_box": 3}
N_REACH = 160; N_APPROACH = 170; N_GRASP = 60; N_LIFT = 170
N_OPEN = 45; N_RETREAT = 70
S_TEA = [4.80, -0.97]     # v61's exact station: its landing attractor [4.508,-1.048,0.896] is measured IN THIS GEOMETRY
S_BOX = [4.79, -1.42]     # reach 0.35 to the die -- matches the proven sugar-station geometry
N_TRANSPORT = 170; N_TILT = 80; N_HOLD = 80; N_UNTILT = 110   # fast tilt: momentum forces the far-lip (early) exit mode
N_RETURN = 150; N_LOWER = 110; N_OPEN = 45; N_RETREAT = 70
LIFT_DZ = 0.12          # low lift: 0.22 extended the arm enough that xy tracking sagged ~15 cm
SAFE_Z = 1.10          # MEASURED (probe): cmd 1.10 -> achieved 1.145-1.19 at EVERY
                       # reach 0.25-0.41, i.e. fingertips 1.11-1.15, clearing the
                       # 1.081 rim. The commanded z is NOT the achieved z at this
                       # station and the map is non-monotonic: cmd 1.24 -> achieved
                       # 0.98-1.15 (fingertips BELOW the rim -> the arm plows the cup,
                       # which is the whole v29-v34 failure), cmd 1.16 -> 1.10-1.22
                       # (erratic). Do not "add margin" by raising this.
STANDOFF_DIST = 0.43     # base SPAWN distance from cup (eef untuck hover must clear the cup rim)

def log(m): print(m, flush=True)
log("SCRIPT_VERSION v86")
def arr(x): return np.array(x.cpu() if hasattr(x, "cpu") else x, float)
def yaw_of(q):
    x, y, z, w = q; return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def build_R(a1, b1, a2, b2):
    def n(v): return v / (np.linalg.norm(v) + 1e-9)
    a1 = n(a1); a2 = n(a2 - (a2 @ a1) * a1); a3 = np.cross(a1, a2)
    b1 = n(b1); b2 = n(b2 - (b2 @ b1) * b1); b3 = np.cross(b1, b2)
    A = np.stack([a1, a2, a3], axis=1); B = np.stack([b1, b2, b3], axis=1)
    return B @ A.T

def axis_angle_quat(axis, ang):
    axis = np.asarray(axis, float); axis = axis / (np.linalg.norm(axis) + 1e-9)
    s = math.sin(ang / 2.0)
    return th.tensor([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(ang / 2.0)], dtype=th.float32)

# ---------------- env ----------------
cfg = load_tidybot_env_config(TEMPLATE)
cfg["scene"]["scene_model"] = "house_single_floor"
cfg["scene"]["scene_instance"] = SCENE_INSTANCE
cfg["scene"]["scene_file"] = None
# ASSISTED grasping. sticky attaches whatever a finger FIRST touches -- and when the
# target is a token sitting INSIDE a vessel, that is the vessel: video of the p2 carry
# (v46 abort frames) shows the arm lifting the whole TEACUP with the sugar cube riding
# inside it, then the cube tumbling out mid-drive. Every "sugar in cup" success to date
# was the cube falling out while the arm happened to pass over the cup, which is what
# made the landings bimodal from an identical command.
# AG casts a ray between the finger pads, so it can only grab what is actually pinched
# -- the cube, not the vessel it sits in. The sticky comment this replaces dates from
# the POUR design, where grasping the vessel WAS the goal. Generation already runs
# assisted (env_args patch), so this also makes source and replay consistent.
cfg["robots"][0]["grasping_mode"] = "sticky"
cfg["robots"][0]["self_collisions"] = False
cfg["task"]["activity_name"] = "datagen_make_coffee"
# LOAD-TIME base spawn at the standoff east of the cup, facing it (west). The exact cup
# xy is only known post-load, so spawn from the template-designed layout constants:
CUP_XY_DESIGN = [float(v) for v in os.environ.get("MC_CUP_XY", "4.40,-1.25").split(",")]
# SPAWN AT THE BOX STATION, not at the cup standoff. env.reset() untucks the arm
# through the corridor dead ahead of the spawn; aimed at the cup that corridor runs
# straight through the 1.082 rim and lays the cup on its side before step 1 (v36).
# The box corridor only holds the 0.951-tall toy box, which the untuck clears -- and
# it is the proven phase-3 station, so the arm is known to work from here.
_sx, _sy = 4.85, -1.42
_syaw = math.atan2(-1.46 - _sy, 4.44 - _sx)
cfg["robots"][0]["position"] = [_sx, _sy, 0.0]
cfg["robots"][0]["orientation"] = [0.0, 0.0, math.sin(_syaw / 2), math.cos(_syaw / 2)]
gm.ENABLE_TRANSITION_RULES = False
log("building env...")
env = og.Environment(configs=cfg)
env = DataCollectionWrapper(env=env, output_path=OUTPUT, only_successes=False)
robot = env.robots[0]; arm = robot.default_arm
env.reset()
for _ in range(40): og.sim.step()

objs = {}
for nm in (CUP, TEACUP, BOX, SUGAR, DIE):
    o = env.scene.object_registry("name", nm)
    assert o is not None, "object not found: %s" % nm
    objs[nm] = o
    lo, hi = (arr(a) for a in o.aabb)
    log("OBJ %-18s pos=%s aabb_mm=%s" % (nm, np.round(arr(o.get_position_orientation()[0]), 3).tolist(),
                                         np.round((hi - lo) * 1000, 1).tolist()))
# LAYOUT ASSERTIONS: the whole v29-v34 saga was a cup that spawned 3cm inside the
# counter and overlapping the teacup, so PhysX ejected it somewhere new every run.
# Fail loudly at load rather than 34 versions later.
COUNTER_TOP = 0.888
_ASSERT_LAYOUT = not os.environ.get("MC_PROBE")   # probes exist to DIAGNOSE a failed layout
for _a, _b in ((CUP, TEACUP), (CUP, BOX), (TEACUP, BOX)) if _ASSERT_LAYOUT else ():
    _la, _ha = (arr(x) for x in objs[_a].aabb)
    _lb, _hb = (arr(x) for x in objs[_b].aabb)
    _ov = [min(_ha[i], _hb[i]) - max(_la[i], _lb[i]) for i in range(3)]
    if all(o > 0 for o in _ov):
        log("LAYOUT_FAIL %s/%s interpenetrate by %s mm" % (_a, _b, np.round(np.array(_ov) * 1000, 1).tolist()))
        og.shutdown(); os._exit(4)
for _nm in (CUP, TEACUP, BOX) if _ASSERT_LAYOUT else ():
    _lo = float(arr(objs[_nm].aabb[0])[2])
    if _lo < COUNTER_TOP - 0.003:
        log("LAYOUT_FAIL %s is %.1f mm below the counter top" % (_nm, (COUNTER_TOP - _lo) * 1000))
        og.shutdown(); os._exit(4)
# UPRIGHT = the cup's quat is pure yaw (local +z still world-up). Render-verified:
# upright is 152.3mm tall / 194mm wide (a wide bowl); a TIPPED cup stands its width up
# and reads ~194-230mm in z, which is what the old ejected pose was doing.
_cq = arr(objs[CUP].get_position_orientation()[1])
if _ASSERT_LAYOUT and (abs(float(_cq[0])) > 0.05 or abs(float(_cq[1])) > 0.05):
    log("LAYOUT_FAIL %s not upright: quat=%s (upright => qx,qy ~ 0)" % (
        CUP, np.round(_cq, 4).tolist()))
    og.shutdown(); os._exit(4)
log("LAYOUT_OK all vessels clear and resting on the counter")

cup_p = arr(objs[CUP].get_position_orientation()[0])
cup_hi = arr(objs[CUP].aabb[1])
CUP_RIM_Z = float(cup_hi[2])
bp0 = arr(robot.get_position_orientation()[0])
_ep0 = arr(robot.get_eef_pose(arm)[0])
log("spawn=%s untuck_eef=%s cup=%s rim_z=%.3f dist=%.2f" % (
    np.round(bp0, 2).tolist(), np.round(_ep0, 3).tolist(), np.round(cup_p, 3).tolist(),
    CUP_RIM_Z, float(np.linalg.norm(bp0[:2] - cup_p[:2]))))

# ---------------- camera + recording (chase + close-up focus) ----------------
import omni.replicator.core as rep
from pxr import UsdGeom, Gf
stg = og.sim.stage
cam = UsdGeom.Camera.Define(stg, "/World/jc_cam"); cam.GetFocalLengthAttr().Set(16.0)
cam.GetHorizontalApertureAttr().Set(24.0); cam.GetClippingRangeAttr().Set((0.02, 200.0))
camxf = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
cam_focus = [None]
def set_cam():
    p, q = robot.get_position_orientation()
    bp = arr(p); yw = yaw_of(arr(q))
    if cam_focus[0] is not None:
        Tc, dist = cam_focus[0]
        Tc = np.array(Tc, float)
        az = yw + math.pi - math.radians(50.0); el = math.radians(33.0)
    else:
        Tc = np.array([bp[0], bp[1], 0.5]); dist = 2.6
        az = yw + math.pi + math.radians(35.0); el = math.radians(52.0)
    eye = Tc + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    up = np.array([0, 0, 1.]); f = Tc - eye; f /= np.linalg.norm(f)
    rr = np.cross(f, up); rr /= np.linalg.norm(rr); uu = np.cross(rr, f)
    M = np.eye(4); M[:3, 0] = rr; M[:3, 1] = uu; M[:3, 2] = -f; M[:3, 3] = eye
    camxf.Set(Gf.Matrix4d(*M.T.flatten().tolist()))
set_cam()
rp = rep.create.render_product("/World/jc_cam", (640, 384))
ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
frames = []
def grab():
    d = ann.get_data()
    if d is None: return
    fr = np.array(d)
    if fr.ndim == 3 and fr.shape[-1] >= 3: frames.append(fr[:, :, :3].astype(np.uint8))

# ---------------- control primitives (proven DLS-IK from the trash script) ----------------
step_count = [0]
def compute_action(tp, tq, gripper_closed, base_cmd):
    cd = robot.get_control_dict()
    ac = robot.controllers["arm_" + arm]; dof = ac.dof_idx
    q = th.as_tensor(arr(cd["joint_position"]), dtype=th.float32)[dof]
    J = th.as_tensor(arr(cd["eef_%s_jacobian_relative" % arm]), dtype=th.float32)[:, dof]
    pos_rel, quat_rel = robot.get_relative_eef_pose(arm)
    dpos = th.as_tensor(tp, dtype=th.float32) - th.as_tensor(arr(pos_rel), dtype=th.float32)
    dori = T.orientation_error(T.quat2mat(th.as_tensor(tq, dtype=th.float32)),
                               T.quat2mat(th.as_tensor(arr(quat_rel), dtype=th.float32)))
    err = th.cat([dpos, dori]); JT = J.T
    dq = JT @ th.linalg.solve(J @ JT + 1e-4 * th.eye(6), err)
    target_q = q + th.clamp(dq, -0.05, 0.05)
    action = th.zeros(robot.action_dim)
    action[robot.controller_action_idx["arm_" + arm]] = ac._reverse_preprocess_command(target_q)
    action[robot.controller_action_idx["base"]] = th.as_tensor(base_cmd, dtype=th.float32)
    action[robot.controller_action_idx["gripper_" + arm]] = -1.0 if gripper_closed else 1.0
    return action

def sim_step(tp, tq, gripper_closed):
    set_cam()
    env.step(compute_action(tp, tq, gripper_closed, np.zeros(3)))
    step_count[0] += 1
    if step_count[0] % 2 == 0: grab()

def goto_world(target_pos_w, target_quat_w, nsteps, gripper_closed):
    for _ in range(nsteps):
        rp_, rq_ = robot.get_position_orientation()
        tp, tq = T.relative_pose_transform(th.as_tensor(target_pos_w, dtype=th.float32),
                                           th.as_tensor(target_quat_w, dtype=th.float32), rp_, rq_)
        sim_step(arr(tp), arr(tq), gripper_closed)

def face_base(target_xy, gripper_closed, tol=0.04, max_steps=260):
    """Rotate the holonomic base IN PLACE (velocity commands, arm holding its current
    RELATIVE pose) until facing target_xy. v4: every reach/transport/set-down leg first
    faces its target -- the arm tracks poorly >~5 cm off-axis (v3 plateaued 24 cm short
    of the southward box target), but dead-ahead <=0.40 m is the trash-proven envelope."""
    hp, hq = robot.get_relative_eef_pose(arm)
    hp, hq = arr(hp), arr(hq)
    n = 0
    while n < max_steps:
        bp, bq = robot.get_position_orientation()
        bxy = arr(bp)[:2]; yw = yaw_of(arr(bq))
        d = np.array(target_xy, float) - bxy
        yerr = (math.atan2(d[1], d[0]) - yw + math.pi) % (2 * math.pi) - math.pi
        if abs(yerr) < tol:
            break
        set_cam()
        env.step(compute_action(hp, hq, gripper_closed,
                                np.array([0.0, 0.0, float(np.clip(2.0 * yerr, -0.5, 0.5))])))
        step_count[0] += 1
        if step_count[0] % 2 == 0: grab()
        n += 1
    for _ in range(12):
        sim_step(hp, hq, gripper_closed)
    bq = robot.get_position_orientation()[1]
    log("FACE done yaw=%.3f after %d steps (target %s)" % (yaw_of(arr(bq)), n, np.round(target_xy, 2).tolist()))

def wrap_ang(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def drive_to(target_xy, gripper_closed, tol=0.05, max_steps=800):
    """Drive the holonomic base to target_xy by TURNING then driving FORWARD (the
    trash-proven scheme). v8 tried strafing -- TidyBot's base tracks vx but barely
    responds to vy commands, leaving every standoff 10-19 cm short. Arm holds its
    current RELATIVE pose. Free-space segments only."""
    hp, hq = robot.get_relative_eef_pose(arm)
    hp, hq = arr(hp), arr(hq)
    n = 0
    while n < max_steps:
        bp, bq = robot.get_position_orientation()
        bxy = arr(bp)[:2]; yw = yaw_of(arr(bq))
        d = np.array(target_xy, float) - bxy
        dist = float(np.linalg.norm(d))
        if dist < tol:
            break
        des_yaw = math.atan2(d[1], d[0])
        yerr = (des_yaw - yw + math.pi) % (2 * math.pi) - math.pi
        if abs(yerr) > 0.30:
            cmd = np.array([0.0, 0.0, float(np.clip(2.0 * yerr, -0.6, 0.6))])
        else:
            cmd = np.array([min(0.35, 1.2 * dist), 0.0, float(np.clip(1.5 * yerr, -0.4, 0.4))])
        set_cam()
        env.step(compute_action(hp, hq, gripper_closed, cmd))
        step_count[0] += 1
        if step_count[0] % 2 == 0: grab()
        n += 1
    for _ in range(12): sim_step(hp, hq, gripper_closed)
    bp = arr(robot.get_position_orientation()[0])
    _res = float(np.linalg.norm(np.array(target_xy, float) - bp[:2]))
    log("DRIVE done at %s (target %s, %d steps)%s" % (np.round(bp[:2], 2).tolist(),
        np.round(target_xy, 2).tolist(), n, "  DRIVE_SHORT %.2fm" % _res if _res > 0.10 else ""))

boundaries = {}
def mark(name):
    boundaries[name] = step_count[0]
    log("BOUNDARY %s = %d" % (name, step_count[0]))
    for _nm in objs:
        _p = arr(objs[_nm].get_position_orientation()[0])
        log("  POSE %-16s %s" % (_nm, np.round(_p, 3).tolist()))

def linkpos(nm): return arr(robot.links[nm].get_position_orientation()[0])

# MEASURE the fingertip hang rather than assume it: every clearance decision in this
# script (SAFE_Z, the drop release height, the travel margin over the 1.040 rim) rests
# on it, and the probe only ever logged the eef point.
_lf0 = linkpos("hande_left_finger"); _rf0 = linkpos("hande_right_finger")
_ep00 = arr(robot.get_eef_pose(arm)[0])
FINGER_HANG = float(_ep00[2] - min(_lf0[2], _rf0[2]))
log("FINGER_HANG=%.3f (eef_z=%.3f left_z=%.3f right_z=%.3f span=%.3f)" % (
    FINGER_HANG, float(_ep00[2]), float(_lf0[2]), float(_rf0[2]),
    float(np.linalg.norm(_lf0 - _rf0))))

def rim_grasp_quat(u, tilt_deg=None):
    """Grasp orientation for pinching a vertical wall whose outward horizontal normal
    is +u: approach near-top-down tilted slightly toward the vessel center (-u), slide
    (finger-finger) axis radial (= u). v2: u = v = vessel->cup direction, i.e. the
    grip is on the CUP-FACING side, keeping every contact target within ~0.39 m and
    <=17 deg of the fixed base standoff (far-side grips saturated the DLS-IK arm)."""
    epos, equat = robot.get_eef_pose(arm)
    R_eef = arr(T.quat2mat(equat))
    lf = linkpos("hande_left_finger"); rf = linkpos("hande_right_finger")
    slide_world = rf - lf; slide_world /= (np.linalg.norm(slide_world) + 1e-9)
    slide_eef = R_eef.T @ slide_world
    t = math.radians(TILT_DEG if tilt_deg is None else tilt_deg)
    into = -np.array([u[0], u[1], 0.0])
    approach_world = math.sin(t) * into + math.cos(t) * np.array([0.0, 0.0, -1.0])
    approach_world /= np.linalg.norm(approach_world)
    R_target = build_R(np.array([0, 0, 1.0]), approach_world, slide_eef, np.array([u[0], u[1], 0.0]))
    return T.mat2quat(th.tensor(R_target, dtype=th.float32)), approach_world

def do_grasp_token(token_name, grip_u, standoff_pt, sat_key, phase):
    """Grasp the TOKEN itself from inside its open vessel: drive + face + staged high
    travel + near-vertical descend + close + lift. grip_u = horizontal slide-axis
    direction for the pinch; sat_key selects the region SAT_BIAS."""
    tk = objs[token_name]
    tp = arr(tk.get_position_orientation()[0])
    grip = np.array([tp[0], tp[1], tp[2]]) + SAT_BIAS[sat_key]

    ep_w = arr(robot.get_eef_pose(arm)[0])
    gq_cur = arr(robot.get_eef_pose(arm)[1])
    bxy = arr(robot.get_position_orientation()[0])[:2]
    ret = bxy - ep_w[:2]; ret = ret / (np.linalg.norm(ret) + 1e-9)
    if phase == 3:
        # PHASE 3 ONLY. p3 starts with the arm parked over the cup (the drop pose), and
        # drive_to holds the arm's RELATIVE pose while the base turns 114deg toward the
        # box station -- sweeping the eef through the cup (v39: shoved 18cm W / 21cm S,
        # then tipped -> p4 abort). Retract along the eef->base axis to ~0.20 reach so
        # the swept arc clears the cup's near edge (0.31 from base).
        # NOT applied to p1: there the arm starts at the untuck pose, far from the cup,
        # and v39's proven retract already leaves it at ~0.20 reach. v41 tucked BOTH
        # phases and broke p1/p2 (which had been landing the cube 4/6).
        _reach_now = float(np.linalg.norm(ep_w[:2] - bxy))
        _pull = max(0.0, _reach_now - 0.20)
        tuck = th.tensor([ep_w[0] + _pull * ret[0], ep_w[1] + _pull * ret[1], SAFE_Z],
                         dtype=th.float32)
        goto_world(tuck, th.tensor(gq_cur, dtype=th.float32), 90, False)
        _et = arr(robot.get_eef_pose(arm)[0])
        log("@tuck p%d eef=%s reach=%.3f->%.3f (cup at %.3f from base)" % (
            phase, np.round(_et, 3).tolist(), _reach_now,
            float(np.linalg.norm(_et[:2] - bxy)),
            float(np.linalg.norm(arr(objs[CUP].get_position_orientation()[0])[:2] - bxy))))
    else:
        # p1's PROVEN pre-drive (v39: sugar grasped + dropped in on 4/6 attempts)
        goto_world(th.tensor([ep_w[0] + 0.08 * ret[0], ep_w[1] + 0.08 * ret[1] + 0.08, SAFE_Z],
                             dtype=th.float32),
                   th.tensor(gq_cur, dtype=th.float32), 60, False)
    drive_to(np.array(standoff_pt, float), False)
    face_base(grip[:2], False)
    # VERTICAL approach: a tilted approach's lateral descent line catches the vessel
    # rim (v27b) and lag-biasing it overshoots (v27c) -- descend straight down the bore
    gq, appr = rim_grasp_quat(np.array([grip_u[0], grip_u[1], 0.0]), tilt_deg=0.0)
    cam_focus[0] = ([tp[0], tp[1], tp[2] + 0.05], 0.85)
    pre = th.tensor(grip - 0.10 * appr, dtype=th.float32)   # hover low: centring rings at z>1.1
    tgt = th.tensor(grip - PAD_OFF * appr, dtype=th.float32)
    goto_world(th.tensor([grip[0], grip[1], SAFE_Z], dtype=th.float32), gq, 90, False)
    goto_world(pre, gq, N_REACH, False)
    mark("p%d_MP_end" % phase)
    def _fm():
        return (linkpos("hande_left_finger") + linkpos("hande_right_finger")) / 2.0
    cmd_xy = np.array([float(grip[0]), float(grip[1])])
    for _it in range(5):
        _err = _fm()[:2] - grip[:2]
        log("@grasp-aim p%d centre%d err_mm=%s" % (
            phase, _it, np.round(_err * 1000, 1).tolist()))
        if float(np.linalg.norm(_err)) < 0.012:
            break
        cmd_xy = cmd_xy - 0.6 * _err   # damped: gain-1 rings at this posture
        goto_world(th.tensor([cmd_xy[0], cmd_xy[1], float(pre[2])], dtype=th.float32),
                   gq, 70, False)
    goto_world(th.tensor([cmd_xy[0], cmd_xy[1], float(tgt[2])], dtype=th.float32),
               gq, N_APPROACH, False)
    _mid = _fm()
    log("@grasp-aim p%d landed err_vs_bait_mm=%s err_vs_token_mm=%s finger_z=%.3f" % (
        phase, np.round((_mid[:2] - grip[:2]) * 1000, 1).tolist(),
        np.round((_mid[:2] - tp[:2]) * 1000, 1).tolist(), float(_mid[2])))
    tgt2 = th.tensor([cmd_xy[0], cmd_xy[1], float(tgt[2])], dtype=th.float32)
    mid = (linkpos("hande_left_finger") + linkpos("hande_right_finger")) / 2.0
    log("@pre-close finger_mid=%s grip_tgt=%s diff_mm=%s" % (
        np.round(mid, 3).tolist(), np.round(grip, 3).tolist(), np.round((mid - grip) * 1000, 1).tolist()))
    goto_world(tgt2, gq, N_GRASP, True)
    log("@post-close is_grasping=%s" % robot.is_grasping(arm))
    lift = tgt2.clone(); lift[2] += 0.16   # STRAIGHT up: lateral retract inside the bore clips the rim
    goto_world(lift, gq, N_LIFT, True)
    tz = float(arr(tk.get_position_orientation()[0])[2])
    grasped = str(robot.is_grasping(arm)).endswith("TRUE")
    log("after lift: %s z=%.3f (start %.3f) is_grasping=%s" % (token_name, tz, tp[2], robot.is_grasping(arm)))
    if tz - tp[2] < 0.06 and not grasped:
        log("GRASP_FAILED phase %d: %s not lifted and not attached (dz=%.3f)" % (phase, token_name, tz - tp[2]))
        early_abort("p%d token not grasped" % phase)
    # attached-but-jammed lifts (hand body under the vessel rim) resolve during the
    # next phase's safe-lift; the p2/p4 in-cup checks remain the true gates
    mark("p%d_term" % phase)
    return gq

def do_drop(token_name, gq, phase):
    """Carry the held token over the cup and release: the trash-task drop recipe.
    Free-space: safe-lift + drive to the cup standoff + face + hover (MP_end).
    Contact replay: short descend + open + retreat (term)."""
    ep_w = arr(robot.get_eef_pose(arm)[0])
    _bxy = arr(robot.get_position_orientation()[0])[:2]
    _tk_lift = arr(objs[token_name].get_position_orientation()[0])
    _ep_lift = arr(robot.get_eef_pose(arm)[0])
    _ride = float(_ep_lift[2] - _tk_lift[2])
    _pinch = str(robot.is_grasping(arm)).endswith("TRUE")
    log("@carry p%d after-lift token=%s eef=%s below_eef=%.3f pinched=%s" % (
        phase, np.round(_tk_lift, 3).tolist(), np.round(_ep_lift, 3).tolist(),
        _ride, _pinch))
    if not _pinch:
        early_abort("p%d token not attached (rides %.3f below eef)" % (phase, _ride))
    _vessel = TEACUP if phase == 2 else BOX
    _v_bot = float(arr(objs[_vessel].aabb[0])[2])
    if _v_bot > 0.92:
        early_abort("p%d VESSEL STOLEN: %s bottom at %.3f (counter 0.888)" % (
            phase, _vessel, _v_bot))
    _excess = _ride - FINGER_HANG
    _fly_z = SAFE_Z + (_excess if _excess > 0.03 else 0.0)
    log("@carry p%d fly_z=%.3f (ride %.3f, hang %.3f)" % (phase, _fly_z, _ride, FINGER_HANG))
    goto_world(th.tensor([ep_w[0] + 0.06, ep_w[1], _fly_z], dtype=th.float32), gq, 70, True)
    if phase == 4:
        _ep_t = arr(robot.get_eef_pose(arm)[0])
        _ret = _bxy - _ep_t[:2]; _ret = _ret / (np.linalg.norm(_ret) + 1e-9)
        _reach_now = float(np.linalg.norm(_ep_t[:2] - _bxy))
        _pull = max(0.0, _reach_now - 0.20)
        goto_world(th.tensor([_ep_t[0] + _pull * _ret[0], _ep_t[1] + _pull * _ret[1],
                              _fly_z], dtype=th.float32), gq, 90, True)
        _ep_t2 = arr(robot.get_eef_pose(arm)[0])
        log("@tuck p%d eef=%s reach=%.3f->%.3f (cup at %.3f from base)" % (
            phase, np.round(_ep_t2, 3).tolist(), _reach_now,
            float(np.linalg.norm(_ep_t2[:2] - _bxy)),
            float(np.linalg.norm(arr(objs[CUP].get_position_orientation()[0])[:2] - _bxy))))
    cup_now = arr(objs[CUP].get_position_orientation()[0])
    S = np.array([max(4.79, cup_now[0] + 0.33), cup_now[1]])
    _bd = float(np.linalg.norm(_bxy - S))
    if _bd > 0.08:
        drive_to(S, True, tol=0.03)
    else:
        log("@drop-drive skipped: base %s already %.3f from S %s" % (
            np.round(_bxy, 3).tolist(), _bd, np.round(S, 3).tolist()))
    face_base(cup_now[:2], True)
    _hp, _hq = robot.get_eef_pose(arm)
    for _ in range(50):   # hold still: let the cup's wobble decay before reading it
        goto_world(_hp, _hq, 1, True)
    # RE-LOCK on the live cup once the base is parked: earlier phases' arm sweeps
    # can nudge the cup (v30: 13cm), and a stale aim releases the cube where the
    # cup WAS. One re-read shared by every stage below, so the descent stays
    # vertical -- replay-safe.
    cup_live = arr(objs[CUP].get_position_orientation()[0])
    rim = float(arr(objs[CUP].aabb[1])[2])
    _cq_now = arr(objs[CUP].get_position_orientation()[1])
    if rim < 0.95 or abs(float(_cq_now[0])) > 0.15 or abs(float(_cq_now[1])) > 0.15:
        # rim<0.95 = fallen off the counter; |qx|/|qy|>0.1 = genuinely tipped. A cup
        # mid-wobble reads rim 1.069-1.088 transiently but stays near pure yaw --
        # the old rim-band guard aborted those healthy runs.
        early_abort("p%d cup fallen or tipped (rim=%.3f quat=%s)" % (
            phase, rim, np.round(_cq_now, 3).tolist()))
    _cmd_reach = float(S[0] - (cup_live[0] + DROP_BIAS[0]))
    if not (0.25 <= _cmd_reach <= 0.41):
        # the station map is only measured over cmd_reach 0.25-0.41 and is
        # non-monotonic -- outside it the release pose is uncalibrated, so recycle
        early_abort("p%d drop cmd_reach %.3f outside the probed band" % (phase, _cmd_reach))
    log("@drop-relock: cup_live=%s rim=%.3f" % (np.round(cup_live, 3).tolist(), rim))
    cam_focus[0] = ([cup_live[0], cup_live[1], rim + 0.05], 0.9)
    # KEEP THE GRASP'S WRIST. Do not roll the wrist mid-carry: the sugar is grasped
    # with u=[0,-1] and a recompute to the canonical u=[1,0] rolls it 90deg, which
    # drops the cube in transit (v44: token below the fingertips / already on the
    # counter before the release; the die, grasped at u=[1,0] and never rolled,
    # always survives). The station map was probed at u=[1,0], so p2's drop runs
    # off-map -- that is what the @drop-aim correction below measures out.
    dx = float(cup_live[0] + DROP_BIAS[0])
    dy = float(cup_live[1] + DROP_BIAS[1])
    # THE HOVER IS THE RELEASE POSE -- there is no descent stage. Commanding
    # [cup + DROP_BIAS, SAFE_Z] lands the eef over the cup centre at z~1.15, i.e.
    # fingertips ~1.11 = ~3cm above the 1.081 rim, and the cube falls into the
    # 2.2x basin. Every previous version instead commanded a "lower" stage to
    # rim+0.0x: at those z the arm tracks to a DIFFERENT branch (probe: cmd 1.06
    # tracks tight, cmd 1.10-1.24 does not), so the fingers arrived at the rim
    # plane 7-13cm off-centre and shoved the cup instead of dropping into it.
    _tk_pre = arr(objs[token_name].get_position_orientation()[0])
    _ep_pre = arr(robot.get_eef_pose(arm)[0])
    log("@carry p%d at-station token=%s eef=%s below_eef=%.3f pinched=%s" % (
        phase, np.round(_tk_pre, 3).tolist(), np.round(_ep_pre, 3).tolist(),
        float(_ep_pre[2] - _tk_pre[2]),
        abs(float(_ep_pre[2] - _tk_pre[2]) - FINGER_HANG) < 0.03))
    goto_world(th.tensor([dx, dy, _fly_z], dtype=th.float32), gq, 200, True)
    _tk_now = arr(objs[token_name].get_position_orientation()[0])
    _ep_now = arr(robot.get_eef_pose(arm)[0])
    _err = _tk_now[:2] - cup_live[:2]
    log("@drop-aim token=%s eef=%s token_off_eef=%s err_vs_cup=%s" % (
        np.round(_tk_now, 3).tolist(), np.round(_ep_now, 3).tolist(),
        np.round((_tk_now[:2] - _ep_now[:2]) * 1000, 1).tolist(),
        np.round(_err * 1000, 1).tolist()))
    dx, dy = float(dx - 0.7 * _err[0]), float(dy - 0.7 * _err[1])
    drop_pose = th.tensor([dx, dy, _fly_z], dtype=th.float32)
    goto_world(drop_pose, gq, 140, True)
    _tk2 = arr(objs[token_name].get_position_orientation()[0])
    log("@drop-aim corrected: token=%s residual_vs_cup=%s" % (
        np.round(_tk2, 3).tolist(), np.round((_tk2[:2] - cup_live[:2]) * 1000, 1).tolist()))
    ep2 = arr(robot.get_eef_pose(arm)[0])
    log("@drop-stage release: eef=%s cmd=[%.3f,%.3f,%.3f] rim=%.3f fingertip_clear=%+.3f" % (
        np.round(ep2, 3).tolist(), dx, dy, _fly_z, rim, ep2[2] - 0.04 - rim))
    mark("p%d_MP_end" % phase)
    goto_world(drop_pose, gq, N_OPEN, False)
    # NO retreat -- hold the release pose like the proven trash drop. A retreat here
    # sits inside the contact-replay segment, commands a reach below the probed band,
    # and sweeps the hand across the cup's east rim (4mm from the counter edge).
    # Phase 3's safe-lift does the retracting. Per-step telemetry distinguishes a
    # bounce-out from a mis-aimed drop.
    for _i in range(60):
        goto_world(drop_pose, gq, 1, False)
        if _i % 12 == 0:
            _tk = arr(objs[token_name].get_position_orientation()[0])
            _cp = arr(objs[CUP].get_position_orientation()[0])
            log("@drop-settle t=%2d token=%s cup=%s dxy=%.3f" % (_i,
                np.round(_tk, 3).tolist(), np.round(_cp, 3).tolist(),
                float(np.linalg.norm(_tk[:2] - _cp[:2]))))
    tk = arr(objs[token_name].get_position_orientation()[0])
    cup_end = arr(objs[CUP].get_position_orientation()[0])
    log("@drop %s pos=%s cup_end=%s (dxy_mm=%s)" % (token_name, np.round(tk, 3).tolist(),
        np.round(cup_end, 3).tolist(), np.round((tk[:2] - cup_end[:2]) * 1000, 1).tolist()))
    mark("p%d_term" % phase)

# ================= LAYOUT PROBE (MC_PROBE=2) =================
# The cup boots with a 152.3mm z-extent = its BODY DIAMETER, i.e. lying on its side,
# from a saved quat that reads "tilt 0" only if the model's local +z is its up-axis.
# Measure the convention rather than assume it: log the boot quat, then try candidate
# uprights in the same boot and report which one actually stands the cup up.
if os.environ.get("MC_PROBE") == "2":
    # LOOK AT THE CUP. Six versions of inferring its pose from AABB extents produced a
    # contradiction (canonical and "ejected" poses both report origin-above-base=0.081
    # yet differ 152 vs 194 in z-extent), so render it instead of reasoning about it.
    import imageio
    OUT = os.environ.get("MC_PROBE_DIR", "/tmp/jc_cuplook")
    os.makedirs(OUT, exist_ok=True)
    def _shot(tag):
        p_, q_ = objs[CUP].get_position_orientation()
        p_ = arr(p_); q_ = arr(q_)
        lo, hi = (arr(a) for a in objs[CUP].aabb)
        cam_focus[0] = ([float(p_[0]), float(p_[1]), float(p_[2])], 0.62)
        set_cam()
        for _ in range(6):
            og.sim.render(); grab()
        if frames:
            imageio.imwrite(os.path.join(OUT, tag + ".png"), frames[-1])
        log("@look %-16s pos=%s quat=%s ext_mm=%s aabb_z=[%.3f,%.3f]" % (
            tag, np.round(p_, 3).tolist(), np.round(q_, 4).tolist(),
            np.round((hi - lo) * 1000, 1).tolist(), float(lo[2]), float(hi[2])))
    def _qmul(a, b):
        ax, ay, az, aw = a; bx, by, bz, bw = b
        return [aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw,
                aw*bw - ax*bx - ay*by - az*bz]
    _shot("00_boot_as_saved")
    _yaw = 127.1 * math.pi / 180.0
    _qz = [0.0, 0.0, math.sin(_yaw / 2), math.cos(_yaw / 2)]
    _s2 = math.sin(math.pi / 4)
    for _nm, _qc in (("rotY-90", [0.0, -_s2, 0.0, _s2]),
                     ("rotY+90", [0.0, _s2, 0.0, _s2]),
                     ("rotX-90", [-_s2, 0.0, 0.0, _s2])):
        _q = th.tensor(_qmul(_qz, _qc), dtype=th.float32)
        objs[CUP].set_position_orientation(th.tensor([4.40, -1.25, 2.0], dtype=th.float32), _q)
        og.sim.step()
        _lo = float(arr(objs[CUP].aabb[0])[2])
        _seat = 0.8895 + (2.0 - _lo)
        objs[CUP].set_position_orientation(
            th.tensor([4.40, -1.25, _seat], dtype=th.float32), _q)
        og.sim.step()
        _shot("10_seated_" + _nm)          # the commanded pose, before physics reacts
        for _ in range(200): og.sim.step()
        _shot("20_settled_" + _nm)         # where it actually comes to rest
    log("@look DONE dir=%s" % OUT)
    og.shutdown(); os._exit(0)

# ================= PROBE MODE =================
# MC_PROBE=1: map the DROP STATION's achieved-vs-commanded eef, then exit. 30+
# source iterations have tuned biases against an arm whose descent stalls ~9cm
# high and ~11cm off regardless of the command (v32/v33/v34 identical). Measure
# the station instead of guessing at it: park exactly as do_drop does, then sweep
# commanded (x, z) with the hand EMPTY-but-closed and log achieved eef per point.
if os.environ.get("MC_PROBE"):
    cup_live = arr(objs[CUP].get_position_orientation()[0])
    rim = float(arr(objs[CUP].aabb[1])[2])
    log("@probe cup=%s rim=%.3f cup_aabb_lo=%s cup_aabb_hi=%s" % (
        np.round(cup_live, 3).tolist(), rim,
        np.round(arr(objs[CUP].aabb[0]), 3).tolist(),
        np.round(arr(objs[CUP].aabb[1]), 3).tolist()))
    for _nm, _o in objs.items():
        log("@probe OBJ %-18s aabb=%s -> %s" % (_nm,
            np.round(arr(_o.aabb[0]), 3).tolist(), np.round(arr(_o.aabb[1]), 3).tolist()))
    S = np.array([max(4.79, cup_live[0] + 0.33), cup_live[1]])
    drive_to(S, True, tol=0.03)
    face_base(cup_live[:2], True)
    bp, bq = robot.get_position_orientation()
    bxy = arr(bp)[:2]; byaw = yaw_of(arr(bq))
    log("@probe BASE parked xy=%s yaw=%.3f (S=%s)" % (
        np.round(bxy, 3).tolist(), byaw, np.round(S, 3).tolist()))
    # counter footprint: the layout redesign needs the real support surface, not
    # the guess that "objects at x<=4.52 are safe"
    for _o in env.scene.objects:
        try:
            _lo, _hi = (arr(a) for a in _o.aabb)
        except Exception:
            continue
        if (_lo[0] < 4.6 < _hi[0] + 0.4 and _lo[1] < -1.2 < _hi[1]
                and 0.5 < _hi[2] < 1.0 and (_hi[0] - _lo[0]) > 0.3):
            log("@probe SUPPORT %-30s aabb=%s -> %s" % (_o.name,
                np.round(_lo, 3).tolist(), np.round(_hi, 3).tolist()))
    gq, _ = rim_grasp_quat([1.0, 0.0], tilt_deg=0.0)
    # sweep: for each commanded reach (base_x - cmd_x) and z, settle then log achieved
    for reach in [0.25, 0.29, 0.33, 0.37, 0.41]:
        for z in [1.24, 1.16, 1.10, 1.06]:
            cx = float(bxy[0] - reach); cy = float(bxy[1])
            goto_world(th.tensor([cx, cy, 1.24], dtype=th.float32), gq, 90, True)
            goto_world(th.tensor([cx, cy, z], dtype=th.float32), gq, 170, True)
            ep = arr(robot.get_eef_pose(arm)[0])
            ach_reach = float(bxy[0] - ep[0])
            log("@probe cmd_reach=%.2f cmd_z=%.2f -> eef=%s ach_reach=%.3f dz=%+.3f dy=%+.3f" % (
                reach, z, np.round(ep, 3).tolist(), ach_reach, ep[2] - z, ep[1] - cy))
    log("@probe DONE")
    og.shutdown(); os._exit(0)

# ================= PHASES =================
# 1: grasp sugar cube (inside 1.6x teacup)   2: drop into cup
# 3: grasp die (inside open toy box)         4: drop into cup
gq1 = do_grasp_token(SUGAR, [0.0, -1.0], S_TEA, "sugar", 1)
do_drop(SUGAR, gq1, 2)
if not token_in_cup(SUGAR):
    early_abort("p2 sugar not in cup")
gq3 = do_grasp_token(DIE, [0.0, -1.0], S_BOX, "die", 3)   # the sugar's wrist: die landings under u=[1,0] were chaotic
do_drop(DIE, gq3, 4)
if not token_in_cup(DIE):
    early_abort("p4 die not in cup")
if not token_in_cup(SUGAR):
    early_abort("p4 sugar knocked out")

# ================= WRAP UP =================
for _ in range(30): sim_step(*[arr(x) for x in robot.get_relative_eef_pose(arm)], False)
for nm in (SUGAR, DIE):
    p = arr(objs[nm].get_position_orientation()[0])
    log("FINAL %s pos=%s (cup=%s rim_z=%.3f)" % (nm, np.round(p, 3).tolist(),
                                                 np.round(cup_p, 3).tolist(), CUP_RIM_Z))
try: log("BDDL_SUCCESS=%s" % env.task.success)
except Exception as e: log("success err %s" % e)
log("BOUNDARIES: %s" % boundaries)
log("TOTAL_STEPS=%d" % step_count[0])

env.save_data()
with h5py.File(OUTPUT, "r+") as f:
    demos = sorted(f["data"].keys())
    if "mask" not in f: f.create_group("mask")
    if "use" in f["mask"]: del f["mask"]["use"]
    f["mask"].create_dataset("use", data=np.array([demos[-1].encode()]))
log("SAVED %s tagged %s" % (OUTPUT, demos[-1]))
if frames:
    import imageio
    imageio.mimsave(VIDEO, frames[::2], fps=20, macro_block_size=1)
    log("VIDEO_SAVED %s frames=%d" % (VIDEO, len(frames)))
log("SCRIPT_COFFEE_DONE")
og.shutdown()
