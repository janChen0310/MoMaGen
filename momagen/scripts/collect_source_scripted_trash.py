# NOTE: imported from the TidyBot dispose-trash pipeline (see docs/tutorials/tidybot-task-pipelines.md).
# Server-specific absolute paths (e.g. dataset/template locations) may need adjusting to your setup.
"""SCRIPTED TidyBot pick-and-dispose source demo (headless), for MoMaGen datagen.

Task (BDDL datagen_picking_up_trash in house_single_floor): drive from the BDDL spawn to the
kitchen counter, grasp the (0.5-scaled) can_of_soda_595, carry it to trash_can_596 on the
floor, drop it in. Records via DataCollectionWrapper (+ mask/use), logs the phase-boundary
step indices needed for the MoMaGen base config, prints BDDL success, and writes an mp4.

datagen_info (world-frame SE(3) geometry) is recorded INLINE, one entry per executed
env.step, via DatagenInfoRecorder -- so the output of this script is directly
generation-ready. prepare_src_dataset.py (the only OmniGibson-version-coupled stage) does
NOT need to run afterward.

NO base teleports (they corrupt the holonomic articulation): the base is DRIVEN with velocity
commands along traversability-map waypoints. Grasp math reuses the proven tilted-approach
recipe from momagen/scripts/script_tidybot_source_demo.py.
"""
import json, math, os
os.environ["OMNIGIBSON_HEADLESS"] = "1"; os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
# NB: do NOT set OMNIGIBSON_GPU_ID here -- on multi-GPU boxes the worker GPU is picked
# with CUDA_VISIBLE_DEVICES alone; combining the two breaks Vulkan device enumeration
# (Isaac Kit segfaults at boot in the XR viewport extension). Same fix as
# collect_source_scripted_coffee.py; verified by reproducing the segfault on this
# script before removing the setdefault(...) that used to be here.
import h5py
import numpy as np
import torch as th
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataCollectionWrapper
from omnigibson.macros import gm
from momagen.env_interfaces.base import make_interface
from momagen.scripts.collect_tidybot_source_demo import load_tidybot_env_config
from momagen.utils.datagen_info_recorder import DatagenInfoRecorder

ENV_INTERFACE_NAME = "MG_TidyBotPickingUpTrash"
ENV_INTERFACE_TYPE = "omnigibson_tidybot"

REPO = "/root/MoMaGen"
TEMPLATE = REPO + "/momagen/datasets/source_og/r1_picking_up_trash.hdf5"
OUTPUT = REPO + "/momagen/datasets/source_og/tidybot_picking_up_trash.hdf5"
SCENE_INSTANCE = "house_single_floor_task_datagen_picking_up_trash_0_0_template"
CAN = "can_of_soda_595"; TRASH = "trash_can_596"
VIDEO = "/root/rivermind-data/script_trash_demo.mp4"

# grasp tunables (proven values from script_tidybot_source_demo.py)
TILT_DEG = 15.0    # near-TOP-DOWN: pads close parallel to the can axis -> no tipping torque
                   # (the 40deg tilt contacted the upright cylinder on a slant and tipped it over)
PAD_OFF = 0.05; LIFT_DZ = 0.30
GRIP_DROP = 0.0
N_REACH = 180; N_APPROACH = 180; N_GRASP = 70; N_LIFT = 130
# Base in the aisle corner pocket + can nudged to the counter's aisle-side edge: the proven
# grasp recipe works at ~0.4m standoff; the sampled pose (1.0m) is beyond the Gen3 envelope.
# (Generation randomizes object poses per attempt, so adjusting the source layout is standard.)
# Base shifted by the measured finger-residual from v4 (fingers plateaued 5cm NE of the can:
# arm workspace saturation under the tilt constraint -> move the base, not the IK target).
GRASP_STANDOFF = np.array([4.79, -1.28])
CAN_START = [4.44, -1.38, 0.95]            # east edge of the kelker counter, near the corner

def log(m): print(m, flush=True)
def arr(x): return np.array(x.cpu() if hasattr(x, "cpu") else x, float)
def yaw_of(q):
    x, y, z, w = q; return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

def build_R(a1, b1, a2, b2):
    def n(v): return v / (np.linalg.norm(v) + 1e-9)
    a1 = n(a1); a2 = n(a2 - (a2 @ a1) * a1); a3 = np.cross(a1, a2)
    b1 = n(b1); b2 = n(b2 - (b2 @ b1) * b1); b3 = np.cross(b1, b2)
    A = np.stack([a1, a2, a3], axis=1); B = np.stack([b1, b2, b3], axis=1)
    return B @ A.T

cfg = load_tidybot_env_config(TEMPLATE)
cfg["scene"]["scene_model"] = "house_single_floor"
cfg["scene"]["scene_instance"] = SCENE_INSTANCE
cfg["scene"]["scene_file"] = None
cfg["robots"][0]["grasping_mode"] = "physical"
cfg["robots"][0]["self_collisions"] = False
# LOAD-TIME spawn in the kitchen at the validated collision-free standoff, facing the can.
# (The template's living-room spawn is an R1 leftover behind a CLOSED door; BDDL wants the
# agent on the kitchen floor. Load-time placement is safe -- unlike runtime teleports.)
_syaw = math.atan2(CAN_START[1] - GRASP_STANDOFF[1], CAN_START[0] - GRASP_STANDOFF[0])
cfg["robots"][0]["position"] = [float(GRASP_STANDOFF[0]), float(GRASP_STANDOFF[1]), 0.0]
cfg["robots"][0]["orientation"] = [0.0, 0.0, math.sin(_syaw / 2), math.cos(_syaw / 2)]
gm.ENABLE_TRANSITION_RULES = False
log("building env...")
env = og.Environment(configs=cfg)
env = DataCollectionWrapper(env=env, output_path=OUTPUT, only_successes=False)
robot = env.robots[0]; arm = robot.default_arm
env.reset()
for _ in range(40): og.sim.step()

# Built once, before the recorded episode starts (nothing above this point is an env.step
# that DataCollectionWrapper records -- scene load, reset, and settling are raw og.sim.step()
# calls). get_datagen_info() reads only live sim state, so it is safe to call every recorded
# step; doing so here deletes prepare_src_dataset.py from the critical path (see
# momagen/utils/datagen_info_recorder.py).
recorder = DatagenInfoRecorder(
    make_interface(name=ENV_INTERFACE_NAME, interface_type=ENV_INTERFACE_TYPE, env=env),
    ENV_INTERFACE_NAME, ENV_INTERFACE_TYPE)

can = env.scene.object_registry("name", CAN); trash = env.scene.object_registry("name", TRASH)
# nudge the can to the counter's aisle-side edge (in reach of the standoff)
can.set_position_orientation(position=th.tensor(CAN_START, dtype=th.float32))
can.keep_still()
for _ in range(30): og.sim.step()
cpos = arr(can.get_position_orientation()[0]); tpos = arr(trash.get_position_orientation()[0])
clo, chi = (arr(a) for a in can.aabb)
log("can settled pos=%s aabb_mm=%s" % (np.round(cpos, 3).tolist(), np.round((chi - clo) * 1000, 1).tolist()))
log("trash pos=%s" % np.round(tpos, 3).tolist())
bp0 = arr(robot.get_position_orientation()[0])
log("spawn=%s dist_to_can=%.2f" % (np.round(bp0, 2).tolist(), float(np.linalg.norm(bp0[:2] - cpos[:2]))))

# --- camera (chase, follows robot) + recording ---
import omni.replicator.core as rep
from pxr import UsdGeom, Gf
stg = og.sim.stage
cam = UsdGeom.Camera.Define(stg, "/World/jc_cam"); cam.GetFocalLengthAttr().Set(16.0)
cam.GetHorizontalApertureAttr().Set(24.0); cam.GetClippingRangeAttr().Set((0.02, 200.0))
camxf = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
cam_focus = [None]  # None = chase the robot; else (point3, dist) close-up orbit of that point
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

# --- control primitives ---
robot_path = robot.prim_path
def nonfloor_hits():
    hits = set()
    try:
        for c in robot.contact_list():
            for b in (c.body0, c.body1):
                if not b or robot_path in b: continue
                if "floor" in b.lower() or "ground" in b.lower(): continue
                hits.add(b.split("/")[-2])
    except Exception: pass
    return hits

step_count = [0]
hold_rel = [None]  # (pos, quat) eef target in ROBOT frame, held during nav
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

def sim_step(tp, tq, gripper_closed, base_cmd):
    set_cam()
    action = compute_action(tp, tq, gripper_closed, base_cmd)
    env.step(action)
    recorder.record(action=action)  # paired 1:1 with the env.step DataCollectionWrapper just recorded
    step_count[0] += 1
    if step_count[0] % 2 == 0: grab()

def hold_pose():
    p, q = robot.get_relative_eef_pose(arm)
    return arr(p), arr(q)

def goto_world(target_pos_w, target_quat_w, nsteps, gripper_closed):
    """Move eef to a WORLD pose (base still)."""
    for _ in range(nsteps):
        rp, rq = robot.get_position_orientation()
        tp, tq = T.relative_pose_transform(th.as_tensor(target_pos_w, dtype=th.float32),
                                           th.as_tensor(target_quat_w, dtype=th.float32), rp, rq)
        sim_step(arr(tp), arr(tq), gripper_closed, np.zeros(3))

def drive_to(target_xy, gripper_closed, face_xy=None, tol=0.18, max_steps=900):
    """Drive the base along trav-map waypoints; arm holds its current relative pose."""
    hp, hq = hold_pose()
    src = arr(robot.get_position_orientation()[0])[:2]
    try:
        path, _dist = env.scene.trav_map.get_shortest_path(
            0, th.tensor(src, dtype=th.float32), th.tensor(target_xy, dtype=th.float32),
            entire_path=True, robot=robot)
        wps = [arr(w)[:2] for w in path]
        log("NAV path %d waypoints %s -> %s" % (len(wps), np.round(src, 2).tolist(), np.round(target_xy, 2).tolist()))
    except Exception as e:
        log("NAV trav_map failed (%s); straight line" % e)
        wps = [np.array(target_xy, float)]
    wi = 0; n = 0
    while n < max_steps:
        bp, bq = robot.get_position_orientation()
        bxy = arr(bp)[:2]; yw = yaw_of(arr(bq))
        while wi < len(wps) - 1 and np.linalg.norm(wps[wi] - bxy) < 0.35: wi += 1
        tgt = wps[wi]
        d = tgt - bxy; dist = float(np.linalg.norm(d))
        if wi == len(wps) - 1 and dist < tol: break
        des_yaw = math.atan2(d[1], d[0]); yerr = wrap(des_yaw - yw)
        if abs(yerr) > 0.35:
            cmd = np.array([0.0, 0.0, np.clip(2.0 * yerr, -0.6, 0.6)])
        else:
            fwd = min(0.4, 1.2 * dist)
            cmd = np.array([fwd, 0.0, np.clip(1.5 * yerr, -0.4, 0.4)])
        sim_step(hp, hq, gripper_closed, cmd)
        n += 1
        if n % 60 == 0:
            log("NAV n=%d bxy=%s wp=%d/%d dist=%.2f hits=%s" % (
                n, np.round(bxy, 2).tolist(), wi + 1, len(wps), dist, sorted(nonfloor_hits())[:3]))
    if face_xy is not None:
        for _ in range(120):
            bp, bq = robot.get_position_orientation()
            bxy = arr(bp)[:2]; yw = yaw_of(arr(bq))
            d = np.array(face_xy) - bxy
            yerr = wrap(math.atan2(d[1], d[0]) - yw)
            if abs(yerr) < 0.06: break
            sim_step(hp, hq, gripper_closed, np.array([0., 0., np.clip(2.0 * yerr, -0.6, 0.6)]))
    for _ in range(15): sim_step(hp, hq, gripper_closed, np.zeros(3))
    bp = arr(robot.get_position_orientation()[0])
    log("NAV done at %s (target %s)" % (np.round(bp[:2], 2).tolist(), np.round(target_xy, 2).tolist()))

boundaries = {}
def mark(name):
    boundaries[name] = step_count[0]
    log("BOUNDARY %s = %d" % (name, step_count[0]))

# ================= PHASES =================
# NAV 1: spawn -> grasp standoff, face the can
drive_to(GRASP_STANDOFF, gripper_closed=False, face_xy=cpos[:2])
mark("nav1_end")

# GRASP (proven tilted approach)
bp = arr(robot.get_position_orientation()[0])
bar = arr(can.get_position_orientation()[0])   # re-read: can may have settled
outward = bp[:2] - bar[:2]; outward /= (np.linalg.norm(outward) + 1e-9)
def linkpos(nm): return arr(robot.links[nm].get_position_orientation()[0])
epos, equat = robot.get_eef_pose(arm)
R_eef = arr(T.quat2mat(equat))
lf = linkpos("hande_left_finger"); rf = linkpos("hande_right_finger")
slide_world = rf - lf; slide_world /= (np.linalg.norm(slide_world) + 1e-9)
slide_eef = R_eef.T @ slide_world
into = -np.array([outward[0], outward[1], 0.0]); into /= (np.linalg.norm(into) + 1e-9)
t = math.radians(TILT_DEG)
approach_world = math.sin(t) * into + math.cos(t) * np.array([0.0, 0.0, -1.0])
approach_world /= np.linalg.norm(approach_world)
target_slide = np.array([-outward[1], outward[0], 0.0])
R_target = build_R(np.array([0, 0, 1.0]), approach_world, slide_eef, target_slide)
grasp_quat_w = T.mat2quat(th.tensor(R_target, dtype=th.float32))
cam_focus[0] = ([bar[0], bar[1], bar[2] + 0.05], 0.85)   # close-up on the can for the grasp
grip_pt = bar + np.array([0.0, 0.0, -GRIP_DROP])
pre_pos_w = th.tensor(grip_pt - 0.20 * approach_world, dtype=th.float32)
grasp_pos_w = th.tensor(grip_pt - PAD_OFF * approach_world, dtype=th.float32)
lift_pos_w = grasp_pos_w.clone(); lift_pos_w[2] += LIFT_DZ

# SINGLE-PASS approach: pre-bias the target by the measured systematic saturation residual
# (from the v6 run's correction pass) instead of correcting closed-loop. A clean monotone
# approach segment transfers under MoMaGen's re-anchored replay; correction wiggles do not.
SAT_BIAS = np.array([0.01, 0.003, -0.029])
grasp_pos_w = grasp_pos_w + th.tensor(SAT_BIAS, dtype=th.float32)
lift_pos_w = grasp_pos_w.clone(); lift_pos_w[2] += LIFT_DZ
goto_world(pre_pos_w, grasp_quat_w, N_REACH, False)
mark("grasp_MP_end")            # end of free-space reach = MP_end_step for phase 1
goto_world(grasp_pos_w, grasp_quat_w, N_APPROACH + 60, False)
mid = (linkpos("hande_left_finger") + linkpos("hande_right_finger")) / 2.0
log("@pre-close finger_mid=%s can=%s diff=%s" % (np.round(mid, 3).tolist(), np.round(bar, 3).tolist(), np.round(mid - bar, 3).tolist()))
goto_world(grasp_pos_w, grasp_quat_w, N_GRASP, True)
log("@post-grasp is_grasping=%s can_z=%.3f" % (robot.is_grasping(arm), float(arr(can.get_position_orientation()[0])[2])))
goto_world(lift_pos_w, grasp_quat_w, N_LIFT, True)
can_z = float(arr(can.get_position_orientation()[0])[2])
log("after lift: can_z=%.3f (start %.3f) delta=%+.3f is_grasping=%s" % (can_z, bar[2], can_z - bar[2], robot.is_grasping(arm)))
mark("grasp_term")              # subtask_term_step for phase 1

# NAV 2: to trash standoff (short), face the trash
cam_focus[0] = None
toff = tpos[:2] + (GRASP_STANDOFF - tpos[:2]) / np.linalg.norm(GRASP_STANDOFF - tpos[:2]) * 0.60
drive_to(toff, gripper_closed=True, face_xy=tpos[:2])
mark("nav2_end")

# DROP: eef above the trash rim, open
cam_focus[0] = ([tpos[0], tpos[1], tpos[2] + 0.25], 1.0)   # close-up on the trash can
rim_z = tpos[2] + 0.14 + 0.18   # can center 0.13 + half-height + clearance
drop_pos_w = th.tensor([tpos[0], tpos[1], rim_z + 0.12], dtype=th.float32)
goto_world(drop_pos_w, grasp_quat_w, 170, True)
mark("drop_MP_end")             # end of free-space move-over-trash = MP_end for phase 2
goto_world(drop_pos_w, grasp_quat_w, 20, True)
goto_world(drop_pos_w, grasp_quat_w, 50, False)   # open -> can falls in
for _ in range(40): sim_step(*hold_pose(), False, np.zeros(3))
mark("drop_term")               # subtask_term_step for phase 2

cz = arr(can.get_position_orientation()[0])
log("can final pos=%s trash=%s" % (np.round(cz, 3).tolist(), np.round(tpos, 3).tolist()))
try: log("BDDL_SUCCESS=%s" % env.task.success)
except Exception as e: log("success err %s" % e)
log("BOUNDARIES: %s" % boundaries)
log("TOTAL_STEPS=%d" % step_count[0])

env.save_data()
assert len(recorder) == step_count[0], (
    "datagen_info count %d != recorded step count %d -- an env.step was not paired "
    "with a recorder.record() call" % (len(recorder), step_count[0]))
demo_key = recorder.write(OUTPUT)
log("DATAGEN_INFO_WRITTEN %s entries=%d" % (demo_key, len(recorder)))
with h5py.File(OUTPUT, "r+") as f:
    demos = sorted(f["data"].keys())
    if "mask" not in f: f.create_group("mask")
    if "use" in f["mask"]: del f["mask"]["use"]
    f["mask"].create_dataset("use", data=np.array([demos[-1].encode()]))
    # DataCollectionWrapper writes data.attrs["config"] but generate_dataset reads
    # data.attrs["env_args"] (robomimic env metadata, see momagen/utils/file_utils.py and
    # robomimic/utils/env_utils.py); add it so this freshly collected + inline-annotated
    # demo is generation-ready with NO prepare_src_dataset.py pass at all. Precedent:
    # momagen/scripts/script_tidybot_source_demo_curobo.py does the same patch.
    # env_kwargs == the OG config; env_name derived from the task's own activity_name so
    # this stays correct if the template/task ever changes.
    if "config" in f["data"].attrs and "env_args" not in f["data"].attrs:
        cfg_d = json.loads(f["data"].attrs["config"])
        activity_name = cfg_d.get("task", {}).get("activity_name", "datagen")
        f["data"].attrs["env_args"] = json.dumps(
            {"env_name": "%s_D0" % activity_name, "type": 4, "env_kwargs": cfg_d})
log("SAVED %s tagged %s" % (OUTPUT, demos[-1]))
if frames:
    import imageio
    imageio.mimsave(VIDEO, frames[::2], fps=20, macro_block_size=1)
    log("VIDEO_SAVED %s frames=%d" % (VIDEO, len(frames)))
log("SCRIPT_TRASH_DONE")
og.shutdown()
