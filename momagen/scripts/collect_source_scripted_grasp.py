# NOTE: imported from the TidyBot dispose-trash pipeline (see docs/tutorials/tidybot-task-pipelines.md).
# Server-specific absolute paths (e.g. dataset/template locations) may need adjusting to your setup.
"""SCRIPTED TidyBot navigate-and-grasp source demo (headless), for MoMaGen datagen.

Task (tidybot_grasp_can): drive to the kitchen counter and grasp the (0.5-scaled)
can_of_soda_595. This is collect_source_scripted_trash.py with the carry-and-drop half
removed -- same proven grasp recipe, same inline datagen_info recording, one phase instead
of two.

The can position and the base standoff are arguments, because the source demo should cover
more than one approach geometry: MoMaGen re-anchors the contact-rich segment onto a new can
pose each attempt, and a single source geometry is a single point of failure for that.
Positions should come from the swept usable region (momagen/scripts/sweep_can_positions.py),
not be invented -- most of the counter cannot be reached from anywhere the base may stand.

datagen_info (world-frame SE(3) geometry) is recorded INLINE, one entry per executed
env.step, via DatagenInfoRecorder -- so the output is directly generation-ready and
prepare_src_dataset.py (the only OmniGibson-version-coupled stage) never runs.

NO base teleports (they corrupt the holonomic articulation): the base is DRIVEN with velocity
commands along traversability-map waypoints.
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

ENV_INTERFACE_NAME = "MG_TidyBotGraspCan"
ENV_INTERFACE_TYPE = "omnigibson_tidybot"

# Derived, not hardcoded: the original carried an absolute /root/MoMaGen that was wrong on
# every machine but one.
REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEMPLATE = REPO + "/momagen/datasets/source_og/r1_picking_up_trash.hdf5"
OUTPUT = os.environ.get("JC_OUT", REPO + "/momagen/datasets/source_og/tidybot_grasp_can.hdf5")
SCENE_INSTANCE = "house_single_floor_task_datagen_picking_up_trash_0_0_template"
CAN = "can_of_soda_595"
VIDEO = os.environ.get("JC_VIDEO", REPO + "/script_grasp_demo.mp4")

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
# Can position. Default is a high-viability cell from the swept region
# (momagen/datasets/can_region.json); override per source demo to vary the approach geometry.
#   JC_CAN_XY="4.37,-0.06"
_cxy = [float(v) for v in os.environ.get("JC_CAN_XY", "4.37,-0.06").split(",")]
CAN_START = [_cxy[0], _cxy[1], 0.95]
# The standoff is DERIVED from the walkway, not a fixed offset from the can. The trash demo's
# (+0.35, +0.10) offset is specific to where its can sat: applied at a different point on the
# counter it lands the base centre 0.23 m from the counter edge, and the base half-width is
# 0.332 m, so the target pose is inside the slab and the robot simply cannot get there. The
# first collection spent 400 steps failing to park before closing on empty air.
# Systematic saturation residual measured on the trash demo; applied to the grasp target both
# when scoring candidate standoffs and when executing the grasp, so the two agree.
SAT_BIAS_C = np.array([0.01, 0.003, -0.029])

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
# Load-time spawn: the task's own spawn band, out of arm reach, so the source demo contains a
# real drive rather than starting on top of the target. MoMaGen replaces this segment with its
# own plan at generation time, but a demo that never navigates is a demo that cannot be checked
# for navigation.
#
# The spawn is SAMPLED FROM THE WALKWAY, not hardcoded. A hand-picked [5.60, 0.30] that looked
# plausible on a render turned out to be inside a bar: the robot spent 1335 steps pinned against
# furniture, reporting hits=['bar_udatjt_0'], and never moved. The walkway mask is the traversable
# floor eroded by the chassis' circumscribed radius, so anything it returns is somewhere the robot
# can actually stand. It reads PNGs only, so this runs before the simulator exists.
from momagen.utils.kitchen_walkway import WalkwaySampler, default_dirs
_sd, _md = default_dirs(REPO)
_ws = WalkwaySampler(_sd, _md, clearance_m=float(os.environ.get("JC_SPAWN_CLEARANCE", "0.46")))
if os.environ.get("JC_SRC_SPAWN"):
    _SPAWN = [float(v) for v in os.environ["JC_SRC_SPAWN"].split(",")]
else:
    _sc = np.asarray(_ws.candidates(), dtype=float)
    _sd_ = np.linalg.norm(_sc - np.array(CAN_START[:2])[None], axis=1)
    _lo_s = float(os.environ.get("JC_SRC_SPAWN_MIN", "1.2"))
    _hi_s = float(os.environ.get("JC_SRC_SPAWN_MAX", "2.5"))
    _pool = _sc[(_sd_ >= _lo_s) & (_sd_ <= _hi_s)]
    if len(_pool) == 0:
        raise SystemExit("no walkway spawn %.1f-%.1f m from the can" % (_lo_s, _hi_s))
    _SPAWN = _pool[np.random.RandomState(int(os.environ.get("JC_SEED", "0"))).randint(len(_pool))].tolist()
log("spawn (walkway-sampled) %s at %.2f m from the can"
    % (np.round(_SPAWN, 3).tolist(), float(np.linalg.norm(np.array(_SPAWN) - np.array(CAN_START[:2])))))
_syaw = math.atan2(CAN_START[1] - _SPAWN[1], CAN_START[0] - _SPAWN[0])
cfg["robots"][0]["position"] = [_SPAWN[0], _SPAWN[1], 0.0]
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

can = env.scene.object_registry("name", CAN)

# The trash can belongs to the parent picking_up_trash activity and has NO role here. Left where
# the scene instance puts it (~[4.78, -0.15]) it sits squarely in the band the robot must park in
# to grasp -- the first run that got the standoff right then failed with
# "parked in collision with ['trash_can_596']". It is a movable object, so the static walkway mask
# cannot know about it. Park it far away instead of teaching every downstream sampler to dodge a
# prop that is not part of the task.
_tc = env.scene.object_registry("name", "trash_can_596")
if _tc is not None:
    _far = np.asarray(_ws.candidates(), dtype=float)
    _fd = np.linalg.norm(_far - np.array(CAN_START[:2])[None], axis=1)
    _park = _far[int(np.argmax(_fd))]
    _tz = float(arr(_tc.get_position_orientation()[0])[2])
    _tc.set_position_orientation(position=th.tensor([float(_park[0]), float(_park[1]), _tz],
                                                    dtype=th.float32))
    _tc.keep_still()
    for _ in range(20): og.sim.step()
    log("parked trash_can_596 at %s (%.2f m from the can)"
        % (np.round(_park, 2).tolist(), float(np.max(_fd))))
# nudge the can to the counter's aisle-side edge (in reach of the standoff)
can.set_position_orientation(position=th.tensor(CAN_START, dtype=th.float32))
can.keep_still()
for _ in range(30): og.sim.step()
cpos = arr(can.get_position_orientation()[0])
clo, chi = (arr(a) for a in can.aabb)
log("can settled pos=%s aabb_mm=%s" % (np.round(cpos, 3).tolist(), np.round((chi - clo) * 1000, 1).tolist()))

# --- choose the standoff with the KineReady head ---------------------------------------------
# Two constraints have to hold at once and neither is obvious by eye. The base must physically
# fit (the walkway mask, eroded by the chassis' circumscribed radius, guarantees that), AND the
# arm must be able to reach the grasp from there. A fixed offset from the can satisfies neither
# reliably: the trash demo's (+0.35, +0.10) puts the base inside the counter slab at this can
# position, and the nearest standable pose is ~0.6 m out, well beyond the 0.36 m that offset
# assumes.
#
# So: enumerate standable poses near the can, build the grasp each one implies (the approach
# tilts along can->base, so the target rotates with the standoff), and score them with the
# learned IK head -- the same model that agreed with exact IK on 25 of 25 top-ranked poses.
from kineready.reward import ReadinessReward

# A SEPARATE, less conservative erosion for the standoff. The 0.46 m spawn clearance is the
# chassis' circumscribed radius, which guarantees the footprint fits at ANY yaw -- correct for a
# spawn whose heading is arbitrary, but it holds the base 0.13 m further from the counter than
# necessary for a robot that is deliberately facing it. At 0.59 m the DLS tracker plateaus 0.25 m
# short of a near-top-down grasp: a local method cannot find the elbow configuration, even though
# a global IK solver says one exists. The proven recipe works at ~0.4 m. Half the chassis DEPTH
# is the right clearance when the heading is known, and the parked pose is collision-checked
# against the live sim afterwards, so this is verified rather than assumed.
_ws_close = WalkwaySampler(_sd, _md,
                           clearance_m=float(os.environ.get("JC_STANDOFF_CLEARANCE", "0.34")))
_cand = np.asarray(_ws_close.candidates(), dtype=float)
_d = np.linalg.norm(_cand - cpos[:2][None], axis=1)
_lo = float(os.environ.get("JC_STANDOFF_MIN", "0.36"))
_hi = float(os.environ.get("JC_STANDOFF_MAX", "0.60"))
_ok = _cand[(_d >= _lo) & (_d <= _hi)]
if len(_ok) == 0:
    raise SystemExit("no standable pose %.2f-%.2f m from the can at %s -- pick a can cell from "
                     "can_region.json; this one cannot be served"
                     % (_lo, _hi, np.round(cpos[:2], 2).tolist()))

# The gripper's slide axis in the EEF frame is a fixed property of the hand, so reading it once
# at the current pose is enough to build the target orientation for any candidate standoff.
def _linkpos0(nm): return arr(robot.links[nm].get_position_orientation()[0])
_epos, _equat = robot.get_eef_pose(arm)
_R_eef = arr(T.quat2mat(_equat))
_slide_w = _linkpos0("hande_right_finger") - _linkpos0("hande_left_finger")
_slide_eef = _R_eef.T @ (_slide_w / (np.linalg.norm(_slide_w) + 1e-9))
_bar0 = cpos

def _grasp_target_for(base_xy):
    """World grasp pose (4x4) implied by standing at base_xy -- same math as the GRASP block."""
    outw = np.array(base_xy, float) - _bar0[:2]
    outw /= (np.linalg.norm(outw) + 1e-9)
    into = -np.array([outw[0], outw[1], 0.0]); into /= np.linalg.norm(into)
    t_ = math.radians(TILT_DEG)
    appr = math.sin(t_) * into + math.cos(t_) * np.array([0.0, 0.0, -1.0])
    appr /= np.linalg.norm(appr)
    R_t = build_R(np.array([0, 0, 1.0]), appr, _slide_eef, np.array([-outw[1], outw[0], 0.0]))
    Tm = np.eye(4); Tm[:3, :3] = R_t
    Tm[:3, 3] = (_bar0 - PAD_OFF * appr) + SAT_BIAS_C
    return Tm

_reward = ReadinessReward.from_checkpoint(
    os.environ.get("JC_KINEREADY", REPO + "/kineready_models/kineready.pt"))
_yaws = np.arctan2(_bar0[1] - _ok[:, 1], _bar0[0] - _ok[:, 0])
_poses = np.stack([_ok[:, 0], _ok[:, 1], _yaws], axis=1)
_scores = np.array([float(_reward.score(_poses[i:i + 1], _grasp_target_for(_ok[i]))[0])
                    for i in range(len(_ok))])
# Among the poses the model says are reachable, take the CLOSEST. Reachability is necessary but
# not sufficient here: the scripted grasp is executed by a local Jacobian tracker, which needs the
# target near the arm's natural configuration, not merely inside its envelope.
_reach = np.where(_scores >= 0.5)[0]
if len(_reach) == 0:
    raise SystemExit("no standable pose can reach the can at %s (best p_kin %.3f of %d "
                     "candidates) -- pick a different can cell from can_region.json"
                     % (np.round(cpos[:2], 2).tolist(), _scores.max(), len(_ok)))
_dr = np.linalg.norm(_ok[_reach] - _bar0[:2][None], axis=1)
_best = int(_reach[int(np.argmin(_dr))])
GRASP_STANDOFF = _ok[_best]
log("standoff %s at %.3f m, p_kin=%.3f (closest of %d reachable, %d standable candidates)"
    % (np.round(GRASP_STANDOFF, 3).tolist(),
       float(np.linalg.norm(GRASP_STANDOFF - _bar0[:2])), _scores[_best], len(_reach), len(_ok)))

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

def fine_approach(target_xy, face_xy, gripper_closed, tol=0.025, max_steps=400):
    """Null the remaining base error using the HOLONOMIC axes, then face the can.

    drive_to stops at a 0.18 m tolerance and only ever commands forward + yaw, which is fine for
    crossing a room and useless for parking. The grasp recipe tracks the replayed eef path within
    about +-3 cm, so a 0.17 m parking error puts the can outside the arm's envelope and the
    gripper closes on air -- which is exactly what the first collection did.

    The base is holonomic, so the error can be driven out sideways without turning. Commands are
    issued in the BASE frame because that is what the controller expects.
    """
    hp, hq = hold_pose()
    for n in range(max_steps):
        bp, bq = robot.get_position_orientation()
        bxy = arr(bp)[:2]; yw = yaw_of(arr(bq))
        err = np.array(target_xy, float) - bxy
        dist = float(np.linalg.norm(err))
        d = np.array(face_xy, float) - bxy
        yerr = wrap(math.atan2(d[1], d[0]) - yw)
        if dist < tol and abs(yerr) < 0.03:
            break
        # world error -> base frame
        c, sn = math.cos(-yw), math.sin(-yw)
        ex = c * err[0] - sn * err[1]
        ey = sn * err[0] + c * err[1]
        g = 1.2
        cmd = np.array([np.clip(g * ex, -0.25, 0.25),
                        np.clip(g * ey, -0.25, 0.25),
                        np.clip(1.5 * yerr, -0.4, 0.4)])
        sim_step(hp, hq, gripper_closed, cmd)
    else:
        log("FINE did not converge in %d steps -- the target is probably not standable" % max_steps)
    for _ in range(20):
        sim_step(hp, hq, gripper_closed, np.zeros(3))
    bp = arr(robot.get_position_orientation()[0])
    log("FINE done at %s (target %s) err=%.3f m"
        % (np.round(bp[:2], 3).tolist(), np.round(np.array(target_xy), 3).tolist(),
           float(np.linalg.norm(bp[:2] - np.array(target_xy, float)))))


boundaries = {}
def mark(name):
    boundaries[name] = step_count[0]
    log("BOUNDARY %s = %d" % (name, step_count[0]))

# ================= PHASES =================
# NAV 1: spawn -> grasp standoff, face the can
drive_to(GRASP_STANDOFF, gripper_closed=False, face_xy=cpos[:2])
fine_approach(GRASP_STANDOFF, cpos[:2], gripper_closed=False)
_hits = sorted(nonfloor_hits())
if _hits:
    raise SystemExit("parked in collision with %s -- the standoff clearance is too aggressive "
                     "for this can position" % _hits[:4])
log("parked clear (no non-floor contacts)")

# Pad NAV to a fixed length before marking. MoMaGen reads MP_end_step / subtask_term_step as
# GLOBAL values -- data_generator.parse_MP_end_step_local() pulls one number out of the task spec
# and applies it to whichever source demo it selected ("We only have one demo right now"). So
# every source demo has to share the same phase boundaries. Only NAV varies, with the spawn->
# standoff distance; the grasp phase is fixed at +180/+440 steps. Holding still at the standoff
# until a fixed step count makes the demos align exactly. The padding frames land inside the
# motion-planned segment, which generation replans from scratch, so they cost nothing downstream.
NAV_PAD = int(os.environ.get("JC_NAV_PAD", "200"))
if step_count[0] > NAV_PAD:
    raise SystemExit("NAV took %d steps, over JC_NAV_PAD=%d -- raise the pad, or this demo will "
                     "not share boundaries with the others" % (step_count[0], NAV_PAD))
_php, _phq = hold_pose()
while step_count[0] < NAV_PAD:
    sim_step(_php, _phq, False, np.zeros(3))
log("NAV padded to %d steps" % step_count[0])
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
SAT_BIAS = SAT_BIAS_C
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

cz = arr(can.get_position_orientation()[0])
log("can final pos=%s (start z %.3f) lifted=%+.3f" % (np.round(cz, 3).tolist(), CAN_START[2], cz[2] - CAN_START[2]))
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
    # ALL demos in the file, not just the last one. This script is meant to be run several times
    # into the same output to cover different approach geometries, and MoMaGen re-anchors the
    # contact-rich segment from whichever source it selects -- keeping only the final demo would
    # silently throw the earlier geometries away.
    keep = [d for d in demos if "datagen_info" in f["data"][d]]
    f["mask"].create_dataset("use", data=np.array([d.encode() for d in keep]))
    log("mask/use = %s" % keep)
    # DataCollectionWrapper writes data.attrs["config"] but generate_dataset reads
    # data.attrs["env_args"] (robomimic env metadata, see momagen/utils/file_utils.py and
    # robomimic/utils/env_utils.py); add it so this freshly collected + inline-annotated
    # demo is generation-ready with NO prepare_src_dataset.py pass at all. Precedent:
    # momagen/scripts/script_tidybot_source_demo_curobo.py does the same patch.
    # env_kwargs == the OG config; env_name derived from the task's own activity_name so
    # this stays correct if the template/task ever changes.
    if "config" in f["data"].attrs and "env_args" not in f["data"].attrs:
        cfg_d = json.loads(f["data"].attrs["config"])
        # env_name must name THIS task, not the donor BDDL activity: the task branches and the
        # kinematic success override key off "tidybot_grasp_can". (generate_dataset overrides
        # env_name from its own config anyway, but a demo that is wrong on its own is a trap.)
        f["data"].attrs["env_args"] = json.dumps(
            {"env_name": "tidybot_grasp_can_D0", "type": 4, "env_kwargs": cfg_d})
log("SAVED %s tagged %s" % (OUTPUT, demos[-1]))
if frames:
    import imageio
    imageio.mimsave(VIDEO, frames[::2], fps=20, macro_block_size=1)
    log("VIDEO_SAVED %s frames=%d" % (VIDEO, len(frames)))
log("SCRIPT_GRASP_DONE")
og.shutdown()
