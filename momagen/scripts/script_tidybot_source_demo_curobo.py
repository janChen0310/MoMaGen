"""
CuRobo-driven scripted TidyBot++ source demo for MoMaGen, recorded headless.

This replaces the hand-rolled damped-least-squares IK (script_tidybot_source_demo.py)
with the SAME CuRobo motion-planning machinery generate_dataset uses to execute grasps.

Why this is robust where the DLS-IK was not:
  The arm controller is an absolute-position JointController. CuRobo solves the FULL
  joint configuration for a target eef pose in one shot; robot.q_to_action(q) turns that
  into an action and the JointController drives the arm there. There is NO per-step
  Jacobian feedback loop reading the (lagged, in the extended pose) eef/finger state, so
  the stale-read fragility that made the open-loop DLS grasp untunable is gone.

Phases (each = CuRobo plan -> execute the joint trajectory as actions, recorded):
  REACH   - MP to a pre-grasp pose backed off along the approach axis.   gripper OPEN
  DESCEND - MP to the grasp pose (cube ignored as obstacle so fingers can straddle it). OPEN
  CLOSE   - hold arm joints, gripper CLOSE (-1.0) for N steps (physical grasp catches)
  LIFT    - MP straight up (cube ignored), gripper held CLOSED

Grasp geometry: measure the finger-gap-center offset in the eef frame ONCE at rest (a
fixed gripper property), then place eef target poses so the gap-center lands on the cube
center. A small set of approach tilts is tried by PLANNING-ONLY feasibility (no execution)
and the first tilt for which both pre-grasp and grasp poses are reachable is used.

Following generation exactly (starter_semantic_action_primitives._plan_joint_motion):
  cmg.update_obstacles(ignore_objects=[cube] + _markers); compute_trajectories(skip_obstacle_update=True,
  emb_sel=ARM_NO_TORSO). The world collision checker is shared across embodiments
  (curobo.py:130), so updating via DEFAULT applies to ARM_NO_TORSO planning; ignoring the
  cube also avoids the geom_type=="Mesh" assert on the PrimitiveObject.

Usage (headless):
    OMNIGIBSON_HEADLESS=1 OMNIGIBSON_GPU_ID=5 [JC_RENDER=1] \
    python momagen/scripts/script_tidybot_source_demo_curobo.py \
        --template momagen/datasets/processed_source_demos/r1_pick_cup.hdf5 \
        --output momagen/datasets/source_og/tidybot_pick_cup.hdf5
"""
import argparse
import json
import math
import os
import traceback

import h5py
import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataCollectionWrapper
from omnigibson.macros import gm

from momagen.scripts.collect_tidybot_source_demo import load_tidybot_env_config

LOG = "/home/ubuntu/yhu/jc_scriptdemo_curobo.log"
def log(m):
    with open(LOG, "a") as f:
        f.write(str(m) + "\n")
    print(m, flush=True)

# the pick target cube (added to the scene)
CUBE_NAME = "pick_cube"
CUBE_POS  = [1.02, -0.05, 0.85]          # dropped onto the breakfast table near the cup area
CUBE_SCALE = [0.03, 0.03, 0.06]          # 30x30 footprint (<=42mm diagonal < 50mm gripper), 60mm tall

# grasp tunables
BASE_DIST = 0.40                         # base standoff from the cube along outward
PRE_DIST  = 0.12                         # pre-grasp standoff back along the approach axis
LIFT_DZ   = 0.18
N_GRASP   = 45                           # steps to hold the gripper closed so the grasp catches
SETTLE    = 18                           # steps to hold each reach's final config so the position
                                         # controller converges (else the arm lags ~14mm and the
                                         # gripper closes off-target)
# approach tilt from vertical (deg); tried in order, first reachable (pre+grasp) wins.
# Prefer near-vertical (top-down): the fingers descend straight around the cube and straddle it,
# rather than an angled approach that shoves the cube.
CANDIDATE_TILTS = [10.0, 20.0, 5.0, 30.0, 15.0, 40.0, 50.0]


def build_R(a1, b1, a2, b2):
    """Rotation mapping eef-frame axes (a1,a2) onto world axes (b1,b2)."""
    def n(v): return v / (np.linalg.norm(v) + 1e-9)
    a1 = n(a1); a2 = n(a2 - (a2 @ a1) * a1); a3 = np.cross(a1, a2)
    b1 = n(b1); b2 = n(b2 - (b2 @ b1) * b1); b3 = np.cross(b1, b2)
    A = np.stack([a1, a2, a3], axis=1); B = np.stack([b1, b2, b3], axis=1)
    return B @ A.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    open(LOG, "w").close()

    cfg = load_tidybot_env_config(args.template)
    cfg["scene"]["scene_file"] = os.path.join(os.getcwd(), "momagen", "scene_instances", "Rs_int",
                                              "Rs_int_task_datagen_pick_0_0_template.json")
    cfg["scene"]["scene_instance"] = None
    cfg["robots"][0]["grasping_mode"] = "physical"
    cfg["objects"] = [{
        "type": "PrimitiveObject",
        "name": CUBE_NAME,
        "primitive_type": "Cube",
        "rgba": [0.1, 0.35, 0.9, 1.0],
        "scale": CUBE_SCALE,
        "mass": 0.3,
        "position": CUBE_POS,
    }]
    # Visualization markers (visual_only -> no physics, no CuRobo collision): an axis triad at the
    # eef_link IK-TARGET frame CuRobo plans for (X=red, Y=green, Z=blue), a sphere at the ACTUAL
    # eef_link, and a sphere at the finger-GAP (grasp) point (offset +51.5mm along eef +z).
    _AXL = 0.14  # axis length (long enough to stick out past the gripper body)
    _AXW = 0.009
    # Markers ONLY when rendering -- a clean RE-RECORD (no JC_RENDER) must not add scene objects,
    # or prepare_src_dataset/generation would inherit them. (When absent, the marker helpers no-op
    # because object_registry returns None for their names.)
    if bool(os.environ.get("JC_RENDER")):
        for _m in [
            {"name": "jc_ax_x", "primitive_type": "Cube", "scale": [_AXL, _AXW, _AXW], "rgba": [1.0, 0.15, 0.15, 1.0]},
            {"name": "jc_ax_y", "primitive_type": "Cube", "scale": [_AXW, _AXL, _AXW], "rgba": [0.15, 1.0, 0.15, 1.0]},
            {"name": "jc_ax_z", "primitive_type": "Cube", "scale": [_AXW, _AXW, _AXL], "rgba": [0.2, 0.4, 1.0, 1.0]},
            {"name": "jc_eef", "primitive_type": "Sphere", "scale": [0.03, 0.03, 0.03], "rgba": [1.0, 0.4, 1.0, 1.0]},
            {"name": "jc_gap", "primitive_type": "Sphere", "scale": [0.028, 0.028, 0.028], "rgba": [1.0, 0.95, 0.1, 1.0]},
        ]:
            cfg["objects"].append({"type": "PrimitiveObject", "visual_only": True,
                                   "position": [0.0, 0.0, 2.5], **_m})
    gm.ENABLE_TRANSITION_RULES = False
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # --- build env, then the CuRobo primitive BEFORE wrapping (its warmup steps the sim;
    #     doing it pre-wrap keeps those steps out of the recorded demo) ---
    env_og = og.Environment(configs=cfg)
    robot = env_og.robots[0]
    arm = robot.default_arm

    from omnigibson.action_primitives.curobo import CuRoboMotionGenerator, CuRoboEmbodimentSelection

    scene_model = (
        env_og.scene.scene_model.lower()
        if isinstance(env_og.scene, og.scenes.interactive_traversable_scene.InteractiveTraversableScene)
        else "empty"
    )
    # Instantiate CuRobo directly (we only need the motion generator, not the full
    # StarterSemanticActionPrimitives). CuRobo MUST run on a CUDA device (its fused kernels
    # assert is_cuda); OG's physics tensor API here reports "cpu", so we use cuda:0 (the
    # generation-proven default) and run OG on GPU 0 via OMNIGIBSON_GPU_ID=0 to co-locate.
    # (CUDA_VISIBLE_DEVICES masking is avoided: it breaks Isaac's RTX/XR device discovery.)
    # CuRobo reads the robot's (CPU) joint values and moves them onto its device internally.
    cmg = CuRoboMotionGenerator(
        robot=robot,
        batch_size=int(os.environ.get("JC_CUROBO_BATCH", "6")),
        use_cuda_graph=False,
        scene_model=scene_model,
        use_eyes_targets=False,
        device="cuda:0",  # CuRobo has internal cuda:0 buffers -> run OG on GPU 0 too (co-locate)
    )
    emb_sel = CuRoboEmbodimentSelection.ARM_NO_TORSO
    eef_link = robot.eef_link_names[arm]
    log(f"PREFLIGHT eef_link={eef_link} batch_size={cmg.batch_size} "
        f"embodiments={list(cmg.mg.keys())} arm={arm}")

    env = DataCollectionWrapper(env=env_og, output_path=args.output, only_successes=False)
    env.reset()
    for _ in range(40):                  # let the cube settle onto the table (NOT recorded)
        og.sim.step()

    def arr(x): return np.array(x.cpu() if hasattr(x, "cpu") else x, float)
    def linkpos(name): return arr(robot.links[name].get_position_orientation()[0])

    # --- visualization marker handles + updaters ---
    _mk = {n: env.scene.object_registry("name", n) for n in ["jc_ax_x", "jc_ax_y", "jc_ax_z", "jc_eef", "jc_gap"]}
    _markers = [m for m in _mk.values() if m is not None]
    _AXL_OFF = {"jc_ax_x": np.array([_AXL / 2, 0, 0]), "jc_ax_y": np.array([0, _AXL / 2, 0]),
                "jc_ax_z": np.array([0, 0, _AXL / 2])}
    _IDQ = th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)

    def set_target_triad(pos_w, quat_w):
        """Place the X/Y/Z axis triad at the eef_link IK-target frame CuRobo plans for."""
        qt = th.as_tensor(np.asarray(quat_w.cpu() if hasattr(quat_w, "cpu") else quat_w), dtype=th.float32)
        R_w = arr(T.quat2mat(qt))
        for nm, off in _AXL_OFF.items():
            if _mk[nm] is None: continue
            p = np.asarray(pos_w, float) + R_w @ off
            _mk[nm].set_position_orientation(th.tensor(p, dtype=th.float32), qt)

    _HAS_HEND = "hande_end" in robot.links
    def update_actual_markers():
        """Track the ACTUAL eef_link (magenta) and the FINGERTIP grasp point hande_end (yellow)."""
        if _mk["jc_eef"] is not None:
            _mk["jc_eef"].set_position_orientation(th.tensor(linkpos(eef_link), dtype=th.float32), _IDQ)
        if _mk["jc_gap"] is not None:
            gp = linkpos("hande_end") if _HAS_HEND else \
                (linkpos("hande_left_finger") + linkpos("hande_right_finger")) / 2.0
            _mk["jc_gap"].set_position_orientation(th.tensor(gp, dtype=th.float32), _IDQ)

    cube = env.scene.object_registry("name", CUBE_NAME)
    cpos = arr(cube.get_position_orientation()[0])
    lo, hi = (arr(a) for a in cube.aabb)
    log(f"cube settled pos={np.round(cpos,4).tolist()} aabb_size_mm={np.round((hi-lo)*1000,1).tolist()}")

    # place base on the outward side of the cube, facing it (NOT recorded)
    base_now = arr(robot.get_position_orientation()[0])
    outward = (base_now[:2] - cpos[:2]); outward = outward / (np.linalg.norm(outward) + 1e-9)
    base_xy = cpos[:2] + outward * BASE_DIST
    base_pos = th.tensor([base_xy[0], base_xy[1], float(base_now[2])], dtype=th.float32)
    yaw = math.atan2(-outward[1], -outward[0])
    base_quat = T.euler2quat(th.tensor([0.0, 0.0, yaw], dtype=th.float32))
    robot.set_position_orientation(position=base_pos, orientation=base_quat)
    for _ in range(20):
        og.sim.step()
    robot.keep_still()
    log(f"placed base at {np.round(arr(robot.get_position_orientation()[0]),3).tolist()} "
        f"yaw={math.degrees(yaw):.1f}deg eef={np.round(linkpos('eef_link'),3).tolist()}")

    # fixed gripper geometry, measured once at rest (gripper open)
    epos, equat = robot.get_eef_pose(arm)
    R_eef = arr(T.quat2mat(equat))                       # world_R_eef
    lf = linkpos("hande_left_finger"); rf = linkpos("hande_right_finger")
    # GRASP POINT = "hande_end", the tool center BETWEEN THE FINGERTIPS (hande_link+0.1455). The
    # finger LINK ORIGINS (hande_left/right_finger) sit at the finger MOUNT (hande_link+0.099),
    # ~46.5mm short of the tips -> using their midpoint put the cube at the gripper base and the
    # fingertips into the table. Fall back to the mount midpoint + 46.5mm along eef +z if the link
    # isn't present.
    if "hande_end" in robot.links:
        gap_w = linkpos("hande_end")
    else:
        gap_w = (lf + rf) / 2.0 + R_eef @ np.array([0.0, 0.0, 0.0465])
    gap_off_eef = R_eef.T @ (gap_w - arr(epos))          # fingertip tool center in the eef frame (fixed)
    slide_world = rf - lf; slide_world = slide_world / (np.linalg.norm(slide_world) + 1e-9)
    slide_eef = R_eef.T @ slide_world
    log(f"grasp_pt={'hande_end' if 'hande_end' in robot.links else 'mount+46.5'} "
        f"gap_off_eef(mm)={np.round(gap_off_eef*1000,1).tolist()} slide_eef={np.round(slide_eef,3).tolist()}")

    bar = cpos.copy()                                    # grasp at the cube center

    def poses_for_tilt(tilt_deg):
        into = -np.array([outward[0], outward[1], 0.0]); into = into / (np.linalg.norm(into) + 1e-9)
        t = math.radians(tilt_deg)
        approach = math.sin(t) * into + math.cos(t) * np.array([0.0, 0.0, -1.0])
        approach = approach / np.linalg.norm(approach)
        target_slide = np.array([-outward[1], outward[0], 0.0])
        R_target = build_R(np.array([0, 0, 1.0]), approach, slide_eef, target_slide)
        quat = T.mat2quat(th.tensor(R_target, dtype=th.float32))
        grasp_pos = bar - R_target @ gap_off_eef         # gap-center -> cube center
        pre_pos = grasp_pos - PRE_DIST * approach
        lift_pos = grasp_pos.copy(); lift_pos[2] += LIFT_DZ
        return approach, quat, pre_pos, grasp_pos, lift_pos

    def _targets(pos_w, quat_w):
        bs = cmg.batch_size
        p = th.as_tensor(np.asarray(pos_w), dtype=th.float32)
        q = th.as_tensor(np.asarray(quat_w.cpu() if hasattr(quat_w, "cpu") else quat_w), dtype=th.float32)
        return {eef_link: th.stack([p for _ in range(bs)])}, {eef_link: th.stack([q for _ in range(bs)])}

    def _full_q(out):
        t = out.cpu().float() if hasattr(out, "cpu") else th.as_tensor(out, dtype=th.float32)
        return t[-1] if t.dim() == 2 else t

    def plan_mp(pos_w, quat_w):
        """Full collision-aware ARM_NO_TORSO motion plan. Returns (interp q_traj (T,D) | None, status)."""
        bs = cmg.batch_size
        tp, tq = _targets(pos_w, quat_w)
        cmg.update_obstacles(ignore_objects=[cube] + _markers)      # cube ignored (straddle + avoids geom assert)
        results, paths = cmg.compute_trajectories(
            target_pos=tp, target_quat=tq, initial_joint_pos=None, is_local=False,
            max_attempts=50, timeout=30.0, ik_fail_return=10,
            enable_finetune_trajopt=True, finetune_attempts=1, return_full_result=True,
            success_ratio=1.0 / bs, attached_obj=None, attached_obj_scale=None,
            skip_obstacle_update=True, ik_only=False, emb_sel=emb_sel,
        )
        idx = th.where(results[0].success)[0].cpu()
        if len(idx) == 0:
            return None, str(results[0].status)
        q_traj = cmg.path_to_joint_trajectory(paths[idx[0]], get_full_js=True, emb_sel=emb_sel).cpu().float()
        return cmg.add_linearly_interpolated_waypoints(traj=q_traj, max_inter_dist=0.01), "ok"

    name_to_dof = {n: i for i, n in enumerate(cmg.robot_joint_names)}  # robot joint-name -> dof index

    def ik_goal(pos_w, quat_w):
        """Collision-aware IK only. Returns the full target joint vector (D,) | None.

        Extracts joints straight from the CuRobo JointState (position + joint_names) and writes
        them into a copy of the current full DOF vector. (We can't use path_to_joint_trajectory
        with get_full_js=True on an IK result -- CuRobo re-augments the locked joints and raises
        'lock_joints is also listed in self.joint_names'.)"""
        bs = cmg.batch_size
        tp, tq = _targets(pos_w, quat_w)
        cmg.update_obstacles(ignore_objects=[cube] + _markers)
        succ, js = cmg.compute_trajectories(
            target_pos=tp, target_quat=tq, initial_joint_pos=None, is_local=False,
            max_attempts=50, timeout=20.0, ik_fail_return=10,
            enable_finetune_trajopt=False, finetune_attempts=0, return_full_result=False,
            success_ratio=1.0 / bs, skip_obstacle_update=True, ik_only=True,
            ik_world_collision_check=True, emb_sel=emb_sel,
        )
        idx = th.where(succ)[0].cpu()
        if len(idx) == 0:
            return None
        sol = js[int(idx[0])]
        pos = th.as_tensor(sol.position).detach().cpu().float().flatten()
        names = list(sol.joint_names)
        q_full = robot.get_joint_positions().clone().cpu().float()
        for jn, p in zip(names, pos):
            if jn in name_to_dof:
                q_full[name_to_dof[jn]] = float(p)
        return q_full

    RENDER = bool(os.environ.get("JC_RENDER"))
    _frames = []
    if RENDER:
        import omni.replicator.core as rep
        from pxr import UsdGeom, Gf
        stg = og.sim.stage
        cam = UsdGeom.Camera.Define(stg, "/World/jc_grip_cam"); cam.GetFocalLengthAttr().Set(28.0)
        cam.GetHorizontalApertureAttr().Set(20.955); cam.GetClippingRangeAttr().Set((0.01, 100.0))
        eye = bar + np.array([-0.13, -0.52, 0.17]); tgt = bar + np.array([0.0, 0.0, 0.05])
        up = np.array([0, 0, 1.0]); f = tgt - eye; f = f / np.linalg.norm(f)
        rr = np.cross(f, up); rr = rr / np.linalg.norm(rr); uu = np.cross(rr, f)
        M = np.eye(4); M[:3, 0] = rr; M[:3, 1] = uu; M[:3, 2] = -f; M[:3, 3] = eye
        UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*M.T.flatten().tolist()))
        _rp = rep.create.render_product("/World/jc_grip_cam", (640, 480))
        _ann = rep.AnnotatorRegistry.get_annotator("rgb"); _ann.attach([_rp])
        def _grab():
            d = _ann.get_data()
            if d is None: return
            fr = np.array(d)
            if fr.ndim == 3 and fr.shape[-1] >= 3: _frames.append(fr[:, :, :3].astype(np.uint8))
    else:
        def _grab(): return None

    gidx = robot.gripper_action_idx[arm]

    def do_step(action, gripper_closed):
        action = action.clone() if hasattr(action, "clone") else th.as_tensor(action, dtype=th.float32)
        action[gidx] = -1.0 if gripper_closed else 1.0
        env.step(action); update_actual_markers(); _grab()

    def execute_traj(q_traj, gripper_closed):
        for j_pos in q_traj:
            do_step(robot.q_to_action(j_pos), gripper_closed)
        return len(q_traj)

    def hold_close(nsteps):
        for _ in range(nsteps):
            do_step(robot.q_to_action(robot.get_joint_positions()), True)
        return nsteps

    def reach(pos_w, quat_w, gripper_closed, label):
        """Move eef to (pos_w, quat_w): collision-aware MP if available (preferred; matches
        generation's free-space arm MP), else collision-aware IK + joint interpolation
        (proven reachable for these poses). Holds the final config for SETTLE steps so the
        position controller converges (avoids the ~14mm lag that misplaced the grasp).
        Returns nsteps executed (0 if both MP and IK fail)."""
        set_target_triad(pos_w, quat_w)   # show the eef_link IK target frame for this phase
        q_traj, status = plan_mp(pos_w, quat_w)
        via = "MP"
        if q_traj is None:
            qg = ik_goal(pos_w, quat_w)
            if qg is None:
                log(f"  reach[{label}] FAILED: MP={status}, IK=none")
                return 0
            q_now = robot.get_joint_positions().cpu().float()
            q_traj = cmg.add_linearly_interpolated_waypoints(traj=th.stack([q_now, qg]), max_inter_dist=0.01)
            via = f"IK+interp (MP was {status})"
        n = execute_traj(q_traj, gripper_closed)
        q_final = q_traj[-1]
        for _ in range(SETTLE):
            do_step(robot.q_to_action(q_final), gripper_closed); n += 1
        log(f"  reach[{label}] via {via}: {len(q_traj)}+{SETTLE} steps")
        return n

    # --- choose an approach tilt by IK feasibility (pre + grasp both reachable) ---
    chosen = None
    for tilt in CANDIDATE_TILTS:
        approach, quat, pre_pos, grasp_pos, lift_pos = poses_for_tilt(tilt)
        if ik_goal(pre_pos, quat) is None:
            log(f"tilt {tilt:>4.1f}: PRE IK fail"); continue
        if ik_goal(grasp_pos, quat) is None:
            log(f"tilt {tilt:>4.1f}: PRE ok but GRASP IK fail"); continue
        chosen = (tilt, approach, quat, pre_pos, grasp_pos, lift_pos)
        log(f"tilt {tilt:>4.1f}: REACHABLE -> CHOSEN")
        break

    if chosen is None:
        log("NO reachable approach tilt found. Aborting (no demo written).")
        og.shutdown(); return

    tilt, approach, quat, pre_pos, grasp_pos, lift_pos = chosen
    log(f"approach={np.round(approach,3).tolist()} grasp_eef={np.round(grasp_pos,4).tolist()} "
        f"pre_eef={np.round(pre_pos,4).tolist()} lift_eef={np.round(lift_pos,4).tolist()}")

    frames = {}; n = 0
    # REACH -- gripper open
    n += reach(pre_pos, quat, False, "reach"); frames["end_reach(MP_end)"] = n
    mid = (linkpos("hande_left_finger") + linkpos("hande_right_finger")) / 2.0
    log(f"@post-reach finger_mid={np.round(mid,4).tolist()} eef={np.round(linkpos('eef_link'),4).tolist()}")

    # DESCEND to grasp pose -- gripper open
    n += reach(grasp_pos, quat, False, "descend"); frames["end_descend"] = n
    mid = (linkpos("hande_left_finger") + linkpos("hande_right_finger")) / 2.0
    log(f"@pre-close finger_mid={np.round(mid,4).tolist()} cube={np.round(bar,4).tolist()} "
        f"mid-cube(mm)={np.round((mid-bar)*1000,1).tolist()}")

    # CLOSE -- hold arm, close gripper
    n += hold_close(N_GRASP); frames["end_grasp"] = n
    gq_idx = np.array(arr(robot.gripper_control_idx[arm]), dtype=int)
    sep = float(np.linalg.norm(linkpos("hande_left_finger") - linkpos("hande_right_finger")))
    cz_grasp = float(arr(cube.get_position_orientation()[0])[2])
    log(f"@post-grasp is_grasping={robot.is_grasping(arm)} "
        f"gripper_qpos={np.round(arr(robot.get_joint_positions())[gq_idx],4).tolist()} "
        f"finger_sep={sep:.4f} cube_z={cz_grasp:.3f}")

    # LIFT -- gripper held closed; try decreasing heights until one is reachable
    lifted = 0
    for dz in [LIFT_DZ, 0.13, 0.09, 0.06]:
        lp = grasp_pos.copy(); lp[2] += dz
        c = reach(lp, quat, True, f"lift+{dz:.2f}")
        if c > 0:
            lifted = c; break
    if lifted == 0:
        log("LIFT failed at all heights; holding closed")
        lifted = hold_close(40)
    n += lifted; frames["end_lift(total)"] = n

    cube_z_after = float(arr(cube.get_position_orientation()[0])[2])
    log(f"PHASE FRAMES: {frames}")
    log(f"cube z: start={cpos[2]:.3f} after-lift={cube_z_after:.3f}  LIFT_DELTA={cube_z_after-cpos[2]:+.3f}")
    log(f"FINAL is_grasping={robot.is_grasping(arm)}")

    if RENDER and _frames:
        import imageio
        imageio.mimwrite("/home/ubuntu/yhu/jc_scriptdemo_curobo_grasp.mp4", _frames, fps=30, quality=8, macro_block_size=8)
        log(f"wrote grasp mp4 ({len(_frames)} frames)")

    env.save_data()
    with h5py.File(args.output, "r+") as f:
        demos = sorted(f["data"].keys())
        if "mask" not in f:
            f.create_group("mask")
        if "use" in f["mask"]:
            del f["mask"]["use"]
        f["mask"].create_dataset("use", data=np.array([demos[-1].encode()]))
        # DataCollectionWrapper writes data.attrs["config"] but generate_dataset reads
        # data.attrs["env_args"] (robomimic env metadata); add it so prepare_src_dataset ->
        # generate_dataset works without manual injection. env_kwargs == the OG config.
        if "config" in f["data"].attrs and "env_args" not in f["data"].attrs:
            cfg_d = json.loads(f["data"].attrs["config"])
            f["data"].attrs["env_args"] = json.dumps(
                {"env_name": "datagen_pick_D0", "type": 4, "env_kwargs": cfg_d})
    log(f"SAVED {args.output}; tagged {demos[-1]} 'use'. DONE")
    og.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("FATAL " + repr(e) + "\n" + traceback.format_exc())
        raise
