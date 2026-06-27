"""
SCRIPTED (no-teleop) TidyBot++ source demo for MoMaGen, recorded headless.

The Hand-E gripper (~50 mm) can't wrap coffee_cup_7 (71 mm) and the thin mug handle
won't pinch reliably, and the Rs_int pick scene has no other graspable object — so we
ADD a small graspable cube ("pick_cube") on the table and pick THAT (a reliable body
grasp). The cube is added to the env config, so it is saved into the demo's config and
propagates through prepare_src_dataset -> generate_dataset.

Pipeline: reuse collect_tidybot_source_demo.load_tidybot_env_config + DataCollectionWrapper;
place the base near the cube; tilted down-and-forward approach (reachable); pads onto the
cube; close; lift. Steps the sim every frame so motion is recorded. Prints geometry,
grasp result, per-phase frame indices.

Usage (headless):
    OMNIGIBSON_HEADLESS=1 OMNIGIBSON_GPU_ID=5 [JC_RENDER=1] \
    python momagen/scripts/script_tidybot_source_demo.py \
        --template momagen/datasets/processed_source_demos/r1_pick_cup.hdf5 \
        --output momagen/datasets/source_og/tidybot_pick_cup.hdf5
"""
import argparse
import math
import os

import h5py
import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataCollectionWrapper
from omnigibson.macros import gm

from momagen.scripts.collect_tidybot_source_demo import load_tidybot_env_config

LOG = "/home/ubuntu/yhu/jc_scriptdemo.log"
def log(m):
    with open(LOG, "a") as f:
        f.write(str(m) + "\n")
    print(m, flush=True)

# the pick target cube (added to the scene)
CUBE_NAME = "pick_cube"
CUBE_SIZE = 0.03                       # 30 mm edge (50 mm gripper -> generous clearance)
CUBE_POS = [1.02, -0.05, 0.85]         # dropped onto the breakfast table near the cup area

# grasp tunables
BASE_DIST   = 0.40     # base standoff from the cube along outward
TILT_DEG    = 40.0     # the sweep showed 40deg is reachable + IK-stable (50-80 diverge)
PAD_OFF     = 0.05     # origins-at-cube targeting (reliable via linkpos; .aabb gap-center is
                       # stale in the extended pose). 40deg (more vertical) ejects less than 50.
PRE_DIST    = 0.14     # pre-grasp standoff back along the approach axis
LIFT_DZ     = 0.18
N_REACH     = 180      # more steps so the DLS-IK fully converges (the sweep needed ~200)
N_APPROACH  = 180
N_GRASP     = 45
N_LIFT      = 55


def build_R(a1, b1, a2, b2):
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
    # ADD the graspable cube target
    cfg["objects"] = [{
        "type": "PrimitiveObject",
        "name": CUBE_NAME,
        "primitive_type": "Cube",
        "rgba": [0.1, 0.35, 0.9, 1.0],
        "scale": [0.05, 0.03, 0.06],   # 50(x)deep x 30(y)graspable x 60(z)tall mm: narrow in
                                        # the slide axis (y), tall so it can't be ejected upward
        "mass": 0.4,
        "position": CUBE_POS,
    }]
    gm.ENABLE_TRANSITION_RULES = False
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    env = og.Environment(configs=cfg)
    env = DataCollectionWrapper(env=env, output_path=args.output, only_successes=False)
    robot = env.robots[0]
    arm = robot.default_arm
    env.reset()
    for _ in range(40):                 # let the cube settle onto the table
        og.sim.step()

    def arr(x): return np.array(x.cpu() if hasattr(x, "cpu") else x, float)
    def linkpos(name): return arr(robot.links[name].get_position_orientation()[0])

    cube = env.scene.object_registry("name", CUBE_NAME)
    cpos = arr(cube.get_position_orientation()[0])
    lo, hi = (arr(a) for a in cube.aabb)
    log(f"cube settled pos={np.round(cpos,4).tolist()} aabb_size_mm={np.round((hi-lo)*1000,1).tolist()}")
    base_now = arr(robot.get_position_orientation()[0])
    log(f"robot spawn={np.round(base_now,3).tolist()}  robot<->cube dist={float(np.linalg.norm(cpos[:2]-base_now[:2])):.3f}")

    # grasp target = cube center; approach from the robot's side (outward = cube->robot)
    bar = cpos.copy()
    outward = (base_now[:2] - cpos[:2]); outward = outward / (np.linalg.norm(outward) + 1e-9)

    # place base on the outward side, facing the cube
    base_xy = cpos[:2] + outward * BASE_DIST
    base_pos = th.tensor([base_xy[0], base_xy[1], float(base_now[2])], dtype=th.float32)
    yaw = math.atan2(-outward[1], -outward[0])
    base_quat = T.euler2quat(th.tensor([0.0, 0.0, yaw], dtype=th.float32))
    robot.set_position_orientation(position=base_pos, orientation=base_quat)
    for _ in range(20):
        og.sim.step()
    log(f"placed base at {np.round(arr(robot.get_position_orientation()[0]),3).tolist()} yaw={math.degrees(yaw):.1f}deg  eef={np.round(linkpos('eef_link'),3).tolist()}")

    # tilted approach orientation
    epos, equat = robot.get_eef_pose(arm)
    R_eef = arr(T.quat2mat(equat))
    lf = linkpos("hande_left_finger"); rf = linkpos("hande_right_finger")
    slide_world = rf - lf; slide_world = slide_world / (np.linalg.norm(slide_world) + 1e-9)
    slide_eef = R_eef.T @ slide_world
    into = -np.array([outward[0], outward[1], 0.0]); into /= (np.linalg.norm(into) + 1e-9)
    t = math.radians(TILT_DEG)
    approach_world = math.sin(t) * into + math.cos(t) * np.array([0.0, 0.0, -1.0])
    approach_world /= np.linalg.norm(approach_world)
    target_slide = np.array([-outward[1], outward[0], 0.0])
    R_target = build_R(np.array([0, 0, 1.0]), approach_world, slide_eef, target_slide)
    grasp_quat_w = T.mat2quat(th.tensor(R_target, dtype=th.float32))
    # rough pre-grasp up-and-back along the approach; exact grasp target is MEASURED below
    pre_pos_w = th.tensor(bar - 0.20 * approach_world, dtype=th.float32)
    grasp_pos_w = th.tensor(bar - 0.078 * approach_world, dtype=th.float32)  # placeholder
    lift_pos_w = grasp_pos_w.clone(); lift_pos_w[2] += LIFT_DZ
    log(f"approach={np.round(approach_world,2).tolist()}")

    RENDER = bool(os.environ.get("JC_RENDER"))
    _frames = []
    if RENDER:
        import imageio, omni.replicator.core as rep
        from pxr import UsdGeom, Gf
        stg = og.sim.stage
        cam = UsdGeom.Camera.Define(stg, "/World/jc_grip_cam"); cam.GetFocalLengthAttr().Set(28.0)
        cam.GetHorizontalApertureAttr().Set(20.955); cam.GetClippingRangeAttr().Set((0.01, 100.0))
        eye = bar + np.array([-0.18, -0.55, 0.34]); tgt = bar + np.array([0, 0, 0.06])
        up = np.array([0, 0, 1.0]); f = tgt - eye; f /= np.linalg.norm(f)
        rr = np.cross(f, up); rr /= np.linalg.norm(rr); uu = np.cross(rr, f)
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

    def compute_action(target_pos_robot, target_quat_robot, gripper_closed):
        cd = robot.get_control_dict()
        ac = robot.controllers[f"arm_{arm}"]; dof = ac.dof_idx
        q = th.as_tensor(arr(cd["joint_position"]), dtype=th.float32)[dof]
        J = th.as_tensor(arr(cd[f"eef_{arm}_jacobian_relative"]), dtype=th.float32)[:, dof]
        pos_rel, quat_rel = robot.get_relative_eef_pose(arm)
        dpos = th.as_tensor(target_pos_robot, dtype=th.float32) - th.as_tensor(arr(pos_rel), dtype=th.float32)
        dori = T.orientation_error(T.quat2mat(th.as_tensor(target_quat_robot, dtype=th.float32)),
                                   T.quat2mat(th.as_tensor(arr(quat_rel), dtype=th.float32)))
        err = th.cat([dpos, dori])
        JT = J.T
        dq = JT @ th.linalg.solve(J @ JT + 1e-4 * th.eye(6), err)
        target_q = q + th.clamp(dq, -0.05, 0.05)
        action = th.zeros(robot.action_dim)
        action[robot.controller_action_idx[f"arm_{arm}"]] = ac._reverse_preprocess_command(target_q)
        action[robot.controller_action_idx["base"]] = th.zeros(3)
        action[robot.controller_action_idx[f"gripper_{arm}"]] = -1.0 if gripper_closed else 1.0
        return action

    def goto(target_pos_w, target_quat_w, nsteps, gripper_closed):
        for _ in range(nsteps):
            rp, rq = robot.get_position_orientation()
            tp, tq = T.relative_pose_transform(target_pos_w, target_quat_w, rp, rq)
            env.step(compute_action(tp, tq, gripper_closed)); _grab()

    frames = {}; n = 0
    goto(pre_pos_w, grasp_quat_w, N_REACH, False); n += N_REACH; frames["end_reach(MP_end)"] = n
    # MEASURE the open-gripper finger-gap center (gc) vs the eef at the grasp orientation,
    # then target so gc lands on the box center (this is exactly what gripped in the pinned test).
    def aabb_center(n_):
        lo, hi = robot.links[n_].aabb
        return (arr(lo) + arr(hi)) / 2.0
    grasp_pos_w = th.tensor(bar - PAD_OFF * approach_world, dtype=th.float32)  # origins -> cube
    lift_pos_w = grasp_pos_w.clone(); lift_pos_w[2] += LIFT_DZ
    log(f"grasp_eef={np.round(arr(grasp_pos_w),4).tolist()}")
    goto(grasp_pos_w, grasp_quat_w, N_APPROACH, False); n += N_APPROACH; frames["end_approach"] = n
    mid = (linkpos("hande_left_finger") + linkpos("hande_right_finger")) / 2.0
    log(f"@pre-close finger_mid={np.round(mid,4).tolist()} cube={np.round(bar,4).tolist()} mid-cube={np.round(mid-bar,4).tolist()}")
    goto(grasp_pos_w, grasp_quat_w, N_GRASP, True); n += N_GRASP; frames["end_grasp"] = n
    gidx = np.array(arr(robot.gripper_control_idx[arm]), dtype=int)
    sep = float(np.linalg.norm(linkpos("hande_left_finger") - linkpos("hande_right_finger")))
    log(f"@post-grasp is_grasping={robot.is_grasping(arm)} gripper_qpos={np.round(arr(robot.get_joint_positions())[gidx],4).tolist()} finger_sep={sep:.4f} cube_z={float(arr(cube.get_position_orientation()[0])[2]):.3f}")
    goto(lift_pos_w, grasp_quat_w, N_LIFT, True); n += N_LIFT; frames["end_lift(total)"] = n

    cube_z_after = float(arr(cube.get_position_orientation()[0])[2])
    log(f"PHASE FRAMES: {frames}")
    log(f"cube z: start={cpos[2]:.3f} after-lift={cube_z_after:.3f}  LIFT_DELTA={cube_z_after-cpos[2]:+.3f}")
    log(f"is_grasping={robot.is_grasping(arm)}")

    if RENDER and _frames:
        import imageio
        imageio.mimwrite("/home/ubuntu/yhu/jc_scriptdemo_grasp.mp4", _frames, fps=30, quality=8, macro_block_size=8)
        log(f"wrote grasp mp4 ({len(_frames)} frames)")

    env.save_data()
    with h5py.File(args.output, "r+") as f:
        demos = sorted(f["data"].keys())
        if "mask" not in f:
            f.create_group("mask")
        if "use" in f["mask"]:
            del f["mask"]["use"]
        f["mask"].create_dataset("use", data=np.array([demos[-1].encode()]))
    log(f"SAVED {args.output}; tagged {demos[-1]} 'use'. DONE")
    og.shutdown()


if __name__ == "__main__":
    main()
