"""Which exact-IK verdict is correct: base_pose_metric's, or the teacher's?

They disagree sharply. On the trash grasp, `base_pose_metric._solve_ik` calls every sampled base
pose feasible, while `kineready.teacher` (whose base frame passed the 1e-7 m frame gate) calls
~22% feasible. The learned model agrees with the teacher at AUROC 0.9996. One of the two exact
solvers is wrong, and "the exact solver" is exactly the thing everything else is measured against.

The suspicion is a frame inconsistency in `_solve_ik`: it passes `is_local=False`, so the wrapper
converts the world target using the robot's **actual current pose** (curobo.py ~line 710), while
`initial_joint_pos` simultaneously locks the base joints at the **candidate** pose. Those are the
same only when the robot is really standing at the candidate — and in a sweep it never is.

This script settles it with GROUND TRUTH: physically move the base to each candidate, let physics
settle, and solve IK with the robot genuinely there. Under those conditions every convention
agrees, so whichever predictor matches the moved-robot answer is the correct one.
"""
import argparse
import os

import numpy as np

REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    from robomimic.utils.file_utils import get_env_metadata_from_dataset
    import momagen.utils.robomimic_utils as RobomimicUtils
    from momagen.utils.robot_config import configure_tidybot_env_meta

    src = os.path.join(REPO, "momagen/datasets/processed_source_demos/tidybot_picking_up_trash.hdf5")
    env_meta = configure_tidybot_env_meta(get_env_metadata_from_dataset(dataset_path=src))
    env = RobomimicUtils.create_env(
        env_meta=env_meta, env_class=None, env_name="tidybot_picking_up_trash_task_D0",
        robot=None, gripper=None, camera_names=[], camera_height=84, camera_width=84,
        render=False, render_offscreen=True, use_image_obs=False, use_depth_obs=False,
        manipulation_only=False, real_robot_mode=False, baseline=None)
    print("ENV_READY", flush=True)

    import torch as th
    from omnigibson.action_primitives.curobo import (CuRoboEmbodimentSelection,
                                                     CuRoboMotionGenerator)
    from base_pose_metric.metric import BasePoseMetric
    from base_pose_metric.examples.render_pose_grid import SODA, grasp_pose_from_source
    from kineready.frames import pose_to_matrix, targets_in_base_frame
    from kineready.teacher import IKTeacher

    robot = env.env.robots[0]
    scene = env.env.scene
    target = scene.object_registry("name", SODA)

    # ONE motion generator shared by both consumers: the kitchen already holds ~16 GB of VRAM and
    # two generators will not fit beside it.
    # batch_size=4: the kitchen scene leaves only ~4 GB beside Isaac, and the constructor's
    # trajopt warmup is the peak. Throughput is irrelevant here -- this is a dozen solves.
    mg = CuRoboMotionGenerator(
        robot=robot, batch_size=4, use_cuda_graph=False,
        embodiment_types=[CuRoboEmbodimentSelection.ARM_NO_TORSO,
                          CuRoboEmbodimentSelection.DEFAULT],
        scene_model=str(getattr(scene, "scene_model", "empty")).lower())

    metric = BasePoseMetric(robot, motion_generator=mg, distance_band=(0.30, 0.75), verbose=False)
    teacher = IKTeacher(robot, motion_generator=mg, batch_size=4)
    ok, controls = teacher.run_controls(n_fk=32, rng=np.random.default_rng(3))
    if not ok:
        raise RuntimeError("teacher controls failed: %s" % controls)

    eef_pose = grasp_pose_from_source(src, target, SODA)
    T_world = pose_to_matrix(np.asarray(eef_pose[0]), np.asarray(eef_pose[1]))
    print("grasp target (world): %s" % np.round(T_world[:3, 3], 4), flush=True)

    lo, hi = target.aabb
    centre = 0.5 * (np.asarray(lo.cpu() if hasattr(lo, "cpu") else lo, float)
                    + np.asarray(hi.cpu() if hasattr(hi, "cpu") else hi, float))

    rng = np.random.default_rng(args.seed)
    radii = np.linspace(0.35, 1.35, args.n)
    poses = []
    for r in radii:
        a = rng.uniform(-np.pi, np.pi)
        xy = centre[:2] + r * np.array([np.cos(a), np.sin(a)])
        yaw = np.arctan2(centre[1] - xy[1], centre[0] - xy[0])
        poses.append([xy[0], xy[1], yaw])
    poses = np.array(poses)

    # ---- predictor A: base_pose_metric's exact solve, robot NOT moved ------------------------
    base_q = robot.get_joint_positions().clone()
    a_verdicts = []
    for (x, y, yaw) in poses:
        q = base_q.clone()
        q[metric._slot["x"]] = float(x)
        q[metric._slot["y"]] = float(y)
        q[metric._slot["yaw"]] = float(yaw)
        a_verdicts.append(metric._solve_ik(q, eef_pose) is not None)
    a_verdicts = np.array(a_verdicts)

    # ---- predictor B: the teacher, robot NOT moved -------------------------------------------
    local = targets_in_base_frame(poses, T_world)[:, 0]
    b_verdicts = teacher.label(local)

    # ---- GROUND TRUTH: actually stand there and solve ----------------------------------------
    saved = og.sim.dump_state()
    gt = []
    for (x, y, yaw) in poses:
        og.sim.load_state(saved)
        robot.set_position_orientation(
            position=th.tensor([float(x), float(y), 0.0], dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))],
                                  dtype=th.float32))
        robot.keep_still()
        for _ in range(3):
            og.sim.step()
        # Robot genuinely at the candidate, so every frame convention coincides here.
        gt.append(metric._solve_ik(robot.get_joint_positions().clone(), eef_pose) is not None)
    og.sim.load_state(saved)
    gt = np.array(gt)

    print("\n%-8s %-9s %-11s %-9s %-9s" % ("radius", "ground", "metric._solve_ik", "teacher", ""),
          flush=True)
    for r, g, a, b in zip(radii, gt, a_verdicts, b_verdicts):
        flag = ""
        if a != g:
            flag += " metric-WRONG"
        if b != g:
            flag += " teacher-WRONG"
        print("%-8.2f %-9s %-17s %-9s%s" % (r, g, a, b, flag), flush=True)

    print("\nagreement with ground truth: base_pose_metric %.0f%%  |  kineready teacher %.0f%%"
          % (100 * (a_verdicts == gt).mean(), 100 * (b_verdicts == gt).mean()), flush=True)
    print("ground-truth feasible: %d/%d" % (int(gt.sum()), len(gt)), flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
