"""Why does the teacher call close-range poses infeasible when they are reachable?

The arbitration run (`verify_exact_ik.py`) found `base_pose_metric._solve_ik` agreeing with ground
truth 12/12 while the teacher missed BOTH true positives, at the two closest radii. The teacher
labeled the entire training set, so a systematic close-range false negative would be baked into
every downstream number.

Three candidate causes, separated here:

  FRAME   The target-in-base-frame transform is wrong for poses away from the spawn. The frame
          gate only ever tested the spawn pose, so it could not have seen this.
  SETTINGS The solve is configured more weakly than `_solve_ik` (max_attempts, timeout, and
          crucially `is_local`), and simply gives up on hard configurations.
  STOCHASTIC The solver is seeded randomly and these poses sit near the boundary.

The discriminator: move the robot to the candidate and ask the teacher again. If the teacher's
answer CHANGES when the robot physically stands there, the transform is at fault. If it stays
wrong, the transform is fine and the solve settings are.
"""
import argparse
import os

import numpy as np

REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
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

    mg = CuRoboMotionGenerator(
        robot=robot, batch_size=4, use_cuda_graph=False,
        embodiment_types=[CuRoboEmbodimentSelection.ARM_NO_TORSO,
                          CuRoboEmbodimentSelection.DEFAULT],
        scene_model=str(getattr(scene, "scene_model", "empty")).lower())
    metric = BasePoseMetric(robot, motion_generator=mg, distance_band=(0.30, 0.75), verbose=False)
    teacher = IKTeacher(robot, motion_generator=mg, batch_size=4)

    eef_pose = grasp_pose_from_source(src, target, SODA)
    T_world = pose_to_matrix(np.asarray(eef_pose[0]), np.asarray(eef_pose[1]))

    lo, hi = target.aabb
    centre = 0.5 * (np.asarray(lo.cpu() if hasattr(lo, "cpu") else lo, float)
                    + np.asarray(hi.cpu() if hasattr(hi, "cpu") else hi, float))

    # Concentrate on the close range where the disagreement lives.
    rng = np.random.default_rng(0)
    radii = np.linspace(0.30, 0.75, args.n)
    poses = []
    for r in radii:
        a = rng.uniform(-np.pi, np.pi)
        xy = centre[:2] + r * np.array([np.cos(a), np.sin(a)])
        yaw = np.arctan2(centre[1] - xy[1], centre[0] - xy[0])
        poses.append([xy[0], xy[1], yaw])
    poses = np.array(poses)

    base_q = robot.get_joint_positions().clone()
    saved = og.sim.dump_state()

    rows = []
    for (x, y, yaw), r in zip(poses, radii):
        # (1) teacher, robot NOT moved -- the labeling-time configuration
        local = targets_in_base_frame(np.array([[x, y, yaw]]), T_world)[:, 0]
        t_still = bool(teacher.label(local)[0])

        # (2) ground truth: stand there, solve with the metric's settings
        og.sim.load_state(saved)
        robot.set_position_orientation(
            position=th.tensor([float(x), float(y), 0.0], dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))],
                                  dtype=th.float32))
        robot.keep_still()
        for _ in range(3):
            og.sim.step()
        gt = metric._solve_ik(robot.get_joint_positions().clone(), eef_pose) is not None

        # (3) teacher WHILE standing there, using the robot's own measured pose. If (3) != (1) the
        # transform is the problem; if (3) == (1) != (2) the solve settings are.
        bp, bq = robot.get_position_orientation()
        bp = np.asarray(bp.cpu()); bq = np.asarray(bq.cpu())
        yaw_m = np.arctan2(2 * (bq[3] * bq[2] + bq[0] * bq[1]), 1 - 2 * (bq[1] ** 2 + bq[2] ** 2))
        local_m = targets_in_base_frame(np.array([[bp[0], bp[1], yaw_m]]), T_world)[:, 0]
        t_moved = bool(teacher.label(local_m)[0])

        # (4) the metric's own solver, robot NOT moved (predictor A from the arbitration)
        og.sim.load_state(saved)
        q = base_q.clone()
        q[metric._slot["x"]] = float(x)
        q[metric._slot["y"]] = float(y)
        q[metric._slot["yaw"]] = float(yaw)
        m_still = metric._solve_ik(q, eef_pose) is not None

        rows.append((r, gt, t_still, t_moved, m_still,
                     float(np.linalg.norm(local[0][:3, 3]))))
        print("r=%.2f  ground=%-5s teacher_still=%-5s teacher_moved=%-5s metric_still=%-5s  "
              "|target|_base=%.3f m" % rows[-1], flush=True)

    og.sim.load_state(saved)
    arr = np.array([(g, ts, tm, ms) for _, g, ts, tm, ms, _ in rows], dtype=bool)
    gt, ts, tm, ms = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    print("\nagreement with ground truth over %d poses (%d feasible):" % (len(gt), int(gt.sum())),
          flush=True)
    print("  teacher, robot still : %.0f%%" % (100 * (ts == gt).mean()), flush=True)
    print("  teacher, robot moved : %.0f%%" % (100 * (tm == gt).mean()), flush=True)
    print("  metric,  robot still : %.0f%%" % (100 * (ms == gt).mean()), flush=True)
    print("\nteacher still == teacher moved on %.0f%% of poses -> %s"
          % (100 * (ts == tm).mean(),
             "TRANSFORM is fine, look at solve settings" if (ts == tm).all()
             else "TRANSFORM differs once the robot moves"), flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
