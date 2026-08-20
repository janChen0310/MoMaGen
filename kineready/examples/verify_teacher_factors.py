"""Isolate WHICH difference between the two exact solvers produces the disagreement.

Established so far: the teacher's answer does not change when the robot physically moves to the
candidate, so the base-frame transform is not the cause. That leaves the arguments of the solve
call itself. `base_pose_metric._solve_ik` and `kineready.teacher.label` differ in exactly four:

    initial_joint_pos     candidate base pose   vs  the pose captured at teacher construction
    is_local              False                 vs  True
    max_attempts/timeout  5 / 10 s              vs  1 / 5 s
    ik_world_collision_check   default True     vs  False

Each variant below changes ONE of them from the teacher's setting toward the metric's, so the
column that flips the answers names the cause. Ground truth is the robot physically standing at
the candidate.
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
    from kineready.teacher import IKTeacher, _mat_to_quat_xyzw

    robot = env.env.robots[0]
    scene = env.env.scene
    target = scene.object_registry("name", SODA)
    emb = CuRoboEmbodimentSelection.ARM_NO_TORSO

    mg = CuRoboMotionGenerator(
        robot=robot, batch_size=4, use_cuda_graph=False,
        embodiment_types=[emb, CuRoboEmbodimentSelection.DEFAULT],
        scene_model=str(getattr(scene, "scene_model", "empty")).lower())
    metric = BasePoseMetric(robot, motion_generator=mg, distance_band=(0.30, 0.75), verbose=False)
    teacher = IKTeacher(robot, motion_generator=mg, batch_size=4)
    eef_link = list(robot.eef_link_names.values())[0]

    eef_pose = grasp_pose_from_source(src, target, SODA)
    T_world = pose_to_matrix(np.asarray(eef_pose[0]), np.asarray(eef_pose[1]))
    lo, hi = target.aabb
    centre = 0.5 * (np.asarray(lo.cpu() if hasattr(lo, "cpu") else lo, float)
                    + np.asarray(hi.cpu() if hasattr(hi, "cpu") else hi, float))

    rng = np.random.default_rng(0)
    radii = np.linspace(0.30, 0.75, args.n)
    poses = []
    for r in radii:
        a = rng.uniform(-np.pi, np.pi)
        xy = centre[:2] + r * np.array([np.cos(a), np.sin(a)])
        poses.append([xy[0], xy[1],
                      np.arctan2(centre[1] - xy[1], centre[0] - xy[0])])
    poses = np.array(poses)

    base_q = robot.get_joint_positions().clone()
    saved = og.sim.dump_state()

    def solve(T_local, q_init, max_attempts, timeout, world_check):
        pos = th.tensor(T_local[None, :3, 3], dtype=th.float32)
        quat = th.tensor(_mat_to_quat_xyzw(T_local[:3, :3])[None], dtype=th.float32)
        successes, _ = mg.compute_trajectories(
            target_pos={eef_link: pos}, target_quat={eef_link: quat},
            initial_joint_pos=q_init, is_local=True,
            max_attempts=max_attempts, timeout=timeout, ik_fail_return=5,
            enable_finetune_trajopt=False, finetune_attempts=0, return_full_result=False,
            success_ratio=1.0, skip_obstacle_update=True, ik_only=True,
            ik_world_collision_check=world_check, emb_sel=emb)
        return bool(np.asarray(successes.cpu(), dtype=bool).reshape(-1)[0])

    cols = ["ground", "teacher", "+attempts", "+worldcol", "+lockedq", "all3"]
    print("\n%-6s " % "r" + " ".join("%-10s" % c for c in cols), flush=True)
    tally = {c: 0 for c in cols}
    for (x, y, yaw), r in zip(poses, radii):
        T_local = targets_in_base_frame(np.array([[x, y, yaw]]), T_world)[0, 0]
        q_cand = base_q.clone()
        q_cand[metric._slot["x"]] = float(x)
        q_cand[metric._slot["y"]] = float(y)
        q_cand[metric._slot["yaw"]] = float(yaw)

        og.sim.load_state(saved)
        robot.set_position_orientation(
            position=th.tensor([float(x), float(y), 0.0], dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))],
                                  dtype=th.float32))
        robot.keep_still()
        for _ in range(3):
            og.sim.step()
        gt = metric._solve_ik(robot.get_joint_positions().clone(), eef_pose) is not None
        og.sim.load_state(saved)

        vals = {
            "ground": gt,
            "teacher":   solve(T_local, teacher._rest_q, 1, 5.0, False),
            "+attempts": solve(T_local, teacher._rest_q, 5, 10.0, False),
            "+worldcol": solve(T_local, teacher._rest_q, 1, 5.0, True),
            "+lockedq":  solve(T_local, q_cand, 1, 5.0, False),
            "all3":      solve(T_local, q_cand, 5, 10.0, True),
        }
        for c in cols:
            tally[c] += int(vals[c] == gt)
        print("%-6.2f " % r + " ".join("%-10s" % vals[c] for c in cols), flush=True)

    print("\nagreement with ground truth (%d poses):" % len(poses), flush=True)
    for c in cols:
        print("  %-10s %3.0f%%" % (c, 100 * tally[c] / len(poses)), flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
