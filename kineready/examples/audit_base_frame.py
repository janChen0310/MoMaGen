"""What frame does `is_local=True` actually mean, and does `targets_in_base_frame` produce it?

The factor sweep ruled out every solve argument: attempts, timeout, world-collision, and the
locked joint vector all leave the answer unchanged. Only `is_local` remains, which means the
target FRAME is the problem.

Why the earlier checks missed it:

  * The frame gate ran in the EMPTY scene with the robot at the origin with zero yaw. There, an
    idealized (x, y, yaw, z=0) frame and the true base-link frame coincide no matter what the
    offset is. A gate that cannot fail is not a gate -- the same mistake as a control that cannot
    see orientation.
  * The "robot moved" check compared the teacher against ITSELF under two conditions. Both used
    `targets_in_base_frame`, so a shared wrong frame is invisible to it.

This measures the frames directly, at the spawn AND after moving, and reports the discrepancy
that `targets_in_base_frame` introduces.
"""
import os

import numpy as np

REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
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
    from kineready.frames import base_pose_to_matrix, pose_to_matrix, targets_in_base_frame

    robot = env.env.robots[0]

    def report(tag):
        rp, rq = robot.get_position_orientation()
        rp = np.asarray(rp.cpu(), float); rq = np.asarray(rq.cpu(), float)
        bp, bq = robot.links["base_footprint_x"].get_position_orientation()
        bp = np.asarray(bp.cpu(), float); bq = np.asarray(bq.cpu(), float)
        jp = robot.get_joint_positions().cpu().numpy()
        names = list(robot.joints.keys())
        base_vals = {n: round(float(jp[names.index(n)]), 4)
                     for n in robot.base_joint_names if n in names}

        print("\n--- %s ---" % tag, flush=True)
        print("robot.get_position_orientation() pos=%s quat=%s" % (rp.round(4), rq.round(4)),
              flush=True)
        print("links[base_footprint_x]          pos=%s quat=%s" % (bp.round(4), bq.round(4)),
              flush=True)
        print("base joint values                %s" % base_vals, flush=True)
        print("root->baselink offset            %s m" % (bp - rp).round(4), flush=True)

        # The frame `is_local=True` targets are interpreted in, per curobo.py's own conversion.
        T_true = pose_to_matrix(bp, bq)
        # The frame `targets_in_base_frame` assumes: planar (x, y, yaw) at z = 0.
        yaw = np.arctan2(2 * (rq[3] * rq[2] + rq[0] * rq[1]), 1 - 2 * (rq[1] ** 2 + rq[2] ** 2))
        T_assumed = base_pose_to_matrix(rp[0], rp[1], yaw)
        delta = np.linalg.inv(T_true) @ T_assumed
        print("ASSUMED vs TRUE base frame: translation %s m, rotation %.4f (Frobenius)"
              % (delta[:3, 3].round(4), np.linalg.norm(delta[:3, :3] - np.eye(3))), flush=True)

        # What that misalignment does to a concrete target 1 m in front of the robot.
        T_target = pose_to_matrix(rp + np.array([1.0, 0.0, 1.0]), [0, 0, 0, 1])
        via_assumed = targets_in_base_frame(np.array([[rp[0], rp[1], yaw]]), T_target)[0, 0]
        via_true = np.linalg.inv(T_true) @ T_target
        print("target-in-base-frame error: %s m (|%.4f|)"
              % ((via_assumed[:3, 3] - via_true[:3, 3]).round(4),
                 float(np.linalg.norm(via_assumed[:3, 3] - via_true[:3, 3]))), flush=True)
        return float(np.linalg.norm(via_assumed[:3, 3] - via_true[:3, 3]))

    e_spawn = report("AT SPAWN (kitchen)")

    saved = og.sim.dump_state()
    robot.set_position_orientation(
        position=th.tensor([3.6, -0.9, 0.0], dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, float(np.sin(1.1 / 2)), float(np.cos(1.1 / 2))],
                              dtype=th.float32))
    robot.keep_still()
    for _ in range(3):
        og.sim.step()
    e_moved = report("AFTER MOVING to (3.6, -0.9, yaw=1.1)")
    og.sim.load_state(saved)

    print("\nVERDICT: target-frame error %.4f m at spawn, %.4f m after moving."
          % (e_spawn, e_moved), flush=True)
    print("An error of a few centimetres is enough to flip IK verdicts near the workspace "
          "boundary, which is exactly where base-pose selection operates.", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
