"""Verify the head camera landed and that adding it broke nothing.

The TidyBot USD is shared with the generation fleet and with recorded demos, so an additive edit has
to be proven inert before it is trusted. Four checks:

  DOF        a fixed joint must add no degrees of freedom, or every recorded action vector shifts.
  STATE      EntityPrim._dump_state serializes is_asleep + root_link + joint_pos + joint_vel and
             NOTHING per-link, so the serialized layout cannot change while n_joints holds. Do not
             "verify" this by comparing against a source demo's recorded state_size -- that demo was
             recorded in a different scene and the lengths legitimately differ. That comparison
             produced a false alarm once already.
  CAMERAS    three VisionSensors, with the original two unmoved.
  VISIBILITY the point of the exercise: the can should now be visible from a range of base poses.

Run:  OMNIGIBSON_HEADLESS=1 CUDA_VISIBLE_DEVICES=N python momagen/scripts/verify_head_camera.py
"""
import os

import numpy as np

REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXPECTED = {"base_camera_link": (0.315, 45.0), "head_camera_link": (1.269, 30.0),
            "arm_camera_link": (0.753, 90.0)}


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

    from omnigibson.sensors.vision_sensor import VisionSensor
    from base_pose_metric.metric import BasePoseMetric
    from base_pose_metric.geometry import aabb_sample_points, base_pose_to_matrix, visible_fraction

    robot = env.env.robots[0]
    can = env.env.scene.object_registry("name", "can_of_soda_595")
    ok = True

    n_j, n_q = len(robot.joints), len(robot.get_joint_positions())
    print("\n[1] joints=%d qpos=%d (expect 12/12) -- a fixed joint adds no DOF" % (n_j, n_q))
    ok &= (n_j == 12 and n_q == 12)

    print("[2] head_camera_link present: %s" % ("head_camera_link" in robot.links))
    ok &= "head_camera_link" in robot.links

    print("[3] dump_state len=%d, n_joints unchanged -> layout invariant"
          % len(og.sim.dump_state(serialized=True)))

    metric = BasePoseMetric(robot, distance_band=(0.30, 0.75), verbose=False)
    n_vs = len([s for s in robot.sensors.values() if isinstance(s, VisionSensor)])
    print("\n[4] cameras: %d (expect 3)" % n_vs)
    ok &= (n_vs == 3)
    for name, c in metric._cameras.items():
        short = name.split(":")[-3]
        m = c["mount"]
        fwd = -m[:3, 2]
        pitch = -np.degrees(np.arcsin(fwd[2] / np.linalg.norm(fwd)))
        exp = EXPECTED.get(short)
        good = exp is not None and abs(m[2, 3] - exp[0]) < 0.02 and abs(pitch - exp[1]) < 2.0
        print("     %-20s z=%.3f pitch=%.1f  %s" % (short, m[2, 3], pitch, "ok" if good else "UNEXPECTED"))
        ok &= good

    lo, hi = can.aabb
    lo = np.asarray(lo.cpu() if hasattr(lo, "cpu") else lo, float)
    hi = np.asarray(hi.cpu() if hasattr(hi, "cpu") else hi, float)
    pts = aabb_sample_points(lo, hi)
    centre = pts[-1]
    hits = 0
    print("\n[5] visibility with the robot facing the can")
    for r in np.arange(0.4, 1.75, 0.15):
        xy = centre[:2] + r * np.array([np.cos(0.6), np.sin(0.6)])
        yaw = np.arctan2(centre[1] - xy[1], centre[0] - xy[0])
        T_base = base_pose_to_matrix(xy[0], xy[1], yaw)
        best = max(visible_fraction(c["K"], T_base @ c["mount"], pts, c["width"], c["height"])
                   for c in metric._cameras.values())
        hits += best > 0
        print("     r=%.2f  vis[any]=%.2f" % (r, best))
    print("     radii with vis>0: %d of 9" % hits)
    ok &= hits > 0

    print("\n=== %s ===" % ("ALL CHECKS PASSED" if ok else "SOMETHING FAILED"))
    og.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
