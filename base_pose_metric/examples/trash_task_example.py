"""Worked example: score base poses for pick-trash-and-dispose.

The metric itself knows nothing about this task -- everything task-specific lives here. Swapping
in a different scene means changing the env construction and the two object names below.

Run on a machine with a working OmniGibson/Isaac:
  MOMAGEN_REPO=... OMNIGIBSON_HEADLESS=1 python base_pose_metric/examples/trash_task_example.py
"""
import argparse
import os

import numpy as np
import torch as th

REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SODA = "can_of_soda_595"
TRASH = "trash_can_596"


def grasp_pose_from_source(path, target):
    """Re-anchor the source demo's actual grasp pose onto the target's current pose.

    The invariant a demo carries is the eef pose RELATIVE to the object; re-anchoring it onto
    wherever the object is now is what MoMaGen itself does when generating. Taking the frame where
    the gripper first closes gives the grasp.
    """
    import h5py
    from scipy.spatial.transform import Rotation as R

    with h5py.File(path, "r") as f:
        dg = f["data"][sorted(f["data"].keys())[0]]["datagen_info"]
        eef, ga = dg["eef_pose"][:], dg["gripper_action"][:]
        obj = dg["object_poses"][SODA][:]
    g = ga[:, 0]
    trans = [i for i in range(1, len(g)) if g[i] != g[i - 1]]
    gf = trans[0] if trans else 0                      # first close == the grasp
    rel = np.linalg.inv(obj[gf]) @ eef[gf][0:4]

    p, q = target.get_position_orientation()
    p = np.asarray(p.cpu() if hasattr(p, "cpu") else p, float)
    q = np.asarray(q.cpu() if hasattr(q, "cpu") else q, float)
    cur = np.eye(4)
    cur[:3, :3] = R.from_quat(q).as_matrix()
    cur[:3, 3] = p
    new = cur @ rel
    return (th.tensor(new[:3, 3], dtype=th.float32),
            th.tensor(R.from_matrix(new[:3, :3]).as_quat(), dtype=th.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, default=0.25, help="sweep spacing (m)")
    ap.add_argument("--extent", type=float, default=1.0, help="sweep half-extent around the object (m)")
    ap.add_argument("--yaws", type=int, default=8, help="yaw samples per position")
    ap.add_argument("--target", choices=["soda", "trash"], default="soda")
    ap.add_argument("--top", type=int, default=15)
    # The robot grasps while parked against the counter, so with a truthfully-sized base that
    # pose is legitimately "in collision" with the counter run. Excluding the furniture the robot
    # is deliberately touching -- the same trick the grasp planner uses on its target object -- is
    # what makes the pose scoreable. Everything else in the kitchen still collides normally.
    ap.add_argument("--ignore", default="",
                    help="comma-separated object-name substrings to exclude from collision checks")
    # The metric's built-in default (hover above the AABB top, gripper straight down) does NOT
    # solve for this arm: the real grasp sits 7.5 cm above the can and is tilted ~23 deg off
    # vertical, and TidyBot's arm is tilt-constrained. Reading the actual pose out of the source
    # demo is what a real caller does -- hence eef_pose being an explicit option on evaluate().
    ap.add_argument("--source-demo", default=None,
                    help="hdf5 whose datagen_info supplies the true grasp pose (recommended)")
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

    from base_pose_metric import BasePoseMetric

    robot = env.env.robots[0]
    scene = env.env.scene
    target = scene.object_registry("name", SODA if args.target == "soda" else TRASH)
    assert target is not None, "target object not in registry"

    ignore = None
    if args.ignore:
        wants = [s.strip() for s in args.ignore.split(",") if s.strip()]
        ignore = [o for o in scene.objects if any(w in o.name for w in wants)]

    metric = BasePoseMetric(robot, distance_band=(0.30, 0.75), ignore_objects=ignore)

    eef_pose = None
    if args.source_demo:
        eef_pose = grasp_pose_from_source(args.source_demo, target)
        print("[example] using the source demo's real grasp pose", flush=True)
    else:
        print("[example] using the metric's derived top-down default -- expect IK to fail for "
              "TidyBot, whose real grasp is tilted ~23 deg (pass --source-demo)", flush=True)

    tgt_xy = np.asarray((target.aabb[0].cpu() + target.aabb[1].cpu()) / 2.0, dtype=float)[:2]
    base_now = robot.get_position_orientation()[0]
    base_now = np.asarray(base_now.cpu() if hasattr(base_now, "cpu") else base_now, float)[:2]

    # --- reference poses: the one generation actually uses, and a deliberately bad one ----------
    yaw_at = lambda p: float(np.arctan2(tgt_xy[1] - p[1], tgt_xy[0] - p[0]))
    known_good = [base_now[0], base_now[1], yaw_at(base_now)]
    into_counter = [tgt_xy[0] + 0.05, tgt_xy[1] + 0.05, yaw_at(tgt_xy + 0.05)]

    print("\n=== reference poses ===", flush=True)
    for label, pose in (("current spawn (known good)", known_good),
                        ("0.05 m from the object (inside the counter)", into_counter)):
        r = metric.evaluate([pose], target, eef_pose=eef_pose)[0]
        print("%-44s d=%.3f vis=%.2f ik=%-5s static=%-5s reach=%-5s feasible=%-5s score=%.3f"
              % (label, r["distance"], r["visibility"]["any"], r["ik_ok"],
                 r["collision_static"], r["collision_reach"], r["feasible"], r["score"]), flush=True)

    # --- sweep -------------------------------------------------------------------------------
    offs = np.arange(-args.extent, args.extent + 1e-9, args.grid)
    poses = []
    for dx in offs:
        for dy in offs:
            p = tgt_xy + np.array([dx, dy])
            if np.linalg.norm([dx, dy]) < 0.2:
                continue                      # inside the object itself
            for yaw in np.linspace(-np.pi, np.pi, args.yaws, endpoint=False):
                poses.append([p[0], p[1], yaw])
    print("\n=== sweeping %d poses around %s ===" % (len(poses), target.name), flush=True)

    res = metric.evaluate(poses, target, eef_pose=eef_pose)
    feasible = [r for r in res if r["feasible"]]
    res.sort(key=lambda r: -r["score"])

    print("feasible: %d / %d" % (len(feasible), len(res)), flush=True)
    if not feasible and metric.ik_diagnostics:
        # A sweep with zero feasible poses is ambiguous: genuinely bad geometry, or a broken IK
        # setup? Print why IK failed so the two cannot be confused.
        print("IK diagnostics (why no solution was found):", flush=True)
        for emb, why in metric.ik_diagnostics.items():
            print("   %-40s %s" % (emb, why), flush=True)
    print("\n%-26s %7s %6s %5s %7s %6s %6s" %
          ("(x, y, yaw_deg)", "dist", "vis", "ik", "static", "reach", "score"), flush=True)
    for r in res[: args.top]:
        x, y, yaw = r["base_pose"]
        print("(%6.2f,%6.2f,%7.1f) %7.3f %6.2f %5s %7s %6s %6.3f"
              % (x, y, np.degrees(yaw), r["distance"], r["visibility"]["any"],
                 r["ik_ok"], r["collision_static"], r["collision_reach"], r["score"]), flush=True)

    og.shutdown()


if __name__ == "__main__":
    main()
