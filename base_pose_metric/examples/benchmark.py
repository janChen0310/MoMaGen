"""Measure what scoring a base pose actually costs, broken down by component.

The four terms differ by orders of magnitude, and the headline "seconds per pose" is misleading
without the split: visibility is numpy plus a handful of rays, collision is one batched GPU call
for the whole sweep, and IK is a per-pose solver call that cannot be batched (locked joints are
global to the call). Which of those dominates depends entirely on how many poses survive to the
IK stage, so the benchmark reports both a collision-heavy and an IK-heavy case.
"""
import argparse
import os
import time

import numpy as np
import torch as th

REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SODA = "can_of_soda_595"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,10,50", help="sweep sizes to time")
    ap.add_argument("--ignore", default="")
    ap.add_argument("--source-demo", default=None)
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
    from base_pose_metric.examples.render_pose_grid import grasp_pose_from_source

    robot, scene = env.env.robots[0], env.env.scene
    target = scene.object_registry("name", SODA)
    ignore = None
    if args.ignore:
        wants = [w.strip() for w in args.ignore.split(",") if w.strip()]
        ignore = [o for o in scene.objects if any(w in o.name for w in wants)]

    t0 = time.perf_counter()
    metric = BasePoseMetric(robot, distance_band=(0.30, 0.75), ignore_objects=ignore, verbose=False)
    construct_s = time.perf_counter() - t0
    print("\nconstruction (builds CuRoboMotionGenerator, caches mounts/intrinsics): %.2f s"
          % construct_s, flush=True)

    eef = grasp_pose_from_source(args.source_demo, target, SODA) if args.source_demo else None

    lo, hi = target.aabb
    centre = ((np.asarray(lo.cpu(), float) + np.asarray(hi.cpu(), float)) / 2.0)[:2]
    rng = np.random.default_rng(0)

    def sample(n, near):
        """near=True clusters poses where IK tends to succeed, so the IK cost is exercised."""
        out = []
        while len(out) < n:
            r = rng.uniform(0.35, 0.65) if near else rng.uniform(0.35, 1.10)
            a = rng.uniform(-np.pi, np.pi)
            xy = centre + r * np.array([np.cos(a), np.sin(a)])
            out.append([xy[0], xy[1], np.arctan2(centre[1] - xy[1], centre[0] - xy[0])])
        return np.array(out)

    # Worst cases: (a) every pose reaches IK and SUCCEEDS -- no filtering relief; (b) every pose
    # reaches IK and FAILS -- the solver exhausts its attempts before giving up, which is the
    # expensive direction and the one extrapolating from successes would understate.
    base_now = robot.get_position_orientation()[0]
    base_now = np.asarray(base_now.cpu() if hasattr(base_now, "cpu") else base_now, float)[:2]

    def cluster(n, jitter=0.04):
        """Tight cluster on the known-good spawn so every pose is collision-free AND reachable."""
        out = []
        for _ in range(n):
            xy = base_now + rng.uniform(-jitter, jitter, 2)
            out.append([xy[0], xy[1], np.arctan2(centre[1] - xy[1], centre[0] - xy[0])])
        return np.array(out)

    print("\n=== ALL poses reach IK and SUCCEED (worst realistic case) ===", flush=True)
    print("%6s %9s %9s %9s %9s" % ("n", "total_s", "per_pose", "ik_s", "ik_per_call"), flush=True)
    for n in [int(x) for x in args.sizes.split(",")]:
        poses = cluster(n)
        t0 = time.perf_counter()
        res = metric.evaluate(poses, target, eef_pose=eef)
        wall = time.perf_counter() - t0
        t = metric.last_timings
        ran = sum(1 for r in res if not r["collision_static"])
        print("%6d %9.3f %9.4f %9.4f %9.4f   (%d ran IK, %d feasible)"
              % (n, wall, wall / n, t["ik"], t["ik"] / max(ran, 1), ran,
                 sum(r["feasible"] for r in res)), flush=True)

    print("\n=== ALL poses reach IK and FAIL (unreachable target) ===", flush=True)
    far = (th.tensor([float(centre[0]) + 6.0, float(centre[1]), 1.6], dtype=th.float32),
           eef[1] if eef is not None else th.tensor([0., 0., 0., 1.], dtype=th.float32))
    print("%6s %9s %9s %9s %9s" % ("n", "total_s", "per_pose", "ik_s", "ik_per_call"), flush=True)
    for n in [int(x) for x in args.sizes.split(",")][:2]:
        poses = cluster(n)
        t0 = time.perf_counter()
        res = metric.evaluate(poses, target, eef_pose=far)
        wall = time.perf_counter() - t0
        t = metric.last_timings
        ran = sum(1 for r in res if not r["collision_static"])
        print("%6d %9.3f %9.4f %9.4f %9.4f   (%d ran IK, %d feasible)"
              % (n, wall, wall / n, t["ik"], t["ik"] / max(ran, 1), ran,
                 sum(r["feasible"] for r in res)), flush=True)

    for label, near in (("mixed (most poses collide -> IK skipped)", False),
                        ("close-in (IK actually runs)", True)):
        print("\n=== %s ===" % label, flush=True)
        print("%6s %9s %9s %9s %9s %9s %9s" %
              ("n", "total_s", "per_pose", "vis_s", "ik_s", "coll_s", "obst_s"), flush=True)
        for n in [int(x) for x in args.sizes.split(",")]:
            poses = sample(n, near)
            t0 = time.perf_counter()
            res = metric.evaluate(poses, target, eef_pose=eef)
            wall = time.perf_counter() - t0
            t = metric.last_timings
            n_ik = sum(1 for r in res if not r["collision_static"])
            print("%6d %9.3f %9.4f %9.4f %9.4f %9.4f %9.4f   (%d reached IK, %d feasible)"
                  % (n, wall, wall / n, t["visibility"], t["ik"],
                     t["collision_static_batched"] + t["collision_reach"],
                     t["update_obstacles"], n_ik, sum(r["feasible"] for r in res)), flush=True)

    og.shutdown()


if __name__ == "__main__":
    main()
