"""Sweep the counter surface: where can the can go, and how hard is each spot?

WHY THIS RUNS BEFORE ANY DEMO IS COLLECTED
-------------------------------------------
The whole point of the task is to test a base-placement reward. If the robot can grasp the can from
almost anywhere it might plausibly stop, then placement is not the deciding factor, the reward has
nothing to contribute, and the ablation returns null BY CONSTRUCTION -- after weeks of data
generation and training. So the difficulty is measured first, offline, at ~0.1 ms per pose.

OUTPUT
------
`can_region.json`: one row per grid cell with `on_surface` and `viable` (the fraction of nearby
walkway positions that can both stand there and reach the can). `_counter_sample` samples only
cells above a threshold, so every episode is solvable by construction.

The first random-sampling pass over the raw counter AABB found SIX OF TWELVE can positions with
ZERO viable grasp stations -- the can lands deep on the slab, out of arm reach from anywhere the
base can legally stand. Half of all episodes would have been unsolvable, and generation would have
burned attempts discovering that one trial at a time. Hence this sweep.

TWO QUESTIONS, and the second is the one that matters
-----------------------------------------------------
1. Can the robot grasp straight from its spawn? The spawn band is deliberately outside arm reach,
   so this should be ~0%. It is a control: a non-zero number here means the band is wrong and some
   episodes never exercise navigation at all.

2. Of the walkway positions NEAR the can, what fraction are actually viable grasp stations --
   collision-free AND kinematically able to reach a grasp? This is the real difficulty knob:
     high  (>70%)  the robot can stop almost anywhere; placement barely matters; weak ablation
     low   (<10%)  placement is nearly impossible; the policy may never learn the task at all
     middle        placement decides success, which is what we want

Scoring avoids BasePoseMetric.evaluate()'s exact-IK path, which has a known open over-reporting
bug. Collision comes from the batched checker and reachability from the KineReady head over the
K=8 grasp orbit -- the same combination rank_and_render.py uses and which agreed with exact IK on
25 of 25 top-ranked poses.
"""
import argparse
import json
import os

import numpy as np

REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kineready_models/kineready.pt")
    ap.add_argument("--spacing", type=float, default=0.10,
                    help="grid spacing over the counter surface, metres")
    ap.add_argument("--min-viable", type=float, default=0.05,
                    help="a cell counts as USABLE if at least this fraction of nearby walkway "
                         "positions can both stand there and reach the can")
    ap.add_argument("--near-lo", type=float, default=0.30)
    ap.add_argument("--near-hi", type=float, default=1.20)
    ap.add_argument("--spawn-lo", type=float, default=1.20)
    ap.add_argument("--spawn-hi", type=float, default=2.50)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-per-band", type=int, default=300,
                    help="the walkway mask is 0.01 m/px, so a band holds tens of thousands of "
                         "cells; subsample to keep the collision batch sane")
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="can_region.json")
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
    from base_pose_metric.examples.render_pose_grid import SODA, grasp_pose_from_source
    from kineready.examples.trash_task import grasp_priors, symmetry_orbit
    from kineready.frames import pose_to_matrix
    from kineready.reward import ReadinessReward
    from momagen.utils.kitchen_walkway import WalkwaySampler, default_dirs

    robot = env.env.robots[0]
    scene = env.env.scene
    can = scene.object_registry("name", SODA)
    support = scene.object_registry("name", "countertop_kelker_0")

    mg = CuRoboMotionGenerator(
        robot=robot, batch_size=4, use_cuda_graph=False,
        embodiment_types=[CuRoboEmbodimentSelection.ARM_NO_TORSO,
                          CuRoboEmbodimentSelection.DEFAULT],
        scene_model=str(getattr(scene, "scene_model", "empty")).lower())
    reward = ReadinessReward.from_checkpoint(args.model)

    # Walkway at BASE clearance, restricted to positions CuRobo can actually plan to if a
    # navigable set is available -- the mask alone only proves the footprint fits.
    scene_dir, meta_dir = default_dirs(REPO)
    sampler = WalkwaySampler(scene_dir, meta_dir,
                             clearance_m=float(os.environ.get("JC_SPAWN_CLEARANCE", "0.40")))
    navset = os.environ.get("JC_TRASH_NAVSET")
    if navset and os.path.exists(navset):
        rows = json.load(open(navset))
        sampler.restrict_to_points([r["xy"] for r in rows if r.get("plan_ok")], radius_m=0.15)
        print("[diff] restricted to navigable set: %s" % navset, flush=True)
    walkway = np.asarray(sampler.candidates(), dtype=float)
    print("[diff] walkway candidates: %d" % len(walkway), flush=True)

    joint_names = list(robot.joints.keys())
    slot = {("yaw" if n.endswith("rz_joint") else ("x" if n.endswith("x_joint") else "y")):
            joint_names.index(n) for n in robot.base_joint_names}
    base_q = robot.get_joint_positions().clone()

    def score(poses, orbit):
        """-> (collision_free, p_kin) for a batch of (x, y, yaw). Chunked: the sphere tensor for
        thousands of poses is needlessly large beside a kitchen already holding ~16 GB."""
        free = []
        for i in range(0, len(poses), args.chunk):
            sub = poses[i:i + args.chunk]
            q = th.stack([base_q.clone() for _ in range(len(sub))])
            for j, (x, y, yaw) in enumerate(sub):
                q[j, slot["x"]] = float(x); q[j, slot["y"]] = float(y); q[j, slot["yaw"]] = float(yaw)
            free.append(~np.asarray(mg.check_collisions(q, skip_obstacle_update=True).cpu(), dtype=bool).reshape(-1))
        p = np.asarray(reward.score(poses, orbit, priors=grasp_priors(args.k)), dtype=float)
        return np.concatenate(free), p

    lo, hi = support.aabb
    lo = np.asarray(lo.cpu(), float); hi = np.asarray(hi.cpu(), float)
    inset = float(os.environ.get("JC_CAN_INSET", "0.10"))
    rng = np.random.default_rng(args.seed)
    saved = og.sim.dump_state()
    z0 = float(can.get_position_orientation()[0][2])

    xs = np.arange(lo[0] + inset, hi[0] - inset + 1e-9, args.spacing)
    ys = np.arange(lo[1] + inset, hi[1] - inset + 1e-9, args.spacing)
    print("[sweep] counter AABB x[%.2f,%.2f] y[%.2f,%.2f] -> %d x %d = %d cells at %.2f m"
          % (lo[0], hi[0], lo[1], hi[1], len(xs), len(ys), len(xs) * len(ys), args.spacing),
          flush=True)

    rows = []
    for xi, x in enumerate(xs):
        for y in ys:
            og.sim.load_state(saved)
            can.set_position_orientation(th.tensor([float(x), float(y), z0]).float(),
                                         can.get_position_orientation()[1])
            can.keep_still()
            for _ in range(10):
                og.sim.step()
            p = np.asarray(can.get_position_orientation()[0].cpu(), float)
            # The counter is L-shaped, so its AABB covers an inner corner with no surface under
            # it. A cell there simply drops the can, and physics is the arbiter.
            if abs(p[2] - z0) > 0.05 or np.linalg.norm(p[:2] - np.array([x, y])) > 0.05:
                rows.append(dict(xy=[float(x), float(y)], on_surface=False, viable=0.0))
                continue

            mg.update_obstacles()
            eef_pose = grasp_pose_from_source(src, can, SODA)
            T_grasp = pose_to_matrix(np.asarray(eef_pose[0]), np.asarray(eef_pose[1]))
            orbit = symmetry_orbit(T_grasp, p, k=args.k)
            d = np.linalg.norm(walkway - p[:2][None], axis=1)

            def band(a, b):
                pts = walkway[(d >= a) & (d <= b)]
                n_all = len(pts)
                if n_all == 0:
                    return dict(n=0, free=0.0, reach=0.0, viable=0.0)
                if n_all > args.max_per_band:
                    pts = pts[rng.choice(n_all, args.max_per_band, replace=False)]
                yaw = np.arctan2(p[1] - pts[:, 1], p[0] - pts[:, 0])
                poses = np.stack([pts[:, 0], pts[:, 1], yaw], axis=1)
                free, pk = score(poses, orbit)
                return dict(n=int(n_all), free=float(free.mean()),
                            reach=float((pk >= 0.5).mean()),
                            viable=float((free & (pk >= 0.5)).mean()))

            near = band(args.near_lo, args.near_hi)
            spawn = band(args.spawn_lo, args.spawn_hi)
            rows.append(dict(xy=[float(p[0]), float(p[1])], on_surface=True,
                             viable=near["viable"], near=near, spawn=spawn))
        print("[sweep] column %d/%d done (x=%.2f)" % (xi + 1, len(xs), x), flush=True)

    og.sim.load_state(saved)

    on_surf = [r for r in rows if r["on_surface"]]
    usable = [r for r in on_surf if r["viable"] >= args.min_viable]
    nv = [r["viable"] for r in usable]
    sv = [r["spawn"]["viable"] for r in on_surf]
    summary = dict(
        cells=len(rows), on_surface=len(on_surf), usable=len(usable),
        min_viable=args.min_viable, spacing=args.spacing,
        near_band=[args.near_lo, args.near_hi], spawn_band=[args.spawn_lo, args.spawn_hi],
        usable_viable_mean=float(np.mean(nv)) if nv else None,
        usable_viable_min=float(np.min(nv)) if nv else None,
        usable_viable_max=float(np.max(nv)) if nv else None,
        spawn_directly_graspable_mean=float(np.mean(sv)) if sv else None,
    )
    print("\n=== CAN REGION ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\n%d of %d on-surface cells are usable (>= %.0f%% of nearby walkway can grasp them)"
          % (len(usable), len(on_surf), 100 * args.min_viable), flush=True)
    if nv:
        print("Q1 spawn directly graspable: %.1f%%  (want ~0 -- every episode must navigate)"
              % (100 * summary["spawn_directly_graspable_mean"]), flush=True)
        print("Q2 among USABLE cells, viable grasp stations: mean %.1f%% (min %.1f%%, max %.1f%%)"
              % (100 * summary["usable_viable_mean"], 100 * summary["usable_viable_min"],
                 100 * summary["usable_viable_max"]), flush=True)
        print("   (>70%% = placement barely matters; <10%% = may be unlearnable; middle = good)",
              flush=True)

    json.dump(dict(summary=summary, rows=rows), open(args.out, "w"), indent=2)
    print("\nwrote %s" % args.out, flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
