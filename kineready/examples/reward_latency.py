"""End-to-end latency of the FULL readiness reward in a furnished scene.

`kineready/examples/latency.py` times the learned IK head alone, in isolation, with no simulator.
That is the right number for the surrogate but the wrong number for the reward: the other three
terms are exact and run against live scene geometry, so the total is dominated by whichever of them
is slowest -- not by the MLP.

Reported two ways, because they answer different questions:

  cold   including `update_obstacles()`, which rebuilds the collision world. Correct for the first
         evaluation after anything in the scene moves.
  warm   excluding it. Correct for scoring many candidate base poses within one decision, where
         the scene is static and the rebuild is hoisted out of the loop -- which is how a
         navigation policy would actually use this.
"""
import argparse
import json
import os
import time

import numpy as np

REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kineready_models/kineready.pt")
    ap.add_argument("--sizes", default="64,250,1000")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
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

    from omnigibson.action_primitives.curobo import (CuRoboEmbodimentSelection,
                                                     CuRoboMotionGenerator)
    from base_pose_metric.metric import BasePoseMetric
    from base_pose_metric.geometry import distance_score
    from base_pose_metric.examples.render_pose_grid import SODA, grasp_pose_from_source
    from kineready.examples.trash_task import grasp_priors, symmetry_orbit
    from kineready.frames import pose_to_matrix
    from kineready.reward import ReadinessReward, readiness

    robot = env.env.robots[0]
    scene = env.env.scene
    can = scene.object_registry("name", SODA)

    mg = CuRoboMotionGenerator(
        robot=robot, batch_size=4, use_cuda_graph=False,
        embodiment_types=[CuRoboEmbodimentSelection.ARM_NO_TORSO,
                          CuRoboEmbodimentSelection.DEFAULT],
        scene_model=str(getattr(scene, "scene_model", "empty")).lower())
    metric = BasePoseMetric(robot, motion_generator=mg, distance_band=(0.30, 0.75),
                            ik_predictor=ReadinessReward.from_checkpoint(args.model),
                            verbose=False)
    reward = metric.ik_predictor

    eef_pose = grasp_pose_from_source(src, can, SODA)
    T_grasp = pose_to_matrix(np.asarray(eef_pose[0]), np.asarray(eef_pose[1]))
    lo, hi = can.aabb
    lo = np.asarray(lo.cpu() if hasattr(lo, "cpu") else lo, float)
    hi = np.asarray(hi.cpu() if hasattr(hi, "cpu") else hi, float)
    centre = 0.5 * (lo + hi)
    orbit = symmetry_orbit(T_grasp, centre, k=args.k)
    priors = grasp_priors(args.k)

    rng = np.random.default_rng(0)
    out = {}
    for n in [int(x) for x in args.sizes.split(",")]:
        r = rng.uniform(0.35, 1.8, n)
        a = rng.uniform(-np.pi, np.pi, n)
        xy = centre[:2][None] + np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
        face = np.arctan2(centre[1] - xy[:, 1], centre[0] - xy[:, 0])
        poses = np.stack([xy[:, 0], xy[:, 1],
                          face + rng.uniform(-1.2, 1.2, n)], axis=1)

        metric.evaluate(poses[:8], can, eef_pose=eef_pose, use_learned_ik=True)   # warm up
        acc = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            results = metric.evaluate(poses, can, eef_pose=eef_pose, use_learned_ik=True)
            t_eval = time.perf_counter() - t0
            t = dict(metric.last_timings)

            t1 = time.perf_counter()
            S_d = np.array([distance_score(x["distance"], metric.distance_band) for x in results])
            S_v = np.array([x["visibility"]["any"] for x in results])
            free = np.array([not x["collision_static"] for x in results], dtype=float)
            t_gather = time.perf_counter() - t1

            t1 = time.perf_counter()
            P_kin = np.asarray(reward.score(poses, orbit, priors=priors), dtype=float)
            t_pkin = time.perf_counter() - t1

            t1 = time.perf_counter()
            readiness(S_d, S_v, free, P_kin)
            t_compose = time.perf_counter() - t1

            acc.append({"total_cold": t_eval + t_gather + t_pkin + t_compose,
                        "obstacles": t["update_obstacles"],
                        "collision": t["collision_static_batched"],
                        "visibility": t["visibility"],
                        "pkin_K1_inside_evaluate": t["ik"],
                        "pkin_K%d" % args.k: t_pkin,
                        "gather": t_gather,
                        "compose": t_compose})
        m = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        m["total_warm"] = m["total_cold"] - m["obstacles"]
        m["per_pose_warm_ms"] = 1000 * m["total_warm"] / n
        m["per_pose_cold_ms"] = 1000 * m["total_cold"] / n
        out[n] = m

        print("\n=== N = %d base poses (mean of %d) ===" % (n, args.repeats), flush=True)
        for k in ("obstacles", "collision", "visibility", "pkin_K1_inside_evaluate",
                  "pkin_K%d" % args.k, "gather", "compose"):
            share = 100 * m[k] / m["total_cold"]
            print("  %-26s %8.2f ms  (%4.1f%% of cold total)" % (k, 1000 * m[k], share), flush=True)
        print("  %-26s %8.2f ms   -> %.3f ms per pose"
              % ("TOTAL cold", 1000 * m["total_cold"], m["per_pose_cold_ms"]), flush=True)
        print("  %-26s %8.2f ms   -> %.3f ms per pose"
              % ("TOTAL warm", 1000 * m["total_warm"], m["per_pose_warm_ms"]), flush=True)

    print("\n=== summary ===", flush=True)
    print("%-8s %-14s %-14s %-16s" % ("N", "cold ms", "warm ms", "warm ms/pose"), flush=True)
    for n, m in out.items():
        print("%-8d %-14.1f %-14.1f %-16.4f"
              % (n, 1000 * m["total_cold"], 1000 * m["total_warm"], m["per_pose_warm_ms"]),
              flush=True)
    with open("kineready_reward_latency.json", "w") as f:
        json.dump({str(k): v for k, v in out.items()}, f, indent=2)
    print("\nwrote kineready_reward_latency.json", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
