"""Score N sampled base poses and dump EVERY component, for distribution analysis.

The grid runs only kept the 25 rows they rendered. Understanding the reward as a training signal
needs the whole population: how much mass sits at exactly zero, how graded the rest is, and which
term is responsible for each part of the shape.
"""
import argparse
import json
import os

import numpy as np

REPO = os.environ.get("MOMAGEN_REPO", "/home/ubuntu/DATA4/backup_root_home/yhu/MoMaGen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kineready_models/kineready.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="kineready_reward_distribution.npz")
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
                            ik_predictor=ReadinessReward.from_checkpoint(args.model), verbose=False)
    reward = metric.ik_predictor

    eef_pose = grasp_pose_from_source(src, can, SODA)
    T_grasp = pose_to_matrix(np.asarray(eef_pose[0]), np.asarray(eef_pose[1]))
    lo, hi = can.aabb
    lo = np.asarray(lo.cpu() if hasattr(lo, "cpu") else lo, float)
    hi = np.asarray(hi.cpu() if hasattr(hi, "cpu") else hi, float)
    centre = 0.5 * (lo + hi)
    orbit = symmetry_orbit(T_grasp, centre, k=args.k)
    priors = grasp_priors(args.k)

    rng = np.random.default_rng(args.seed)
    r = rng.uniform(0.35, 1.8, args.n)
    a = rng.uniform(-np.pi, np.pi, args.n)
    xy = centre[:2][None] + np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    face = np.arctan2(centre[1] - xy[:, 1], centre[0] - xy[:, 0])
    poses = np.stack([xy[:, 0], xy[:, 1], face + rng.uniform(-1.2, 1.2, args.n)], axis=1)

    results = []
    for i in range(0, args.n, args.chunk):
        results += metric.evaluate(poses[i:i + args.chunk], can, eef_pose=eef_pose,
                                   use_learned_ik=True)
        print("[dump] %d/%d" % (len(results), args.n), flush=True)

    cam_names = list(metric._cameras)
    S_d = np.array([distance_score(x["distance"], metric.distance_band) for x in results])
    S_v = np.array([x["visibility"]["any"] for x in results])
    per_cam = {n: np.array([x["visibility"].get(n, 0.0) for x in results]) for n in cam_names}
    free = np.array([not x["collision_static"] for x in results], dtype=float)
    dist = np.array([x["distance"] for x in results])

    P_kin, comp = reward.score(poses, orbit, priors=priors, return_components=True)
    P_kin = np.asarray(P_kin, dtype=float)
    margin = np.asarray(comp["robust"], dtype=float).max(axis=1)
    p_std = np.asarray(comp["p_std"], dtype=float).mean(axis=1)

    R4 = readiness(S_d, S_v, free, P_kin)
    eps = 1e-6
    logs = (np.log(np.clip(S_d, eps, 1)) + np.log(np.clip(S_v, eps, 1))
            + 2 * np.log(np.clip(P_kin, eps, 1)) + np.log(np.clip(margin, eps, 1)))
    R5 = free * np.exp(logs / 5.0)

    np.savez(args.out, poses=poses, distance=dist, S_d=S_d, S_v=S_v, free=free,
             p_kin=P_kin, p_std=p_std, margin=margin, R4=R4, R5=R5,
             can_centre=centre, cam_names=np.array(cam_names),
             **{("vis_" + n.split(":")[-3]): v for n, v in per_cam.items()})
    print("\nwrote %s" % args.out, flush=True)
    print(json.dumps({
        "n": int(args.n),
        "R4_zero": int((R4 == 0).sum()), "R4_nonzero": int((R4 > 0).sum()),
        "R4_median_nonzero": float(np.median(R4[R4 > 0])),
        "R4_max": float(R4.max()),
        "collision_free": int(free.sum()), "visible": int((S_v > 0).sum()),
        "p_kin_over_half": int((P_kin >= 0.5).sum()),
        "all_three_pass": int(((free > 0) & (S_v > 0) & (P_kin >= 0.5)).sum()),
    }, indent=2), flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
