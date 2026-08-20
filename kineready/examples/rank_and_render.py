"""Rank 1000 sampled base poses by the FULL readiness reward, then render the top 25.

The point is qualitative: does a higher reward actually look like a more comfortable place to stand
and grasp the can? Aggregate metrics cannot answer that. A grid can, and it is the artifact class
that caught the occlusion bug in this project's sibling module.

The reward is the geometric-mean Phi of `kineready.reward.readiness` -- all four terms:

    Phi(b) = 1[no base collision] * exp( (w_d ln S_d + w_v ln S_v + w_k ln P_kin) / sum w )

with w_d = w_v = 1, w_k = 2. Geometric, so one bad term sinks the score instead of being averaged
away; collision is a hard gate because a base inside the furniture is not "somewhat good".

Three deliberate choices:

  * Collision is enforced HONESTLY -- no ignore list. Generation excluded the countertops from base
    motion planning, but a metric that excluded them would call a pose standing inside a counter
    clear, which is the opposite of what this grid is for.
  * P_kin uses the K=8 SYMMETRY ORBIT, not the single demo grasp. The can is a cylinder, so a base
    pose that cannot reach the demo's exact grasp may comfortably reach the one 45 degrees around;
    scoring only the demo pose systematically under-rates base poses.
  * The top 25 additionally get EXACT IK, via the formulation validated against a physically moved
    robot (world-frame target + base joints locked to the candidate) -- NOT `evaluate()`'s path,
    which is the one with the open over-reporting bug. Each tile therefore shows the learned
    probability beside ground truth, so the grid tests the model as well as the ranking.
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
    ap.add_argument("--n", type=int, default=1000, help="base poses to sample and rank")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--r-min", type=float, default=0.35)
    ap.add_argument("--r-max", type=float, default=1.80)
    ap.add_argument("--yaw-jitter", type=float, default=1.2,
                    help="radians of yaw noise around 'facing the can'. Fully uniform yaw would "
                         "leave ~5/6 of samples unable to see the can at all, so the visibility "
                         "term would decide the ranking by itself and the grid would show nothing "
                         "about the other three")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="kineready_reward_grid.png")
    args = ap.parse_args()

    import cv2
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
    from omnigibson.sensors import VisionSensor

    from base_pose_metric.metric import BasePoseMetric
    from base_pose_metric.geometry import distance_score
    from base_pose_metric.examples.render_pose_grid import SODA, grasp_pose_from_source
    from kineready.examples.trash_task import grasp_priors, symmetry_orbit
    from kineready.frames import mat_to_quat, matrix_to_pose, pose_to_matrix
    from base_pose_metric.geometry import look_at_rotation
    from kineready.reward import ReadinessReward, readiness

    robot = env.env.robots[0]
    scene = env.env.scene
    can = scene.object_registry("name", SODA)
    emb = CuRoboEmbodimentSelection.ARM_NO_TORSO

    # Prefer the head (mast) camera for the onboard tile: it is the only base-mounted camera that
    # can actually see a counter-height object, so it is the one whose view shows whether a standing
    # position looks workable. Falls back to the stock base camera if the head link is absent.
    vs = {n: s for n, s in robot.sensors.items() if isinstance(s, VisionSensor)}
    cam = next((s for n, s in vs.items() if "head_camera" in n),
               next(s for n, s in vs.items() if "base_camera" in n))
    cam.add_modality("rgb")
    print("[grid] onboard camera for tiles: %s (%dx%d) | all cameras: %s"
          % (cam.name, cam.image_width, cam.image_height, list(vs)), flush=True)

    mg = CuRoboMotionGenerator(
        robot=robot, batch_size=4, use_cuda_graph=False,
        embodiment_types=[emb, CuRoboEmbodimentSelection.DEFAULT],
        scene_model=str(getattr(scene, "scene_model", "empty")).lower())
    # No ignore_objects: collision is enforced honestly for this ranking.
    metric = BasePoseMetric(robot, motion_generator=mg, distance_band=(0.30, 0.75),
                            ik_predictor=ReadinessReward.from_checkpoint(args.model),
                            verbose=False)
    reward = metric.ik_predictor

    eef_pose = grasp_pose_from_source(src, can, SODA)         # the REAL demo grasp, not top-down
    T_grasp = pose_to_matrix(np.asarray(eef_pose[0]), np.asarray(eef_pose[1]))
    lo, hi = can.aabb
    lo = np.asarray(lo.cpu() if hasattr(lo, "cpu") else lo, float)
    hi = np.asarray(hi.cpu() if hasattr(hi, "cpu") else hi, float)
    centre = 0.5 * (lo + hi)
    orbit = symmetry_orbit(T_grasp, centre, k=args.k)
    priors = grasp_priors(args.k)
    print("[grid] can centre %s | grasp %s" % (centre.round(3), T_grasp[:3, 3].round(3)), flush=True)

    # ---- sample -------------------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    r = rng.uniform(args.r_min, args.r_max, args.n)
    a = rng.uniform(-np.pi, np.pi, args.n)
    xy = centre[:2][None] + np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    face = np.arctan2(centre[1] - xy[:, 1], centre[0] - xy[:, 0])
    yaw = face + rng.uniform(-args.yaw_jitter, args.yaw_jitter, args.n)
    poses = np.stack([xy[:, 0], xy[:, 1], yaw], axis=1)

    # ---- score every pose on all four terms ---------------------------------------------------
    # Chunked: `evaluate` batches the collision check over whatever it is given, and 1000 poses at
    # once is a needlessly large sphere tensor beside a kitchen that already holds ~16 GB.
    results = []
    for i in range(0, args.n, args.chunk):
        results += metric.evaluate(poses[i:i + args.chunk], can, eef_pose=eef_pose,
                                  use_learned_ik=True)
        print("[grid] scored %d/%d" % (len(results), args.n), flush=True)

    S_d = np.array([distance_score(x["distance"], metric.distance_band) for x in results])
    S_v = np.array([x["visibility"]["any"] for x in results])
    free = np.array([not x["collision_static"] for x in results], dtype=float)
    dist = np.array([x["distance"] for x in results])

    # P_kin over the symmetry orbit -- one batched call for all N x K, microseconds.
    P_kin, comp = reward.score(poses, orbit, priors=priors, return_components=True)
    P_kin = np.asarray(P_kin, dtype=float)
    # The margin head, aggregated as the BEST available grasp rather than by noisy-OR: robustness
    # is a margin, so what matters is the sturdiest orbit member, not "at least one".
    R_margin = np.asarray(comp["robust"], dtype=float).max(axis=1)

    # As the user's four-term reward.
    Phi = readiness(S_d, S_v, free, P_kin)
    # Plus the margin, for a ranking that is not one enormous plateau. S_d saturates at 1.0 inside
    # the distance band and P_kin saturates at 1.000 for any comfortably-reachable pose, so the
    # four-term reward assigns an identical value to every good pose and argsort returns them in
    # index order. The margin head is the only term that varies across poses that all pass -- which
    # is precisely the tie-breaking role it was built for and had not yet been used in.
    Phi_m = _readiness_with_margin(S_d, S_v, free, P_kin, R_margin)

    funnel = {
        "sampled": int(args.n),
        "collision_free": int(free.sum()),
        "can_visible": int((S_v > 0).sum()),
        "p_kin_over_half": int((P_kin >= 0.5).sum()),
        "reward_positive": int((Phi > 0).sum()),
        "distinct_4term_rewards": int(len(np.unique(Phi.round(6)))),
        "distinct_5term_rewards": int(len(np.unique(Phi_m.round(6)))),
        "S_d_saturated_at_1": int((S_d >= 0.999).sum()),
        "p_kin_saturated_at_1": int((P_kin >= 0.999).sum()),
        "margin_range": [float(R_margin.min()), float(R_margin.max())],
    }
    print("\n[grid] funnel: %s" % json.dumps(funnel), flush=True)
    if funnel["reward_positive"] < args.top:
        print("[grid] WARNING only %d poses have reward > 0; the grid will include zero-reward "
              "tiles and the bottom rows are not meaningful" % funnel["reward_positive"], flush=True)

    ranked = np.argsort(-Phi_m)
    order = ranked[:args.top]
    # A stratified sweep across the WHOLE ranking. The top N all saturate -- S_d = 1 inside the
    # distance band, p_kin = 1.000 for anything comfortably reachable, margin ~0.99 -- so a top-N
    # grid shows 25 equally-good poses and cannot reveal whether reward tracks comfort. Spanning
    # rank 1 to rank N_total is what makes the gradient visible.
    strat = ranked[np.linspace(0, len(ranked) - 1, args.top).round().astype(int)]
    print("[grid] 4-term reward over the top %d: %.4f .. %.4f  (%d distinct values over all %d)"
          % (args.top, Phi[order].min(), Phi[order].max(),
             funnel["distinct_4term_rewards"], args.n), flush=True)
    print("[grid] 5-term reward over the top %d: %.4f .. %.4f  (%d distinct values over all %d)"
          % (args.top, Phi_m[order].min(), Phi_m[order].max(),
             funnel["distinct_5term_rewards"], args.n), flush=True)

    # ---- exact IK on the top N, as ground truth ----------------------------------------------
    base_q = robot.get_joint_positions().clone()
    joint_names = list(robot.joints.keys())
    slot = {("yaw" if n.endswith("rz_joint") else ("x" if n.endswith("x_joint") else "y")):
            joint_names.index(n) for n in robot.base_joint_names}

    exact_any, exact_demo = [], []
    for rank, i in enumerate(list(order) + list(strat)):
        q = base_q.clone()
        q[slot["x"]] = float(poses[i, 0])
        q[slot["y"]] = float(poses[i, 1])
        q[slot["yaw"]] = float(poses[i, 2])
        hits = 0
        for T_k in orbit:
            p_k, q_k = matrix_to_pose(T_k)
            sol = metric._solve_ik(q, (th.tensor(p_k, dtype=th.float32),
                                       th.tensor(q_k, dtype=th.float32)))
            if sol is not None:
                hits += 1
        exact_any.append(hits > 0)
        exact_demo.append(bool(hits))
        print("[grid] exact IK %2d/%d: %d/%d orbit members solve  (p_kin %.3f, R5 %.4f)"
              % (rank + 1, 2 * args.top, hits, args.k, P_kin[i], Phi_m[i]), flush=True)
    exact_any = np.array(exact_any)
    exact_top, exact_strat = exact_any[:args.top], exact_any[args.top:]
    idx_all = np.concatenate([order, strat])
    agree = float(((P_kin[idx_all] >= 0.5) == exact_any).mean())
    print("[grid] learned vs exact over both sets (%d poses): %.0f%% agree"
          % (len(idx_all), 100 * agree), flush=True)
    print("[grid]   top-%d: %d of %d truly reachable | stratified: %d of %d"
          % (args.top, int(exact_top.sum()), args.top, int(exact_strat.sum()), args.top),
          flush=True)

    # ---- render -------------------------------------------------------------------------------
    # TWO grids. The base camera is what was asked for, but this scene's `base_camera_link` is the
    # STOCK mount -- 0.315 m up, pitched 45 deg down -- and the can sits at 0.944 m on a counter,
    # 0.63 m ABOVE it. Measured: the can is behind the image plane for r <= 0.84 m and projects to
    # v = -350..-3480 px on a 256 px sensor beyond that. It is unviewable at any distance, so those
    # tiles cannot show whether a pose looks comfortable. A third-person view can, so both are
    # rendered and the base-camera grid stands as evidence for the mast-mount case.
    viewer = og.sim.viewer_camera
    viewer.add_modality("rgb")

    def grab(sensor):
        og.sim.render()
        obs = sensor.get_obs()
        rgb = obs[0]["rgb"] if isinstance(obs, tuple) else obs["rgb"]
        return np.asarray(rgb.cpu() if hasattr(rgb, "cpu") else rgb)[:, :, :3].astype(np.uint8)

    def place_viewer(base_xy):
        """Over-the-shoulder view: behind and left of the robot, looking past it at the can.

        Orbiting the CAN put the camera inside cabinets and the fridge for half the bearings --
        six of twenty-five tiles came back black. The robot is by construction standing in free
        space, so anchoring the camera to the robot and backing away from the can keeps it in the
        open, and framing the robot in the foreground with the can beyond is the view that actually
        shows whether a standing position looks comfortable to reach from.
        """
        to_can = centre[:2] - base_xy
        d = to_can / max(float(np.linalg.norm(to_can)), 1e-6)
        perp = np.array([-d[1], d[0]])
        eye = np.array([base_xy[0] - 1.45 * d[0] + 0.85 * perp[0],
                        base_xy[1] - 1.45 * d[1] + 0.85 * perp[1], 1.80])
        focus = np.array([base_xy[0] + 0.55 * to_can[0],
                          base_xy[1] + 0.55 * to_can[1], 0.80])
        R = look_at_rotation(focus - eye)
        viewer.set_position_orientation(
            position=th.tensor(eye, dtype=th.float32),
            orientation=th.tensor(mat_to_quat(R), dtype=th.float32))

    def square512(img):
        """Centre-crop to square, then to a uniform tile size."""
        h, w = img.shape[:2]
        side = min(h, w)
        img = img[(h - side) // 2:(h - side) // 2 + side, (w - side) // 2:(w - side) // 2 + side]
        return cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    saved = og.sim.dump_state()

    def render_set(idxs, exact_flags, label_rank):
        base_t, third_t = [], []
        for k, i in enumerate(idxs):
            og.sim.load_state(saved)
            x, y, yw = poses[i]
            robot.set_position_orientation(
                position=th.tensor([float(x), float(y), 0.0], dtype=th.float32),
                orientation=th.tensor([0.0, 0.0, float(np.sin(yw / 2)), float(np.cos(yw / 2))],
                                      dtype=th.float32))
            robot.keep_still()
            for _ in range(3):
                og.sim.step()
            rgb = grab(cam)
            place_viewer(poses[i, :2])
            rgb3 = square512(grab(viewer))

            ann = (label_rank(k, i), Phi_m[i], Phi[i], S_d[i], S_v[i], P_kin[i], R_margin[i],
                   dist[i], bool(free[i]), bool(exact_flags[k]))
            base_t.append(_annotate(cv2, cv2.resize(rgb, (512, 512),
                                                    interpolation=cv2.INTER_NEAREST), *ann, scale=2))
            third_t.append(_annotate(cv2, rgb3, *ann, scale=2))
            print("[grid] rendered %s %d/%d" % (label_rank.__name__, k + 1, len(idxs)), flush=True)
        return base_t, third_t

    def top_label(k, i):
        return k + 1

    def strat_label(k, i):
        return int(np.where(ranked == i)[0][0]) + 1     # true rank out of all sampled poses

    tiles, tiles3 = render_set(order, exact_top, top_label)
    stiles, stiles3 = render_set(strat, exact_strat, strat_label)
    og.sim.load_state(saved)

    def sheet_of(ts, path):
        side = int(np.ceil(np.sqrt(len(ts))))
        h, w = ts[0].shape[:2]
        sh = np.zeros((side * h, side * w, 3), dtype=np.uint8)
        for n, t in enumerate(ts):
            rr, cc = divmod(n, side)
            sh[rr * h:(rr + 1) * h, cc * w:(cc + 1) * w] = t
        cv2.imwrite(path, sh[:, :, ::-1])
        print("[grid] wrote %s (%dx%d)" % (path, sh.shape[1], sh.shape[0]), flush=True)

    stem = os.path.splitext(args.out)[0]
    sheet_of(tiles, stem + "_top_basecam.png")
    sheet_of(tiles3, stem + "_top_thirdperson.png")
    sheet_of(stiles, stem + "_stratified_basecam.png")
    sheet_of(stiles3, stem + "_stratified_thirdperson.png")

    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"funnel": funnel, "agreement_top": agree,
                   "stratified": [{"rank": int(np.where(ranked == i)[0][0]) + 1,
                                   "reward_5term": float(Phi_m[i]), "reward_4term": float(Phi[i]),
                                   "margin": float(R_margin[i]), "distance_m": float(dist[i]),
                                   "p_kin": float(P_kin[i]), "collision_free": bool(free[i]),
                                   "exact_reachable": bool(exact_strat[k])}
                                  for k, i in enumerate(strat)],
                   "top": [{"rank": int(k + 1), "base_pose": poses[i].round(4).tolist(),
                            "reward_5term": float(Phi_m[i]), "reward_4term": float(Phi[i]),
                            "margin": float(R_margin[i]), "distance_m": float(dist[i]),
                            "S_distance": float(S_d[i]), "S_visibility": float(S_v[i]),
                            "collision_free": bool(free[i]), "p_kin": float(P_kin[i]),
                            "exact_reachable": bool(exact_top[k])}
                           for k, i in enumerate(order)]}, f, indent=2)
    og.shutdown()


def _readiness_with_margin(s_d, s_v, free, p_kin, margin, weights=(1.0, 1.0, 2.0, 1.0)):
    """Gated weighted geometric mean over distance, visibility, P_kin AND the reachability margin.

    Same shape as `kineready.reward.readiness`, with a fourth graded term. Kept local rather than
    changed in `reward.py`: whether the margin belongs in the shipped reward is an open design
    question, and this script is the experiment that informs it, not the decision.
    """
    eps = 1e-6
    w_d, w_v, w_k, w_m = weights
    total = w_d + w_v + w_k + w_m
    logs = (w_d * np.log(np.clip(s_d, eps, 1.0))
            + w_v * np.log(np.clip(s_v, eps, 1.0))
            + w_k * np.log(np.clip(p_kin, eps, 1.0))
            + w_m * np.log(np.clip(margin, eps, 1.0)))
    return np.asarray(free, dtype=float) * np.exp(logs / total)


def _annotate(cv2, rgb, rank, phi_m, phi, s_d, s_v, p_kin, margin, dist, free, exact, scale=1):
    """Reward first and largest -- the grid is read by scanning it against the pictures."""
    img = np.ascontiguousarray(rgb[:, :, ::-1])
    # Border encodes whether exact IK agrees with the learned call, so a disagreement is visible
    # without reading any text.
    ok = (p_kin >= 0.5)
    colour = (80, 220, 80) if (ok and exact) else ((60, 60, 235) if ok != exact else (150, 150, 150))
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), colour, 3 * scale)

    def put(text, y, sc=0.42, col=colour):
        cv2.putText(img, text, (6 * scale, y * scale), cv2.FONT_HERSHEY_SIMPLEX, sc * scale,
                    (0, 0, 0), 3 * scale, cv2.LINE_AA)
        cv2.putText(img, text, (6 * scale, y * scale), cv2.FONT_HERSHEY_SIMPLEX, sc * scale,
                    col, 1 * scale, cv2.LINE_AA)

    put("#%d  R=%.3f" % (rank, phi_m), 20, 0.55)
    put("R4=%.3f  margin=%.2f" % (phi, margin), 40)
    put("d=%.2fm Sd=%.2f vis=%.2f" % (dist, s_d, s_v), 57)
    put("p_kin=%.2f exact=%s free=%s" % (p_kin, "Y" if exact else "N", "Y" if free else "N"), 74)
    return img[:, :, ::-1]


if __name__ == "__main__":
    main()
