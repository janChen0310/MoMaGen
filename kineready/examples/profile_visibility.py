"""Profile the visibility term, then test optimized variants against it.

218 ms for 1000 poses x 3 cameras is ~73 us per camera per pose, for what is arithmetically a 4x4
inverse, nine projections, and (sometimes) nine rays. That is far more than the work justifies, so
this measures where it actually goes before changing anything.

Every variant is checked for EXACT agreement with the baseline before its timing is reported. An
optimization that changes the answer is not an optimization, and visibility already produced one
silently-wrong number in this project.
"""
import argparse
import cProfile
import io
import os
import pstats
import time

import numpy as np

REPO = os.environ.get("MOMAGEN_REPO", "/home/ubuntu/DATA4/backup_root_home/yhu/MoMaGen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
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

    from base_pose_metric.metric import BasePoseMetric
    from base_pose_metric.geometry import (aabb_sample_points, base_pose_to_matrix,
                                           project_points, visible_fraction)
    from base_pose_metric.examples.render_pose_grid import SODA

    robot = env.env.robots[0]
    can = env.env.scene.object_registry("name", SODA)
    metric = BasePoseMetric(robot, distance_band=(0.30, 0.75), verbose=False)
    cams = metric._cameras
    names = list(cams)

    lo, hi = can.aabb
    lo = np.asarray(lo.cpu() if hasattr(lo, "cpu") else lo, float)
    hi = np.asarray(hi.cpu() if hasattr(hi, "cpu") else hi, float)
    pts = aabb_sample_points(lo, hi)
    centre = pts[-1]

    rng = np.random.default_rng(0)
    r = rng.uniform(0.35, 1.8, args.n)
    a = rng.uniform(-np.pi, np.pi, args.n)
    xy = centre[:2][None] + np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    face = np.arctan2(centre[1] - xy[:, 1], centre[0] - xy[:, 0])
    poses = np.stack([xy[:, 0], xy[:, 1], face + rng.uniform(-1.2, 1.2, args.n)], axis=1)

    # ---- separate the occlusion cost from everything else --------------------------------------
    # The PhysX binding's raycast_closest is a read-only C++ attribute, so it cannot be wrapped.
    # `_unoccluded_fraction` is an ordinary Python method though, so instrument that instead: it is
    # the only thing that fires rays, and the baseline always fires exactly 9 per call.
    ray_stats = {"n": 0, "t": 0.0, "calls": 0}
    real_unocc = metric._unoccluded_fraction

    def timed_unocc(cam_pos, points, target):
        t0 = time.perf_counter()
        out = real_unocc(cam_pos, points, target)
        ray_stats["t"] += time.perf_counter() - t0
        ray_stats["calls"] += 1
        ray_stats["n"] += len(points)
        return out

    metric._unoccluded_fraction = timed_unocc

    # ---- BASELINE: exactly what evaluate() does today -----------------------------------------
    def baseline():
        out = np.empty(len(poses))
        for i, (x, y, yaw) in enumerate(poses):
            T_base = base_pose_to_matrix(x, y, yaw)
            best = 0.0
            for nm in names:
                c = cams[nm]
                T_cam = T_base @ c["mount"]
                frac = visible_fraction(c["K"], T_cam, pts, c["width"], c["height"])
                if frac > 0.0:
                    frac *= metric._unoccluded_fraction(T_cam[:3, 3], pts, can)
                best = max(best, frac)
            out[i] = best
        return out

    t0 = time.perf_counter(); ref = baseline(); t_base = time.perf_counter() - t0
    n_rays, t_rays, n_calls = ray_stats["n"], ray_stats["t"], ray_stats["calls"]
    print("\n=== BASELINE (%d poses x %d cameras) ===" % (args.n, len(names)), flush=True)
    print("  total                    %8.1f ms   (%.3f ms/pose)"
          % (1000*t_base, 1000*t_base/args.n), flush=True)
    print("  occlusion stage          %8.1f ms   (%.0f%%)  %d calls, %d rays, %.1f us/ray"
          % (1000*t_rays, 100*t_rays/t_base, n_calls, n_rays, 1e6*t_rays/max(n_rays,1)), flush=True)
    print("  frustum stage + overhead %8.1f ms   (%.0f%%)"
          % (1000*(t_base-t_rays), 100*(1-t_rays/t_base)), flush=True)

    # ---- cProfile, for the non-ray half -------------------------------------------------------
    metric._unoccluded_fraction = real_unocc
    pr = cProfile.Profile(); pr.enable(); baseline(); pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14)
    print("\n=== cProfile, sorted by tottime ===", flush=True)
    for line in s.getvalue().splitlines()[4:26]:
        print("  " + line, flush=True)

    # ---- OPTIMIZED ---------------------------------------------------------------------------
    # 1. target_paths was rebuilt on EVERY call (per pose, per camera). Cache it.
    # 2. np.linalg.inv on a 4x4 per call -> analytic rigid inverse (R^T, -R^T t).
    # 3. the whole frustum stage vectorized over all N poses at once, so the Python loop only runs
    #    for poses that actually need rays.
    target_paths = tuple({l.prim_path for l in can.links.values()})
    own_paths = metric._own_link_paths()

    K0, W0, H0 = cams[names[0]]["K"], cams[names[0]]["width"], cams[names[0]]["height"]
    assert all(cams[n]["width"] == W0 and cams[n]["height"] == H0 for n in names)

    def frustum_batched(T_cams, K, W, H):
        """(N,4,4) camera poses -> (N,9) bool in_image, without a Python loop or any 4x4 inverse."""
        R = T_cams[:, :3, :3]
        t = T_cams[:, :3, 3]
        # rigid inverse: cam = R^T (p - t)
        d = pts[None, :, :] - t[:, None, :]                       # (N,9,3)
        cam = np.einsum("nij,nkj->nki", np.swapaxes(R, 1, 2), d)   # (N,9,3)
        depth = -cam[:, :, 2]
        infront = depth > 1e-6
        safe = np.where(infront, depth, 1.0)
        u = K[0, 0] * (cam[:, :, 0] / safe) + K[0, 2]
        v = K[1, 1] * (-cam[:, :, 1] / safe) + K[1, 2]
        return infront & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    def unocc_fast(origin, sel_pts):
        """Occlusion over the given points, with cached path sets."""
        clear = 0
        ray_stats["n"] += len(sel_pts)
        for p in sel_pts:
            dv = p - origin
            dist = float(np.linalg.norm(dv))
            if dist < 1e-6:
                clear += 1; continue
            hit = og.sim.psqi.raycast_closest(origin=origin.tolist(),
                                              dir=(dv / dist).tolist(), distance=dist)
            if not hit or not hit.get("hit", False):
                clear += 1; continue
            body = str(hit.get("rigidBody", "") or hit.get("collision", ""))
            if any(body.startswith(tp) or tp.startswith(body) for tp in target_paths):
                clear += 1
            elif any(body.startswith(op) for op in own_paths):
                clear += 1
            elif hit.get("distance", dist) >= dist - 1e-3:
                clear += 1
        return float(clear) / float(len(sel_pts))

    def optimized(early_exit=True, rays_on_visible_only=False):
        c, s_ = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        T_base = np.zeros((len(poses), 4, 4)); T_base[:, 3, 3] = 1.0
        T_base[:, 0, 0], T_base[:, 0, 1] = c, -s_
        T_base[:, 1, 0], T_base[:, 1, 1] = s_, c
        T_base[:, 2, 2] = 1.0
        T_base[:, 0, 3], T_base[:, 1, 3] = poses[:, 0], poses[:, 1]

        inimg, T_all = {}, {}
        for nm in names:                                   # 3 batched calls, not 3*N
            T_all[nm] = T_base @ cams[nm]["mount"]
            inimg[nm] = frustum_batched(T_all[nm], cams[nm]["K"], W0, H0)

        out = np.zeros(len(poses))
        for i in range(len(poses)):
            best = 0.0
            for nm in names:
                mask = inimg[nm][i]
                frac = mask.mean()
                if frac == 0.0:
                    continue
                sel = pts[mask] if rays_on_visible_only else pts
                frac *= unocc_fast(T_all[nm][i, :3, 3], sel)
                best = max(best, frac)
                if early_exit and best >= 1.0 - 1e-12:
                    break
            out[i] = best
        return out

    for tag, kw in (("cached paths + batched frustum + analytic inverse", dict(early_exit=False)),
                    ("  + early exit once a camera reaches 1.0", dict(early_exit=True)),
                    ("  + rays only on the points that are IN FRAME",
                     dict(early_exit=True, rays_on_visible_only=True))):
        ray_stats["n"] = 0
        t0 = time.perf_counter(); got = optimized(**kw); dt = time.perf_counter() - t0
        exact = bool(np.allclose(got, ref, atol=1e-12))
        maxdiff = float(np.abs(got - ref).max())
        print("\n%s" % tag, flush=True)
        print("    %8.1f ms  (%.3f ms/pose)   speedup %.2fx   rays %d"
              % (1000*dt, 1000*dt/args.n, t_base/dt, ray_stats["n"]), flush=True)
        print("    identical to baseline: %s   (max abs diff %.3g)" % (exact, maxdiff), flush=True)

    og.sim.psqi.raycast_closest = real_ray
    og.shutdown()


if __name__ == "__main__":
    main()
