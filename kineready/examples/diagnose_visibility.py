"""Why does the soda can score visibility 0.00 from all 1000 sampled base poses?

The full-reward ranking collapsed: every pose tied at Phi = 0.0316, which is exactly
(1e-6)^(1/4) -- the epsilon floor in `readiness()` reached when S_v = 0. The funnel confirmed it:
can_visible 0 of 1000.

Visibility is two tests in sequence, and the metric multiplies them, so either can zero the result:

    frustum     is the can inside the camera's field of view at all?
    occlusion   is there a clear line of sight, by PhysX raycast?

They demand opposite fixes, so this separates them and -- when a ray is blocked -- reports WHAT it
hit. A ray stopped by the robot's own arm means something very different from one stopped by a wall.
"""
import argparse
import os
from collections import Counter

import numpy as np

REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
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
    scene = env.env.scene
    can = scene.object_registry("name", SODA)
    metric = BasePoseMetric(robot, distance_band=(0.30, 0.75), verbose=False)

    # EVERY camera. `vis["any"] = max(...)` over all of them, so a per-camera breakdown is the
    # only way to know which one (if any) actually contributes. Reading just the first entry is
    # how the stock mount got mistaken for the whole story.
    print("\n=== cameras discovered by the metric: %d ===" % len(metric._cameras), flush=True)
    cams = []
    for name, c in metric._cameras.items():
        K, W, H, mount = c["K"], c["width"], c["height"], c["mount"]
        fwd = -mount[:3, 2]
        pitch = -np.degrees(np.arcsin(fwd[2] / np.linalg.norm(fwd)))
        fov_h = 2 * np.degrees(np.arctan(0.5 * W / K[0, 0]))
        fov_v = 2 * np.degrees(np.arctan(0.5 * H / K[1, 1]))
        print("\n  %s  (%dx%d)" % (name, W, H), flush=True)
        print("    mount xyz : %s   height %.3f m above the base frame"
              % (mount[:3, 3].round(4), mount[2, 3]), flush=True)
        print("    forward   : %s   pitch %.1f deg below horizontal" % (fwd.round(4), pitch),
              flush=True)
        print("    FOV       : %.1f deg h, %.1f deg v   (fx=%.1f fy=%.1f)"
              % (fov_h, fov_v, K[0, 0], K[1, 1]), flush=True)
        cams.append((name, K, W, H, mount))
    # also list every VisionSensor on the robot, in case the metric filtered one out
    from omnigibson.sensors.vision_sensor import VisionSensor as _VS
    allsens = [n for n, s in robot.sensors.items() if isinstance(s, _VS)]
    print("\n  all VisionSensors on the robot: %s" % allsens, flush=True)
    print("  included by the metric        : %s" % list(metric._cameras), flush=True)

    lo, hi = can.aabb
    lo = np.asarray(lo.cpu() if hasattr(lo, "cpu") else lo, float)
    hi = np.asarray(hi.cpu() if hasattr(hi, "cpu") else hi, float)
    pts = aabb_sample_points(lo, hi)
    centre = pts[-1]
    print("\ncan AABB    : lo %s hi %s" % (lo.round(3), hi.round(3)), flush=True)
    print("can size    : %s m" % (hi - lo).round(3), flush=True)

    print("\n=== per-pose, PER-CAMERA breakdown ===", flush=True)
    hdr = "%-6s" % "r"
    for name, _, _, _, _ in cams:
        hdr += " | %-26s" % name.split(":")[-3][:26]
    print(hdr + " | vis[any]", flush=True)
    print("%-6s" % "" + "".join(" | %-8s %-8s %-6s" % ("frustum", "unoccl", "v_px")
                                for _ in cams) + " |", flush=True)

    all_hits = Counter()
    for r in np.linspace(0.4, 1.6, args.n):
        ang = 0.6                                   # a fixed bearing, so only radius varies
        xy = centre[:2] + r * np.array([np.cos(ang), np.sin(ang)])
        yaw = np.arctan2(centre[1] - xy[1], centre[0] - xy[0])
        T_base = base_pose_to_matrix(xy[0], xy[1], yaw)

        row, best = "%-6.2f" % r, 0.0
        for name, K, W, H, mount in cams:
            T_cam = T_base @ mount
            frac = visible_fraction(K, T_cam, pts, W, H)
            unocc = metric._unoccluded_fraction(T_cam[:3, 3], pts, can) if frac > 0 else 0.0
            res = project_points(K, T_cam, centre[None], W, H)
            row += " | %-8.2f %-8.2f %-6.0f" % (frac, unocc, res["pixels"][0, 1])
            best = max(best, frac * (unocc if frac > 0 else 1.0))
            all_hits.update(_hit_paths(og, T_cam[:3, 3], pts, can))
        print(row + " | %.2f" % best, flush=True)

    print("\n=== everything the rays hit, across all poses ===", flush=True)
    for path, n in all_hits.most_common(12):
        print("  %5d  %s" % (n, path), flush=True)

    own = sum(n for p, n in all_hits.items() if "tidybot" in p.lower() or "robot" in p.lower())
    total = sum(all_hits.values())
    if total:
        print("\nrays blocked by the ROBOT ITSELF: %d of %d (%.0f%%)"
              % (own, total, 100 * own / total), flush=True)
    og.shutdown()


def _hit_paths(og, origin, points, target):
    """What each blocked ray actually hit."""
    origin = np.asarray(origin, dtype=float)
    tp = {l.prim_path for l in target.links.values()} if hasattr(target, "links") else set()
    out = []
    for p in points:
        d = np.asarray(p, float) - origin
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            continue
        hit = og.sim.psqi.raycast_closest(origin=origin.tolist(), dir=(d / dist).tolist(),
                                          distance=dist)
        if not hit or not hit.get("hit", False):
            continue
        body = str(hit.get("rigidBody", "") or hit.get("collision", ""))
        if any(body.startswith(t) or str(t).startswith(body) for t in tp):
            continue
        if hit.get("distance", dist) >= dist - 1e-3:
            continue
        out.append(body)
    return out


if __name__ == "__main__":
    main()
