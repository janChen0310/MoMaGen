"""Sample 25 base poses, score them, and render what the base camera sees at each.

Produces a 5x5 contact sheet of the base-camera view with the four metrics printed on every tile,
so the numbers can be checked against the picture rather than taken on faith.

Note the two halves work differently on purpose:
  * SCORING is hypothetical -- all 25 poses are evaluated in one batch without the robot moving.
  * RENDERING cannot be; the robot is teleported to each pose, rendered, and the full sim state is
    restored afterwards so every tile is independent and the scene is left exactly as found.

Run:
  MOMAGEN_REPO=... OMNIGIBSON_HEADLESS=1 python base_pose_metric/examples/render_pose_grid.py \
      --ignore countertop_kelker_0,... --source-demo .../tidybot_picking_up_trash.hdf5
"""
import argparse
import os

import numpy as np
import torch as th

REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SODA = "can_of_soda_595"
TRASH = "trash_can_596"


def grasp_pose_from_source(path, target, obj_name):
    """The demo's real grasp pose, re-anchored onto the target's current pose."""
    import h5py
    from scipy.spatial.transform import Rotation as R

    with h5py.File(path, "r") as f:
        dg = f["data"][sorted(f["data"].keys())[0]]["datagen_info"]
        eef, ga = dg["eef_pose"][:], dg["gripper_action"][:]
        obj = dg["object_poses"][obj_name][:]
    g = ga[:, 0]
    trans = [i for i in range(1, len(g)) if g[i] != g[i - 1]]
    rel = np.linalg.inv(obj[trans[0] if trans else 0]) @ eef[trans[0] if trans else 0][0:4]

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
    ap.add_argument("--target", choices=["soda", "trash"], default="soda")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--r-min", type=float, default=0.35)
    ap.add_argument("--r-max", type=float, default=1.10)
    ap.add_argument("--yaw-jitter", type=float, default=0.9,
                    help="radians of yaw noise around 'facing the target' -- without this every "
                         "tile frames the object identically and visibility never varies")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ignore", default="")
    ap.add_argument("--source-demo", default=None)
    ap.add_argument("--model", default=None,
                    help="kineready checkpoint; prints learned p_kin beside the exact IK verdict")
    ap.add_argument("--out", default=os.path.join(REPO, "base_pose_grid.png"))
    args = ap.parse_args()

    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    import cv2
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
    from omnigibson.sensors.vision_sensor import VisionSensor

    robot = env.env.robots[0]
    scene = env.env.scene
    obj_name = SODA if args.target == "soda" else TRASH
    target = scene.object_registry("name", obj_name)
    assert target is not None

    # base camera = the mast third-person cam; pick it by link name, not position in the dict
    cam = None
    for name, s in robot.sensors.items():
        if isinstance(s, VisionSensor) and "base_camera" in name:
            cam = s
            break
    assert cam is not None, "no base camera found"
    cam.add_modality("rgb")
    print("[grid] base camera: %s (%dx%d)" % (cam.name, cam.image_width, cam.image_height), flush=True)

    ignore = None
    if args.ignore:
        wants = [w.strip() for w in args.ignore.split(",") if w.strip()]
        ignore = [o for o in scene.objects if any(w in o.name for w in wants)]

    metric = BasePoseMetric(robot, distance_band=(0.30, 0.75), ignore_objects=ignore)
    eef_pose = grasp_pose_from_source(args.source_demo, target, obj_name) if args.source_demo else None
    if eef_pose is None:
        # Resolve it HERE rather than letting evaluate() derive it internally, so the exact solve
        # and the learned prediction are answering about the identical target. Otherwise the two
        # numbers printed side by side on each tile would be about different grasps -- a
        # comparison that looks meaningful and is not.
        eef_pose = metric.derive_eef_pose(target)

    # ---- sample poses in an annulus around the target, roughly facing it -------------------
    lo, hi = target.aabb
    centre = ((np.asarray(lo.cpu(), float) + np.asarray(hi.cpu(), float)) / 2.0)[:2]
    rng = np.random.default_rng(args.seed)
    poses = []
    while len(poses) < args.n:
        r = rng.uniform(args.r_min, args.r_max)
        a = rng.uniform(-np.pi, np.pi)
        xy = centre + r * np.array([np.cos(a), np.sin(a)])
        yaw = np.arctan2(centre[1] - xy[1], centre[0] - xy[0]) + rng.uniform(
            -args.yaw_jitter, args.yaw_jitter)
        poses.append([xy[0], xy[1], yaw])
    poses = np.array(poses)

    # ---- score all of them in ONE batch, robot untouched ------------------------------------
    # The exact solve stays the ground truth on every tile; the learned prediction is computed
    # separately and printed alongside, so the grid compares them rather than replacing one.
    # skip_ik_if_colliding=False is required for the comparison to mean anything. With the default
    # True, a base pose that collides never gets an IK solve at all and reports ik_ok=False -- so
    # comparing the learned p_kin against it would be comparing against "not evaluated" rather than
    # "infeasible". In a kitchen most sampled poses collide, so nearly the whole grid would be
    # meaningless while still printing a confident-looking agreement percentage.
    results = metric.evaluate(poses, target, eef_pose=eef_pose, use_learned_ik=False,
                              skip_ik_if_colliding=not args.model)
    print("[grid] scored %d poses (%d feasible)"
          % (len(results), sum(r["feasible"] for r in results)), flush=True)

    preds = None
    if args.model:
        from kineready.frames import pose_to_matrix
        from kineready.reward import ReadinessReward

        T_target = pose_to_matrix(
            np.asarray(eef_pose[0].cpu() if hasattr(eef_pose[0], "cpu") else eef_pose[0], float),
            np.asarray(eef_pose[1].cpu() if hasattr(eef_pose[1], "cpu") else eef_pose[1], float))
        preds = np.asarray(ReadinessReward.from_checkpoint(args.model).score(poses, T_target))
        exact = np.array([r["ik_ok"] for r in results])
        agree = (preds >= 0.5) == exact
        # The two are NOT answering the same question: the exact solve here runs with
        # ik_world_collision_check defaulting to True, so it also rejects arm configurations that
        # hit the counter, while the learned model is scene-free by construction. Disagreements in
        # the optimistic direction (model says reachable, exact says no) are therefore expected and
        # are the documented scope boundary -- the pessimistic direction is the one that would
        # indicate a model problem.
        opt = int(((preds >= 0.5) & ~exact).sum())
        pes = int(((preds < 0.5) & exact).sum())
        print("[grid] learned (scene-free) vs exact (scene-aware) IK: %.1f%% agree of %d tiles "
              "| %d optimistic (furniture blocks the arm), %d pessimistic"
              % (100 * agree.mean(), len(poses), opt, pes), flush=True)
        print("[grid] exact feasible: %d, base-collision-free: %d"
              % (int(exact.sum()), int(sum(not r["collision_static"] for r in results))), flush=True)

    # ---- render each pose, restoring the scene between tiles ---------------------------------
    saved = og.sim.dump_state()
    tiles = []
    for i, (x, y, yaw) in enumerate(poses):
        og.sim.load_state(saved)
        robot.set_position_orientation(
            position=th.tensor([float(x), float(y), 0.0], dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))],
                                  dtype=th.float32))
        robot.keep_still()
        for _ in range(3):
            og.sim.step()
        og.sim.render()
        obs = cam.get_obs()
        rgb = obs[0]["rgb"] if isinstance(obs, tuple) else obs["rgb"]
        rgb = np.asarray(rgb.cpu() if hasattr(rgb, "cpu") else rgb)[:, :, :3].astype(np.uint8)
        tiles.append(_annotate(cv2, rgb, results[i], None if preds is None else float(preds[i])))
        print("  [%2d/%d] d=%.2f vis=%.2f ik=%-5s coll=%-5s score=%.2f"
              % (i + 1, len(poses), results[i]["distance"], results[i]["visibility"]["any"],
                 results[i]["ik_ok"],
                 results[i]["collision_static"] or results[i]["collision_reach"],
                 results[i]["score"]), flush=True)
    og.sim.load_state(saved)
    for _ in range(3):
        og.sim.step()

    side = int(np.ceil(np.sqrt(len(tiles))))
    h, w = tiles[0].shape[:2]
    sheet = np.zeros((side * h, side * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, side)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    cv2.imwrite(args.out, sheet[:, :, ::-1])
    print("wrote %s (%dx%d)" % (args.out, sheet.shape[1], sheet.shape[0]), flush=True)
    og.shutdown()


def _annotate(cv2, rgb, res, pred=None):
    """Print the metrics onto a tile. Green border = feasible, red = not.

    When `pred` (a learned p_kin for the same pose) is supplied, it is printed BESIDE the exact
    verdict and a disagreement is called out explicitly. Two numbers agreeing on every tile is
    weak evidence; the value of this artifact is that a systematic disagreement -- a whole row or
    a whole distance band -- is immediately visible, which no aggregate metric shows.
    """
    img = np.ascontiguousarray(rgb[:, :, ::-1])          # to BGR for cv2
    ok = res["feasible"]
    colour = (80, 220, 80) if ok else (60, 60, 235)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), colour, 3)
    coll = res["collision_static"] or res["collision_reach"]
    lines = [
        "d=%.2fm  vis=%.2f" % (res["distance"], res["visibility"]["any"]),
        "ik=%s  coll=%s" % ("Y" if res["ik_ok"] else "N", "Y" if coll else "N"),
        "score=%.2f" % res["score"],
    ]
    if pred is not None:
        agrees = (pred >= 0.5) == bool(res["ik_ok"])
        lines.append("p_kin=%.2f %s" % (pred, "" if agrees else "<< DISAGREES"))
    y = 16
    for ln in lines:
        cv2.putText(img, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        y += 17
    return img[:, :, ::-1]                                # back to RGB


if __name__ == "__main__":
    main()
