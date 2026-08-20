"""Empirically map which kitchen-walkway trash-can positions a CORRECTLY-SIZED base can serve.

Pilot 4 showed that with the base collision spheres sized to the measured chassis (max reach
0.381 m vs the shipped 0.301 m), CuRobo base motion planning fails for every randomly sampled
can position -- i.e. the earlier demos were only feasible because the base was modelled ~70 mm
too small and drove through cabinet doors.

This sweep asks, per candidate position: can the primitive sample a collision-free base pose
near the can (the same call generation uses), and can CuRobo plan a path to it from the robot's
spawn? The surviving set becomes the sampling pool, so generated demos are physically valid.

Output: nav_sweep.json  [{xy, pose_found, plan_ok}, ...]
"""
import argparse, json, os, sys
import numpy as np
import torch as th

REPO = os.environ.get("MOMAGEN_REPO")
TRASH = "trash_can_596"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=0.20, help="grid spacing in m")
    ap.add_argument("--limit", type=int, default=0, help="cap positions (0 = all); use small for a smoke test")
    ap.add_argument("--out", default=os.path.join(REPO, "nav_sweep.json"))
    ap.add_argument("--min-dist", type=float, default=2.0)
    ap.add_argument("--clearance", type=float, default=0.28)
    ap.add_argument("--validate-xy", default=None,
                    help="'x,y' (or 'x,y;x,y') control positions tested first. Use the ORIGINAL "
                         "fixed can spot (4.776,-0.151): it produced 20/28 successes, so the probe "
                         "must call it navigable or the sweep's verdicts cannot be trusted.")
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    from momagen.utils.kitchen_walkway import build_walkway, map_to_world, default_dirs, RES

    scene_dir, meta_dir = default_dirs(REPO)
    eroded, size, stats, _, _ = build_walkway(scene_dir, meta_dir, clearance_m=args.clearance)
    rows, cols = np.nonzero(eroded)
    step = max(1, int(round(args.spacing / RES)))
    keep = (rows % step == 0) & (cols % step == 0)
    cand = np.stack([map_to_world((r, c), size) for r, c in zip(rows[keep], cols[keep])])
    print("GRID candidates: %d (spacing %.2f m, usable %.2f m2)" % (len(cand), args.spacing,
                                                                   stats["usable_area_m2"]), flush=True)

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

    scene = env.env.scene
    trash = scene.object_registry("name", TRASH)
    robot = env.env.robots[0]
    prim = env.primitive
    base_xy = robot.get_position_orientation()[0][:2]
    base_xy = np.array(base_xy.cpu() if hasattr(base_xy, "cpu") else base_xy, float)
    z0 = float(trash.get_position_orientation()[0][2])
    _, orn0 = trash.get_position_orientation()
    print("SPAWN_XY=%s  TRASH_Z=%.3f" % (np.round(base_xy, 3).tolist(), z0), flush=True)

    cand = [c for c in cand if np.linalg.norm(c - base_xy) >= args.min_dist]
    if args.limit:
        idx = np.linspace(0, len(cand) - 1, args.limit).astype(int)
        cand = [cand[i] for i in idx]

    # Control positions, tested first. Default is the ORIGINAL fixed can spot, which generated
    # 20/28 successes -- so the probe MUST report it navigable. If it does not, every "not
    # navigable" verdict from this sweep is meaningless and the sweep should be discarded.
    # (It sits 1.13 m from spawn, inside the min-dist filter, so it has to be added explicitly.)
    n_validate = 0
    if args.validate_xy:
        ctrl = []
        for pair in args.validate_xy.split(";"):
            x, y = (float(v) for v in pair.split(","))
            ctrl.append(np.array([x, y]))
        cand = ctrl + list(cand)
        n_validate = len(ctrl)
        print("VALIDATION positions (expect navigable): %s"
              % [[round(float(c[0]), 3), round(float(c[1]), 3)] for c in ctrl], flush=True)
    print("TESTING %d positions (%d validation + %d swept)"
          % (len(cand), n_validate, len(cand) - n_validate), flush=True)

    # Do NOT invent a drop pose. An earlier version put the gripper 0.399 m above the can
    # pointing straight down; the real one is 0.265 m up and tilted ~14 deg, and the invented
    # pose was unreachable for this tilt-constrained arm -- so sampling/planning failed at EVERY
    # position including the one that generated 20/28 successes. Take the source demo's actual
    # release pose relative to the can and re-anchor it, which is what MoMaGen itself does.
    import h5py
    from scipy.spatial.transform import Rotation as _R

    with h5py.File(src, "r") as _f:
        _dg = _f["data"][sorted(_f["data"].keys())[0]]["datagen_info"]
        _eef = _dg["eef_pose"][:]
        _ga = _dg["gripper_action"][:]
        _can = _dg["object_poses"][TRASH][:]
    _g = _ga[:, 0]
    _trans = [i for i in range(1, len(_g)) if _g[i] != _g[i - 1]]
    _rel_f = _trans[-1] if _trans else len(_g) - 1          # last open == release into the can
    REL_T = np.linalg.inv(_can[_rel_f]) @ _eef[_rel_f][0:4]  # eef expressed in the can's frame
    print("SRC release frame %d/%d; eef rel to can t=%s"
          % (_rel_f, len(_g), np.round(REL_T[:3, 3], 3).tolist()), flush=True)

    def drop_pose_for(obj):
        """Re-anchor the source release pose onto the can's CURRENT pose -> (pos, quat) batched.

        Batched (1,3)/(1,4) on purpose: _sample_pose_near_object does
        th.stack([...]).mean(dim=(0,1)), so a flat (3,) collapses to a 0-dim scalar and the
        later target_position[0] raises "invalid index of a 0-dim tensor".
        """
        p, q = obj.get_position_orientation()
        p = np.array(p.cpu() if hasattr(p, "cpu") else p, float)
        q = np.array(q.cpu() if hasattr(q, "cpu") else q, float)
        cur = np.eye(4)
        cur[:3, :3] = _R.from_quat(q).as_matrix()
        cur[:3, 3] = p
        new = cur @ REL_T
        pos = th.tensor(new[:3, 3][None, :], dtype=th.float32)
        quat = th.tensor(_R.from_matrix(new[:3, :3]).as_quat()[None, :], dtype=th.float32)
        return pos, quat

    saved = og.sim.dump_state()
    results = []
    for i, xy in enumerate(cand):
        og.sim.load_state(saved)
        trash.set_position_orientation(
            position=th.tensor([float(xy[0]), float(xy[1]), z0], dtype=th.float32), orientation=orn0)
        trash.keep_still()
        for _ in range(5):
            og.sim.step()

        pose_found, plan_ok, err = False, False, ""
        try:
            # eef_pose=None makes _sample_pose_near_object fall back to _sample_grasp_pose(obj),
            # which throws for the ashcan (nothing graspable). Generation never hits that path --
            # it always passes the re-anchored subtask eef pose. Approximate the DROP pose the
            # same way the source demo ends: gripper above the can, pointing straight down.
            # Approximate, but it is what determines where the base must stand, and the sampler
            # tries many base yaws around it.
            pose = prim._sample_pose_near_object(obj=trash, eef_pose=drop_pose_for(trash))
            pose_found = pose is not None
        except Exception as exc:
            err = "sample:%s:%s" % (type(exc).__name__, str(exc)[:50])
            pose = None
        if pose_found:
            # `pose` is a 2D (x, y, yaw) pose -- _navigate_to_pose converts it via
            # _get_robot_pose_from_2d_pose and plans with the BASE embodiment, yielding None
            # when planning fails. Driving the generator one step runs exactly the planning
            # that generation runs, and since we never execute the yielded action the robot
            # does not move.
            try:
                gen = prim._navigate_to_pose(pose, skip_obstacle_update=True)
                first = next(gen, None)
                gen.close()
                plan_ok = first is not None
            except Exception as exc:
                err = "plan:%s:%s" % (type(exc).__name__, str(exc)[:60])

        results.append({"xy": [float(xy[0]), float(xy[1])],
                        "dist_from_spawn": float(np.linalg.norm(xy - base_xy)),
                        "pose_found": bool(pose_found), "plan_ok": bool(plan_ok), "err": err})
        print("POS %3d/%d (%6.2f,%6.2f) d=%.2f pose=%-5s plan=%-5s %s"
              % (i + 1, len(cand), xy[0], xy[1], results[-1]["dist_from_spawn"],
                 pose_found, plan_ok, err), flush=True)
        if (i + 1) % 10 == 0:
            json.dump(results, open(args.out, "w"), indent=1)

    json.dump(results, open(args.out, "w"), indent=1)
    npose = sum(r["pose_found"] for r in results)
    nplan = sum(r["plan_ok"] for r in results)
    print("SWEEP_SUMMARY tested=%d pose_found=%d (%.1f%%) plan_ok=%d (%.1f%%)"
          % (len(results), npose, 100.0*npose/max(1,len(results)), nplan, 100.0*nplan/max(1,len(results))),
          flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
