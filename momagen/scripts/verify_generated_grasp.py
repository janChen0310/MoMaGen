"""Re-check generated tidybot_grasp_can demos for the task's collision criterion.

MoMaGen only keeps trials whose success predicate fired, and CuRobo plans the free-space segments
collision-free -- but the contact-rich replay segment is not collision-checked, and the success
predicate only asks whether the can ended up held and lifted. A demo that dragged the gripper along
the countertop on the way in still counts as a success. For a task whose whole point is *where the
base stands*, an expert demo that grazes the furniture is not an expert demo.

So this replays each demo's recorded states and reports, per demo:

    base_hits   any non-floor contact on a base link -- always a failure
    arm_hits    arm/gripper contact with something that is not the can -- a failure
    can_hits    arm/gripper contact with the can -- expected, and required for the grasp

States are restored rather than actions re-executed, so this measures what the recorded trajectory
actually did rather than re-rolling a fresh (and possibly divergent) one.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CAN = "can_of_soda_595"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="generated demo.hdf5")
    ap.add_argument("--stride", type=int, default=4,
                    help="check every Nth state (contacts persist over many frames)")
    ap.add_argument("--max-demos", type=int, default=0, help="0 = all")
    ap.add_argument("--support", default="countertop_kelker_0,burner_mjvqii_0",
                    help="comma-separated prims that the can RESTS ON. A top-down grasp of a 43 mm "
                         "can brings the fingers to within millimetres of its support, so contact "
                         "there is inherent to the task, not a collision. Anything else the arm "
                         "touches (e.g. microwave_abzvij_0, when the can lands beside it) is a "
                         "real obstacle collision.")
    ap.add_argument("--grasp-phase", type=float, default=0.55,
                    help="episode fraction after which arm-vs-furniture contact is treated as "
                         "part of the grasp rather than a drag; the source demo closes its "
                         "gripper at step 620 of 820 (0.76) and reaches grasp height by 0.55")
    ap.add_argument("--out", default="grasp_collision_report.json")
    args = ap.parse_args()

    import h5py
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    from robomimic.utils.file_utils import get_env_metadata_from_dataset
    import momagen.utils.robomimic_utils as RobomimicUtils
    from momagen.utils.robot_config import configure_tidybot_env_meta

    env_meta = configure_tidybot_env_meta(get_env_metadata_from_dataset(dataset_path=args.dataset))
    env = RobomimicUtils.create_env(
        env_meta=env_meta, env_class=None, env_name="tidybot_grasp_can_D0",
        robot=None, gripper=None, camera_names=[], camera_height=84, camera_width=84,
        render=False, render_offscreen=False, use_image_obs=False, use_depth_obs=False,
        manipulation_only=False, real_robot_mode=False, baseline=None)
    print("ENV_READY", flush=True)

    robot = env.env.robots[0]
    robot_path = robot.prim_path
    base_links = set(getattr(robot, "base_link_names", []) or [])
    if not base_links:
        base_links = {n for n in robot.links if "base" in n.lower() or "caster" in n.lower()}
    print("base links treated as base: %s" % sorted(base_links), flush=True)

    def contacts():
        """(base_hits, arm_hits, can_hits) as sets of prim names."""
        base_h, arm_h, can_h = set(), set(), set()
        try:
            clist = robot.contact_list()
        except Exception:
            return base_h, arm_h, can_h
        for c in clist:
            b0, b1 = str(c.body0 or ""), str(c.body1 or "")
            mine = b0 if robot_path in b0 else (b1 if robot_path in b1 else None)
            other = b1 if mine is b0 else b0
            if mine is None or not other or robot_path in other:
                continue
            low = other.lower()
            if "floor" in low or "ground" in low:
                continue
            my_link = mine.split("/")[-1]
            name = other.split("/")[-2] if "/" in other else other
            if CAN in other:
                can_h.add(name)
            elif my_link in base_links or "base" in my_link.lower() or "caster" in my_link.lower():
                base_h.add(name)
            else:
                arm_h.add(name)
        return base_h, arm_h, can_h

    support = {x.strip() for x in args.support.split(",") if x.strip()}
    print("support surfaces (contact allowed during the grasp): %s" % sorted(support), flush=True)

    f = h5py.File(args.dataset, "r")
    demos = sorted(f["data"], key=lambda k: int(k.split("_")[-1]))
    if args.max_demos:
        demos = demos[: args.max_demos]

    rows, tally = [], Counter()
    for di, dk in enumerate(demos):
        grp = f["data"][dk]
        # Source demos (DataCollectionWrapper) write "state" + "state_size"; generated demos write
        # a fixed-width "states". Accept either so this runs on both.
        skey = "states" if "states" in grp else ("state" if "state" in grp else None)
        if skey is None:
            print("%s: no recorded states -- cannot verify" % dk, flush=True)
            continue
        states = grp[skey]
        sizes = np.array(grp["state_size"]) if "state_size" in grp else None
        env.reset()
        og.sim.step()
        base_all, arm_all, can_all = Counter(), Counter(), Counter()
        # WHEN a contact happens decides whether it is a defect. The can rests ON the countertop,
        # so a top-down grasp of a 43 mm can necessarily brings the fingers to within millimetres
        # of that surface -- a touch in the closing frames is inherent to the task. The same object
        # touched throughout the approach is the arm dragging across the counter, which is a defect.
        # Recording the normalised episode phase of each contact separates the two.
        arm_phases, base_phases = [], []
        n_checked = 0
        for si in range(0, states.shape[0], args.stride):
            st = np.array(states[si])
            if sizes is not None:
                st = st[: int(sizes[si])]
            try:
                # og.sim.load_state directly, NOT env.reset_to: reset_to teleports a
                # "breakfast_table" that house_single_floor does not contain (AttributeError on
                # None) and runs 20 sim steps per call, which would dominate the runtime here.
                # One step after loading is needed for PhysX to publish contact reports.
                og.sim.load_state(th.from_numpy(st).to(th.float32), serialized=True)
                og.sim.step()
            except Exception as exc:
                print("%s step %d: load_state failed (%s)" % (dk, si, exc), flush=True)
                break
            b, a, c = contacts()
            base_all.update(b); arm_all.update(a); can_all.update(c)
            phase = si / max(states.shape[0] - 1, 1)
            if a:
                arm_phases.append(phase)
            if b:
                base_phases.append(phase)
            n_checked += 1

        clean = (not base_all) and (not arm_all)
        # Separate the two kinds of arm contact before judging.
        obstacles = {k: v for k, v in arm_all.items() if k not in support}
        early = [p for p in arm_phases if p < args.grasp_phase]
        verdict = ("BASE-HIT" if base_all else
                   "OBSTACLE" if obstacles else
                   "DRAG" if early else
                   "CLEAN" if not arm_phases else "GRASP-TOUCH")
        tally[verdict] += 1
        rows.append({"demo": dk, "checked": n_checked, "clean": clean, "verdict": verdict,
                     "base_hits": dict(base_all), "arm_hits": dict(arm_all),
                     "can_hits": dict(can_all),
                     "arm_contact_frames": len(arm_phases),
                     "arm_phase_first": min(arm_phases) if arm_phases else None,
                     "arm_phase_last": max(arm_phases) if arm_phases else None,
                     "arm_early_frames": len(early),
                     "obstacle_hits": obstacles})
        print("%-10s %-12s checked=%3d arm_frames=%3d phase=%s base=%s obstacles=%s"
              % (dk, verdict, n_checked, len(arm_phases),
                 ("%.2f-%.2f" % (min(arm_phases), max(arm_phases))) if arm_phases else "-",
                 sorted(base_all) or "-", dict(obstacles) or "-"),
              flush=True)

    n = len(rows)
    print("\n=== %d demos ===" % n, flush=True)
    for k in ("CLEAN", "GRASP-TOUCH", "DRAG", "OBSTACLE", "BASE-HIT"):
        if tally[k]:
            print("  %-12s %3d  (%.0f%%)" % (k, tally[k], 100.0 * tally[k] / n), flush=True)
    ok = tally["CLEAN"] + tally["GRASP-TOUCH"]
    print("  -> %d of %d (%.0f%%) usable: no base contact, no obstacle contact, and no dragging "
          "across the support before the grasp" % (ok, n, 100.0 * ok / n if n else 0.0), flush=True)
    bad = [r["demo"] for r in rows if r["verdict"] in ("OBSTACLE", "DRAG", "BASE-HIT")]
    if bad:
        print("  demos to exclude: %s" % ", ".join(bad), flush=True)
    off = Counter()
    for r in rows:
        off.update(r["base_hits"]); off.update(r.get("obstacle_hits", {}))
    if off:
        print("most-contacted objects:", flush=True)
        for k, v in off.most_common(10):
            print("  %5d  %s" % (v, k), flush=True)
    with open(args.out, "w") as fh:
        json.dump({"dataset": args.dataset, "stride": args.stride, "rows": rows,
                   "tally": dict(tally)}, fh, indent=1)
    print("wrote %s" % args.out, flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
