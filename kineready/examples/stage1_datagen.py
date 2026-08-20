"""Stage 1: generate the KineReady training set.

Pilot first (`--pilot`), then the full run. The pilot exists to check per-distribution positive
rates BEFORE committing the full budget: a sampler that returns 0% or 100% positives contributes
no gradient, and finding that out after the full run wastes the run.

Run:
  OMNIGIBSON_HEADLESS=1 CUDA_VISIBLE_DEVICES=N \
    python kineready/examples/stage1_datagen.py --pilot --out kineready_data/pilot
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="64 measured as the largest that fits beside Isaac on a 24 GB card")
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import omnigibson as og
    from kineready.robot_env import make_teacher_env, scene_free_report

    env, robot = make_teacher_env()
    print("ENV_READY %s" % scene_free_report(robot), flush=True)

    from kineready.datagen import generate
    from kineready.teacher import IKTeacher

    teacher = IKTeacher(robot, batch_size=args.batch_size)

    # Anchor counts. 'robust' anchors cost (1 + n_perturb) solves each, so its solve budget is
    # ~9x its row count -- sized deliberately smaller for that reason.
    plan = ({"fk": 20000, "uniform": 50000, "robust": 5000} if args.pilot
            else {"fk": 200000, "uniform": 500000, "robust": 100000})

    manifest = generate(teacher, args.out, plan=plan, shard_size=args.shard_size, seed=args.seed)
    print("\n=== STAGE 1 DONE ===", flush=True)
    print(json.dumps(manifest["counts"], indent=2), flush=True)

    _report_rates(args.out)
    og.shutdown()


def _report_rates(out_dir):
    """Per-distribution positive rates -- the pilot's actual deliverable."""
    import h5py
    import numpy as np

    print("\nper-distribution label rates:", flush=True)
    by_sampler = {}
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".hdf5"):
            continue
        with h5py.File(os.path.join(out_dir, name), "r") as f:
            s = f.attrs["sampler"]
            e = np.asarray(f["exist"])
            r = np.asarray(f["robust"])
            acc = by_sampler.setdefault(s, {"n": 0, "pos": 0, "robust": []})
            acc["n"] += len(e)
            acc["pos"] += int(e.sum())
            acc["robust"].append(r[np.isfinite(r)])
    for s, acc in sorted(by_sampler.items()):
        rob = np.concatenate(acc["robust"]) if acc["robust"] else np.array([])
        extra = ""
        if rob.size:
            extra = " | robust mean %.3f, in (0,1) for %.1f%% (the graded middle)" % (
                rob.mean(), 100 * np.mean((rob > 0) & (rob < 1)))
        print("  %-8s n=%-7d positive %.1f%%%s" % (s, acc["n"], 100 * acc["pos"] / acc["n"], extra),
              flush=True)


if __name__ == "__main__":
    main()
