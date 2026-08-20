"""Stage 0: verify the IK teacher is correct, then measure how fast it can label.

Two questions, in this order:
  1. Are the labels TRUSTWORTHY? Controls must pass before any timing number means anything.
  2. How many labels per second? This sizes the Stage-1 dataset budget and answers the proposal's
     Phase-0 go/no-go ("is exact batched IK actually a bottleneck?") with a measurement rather
     than an assumption.

The environment is an empty scene, not the trash house -- see `kineready/robot_env.py`. The label
is scene-free by construction, and the house's ~16 GB of RTX geometry was what forced CuRobo down
to batch_size=8, capping the very throughput this script exists to measure.

Run:
  OMNIGIBSON_HEADLESS=1 CUDA_VISIBLE_DEVICES=N python kineready/examples/stage0_benchmark.py
"""
import argparse
import gc
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", default="64,256,1024")
    ap.add_argument("--n", type=int, default=4096, help="poses per timing run")
    args = ap.parse_args()

    import omnigibson as og
    from kineready.robot_env import make_teacher_env, scene_free_report

    env, robot = make_teacher_env()
    print("ENV_READY %s" % scene_free_report(robot), flush=True)

    import torch as th
    from kineready.frames import quat_to_mat
    from kineready.teacher import IKTeacher

    rng = np.random.default_rng(0)

    def free(obj):
        """Release a motion generator's VRAM before building the next one.

        `batch_size` is fixed at construction, so comparing sizes means constructing several, and
        each holds a large allocation (512 IK seeds + collision world). Without this the third
        construction OOMs the card.
        """
        del obj
        gc.collect()
        th.cuda.empty_cache()

    # Uniform task-space poses: the distribution Stage 1 will actually spend its budget on. Built
    # once so every batch size is timed on identical work.
    def make_targets(n):
        T = np.repeat(np.eye(4)[None], n, axis=0)
        T[:, :3, 3] = np.stack([rng.uniform(-1.1, 1.1, n),
                                rng.uniform(-1.1, 1.1, n),
                                rng.uniform(0.0, 1.5, n)], axis=1)
        for i in range(n):
            q = rng.normal(size=4)
            T[i, :3, :3] = quat_to_mat(q / np.linalg.norm(q))
        return T

    T_uniform = make_targets(args.n)

    results = []
    teacher = None
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        print("\n=== batch_size=%d ===" % bs, flush=True)
        if teacher is not None:
            free(teacher)
            teacher = None
        free_mib = th.cuda.mem_get_info()[0] / 2**20
        print("free VRAM before construction: %.0f MiB" % free_mib, flush=True)
        t0 = time.perf_counter()
        try:
            teacher = IKTeacher(robot, batch_size=bs)
        except th.OutOfMemoryError:
            # A batch size that does not fit is a data point, not a reason to abandon the run:
            # `compute_trajectories` chunks internally, so a smaller batch still labels
            # arbitrarily many targets, just in more chunks.
            print("batch_size=%d does NOT fit in %.0f MiB -- skipping" % (bs, free_mib), flush=True)
            teacher = None
            th.cuda.empty_cache()
            continue
        print("construction: %.2f s | free VRAM after: %.0f MiB"
              % (time.perf_counter() - t0, th.cuda.mem_get_info()[0] / 2**20), flush=True)

        ok, report = teacher.run_controls(n_fk=64, rng=np.random.default_rng(1))
        if not ok:
            # Refuse to report throughput for a labeler that cannot label correctly -- a fast
            # wrong labeler is worse than a slow one, because the dataset looks fine.
            print("CONTROLS FAILED -- aborting, timings would be meaningless: %s" % report,
                  flush=True)
            free(teacher)
            og.shutdown()
            return

        teacher.label(T_uniform[:bs])            # warm up kernels; excluded from the timing
        labels = teacher.label(T_uniform)
        s = teacher.last_stats
        print("uniform  n=%d  %.2f s  -> %.0f labels/s  (positive rate %.1f%%)"
              % (s["n"], s["seconds"], s["per_sec"], 100 * s["positive_rate"]), flush=True)

        t0 = time.perf_counter()
        exist, robust = teacher.label_robust(T_uniform[:256], n_perturb=8, rng=rng)
        dt = time.perf_counter() - t0
        print("robust   n=256 x (1+8)  %.2f s  -> %.0f anchors/s  (mean robust %.3f)"
              % (dt, 256 / dt, float(robust.mean())), flush=True)

        # What the whole plan hinges on: labels/s vs the 0.34 s/pose serial cost measured earlier
        # for the exact-IK term in base_pose_metric.
        print("speedup vs serial exact IK (0.34 s/pose): %.0fx" % (s["per_sec"] * 0.34), flush=True)
        results.append((bs, s["per_sec"]))

    if results:
        best_bs, best_rate = max(results, key=lambda r: r[1])
        print("\n=== STAGE 0 RESULT ===", flush=True)
        print("best: batch_size=%d at %.0f labels/s" % (best_bs, best_rate), flush=True)
        for budget, tag in [(2_300_000, "full plan"), (100_000, "pilot")]:
            print("  %-10s %d solves -> %.1f h" % (tag, budget, budget / best_rate / 3600),
                  flush=True)

    og.shutdown()


if __name__ == "__main__":
    main()
