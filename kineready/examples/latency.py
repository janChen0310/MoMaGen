"""How fast is the surrogate, on CPU and GPU, at the batch sizes a policy actually queries?

The whole justification for training a model is that exact IK costs 0.34 s per pose and needs a
live CuRobo + Isaac process. This measures what replaces it. CPU numbers matter as much as GPU:
an RL rollout worker usually has no spare GPU, and a reward that needs one is not deployable.

Runs anywhere -- no simulator, no dataset, no checkpoint required (an untrained net has identical
compute cost, and this measures cost, not accuracy).
"""
import argparse
import time

import numpy as np
import torch as th

from kineready.frames import pose_to_matrix
from kineready.model import KineReadyEnsemble
from kineready.reward import ReadinessReward

EXACT_IK_SECONDS_PER_POSE = 0.34      # measured for base_pose_metric's exact term


def bench(device, n_poses, k_targets, members, repeats=20):
    model = KineReadyEnsemble(m=members)
    reward = ReadinessReward(model, device=device)
    rng = np.random.default_rng(0)
    base = np.stack([rng.uniform(-2, 2, n_poses), rng.uniform(-2, 2, n_poses),
                     rng.uniform(-np.pi, np.pi, n_poses)], axis=1)
    targets = np.stack([pose_to_matrix(rng.uniform(-1, 1, 3), [0, 0, 0, 1])
                        for _ in range(k_targets)])

    for _ in range(3):                       # warm up allocator / kernels
        reward.score(base, targets)
    if device == "cuda":
        th.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        reward.score(base, targets)
    if device == "cuda":
        th.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeats
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--k", type=int, default=8, help="targets per base pose (symmetry orbit)")
    args = ap.parse_args()

    n_params = sum(p.numel() for p in KineReadyEnsemble(m=args.members).parameters())
    print("ensemble of %d members, %d parameters total\n" % (args.members, n_params), flush=True)

    devices = ["cpu"] + (["cuda"] if th.cuda.is_available() else [])
    print("%-6s %8s %6s %10s %12s %14s" % ("device", "n_poses", "K", "seconds", "us/pose",
                                           "vs exact IK"))
    for device in devices:
        for n in (64, 512, 4096):
            dt = bench(device, n, args.k, args.members)
            speedup = (n * EXACT_IK_SECONDS_PER_POSE) / dt
            print("%-6s %8d %6d %10.5f %12.2f %13.0fx"
                  % (device, n, args.k, dt, 1e6 * dt / n, speedup), flush=True)

    print("\nExact IK for 4096 poses would take %.0f s (%.1f min); the surrogate answers the same "
          "query in milliseconds, which is what makes it usable as a per-step reward."
          % (4096 * EXACT_IK_SECONDS_PER_POSE, 4096 * EXACT_IK_SECONDS_PER_POSE / 60), flush=True)


if __name__ == "__main__":
    main()
