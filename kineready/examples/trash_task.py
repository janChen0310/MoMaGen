"""Worked example: scoring base poses for the pick-trash-and-dispose task.

Everything task-specific lives HERE, above both the model and the reward core. The model answers
one question -- "is this eef target in the base frame reachable?" -- and knows nothing about soda
cans, trash cans, or kitchens. This file supplies the targets.

Its one real idea is symmetry augmentation. A grasp on a cylindrical object is not a single pose:
rotating the gripper about the can's vertical axis produces a different eef pose that grasps the
SAME object equally well. A base pose that cannot reach the demo's exact grasp may comfortably
reach the one 45 degrees around, so scoring only the demo pose systematically under-rates base
poses. The orbit of K=8 rotations is aggregated with noisy-OR (proposal Sec 11): P(at least one
works). That is `max` with a memory -- eight near-misses genuinely are better evidence than one,
and the result stays differentiable for reward shaping.
"""
import numpy as np


def symmetry_orbit(T_grasp, axis_point, axis_dir=(0.0, 0.0, 1.0), k=8):
    """K grasp poses related by rotation about an object symmetry axis -> (K, 4, 4).

    Args:
        T_grasp: (4, 4) world eef pose of one known-good grasp.
        axis_point: a point on the symmetry axis (the object's centre).
        axis_dir: the axis direction; vertical for a can standing on a surface.

    The rotation is applied ABOUT THE OBJECT, not about the gripper: it moves the gripper around
    the can while keeping the same relative grip. Rotating in the gripper frame instead would
    spin the tool in place and produce grasps that miss the object entirely.
    """
    T_grasp = np.asarray(T_grasp, dtype=float)
    p = np.asarray(axis_point, dtype=float)
    a = np.asarray(axis_dir, dtype=float)
    a = a / np.linalg.norm(a)

    out = np.repeat(T_grasp[None], k, axis=0)
    for i, ang in enumerate(np.linspace(0, 2 * np.pi, k, endpoint=False)):
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)       # Rodrigues
        G = np.eye(4)
        G[:3, :3] = R
        G[:3, 3] = p - R @ p                    # rotate about the axis through `p`, not the origin
        out[i] = G @ T_grasp
    return out


def grasp_priors(k=8, demo_index=0, demo_weight=1.0, other_weight=0.85):
    """Per-candidate priors for the noisy-OR.

    The demo's own grasp is the one actually observed to work; its rotated siblings are inferred
    from an assumed symmetry that is only approximately true (a can has a tab, a label, a handle
    nearby). Weighting them slightly lower keeps a base pose from claiming full credit on
    evidence that was never demonstrated.
    """
    w = np.full(k, other_weight, dtype=float)
    w[demo_index] = demo_weight
    return w


def score_base_poses(reward, base_poses, T_grasp, object_centre, k=8, **kw):
    """P_kin for each candidate base pose, aggregated over the symmetry orbit."""
    orbit = symmetry_orbit(T_grasp, object_centre, k=k)
    return reward.score(base_poses, orbit, priors=grasp_priors(k), **kw)


def main():
    """Compare orbit-aggregated scoring against single-grasp scoring on the live task."""
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-poses", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    import omnigibson as og
    from kineready.reward import ReadinessReward
    from kineready.robot_env import make_teacher_env
    from kineready.teacher import IKTeacher
    from kineready.frames import targets_in_base_frame

    env, robot = make_teacher_env()
    print("ENV_READY", flush=True)

    teacher = IKTeacher(robot, batch_size=64)
    ok, controls = teacher.run_controls(n_fk=64, rng=np.random.default_rng(11))
    if not ok:
        raise RuntimeError("controls failed: %s" % controls)
    reward = ReadinessReward.from_checkpoint(args.model)

    # A can standing on a counter, grasped from the side and tilted -- the geometry measured from
    # the real source demo (7.5 cm above the can, tilted ~23 deg), not a straight-down guess.
    centre = np.array([0.55, 0.10, 0.90])
    tilt = np.deg2rad(23.0)
    T_grasp = np.eye(4)
    T_grasp[:3, :3] = (np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0],
                                 [-np.sin(tilt), 0, np.cos(tilt)]])
                       @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    T_grasp[:3, 3] = centre + np.array([0.0, 0.0, 0.075])

    rng = np.random.default_rng(0)
    base = np.stack([rng.uniform(-1.5, 1.5, args.n_poses),
                     rng.uniform(-1.5, 1.5, args.n_poses),
                     rng.uniform(-np.pi, np.pi, args.n_poses)], axis=1)

    single = reward.score(base, T_grasp)
    orbit_scores = score_base_poses(reward, base, T_grasp, centre, k=args.k)

    # Oracle: exact IK over the whole orbit -- a base pose is truly usable if ANY orbit member
    # solves, which is exactly what the aggregation is meant to approximate.
    orbit = symmetry_orbit(T_grasp, centre, k=args.k)
    local = targets_in_base_frame(base, orbit).reshape(-1, 4, 4)
    exact_any = teacher.label(local).reshape(len(base), args.k).any(axis=1)
    exact_demo = teacher.label(targets_in_base_frame(base, T_grasp)[:, 0])

    summary = {
        "n_poses": int(len(base)),
        "k": args.k,
        "exact_feasible_demo_grasp_only": float(exact_demo.mean()),
        "exact_feasible_any_orbit_member": float(exact_any.mean()),
        "auroc_single_vs_orbit_oracle": _auroc(single, exact_any),
        "auroc_orbit_vs_orbit_oracle": _auroc(orbit_scores, exact_any),
    }
    print("\n=== TRASH-TASK SYMMETRY AUGMENTATION ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\nSymmetry augmentation raises the fraction of base poses that can grasp the can from "
          "%.1f%% to %.1f%% -- those are poses a single-grasp metric would have discarded."
          % (100 * summary["exact_feasible_demo_grasp_only"],
             100 * summary["exact_feasible_any_orbit_member"]), flush=True)

    og.shutdown()


def _auroc(p, y):
    from kineready.train import _auroc as impl

    return impl(np.asarray(p, dtype=float), np.asarray(y, dtype=bool))


if __name__ == "__main__":
    main()
