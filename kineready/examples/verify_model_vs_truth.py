"""Are the dataset labels -- and therefore the shipped model -- actually correct?

The frame audit showed the teacher is only valid with the robot's base joints at zero, which is
exactly how every shard was generated (empty scene, robot at the origin). But "the setup was
right" is an argument, not a measurement, and the whole pipeline rests on it.

This measures it against a ground truth that does not depend on the frame convention at all:

    pick a base pose p and a target L expressed in the arm's frame
    -> the world target is  W = T(p) @ L
    -> solve IK with the base joints actually set to p and W given in world coordinates
       (the formulation validated 12/12 and 10/10 against a physically-moved robot)

If the teacher's label for L and the model's prediction for (p, W) both match that, then the
labels are sound wherever the arm stands, and the 1.1 M-row dataset is trustworthy.

Includes a NEGATIVE control on the frame itself: the same comparison run with a deliberately
mismatched base pose must show poor agreement, otherwise the test is not sensitive to the very
error it exists to detect.
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kineready_models/kineready.pt")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--regime", choices=["broad", "grid"], default="broad",
                    help="'grid' reproduces the kitchen 25-tile geometry: target high and in "
                         "front, base 0.35-1.10 m away facing it")
    args = ap.parse_args()

    import omnigibson as og
    from kineready.robot_env import make_teacher_env

    env, robot = make_teacher_env()
    print("ENV_READY", flush=True)

    import torch as th
    from kineready.frames import base_pose_to_matrix, matrix_to_pose
    from kineready.reward import ReadinessReward
    from kineready.teacher import IKTeacher

    teacher = IKTeacher(robot, batch_size=16)
    base_q = robot.get_joint_positions().clone()
    ok, controls = teacher.run_controls(n_fk=32, rng=np.random.default_rng(5))
    if not ok:
        raise RuntimeError("controls failed: %s" % controls)
    reward = ReadinessReward.from_checkpoint(args.model)

    # The ground-truth formulation, lifted from base_pose_metric._solve_ik rather than importing
    # it: that class needs a camera and the empty scene deliberately has none. This is the call
    # that matched a physically-moved robot 12/12 and 10/10 -- world-frame target, base joints
    # locked at the candidate.
    joint_names = list(robot.joints.keys())
    slot = {}
    for n in robot.base_joint_names:
        j = robot.joints[n]
        key = ("yaw" if n.endswith("rz_joint") else ("x" if n.endswith("x_joint") else "y"))
        slot[key] = joint_names.index(n)
    eef_link = list(robot.eef_link_names.values())[0]

    def exact_at(base_pose, W):
        pos, quat = matrix_to_pose(W)
        q = base_q.clone()
        q[slot["x"]] = float(base_pose[0])
        q[slot["y"]] = float(base_pose[1])
        q[slot["yaw"]] = float(base_pose[2])
        successes, _ = teacher.mg.compute_trajectories(
            target_pos={eef_link: th.tensor(pos[None], dtype=th.float32)},
            target_quat={eef_link: th.tensor(quat[None], dtype=th.float32)},
            initial_joint_pos=q, is_local=False, max_attempts=5, timeout=10.0, ik_fail_return=5,
            enable_finetune_trajopt=False, finetune_attempts=0, return_full_result=False,
            success_ratio=1.0, skip_obstacle_update=True, ik_only=True,
            ik_world_collision_check=False, emb_sel=teacher._emb)
        return bool(np.asarray(successes.cpu(), dtype=bool).reshape(-1)[0])

    rng = np.random.default_rng(0)
    # Targets spanning the interesting band: from well inside the workspace to well outside.
    L = np.repeat(np.eye(4)[None], args.n, axis=0)
    for i in range(args.n):
        if args.regime == "grid":
            # The 25-tile grid's geometry: a can on a counter at ~1.02 m, the base standing
            # 0.35-1.10 m away and facing it (yaw jitter +-0.9 rad).
            rad = rng.uniform(0.35, 1.10)
            az = rng.uniform(-0.9, 0.9)
            L[i, :3, 3] = [rad * np.cos(az), rad * np.sin(az), 1.022]
        else:
            rad = rng.uniform(0.2, 1.3)
            el = rng.uniform(-0.2, 1.2)
            az = rng.uniform(-np.pi, np.pi)
            L[i, :3, 3] = [rad * np.cos(az), rad * np.sin(az), 0.35 + el * 0.6]
        tilt = np.deg2rad(rng.uniform(0, 50) if args.regime == "broad" else 23.0)
        yaw = rng.uniform(-np.pi, np.pi)
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]])
        L[i, :3, :3] = Rz @ Ry @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Base poses deliberately AWAY from the origin -- the condition under which a frame error
    # shows up at all.
    P = np.stack([rng.uniform(-2.5, 2.5, args.n), rng.uniform(-2.5, 2.5, args.n),
                  rng.uniform(-np.pi, np.pi, args.n)], axis=1)

    teacher_labels = teacher.label(L)

    truth, learned, learned_shifted = [], [], []
    for i in range(args.n):
        T_p = base_pose_to_matrix(*P[i])
        W = T_p @ L[i]
        truth.append(exact_at(P[i], W))
        learned.append(float(reward.score(P[i:i + 1], W)[0]))
        # NEGATIVE CONTROL: score the same world target against a WRONG base pose. If this agrees
        # as well as the correct pairing, the comparison is insensitive to frame errors and proves
        # nothing.
        wrong = P[(i + 7) % args.n]
        learned_shifted.append(float(reward.score(wrong[None], W)[0]))

    truth = np.array(truth)
    learned = np.array(learned)
    learned_shifted = np.array(learned_shifted)

    def agree(p):
        return float(((p >= 0.5) == truth).mean())

    summary = {
        "n": int(args.n),
        "ground_truth_feasible": int(truth.sum()),
        "teacher_vs_truth_agreement": float((teacher_labels == truth).mean()),
        "model_vs_truth_agreement": agree(learned),
        "model_vs_truth_agreement_WRONG_base_pose": agree(learned_shifted),
        "teacher_positive_rate": float(teacher_labels.mean()),
        "model_mean_p": float(learned.mean()),
    }
    print("\n=== labels and model vs frame-independent ground truth ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    good = summary["teacher_vs_truth_agreement"]
    ctrl = summary["model_vs_truth_agreement_WRONG_base_pose"]
    print("\nteacher agrees with ground truth on %.0f%% of %d poses (%d truly feasible)."
          % (100 * good, args.n, int(truth.sum())), flush=True)
    print("negative control (wrong base pose): %.0f%% -- must be clearly worse, or this test "
          "cannot see frame errors." % (100 * ctrl), flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
