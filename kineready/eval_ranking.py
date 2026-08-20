"""Stage 3: does the surrogate RANK base poses the way exact IK does?

Offline AUROC answers "is each individual prediction right". That is not the question the metric
is for. A navigation policy picks ONE endpoint out of many candidates, so what matters is whether
the pose the model ranks first is actually reachable, and whether the ordering it induces matches
the oracle's. A model can have excellent AUROC and still rank badly if its errors concentrate
among the top candidates -- which is precisely where they hurt.

FRAME GATE. Before any ranking number is reported, this checks that the base frame the model was
trained in is the base frame the metric queries it in. `targets_in_base_frame` builds the base
transform from a planar (x, y, yaw) at z=0; the teacher labeled targets relative to curobo's
`base_footprint_x` link. If those differ by so much as a fixed offset, every prediction is a
coherent answer to the wrong question -- and the ranking metrics would still look plausible,
because the error is systematic rather than noisy. The gate compares the two constructions
directly on the live robot and aborts on mismatch.
"""
import argparse
import json
import time

import numpy as np


def frame_gate(robot, teacher, tol=1e-3):
    """Verify the model's base frame == the teacher's base frame. Returns (ok, detail).

    Takes the robot's ACTUAL eef pose in the world, converts it with the same function the reward
    path uses, and compares against curobo's own forward kinematics in its base frame.
    """
    from .frames import targets_in_base_frame, pose_to_matrix

    eef_link = list(robot.eef_link_names.values())[0]
    pos, quat = robot.links[eef_link].get_position_orientation()
    T_world = pose_to_matrix(np.asarray(pos.cpu()), np.asarray(quat.cpu()))

    bpos, bquat = robot.get_position_orientation()
    bpos = np.asarray(bpos.cpu())
    bq = np.asarray(bquat.cpu())
    yaw = np.arctan2(2 * (bq[3] * bq[2] + bq[0] * bq[1]),
                     1 - 2 * (bq[1] ** 2 + bq[2] ** 2))
    via_reward = targets_in_base_frame(np.array([[bpos[0], bpos[1], yaw]]), T_world)[0, 0]
    via_teacher = teacher.rest_pose_target()

    dp = float(np.linalg.norm(via_reward[:3, 3] - via_teacher[:3, 3]))
    dR = float(np.linalg.norm(via_reward[:3, :3] - via_teacher[:3, :3]))
    ok = dp < 1e-2 and dR < 5e-2
    return ok, {"position_gap_m": dp, "rotation_gap_fro": dR,
                "via_reward_xyz": via_reward[:3, 3].round(4).tolist(),
                "via_teacher_xyz": via_teacher[:3, 3].round(4).tolist()}


def spearman(a, b):
    """Rank correlation without scipy (ties averaged)."""
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(1, len(v) + 1)
        sv = v[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def ranking_metrics(p_pred, exact_ok, top_k=(1, 5)):
    """How well does the predicted ordering serve endpoint selection?"""
    p_pred = np.asarray(p_pred, dtype=float)
    exact_ok = np.asarray(exact_ok, dtype=bool)
    out = {"n": int(len(p_pred)), "n_feasible": int(exact_ok.sum()),
           "spearman": spearman(p_pred, exact_ok.astype(float))}
    order = np.argsort(-p_pred)
    for k in top_k:
        # "Did at least one of the model's top-k candidates actually work?" -- the quantity that
        # decides whether the policy's chosen endpoint succeeds.
        out["top%d_hit" % k] = bool(exact_ok[order[:k]].any()) if len(order) else False
    out["top1_feasible"] = bool(exact_ok[order[0]]) if len(order) else False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-configs", type=int, default=20)
    ap.add_argument("--n-poses", type=int, default=64)
    ap.add_argument("--out", default="kineready_eval_ranking.json")
    args = ap.parse_args()

    import omnigibson as og
    from .robot_env import make_teacher_env, scene_free_report

    env, robot = make_teacher_env()
    print("ENV_READY %s" % scene_free_report(robot), flush=True)

    import torch as th
    from .frames import pose_to_matrix, targets_in_base_frame
    from .reward import ReadinessReward
    from .teacher import IKTeacher

    teacher = IKTeacher(robot, batch_size=64)
    ok, controls = teacher.run_controls(n_fk=64, rng=np.random.default_rng(7))
    if not ok:
        raise RuntimeError("teacher controls failed, oracle is untrustworthy: %s" % controls)

    gate_ok, gate = frame_gate(robot, teacher)
    print("\nFRAME GATE %s: %s" % ("PASS" if gate_ok else "FAIL", json.dumps(gate)), flush=True)
    if not gate_ok:
        print("ABORTING: the model's base frame does not match the teacher's. Every prediction "
              "would be a coherent answer to the wrong question.", flush=True)
        og.shutdown()
        return

    reward = ReadinessReward.from_checkpoint(args.model)

    # Each "config" is one world target; the candidates are base poses around it. This is exactly
    # the decision an endpoint selector faces.
    rng = np.random.default_rng(0)
    per_config, latencies = [], {}
    for c in range(args.n_configs):
        obj_xy = rng.uniform(-1.5, 1.5, 2)
        obj_z = rng.uniform(0.4, 1.1)
        # A tilted grasp, not straight down: the real trash grasp is tilted ~23 deg, and a
        # straight-down assumption already produced a wrong answer once in this project.
        tilt = np.deg2rad(rng.uniform(0, 40))
        yaw_t = rng.uniform(-np.pi, np.pi)
        Rz = np.array([[np.cos(yaw_t), -np.sin(yaw_t), 0], [np.sin(yaw_t), np.cos(yaw_t), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0],
                       [-np.sin(tilt), 0, np.cos(tilt)]])
        T_target = np.eye(4)
        T_target[:3, :3] = Rz @ Ry @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        T_target[:3, 3] = [obj_xy[0], obj_xy[1], obj_z]

        base = np.stack([rng.uniform(-2.0, 2.0, args.n_poses),
                         rng.uniform(-2.0, 2.0, args.n_poses),
                         rng.uniform(-np.pi, np.pi, args.n_poses)], axis=1)

        t0 = time.perf_counter()
        p = reward.score(base, T_target)
        latencies.setdefault("learned_s", []).append(time.perf_counter() - t0)

        local = targets_in_base_frame(base, T_target)[:, 0]
        t0 = time.perf_counter()
        exact = teacher.label(local)
        latencies.setdefault("exact_s", []).append(time.perf_counter() - t0)

        per_config.append(ranking_metrics(p, exact))
        if c % 5 == 0:
            print("config %2d: feasible %d/%d  top1 %s  spearman %.3f"
                  % (c, per_config[-1]["n_feasible"], args.n_poses,
                     per_config[-1]["top1_feasible"], per_config[-1]["spearman"]), flush=True)

    usable = [m for m in per_config if 0 < m["n_feasible"] < m["n"]]
    summary = {
        "controls": controls,
        "frame_gate": gate,
        "n_configs": len(per_config),
        # Configs where every candidate works, or none does, cannot discriminate any ranker --
        # averaging them in would inflate the score with free wins.
        "n_configs_discriminative": len(usable),
        "mean_spearman": float(np.mean([m["spearman"] for m in usable])) if usable else float("nan"),
        "top1_feasible_rate": float(np.mean([m["top1_feasible"] for m in usable])) if usable else float("nan"),
        "top5_hit_rate": float(np.mean([m["top5_hit"] for m in usable])) if usable else float("nan"),
        "learned_s_per_call": float(np.mean(latencies["learned_s"])),
        "exact_s_per_call": float(np.mean(latencies["exact_s"])),
        "poses_per_call": args.n_poses,
    }
    summary["speedup"] = summary["exact_s_per_call"] / max(summary["learned_s_per_call"], 1e-9)

    print("\n=== STAGE 3 RANKING ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_config": per_config}, f, indent=2)

    og.shutdown()


if __name__ == "__main__":
    main()
