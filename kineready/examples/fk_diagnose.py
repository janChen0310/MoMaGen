"""Why do FK-derived poses fail IK 28% of the time?

FK poses come from real joint configurations, so they are reachable BY CONSTRUCTION -- the
config that produced them is itself a solution. A 72% label rate means one of the following is
true, and they demand opposite responses:

  A. The sampler produces poses the solver cannot be asked for (wrong joints varied, wrong
     frame, wrong link).  -> the LABELER is broken; fix it before generating any data.
  B. The solver gives up on poses that are genuinely reachable but hard (workspace boundary,
     few solutions, limited attempts).                -> the labeler is fine but PESSIMISTIC;
     the dataset would carry false negatives, which is the dangerous direction for a
     feasibility model, so the solve budget must go up.
  C. The poses are genuinely unreachable because the sampled q violates something the solver
     enforces and FK does not (self-collision under curobo's own sphere model).
                                                      -> the SAMPLER must filter with the
     solver's own model, not a different one.
  D. Nothing is wrong and the >=99% control threshold was never achievable for uniform
     joint-limit sampling.                            -> the CONTROL is miscalibrated; fix the
     threshold, not the code.

Guessing between these is how the previous two runs were spent. This script measures all four
in one Isaac boot and prints the discriminating numbers side by side.

Run:
  MOMAGEN_REPO=... OMNIGIBSON_HEADLESS=1 python kineready/examples/fk_diagnose.py
"""
import numpy as np

def main():
    # An EMPTY scene, not the trash house. The label is scene-free by definition, and the house
    # ate ~16 GB of VRAM -- enough that CuRobo OOM'd building itself at batch_size=8.
    import omnigibson as og
    from kineready.robot_env import make_teacher_env, scene_free_report

    env, robot = make_teacher_env()
    print("ENV_READY %s" % scene_free_report(robot), flush=True)

    import torch as th
    from kineready.teacher import IKTeacher

    teacher = IKTeacher(robot, batch_size=64)
    emb = teacher._emb
    kin = teacher.mg.mg[emb].kinematics
    print("free VRAM after teacher construction: %.0f MiB"
          % (th.cuda.mem_get_info()[0] / 2**20), flush=True)

    # ---------------------------------------------------------------- (0) what is even in play
    names = list(kin.joint_names)
    lim = kin.get_joint_limits()
    lo = np.asarray(lim.position[0].cpu(), dtype=float)
    hi = np.asarray(lim.position[1].cpu(), dtype=float)
    locked = teacher._locked_joint_names(kin)
    print("\n=== (0) configuration ===", flush=True)
    print("active cspace joints (%d): %s" % (len(names), names), flush=True)
    for n, a, b in zip(names, lo, hi):
        print("    %-28s [%8.3f, %8.3f]" % (n, a, b), flush=True)
    print("locked joints: %s" % locked, flush=True)
    print("curobo ee_link : %s" % teacher.mg.ee_link[emb], flush=True)
    print("teacher target : %s" % teacher._eef_link, flush=True)
    print("curobo base_link: %s" % teacher.mg.base_link[emb], flush=True)
    print("MATCH ee_link  : %s" % (teacher.mg.ee_link[emb] == teacher._eef_link), flush=True)

    # ---------------------------------------------------------------- (1) identity control
    # FK of the robot's own rest configuration. If THIS is not labeled reachable, the frame or
    # the link is wrong and nothing else in this script matters.
    import omnigibson.lazy as lazy

    def fk(q):
        cu_js = lazy.curobo.types.state.JointState(
            position=teacher.mg.tensor_args.to_device(th.tensor(np.atleast_2d(q), dtype=th.float32)),
            joint_names=names)
        out = kin.compute_kinematics(cu_js)
        p = np.asarray(out.ee_position.cpu(), dtype=float)
        qw = np.asarray(out.ee_quaternion.cpu(), dtype=float)
        T = np.repeat(np.eye(4)[None], len(p), axis=0)
        T[:, :3, 3] = p
        from kineready.frames import quat_to_mat
        for i, w in enumerate(qw):
            T[i, :3, :3] = quat_to_mat([w[1], w[2], w[3], w[0]])
        return T

    rest_full = robot.get_joint_positions().cpu().numpy()
    all_names = list(robot.joints.keys())
    q_rest = np.array([rest_full[all_names.index(n)] if n in all_names else 0.0 for n in names])
    T_rest = fk(np.repeat(q_rest[None], 4, axis=0))
    print("\n=== (1) identity control: FK(rest q) ===", flush=True)
    print("eef in base frame: pos=%s" % np.round(T_rest[0, :3, 3], 4), flush=True)
    print("labeled reachable: %.0f%%  (expect 100)" % (100 * teacher.label(T_rest).mean()), flush=True)

    # ---------------------------------------------------------------- (2) near-rest vs uniform
    # If small perturbations of rest label ~100% but uniform-over-limits labels ~72%, the
    # failures live at the workspace boundary (hypothesis B/D), not in the plumbing (A).
    rng = np.random.default_rng(0)
    n = 64
    print("\n=== (2) sampling radius sweep ===", flush=True)
    for scale, tag in [(0.15, "near-rest +-0.15 rad"), (0.5, "mid +-0.5 rad"), (None, "uniform over limits")]:
        if scale is None:
            q = rng.uniform(lo, hi, size=(n, len(lo)))
        else:
            q = np.clip(q_rest[None] + rng.uniform(-scale, scale, size=(n, len(lo))), lo, hi)
        T = fk(q)
        lab = teacher.label(T)
        r = np.linalg.norm(T[:, :3, 3], axis=1)
        print("%-22s reachable %5.1f%% | radius mean %.3f m  fail-radius %.3f  ok-radius %.3f"
              % (tag, 100 * lab.mean(), r.mean(),
                 r[~lab].mean() if (~lab).any() else float("nan"),
                 r[lab].mean() if lab.any() else float("nan")), flush=True)

    # ---------------------------------------------------------------- (3) is it solver effort?
    # Re-label the SAME failing poses with a much larger attempt budget. Recovery means the
    # solver was giving up (B); no recovery means the pose is genuinely rejected (C).
    q = rng.uniform(lo, hi, size=(128, len(lo)))
    T = fk(q)
    lab1 = teacher.label(T)
    fails = np.where(~lab1)[0]
    print("\n=== (3) solver-effort test on %d failures (of 128) ===" % len(fails), flush=True)
    if len(fails):
        teacher.max_attempts = 10
        teacher.timeout = 30.0
        lab2 = teacher.label(T[fails])
        print("recovered with max_attempts=10: %.1f%%" % (100 * lab2.mean()), flush=True)
        teacher.max_attempts = 1
        teacher.timeout = 5.0

        # ------------------------------------------------------------ (4) is it self-collision?
        # Ask curobo's OWN checker about the exact q that produced each failing pose. A high hit
        # rate here means the sampler must filter with this model (C) -- and that the poses were
        # never "guaranteed positives" to begin with.
        hits = teacher._self_collides(q[fails], names)
        ok_hits = teacher._self_collides(q[np.where(lab1)[0]][:len(fails)], names)
        print("failing q in collision (curobo checker): %.1f%%" % (100 * hits.mean()), flush=True)
        print("passing q in collision (curobo checker): %.1f%%" % (100 * ok_hits.mean()), flush=True)
    else:
        print("no failures", flush=True)

    # ------------------------------------------------------------------ (5) the control suite
    ok, report = teacher.run_controls(n_fk=64, rng=np.random.default_rng(1))
    print("\n=== (5) control suite ===\nok=%s %s" % (ok, report), flush=True)

    og.shutdown()


if __name__ == "__main__":
    main()
