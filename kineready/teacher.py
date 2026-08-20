"""Exact IK teacher: labels target eef poses (in the base frame) as reachable or not.

Uses CuRobo through OmniGibson's `CuRoboMotionGenerator`. Three properties of that wrapper make
this cheap and scene-free, all verified by reading the source:

  * `is_local=True` -- targets are interpreted in the robot's base frame, so a candidate base pose
    becomes a transformed target and THE ROBOT NEVER MOVES. No teleporting, no state save/restore.
  * internal chunking -- `compute_trajectories` splits `num_targets` into `batch_size` groups
    itself (curobo.py ~line 868), so one call labels an arbitrary number of poses.
  * `ik_world_collision_check=False` -- disables world-collision cost/constraint for the solve
    (curobo.py ~line 854), giving the scene-free label the proposal specifies. Joint limits and
    self-collision are still enforced (`self_collision_check=True` at construction), which is
    exactly the Q_valid of proposal Sec 6.1.

The solver already runs 512 IK seeds per target, so the proposal's "multi-seed solver-robust
label" (Sec 7) needs no seed loop -- one call is already the multi-seed answer.

CONTROLS. Every run embeds known-answer samples and aborts if they fail. This is not ceremony:
during development of the sibling `base_pose_metric`, five separate bugs each produced confident,
plausible, completely wrong output (a collision model sunk below the floor, an IK embodiment that
silently never solves, a swallowed exception, an invented grasp pose, a frustum test blind to
walls). A labeling run that is silently wrong poisons the dataset and everything trained on it,
and unlike those bugs it would not be visible in any rendering.
"""
import time

import numpy as np
import torch as th

from .frames import encode_features


class IKTeacher:
    """Batched exact-IK labeler for target poses expressed in the robot's base frame."""

    def __init__(self, robot, motion_generator=None, batch_size=128, max_attempts=1,
                 timeout=5.0, verbose=True, embodiment=None, motion_cfg_kwargs=None):
        self.robot = robot
        self.batch_size = int(batch_size)
        self.max_attempts = int(max_attempts)
        self.timeout = float(timeout)
        self.verbose = verbose

        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

        # ARM_NO_TORSO is not a preference -- TidyBot has no torso and the plain ARM embodiment
        # fails EVERY solve while looking exactly like "unreachable". Cost us a full debugging
        # cycle on the sibling module.
        self._emb = embodiment or CuRoboEmbodimentSelection.ARM_NO_TORSO
        if motion_generator is not None:
            self.mg = motion_generator
            # A generator we did not build may lack our embodiment; fail loudly rather than
            # silently labeling everything unreachable under the wrong one.
            if self._emb not in self.mg.mg:
                raise ValueError("supplied motion generator has no %s embodiment (has %s)"
                                 % (self._emb, list(self.mg.mg)))
        else:
            self.mg = self._build(batch_size, motion_cfg_kwargs)
        self._eef_link = list(robot.eef_link_names.values())[0]
        self._rest_q = robot.get_joint_positions().clone()
        self.last_stats = {}
        self._assert_base_at_origin()

    def _assert_base_at_origin(self, tol=1e-3):
        """Refuse to label unless the robot's base joints are at zero.

        `is_local=True` is a MISNOMER for a holonomic base. Measured on TidyBot: the link curobo
        treats as its base (`base_footprint_x`) sits at the WORLD ORIGIN with identity orientation
        no matter where the robot is -- the world pose lives entirely in the base joints. So the
        wrapper's world->base conversion is the identity, and "local" targets are really WORLD
        targets.

        That makes the frame this class documents -- the target in the arm's own base frame --
        correct only while the arm is at the origin. Move the robot to (4.79, -1.28) and ask about
        a target 0.3 m in front of it, and curobo is asked about a point 0.3 m from the origin,
        4.8 m from the arm: everything comes back unreachable, with no error and no warning.

        This cost a long detour, and none of the existing checks could catch it: the controls are
        self-consistent round-trips through the same frame, and the frame gate ran in the empty
        scene at the origin, where the correct and incorrect frames coincide. A gate that cannot
        fail is not a gate. So the invariant is asserted directly instead.
        """
        names = list(self.robot.joints.keys())
        q = self._rest_q.cpu().numpy()
        bad = {}
        for n in getattr(self.robot, "base_joint_names", []):
            if n in names and abs(float(q[names.index(n)])) > tol:
                bad[n] = round(float(q[names.index(n)]), 4)
        if bad:
            raise ValueError(
                "IKTeacher requires the robot's base joints at zero, but found %s. For a holonomic "
                "base curobo's 'local' target frame IS the world origin, so with the base elsewhere "
                "every target would be interpreted %.2f m away from the arm and labeled "
                "unreachable -- silently. Build the teacher in an empty scene with the robot at the "
                "origin (kineready.robot_env.make_teacher_env), or pre-compose the base offset into "
                "the targets yourself." % (bad, float(np.linalg.norm(list(bad.values())[:2]))))

    def _build(self, batch_size, motion_cfg_kwargs=None):
        """Build ONE embodiment's motion generator.

        `CuRoboMotionGenerator` defaults to constructing every embodiment in the robot's config
        (BASE, ARM, ARM_NO_TORSO, DEFAULT), each with 512 IK seeds AND a full trajopt warmup plan
        in its constructor. We call exactly one of them, and only ever through `ik_only=True`, so
        the other three are pure VRAM. Building all four is what made batch_size=8 OOM on a 24 GB
        card -- the cap on IK throughput was self-inflicted.
        """
        from omnigibson.action_primitives.curobo import (CuRoboEmbodimentSelection,
                                                          CuRoboMotionGenerator)

        scene_model = getattr(getattr(self.robot, "scene", None), "scene_model", None)
        # DEFAULT rides along because `check_collisions` hardcodes it (curobo.py ~line 394:
        # "only makes sense for the default embodiment where all the joints are actuated").
        # It costs little: the constructor's trajopt warmup explicitly skips DEFAULT.
        embodiments = [self._emb]
        if CuRoboEmbodimentSelection.DEFAULT not in embodiments:
            embodiments.append(CuRoboEmbodimentSelection.DEFAULT)
        return CuRoboMotionGenerator(
            robot=self.robot, batch_size=batch_size, use_cuda_graph=False,
            embodiment_types=embodiments,
            motion_cfg_kwargs=motion_cfg_kwargs,
            scene_model=scene_model.lower() if isinstance(scene_model, str) else "empty")

    # ------------------------------------------------------------------ labeling

    def label(self, T_base_frame):
        """Label (N, 4, 4) targets-in-base-frame -> (N,) bool array of IK existence.

        Poses are passed with `is_local=True`, so each is answered as "could the arm reach this
        pose relative to its own base" -- independent of where the base actually is.
        """
        T = np.asarray(T_base_frame, dtype=float)
        if T.ndim == 2:
            T = T[None]
        n = len(T)

        pos = th.tensor(T[:, :3, 3], dtype=th.float32)
        quat = th.tensor(np.stack([_mat_to_quat_xyzw(R) for R in T[:, :3, :3]]), dtype=th.float32)

        t0 = time.perf_counter()
        try:
            successes, _ = self.mg.compute_trajectories(
                target_pos={self._eef_link: pos},
                target_quat={self._eef_link: quat},
                initial_joint_pos=self._rest_q,
                is_local=True,                    # base frame -> robot never moves
                max_attempts=self.max_attempts,
                timeout=self.timeout,
                ik_fail_return=5,
                enable_finetune_trajopt=False,
                finetune_attempts=0,
                return_full_result=False,
                success_ratio=1.0,
                skip_obstacle_update=True,
                ik_only=True,
                ik_world_collision_check=False,   # scene-free label (proposal MVP)
                emb_sel=self._emb,
            )
        except Exception as exc:
            # Never collapse a misconfiguration into "unreachable" -- they are indistinguishable
            # downstream and mean opposite things.
            raise RuntimeError("IK teacher failed under %s: %s" % (self._emb, exc)) from exc
        out = np.asarray(successes.cpu(), dtype=bool).reshape(-1)[:n]

        dt = time.perf_counter() - t0
        self.last_stats = {"n": n, "seconds": dt, "per_sec": n / dt if dt > 0 else float("inf"),
                           "positive_rate": float(out.mean()) if n else 0.0,
                           "embodiment": str(self._emb)}
        return out

    def label_robust(self, T_base_frame, n_perturb=8, sigma_pos=0.02, sigma_rot=np.deg2rad(5.0),
                     rng=None):
        """Robust label (proposal Sec 6.2): fraction of perturbed copies that remain reachable.

        Returns (exist, robust) where `exist` is the nominal label and `robust` in [0, 1] measures
        reachability MARGIN -- the difference between a target deep in the workspace and one on
        the boundary, which a binary label cannot express and which is the actually useful
        navigation signal.
        """
        from .frames import perturb_poses

        T = np.asarray(T_base_frame, dtype=float)
        if T.ndim == 2:
            T = T[None]
        rng = rng or np.random.default_rng()

        # One flat batch: nominal poses followed by all perturbations, so the GPU sees a single
        # large call rather than N small ones.
        flat = [T]
        for t in T:
            flat.append(perturb_poses(t, n_perturb, sigma_pos, sigma_rot, rng=rng))
        labels = self.label(np.concatenate(flat, axis=0))

        exist = labels[: len(T)]
        robust = labels[len(T):].reshape(len(T), n_perturb).mean(axis=1)
        return exist, robust

    # ------------------------------------------------------------------ controls

    def run_controls(self, n_fk=64, rng=None):
        """Known-answer checks. Returns (ok, report). Callers MUST abort the run when not ok.

        Three controls, each catching a class the others cannot:

        - IDENTITY: forward kinematics of the robot's own rest configuration. The arm is
          demonstrably in that pose, so it must label reachable 100% of the time. This is the
          only control here that is sensitive to ORIENTATION, and it is the one that caught the
          xyzw/wxyz double-permutation (see `_mat_to_quat_xyzw`) after the other two had passed
          a quaternion scramble for two full runs.
        - FK SAMPLES: poses from random valid configurations, reachable by construction. Catches
          errors that only appear away from the rest pose, and measures how pessimistic the
          solver is across the workspace.
        - FAR TARGET: 6 m away, unreachable for any tabletop arm. Catches a target frame that is
          being ignored -- but is blind to orientation, since nothing at 6 m is reachable at any
          orientation.
        """
        rng = rng or np.random.default_rng(0)
        report = {}

        ident = self.rest_pose_target()
        report["identity_positive_rate"] = float(self.label(np.repeat(ident[None], 8, axis=0)).mean())

        fk = self.sample_fk_poses(n_fk, rng=rng)
        fk_labels = self.label(fk)
        report["fk_positive_rate"] = float(fk_labels.mean())
        report["fk_self_collision_reject_rate"] = float(getattr(self, "last_fk_reject_rate", 0.0))
        report["fk_sampled_joints"] = list(getattr(self, "last_fk_sampled_joints", []))

        far = np.repeat(np.eye(4)[None], 4, axis=0)
        far[:, 0, 3] = 6.0
        report["far_positive_rate"] = float(self.label(far).mean())

        ok = (report["identity_positive_rate"] == 1.0
              and report["fk_positive_rate"] >= 0.95
              and report["far_positive_rate"] == 0.0)
        report["ok"] = ok
        if self.verbose:
            status = "PASS" if ok else "FAIL"
            print("[IKTeacher] controls %s | identity %.0f%% (expect 100) | FK-reachable %.1f%% "
                  "(expect >=95) | 6m-away %.1f%% (expect 0) | %.0f%% of raw q self-collided"
                  % (status, 100 * report["identity_positive_rate"],
                     100 * report["fk_positive_rate"],
                     100 * report["far_positive_rate"],
                     100 * report["fk_self_collision_reject_rate"]), flush=True)
        return ok, report

    def rest_pose_target(self):
        """(4, 4) eef pose of the robot's current configuration, in the curobo base frame.

        Reachable by definition -- the robot is standing in it.
        """
        import omnigibson.lazy as lazy

        kin = self.mg.mg[self._emb].kinematics
        names = list(kin.joint_names)
        all_names = list(self.robot.joints.keys())
        rest = np.asarray(self._rest_q.cpu(), dtype=float)
        q = np.array([rest[all_names.index(n)] for n in names])

        cu_js = lazy.curobo.types.state.JointState(
            position=self.mg.tensor_args.to_device(th.tensor(q[None], dtype=th.float32)),
            joint_names=names)
        out = kin.compute_kinematics(cu_js)
        T = np.eye(4)
        T[:3, 3] = np.asarray(out.ee_position.cpu(), dtype=float)[0]
        T[:3, :3] = _quat_wxyz_to_mat(np.asarray(out.ee_quaternion.cpu(), dtype=float)[0])
        return T

    def sample_fk_poses(self, n, rng=None, reject_self_collision=True, oversample=3.0):
        """Forward kinematics of random VALID arm configurations -> poses on the reachable manifold.

        These carry the true orientation distribution of the reachable set, which uniform
        task-space sampling badly misrepresents: at a given position only a small, strongly
        structured family of orientations is achievable.

        `reject_self_collision` matters because the solver enforces self-collision and forward
        kinematics does not: a self-colliding q reaches a pose that IK will refuse unless some
        other configuration also reaches it. Emitting those as "guaranteed positives" would put
        label noise in the largest slice of the dataset, invisible downstream because nothing
        re-checks it.

        This is why the FK slice is a SAMPLING DISTRIBUTION here, not a free-label shortcut --
        `datagen` labels these poses with the teacher like every other slice. The proposal's
        Sec 12.1 treats FK samples as costless positives; that saves solves at the price of a
        label the pipeline cannot verify, and the cheaper thing to be wrong about is time.
        """
        import omnigibson.lazy as lazy

        emb = self._emb
        kin = self.mg.mg[emb].kinematics
        names = list(kin.joint_names)
        lim = kin.get_joint_limits()
        lo = np.asarray(lim.position[0].cpu(), dtype=float)
        hi = np.asarray(lim.position[1].cpu(), dtype=float)

        # Sample only joints IK is free to move. Measured on TidyBot/ARM_NO_TORSO this is a
        # no-op -- curobo strips locked joints from the active cspace, so `kin.joint_names` is
        # already just joint_1..joint_7 -- but it is not free to assume that for another robot or
        # embodiment, where a locked joint left in the cspace would silently generate FK poses
        # only reachable by moving something IK holds fixed.
        locked = set(self._locked_joint_names(kin))
        free = np.array([i for i, nm in enumerate(names) if nm not in locked], dtype=int)
        if len(free) == 0:
            raise RuntimeError("every joint in the %s cspace is locked -- cannot sample FK poses" % emb)
        self.last_fk_sampled_joints = [names[i] for i in free]

        rng = rng or np.random.default_rng()
        want = int(n * oversample) if reject_self_collision else n
        # Locked joints stay at the value IK will hold them at; only free joints vary.
        q = np.repeat(_locked_reference(lo, hi)[None], want, axis=0)
        q[:, free] = rng.uniform(lo[free], hi[free], size=(want, len(free)))

        self.last_fk_reject_rate = 0.0
        if reject_self_collision:
            keep = ~self._self_collides(q, names)
            # Measure the rate BEFORE truncating to n. Computing it from the kept length after
            # `[:n]` yields 1 - 1/oversample no matter what the collision checker said -- a
            # constant masquerading as a measurement, which is how "67% rejected" got quoted for
            # two runs.
            self.last_fk_reject_rate = float(1.0 - keep.mean())
            q = q[keep][:n]
            while len(q) < n:   # unlucky draw; top up rather than silently return fewer
                extra = np.repeat(_locked_reference(lo, hi)[None], want, axis=0)
                extra[:, free] = rng.uniform(lo[free], hi[free], size=(want, len(free)))
                extra = extra[~self._self_collides(extra, names)]
                if len(extra) == 0:
                    raise RuntimeError("every sampled configuration self-collides; cannot build "
                                       "an FK slice for %s" % emb)
                q = np.concatenate([q, extra])[:n]

        cu_js = lazy.curobo.types.state.JointState(
            position=self.mg.tensor_args.to_device(th.tensor(q, dtype=th.float32)),
            joint_names=names)
        out = kin.compute_kinematics(cu_js)
        pos = np.asarray(out.ee_position.cpu(), dtype=float)
        quat_wxyz = np.asarray(out.ee_quaternion.cpu(), dtype=float)

        T = np.repeat(np.eye(4)[None], len(q), axis=0)
        T[:, :3, 3] = pos
        for i, q_w in enumerate(quat_wxyz):
            T[i, :3, :3] = _quat_wxyz_to_mat(q_w)
        return T

    def _self_collides(self, q_arm, arm_joint_names):
        """(M, n_arm) arm configurations -> (M,) bool self-collision, via the batched checker.

        `check_collisions` wants a FULL robot joint vector, so each arm configuration is embedded
        into the rest pose. It also tests world collision with no way to disable it -- but the
        world model is only populated by `update_obstacles()`, which this module never calls, so
        under the empty-scene teacher the world is empty and every hit is attributable to the arm
        configuration itself. Reusing this with a generator whose obstacles HAVE been loaded would
        silently start rejecting configurations for hitting furniture, biasing the FK slice away
        from low reaches while still calling the rate "self-collision".
        """
        q_arm = np.atleast_2d(np.asarray(q_arm, dtype=float))
        full = th.stack([self._rest_q.clone() for _ in range(len(q_arm))])
        idx = [self._joint_index(n) for n in arm_joint_names]
        for j, col in enumerate(idx):
            if col is not None:
                full[:, col] = th.tensor(q_arm[:, j], dtype=full.dtype)
        hits = self.mg.check_collisions(full, self_collision_check=True, skip_obstacle_update=True)
        return np.asarray(hits.cpu(), dtype=bool).reshape(-1)

    def _joint_index(self, name):
        names = list(self.robot.joints.keys())
        return names.index(name) if name in names else None

    @staticmethod
    def _locked_joint_names(kin):
        """Joints CuRobo holds fixed for this embodiment.

        Read from the live kinematics config rather than the yaml, so it stays correct if the
        config changes underneath us -- the box has no version control and its configs have been
        edited out from under this code before.
        """
        try:
            return [str(x) for x in kin.kinematics_config.lock_jointstate.joint_names]
        except Exception:
            return []


def _locked_reference(lo, hi):
    """Reference values for locked joints: midpoint of their limits.

    Only the locked entries of this vector are used (free entries are overwritten by sampling),
    and for a locked joint CuRobo substitutes its own locked value anyway -- so the midpoint is a
    harmless placeholder that keeps the vector well-formed.
    """
    return 0.5 * (np.asarray(lo, dtype=float) + np.asarray(hi, dtype=float))


def _mat_to_quat_xyzw(R):
    """(3,3) -> (x, y, z, w), the ordering `compute_trajectories` expects.

    NOT wxyz, despite curobo itself using wxyz internally. The wrapper applies
    `target_quat[:, [3, 0, 1, 2]]` unconditionally -- it sits OUTSIDE the `if not is_local`
    block (curobo.py ~line 720), so it runs for local targets too. Handing it wxyz gets the
    permutation applied twice, turning (w,x,y,z) into (z,w,x,y): a valid-looking unit quaternion
    naming a completely different rotation.

    This produced the single most misleading failure of the module. Positions were passed through
    correctly, so targets landed in the right place with a scrambled orientation, and IK succeeded
    whenever that scrambled orientation happened to be reachable. The result was a plausible ~72%
    "reachability" rate that no geometric story explains, and it was invisible to the 6-m-away
    control (unreachable at any orientation). Only the identity control -- FK of the robot's own
    rest pose, which must label 100% and labeled 50% -- could see it.
    """
    from .frames import mat_to_quat

    return mat_to_quat(R)


def _quat_wxyz_to_mat(q):
    from .frames import quat_to_mat

    w, x, y, z = q
    return quat_to_mat([x, y, z, w])
