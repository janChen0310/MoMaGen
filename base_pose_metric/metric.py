"""Score a candidate mobile-base pose: distance, visibility, reachability, collision.

Scene-agnostic and self-contained -- depends only on OmniGibson/Isaac Sim, never on the task or
scene it is used in. Drop this package into any OmniGibson project, hand it a robot, and it will
discover that robot's base joints, cameras and end-effectors on its own.

Why this exists: "where should the base stand" is the decision that quietly governs whether a
mobile-manipulation demo succeeds, and when it is wrong the symptom appears far downstream as an
unexplained motion-planning failure. This turns it into four numbers you can inspect directly.

The central trick is that NOTHING HERE MOVES THE ROBOT. Both CuRobo entry points accept a
hypothetical configuration:
  * `check_collisions(q, ...)` takes an (N, D) batch of joint configurations -- and for a
    holonomic base the base joints ARE part of that vector, so writing (x, y, yaw) into those
    columns places the collision model at a candidate pose directly.
  * IK takes `initial_joint_pos`, which sets the *locked* joints, so passing candidate base values
    makes the solver answer "could the arm reach this from over there?".
So a sweep costs no sim steps and no state save/restore.
"""
import math
import time

import numpy as np
import torch as th

from .geometry import (
    aabb_sample_points,
    base_pose_to_matrix,
    base_poses_to_matrices,
    in_image_batched,
    intrinsics_from_camera_params,
    look_at_rotation,
    matrix_to_pose,
    pose_to_matrix,
    score_from_components,
    visible_fraction,
)


class BasePoseMetric:
    """Scores candidate base poses for a given manipulation target.

    Args:
        robot: an OmniGibson robot with a holonomic base (needs `base_joint_names`).
        motion_generator: an existing `CuRoboMotionGenerator` to reuse. Built if omitted, which
            costs several seconds and a chunk of VRAM -- pass one in if you already have it.
        camera_names: substrings selecting which cameras count for visibility. Default: all
            `VisionSensor`s on the robot.
        weights: soft-term weights, e.g. `{"distance": 0.3, "visibility": 0.7}`.
        distance_band: (lo, hi) metres. Poses inside score 1.0 for distance; outside decays.
        ignore_objects: objects excluded from collision checks. **Recorded and reported**, because
            an exclusion silently turns "collides" into "clear" and that is invisible in results.
        check_occlusion: ray-test each visible sample point for line of sight. Without it,
            visibility means only "inside the frustum", which reports objects behind walls as
            fully visible.
        ik_predictor: optional `kineready.reward.ReadinessReward`. When set, `evaluate` reports a
            learned P(IK feasible) instead of solving. IMPORTANT: a probability is not a joint
            configuration, so the reach-collision check has nothing to test and is reported as
            None rather than False -- "not checked" and "checked and clear" are different claims
            and must not be conflated. Use exact IK, or re-verify the top candidates with it,
            whenever reach-collision matters.
        ik_threshold: p_kin at or above this counts as `ik_ok` when using the predictor.
    """

    def __init__(self, robot, motion_generator=None, camera_names=None, weights=None,
                 distance_band=(0.4, 0.9), ignore_objects=None, check_occlusion=True,
                 verbose=True, ik_predictor=None, ik_threshold=0.5):
        self.robot = robot
        # Optional learned replacement for the exact IK term. Exact IK measured 0.34 s/pose and is
        # 99.7% of this metric's cost; the surrogate answers in microseconds, which is what makes
        # the metric usable as a dense online reward. See kineready/.
        self.ik_predictor = ik_predictor
        self.ik_threshold = float(ik_threshold)
        self.weights = weights
        self.distance_band = tuple(distance_band)
        self.ignore_objects = list(ignore_objects) if ignore_objects else None
        # Frustum containment alone over-reports badly in a furnished room (see
        # _unoccluded_fraction). Set False only if you want the pure-geometry answer.
        self.check_occlusion = check_occlusion
        self.verbose = verbose

        self._joint_names = list(robot.joints.keys())
        base_names = getattr(robot, "base_joint_names", None)
        if not base_names:
            raise ValueError(
                "robot has no `base_joint_names`; this metric needs a holonomic base whose "
                "(x, y, yaw) are joints, so a candidate pose can be evaluated without moving it")
        self._base_joint_names = list(base_names)
        missing = [n for n in self._base_joint_names if n not in self._joint_names]
        if missing:
            raise ValueError("base joints %s are not in robot.joints" % missing)
        self._base_idx = [self._joint_names.index(n) for n in self._base_joint_names]
        # Which of (x, y, yaw) each base joint carries -- name-based so a robot ordering them
        # differently still works.
        self._slot = {}
        for i, n in enumerate(self._base_joint_names):
            for key, tag in (("x", "_x_"), ("y", "_y_"), ("yaw", "_rz_")):
                if tag in n:
                    self._slot[key] = self._base_idx[i]
        if set(self._slot) != {"x", "y", "yaw"}:
            raise ValueError("could not identify x/y/rz among base joints %s" % self._base_joint_names)

        self.mg = motion_generator if motion_generator is not None else self._build_motion_generator()
        self._cameras = self._discover_cameras(camera_names)
        self._rest_q = robot.get_joint_positions().clone()

        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
        # ARM_NO_TORSO first: torso-less arms (TidyBot) only solve under that embodiment, and a
        # wrong choice fails every solve while looking exactly like "the target is unreachable".
        self._ik_embodiments = [CuRoboEmbodimentSelection.ARM_NO_TORSO,
                                CuRoboEmbodimentSelection.ARM,
                                CuRoboEmbodimentSelection.DEFAULT]
        self._ik_errors = {}

        if verbose:
            self._report()

    # ------------------------------------------------------------------ setup

    def _build_motion_generator(self):
        from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
        import omnigibson as og

        scene_model = getattr(getattr(self.robot, "scene", None), "scene_model", None)
        return CuRoboMotionGenerator(
            robot=self.robot, batch_size=1, use_cuda_graph=False,
            scene_model=scene_model.lower() if isinstance(scene_model, str) else "empty",
        )

    def _discover_cameras(self, camera_names):
        """Find the robot's cameras and cache each one's FIXED transform from the robot root.

        Rigidly-mounted cameras move with the base, so their world pose at a candidate base pose is
        T_base(candidate) @ T_mount. Caching T_mount once is what lets visibility be evaluated for
        an arbitrary pose without touching the simulator.
        """
        from omnigibson.sensors.vision_sensor import VisionSensor

        root_pos, root_quat = self.robot.get_position_orientation()
        T_root = pose_to_matrix(_np(root_pos), _np(root_quat))
        T_root_inv = np.linalg.inv(T_root)

        cams = {}
        for name, sensor in self.robot.sensors.items():
            if not isinstance(sensor, VisionSensor):
                continue
            if camera_names and not any(sub in name for sub in camera_names):
                continue
            cpos, cquat = sensor.get_position_orientation()
            T_mount = T_root_inv @ pose_to_matrix(_np(cpos), _np(cquat))
            cams[name] = {"mount": T_mount,
                          "K": self._intrinsics(sensor),
                          "width": int(sensor.image_width),
                          "height": int(sensor.image_height)}
        if not cams:
            raise ValueError("no VisionSensor found on the robot (camera_names=%r)" % (camera_names,))
        return cams

    @staticmethod
    def _intrinsics(sensor):
        """Prefer the sensor's own K; fall back to the analytic pinhole form.

        `intrinsic_matrix` reads the render product's projection matrix, which is only populated
        once the sensor has rendered -- so a metric built before the first render would otherwise
        fail on a detail that has nothing to do with base poses.
        """
        try:
            return _np(sensor.intrinsic_matrix)
        except Exception:
            return intrinsics_from_camera_params(
                sensor.focal_length, sensor.horizontal_aperture,
                sensor.image_width, sensor.image_height)

    def _report(self):
        print("[BasePoseMetric] base joints: %s" % self._base_joint_names, flush=True)
        for n, c in self._cameras.items():
            print("[BasePoseMetric] camera %-42s %dx%d  fx=%.1f"
                  % (n, c["width"], c["height"], c["K"][0, 0]), flush=True)
        if self.ignore_objects:
            print("[BasePoseMetric] WARNING: %d object(s) EXCLUDED from collision checks -- poses "
                  "touching them will report clear: %s"
                  % (len(self.ignore_objects), [o.name for o in self.ignore_objects]), flush=True)
        # Self-check: the robot standing where it already is should not be in collision. If it is,
        # the collision model is broken (e.g. base spheres sunk below the floor plane) and every
        # result from this metric is meaningless -- fail loudly rather than return confident zeros.
        # It must build the SAME world `evaluate` uses, exclusions included, or it reports a
        # collision that evaluate() will not see and the warning becomes a false alarm.
        try:
            q = self._rest_q.unsqueeze(0)
            self.mg.update_obstacles(ignore_objects=self.ignore_objects)
            hit = bool(self.mg.check_collisions(q, skip_obstacle_update=True)[0])
            if hit:
                print("[BasePoseMetric] WARNING: the robot reports COLLISION at its own current "
                      "pose. The collision model is probably wrong (commonly: base spheres placed "
                      "below the floor plane). Treat all results as suspect.", flush=True)
            else:
                print("[BasePoseMetric] self-check OK: current pose is collision-free", flush=True)
        except Exception as exc:
            print("[BasePoseMetric] self-check could not run: %s" % exc, flush=True)

    # ------------------------------------------------------------- evaluation

    def evaluate(self, base_poses, target, eef_pose=None, attached_obj=None,
                 arm_config=None, skip_ik_if_colliding=True, use_learned_ik=None):
        """Score one or many candidate base poses against a manipulation target.

        Args:
            base_poses: (3,) or (N, 3) of (x, y, yaw) in world coordinates.
            target: the object to be manipulated (anything with `.aabb`).
            eef_pose: optional exact (pos, quat) world eef target. Derived from the object if None.
            attached_obj: dict of {eef_link_name: object} already grasped, included in collision.
            arm_config: joint values for the "standing still" check. Defaults to the current arm.
            skip_ik_if_colliding: don't spend IK on poses already known to collide. IK is the
                expensive term.
            use_learned_ik: force the learned predictor on/off. Defaults to True when one was
                supplied.

        Returns:
            list of dicts, one per candidate.
        """
        poses = np.atleast_2d(np.asarray(base_poses, dtype=float))
        if poses.shape[1] != 3:
            raise ValueError("base_poses must be (N, 3) of (x, y, yaw), got %s" % (poses.shape,))

        lo, hi = target.aabb
        pts = aabb_sample_points(_np(lo), _np(hi))
        centre = pts[-1]

        if eef_pose is None:
            eef_pose = self.derive_eef_pose(target)

        base_q = self._rest_q if arm_config is None else arm_config
        q_static = th.stack([base_q.clone() for _ in range(len(poses))])
        for i, (x, y, yaw) in enumerate(poses):
            q_static[i, self._slot["x"]] = float(x)
            q_static[i, self._slot["y"]] = float(y)
            q_static[i, self._slot["yaw"]] = float(yaw)

        # One world rebuild for the whole sweep; every later call skips it.
        t = {"n_poses": len(poses)}
        t0 = time.perf_counter()
        self.mg.update_obstacles(ignore_objects=self.ignore_objects)
        t["update_obstacles"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        coll_static = _np(self.mg.check_collisions(q_static, skip_obstacle_update=True)).astype(bool)
        t["collision_static_batched"] = time.perf_counter() - t0

        # Frustum test for EVERY pose and camera up front: one vectorized call per camera rather
        # than one per pose per camera. The per-pose loop below then only pays for raycasts.
        t0 = time.perf_counter()
        T_bases = base_poses_to_matrices(poses)
        T_cams, in_img = {}, {}
        for name, c in self._cameras.items():
            T_cams[name] = T_bases @ c["mount"]
            in_img[name] = in_image_batched(c["K"], T_cams[name], pts, c["width"], c["height"])
        t["visibility_frustum_batched"] = time.perf_counter() - t0
        t["visibility"] = t["ik"] = t["collision_reach"] = 0.0

        learned = self.ik_predictor is not None if use_learned_ik is None else bool(use_learned_ik)
        if learned and self.ik_predictor is None:
            raise ValueError("use_learned_ik=True but no ik_predictor was supplied")
        p_kin = None
        if learned:
            # One batched call for every candidate, before the loop -- the whole point of the
            # surrogate is that N poses cost about as much as one.
            _t0 = time.perf_counter()
            T_target = pose_to_matrix(_np(eef_pose[0]), _np(eef_pose[1]))
            p_kin = np.asarray(self.ik_predictor.score(poses, T_target), dtype=float)
            t["ik"] += time.perf_counter() - _t0

        results = []
        for i, (x, y, yaw) in enumerate(poses):
            _t0 = time.perf_counter()
            vis = {}
            for name in self._cameras:
                frac = float(in_img[name][i].mean())
                if frac > 0.0 and self.check_occlusion:
                    frac *= self._unoccluded_fraction(T_cams[name][i, :3, 3], pts, target)
                vis[name] = frac
            vis["any"] = max(vis.values()) if vis else 0.0
            t["visibility"] += time.perf_counter() - _t0

            distance = float(np.linalg.norm(np.array([x, y]) - centre[:2]))

            ik_ok, coll_reach = False, False
            if learned:
                ik_ok = bool(p_kin[i] >= self.ik_threshold)
                # No joint solution exists to test, so this is "unknown", not "clear".
                coll_reach = None
            elif not (skip_ik_if_colliding and coll_static[i]):
                _t0 = time.perf_counter()
                sol = self._solve_ik(q_static[i], eef_pose)
                t["ik"] += time.perf_counter() - _t0
                ik_ok = sol is not None
                if ik_ok:
                    q_reach = q_static[i].clone()
                    q_reach[: len(sol)] = sol.to(q_reach.device, q_reach.dtype)[: len(q_reach)]
                    for key in ("x", "y", "yaw"):          # keep the candidate base pose
                        q_reach[self._slot[key]] = q_static[i, self._slot[key]]
                    _t0 = time.perf_counter()
                    coll_reach = bool(self.mg.check_collisions(
                        q_reach.unsqueeze(0), skip_obstacle_update=True,
                        attached_obj=attached_obj)[0])
                    t["collision_reach"] += time.perf_counter() - _t0

            score, feasible = score_from_components(
                distance, vis["any"], ik_ok, bool(coll_static[i]), bool(coll_reach),
                self.distance_band, self.weights)
            results.append({
                "base_pose": (float(x), float(y), float(yaw)),
                "distance": distance,
                "visibility": vis,
                "ik_ok": bool(ik_ok),
                "ik_source": "learned" if learned else "exact",
                "p_kin": None if p_kin is None else float(p_kin[i]),
                "collision_static": bool(coll_static[i]),
                "collision_reach": None if coll_reach is None else bool(coll_reach),
                "feasible": bool(feasible),
                "score": float(score),
            })
        t["total"] = (t["update_obstacles"] + t["collision_static_batched"]
                      + t.get("visibility_frustum_batched", 0.0)
                      + t["visibility"] + t["ik"] + t["collision_reach"])
        self.last_timings = t
        return results

    def _unoccluded_fraction(self, cam_pos, points, target):
        """Fraction of `points` with clear line of sight from the camera.

        Pure frustum containment answers "is it in the field of view", which is NOT the useful
        question in a furnished room: a 5x5 render grid showed roughly a third of poses reporting
        vis=1.00 while the camera was looking at a wall with the object behind it -- one tile was a
        completely black image scoring 1.00. A PhysX ray per sample point costs microseconds (far
        cheaper than rendering a segmentation mask) and turns "in the frustum" into "actually
        visible".

        A hit belonging to the target itself counts as reaching it: the ray naturally terminates on
        the object's own near surface.

        Hits on the ROBOT'S OWN links are also ignored. Measured on TidyBot: 94% of blocked rays
        terminated on the robot's own base or wrist links, because the camera sits on the chassis
        and the arm is parked in front of it. Counting that as occlusion penalizes a base pose for
        the pose of an arm that will move as soon as it reaches -- it describes the parking
        configuration, not the standing position being scored.
        """
        import omnigibson as og

        origin = np.asarray(cam_pos, dtype=float)
        target_paths = self._target_link_paths(target)
        own_paths = self._own_link_paths()
        clear = 0
        for p in points:
            d = np.asarray(p, dtype=float) - origin
            dist = float(np.linalg.norm(d))
            if dist < 1e-6:
                clear += 1
                continue
            try:
                hit = og.sim.psqi.raycast_closest(
                    origin=origin.tolist(), dir=(d / dist).tolist(), distance=dist)
            except Exception:
                return 1.0        # no ray API -> fall back to frustum-only rather than lie
            if not hit or not hit.get("hit", False):
                clear += 1        # nothing in the way at all
                continue
            body = hit.get("rigidBody", "") or hit.get("collision", "")
            if any(str(body).startswith(tp) or str(tp).startswith(str(body)) for tp in target_paths):
                clear += 1        # the ray landed on the target itself
            elif any(str(body).startswith(op) for op in own_paths):
                clear += 1        # the robot's own body -- see the docstring
            elif hit.get("distance", dist) >= dist - 1e-3:
                clear += 1        # hit is at/behind the sample point, so nothing occludes it
        return float(clear) / float(len(points))

    def _target_link_paths(self, target):
        """Prim paths of the target's links, cached per target.

        Rebuilt on every call before this, which meant once per pose PER CAMERA -- 3000 set
        comprehensions in a 1000-pose sweep, and the dominant Python cost inside the occlusion test.
        """
        if getattr(self, "_target_paths_cache", None) is None:
            self._target_paths_cache = {}
        key = getattr(target, "name", id(target))
        if key not in self._target_paths_cache:
            self._target_paths_cache[key] = tuple(
                {l.prim_path for l in target.links.values()}) if hasattr(target, "links") else tuple()
        return self._target_paths_cache[key]

    def _own_link_paths(self):
        """Prim paths of the robot's own links, cached. Used to discount self-occlusion."""
        if getattr(self, "_own_paths_cache", None) is None:
            try:
                self._own_paths_cache = tuple(sorted(
                    {l.prim_path for l in self.robot.links.values()}))
            except Exception:
                self._own_paths_cache = tuple()
        return self._own_paths_cache

    def derive_eef_pose(self, target, standoff=0.12, approach=(0.0, 0.0, -1.0)):
        """A default top-down grasp above the target, used when no exact pose is supplied.

        Deliberately simple: hovering above the AABB top by `standoff`, gripper pointing down. Real
        tasks usually have a specific grasp (the trash task's drop is tilted ~14 deg off vertical),
        so pass `eef_pose` explicitly when the exact pose matters -- this is only a sane default.
        """
        lo, hi = target.aabb
        lo, hi = _np(lo), _np(hi)
        pos = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, hi[2] + standoff])
        R = look_at_rotation(approach)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        _, quat = matrix_to_pose(T)
        return (th.tensor(pos, dtype=th.float32), th.tensor(quat, dtype=th.float32))

    def _solve_ik(self, q_candidate, eef_pose):
        """IK from a hypothetical base pose.

        `initial_joint_pos` sets CuRobo's LOCKED joints, and for an arm embodiment the base joints
        are locked -- so handing it a configuration carrying the candidate (x, y, yaw) asks exactly
        "could the arm reach this target from over there?" without moving anything.

        Note this cannot be batched across base poses: locked joints are global to the call, and
        each candidate needs different ones. IK is therefore the per-pose cost of a sweep.
        """
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

        eef_link = list(self.robot.eef_link_names.values())[0]
        pos, quat = eef_pose
        target_pos = {eef_link: th.stack([th.as_tensor(pos, dtype=th.float32)])}
        target_quat = {eef_link: th.stack([th.as_tensor(quat, dtype=th.float32)])}

        # Embodiment matters: a torso-less arm (TidyBot) must use ARM_NO_TORSO -- asking for ARM
        # silently fails every solve. Try the configured one, then fall back, so this works across
        # robots without the caller knowing which their robot has.
        for emb in self._ik_embodiments:
            try:
                successes, joint_states = self.mg.compute_trajectories(
                    target_pos=target_pos, target_quat=target_quat,
                    initial_joint_pos=q_candidate,
                    is_local=False, max_attempts=5, timeout=10.0, ik_fail_return=5,
                    enable_finetune_trajopt=False, finetune_attempts=0,
                    return_full_result=False, success_ratio=1.0,
                    skip_obstacle_update=True, ik_only=True, emb_sel=emb,
                )
            except Exception as exc:
                # Record WHY rather than collapsing a misconfiguration into "unreachable" --
                # those two look identical downstream and mean completely different things.
                self._ik_errors[str(emb)] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
                continue
            idx = th.where(successes)[0]
            if len(idx) == 0:
                self._ik_errors[str(emb)] = "no IK solution"
                continue
            self._ik_embodiments = [emb]      # lock in the one that works
            return self.mg.path_to_joint_trajectory(
                joint_states[int(idx[0])], get_full_js=False, emb_sel=emb).cpu()
        return None

    @property
    def ik_diagnostics(self):
        """Why IK last failed, per embodiment. Empty when IK has been succeeding."""
        return dict(self._ik_errors)


def _np(x):
    """Tensor/array -> numpy, tolerating CUDA tensors."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=float)
