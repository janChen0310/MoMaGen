"""A minimal OmniGibson environment holding nothing but TidyBot on a floor plane.

The KineReady label is *scene-free by definition* -- `ik_world_collision_check=False` means the
solver is told to ignore world geometry entirely. Loading `house_single_floor` to produce those
labels therefore buys nothing and costs a great deal:

  * ~16 GB of the 24 GB card goes to RTX scene geometry, leaving so little for CuRobo that the
    motion generator OOMs during its own constructor warmup at batch_size=8. That directly caps
    IK throughput, which is the entire quantity Stage 0 exists to measure.
  * ~2 min of scene loading per run.
  * A dependency on the trash-task hdf5, which contradicts the whole point of the model: it is
    object- and task-agnostic, so its training data must not come from one task's scene.

An empty scene fixes all three. The robot's kinematics, joint limits, and self-collision spheres
-- the only things a scene-free label depends on -- are properties of the robot, not the scene.

The one thing that does change: `HOLONOMIC_BASE_PRISMATIC_JOINT_LIMIT` is keyed by scene, and
"empty" maps to +-5 m instead of the house's box. That affects only the BASE embodiment's
prismatic joints, which are LOCKED for ARM_NO_TORSO and never enter an IK query here.
"""
import numpy as np


def make_teacher_env(robot_type="TidyBot", position=(0.0, 0.0, 0.0), device_gpu_id=None):
    """Build (env, robot) with an empty scene and no cameras. Caller owns `og.shutdown()`.

    `obs_modalities=[]` is load-bearing, not tidiness: it stops OmniGibson from creating the
    robot's VisionSensors, and with them the render products that dominate VRAM. Nothing in the
    teacher looks at an image.
    """
    import omnigibson as og
    from omnigibson.macros import gm

    gm.HEADLESS = True
    # Object states / transition rules simulate cooking, wetting, slicing and friends. There are
    # no objects here, so they are pure overhead.
    gm.ENABLE_OBJECT_STATES = False
    gm.ENABLE_TRANSITION_RULES = False

    from momagen.utils.robot_config import get_tidybot_config

    _, controller_config = get_tidybot_config()

    cfg = {
        "env": {"action_frequency": 30, "physics_frequency": 120},
        "scene": {"type": "Scene", "use_floor_plane": True},
        "robots": [{
            "type": robot_type,
            "name": "robot0",
            "obs_modalities": [],
            "position": list(position),
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "controller_config": controller_config,
            "default_reset_mode": "untuck",
            "grasping_mode": "physical",
            # TidyBot's imported collision meshes overlap slightly between adjacent links; leaving
            # this True makes the robot vibrate at rest. Same reason as the task envs.
            "self_collisions": False,
        }],
    }
    if device_gpu_id is not None:
        cfg["env"]["device"] = "cuda:%d" % device_gpu_id

    env = og.Environment(configs=cfg)
    env.reset()
    robot = env.robots[0]
    # Park the robot at its rest configuration and let physics settle, so `get_joint_positions()`
    # returns the pose the teacher will lock its base and fingers at.
    for _ in range(5):
        og.sim.step()
    return env, robot


def scene_free_report(robot):
    """One-line provenance for a run log: what the labels are actually a function of."""
    q = robot.get_joint_positions()
    return {
        "robot": type(robot).__name__,
        "n_joints": int(len(q)),
        "rest_q": np.asarray(q.cpu(), dtype=float).round(4).tolist(),
        "scene": "empty (Scene + floor plane)",
    }
