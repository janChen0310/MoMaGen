"""
Robot configuration utilities for MoMaGen.
"""
import torch as th
from omnigibson.robots.r1 import R1
from omnigibson.robots.tiago import Tiago
from omnigibson.robots.tidybot import TidyBot

# Robot types
ROBOT_R1 = "R1"
ROBOT_TIAGO = "Tiago"
ROBOT_TIDYBOT = "TidyBot"

# Default robot type
DEFAULT_ROBOT_TYPE = ROBOT_R1

# Robot-specific link names
ROBOT_LINK_NAMES = {
    ROBOT_R1: {
        "torso": "torso_link4"
    },
    ROBOT_TIAGO: {
        "torso": "torso_lift_link"
    },
    ROBOT_TIDYBOT: {
        "torso": None  # no articulated trunk
    }
}

def get_torso_link_name(robot):
    """Get the torso link name for a given robot instance (None if the robot has no trunk)."""
    if isinstance(robot, Tiago):
        return ROBOT_LINK_NAMES[ROBOT_TIAGO]["torso"]
    elif isinstance(robot, R1):
        return ROBOT_LINK_NAMES[ROBOT_R1]["torso"]
    elif isinstance(robot, TidyBot):
        return ROBOT_LINK_NAMES[ROBOT_TIDYBOT]["torso"]
    else:
        raise ValueError(f"Robot type {type(robot)} not supported")

def get_robot_type_from_instance(robot):
    """Get robot type string from robot instance."""
    if isinstance(robot, Tiago):
        return ROBOT_TIAGO
    elif isinstance(robot, R1):
        return ROBOT_R1
    elif isinstance(robot, TidyBot):
        return ROBOT_TIDYBOT
    else:
        raise ValueError(f"Robot type {type(robot)} not supported")

def get_tiago_config():
    """Get Tiago robot configuration."""
    reset_joint_pos = th.tensor([
        0.0000,  0.0000,  0.0003,  0.0000,  0.0000,
       -0.0000,  0.3500,  0.8637,      0.8401,      0.0000,
        -0.8935,     -0.8862,     -0.4500,      1.8286,      1.8267,
         1.1199,      1.1741,      1.1771,      1.1749,     -1.4134,
        -1.2823, -1.0891, -1.0891,  0.0450,  0.0450,
        0.0450,  0.0450
    ])

    controller_config = {
        'arm_left': {
            'name': 'JointController',
            'motor_type': 'position',
            'pos_kp': 150,
            'command_input_limits': None,
            'command_output_limits': None,
            'use_impedances': False,
            'use_delta_commands': False
        },
        'arm_right': {
            'name': 'JointController',
            'motor_type': 'position',
            'pos_kp': 150,
            'command_input_limits': None,
            'command_output_limits': None,
            'use_impedances': False,
            'use_delta_commands': False
        },
        'gripper_left': {
            'name': 'MultiFingerGripperController',
            'mode': 'smooth',
            'command_input_limits': 'default',
            'command_output_limits': 'default'
        },
        'gripper_right': {
            'name': 'MultiFingerGripperController',
            'mode': 'smooth',
            'command_input_limits': 'default',
            'command_output_limits': 'default'
        },
        'base': {
            'name': 'HolonomicBaseJointController',
            'motor_type': 'velocity',
            'vel_kp': 150,
            'command_input_limits': [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            'command_output_limits': [[-1.5, -1.5, -3.1415927], [1.5, 1.5, 3.1415927]],
            'use_impedances': False
        },
        'trunk': {
            'name': 'JointController',
            'motor_type': 'position',
            'pos_kp': 150,
            'command_input_limits': None,
            'command_output_limits': None,
            'use_impedances': False,
            'use_delta_commands': False
        },
        'camera': {
            'name': 'JointController',
            'motor_type': 'position',
            'use_impedances': False,
            'use_delta_commands': False
        }
    }

    return reset_joint_pos, controller_config

def configure_tiago_env_meta(env_meta):
    """Configure environment metadata for Tiago robot."""
    env_meta["env_kwargs"]["robots"][0]["type"] = "Tiago"
    reset_joint_pos, controller_config = get_tiago_config()
    env_meta["env_kwargs"]["robots"][0]["reset_joint_pos"] = reset_joint_pos.tolist()
    env_meta["env_kwargs"]["robots"][0]["controller_config"] = controller_config

    if env_meta["env_kwargs"]["scene"]["scene_model"] == "house_single_floor":
        env_meta["env_kwargs"]["scene"]["load_room_types"] = ["kitchen"]

    return env_meta

def get_tidybot_config():
    """
    Get TidyBot++ robot configuration.

    Same controller scheme as Tiago/R1: absolute-position JointController for the
    arm (required by MoMaGen's q_to_action MP replay and Jacobian-QP IK), velocity
    HolonomicBaseJointController for the base, MultiFingerGripperController for
    the 2F-85 (q=0 open, q~0.82 closed for both driven joints; explicit
    open/closed qpos remove any sign ambiguity: action >= 0 opens, < 0 closes).
    """
    controller_config = {
        'arm_0': {
            'name': 'JointController',
            'motor_type': 'position',
            'pos_kp': 150,
            'command_input_limits': None,
            'command_output_limits': None,
            'use_impedances': False,
            'use_delta_commands': False
        },
        'gripper_0': {
            'name': 'MultiFingerGripperController',
            'mode': 'smooth',
            # Hand-E prismatic fingers (range [0, 0.025] m): q=0 fully closed (fingers
            # together), q=0.025 fully open (max 25 mm stroke). (Was stale 2F-85 values
            # open=[0,0]/closed=[0.82], which are out of range and inverted for Hand-E.)
            'open_qpos': [0.025, 0.025],
            'closed_qpos': [0.0, 0.0],
            'command_input_limits': 'default',
            'command_output_limits': 'default'
        },
        'base': {
            'name': 'HolonomicBaseJointController',
            'motor_type': 'velocity',
            'vel_kp': 150,
            'command_input_limits': [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            'command_output_limits': [[-1.5, -1.5, -3.1415927], [1.5, 1.5, 3.1415927]],
            'use_impedances': False
        },
    }

    # reset_joint_pos is intentionally not specified: TidyBot's class defaults
    # (untucked = Gen3 retract + open gripper, built from joint-name index maps)
    # are order-correct for the imported USD, unlike a hand-written flat list.
    return None, controller_config

def configure_tidybot_env_meta(env_meta):
    """Configure environment metadata for the TidyBot++ robot (swaps the robot in
    an existing task env_meta, e.g. the R1 pick_cup one)."""
    robot_kwargs = env_meta["env_kwargs"]["robots"][0]
    robot_kwargs["type"] = "TidyBot"
    _, controller_config = get_tidybot_config()
    robot_kwargs["controller_config"] = controller_config
    # Drop R1-specific overrides that don't transfer across embodiments
    robot_kwargs.pop("reset_joint_pos", None)
    robot_kwargs["default_reset_mode"] = "untuck"
    # Force physical grasping. The source (R1) env_meta carries grasping_mode
    # "assisted"/"sticky"; inheriting that drives OmniGibson's assisted-grasping path,
    # which (a) freezes the gripper on a false grasp (breaks repeated open/close) and
    # (b) crashes for TidyBot (_find_gripper_raycast_collisions KeyError on the
    # phantom-arm "0" key). TidyBot is a real parallel-jaw gripper -> physical mode.
    robot_kwargs["grasping_mode"] = "physical"

    return env_meta