"""
MoMaGen environment interface classes for OmniGibson environments.
Refactored to use configuration-driven tasks instead of hardcoded classes.
"""
import os
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass, field
import cvxpy as cp
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.object_states import *
from omnigibson.controllers import ControlType
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

from momagen.env_interfaces.base import MG_EnvInterface
from momagen.datagen.datagen_info import DatagenInfo


# --- GUARD: PhysX's PxArticulationJointReducedCoordinate.setDriveTarget rejects revolute drive
# targets outside [-2pi, 2pi]. TidyBot's Kinova continuous joints (notably joint_3, dof 8) get
# targets just over 2pi from the controller/normalization path (verified via JC_DRIVE_OOB:
# joint_3 -> 6.288/6.331); upstream clamps (q_to_action, the replay QP) don't fully prevent it.
# Unchecked, the flood of PhysX errors destabilizes the articulation (physics view -> None -> sim
# stops). Clamp revolute-joint drive targets to [-2pi, 2pi] at the final sink before PhysX. ---
_JC_ORIG_SJPT = ControllableObjectViewAPI.set_joint_position_targets.__func__
_JC_REV_DOF = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}  # TidyBot revolute dofs: base rx,ry,rz + arm joint_1..7
_JC_LIM = 2.0 * np.pi - 0.02
def _jc_clamp_set_joint_position_targets(cls, prim_path, positions, indices):
    try:
        if "tidybot" in str(prim_path):
            idx_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            rev = [k for k, di in enumerate(idx_list) if int(di) in _JC_REV_DOF]
            if rev:
                positions = positions.clone() if hasattr(positions, "clone") else positions.copy()
                for k in rev:
                    v = float(positions[k])
                    if v > _JC_LIM:
                        positions[k] = _JC_LIM
                    elif v < -_JC_LIM:
                        positions[k] = -_JC_LIM
    except Exception:
        pass
    return _JC_ORIG_SJPT(cls, prim_path, positions, indices)
ControllableObjectViewAPI.set_joint_position_targets = classmethod(_jc_clamp_set_joint_position_targets)


@dataclass
class TaskConfig:
    """Configuration for a task, defining objects and termination signals."""
    name: str
    tracked_objects: Dict[str, str] = field(default_factory=dict)  # logical_name -> registry_name
    termination_signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # signal_name -> config
    robot_specific_objects: Dict[str, Dict[str, str]] = field(default_factory=dict)  # robot_type -> object_mapping
    bimanual: bool = True  # Whether this task uses bimanual interface


class OmniGibsonInterface(MG_EnvInterface):
    """
    MoMaGen environment interface base class for basic omnigibson environments.
    """

    # Note: base simulator interface class must fill out interface type as a class property
    INTERFACE_TYPE = "omnigibson"

    def __init__(self, env, task_config: TaskConfig = None):
        super(OmniGibsonInterface, self).__init__(env)
        self.task_config = task_config
        self.robot = env.robots[0]
        self._setup_arm_controller()

    def _setup_arm_controller(self):
        """
        Sets up the arm controller for the robot. This is necessary to know where the arm command
        starts and ends in the action vector.
        """
        start_idx = 0
        end_idx = None
        arm_controller = None
        for controller_type, controller in self.robot.controllers.items():
            if controller_type != f"arm_{self.robot.default_arm}":
                start_idx += controller.command_dim
            else:
                end_idx = start_idx + controller.command_dim
                arm_controller = controller
                break
        assert end_idx is not None and arm_controller is not None
        self.arm_controller = arm_controller

    def get_robot_eef_pose(self):
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Returns:
            pose (np.array): 4x4 eef pose matrix
        """
        return self.get_object_pose(self.robot.eef_links[self.robot.default_arm])

    def get_object_pose(self, obj):
        """
        Returns 4x4 object pose given the name of the object and the type.

        Args:
            obj (BaseObject): OG object

        Returns:
            obj_pose (np.array): 4x4 object pose
        """
        return T.pose2mat(obj.get_position_orientation())

    def _get_object_by_name(self, name: str):
        """Get object from scene by name, with fallback strategies."""
        try:
            # Try object scope first (for newer tasks)
            if hasattr(self.env.task, 'object_scope') and name in self.env.task.object_scope:
                return self.env.task.object_scope[name]

            # Try scene registry
            return self.env.scene.object_registry("name", name)
        except (KeyError, AttributeError):
            # If object not found, return None (caller should handle)
            return None

    def get_object_poses(self):
        """Gets poses of task-relevant objects based on configuration."""
        if not self.task_config:
            return {}

        object_poses = {}

        # Get robot-specific objects if applicable
        robot_type = type(self.robot).__name__.lower()
        if robot_type in self.task_config.robot_specific_objects:
            tracked_objects = {**self.task_config.tracked_objects,
                             **self.task_config.robot_specific_objects[robot_type]}
        else:
            tracked_objects = self.task_config.tracked_objects

        for logical_name, registry_name in tracked_objects.items():
            # Special handling for robot links (e.g., torso_link4)
            if logical_name == "torso_link4" and registry_name in ["torso_lift_link", "torso_link4"]:
                # This is a robot link, not a scene object
                robot_link = self.env.robots[0].links[registry_name]
                object_poses[logical_name] = self.get_object_pose(robot_link)
            else:
                obj = self._get_object_by_name(registry_name)
                if obj is not None:
                    object_poses[logical_name] = self.get_object_pose(obj)

        return object_poses

    def get_subtask_term_signals(self):
        """Gets termination signals based on configuration."""
        if not self.task_config:
            return {}

        signals = {}

        for signal_name, signal_config in self.task_config.termination_signals.items():
            signal_type = signal_config.get("type")
            obj_name = signal_config.get("object")

            obj = self._get_object_by_name(obj_name) if obj_name else None

            if signal_type == "grasp" and obj:
                arm = signal_config.get("arm", "default")
                signals[signal_name] = int(self.robot.is_grasping(arm=arm, candidate_obj=obj))

            elif signal_type == "touch" and obj:
                signals[signal_name] = int(self.robot.states[Touching].get_value(obj))

            elif signal_type == "custom":
                # Allow custom signal evaluation via callback
                callback = signal_config.get("callback")
                if callable(callback):
                    signals[signal_name] = callback(self.env, self.robot, obj)

        return signals

    def target_pose_to_action(self, target_pose, relative=True):
        """
        Takes a target pose for the end effector controller (in the world frame) and returns an action
        (usually a normalized delta pose action in the robot frame) to try and achieve that target pose.

        Args:
            target_pose (np.array): 4x4 target eef pose, in the world frame

        Returns:
            action (np.array): action compatible with env.step (minus gripper actuation), in the robot frame
        """
        # Legacy
        del relative

        # Ensure float32
        target_pose = target_pose.astype(np.float32)

        # Convert to torch tensor
        target_pose = th.from_numpy(target_pose)

        # Compute the eef target pose in the robot frame
        target_pos, target_quat = T.relative_pose_transform(*T.mat2pose(target_pose), *self.robot.get_position_orientation())

        # Get the current eef pose in the robot frame
        pos_relative, quat_relative = self.robot.get_relative_eef_pose()

        # Find the relative pose between the current eef pose and the target eef pose in the robot frame (delta pose)
        dpos = target_pos - pos_relative

        dori = T.orientation_error(T.quat2mat(target_quat), T.quat2mat(quat_relative))

        # Assemble the arm command and undo the preprocessing
        arm_command = th.cat([dpos, dori])
        arm_action = self.robot.controllers[f"arm_{self.robot.default_arm}"]._reverse_preprocess_command(arm_command)

        # Get an all-zero action (minus gripper actuation) and set the arm command part
        # This assumes other parts of the action (e.g. base, head) are zero
        action = th.from_numpy(np.zeros_like(self.robot.action_space.sample())[:-1])
        action[self.robot.arm_action_idx[self.robot.default_arm]] = arm_action

        # Convert to numpy tensor
        action = action.numpy()

        return action

    def action_to_target_pose(self, action, relative=True):
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action (np.array): environment action

        Returns:
            target_pose (np.array): 4x4 target eef pose that @action corresponds to
        """
        # Legacy
        del relative

        # Ensure float32
        action = action.astype(np.float32)

        # Convert to torch tensor
        action = th.from_numpy(action)

        # Extract the arm command part of the action and preprocess it
        arm_action = action[self.robot.arm_action_idx[self.robot.default_arm]]
        arm_command = self.robot.controllers[f"arm_{self.robot.default_arm}"]._preprocess_command(arm_action)

        # Get the current eef pose in the robot frame
        pos_relative, quat_relative = self.robot.get_relative_eef_pose()

        # Extract the delta pose from the arm command and compute the target pose in the robot frame
        dpos = arm_command[:3]
        target_pos = pos_relative + dpos
        dori = T.quat2mat(T.axisangle2quat(arm_command[3:6]))
        target_quat = T.mat2quat(dori @ T.quat2mat(quat_relative))

        # Convert the target pose to the world frame
        target_pose = T.pose2mat(T.pose_transform(*self.robot.get_position_orientation(), target_pos, target_quat))
        target_pose = target_pose.numpy()

        # Sanity check cycle consistency (not technically necessary)
        new_action = self.target_pose_to_action(target_pose)
        # @new_action has one less element than @action because it doesn't have the gripper actuation
        assert th.allclose(action[:-1], th.from_numpy(new_action), atol=1e-2)

        return target_pose

    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (np.array): environment action

        Returns:
            gripper_action (np.array): subset of environment action for gripper actuation
        """
        # last dimension is gripper action
        return action[-1:]


class OmniGibsonInterfaceBimanual(OmniGibsonInterface):
    """
    MimicGen environment interface class for bimanual robots.
    """
    INTERFACE_TYPE = "omnigibson_bimanual"

    def __init__(self, env, task_config: TaskConfig = None):
        super(OmniGibsonInterfaceBimanual, self).__init__(env, task_config)
        self._setup_arm_controller()
        self.gripper_action_dim = th.cat([self.robot.controller_action_idx["gripper_left"], self.robot.controller_action_idx["gripper_right"]])

    def _setup_arm_controller(self):
        """
        Sets up the arm controller for the robot. This is necessary to know where the arm command
        starts and ends in the action vector.
        """
        self.robot = self.env.robots[0]
        self.arm_controller = {}

    def get_robot_eef_pose(self, arm_name=None):
        """
        Get current robot end effector pose for specified arm or both.

        Returns:
            pose (np.array): 4x4 eef pose matrix for single arm, or 8x4 for both arms
        """
        if arm_name:
            assert arm_name in ["left", "right"]
            return self.get_object_pose(self.robot.eef_links[arm_name])
        else:
            # Return concatenated poses for both arms
            left_pose = self.get_object_pose(self.robot.eef_links["left"])
            right_pose = self.get_object_pose(self.robot.eef_links["right"])
            return np.concatenate([left_pose, right_pose], axis=0)

    def get_datagen_info(self, action=None):
        """
        Get information needed for data generation, at the current
        timestep of simulation. If @action is provided, it will be used to
        compute the target eef pose for the controller, otherwise that
        will be excluded.

        Returns:
            datagen_info (DatagenInfo instance)
        """

        # current eef pose
        eef_pose = self.get_robot_eef_pose()  # 8x4 for bimanual
        base_pose = self.get_object_pose(self.robot)

        # object poses
        object_poses = self.get_object_poses()

        # subtask termination signals
        subtask_term_signals = self.get_subtask_term_signals()

        # these must be extracted from provided action
        # Only record eef_pose that are actually achieved, not the target_pose
        gripper_action = None
        if action is not None:
            gripper_action = self.action_to_gripper_action(action)

        datagen_info = DatagenInfo(
            base_pose=base_pose,
            eef_pose=eef_pose,
            object_poses=object_poses,
            subtask_term_signals=subtask_term_signals,
            gripper_action=gripper_action,
        )
        return datagen_info

    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (np.array): environment action

        Returns:
            gripper_action (np.array): subset of environment action for gripper actuation
        """
        gripper_action = action[self.gripper_action_dim]
        return gripper_action

    def target_pose_to_action(self, target_pose, relative=True):
        """
        Takes a target pose for the end effector controller (in the world frame) and returns an action
        (usually a normalized delta pose action in the robot frame) to try and achieve that target pose.

        Args:
            target_pose (np.array): 4x4 target eef pose, in the world frame

        Returns:
            action (np.array): action compatible with env.step (minus gripper actuation), in the robot frame
        """
        # Legacy
        del relative

        # Ensure float32
        target_pose = target_pose.astype(np.float32)

        # Convert to torch tensor
        target_pose = th.from_numpy(target_pose)
        target_pose_dict = {}
        target_pose_dict["left"] = target_pose[:4,:]
        target_pose_dict["right"] = target_pose[4:,:]

        # Get an all-zero action (minus gripper actuation) and set the arm command part
        # This assumes other parts of the action (e.g. base, head) are zero
        action = np.zeros_like(self.robot.action_space.sample())
        
        control_dict = self.robot.get_control_dict()

        for arm_name in ["left", "right"]:
            target_pose = target_pose_dict[arm_name]

            # Compute the eef target pose in the robot frame
            target_pos, target_quat = T.relative_pose_transform(*T.mat2pose(target_pose), *self.robot.get_position_orientation())

            # Get the current eef pose in the robot frame
            pos_relative, quat_relative = self.robot.get_relative_eef_pose(arm_name)

            # Find the relative pose between the current eef pose and the target eef pose in the robot frame (delta pose)
            dpos = target_pos - pos_relative

            dori = T.orientation_error(T.quat2mat(target_quat), T.quat2mat(quat_relative))

            # Compute delta pose
            err = th.cat([dpos, dori])

            # Replicate the logic from IKController
            arm_controller = self.robot.controllers[f"arm_{arm_name}"]
            arm_dof_idx = arm_controller.dof_idx
            manipulation_dof_idx = arm_dof_idx

            # Assume the trunk is excluded
            # if arm_name == "left":
            #     trunk_controller = self.robot.controllers["trunk"]
            #     trunk_controller_dof_idx = trunk_controller.dof_idx
            #     manipulation_dof_idx = th.cat([arm_dof_idx, trunk_controller_dof_idx])

            j_eef = control_dict[f"eef_{arm_name}_jacobian_relative"][:, manipulation_dof_idx]
            q = control_dict["joint_position"][manipulation_dof_idx]
            q_lower_limit = arm_controller._control_limits[ControlType.get_type("position")][0][manipulation_dof_idx]
            q_upper_limit = arm_controller._control_limits[ControlType.get_type("position")][1][manipulation_dof_idx]

            # percentile = 0.95
            # q_range = q_upper_limit - q_lower_limit
            # q_lower_limit = q_lower_limit + (1 - percentile) / 2 * q_range
            # q_upper_limit = q_upper_limit - (1 - percentile) / 2 * q_range
            # q = np.clip(q, q_lower_limit, q_upper_limit)

            q_dot_lower_limit = arm_controller._control_limits[ControlType.get_type("velocity")][0][manipulation_dof_idx]
            q_dot_upper_limit = arm_controller._control_limits[ControlType.get_type("velocity")][1][manipulation_dof_idx]

            vel_err = err.numpy() / og.sim.get_physics_dt()
            proportional_gain = 0.5

            n = j_eef.shape[1]
            epsilon = 1e-6
            P = j_eef.T @ j_eef + epsilon * np.eye(j_eef.shape[1])
            r = -proportional_gain * vel_err @ j_eef

            velocity_gain = 0.5
            q_dot_upper_limit_by_joint_limit = velocity_gain * (q_upper_limit - q) / og.sim.get_physics_dt()
            q_dot_lower_limit_by_joint_limit = velocity_gain * (q_lower_limit - q) / og.sim.get_physics_dt()

            q_dot_upper_limit = np.minimum(q_dot_upper_limit, q_dot_upper_limit_by_joint_limit)
            q_dot_lower_limit = np.maximum(q_dot_lower_limit, q_dot_lower_limit_by_joint_limit)

            G = np.vstack([np.eye(n), -np.eye(n)])
            h = np.concatenate([q_dot_upper_limit, -q_dot_lower_limit])

            q_dot = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(q_dot, P) + r.T @ q_dot), [G @ q_dot <= h])
            try:
                prob.solve()
            except cp.error.SolverError:
                target_joint_pos = q
            else:
                if prob.status == "optimal":
                    q_dot_val = q_dot.value
                    delta_j = q_dot_val * og.sim.get_physics_dt()
                    target_joint_pos = q + delta_j
                else:
                    target_joint_pos = q

            # NOTE: This clipping is important because the convex optimization (cvxpy) solver does not guarantee that the solution will be STRICTLY within the limits
            # The result is that sometimes the joint positions obtained from the solver are just slightly (even in the order of 1e-5) out of the limits
            # So, making the limits of target_joint_pos (in radians) a bit more stricter will help avoid this issue
            target_joint_pos = np.clip(target_joint_pos, q_lower_limit + 0.02, q_upper_limit - 0.02)

            arm_command = target_joint_pos
            if arm_name == "left":
                # arm_command, trunk_command = arm_command[:arm_dof_idx.shape[0]], arm_command[arm_dof_idx.shape[0]:]
                arm_action = arm_controller._reverse_preprocess_command(arm_command)
                # trunk_command = trunk_controller._reverse_preprocess_command(trunk_command)
                action[self.robot.controller_action_idx[f"arm_{arm_name}"]] = arm_action
                # action[self.robot.controller_action_idx["trunk"]] = trunk_command
            else:
                arm_action = arm_controller._reverse_preprocess_command(arm_command)
                action[self.robot.controller_action_idx[f"arm_{arm_name}"]] = arm_action

        # fill in the no operation actions for the base, camera and trunk
        for name, controller in self.robot.controllers.items():
            if name == 'base' or name == 'camera' or name == "trunk":
                partial_action = controller.compute_no_op_action(control_dict)
                action_idx = self.robot.controller_action_idx[name]
                action[action_idx] = partial_action

        return action

    def generate_action(self, target_pose):
        """
        Generate a no-op action that will keep the robot still but aim to move the arms to the saved pose targets, if possible

        Returns:
            th.tensor or None: Action array for one step for the robot to do nothing
        """
        # change to quaternion 

        # Ensure float32
        target_pose = target_pose.astype(np.float32)

        # Convert to torch tensor
        target_pose = th.from_numpy(target_pose)
        target_pose_dict = {}
        target_pose_dict["left"] = T.mat2pose(target_pose[:4,:]) # T.mat2pose(target_pose)
        target_pose_dict["right"] = T.mat2pose(target_pose[4:,:])
        
        arm_targets = {
            'arm_left': (target_pose_dict["left"][0], target_pose_dict["left"][1], 0),
            'arm_right': (target_pose_dict["right"][0], target_pose_dict["right"][1], 0),
        }

        action = th.zeros(self.robot.action_dim)
        for name, controller in self.robot.controllers.items():
            # if desired arm targets are available, generate an action that moves the arms to the saved pose targets
            if name in arm_targets:
                arm = name.replace("arm_", "")
                # change to robot base frame
                target_pos, target_orn, gripper_state = arm_targets[name] # in world fixed frame

                current_pos, current_orn = self.robot.get_eef_pose(arm)
                if target_orn is None:
                    target_orn = current_orn
                if target_pos is None:
                    target_pos = current_pos
                arm_targets[name] = (target_pos, target_orn, gripper_state)

                delta_pos = target_pos - current_pos
                delta_orn = T.orientation_error(T.quat2mat(target_orn), T.quat2mat(current_orn))
                partial_action = th.cat((delta_pos, delta_orn))
            else:
                partial_action = controller.compute_no_op_action(self.robot.get_control_dict())
            action_idx = self.robot.controller_action_idx[name]
            action[action_idx] = partial_action

            # set the gripper no operation action to 0
            action[11] = 0
            action[-1] = 0
        
        # bug: change to robot base frame

        # Convert to numpy tensor
        action = action.numpy()
        print('generated action')
        print('arm left')
        print(action[5:12])
        print('arm right')
        print(action[12:19])
        return action

    def action_to_target_pose(self, action, relative=True):
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action (np.array): environment action

        Returns:
            target_pose (np.array): 4x4 target eef pose that @action corresponds to
        """
        # Legacy
        del relative

        # Ensure float32
        action = action.astype(np.float32)

        # Convert to torch tensor
        action = th.from_numpy(action)

        target_pose_dict = {}

        for arm_name in ["left", "right"]:
            # Extract the arm command part of the action and preprocess it
            arm_action = action[self.robot.arm_action_idx[arm_name]]
            arm_command = self.robot.controllers[f"arm_{arm_name}"]._preprocess_command(arm_action)

            # Get the current eef pose in the robot frame
            pos_relative, quat_relative = self.robot.get_relative_eef_pose(arm_name)

            # Extract the delta pose from the arm command and compute the target pose in the robot frame
            dpos = arm_command[:3]
            target_pos = pos_relative + dpos
            dori = T.quat2mat(T.axisangle2quat(arm_command[3:6]))
            target_quat = T.mat2quat(dori @ T.quat2mat(quat_relative))

            # Convert the target pose to the world frame
            target_pose = T.pose2mat(T.pose_transform(*self.robot.get_position_orientation(), target_pos, target_quat))
            target_pose = target_pose.numpy()
            
            target_pose_dict[arm_name] = target_pose

        target_pose = np.concatenate([target_pose_dict["left"], target_pose_dict["right"]], axis=0) # 8x4

        # Sanity check cycle consistency (not technically necessary)
        new_action = self.target_pose_to_action(target_pose)
        new_action[self.gripper_action_dim] = action[self.gripper_action_dim]
        new_action[:5] = action[:5]

        # @new_action has one less element than @action because it doesn't have the gripper actuation
        assert th.allclose(action, th.from_numpy(new_action), atol=1e-2)


        return target_pose


class OmniGibsonInterfaceTidyBot(OmniGibsonInterfaceBimanual):
    """
    MoMaGen environment interface for the single-arm TidyBot++ robot, presented
    through the bimanual API (the "phantom arm" scheme).

    The TidyBot robot class aliases the bimanual arm names ("left"/"right") to its
    single arm "0", so the inherited bimanual methods (get_robot_eef_pose -> 8x4
    with both halves identical, get_datagen_info, action_to_gripper_action -> 2
    identical entries) work unchanged. Task configs must put the real subtasks on
    arm_left and set arm_right's object_ref to null everywhere.

    Only the pose<->action converters are overridden: the bimanual versions
    iterate both arms with per-arm controllers, while here a single Jacobian-QP
    solve on the left (real) half of the 8x4 pose suffices.

    Gripper action convention (same as R1): >= 0 opens, < 0 closes.
    """

    INTERFACE_TYPE = "omnigibson_tidybot"

    ARM = "0"

    # control_dict["eef_0_jacobian_relative"] caches link_idx = _articulation_view.get_body_index
    # (eef_link), which TRANSIENTLY returns None mid-generation (after sim manipulation) -> the
    # fcn then does `n_links - None` -> TypeError. eef_link is in fact a real body with a stable
    # index (verified via av.body_names); we resolve the relative-jacobian row from the stable
    # body_names metadata and cache it, with this hardcoded fallback (eef_link is body index 17
    # of the 23-entry TidyBot articulation body list).
    EEF_BODY_INDEX_FALLBACK = 17

    def _eef_jac_row(self):
        """Row of eef_link in ControllableObjectViewAPI.get_relative_jacobian, computed once and
        cached. Uses stable body_names metadata (then get_body_index, then a hardcoded fallback)
        rather than get_body_index alone, which returns None at points during generation."""
        if getattr(self, "_jac_row_cached", None) is not None:
            return self._jac_row_cached
        link = self.robot.eef_link_names[self.ARM]
        av = self.robot._articulation_view
        bidx = None
        try:
            bnames = av.body_names
            if bnames is not None:
                bidx = list(bnames).index(link)
        except Exception:
            bidx = None
        if bidx is None:
            try:
                idx = av.get_body_index(link)
                bidx = None if idx is None else int(idx)
            except Exception:
                bidx = None
        if bidx is None:
            bidx = self.EEF_BODY_INDEX_FALLBACK
        self._jac_row_cached = -(int(self.robot.n_links) - bidx)
        return self._jac_row_cached

    def _get_cmg(self):
        """Lazily build a CuRobo motion generator for replay IK (the env's own CuRobo lives on the
        robomimic wrapper, which the interface doesn't hold; self.env is the underlying OG env)."""
        if getattr(self, "_cmg", None) is None:
            from omnigibson.action_primitives.curobo import CuRoboMotionGenerator, CuRoboEmbodimentSelection
            scene_model = (
                self.env.scene.scene_model.lower()
                if isinstance(self.env.scene, og.scenes.interactive_traversable_scene.InteractiveTraversableScene)
                else "empty"
            )
            self._cmg = CuRoboMotionGenerator(
                robot=self.robot, batch_size=6, use_cuda_graph=False,
                scene_model=scene_model, use_eyes_targets=False,
                device=f"cuda:{os.environ.get('OMNIGIBSON_GPU_ID', '0')}",
            )
            self._emb_sel = CuRoboEmbodimentSelection.ARM_NO_TORSO
            self._cmg_eef_link = self.robot.eef_link_names[self.ARM]
            self._cmg_name_to_dof = {n: i for i, n in enumerate(self._cmg.robot_joint_names)}
            cube = self.robot.scene.object_registry("name", "pick_cube")
            self._cmg.update_obstacles(ignore_objects=[cube] if cube is not None else None)
        return self._cmg

    def _curobo_ik_arm_q(self, tgt_pos, tgt_quat):
        """CuRobo IK for a WORLD eef target -> the arm's joint-position vector (tensor) or None.
        Analytic from the target pose (no eef-pose feedback), seeded from the current config.

        Two things keep the descend in the GOOD kinematic branch (the source executor used a
        collision-aware MP path, so it never branch-jumped; per-waypoint IK can):
          * ik_world_collision_check=True -- filters out arm-through-table / swung-to-the-side
            solutions (the cube is ignored as an obstacle in _get_cmg, so the gripper can still
            descend onto it). Matches the source executor's ik_goal.
          * pick the successful solution NEAREST the current arm config (continuity) rather than
            the first -- avoids snapping to a far IK branch between adjacent waypoints.
        """
        cmg = self._get_cmg()
        bs = cmg.batch_size
        p = th.as_tensor(np.asarray(tgt_pos), dtype=th.float32)
        q = th.as_tensor(np.asarray(tgt_quat.cpu() if hasattr(tgt_quat, "cpu") else tgt_quat), dtype=th.float32)
        tp = {self._cmg_eef_link: th.stack([p for _ in range(bs)])}
        tq = {self._cmg_eef_link: th.stack([q for _ in range(bs)])}
        try:
            succ, js = cmg.compute_trajectories(
                target_pos=tp, target_quat=tq, initial_joint_pos=None, is_local=False,
                max_attempts=50, timeout=20.0, ik_fail_return=10, enable_finetune_trajopt=False,
                finetune_attempts=0, return_full_result=False, success_ratio=1.0 / bs,
                skip_obstacle_update=True, ik_only=True, ik_world_collision_check=False,
                emb_sel=self._emb_sel,
            )
        except Exception:
            return None
        idx = th.where(succ)[0].cpu()
        if len(idx) == 0:
            return None
        arm_idx = self.robot.arm_control_idx[self.ARM]
        cur_arm = self.robot.get_joint_positions()[arm_idx].detach().cpu().float()
        best_arm_q, best_dist = None, None
        for k in idx.tolist():
            sol = js[int(k)]
            pos = th.as_tensor(sol.position).detach().cpu().float().flatten()
            names = list(sol.joint_names)
            full_q = self.robot.get_joint_positions().clone().cpu().float()
            for jn, v in zip(names, pos):
                di = self._cmg_name_to_dof.get(jn)
                if di is not None:
                    full_q[di] = float(v)
            arm_q = full_q[arm_idx]
            d = float((arm_q - cur_arm).abs().sum())
            if best_dist is None or d < best_dist:
                best_dist, best_arm_q = d, arm_q
        return best_arm_q

    def _curobo_mp_q(self, tgt_pos, tgt_quat):
        """Full collision-aware MOTION PLAN (cube ignored) to the target -> final arm config (tensor)
        or None. Stronger than pure IK: trajopt finds configs/paths pure IK misses. Matches the
        source executor's plan_mp (which reached the grasp pose reliably)."""
        cmg = self._get_cmg()
        bs = cmg.batch_size
        p = th.as_tensor(np.asarray(tgt_pos), dtype=th.float32)
        qq = th.as_tensor(np.asarray(tgt_quat.cpu() if hasattr(tgt_quat, "cpu") else tgt_quat), dtype=th.float32)
        tp = {self._cmg_eef_link: th.stack([p for _ in range(bs)])}
        tq = {self._cmg_eef_link: th.stack([qq for _ in range(bs)])}
        try:
            results, paths = cmg.compute_trajectories(
                target_pos=tp, target_quat=tq, initial_joint_pos=None, is_local=False,
                max_attempts=50, timeout=30.0, ik_fail_return=10, enable_finetune_trajopt=True,
                finetune_attempts=1, return_full_result=True, success_ratio=1.0 / bs,
                skip_obstacle_update=True, ik_only=False, ik_world_collision_check=False,
                emb_sel=self._emb_sel,
            )
        except Exception as e:
            return None
        idx = th.where(results[0].success)[0].cpu()
        if len(idx) == 0:
            return None
        q_traj = cmg.path_to_joint_trajectory(paths[int(idx[0])], get_full_js=True, emb_sel=self._emb_sel).cpu().float()
        return q_traj[-1][self.robot.arm_control_idx[self.ARM]]

    def target_pose_to_action(self, target_pose, relative=True):
        """MoMaGen-faithful control: R1's live-Jacobian damped-QP, on TidyBot's single arm.

        One damped least-squares joint step toward the world eef target per call, using R1's
        EXACT scheme + params (proportional_gain 0.5, velocity_gain 0.5, eps 1e-6, integrate
        q + q_dot*dt, clip to limits). This is the same method MoMaGen uses for R1; the only
        TidyBot modifications are (a) the single (real) arm instead of left+right, and (b) the
        eef relative jacobian taken via the crash-safe manual body-row when R1's control_dict
        entry trips the transient get_body_index->None.

        This only tracks because the arm runs as an ABSOLUTE-position JointController -- the
        env_omnigibson reload_controllers was reverting arm_0 to delta mode (absolute targets
        applied as current_qpos+target); fixed by adding the arm_0 key to that reload config.
        """
        # Legacy
        del relative

        target_pose = th.from_numpy(target_pose.astype(np.float32))[:4, :]  # left/real half (world)

        action = np.zeros_like(self.robot.action_space.sample())
        control_dict = self.robot.get_control_dict()

        arm_name = self.ARM
        arm_controller = self.robot.controllers[f"arm_{arm_name}"]

        # Compute the eef target pose in the robot frame, and the delta twist from the current eef
        target_pos, target_quat = T.relative_pose_transform(*T.mat2pose(target_pose), *self.robot.get_position_orientation())
        pos_relative, quat_relative = self.robot.get_relative_eef_pose(arm_name)
        dpos = target_pos - pos_relative
        dori = T.orientation_error(T.quat2mat(target_quat), T.quat2mat(quat_relative))
        err = th.cat([dpos, dori])

        manipulation_dof_idx = arm_controller.dof_idx

        # eef relative jacobian: R1's control_dict entry, with the manual body-row as a crash-safe
        # fallback (numerically identical -- the control_dict builder's get_body_index transiently
        # returns None mid-generation -> n_links-None TypeError).
        try:
            j_eef_full = control_dict[f"eef_{arm_name}_jacobian_relative"]
        except Exception:
            start_idx = 0 if self.robot.fixed_base else 6
            j_eef_full = ControllableObjectViewAPI.get_relative_jacobian(self.robot.articulation_root_path)[
                self._eef_jac_row(), :, start_idx : start_idx + self.robot.n_joints
            ]
        j_eef = j_eef_full[:, manipulation_dof_idx]

        q = control_dict["joint_position"][manipulation_dof_idx]
        q_lower_limit = arm_controller._control_limits[ControlType.get_type("position")][0][manipulation_dof_idx]
        q_upper_limit = arm_controller._control_limits[ControlType.get_type("position")][1][manipulation_dof_idx]
        q_dot_lower_limit = arm_controller._control_limits[ControlType.get_type("velocity")][0][manipulation_dof_idx]
        q_dot_upper_limit = arm_controller._control_limits[ControlType.get_type("velocity")][1][manipulation_dof_idx]

        vel_err = err.numpy() / og.sim.get_physics_dt()
        proportional_gain = 0.5

        n = j_eef.shape[1]
        epsilon = 1e-6
        P = j_eef.T @ j_eef + epsilon * np.eye(j_eef.shape[1])
        r = -proportional_gain * vel_err @ j_eef

        velocity_gain = 0.5
        q_dot_upper_limit_by_joint_limit = velocity_gain * (q_upper_limit - q) / og.sim.get_physics_dt()
        q_dot_lower_limit_by_joint_limit = velocity_gain * (q_lower_limit - q) / og.sim.get_physics_dt()
        q_dot_upper_limit = np.minimum(q_dot_upper_limit, q_dot_upper_limit_by_joint_limit)
        q_dot_lower_limit = np.maximum(q_dot_lower_limit, q_dot_lower_limit_by_joint_limit)

        G = np.vstack([np.eye(n), -np.eye(n)])
        h = np.concatenate([q_dot_upper_limit, -q_dot_lower_limit])

        q_dot = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(q_dot, P) + r.T @ q_dot), [G @ q_dot <= h])
        try:
            prob.solve()
        except cp.error.SolverError:
            target_joint_pos = q
        else:
            if prob.status == "optimal":
                target_joint_pos = q + q_dot.value * og.sim.get_physics_dt()
            else:
                target_joint_pos = q

        # Keep strictly inside limits (the QP can exceed them by ~1e-5)
        target_joint_pos = np.clip(target_joint_pos, q_lower_limit + 0.02, q_upper_limit - 0.02)

        action[self.robot.controller_action_idx[f"arm_{arm_name}"]] = arm_controller._reverse_preprocess_command(
            th.as_tensor(target_joint_pos, dtype=th.float32)
        )

        # No-op for the base (gripper is filled by callers)
        for name, controller in self.robot.controllers.items():
            if name == "base":
                action[self.robot.controller_action_idx[name]] = controller.compute_no_op_action(control_dict)

        return action

    def action_to_target_pose(self, action, relative=True):
        """
        Inverse of @target_pose_to_action; returns the bimanual 8x4 format with
        both halves equal to the single arm's target pose.
        """
        # Legacy
        del relative

        action = th.from_numpy(action.astype(np.float32))

        arm_action = action[self.robot.arm_action_idx[self.ARM]]
        arm_command = self.robot.controllers[f"arm_{self.ARM}"]._preprocess_command(arm_action)

        # The arm command is absolute target joint positions; convert to an eef
        # pose via the current eef pose + the commanded joint delta through the
        # relative jacobian (small-displacement approximation, consistent with
        # how target_pose_to_action produces commands)
        control_dict = self.robot.get_control_dict()
        manipulation_dof_idx = self.robot.controllers[f"arm_{self.ARM}"].dof_idx
        q = control_dict["joint_position"][manipulation_dof_idx]
        j_eef = control_dict[f"eef_{self.ARM}_jacobian_relative"][:, manipulation_dof_idx]
        twist = j_eef @ (arm_command - q)

        pos_relative, quat_relative = self.robot.get_relative_eef_pose(self.ARM)
        target_pos = pos_relative + twist[:3]
        dori = T.quat2mat(T.axisangle2quat(twist[3:6]))
        target_quat = T.mat2quat(dori @ T.quat2mat(quat_relative))

        target_pose = T.pose2mat(T.pose_transform(*self.robot.get_position_orientation(), target_pos, target_quat))
        target_pose = target_pose.numpy()

        return np.concatenate([target_pose, target_pose], axis=0)  # 8x4


# Task configuration definitions
TASK_CONFIGS = {

    "test_tiago_cup": TaskConfig(
        name="test_tiago_cup",
        tracked_objects={
            "coffee_cup": "coffee_cup",
            "teacup": "teacup",
            "breakfast_table": "breakfast_table",
        },
        termination_signals={
            "grasp_right": {
                "type": "grasp",
                "object": "coffee_cup",
                "arm": "right"
            }
        }
    ),

    "test_r1_cup": TaskConfig(
        name="test_r1_cup",
        tracked_objects={
            "coffee_cup": "coffee_cup",
            "teacup": "teacup",
            "breakfast_table": "breakfast_table",
        },
        termination_signals={
            "grasp_right": {
                "type": "grasp",
                "object": "coffee_cup",
                "arm": "right"
            }
        }
    ),

    "r1_tidy_table": TaskConfig(
        name="r1_tidy_table",
        tracked_objects={
            "teacup_601": "teacup_601",
            "drop_in_sink_awvzkn_0": "drop_in_sink_awvzkn_0",
        },
    ),

    "r1_pick_cup": TaskConfig(
        name="r1_pick_cup",
        tracked_objects={
            "coffee_cup_7": "coffee_cup_7",
            "breakfast_table_6": "breakfast_table_6",
        },
    ),

    "r1_dishes_away": TaskConfig(
        name="r1_dishes_away",
        tracked_objects={
            "countertop_kelker_0": "countertop_kelker_0",
            "shelf_pfusrd_1": "shelf_pfusrd_1",
            "plate_603": "plate_603",
            "plate_602": "plate_602",
            "plate_601": "plate_601",
        },
    ),

    "r1_clean_pan": TaskConfig(
        name="r1_clean_pan",
        tracked_objects={
            "frying_pan_602": "frying_pan_602",
            "scrub_brush_601": "scrub_brush_601",
            "robot_r1": "robot_r1",
        },
        robot_specific_objects={
            "tiago": {"torso_link4": "torso_lift_link"},
            "r1": {"torso_link4": "torso_link4"},
        },
    ),

    "r1_bringing_water": TaskConfig(
        name="r1_bringing_water",
        tracked_objects={
            "beer_bottle_595": "beer_bottle_595",
            "fridge_dszchb_0": "fridge_dszchb_0",
        },
    ),

    "r1_picking_up_trash": TaskConfig(
        name="r1_picking_up_trash",
        tracked_objects={
            "can_of_soda_261": "can_of_soda_261",
            "trash_can_262": "trash_can_262",
        },
    ),

    "tidybot_pick_cup": TaskConfig(
        name="tidybot_pick_cup",
        tracked_objects={
            # pick_cube is the grasped object: object_ref for the manipulation subtask, so its
            # pose MUST be in datagen_info for the trajectory transform. (The Hand-E can't wrap
            # the 71mm coffee cup, so we pick an added 30x30x60mm cube instead.)
            "pick_cube": "pick_cube",
            "coffee_cup_7": "coffee_cup_7",
            "breakfast_table_6": "breakfast_table_6",
        },
    ),

    "tidybot_tidy_table": TaskConfig(
        name="tidybot_tidy_table",
        tracked_objects={
            # teacup_601 is grasped by its HANDLE (the Hand-E's ~50mm opening can't wrap the
            # bowl). drop_in_sink_awvzkn_0 is the place target. Both poses must be in datagen_info.
            "teacup_601": "teacup_601",
            "drop_in_sink_awvzkn_0": "drop_in_sink_awvzkn_0",
        },
    ),

    "tidybot_picking_up_trash": TaskConfig(
        name="tidybot_picking_up_trash",
        tracked_objects={
            # datagen_picking_up_trash sampled in house_single_floor (kitchen countertop -> floor
            # trash can). can_of_soda_595 = grasped item (object_ref for phase 1, attached_obj for
            # phase 2); trash_can_596 = drop target (object_ref for phase 2). Both poses in datagen_info.
            "can_of_soda_595": "can_of_soda_595",
            "trash_can_596": "trash_can_596",
        },
    ),

    "tidybot_make_coffee": TaskConfig(
        name="tidybot_make_coffee",
        tracked_objects={
            # datagen_make_coffee sampled in house_single_floor (kelker kitchen countertop).
            # teacup_598 ("milk" vessel) = object_ref phase 1 / attached phase 2;
            # toy_box_597 ("coffee" box) = object_ref phase 3 / attached phase 4;
            # coffee_cup_599 (1.4x target cup) = object_ref phases 2+4. Tokens
            # (sugar_cube_596, dice_595) move by physics only but are tracked so their
            # poses land in datagen_info for debugging.
            "teacup_598": "teacup_598",
            "toy_box_597": "toy_box_597",
            "coffee_cup_599": "coffee_cup_599",
            "sugar_cube_596": "sugar_cube_596",
            "dice_595": "dice_595",
        },
    ),

    # ------------------------------------------------------------------------------------------------
    # Add new task configs here
    # Note 1: tracked_objects is a dictionary with the same key and value. Furthermore, the tracked_object is the
    # OG specific name of the object which you can find by clicking on the object on the GUI
    # Note 2: we are currently not using the termination_signals for the data generation, so you can leave it empty
    # ------------------------------------------------------------------------------------------------

    "datagen_pick": TaskConfig(
        name="datagen_pick",
        tracked_objects={
            "coffee_cup_7": "coffee_cup_7",
            "breakfast_table_6": "breakfast_table_6",
        },
    ),

    "r1_put_away_cup": TaskConfig(
        name="r1_put_away_cup",
        tracked_objects={
            "coffee_cup_7": "coffee_cup_7",
            "breakfast_table_6": "breakfast_table_6",
        },
    ),

}

# Backward compatibility - create legacy classes that use the new system
class MG_TestTiagoCup(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["test_tiago_cup"])

class MG_TestR1Cup(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["test_r1_cup"])

class MG_R1PutAwayCup(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["r1_put_away_cup"])

class MG_R1TidyTable(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["r1_tidy_table"])

class MG_R1PickCup(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["r1_pick_cup"])

class MG_R1DishesAway(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["r1_dishes_away"])

class MG_R1CleanPan(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["r1_clean_pan"])

class MG_R1BringingWater(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["r1_bringing_water"])

# ---------------------------------
# Add new class here for new tasks
# ---------------------------------

class MG_R1PickingUpTrash(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["r1_picking_up_trash"])

class MG_TidyBotPickCup(OmniGibsonInterfaceTidyBot):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["tidybot_pick_cup"])

class MG_TidyBotTidyTable(OmniGibsonInterfaceTidyBot):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["tidybot_tidy_table"])

class MG_TidyBotPickingUpTrash(OmniGibsonInterfaceTidyBot):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["tidybot_picking_up_trash"])

class MG_TidyBotMakeCoffee(OmniGibsonInterfaceTidyBot):
    def __init__(self, env):
        super().__init__(env, TASK_CONFIGS["tidybot_make_coffee"])



