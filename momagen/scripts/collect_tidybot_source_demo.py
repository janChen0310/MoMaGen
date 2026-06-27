"""
Collect a TidyBot++ source demonstration for MoMaGen, inside OmniGibson.

Loads the OmniGibson env config embedded in an existing R1 source dataset
(scene, BDDL task, objects), swaps the robot for TidyBot, and records a
keyboard-teleoperated episode with OmniGibson's DataCollectionWrapper — which
writes the same HDF5 layout as the R1 source demos
(data attrs config/n_episodes/n_steps + data/demo_X/{action,state,state_size,...}).

After collecting, annotate it exactly like an R1 source demo:
    python momagen/scripts/prepare_src_dataset.py \
        --dataset momagen/datasets/source_og/tidybot_pick_cup.hdf5 \
        --env_interface MG_TidyBotPickCup --env_interface_type omnigibson_tidybot ...

Teleop keys (tap to step; this is a minimal collection utility, not a polished
teleop — telemoma/JoyLo can be substituted if preferred):
    Base:    W/S forward/back, A/D strafe left/right, Q/E rotate
    Arm eef: I/K +x/-x, J/L +y/-y, U/O +z/-z (robot frame)
             T/G pitch, F/H yaw, R/Y roll
    Gripper: SPACE toggles open/close
    Episode: C  finish + save episode      ESC quit (saves what was recorded)

Usage (GUI required, run on the GPU machine):
    python momagen/scripts/collect_tidybot_source_demo.py \
        --template momagen/datasets/processed_source_demos/r1_pick_cup.hdf5 \
        --output momagen/datasets/source_og/tidybot_pick_cup.hdf5
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataCollectionWrapper
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardEventHandler

from momagen.utils.robot_config import get_tidybot_config

POS_STEP = 0.02      # m per key tap
ORI_STEP = 0.05      # rad per key tap
BASE_LIN = 0.4       # normalized base command per held tap
BASE_ANG = 0.4


def load_tidybot_env_config(template_hdf5):
    """OG env config from the template dataset, with the robot swapped to TidyBot."""
    with h5py.File(template_hdf5, "r") as f:
        cfg = json.loads(f["data"].attrs["config"])

    robot_cfg = cfg["robots"][0]
    robot_cfg["type"] = "TidyBot"
    _, controller_config = get_tidybot_config()
    robot_cfg["controller_config"] = controller_config
    robot_cfg.pop("reset_joint_pos", None)
    robot_cfg["default_reset_mode"] = "untuck"
    # Keep obs modalities lean for collection; datagen re-renders what it needs
    robot_cfg["obs_modalities"] = ["rgb", "proprio"]
    return cfg


class CartesianTeleop:
    """Tap-to-step Cartesian teleop producing MoMaGen-compatible actions
    (absolute joint-position arm commands via damped-least-squares IK)."""

    def __init__(self, robot):
        self.robot = robot
        self.arm = robot.default_arm
        self.base_cmd = th.zeros(3)
        self.gripper_closed = False
        self.done = False
        self.quit = False
        # Persistent eef target in the robot frame
        pos, quat = robot.get_relative_eef_pose(self.arm)
        self.target_pos = pos.clone()
        self.target_quat = quat.clone()

    def nudge(self, dpos=None, dori=None):
        if dpos is not None:
            self.target_pos = self.target_pos + th.tensor(dpos, dtype=th.float32)
        if dori is not None:
            dmat = T.euler2mat(th.tensor(dori, dtype=th.float32))
            self.target_quat = T.mat2quat(dmat @ T.quat2mat(self.target_quat))

    def action(self):
        robot = self.robot
        control_dict = robot.get_control_dict()
        arm_controller = robot.controllers[f"arm_{self.arm}"]
        dof_idx = arm_controller.dof_idx

        q = control_dict["joint_position"][dof_idx]
        j_eef = control_dict[f"eef_{self.arm}_jacobian_relative"][:, dof_idx]

        pos_rel, quat_rel = robot.get_relative_eef_pose(self.arm)
        dpos = self.target_pos - pos_rel
        dori = T.orientation_error(T.quat2mat(self.target_quat), T.quat2mat(quat_rel))
        err = th.cat([dpos, dori])

        # Damped least squares step toward the target
        lam = 1e-4
        JT = j_eef.T
        dq = JT @ th.linalg.solve(j_eef @ JT + lam * th.eye(6), err)
        target_q = q + th.clamp(dq, -0.05, 0.05)

        action = th.zeros(robot.action_dim)
        action[robot.controller_action_idx[f"arm_{self.arm}"]] = (
            arm_controller._reverse_preprocess_command(target_q)
        )
        action[robot.controller_action_idx["base"]] = self.base_cmd
        action[robot.controller_action_idx[f"gripper_{self.arm}"]] = -1.0 if self.gripper_closed else 1.0
        # base command decays so taps produce bounded motion
        self.base_cmd = self.base_cmd * 0.9
        return action

    def register_keys(self):
        K = lazy.carb.input.KeyboardInput
        add = KeyboardEventHandler.add_keyboard_callback

        def base(dx=0.0, dy=0.0, dth=0.0):
            def cb():
                self.base_cmd = th.tensor([dx, dy, dth])
            return cb

        add(K.W, base(dx=BASE_LIN)); add(K.S, base(dx=-BASE_LIN))
        add(K.A, base(dy=BASE_LIN)); add(K.D, base(dy=-BASE_LIN))
        add(K.Q, base(dth=BASE_ANG)); add(K.E, base(dth=-BASE_ANG))

        add(K.I, lambda: self.nudge(dpos=[POS_STEP, 0, 0])); add(K.K, lambda: self.nudge(dpos=[-POS_STEP, 0, 0]))
        add(K.J, lambda: self.nudge(dpos=[0, POS_STEP, 0])); add(K.L, lambda: self.nudge(dpos=[0, -POS_STEP, 0]))
        add(K.U, lambda: self.nudge(dpos=[0, 0, POS_STEP])); add(K.O, lambda: self.nudge(dpos=[0, 0, -POS_STEP]))
        add(K.T, lambda: self.nudge(dori=[0, ORI_STEP, 0])); add(K.G, lambda: self.nudge(dori=[0, -ORI_STEP, 0]))
        add(K.F, lambda: self.nudge(dori=[0, 0, ORI_STEP])); add(K.H, lambda: self.nudge(dori=[0, 0, -ORI_STEP]))
        add(K.R, lambda: self.nudge(dori=[ORI_STEP, 0, 0])); add(K.Y, lambda: self.nudge(dori=[-ORI_STEP, 0, 0]))

        def toggle_gripper():
            self.gripper_closed = not self.gripper_closed
        add(K.SPACE, toggle_gripper)

        def finish():
            self.done = True
        add(K.C, finish)

        def quit_():
            self.done = True
            self.quit = True
        add(K.ESCAPE, quit_)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True,
                        help="existing R1 source hdf5 whose env config (scene/task) is reused")
    parser.add_argument("--output", required=True, help="output hdf5 path")
    args = parser.parse_args()

    cfg = load_tidybot_env_config(args.template)
    gm.ENABLE_TRANSITION_RULES = False

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    env = og.Environment(configs=cfg)
    env = DataCollectionWrapper(env=env, output_path=args.output, only_successes=False)

    robot = env.robots[0]
    env.reset()
    for _ in range(10):
        og.sim.step()

    teleop = CartesianTeleop(robot)
    KeyboardEventHandler.initialize()
    teleop.register_keys()

    print(__doc__)
    print(f"Robot: {type(robot).__name__}, action_dim={robot.action_dim}")
    print("Teleop running. Press C to finish + save the episode.")

    while not teleop.done:
        env.step(teleop.action())
        success = env.task.success if hasattr(env.task, "success") else None
        if success:
            print("Task success detected! Press C to finish + save.")

    env.save_data()
    print(f"Saved episode(s) to {args.output}")

    # Add the robomimic-style filter key used by MoMaGen's source loader
    with h5py.File(args.output, "r+") as f:
        demos = sorted(f["data"].keys())
        if "mask" not in f:
            f.create_group("mask")
        if "use" not in f["mask"]:
            f["mask"].create_dataset("use", data=np.array([demos[-1].encode()]))
    print(f"Tagged {demos[-1]} as the 'use' source demo.")

    og.shutdown()


if __name__ == "__main__":
    main()
