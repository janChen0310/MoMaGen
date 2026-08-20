"""Closed-loop rollout of a finetuned pi0.5 policy on the TidyBot dispose-trash task.

Runs in the `momagen` env (OmniGibson side). The policy runs in a separate process
(`openpi/scripts/serve_policy.py --env=... policy:checkpoint ...`) and is queried over
openpi's websocket protocol, keeping the JAX and Isaac CUDA stacks apart.

Env construction mirrors momagen/scripts/generate_dataset.py exactly (same config JSON,
same env_meta surgery, same robomimic wrapper), so `env.reset()` draws task instances
from the SAME D0 randomization distribution the training data was generated from.
Held-out-ness comes from the RNG seeds (--seed-start far from the fleet's 3001..3004).

Per control step the policy sees what the converter recorded:
  observation/base_image  <- base_camera_link rgb (camera C, rear-left mast)
  observation/wrist_image <- arm_camera_link rgb
  observation/state (11,) <- [base_qvel(3), arm qpos(7), gripper qpos(1)]
and returns an action chunk [H, 11]; the first --execute-horizon steps are executed
open-loop, then we re-infer.

Usage (box):
  python momagen/scripts/rollout_pi05.py \
      --config momagen/datasets/configs/demo_src_tidybot_picking_up_trash_task_D0.json \
      --host localhost --port 8000 --num-episodes 20 --out-dir rollout_out
"""

import argparse
import json
import os
import random
import time

import imageio
import numpy as np
import torch as th

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="datagen config JSON (same one the fleet uses)")
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--num-episodes", type=int, default=20)
parser.add_argument("--seed-start", type=int, default=9001, help="held out: fleet used 3001..3004")
parser.add_argument("--max-steps", type=int, default=3000)
parser.add_argument("--execute-horizon", type=int, default=20,
                    help="env steps executed per inference (<= model action_horizon)")
parser.add_argument("--prompt", default="pick up the trash and put it in the trash can")
parser.add_argument("--out-dir", default="rollout_out")
parser.add_argument("--video-every", type=int, default=1, help="save video every Nth episode")
args = parser.parse_args()

# Isaac import order matters: config/env utils pull in omnigibson
from robomimic.utils.file_utils import get_env_metadata_from_dataset
import momagen.utils.robomimic_utils as RobomimicUtils
from momagen.utils.robot_config import configure_tidybot_env_meta
from momagen.configs.config import config_factory
import omnigibson as og

from openpi_client import websocket_client_policy


def build_env(mg_config):
    """Identical construction to generate_dataset.py (obs-collection branch)."""
    source_dataset_path = os.path.expandvars(os.path.expanduser(mg_config.experiment.source.dataset_path))
    env_meta = get_env_metadata_from_dataset(dataset_path=source_dataset_path)
    env_meta = configure_tidybot_env_meta(env_meta)
    env = RobomimicUtils.create_env(
        env_meta=env_meta,
        env_class=None,
        env_name=mg_config.experiment.task.name,
        robot=mg_config.experiment.task.robot,
        gripper=mg_config.experiment.task.gripper,
        camera_names=mg_config.obs.camera_names,
        camera_height=mg_config.obs.camera_height,
        camera_width=mg_config.obs.camera_width,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        use_depth_obs=False,
        manipulation_only=False,
        real_robot_mode=False,
        baseline=None,
    )
    return env


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def get_policy_obs(env, prompt):
    """Assemble the policy observation from the same sources the converter recorded."""
    robot = env.env.robots[0]
    arm = robot.default_arm
    jq = robot.get_joint_positions()
    jv = robot.get_joint_velocities()
    state = np.concatenate([
        to_np(jv[robot.base_control_idx]),          # base_qvel (3)  [locomotion_robot.py]
        to_np(jq[robot.arm_control_idx[arm]]),      # arm_0_qpos (7)
        to_np(jq[robot.gripper_control_idx[arm]])[:1],  # gripper_0_qpos first dof
    ]).astype(np.float32)

    imgs = {}
    for name, sensor in robot.sensors.items():
        if "base_camera_link" in name:
            key = "base"
        elif "arm_camera_link" in name:
            key = "wrist"
        else:
            continue
        rgb = to_np(sensor.get_obs()[0]["rgb"])[:, :, :3]
        imgs[key] = np.ascontiguousarray(rgb, dtype=np.uint8)
    assert set(imgs) == {"base", "wrist"}, f"missing robot cameras, found {list(imgs)}"

    return {
        "observation/base_image": imgs["base"],
        "observation/wrist_image": imgs["wrist"],
        "observation/state": state,
        "prompt": prompt,
    }, imgs


def _resize_nn(im, H):
    """Nearest-neighbour resize to height H (width scaled to keep aspect). Pure numpy
    so it runs in the momagen env (no PIL)."""
    h, w = im.shape[:2]
    if h == H:
        return im
    W = max(1, int(round(w * H / h)))
    ri = (np.arange(H) * h / H).astype(int)
    ci = (np.arange(W) * w / W).astype(int)
    return im[ri][:, ci]


def video_frame(env, imgs):
    """world debug cam | base cam | wrist cam side by side. All tiles are resized to a
    common height (the tallest) so NONE is cropped -- the debug world cam renders small
    (128) while base/wrist are the full 224x224 the policy actually sees."""
    ext = env.env._external_sensors["external_sensor2"]
    world = to_np(ext.get_obs()[0]["rgb"])[:, :, :3].astype(np.uint8)
    tiles = [world, imgs["base"], imgs["wrist"]]
    H = max(t.shape[0] for t in tiles)
    return np.concatenate([_resize_nn(t, H) for t in tiles], axis=1)


def main():
    ext_cfg = json.load(open(os.path.expanduser(args.config)))
    ext_cfg.pop("meta", None)  # robomimic config-generator block, unused by MoMaGen
    mg_config = config_factory(ext_cfg["name"], config_type=ext_cfg["type"])
    with mg_config.values_unlocked():
        mg_config.update(ext_cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.jsonl")

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[rollout] policy server metadata: {client.get_server_metadata()}", flush=True)

    env = build_env(mg_config)

    n_success = 0
    for ep in range(args.num_episodes):
        seed = args.seed_start + ep
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)

        env.reset()
        env.sensor_setup()
        for _ in range(5):
            og.sim.render()

        frames = []
        record = args.video_every > 0 and (ep % args.video_every == 0)
        success = False
        step = 0
        t0 = time.time()
        while step < args.max_steps and not success:
            obs, imgs = get_policy_obs(env, args.prompt)
            chunk = np.asarray(client.infer(obs)["actions"], dtype=np.float32)[:, :11]
            for a in chunk[: args.execute_horizon]:
                env.step(th.as_tensor(a, dtype=th.float32))
                step += 1
                if record:
                    _, imgs_now = get_policy_obs(env, args.prompt)
                    frames.append(video_frame(env, imgs_now))
                if env.is_success()["task"]:
                    success = True
                    break
                if step >= args.max_steps:
                    break

        n_success += int(success)
        dur = time.time() - t0
        tag = "succ" if success else "fail"
        if record and frames:
            vp = os.path.join(args.out_dir, f"ep{ep:03d}_seed{seed}_{tag}.mp4")
            imageio.mimwrite(vp, frames, fps=20, quality=6)
            print(f"[rollout] video -> {vp}", flush=True)
        rec = {"episode": ep, "seed": seed, "success": success, "steps": step, "wall_s": round(dur, 1)}
        with open(results_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[rollout] ep {ep}: {tag} in {step} steps ({dur:.0f}s) | "
              f"running success {n_success}/{ep + 1}", flush=True)

    print(f"[rollout] DONE success rate {n_success}/{args.num_episodes} = "
          f"{n_success / max(1, args.num_episodes):.2f}", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
