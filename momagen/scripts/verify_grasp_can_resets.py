"""Render N real env.reset() samples of tidybot_grasp_can, for human review.

Numbers said the task is well-posed: 0% of spawns can grasp directly, and 23% of nearby walkway
positions are viable grasp stations. That is necessary but not sufficient -- it says nothing about
whether the can is somewhere sensible on the counter, whether the robot spawns somewhere a person
would call reasonable, or whether the scene looks like the task we described.

Every silent bug in this project was caught by looking, so this looks. It drives the REAL reset
path (the env is constructed with the grasp_can name, so _counter_sample and
_spawn_sample_grasp_can are the ones that run), and renders each sampled episode start from an
over-the-shoulder view plus the robot's own head camera.
"""
import argparse
import os

import numpy as np

REPO = os.environ.get("MOMAGEN_REPO",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default="grasp_can_resets.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cv2
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    from robomimic.utils.file_utils import get_env_metadata_from_dataset
    import momagen.utils.robomimic_utils as RobomimicUtils
    from momagen.utils.robot_config import configure_tidybot_env_meta

    # grasp_can reuses the trash task's BDDL and scene instance, so the trash source demo supplies
    # a valid env config. The env NAME is what selects the grasp_can task branches.
    src = os.path.join(REPO, "momagen/datasets/processed_source_demos/tidybot_picking_up_trash.hdf5")
    env_meta = configure_tidybot_env_meta(get_env_metadata_from_dataset(dataset_path=src))
    env = RobomimicUtils.create_env(
        env_meta=env_meta, env_class=None, env_name="tidybot_grasp_can_D0",
        robot=None, gripper=None, camera_names=[], camera_height=84, camera_width=84,
        render=False, render_offscreen=True, use_image_obs=False, use_depth_obs=False,
        manipulation_only=False, real_robot_mode=False, baseline=None)
    print("ENV_READY name=%s" % env.env_name if hasattr(env, "env_name") else "ENV_READY", flush=True)

    import torch as th
    from omnigibson.sensors import VisionSensor
    from base_pose_metric.geometry import look_at_rotation
    from kineready.frames import mat_to_quat

    robot = env.env.robots[0]
    can = env.env.scene.object_registry("name", "can_of_soda_595")
    vs = {n: s for n, s in robot.sensors.items() if isinstance(s, VisionSensor)}
    head = next((s for n, s in vs.items() if "head_camera" in n), None)
    if head is not None:
        head.add_modality("rgb")
    viewer = og.sim.viewer_camera
    viewer.add_modality("rgb")

    def grab(sensor):
        og.sim.render()
        obs = sensor.get_obs()
        rgb = obs[0]["rgb"] if isinstance(obs, tuple) else obs["rgb"]
        return np.asarray(rgb.cpu() if hasattr(rgb, "cpu") else rgb)[:, :, :3].astype(np.uint8)

    def square(img, n=384):
        h, w = img.shape[:2]
        side = min(h, w)
        img = img[(h - side) // 2:(h - side) // 2 + side, (w - side) // 2:(w - side) // 2 + side]
        return cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)

    np.random.seed(args.seed)
    tiles = []
    for i in range(args.n):
        env.reset()
        bp = np.asarray(robot.get_position_orientation()[0].cpu(), float)
        cp = np.asarray(can.get_position_orientation()[0].cpu(), float)
        dist = float(np.linalg.norm(bp[:2] - cp[:2]))

        # Over the shoulder: behind and left of the robot, looking past it at the can. Anchored to
        # the robot because the robot is by construction standing in free space.
        to_can = cp[:2] - bp[:2]
        d = to_can / max(float(np.linalg.norm(to_can)), 1e-6)
        perp = np.array([-d[1], d[0]])
        eye = np.array([bp[0] - 1.5 * d[0] + 0.9 * perp[0], bp[1] - 1.5 * d[1] + 0.9 * perp[1], 1.9])
        focus = np.array([bp[0] + 0.55 * to_can[0], bp[1] + 0.55 * to_can[1], 0.85])
        viewer.set_position_orientation(
            position=th.tensor(eye, dtype=th.float32),
            orientation=th.tensor(mat_to_quat(look_at_rotation(focus - eye)), dtype=th.float32))

        third = square(grab(viewer))
        if head is not None:
            hd = square(grab(head), 128)
            third[-128:, -128:] = hd            # head-camera inset, bottom-right

        img = np.ascontiguousarray(third[:, :, ::-1])
        col = (80, 220, 80)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), col, 2)
        for k, txt in enumerate(["#%d  d=%.2fm" % (i + 1, dist),
                                 "can [%.2f, %.2f]" % (cp[0], cp[1]),
                                 "base [%.2f, %.2f]" % (bp[0], bp[1])]):
            y = 18 + 16 * k
            cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        tiles.append(img[:, :, ::-1])
        print("[reset] %2d/%d  can=[%.2f,%.2f] base=[%.2f,%.2f] dist=%.2f m"
              % (i + 1, args.n, cp[0], cp[1], bp[0], bp[1], dist), flush=True)

    side = int(np.ceil(np.sqrt(len(tiles))))
    h, w = tiles[0].shape[:2]
    sheet = np.zeros((side * h, side * w, 3), dtype=np.uint8)
    for n, t in enumerate(tiles):
        r, c = divmod(n, side)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    cv2.imwrite(args.out, sheet[:, :, ::-1])
    print("wrote %s (%dx%d)" % (args.out, sheet.shape[1], sheet.shape[0]), flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
