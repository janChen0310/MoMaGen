"""Place the trash can at sampled kitchen-walkway positions and render top-down views.

The map-based sampler (analyze_kitchen_walkway.py) only knows the shipped layout PNGs. This
script is the physical check: it teleports the real asset to each sampled pose, lets it
settle, and reports whether it stayed put and upright -- then renders a top-down frame so a
human can eyeball the placement. Run on a box with a working OmniGibson/Isaac.

Usage (server):
  PYTHONPATH=$NY:... OMNIGIBSON_HEADLESS=1 python momagen/scripts/render_trash_can_samples.py \
      --positions kitchen_walkway.json --n 12 --out trash_can_samples
"""
import argparse
import json
import os

import numpy as np
import torch as th

REPO = os.environ.get("MOMAGEN_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRASH = "trash_can_596"
CAN = "can_of_soda_595"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default=os.path.join(REPO, "kitchen_walkway.json"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(REPO, "trash_can_samples"))
    ap.add_argument("--settle-steps", type=int, default=40)
    ap.add_argument("--cam-height", type=float, default=7.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    positions = json.load(open(args.positions))["positions"][: args.n]
    print("SAMPLES=%d" % len(positions), flush=True)

    import omnigibson as og
    from omnigibson.macros import gm
    import imageio

    gm.HEADLESS = True
    from robomimic.utils.file_utils import get_env_metadata_from_dataset
    import momagen.utils.robomimic_utils as RobomimicUtils
    from momagen.utils.robot_config import configure_tidybot_env_meta

    src = os.path.join(REPO, "momagen/datasets/processed_source_demos/tidybot_picking_up_trash.hdf5")
    env_meta = configure_tidybot_env_meta(get_env_metadata_from_dataset(dataset_path=src))
    env = RobomimicUtils.create_env(
        env_meta=env_meta, env_class=None, env_name="tidybot_picking_up_trash_task_D0",
        robot=None, gripper=None, camera_names=[], camera_height=84, camera_width=84,
        render=False, render_offscreen=True, use_image_obs=False, use_depth_obs=False,
        manipulation_only=False, real_robot_mode=False, baseline=None,
    )
    print("ENV_READY", flush=True)

    scene = env.env.scene
    trash = scene.object_registry("name", TRASH)
    can = scene.object_registry("name", CAN)
    assert trash is not None, "trash can not in registry"

    ext = trash.aabb_extent.cpu().numpy() if hasattr(trash.aabb_extent, "cpu") else np.asarray(trash.aabb_extent)
    print("TRASH_AABB_EXTENT=%s (radius~%.3f m, height %.3f m)"
          % (np.round(ext, 3).tolist(), float(max(ext[0], ext[1]) / 2), float(ext[2])), flush=True)
    z0 = float(trash.get_position_orientation()[0][2])
    print("TRASH_Z0=%.3f" % z0, flush=True)

    # Hide the ceiling/roof shells: they sit at z~2.4-3.2, so a top-down camera above them
    # renders the roof and the frame comes back a flat grey blur (learned the hard way).
    hidden = 0
    for obj in scene.objects:
        if obj.name.startswith(("ceilings", "roof")):
            try:
                obj.visible = False
                hidden += 1
            except Exception as exc:
                print("could not hide %s: %s" % (obj.name, exc), flush=True)
    print("HID_CEILING_OBJECTS=%d" % hidden, flush=True)

    # A bright marker directly above the can makes the position unmistakable from above.
    from omnigibson.objects.primitive_object import PrimitiveObject

    marker = PrimitiveObject(
        relative_prim_path="/marker_trash", name="marker_trash", primitive_type="Sphere",
        radius=0.16, rgba=[1.0, 0.05, 0.05, 1.0], visual_only=True, fixed_base=True,
    )
    scene.add_object(marker)
    spawn_marker = PrimitiveObject(
        relative_prim_path="/marker_spawn", name="marker_spawn", primitive_type="Sphere",
        radius=0.16, rgba=[0.05, 0.3, 1.0, 1.0], visual_only=True, fixed_base=True,
    )
    scene.add_object(spawn_marker)
    robot_xy = env.env.robots[0].get_position_orientation()[0]
    robot_xy = robot_xy.cpu().numpy() if hasattr(robot_xy, "cpu") else np.asarray(robot_xy)
    spawn_marker.set_position_orientation(
        position=th.tensor([float(robot_xy[0]), float(robot_xy[1]), 0.30], dtype=th.float32))
    print("ROBOT_XY=%s" % np.round(robot_xy[:2], 3).tolist(), flush=True)

    # top-down camera over the kitchen
    cam = og.sim.viewer_camera
    cam.image_height = cam.image_width = 900
    cam.add_modality("rgb")

    results = []
    for i, xy in enumerate(positions):
        target = th.tensor([float(xy[0]), float(xy[1]), z0], dtype=th.float32)
        trash.set_position_orientation(position=target, orientation=th.tensor([0.0, 0.0, 0.0, 1.0]))
        trash.keep_still()
        for _ in range(args.settle_steps):
            og.sim.step()

        marker.set_position_orientation(
            position=th.tensor([float(xy[0]), float(xy[1]), 0.30], dtype=th.float32))
        p, q = trash.get_position_orientation()
        p = p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)
        q = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
        drift = float(np.linalg.norm(p[:2] - np.asarray(xy)))
        dz = float(p[2] - z0)
        # upright: the body z-axis must still point up (quat w near +/-1 for no tilt)
        tilt_deg = float(np.degrees(2 * np.arccos(min(1.0, abs(float(q[3]))))))
        ok = drift < 0.10 and abs(dz) < 0.05 and tilt_deg < 15.0
        results.append({"i": i, "requested": list(map(float, xy)), "settled": p.tolist(),
                        "drift_m": drift, "dz_m": dz, "tilt_deg": tilt_deg, "ok": bool(ok)})
        print("POS %2d req=(%.2f,%.2f) settled=(%.2f,%.2f,%.3f) drift=%.3f dz=%+.3f tilt=%.1f %s"
              % (i, xy[0], xy[1], p[0], p[1], p[2], drift, dz, tilt_deg, "OK" if ok else "REJECT"),
              flush=True)

        # frame the kitchen from above, looking straight down
        cam.set_position_orientation(
            position=th.tensor([6.6, -0.1, args.cam_height], dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
        )
        for _ in range(3):
            og.sim.render()
        obs = cam.get_obs()
        rgb = obs[0]["rgb"] if isinstance(obs, tuple) else obs["rgb"]
        rgb = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
        imageio.imwrite(os.path.join(args.out, "topdown_%02d.png" % i), rgb[:, :, :3].astype(np.uint8))

    json.dump(results, open(os.path.join(args.out, "placement_results.json"), "w"), indent=1)
    nok = sum(r["ok"] for r in results)
    print("PLACEMENT_SUMMARY ok=%d/%d" % (nok, len(results)), flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
