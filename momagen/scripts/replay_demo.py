# NOTE: imported from the TidyBot dispose-trash pipeline (see docs/tutorials/tidybot-task-pipelines.md).
# Server-specific absolute paths (e.g. dataset/template locations) may need adjusting to your setup.
"""Minimal trajectory replay for the TidyBot dispose-trash demos.

Usage:
  python replay_demo.py demo_000.hdf5 [--states] [--video out.mp4]

Builds the environment from the demo's embedded env_args, restores the recorded
initial simulator state, then replays recorded actions (default) or exact
per-step states (--states).
"""
import argparse, json, os
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import h5py
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--states", action="store_true", help="state-sync replay instead of action playback")
    ap.add_argument("--video", default=None, help="write an mp4 of the replay (external overview cam)")
    args = ap.parse_args()

    f = h5py.File(args.demo, "r")
    env_args = json.loads(f["data"].attrs["env_args"])
    cfg = env_args.get("env_kwargs", env_args)
    # strip wrapper-level keys; keep the raw OmniGibson config
    og_cfg = {k: cfg[k] for k in ("env", "render", "scene", "robots", "objects", "task") if k in cfg}

    import omnigibson as og
    from omnigibson.macros import gm
    gm.ENABLE_TRANSITION_RULES = False
    env = og.Environment(configs=og_cfg)
    env.reset()

    d = f["data/demo_0"]
    states = d["states"]
    actions = d["actions"][:]
    og.sim.load_state(states[0], serialized=True)
    for _ in range(5):
        og.sim.step()

    writer = None
    if args.video:
        import imageio
        writer = imageio.get_writer(args.video, fps=20)
        import omni.replicator.core as rep
        from pxr import UsdGeom, Gf
        cam = UsdGeom.Camera.Define(og.sim.stage, "/World/replay_cam")
        cam.GetFocalLengthAttr().Set(16.0)
        xf = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(7.2, -3.2, 3.2), Gf.Vec3d(4.6, -0.8, 0.6), Gf.Vec3d(0, 0, 1))
        xf.Set(m.GetInverse())
        rp = rep.create.render_product("/World/replay_cam", (720, 720))
        rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb"); rgb_ann.attach(rp)

    T = actions.shape[0]
    for t in range(T):
        if args.states:
            og.sim.load_state(states[t], serialized=True)
            og.sim.step()
        else:
            env.step(np.asarray(actions[t], dtype=np.float32))
        if writer and t % 2 == 0:
            og.sim.render()
            frame = rgb_ann.get_data()
            if frame is not None and getattr(frame, "size", 0):
                writer.append_data(np.asarray(frame)[..., :3])
        if t % 100 == 0:
            print(f"step {t}/{T}", flush=True)

    succ = None
    try:
        succ = env.task.success
    except Exception:
        pass
    print("replay finished; task success:", succ)
    if writer:
        writer.close()
        print("video:", args.video)
    og.shutdown()

if __name__ == "__main__":
    main()
