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
    ap.add_argument("--index", default="0",
                    help="which demo to replay: an integer, a key like demo_7, or 'list' to just "
                         "print what the file contains and exit. Generated datasets hold hundreds "
                         "of demos in one file, so demo_0 is rarely the one you want.")
    ap.add_argument("--camera", default=None,
                    help="'x,y,z:tx,ty,tz' to override the overview camera eye:target. The default "
                         "framing was set for the dispose-trash counter and can miss a can placed "
                         "at the far end of the kitchen run.")
    args = ap.parse_args()

    f = h5py.File(args.demo, "r")

    demo_keys = sorted(f["data"], key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k)
    if args.index == "list":
        print("%s holds %d demos:" % (args.demo, len(demo_keys)))
        for k in demo_keys:
            n = f["data"][k].attrs.get("num_samples", "?")
            print("  %-10s %s steps" % (k, n))
        return
    key = args.index if args.index.startswith("demo_") else "demo_%s" % int(args.index)
    if key not in f["data"]:
        raise SystemExit("%s not in %s -- it holds %d demos (%s ... %s). Use --index list."
                         % (key, args.demo, len(demo_keys), demo_keys[0], demo_keys[-1]))

    env_args = json.loads(f["data"].attrs["env_args"])
    cfg = env_args.get("env_kwargs", env_args)
    # strip wrapper-level keys; keep the raw OmniGibson config
    og_cfg = {k: cfg[k] for k in ("env", "render", "scene", "robots", "objects", "task") if k in cfg}

    import omnigibson as og
    from omnigibson.macros import gm
    gm.ENABLE_TRANSITION_RULES = False
    env = og.Environment(configs=og_cfg)
    env.reset()

    d = f["data"][key]
    # Source demos (DataCollectionWrapper) name these "state"/"action"; generated ones use the
    # plural. Accept either so this replays both.
    states = d["states"] if "states" in d else d["state"]
    actions = (d["actions"] if "actions" in d else d["action"])[:]
    print("replaying %s: %d steps" % (key, actions.shape[0]), flush=True)
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
        eye, tgt = (7.2, -3.2, 3.2), (4.6, -0.8, 0.6)
        if args.camera:
            _e, _t = args.camera.split(":")
            eye = tuple(float(v) for v in _e.split(","))
            tgt = tuple(float(v) for v in _t.split(","))
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1))
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
