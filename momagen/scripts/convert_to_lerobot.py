# NOTE: standard post-generation delivery step (see docs/tutorials/tidybot-task-pipelines.md):
# MoMaGen generates per-demo hdf5 files (its internal working format, needed for
# datagen_info/replay); this converts them into a LeRobot v3.0 dataset for delivery.
"""Convert MoMaGen per-demo hdf5 files into a LeRobot dataset.

Feature mapping (TidyBot, 20 Hz):
  action              (11,) = [base vx, vy, wz] + 7 arm joint targets + gripper cmd
  observation.state   (11,) = base_qvel(3: vx,vy,wz) + arm_0_qpos(7) + gripper_0_qpos(1)
  observation.velocity(12,) = base_qvel(3) + arm_0_qvel(7) + gripper_0_qvel(2)
  observation.eef_pose (7,) = eef_0_pos(3) + eef_0_quat(4, xyzw)
  observation.images.wrist  = arm_camera_link rgb (alpha dropped)
  observation.images.base   = base_camera_link rgb (alpha dropped)

Depth / segmentation / sim states / datagen_info are NOT representable in the LeRobot
feature set; the raw hdf5s on the generation server remain the archival copy of those.

Usage:
  python convert_to_lerobot.py --inputs DIR_OR_FILE [...] --root OUT_DIR \
      --repo-id local/tidybot_picking_up_trash --task "..." [--fps 20] [--limit N]
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np


def find_obs_key(obs, suffix):
    for k in obs.keys():
        if k.endswith(suffix):
            return k
    raise KeyError(suffix)


def episode_files(inputs):
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.hdf5"))))
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="convert at most N episodes (smoke test)")
    ap.add_argument("--vcodec", default="libsvtav1")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Probe the actual recorded frame size instead of hardcoding it. The trash run recorded
    # 224x224; the grasp_can run records 256x256x4 (alpha included). Declaring 224 while writing
    # 256 makes LeRobot store frames that disagree with their own feature spec, so read it from
    # the data and drop only the alpha channel.
    probe_files = episode_files(args.inputs)
    if not probe_files:
        raise SystemExit("no input hdf5 files matched %s" % (args.inputs,))
    img_hw = None
    for _p in probe_files:
        try:
            with h5py.File(_p, "r") as _f:
                for _dk in _f["data"]:
                    _obs = _f["data"][_dk]["obs"]
                    _k = find_obs_key(_obs, "base_camera_link:Camera:0::rgb")
                    img_hw = tuple(int(v) for v in _obs[_k].shape[1:3])
                    break
            if img_hw:
                break
        except Exception:
            continue
    if img_hw is None:
        raise SystemExit("could not find a base_camera rgb stream in %s" % (probe_files[0],))
    IMG_SHAPE = (img_hw[0], img_hw[1], 3)
    print("recorded image size %dx%d -> feature shape %s" % (img_hw[0], img_hw[1], IMG_SHAPE),
          flush=True)

    features = {
        "action": {"dtype": "float32", "shape": (11,), "names": [
            "base_vx", "base_vy", "base_wz",
            "arm_j1", "arm_j2", "arm_j3", "arm_j4", "arm_j5", "arm_j6", "arm_j7",
            "gripper"]},
        # state mirrors the action layout (pi0.5 finetuning): base VELOCITY, not
        # absolute world pose -- pi0.5 discretises state into text tokens, and world
        # x/y/yaw is memorisable noise; PI's own mobile manipulators used base velocity.
        "observation.state": {"dtype": "float32", "shape": (11,), "names": [
            "base_vx", "base_vy", "base_wz",
            "arm_j1", "arm_j2", "arm_j3", "arm_j4", "arm_j5", "arm_j6", "arm_j7",
            "gripper"]},
        "observation.velocity": {"dtype": "float32", "shape": (12,), "names": [
            "base_vx", "base_vy", "base_wyaw",
            "arm_dj1", "arm_dj2", "arm_dj3", "arm_dj4", "arm_dj5", "arm_dj6", "arm_dj7",
            "gripper_dl", "gripper_dr"]},
        "observation.eef_pose": {"dtype": "float32", "shape": (7,), "names": [
            "x", "y", "z", "qx", "qy", "qz", "qw"]},
        "observation.images.wrist": {"dtype": "video", "shape": IMG_SHAPE,
                                     "names": ["height", "width", "channels"]},
        "observation.images.base": {"dtype": "video", "shape": IMG_SHAPE,
                                    "names": ["height", "width", "channels"]},
    }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=features, root=args.root,
        robot_type="tidybot", use_videos=True, vcodec=args.vcodec,
        image_writer_threads=8,
    )

    files = probe_files
    # Enumerate ALL demos in each hdf5: per-episode tmp files hold 1 demo each, but a
    # merged demo.hdf5 (from a shard that ran to completion) holds many (e.g. 40).
    episodes = []
    bad_files = []
    for path in files:
        try:  # a shard killed mid-write leaves one truncated/corrupt hdf5 (bad object header)
            with h5py.File(path, "r") as f:
                # Numeric, not lexicographic: plain sorted() gives demo_0, demo_1, demo_10,
                # demo_11, ..., demo_2, so LeRobot episode 2 would be demo_10. The mapping has to
                # stay obvious because replaying a trajectory needs the hdf5 demo, not the episode.
                dks = sorted(f["data"].keys(),
                             key=lambda k: (int(k.split("_")[-1])
                                            if k.split("_")[-1].isdigit() else float("inf"), k))
        except Exception as e:
            bad_files.append(path)
            print(f"[skip-file] {os.path.basename(path)} unreadable: {e!r}", flush=True)
            continue
        for dk in dks:
            episodes.append((path, dk))
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"converting {len(episodes)} episodes from {len(files)} files "
          f"(skipped {len(bad_files)} unreadable) -> {args.root}", flush=True)

    saved_map = []   # only episodes that actually made it into the dataset
    repaired_frames = [0]
    for ei, (path, dk) in enumerate(episodes):
        try:
            with h5py.File(path, "r") as f:
                d = f["data"][dk]
                obs = d["obs"]
                actions = np.asarray(d["actions"], dtype=np.float32)
                # Repair the gripper command. During the motion-planned segment MoMaGen only
                # overwrites the gripper slot when an object is attached (waypoint.py ~1440), so
                # while navigating it keeps whatever CuRobo's action generator left there -- in
                # practice the constant 79, for roughly the first fifth of every episode.
                # TidyBot's MultiFingerGripperController runs in "binary" mode, where a preprocessed
                # value > 0 means max-open and anything else max-close, so 79 EXECUTED exactly like
                # +1 and the demos are physically correct. Only the recorded number is wrong, and a
                # policy trained on it would learn to emit 79 while driving and would have its
                # normalisation statistics wrecked by the outlier. Taking the sign reproduces
                # precisely what the simulator did.
                gidx = actions.shape[1] - 1
                n_bad = int(np.sum(np.abs(actions[:, gidx]) > 1.0 + 1e-6))
                if n_bad:
                    repaired_frames[0] += n_bad
                actions[:, gidx] = np.where(actions[:, gidx] > 0, 1.0, -1.0).astype(np.float32)
                state = np.concatenate([
                    np.asarray(obs["base_qvel"], dtype=np.float32),
                    np.asarray(obs["arm_0_qpos"], dtype=np.float32),
                    np.asarray(obs["gripper_0_qpos"], dtype=np.float32)[:, :1]], axis=1)
                vel = np.concatenate([
                    np.asarray(obs["base_qvel"], dtype=np.float32),
                    np.asarray(obs["arm_0_qvel"], dtype=np.float32),
                    np.asarray(obs["gripper_0_qvel"], dtype=np.float32)], axis=1)
                eef = np.concatenate([
                    np.asarray(obs["eef_0_pos"], dtype=np.float32),
                    np.asarray(obs["eef_0_quat"], dtype=np.float32)], axis=1)
                wrist_key = find_obs_key(obs, "arm_camera_link:Camera:0::rgb")
                base_key = find_obs_key(obs, "base_camera_link:Camera:0::rgb")
                wrist = np.asarray(obs[wrist_key])[:, :, :, :3]
                base_im = np.asarray(obs[base_key])[:, :, :, :3]
        except Exception as e:
            print(f"[skip-demo] {os.path.basename(path)}:{dk} unreadable: {e!r}", flush=True)
            continue
        T = actions.shape[0]
        assert state.shape[0] == T and wrist.shape[0] == T, (path, T, state.shape, wrist.shape)
        for t in range(T):
            ds.add_frame({
                "action": actions[t],
                "observation.state": state[t],
                "observation.velocity": vel[t],
                "observation.eef_pose": eef[t],
                "observation.images.wrist": np.ascontiguousarray(wrist[t]),
                "observation.images.base": np.ascontiguousarray(base_im[t]),
                "task": args.task,
            })
        ds.save_episode()
        saved_map.append((path, dk, T))
        print(f"[{ei + 1}/{len(episodes)}] {os.path.basename(path)}:{dk} T={T} saved", flush=True)

    if repaired_frames[0]:
        print("repaired out-of-range gripper commands on %d frames (binary mode: sign is what the "
              "simulator acted on)" % repaired_frames[0], flush=True)

    if hasattr(ds, "finalize"):
        ds.finalize()

    # Write the episode -> (file, demo) map. Without it there is no way back from a LeRobot episode
    # index to the hdf5 demo that replay_demo.py needs.
    try:
        import json as _json
        with open(os.path.join(args.root, "episode_source_map.json"), "w") as _fh:
            _json.dump({"episodes": [{"episode_index": i, "file": os.path.basename(p_),
                                      "path": p_, "demo": d_, "frames": t_}
                                     for i, (p_, d_, t_) in enumerate(saved_map)]},
                       _fh, indent=1)
        print("wrote episode_source_map.json", flush=True)
    except Exception as _e:
        print("could not write episode_source_map.json: %r" % (_e,), flush=True)
        print("finalized", flush=True)
    print("DONE episodes=%d frames=%d root=%s" % (ds.meta.total_episodes, ds.meta.total_frames, args.root), flush=True)


if __name__ == "__main__":
    sys.exit(main())
