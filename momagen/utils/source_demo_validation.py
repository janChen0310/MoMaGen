"""Validate a PROCESSED source demo before syncing it to the generation server.

Generation reads only world-frame SE(3) geometry, so a processed demo is portable
across OmniGibson versions. A RAW demo is not: its `state` blobs are version-specific
and OmniGibson 3.7.1 has no version gate, so a newer file fails SILENTLY. This module
is that missing gate.
"""
import json

import h5py


def validate_processed_source(path, expected_versions=None):
    """Return a list of problems; empty list means the file is safe to sync."""
    problems = []
    with h5py.File(path, "r") as f:
        if "data" not in f:
            return ["missing top-level 'data' group"]
        data = f["data"]

        if "env_args" not in data.attrs:
            problems.append("data.attrs['env_args'] missing (generation needs it to build the env)")

        if "mask" not in f or "use" not in f["mask"]:
            problems.append("missing mask/use (generation selects demos through it)")

        if expected_versions:
            raw = data.attrs.get("versions")
            if raw is not None:
                try:
                    got = json.loads(raw if isinstance(raw, str) else raw.decode())
                except Exception:
                    got = {}
                for key, want in expected_versions.items():
                    have = got.get(key)
                    if have is not None and str(have) != str(want):
                        problems.append(
                            f"version mismatch {key}: file has {have}, server expects {want}")

        demos = [k for k in data if k.startswith("demo")]
        if not demos:
            problems.append("no demo_* groups found")

        for demo in demos:
            g = data[demo]
            if "state" in g or "state_size" in g:
                problems.append(
                    f"{demo}: contains raw 'state'/'state_size' — this is a RAW demo. "
                    "Run prepare_src_dataset.py locally and sync the processed file instead.")
            if "action" not in g:
                problems.append(f"{demo}: missing 'action' (its length defines the trajectory)")
                continue
            T = g["action"].shape[0]

            if "datagen_info" not in g:
                problems.append(f"{demo}: missing 'datagen_info' — run prepare_src_dataset.py")
                continue
            dg = g["datagen_info"]

            # MG_FileUtils.get_env_interface_info_from_dataset reads these off the
            # datagen_info group (file_utils.py:92); generation cannot start without them.
            for attr in ("env_interface_name", "env_interface_type"):
                if attr not in dg.attrs:
                    problems.append(f"{demo}: datagen_info.attrs['{attr}'] missing")

            if "eef_pose" not in dg:
                problems.append(f"{demo}: datagen_info/eef_pose missing")
            else:
                shape = dg["eef_pose"].shape
                if len(shape) != 3 or shape[1:] != (8, 4):
                    problems.append(
                        f"{demo}: eef_pose shape {shape}, expected (T, 8, 4) "
                        "(bimanual layout; TidyBot duplicates its single arm)")
                elif shape[0] != T:
                    problems.append(f"{demo}: eef_pose length {shape[0]} != action length {T}")

            if "gripper_action" not in dg:
                problems.append(f"{demo}: datagen_info/gripper_action missing")
            else:
                ga = dg["gripper_action"]
                if ga.shape[0] != T:
                    problems.append(f"{demo}: gripper_action length {ga.shape[0]} != action length {T}")

            if "object_poses" not in dg or len(dg["object_poses"]) == 0:
                problems.append(
                    f"{demo}: datagen_info/object_poses missing or empty — "
                    "generation re-anchors every subtask against these")
            else:
                for obj in dg["object_poses"]:
                    shape = dg["object_poses"][obj].shape
                    if len(shape) != 3 or shape[1:] != (4, 4):
                        problems.append(f"{demo}: object_poses/{obj} shape {shape}, expected (T, 4, 4)")
                    elif shape[0] != T:
                        problems.append(
                            f"{demo}: object_poses/{obj} length {shape[0]} != action length {T}")
    return problems
