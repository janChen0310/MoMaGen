"""Validate a PROCESSED source demo before syncing it to the generation server.

Generation reads only world-frame SE(3) geometry (`datagen_info`), so a processed
demo is portable across OmniGibson versions once that group exists. A RAW demo is
one with no `datagen_info` at all — its only content is the version-specific `state`
blob, and OmniGibson 3.7.1 has no version gate, so a newer file fails SILENTLY if
someone tries to replay that state on the server. This module is that missing gate.

Leftover `state`/`state_size` alongside a complete `datagen_info` is legal (generation
never reads it) but bulky — `sync_advisories` flags that as non-fatal hygiene advice.
"""
import json

import h5py


def read_versions(data_attrs):
    """Extract {component: version} from a demo's attrs.

    Versions are NOT a flat `data.attrs["versions"]` key (verified against the real
    files). They live nested inside the `scene_file` JSON blob that OmniGibson writes:
    scene_file["versions"]["omnigibson"]["version"]. Returns None when unreadable.
    """
    raw = data_attrs.get("scene_file")
    if raw is None:
        return None
    try:
        scene_file = json.loads(raw if isinstance(raw, str) else raw.decode())
    except Exception:
        return None
    versions = scene_file.get("versions")
    if not isinstance(versions, dict):
        return None
    out = {}
    for key, val in versions.items():
        version = val.get("version") if isinstance(val, dict) else val
        if version is not None:
            out[key] = str(version)
    return out


def _open_or_error(path):
    """Open @path as HDF5, or return (None, [problem]) instead of raising.

    Factored out so the CLI's main() can open the file once and reuse the handle for
    both the fatal-problem scan and the advisory scan, instead of each of
    validate_processed_source/sync_advisories opening the file independently.
    """
    try:
        return h5py.File(path, "r"), []
    except OSError as exc:
        return None, [f"cannot open as HDF5 ({exc.__class__.__name__}: {exc})"]


def _scan_problems(f, expected_versions=None):
    """Core fatal-problem scan against an already-open HDF5 file handle."""
    problems = []
    if "data" not in f:
        return ["missing top-level 'data' group"]
    data = f["data"]

    if "env_args" not in data.attrs:
        problems.append("data.attrs['env_args'] missing (generation needs it to build the env)")

    if "mask" not in f or "use" not in f["mask"]:
        problems.append("missing mask/use (generation selects demos through it)")

    if expected_versions:
        got = read_versions(data.attrs)
        if got is None:
            # Fail CLOSED: a guardrail that cannot read the versions must not
            # report VALID, because a false PASS is the failure mode that silently
            # corrupts generation.
            problems.append(
                "cannot determine file versions (no readable scene_file/versions) — "
                "refusing to certify against --expect-versions")
        else:
            for key, want in expected_versions.items():
                have = got.get(key)
                if have is None:
                    problems.append(f"file declares no version for '{key}' (server expects {want})")
                elif str(have) != str(want):
                    problems.append(
                        f"version mismatch {key}: file has {have}, server expects {want}")

    demos = [k for k in data if k.startswith("demo")]
    if not demos:
        problems.append("no demo_* groups found")

    for demo in demos:
        g = data[demo]
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
            if len(ga.shape) != 2 or ga.shape[1] != 2:
                problems.append(f"{demo}: gripper_action shape {ga.shape}, expected (T, 2)")
            elif ga.shape[0] != T:
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


def _scan_advisories(f):
    """Core non-fatal advisory scan against an already-open HDF5 file handle."""
    advisories = []
    for demo in [k for k in f.get("data", {}) if k.startswith("demo")]:
        g = f["data"][demo]
        if "state" in g or "state_size" in g:
            nbytes = g["state"].size * g["state"].dtype.itemsize if "state" in g else 0
            advisories.append(
                f"{demo}: carries leftover 'state' (~{nbytes / 1e6:.1f} MB) that generation "
                "never reads — strip it with make_minimal_source.py before syncing")
    return advisories


def validate_processed_source(path, expected_versions=None):
    """Return a list of problems; empty list means the file is safe to sync."""
    f, open_problems = _open_or_error(path)
    if open_problems:
        return open_problems
    with f:
        return _scan_problems(f, expected_versions)


def sync_advisories(path):
    """Non-fatal hygiene advice. Leftover `state` is legal in a processed demo (generation
    never reads it) but it is ~95% of the file size, so advise stripping before syncing."""
    f, open_problems = _open_or_error(path)
    if open_problems:
        return []
    with f:
        return _scan_advisories(f)
