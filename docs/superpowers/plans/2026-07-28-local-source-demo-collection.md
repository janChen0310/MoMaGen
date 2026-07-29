# Local Source-Demo Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MoMaGen source-demo collection comfortable and version-safe — by fixing the remote teleop latency, moving the local↔server sync boundary to the *processed* demo, and replacing keyboard input with the existing tidybot_ros WebXR phone app.

**Architecture:** Collection and `prepare_src_dataset.py` always run *on the same machine under the same OmniGibson*; only a ~5.8 MB processed demo + a 4.6 KB base-config JSON cross the wire. Generation consumes nothing but world-frame SE(3) geometry, so that artifact is version-portable while the raw demo is not. Latency is fixed in-place in the existing web teleop server (three specific defects), and the phone app replaces keystrokes with continuous 6-DOF pose targets. Running OmniGibson locally is deferred to an explicitly gated Phase 2.

**Tech Stack:** Python 3.10 (OmniGibson `momagen` env), OmniGibson 3.7.1 / Isaac Sim 4.5 (server), h5py, `http.server`, Flask + Flask-SocketIO (WebXR app, vendored from tidybot_ros), NumPy, PyTorch.

## Global Constraints

- **Never sync `momagen/datasets/source_og/*.hdf5` (raw demos) between machines.** Raw demos contain `state` (T,1270) + `state_size` — serialized `og.sim.dump_state()` blobs whose layout is version-specific. `prepare_src_dataset.py` replays them through `og.sim.load_state()`.
- **The raw/processed discriminator is `datagen_info`, NOT the absence of `state`** (empirically verified 2026-07-28). A *processed* demo keeps `state`/`state_size` alongside the `datagen_info` group that `prepare_src_dataset.py` adds; a *raw* demo has `state` but no `datagen_info`. Carrying `state` into a processed file is harmless to the server (generation never reads it — it consumes only `datagen_info` + `action.shape[0]`) but it is ~4.94 MB of the 5.8 MB and a latent hazard if any future tool replays it. So: **missing `datagen_info` is fatal; leftover `state` is a warning** advising `make_minimal_source.py` (Task 1), which strips it to ~260 KB.
- **Sync exactly two files:** `momagen/datasets/processed_source_demos/tidybot_<task>.hdf5` and `momagen/datasets/base_configs/tidybot_<task>.json`. Run `momagen/scripts/generate_configs.py` server-side afterward to regenerate absolute paths.
- **Direction matters.** OmniGibson 3.7.1 has *no* version check (`scene_base.py:757-760` is a TODO) and asset-hash mismatch is only `log.warn`. A 3.9.0-authored file fed to the 3.7.1 server fails **silently**. The validator in Task 2 is the only guardrail.
- **`action_frequency` must be 30 on both sides.** The executor steps once per replayed waypoint; a 20 Hz trajectory replays at 2/3 speed in server sim-time.
- **Server is `ubuntu@116.172.96.80`** (`sshpass -p '2aystbz-qzygwc3!' ssh ...`), MoMaGen at `/home/ubuntu/DATA4/backup_root_home/yhu/MoMaGen`, conda env `/home/ubuntu/DATA4/conda/envs/momagen`. **Shared box — never shut down, never touch other users' GPUs/processes.** Always `OMNIGIBSON_HEADLESS=1`.
- **SSH was DOWN at plan time** (`kex_exchange_identification: Connection closed`). Tasks 1, 4 and the Task 3 verification need the server. Task 2 and Task 5 are pure-local and can proceed regardless.
- **Do not introduce ROS into MoMaGen.** ROS 2 Jazzy is Python 3.12; OmniGibson pins 3.10 (ompl cp310 wheel). Port the WebXR *webapp*, not the ROS graph.
- Keep contact-rich source-demo segments **monotone** — MoMaGen replays them open-loop re-anchored, so baked-in corrections do not transfer.

---

## File Structure

**Create:**
- `momagen/scripts/validate_processed_source.py` — schema + version gate for the sync contract. The only defense against silent cross-version corruption.
- `momagen/scripts/make_minimal_source.py` — strips a processed demo to the fields generation actually reads; used to *prove* the contract.
- `momagen/utils/webxr_server.py` — Flask + SocketIO server (vendored from tidybot_ros `phone_teleop_server.py`, rclpy removed).
- `momagen/utils/webxr_teleop.py` — WebXR pose → OmniGibson eef/base delta math (ported from tidybot_ros `phone_policy.py`).
- `momagen/assets/webxr/index.html`, `momagen/assets/webxr/socket.io.min.js` — the phone client, vendored so the phone needs no internet.
- `tests/test_validate_processed_source.py`, `tests/test_webxr_teleop.py` — pure-python unit tests (no simulator).

**Modify:**
- `momagen/scripts/collect_source_web_teleop.py` — the three latency defects (`H` class at :174, MJPEG loop at :186-191, main loop at :343-354).
- `momagen/scripts/collect_tidybot_source_demo.py` — add inline `datagen_info` recording.

**Never modify:** `momagen/datagen/data_generator.py`, `momagen/utils/file_utils.py` (the generation-side contract is already correct; changing it invalidates the 312-demo dataset).

---

### Task 1: Prove the sync contract with a minimal-subfile round trip

Settles the whole plan's foundation for ~260 KB of data and no simulator install. If generation reads a field we planned to drop, we find out now.

**Files:**
- Create: `momagen/scripts/make_minimal_source.py`
- Test: run `generate_dataset.py` on the stripped file (server-side)

**Interfaces:**
- Consumes: existing `momagen/datasets/processed_source_demos/tidybot_picking_up_trash.hdf5`
- Produces: `make_minimal_source.py` CLI — `python make_minimal_source.py --input <path> --output <path>`; keeps `data.attrs`, `mask/`, and per-demo `action` + `datagen_info/{eef_pose,object_poses/*,gripper_action}`.

- [ ] **Step 1: Write the stripping script**

```python
"""Strip a processed source demo to exactly the fields generate_dataset.py reads.

Traced via DataGenerator._load_dataset -> MG_FileUtils.parse_source_dataset_bimanual
(momagen/utils/file_utils.py): only datagen_info/{eef_pose, object_poses/*, gripper_action}
plus action.shape[0] are consumed. state/state_size/scene_file are prepare-time only.
"""
import argparse
import h5py


KEEP_DATAGEN = ("eef_pose", "gripper_action")


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "w") as fout:
        din = fin["data"]
        dout = fout.create_group("data")
        copy_attrs(din, dout)

        if "mask" in fin:
            fin.copy("mask", fout)

        for demo in din:
            gin = din[demo]
            gout = dout.create_group(demo)
            copy_attrs(gin, gout)
            gout.create_dataset("action", data=gin["action"][:])

            din_dg = gin["datagen_info"]
            dout_dg = gout.create_group("datagen_info")
            copy_attrs(din_dg, dout_dg)
            for key in KEEP_DATAGEN:
                dout_dg.create_dataset(key, data=din_dg[key][:])
            op_in = din_dg["object_poses"]
            op_out = dout_dg.create_group("object_poses")
            for obj in op_in:
                op_out.create_dataset(obj, data=op_in[obj][:])
            print(f"{demo}: kept action{gin['action'].shape} "
                  f"eef_pose{din_dg['eef_pose'].shape} objects={list(op_in)}")

    print("WROTE", args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the server and compare sizes**

```bash
NY=/home/ubuntu/DATA4/backup_root_home/yhu/MoMaGen
PY=/home/ubuntu/DATA4/conda/envs/momagen/bin/python
$PY $NY/momagen/scripts/make_minimal_source.py \
  --input  $NY/momagen/datasets/processed_source_demos/tidybot_picking_up_trash.hdf5 \
  --output /tmp/minimal_trash.hdf5
ls -la /tmp/minimal_trash.hdf5 $NY/momagen/datasets/processed_source_demos/tidybot_picking_up_trash.hdf5
```

Expected: minimal file is ~260 KB vs ~5.8 MB. It must print `eef_pose(971, 8, 4)` and objects `['can_of_soda_595', 'trash_can_596']`.

- [ ] **Step 3: Run generation against the stripped file**

Point a copy of the datagen config at `/tmp/minimal_trash.hdf5`, then:

```bash
cd $NY && PYTHONPATH=$NY:$NY/robomimic:$NY/BEHAVIOR-1K/OmniGibson:$NY/BEHAVIOR-1K/bddl \
OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=3 \
$PY -u momagen/scripts/generate_dataset.py \
  --config momagen/datasets/configs/demo_src_tidybot_picking_up_trash_task_D0_minimal.json \
  --num_demos 20 --bimanual --robot_type TidyBot \
  --folder $NY/gen_out_minimal --seed 7001 --auto-remove-exp
```

Expected: PASS = runs to completion with a success rate comparable to the full file (historically ~57%). FAIL = a `KeyError`/`None` on any dropped field — if so, add that field to `KEEP_DATAGEN` and re-run.

- [ ] **Step 4: Commit**

```bash
git add momagen/scripts/make_minimal_source.py
git commit -m "feat: add minimal source-demo stripper to prove the sync contract"
```

---

### Task 2: Sync-contract validator (the silent-corruption guardrail)

Pure python, no simulator, runs in seconds. This is the guardrail OmniGibson itself declined to write.

**Files:**
- Create: `momagen/scripts/validate_processed_source.py`
- Test: `tests/test_validate_processed_source.py`

**Interfaces:**
- Produces: `validate_processed_source(path, expected_versions=None) -> list[str]` returning FATAL problems (empty list == safe to sync), and `sync_advisories(path) -> list[str]` returning non-fatal size/hygiene advice (leftover `state`). Both live in `momagen/utils/source_demo_validation.py`; `momagen/scripts/validate_processed_source.py` is a thin CLI wrapper. CLI: `python validate_processed_source.py <file.hdf5> [--expect-versions omnigibson=3.7.1,bddl=3.7.0]`, exit 0 when clean, exit 1 otherwise.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate_processed_source.py
import json

import h5py
import numpy as np
import pytest

from momagen.utils.source_demo_validation import validate_processed_source, sync_advisories


def _make_demo(path, T=10, with_state=False, eef_shape=(8, 4),
               objects=("can_of_soda_595",), og_version="3.7.1"):
    with h5py.File(path, "w") as f:
        d = f.create_group("data")
        d.attrs["env_args"] = '{"env_name": "tidybot_picking_up_trash_task_D0"}'
        # Versions live nested in the scene_file JSON, exactly as OmniGibson writes them.
        d.attrs["scene_file"] = json.dumps({"versions": {
            "omnigibson": {"version": og_version, "git_hash": "deadbeef"},
            "bddl": {"version": "3.7.0", "git_hash": "deadbeef"},
            "behavior-1k-assets": {"version": "3.7.2rc1"},
        }})
        f.create_group("mask").create_dataset("use", data=np.array([b"demo_0"]))
        g = d.create_group("demo_0")
        g.create_dataset("action", data=np.zeros((T, 11), dtype=np.float32))
        dg = g.create_group("datagen_info")
        dg.attrs["env_interface_name"] = "MG_TidyBotPickingUpTrash"
        dg.attrs["env_interface_type"] = "omnigibson_tidybot"
        dg.create_dataset("eef_pose", data=np.zeros((T,) + eef_shape, dtype=np.float32))
        dg.create_dataset("gripper_action", data=np.zeros((T, 2), dtype=np.float32))
        op = dg.create_group("object_poses")
        for o in objects:
            op.create_dataset(o, data=np.zeros((T, 4, 4), dtype=np.float32))
        if with_state:
            g.create_dataset("state", data=np.zeros((T, 1270), dtype=np.float32))


def test_valid_file_has_no_problems(tmp_path):
    p = tmp_path / "ok.hdf5"
    _make_demo(p)
    assert validate_processed_source(str(p)) == []


def test_raw_demo_missing_datagen_info_is_rejected(tmp_path):
    # A RAW demo is one with no datagen_info group — that, not the presence of
    # `state`, is what makes a file unsafe to sync (verified against the real files).
    p = tmp_path / "raw.hdf5"
    _make_demo(p, with_state=True)
    with h5py.File(p, "a") as f:
        del f["data/demo_0/datagen_info"]
    problems = validate_processed_source(str(p))
    assert any("datagen_info" in x for x in problems)


def test_leftover_state_is_advisory_not_fatal(tmp_path):
    # Processed demos legitimately keep state/state_size (~4.94MB of 5.8MB); generation
    # never reads it. It should be advised away, not rejected.
    p = tmp_path / "withstate.hdf5"
    _make_demo(p, with_state=True)
    assert validate_processed_source(str(p)) == []
    assert any("state" in a for a in sync_advisories(str(p)))


def test_no_advisory_for_stripped_file(tmp_path):
    p = tmp_path / "stripped.hdf5"
    _make_demo(p, with_state=False)
    assert sync_advisories(str(p)) == []


def test_wrong_eef_shape_is_rejected(tmp_path):
    p = tmp_path / "badeef.hdf5"
    _make_demo(p, eef_shape=(4, 4))
    problems = validate_processed_source(str(p))
    assert any("eef_pose" in x for x in problems)


def test_length_mismatch_is_rejected(tmp_path):
    p = tmp_path / "badlen.hdf5"
    _make_demo(p, T=10)
    with h5py.File(p, "a") as f:
        del f["data/demo_0/datagen_info/gripper_action"]
        f["data/demo_0/datagen_info"].create_dataset(
            "gripper_action", data=np.zeros((7, 2), dtype=np.float32))
    problems = validate_processed_source(str(p))
    assert any("length" in x.lower() for x in problems)


def test_missing_objects_rejected(tmp_path):
    p = tmp_path / "noobj.hdf5"
    _make_demo(p, objects=())
    assert any("object_poses" in x for x in validate_processed_source(str(p)))


def test_missing_interface_attrs_rejected(tmp_path):
    p = tmp_path / "noattr.hdf5"
    _make_demo(p)
    with h5py.File(p, "a") as f:
        del f["data/demo_0/datagen_info"].attrs["env_interface_name"]
    assert any("env_interface_name" in x for x in validate_processed_source(str(p)))


def test_version_mismatch_rejected(tmp_path):
    # A 3.9.0-authored file fed to the 3.7.1 server is the silent-corruption case.
    p = tmp_path / "ver.hdf5"
    _make_demo(p, og_version="3.9.0")
    problems = validate_processed_source(str(p), expected_versions={"omnigibson": "3.7.1"})
    assert any("3.9.0" in x for x in problems)


def test_matching_version_accepted(tmp_path):
    p = tmp_path / "vok.hdf5"
    _make_demo(p, og_version="3.7.1")
    assert validate_processed_source(str(p), expected_versions={"omnigibson": "3.7.1"}) == []


def test_unreadable_versions_fail_closed(tmp_path):
    # No scene_file at all: the guardrail must refuse to certify rather than pass.
    p = tmp_path / "nover.hdf5"
    _make_demo(p)
    with h5py.File(p, "a") as f:
        del f["data"].attrs["scene_file"]
    problems = validate_processed_source(str(p), expected_versions={"omnigibson": "3.7.1"})
    assert problems, "must not silently PASS when versions cannot be read"


def test_real_shipped_demo_versions_are_readable():
    # Guards against the fixture drifting from the real on-disk schema, which is what
    # masked the original dead-code version check.
    import os

    from momagen.utils.source_demo_validation import read_versions

    real = "momagen/datasets/processed_source_demos/tidybot_picking_up_trash.hdf5"
    if not os.path.exists(real):
        pytest.skip("shipped demo not present")
    with h5py.File(real, "r") as f:
        versions = read_versions(f["data"].attrs)
    assert versions is not None and versions.get("omnigibson") == "3.7.1"


def test_gripper_action_width_rejected(tmp_path):
    p = tmp_path / "badga.hdf5"
    _make_demo(p, T=10)
    with h5py.File(p, "a") as f:
        del f["data/demo_0/datagen_info/gripper_action"]
        f["data/demo_0/datagen_info"].create_dataset(
            "gripper_action", data=np.zeros((10, 5), dtype=np.float32))
    assert any("gripper_action" in x for x in validate_processed_source(str(p)))


def test_malformed_file_returns_problem_not_traceback(tmp_path):
    p = tmp_path / "empty.hdf5"
    p.write_bytes(b"")
    problems = validate_processed_source(str(p))
    assert problems and any("HDF5" in x for x in problems)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/janchen/Documents/MoMaGen && python -m pytest tests/test_validate_processed_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'momagen.scripts.validate_processed_source'`

- [ ] **Step 3: Write the validator**

```python
"""Validate a PROCESSED source demo before syncing it to the generation server.

Generation reads only world-frame SE(3) geometry, so a processed demo is portable
across OmniGibson versions. A RAW demo is not: its `state` blobs are version-specific
and OmniGibson 3.7.1 has no version gate, so a newer file fails SILENTLY. This script
is that missing gate.
"""
import argparse
import json
import sys

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


def validate_processed_source(path, expected_versions=None):
    """Return a list of problems; empty list means the file is safe to sync."""
    problems = []
    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        return [f"cannot open as HDF5 ({exc.__class__.__name__}: {exc})"]
    with f:
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


def sync_advisories(path):
    """Non-fatal hygiene advice. Leftover `state` is legal in a processed demo (generation
    never reads it) but it is ~95% of the file size, so advise stripping before syncing."""
    advisories = []
    with h5py.File(path, "r") as f:
        for demo in [k for k in f.get("data", {}) if k.startswith("demo")]:
            g = f["data"][demo]
            if "state" in g or "state_size" in g:
                nbytes = g["state"].size * g["state"].dtype.itemsize if "state" in g else 0
                advisories.append(
                    f"{demo}: carries leftover 'state' (~{nbytes / 1e6:.1f} MB) that generation "
                    "never reads — strip it with make_minimal_source.py before syncing")
    return advisories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--expect-versions", default=None,
                    help="comma list, e.g. omnigibson=3.7.1,bddl=3.7.0")
    args = ap.parse_args()

    expected = None
    if args.expect_versions:
        expected = dict(kv.split("=", 1) for kv in args.expect_versions.split(","))

    problems = validate_processed_source(args.path, expected)
    if problems:
        print("INVALID — do not sync:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    for advisory in sync_advisories(args.path):
        print("advisory:", advisory)
    print("VALID — safe to sync:", args.path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/janchen/Documents/MoMaGen && python -m pytest tests/test_validate_processed_source.py -v`
Expected: 14 passed

- [ ] **Step 5: Validate the real shipped demo**

```bash
python momagen/scripts/validate_processed_source.py \
  momagen/datasets/processed_source_demos/tidybot_picking_up_trash.hdf5
```

Expected: `VALID — safe to sync: ...`. If it reports problems on the known-good file, the validator is wrong — fix the validator, not the demo.

- [ ] **Step 6: Commit**

```bash
git add momagen/scripts/validate_processed_source.py tests/test_validate_processed_source.py
git commit -m "feat: add processed-source-demo validator to guard the sync contract"
```

---

### Task 3: Fix the three teleop latency defects

All three defects live in `momagen/scripts/collect_source_web_teleop.py`. The network is healthy
(37.6 ms min RTT, 0% loss over 120 packets) — this is queueing and connection churn, not bandwidth.

**Design note (revised):** the script imports `omnigibson` at module scope and executes its server and
sim loop at import time, so nothing in it can be imported or tested without a simulator — and the teleop
box is not reachable this session, so a live latency measurement is not available either. Rather than
ship three unverifiable performance edits, extract the HTTP/streaming layer into an importable module
with real tests (an ephemeral-port server exercised over actual TCP), then rewire the script to use it.
This mirrors how `momagen/utils/webxr_server.py` was structured, and it is what makes the fixes reviewable.

**Files:**
- Create: `momagen/utils/mjpeg_stream.py`
- Test: `tests/test_mjpeg_stream.py`
- Modify: `momagen/scripts/collect_source_web_teleop.py` (use the module; keep behaviour otherwise identical)

**Interfaces:**
- Produces: `FrameBuffer` — thread-safe newest-frame holder with `publish(jpeg_bytes)`, `latest()` returning
  `(frame_id, bytes)`, and an `encode_worker(encode_fn)` that drains raw frames off the caller's thread.
- Produces: `make_handler(frame_buffer, key_sink, html)` returning a `BaseHTTPRequestHandler` subclass with
  `protocol_version = "HTTP/1.1"`, and `serve(handler_cls, host, port)` returning a started
  `ThreadingHTTPServer` plus a `shutdown()` that genuinely stops it (see the Werkzeug lesson from Task 5:
  a stop path that silently no-ops is worse than none).

**The three defects to fix (each must be pinned by a test):**

1. **HTTP/1.0 with no keep-alive.** `BaseHTTPRequestHandler` defaults to HTTP/1.0, so every one of the
   20-per-second-per-held-key `fetch('/key?k=...')` calls pays a fresh TCP handshake (~40 ms) and competes
   with the MJPEG stream for the browser's ~6-connections-per-host budget. Set `protocol_version = "HTTP/1.1"`
   and send an accurate `Content-Length` on every non-streaming response (HTTP/1.1 requires correct framing —
   a missing/incorrect length hangs the client).
   *Test:* two sequential `/key` requests served over a **single** TCP connection (use
   `http.client.HTTPConnection` without closing between requests) both return 204, and the response advertises
   HTTP/1.1.

2. **Blocking MJPEG writes into a 4 MB kernel send buffer.** At a 5-11 Mbps encode rate, steady-state lag
   becomes `queued_bytes / bandwidth` — potentially seconds — and the existing "newest frame only" guard cannot
   help because the stall is *inside* `write()`. Bound `SO_SNDBUF` to ~1-2 frames, set a socket timeout, drop
   frames on a slow client instead of blocking, and cap the send rate.
   *Test:* `FrameBuffer` returns only the newest frame after many rapid `publish()` calls (no unbounded
   backlog), and the stream handler survives a client that stops reading — i.e. the server does not hang and
   remains able to serve a subsequent request.

3. **Render + JPEG encode inline on the sim's critical path.** `ann.get_data()` (a GPU→CPU readback) plus the
   PIL encode run inside the `env.step()` loop, so encode time is added to every simulation step.
   *Test:* `encode_worker` invokes the encode function off the publishing thread — publish raw frames, assert
   the publishing call returns without having run the (deliberately slow) encode, and that the encoded frame
   appears afterwards.

- [ ] **Step 1: Write the failing tests** in `tests/test_mjpeg_stream.py` covering the three behaviours above,
  plus a `serve()`/`shutdown()` start-stop-start cycle on an ephemeral port (port 0) proving the port is
  genuinely released.

- [ ] **Step 2: Run them and confirm they fail** for the expected reason (module does not exist).

- [ ] **Step 3: Implement `momagen/utils/mjpeg_stream.py`.** Pure stdlib + whatever encoder callable the
  caller passes — it must import with **no** `omnigibson`, `torch`, or PIL dependency so it is testable
  anywhere. The caller supplies the encode function.

- [ ] **Step 4: Run the tests and confirm they pass.**

- [ ] **Step 5: Rewire `collect_source_web_teleop.py`** to use `FrameBuffer` + `make_handler` + `serve`,
  removing the inline `H` class, the inline MJPEG loop, and the inline encode. Behaviour must otherwise be
  unchanged: same `/`, `/key`, `/stream` endpoints, same key semantics, same HTML. Also switch base control
  from the decaying rate command (`base_cmd *= 0.9`) to absolute pose targets so lag cannot integrate into
  positional drift. Since this file cannot be imported without a simulator, verify the rewiring by reading it
  carefully and by `python -m py_compile`; state plainly in your report that it is not runtime-verified.

- [ ] **Step 6: Commit.**

- [ ] **Step 7: Live latency measurement — DEFERRED.** Requires the teleop box, which is not reachable this
  session (the webapp targets the rivermind host, whose per-session password is not available). When it is
  reachable: instrument the loop with wall-clock timers around `env.step()`, `ann.get_data()` and the encode,
  measure keypress-to-motion end-to-end, and compare before/after. **If `env.step()` alone is >100 ms on the
  597-object scene, the simulator is the bottleneck and no transport change will help** — in that case pivot
  to scripted collection rather than optimising the transport further.

### Task 4: Record `datagen_info` inline during collection

This is the step that actually decouples the two machines' OmniGibson versions: it removes
`prepare_src_dataset.py` — the only version-coupled stage — from the critical path.
`get_datagen_info` reads only live sim (`robot.eef_links`, `scene.object_registry`,
`T.pose2mat(...)`) and never touches serialized state, so it is safe to call inside a collection loop.

**Design note (revised after inspecting both collectors):** the recording logic lives in a reusable,
unit-tested helper rather than being inlined, because it must serve two collectors with very different
testability. `collect_tidybot_source_demo.py` is **interactive** (`KeyboardEventHandler`) and cannot be
driven headlessly; `collect_source_scripted_trash.py` is **batch** (scripted waypoints + `env.save_data()`)
and is the script that actually produced the shipped source demo. Wiring the same helper into both lets
the batch collector serve as the end-to-end proof.

**Files:**
- Create: `momagen/utils/datagen_info_recorder.py`
- Test: `tests/test_datagen_info_recorder.py`
- Modify: `momagen/scripts/collect_source_scripted_trash.py` (batch — the verifiable path)
- Modify: `momagen/scripts/collect_tidybot_source_demo.py` (interactive — same helper, wired but not runtime-verified)

**Interfaces:**
- Consumes: `make_interface(name, interface_type, env)` from `momagen/env_interfaces/base.py:19`;
  `env_interface.get_datagen_info(action=...) -> DatagenInfo` whose `.to_dict()` yields
  `base_pose`, `eef_pose`, `object_poses`, `gripper_action`, `subtask_term_signals`.
  For the trash task the interface is `MG_TidyBotPickingUpTrash` / `omnigibson_tidybot`
  (`momagen/env_interfaces/omnigibson.py:1113` and `:634`).
- Produces: `DatagenInfoRecorder(env_interface, interface_name, interface_type)` with
  `.record(action)` (call once per executed `env.step`), `.__len__()`, and
  `.write(hdf5_path, demo_key=None)` which adds a `datagen_info` group to an existing
  per-demo group, including the required attrs.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_datagen_info_recorder.py
import h5py
import numpy as np
import pytest

from momagen.utils.datagen_info_recorder import DatagenInfoRecorder


class FakeDatagenInfo:
    def __init__(self, t):
        self.t = t

    def to_dict(self):
        return {
            "base_pose": np.full((4, 4), self.t, dtype=np.float32),
            "eef_pose": np.full((8, 4), self.t, dtype=np.float32),
            "object_poses": {"can_of_soda_595": np.full((4, 4), self.t, dtype=np.float32)},
            "gripper_action": np.array([self.t, self.t], dtype=np.float32),
        }


class FakeInterface:
    """Stands in for OmniGibsonInterfaceTidyBot — records the actions it was given."""

    def __init__(self):
        self.seen_actions = []
        self.t = 0

    def get_datagen_info(self, action=None):
        self.seen_actions.append(action)
        self.t += 1
        return FakeDatagenInfo(self.t - 1)


def _empty_demo(path, T):
    with h5py.File(path, "w") as f:
        g = f.create_group("data").create_group("demo_0")
        g.create_dataset("action", data=np.zeros((T, 11), dtype=np.float32))


def test_record_collects_one_entry_per_step():
    rec = DatagenInfoRecorder(FakeInterface(), "MG_TidyBotPickingUpTrash", "omnigibson_tidybot")
    for i in range(5):
        rec.record(action=np.zeros(11, dtype=np.float32))
    assert len(rec) == 5


def test_record_passes_action_through():
    iface = FakeInterface()
    rec = DatagenInfoRecorder(iface, "MG_TidyBotPickingUpTrash", "omnigibson_tidybot")
    a = np.arange(11, dtype=np.float32)
    rec.record(action=a)
    assert iface.seen_actions[0] is a, "gripper_action is derived from the action — it must be passed"


def test_write_produces_the_expected_schema(tmp_path):
    p = tmp_path / "demo.hdf5"
    _empty_demo(p, T=3)
    rec = DatagenInfoRecorder(FakeInterface(), "MG_TidyBotPickingUpTrash", "omnigibson_tidybot")
    for _ in range(3):
        rec.record(action=np.zeros(11, dtype=np.float32))
    rec.write(str(p))

    with h5py.File(p, "r") as f:
        dg = f["data/demo_0/datagen_info"]
        assert dg.attrs["env_interface_name"] == "MG_TidyBotPickingUpTrash"
        assert dg.attrs["env_interface_type"] == "omnigibson_tidybot"
        assert dg["eef_pose"].shape == (3, 8, 4)
        assert dg["base_pose"].shape == (3, 4, 4)
        assert dg["gripper_action"].shape == (3, 2)
        assert dg["object_poses"]["can_of_soda_595"].shape == (3, 4, 4)


def test_written_file_passes_the_sync_validator(tmp_path):
    # The whole point: an inline-recorded demo must be syncable WITHOUT prepare_src_dataset.py.
    from momagen.utils.source_demo_validation import validate_processed_source

    p = tmp_path / "demo.hdf5"
    with h5py.File(p, "w") as f:
        d = f.create_group("data")
        d.attrs["env_args"] = '{"env_name": "tidybot_picking_up_trash_task_D0"}'
        f.create_group("mask").create_dataset("use", data=np.array([b"demo_0"]))
        d.create_group("demo_0").create_dataset(
            "action", data=np.zeros((3, 11), dtype=np.float32))

    rec = DatagenInfoRecorder(FakeInterface(), "MG_TidyBotPickingUpTrash", "omnigibson_tidybot")
    for _ in range(3):
        rec.record(action=np.zeros(11, dtype=np.float32))
    rec.write(str(p))

    assert validate_processed_source(str(p)) == []


def test_write_rejects_length_mismatch(tmp_path):
    # Recording fewer steps than the demo has actions would silently corrupt the source.
    p = tmp_path / "demo.hdf5"
    _empty_demo(p, T=5)
    rec = DatagenInfoRecorder(FakeInterface(), "MG_TidyBotPickingUpTrash", "omnigibson_tidybot")
    for _ in range(3):
        rec.record(action=np.zeros(11, dtype=np.float32))
    with pytest.raises(ValueError, match="length"):
        rec.write(str(p))


def test_write_with_no_records_raises(tmp_path):
    p = tmp_path / "demo.hdf5"
    _empty_demo(p, T=0)
    rec = DatagenInfoRecorder(FakeInterface(), "MG_TidyBotPickingUpTrash", "omnigibson_tidybot")
    with pytest.raises(ValueError):
        rec.write(str(p))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. .venv-test/bin/python -m pytest tests/test_datagen_info_recorder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'momagen.utils.datagen_info_recorder'`

- [ ] **Step 3: Implement the recorder**

```python
"""Record `datagen_info` inline during source-demo collection.

This removes `prepare_src_dataset.py` — the only OmniGibson-version-coupled stage — from
the critical path. `env_interface.get_datagen_info()` reads only live sim state (eef links,
the object registry, world poses), never the serialized `og.sim.dump_state()` blob, so it is
safe to call inside a collection loop and its output is portable across OmniGibson versions.
"""
import h5py
import numpy as np


class DatagenInfoRecorder:
    """Collects one DatagenInfo per executed env.step and writes them into the saved hdf5."""

    def __init__(self, env_interface, interface_name, interface_type):
        self._env_interface = env_interface
        self._interface_name = interface_name
        self._interface_type = interface_type
        self._infos = []

    def __len__(self):
        return len(self._infos)

    def record(self, action):
        """Call once per executed env.step, with the action that was executed.

        The action matters: gripper_action is derived from it by the interface.
        """
        self._infos.append(self._env_interface.get_datagen_info(action=action).to_dict())

    def write(self, hdf5_path, demo_key=None):
        """Add a `datagen_info` group to a demo in an already-saved hdf5."""
        if not self._infos:
            raise ValueError("no datagen_info recorded — call record() once per env.step")

        infos = self._infos
        with h5py.File(hdf5_path, "a") as f:
            data = f["data"]
            key = demo_key or sorted(k for k in data if k.startswith("demo"))[0]
            grp = data[key]

            n_actions = grp["action"].shape[0]
            if n_actions != len(infos):
                raise ValueError(
                    f"datagen_info length {len(infos)} != action length {n_actions} for {key}; "
                    "record() must be called exactly once per executed env.step")

            if "datagen_info" in grp:
                del grp["datagen_info"]
            dg = grp.create_group("datagen_info")
            # Required: MG_FileUtils.get_env_interface_info_from_dataset reads these off the
            # datagen_info group (file_utils.py:92); generation cannot start without them.
            dg.attrs["env_interface_name"] = self._interface_name
            dg.attrs["env_interface_type"] = self._interface_type

            for field in ("eef_pose", "base_pose", "gripper_action"):
                dg.create_dataset(
                    field, data=np.stack([i[field] for i in infos]).astype(np.float32))

            obj_grp = dg.create_group("object_poses")
            for name in infos[0]["object_poses"]:
                obj_grp.create_dataset(
                    name,
                    data=np.stack([i["object_poses"][name] for i in infos]).astype(np.float32))
        return key
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/janchen/Documents/MoMaGen && PYTHONPATH=. .venv-test/bin/python -m pytest tests/test_datagen_info_recorder.py -v`
Expected: 6 passed

- [ ] **Step 5: Wire into both collectors**

In each collector, build the interface once before the loop:

```python
from momagen.env_interfaces.base import make_interface
from momagen.utils.datagen_info_recorder import DatagenInfoRecorder

ENV_INTERFACE_NAME = "MG_TidyBotPickingUpTrash"
ENV_INTERFACE_TYPE = "omnigibson_tidybot"
recorder = DatagenInfoRecorder(
    make_interface(name=ENV_INTERFACE_NAME, interface_type=ENV_INTERFACE_TYPE, env=env),
    ENV_INTERFACE_NAME, ENV_INTERFACE_TYPE)
```

Call `recorder.record(action=<the action just executed>)` immediately after **every** `env.step(...)`
that the wrapper records, and after `env.save_data()` call `recorder.write(<output hdf5 path>)`.

The length check in `write()` is the guard that the `record` calls and the wrapper's recorded steps
stayed in lockstep — if it raises, find the `env.step` that was not paired with a `record`.

- [ ] **Step 6: Commit**

```bash
git add momagen/utils/datagen_info_recorder.py tests/test_datagen_info_recorder.py \
        momagen/scripts/collect_source_scripted_trash.py momagen/scripts/collect_tidybot_source_demo.py
git commit -m "feat: record datagen_info inline during collection to decouple OmniGibson versions"
```

- [ ] **Step 7: End-to-end proof on the server (batch collector only)**

Deploy `collect_source_scripted_trash.py`, the recorder, and the validator to the server; run the
scripted collector headless; then **without running `prepare_src_dataset.py` at all**:
1. `validate_processed_source.py <collected>.hdf5` → must be `VALID`.
2. Run `generate_dataset.py --num_demos 5` against it → must reach arm-MP/replay without a
   `KeyError` on any datagen_info field.
This is the claim of the whole task: a freshly collected demo is directly generation-ready.

## Phase 2 (GATED): local OmniGibson on Isaac Sim 5.1

**Do not start this until the gate below passes.** It is a multi-week port, not an install.

**Why it is gated:** Isaac Sim 4.5 — which this repo pins — *architecturally cannot* run on this machine's Blackwell GPU. BEHAVIOR-1K collaborator wensi-ai, closing issue #2270 on 2026-07-01: *"Blackwell GPUs are not supported on Isaac Sim 4.5 track because of underlying simulator constraints. There's no workaround."* This repo's own `docker/Dockerfile:10-11` already says so. The only supported path is BEHAVIOR-1K v3.9.0 / Isaac Sim 5.1 / Python 3.11 — which means porting the fork's 477-line `TidyBot` robot class across upstream PR #1905, which *deleted* `HolonomicBaseRobot` and `MobileManipulationRobot`, the classes TidyBot inherits from.

**Also missing locally:** `tidybot.usda` does not exist on this machine (`find /home/janchen -iname 'tidybot*.usd*'` returns nothing; only two CuRobo YAMLs are present). The only copy with the camera-C mount edit lives on the server.

- [ ] **GATE — Isaac 5.1 Blackwell smoke test (90 minutes). Do this BEFORE the 35 GB asset download and before any porting.**

```bash
conda create -n og391 python=3.11 -y && conda activate og391
OMNI_KIT_ACCEPT_EULA=YES pip install 'isaacsim[all,extscache]==5.1.0.0' \
  --extra-index-url https://pypi.nvidia.com
# Boot a stock BEHAVIOR-1K v3.9.0 clone with a trivial scene, cameras ON, for 2 minutes.
```

PASS = it renders frames. FAIL = segfault after "app ready" with `createDLSSContext error` (IsaacSim #643 — the exact `nvidia-driver-580-open` + Ubuntu 24.04 + Blackwell configuration running here), or an indefinite hang at 100% CPU (IsaacLab #4951).

**If it fails:** retry once on the *proprietary* 580 driver instead of `580-open` — the cheapest mitigation. If it still fails, **the local-sim path is dead**: there is no version below 5.1 (upstream never tagged an Isaac-5.0 release, no v3.8.x exists) and Isaac 6.0 needs driver 595.58.03, itself a known Blackwell crash branch. Stay on 580.x.

- [ ] **If the gate passes:** re-express the TidyBot fork as a v3.9.0 YAML robot definition against `definition_schema.py` — do **not** rebase 149 commits across 575. Then run the geometric parity probe: load the same scene instance on both machines and diff the 4×4 world poses of `can_of_soda_595`, `trash_can_596`, and the robot `eef_link`. Any offset that does not cancel in the re-anchor `new_eef = cur_obj_pose · inv(src_obj_pose) · src_eef_pose` silently displaces every grasp target.

---

## Deferred Work / Notes

- **Scripted sources may be the real answer.** `docs/tutorials/tidybot-task-pipelines.md` §1.3 states the shipped trash source demo was *scripted* (`collect_source_scripted_trash.py`), not teleoperated, and that scripted sources are "more repeatable than teleop and produce cleaner replay segments." Because MoMaGen replays contact segments open-loop re-anchored, human teleop wiggle is actively harmful — and scripted collection is a batch job, indifferent to both the 40 ms link and the SSH outages seen at plan time. If the goal is *more demos* rather than *better teleop feel*, Tasks 1-4 plus scripted sources may be the whole answer and Task 5 becomes optional polish.
- **Settled-pose utility.** The scene-instance JSON puts `can_of_soda_595` at z=0.947 but the settled recorded pose is z=0.9101 — a 37 mm delta against a 5 mm settling threshold in `data_generator.py`. Any offline waypoint authoring needs a short server-side load-and-settle job that dumps exact settled poses to JSON.
- **Server reachability.** SSH failed at `kex_exchange_identification` at plan time. Set a 15-minute retry loop; if the box does not return within 24h, escalate — it holds the only copy of the camera-C `tidybot.usda`.
