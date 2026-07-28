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


def test_cli_exit_codes_and_expect_versions(tmp_path):
    # Small subprocess smoke test covering main()'s exit-code contract and
    # --expect-versions parsing, neither of which the two library functions
    # above exercise directly (that logic lives only in the CLI's main()).
    import os
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(repo_root, "momagen", "scripts", "validate_processed_source.py")
    env = dict(os.environ, PYTHONPATH=repo_root)

    p = tmp_path / "cli.hdf5"
    _make_demo(p, og_version="3.7.1")

    ok = subprocess.run([sys.executable, script, str(p)],
                        capture_output=True, text=True, cwd=repo_root, env=env)
    assert ok.returncode == 0 and "VALID" in ok.stdout

    match = subprocess.run(
        [sys.executable, script, str(p), "--expect-versions", "omnigibson=3.7.1"],
        capture_output=True, text=True, cwd=repo_root, env=env)
    assert match.returncode == 0 and "VALID" in match.stdout

    mismatch = subprocess.run(
        [sys.executable, script, str(p), "--expect-versions", "omnigibson=3.9.0"],
        capture_output=True, text=True, cwd=repo_root, env=env)
    assert mismatch.returncode == 1 and "INVALID" in mismatch.stdout
