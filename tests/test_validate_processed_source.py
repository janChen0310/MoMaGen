import json

import h5py
import numpy as np
import pytest

from momagen.utils.source_demo_validation import validate_processed_source, sync_advisories


def _make_demo(path, T=10, with_state=False, eef_shape=(8, 4),
               objects=("can_of_soda_595",), og_version="3.7.1", extra_versions=None):
    with h5py.File(path, "w") as f:
        d = f.create_group("data")
        d.attrs["env_args"] = '{"env_name": "tidybot_picking_up_trash_task_D0"}'
        # Versions live nested in the scene_file JSON, exactly as OmniGibson writes them.
        versions = {
            "omnigibson": {"version": og_version, "git_hash": "deadbeef"},
            "bddl": {"version": "3.7.0", "git_hash": "deadbeef"},
            "behavior-1k-assets": {"version": "3.7.2rc1"},
        }
        versions.update(extra_versions or {})
        d.attrs["scene_file"] = json.dumps({"versions": versions})
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


# The complete pin for the fixture (and for the real shipped demo): every component
# the file declares must be named, otherwise the unnamed ones go unchecked.
FULL_PIN = {"omnigibson": "3.7.1", "bddl": "3.7.0", "behavior-1k-assets": "3.7.2rc1"}


def test_version_mismatch_rejected(tmp_path):
    # A 3.9.0-authored file fed to the 3.7.1 server is the silent-corruption case.
    p = tmp_path / "ver.hdf5"
    _make_demo(p, og_version="3.9.0")
    problems = validate_processed_source(str(p), expected_versions=FULL_PIN)
    assert any("3.9.0" in x for x in problems)


def test_matching_version_accepted(tmp_path):
    p = tmp_path / "vok.hdf5"
    _make_demo(p, og_version="3.7.1")
    assert validate_processed_source(str(p), expected_versions=FULL_PIN) == []


def test_partial_version_pin_is_rejected(tmp_path):
    # THE silent-corruption path this module exists to close: pinning only the
    # components the caller happened to name lets every OTHER declared component
    # (here behavior-1k-assets, which decides asset hashes and therefore grasp
    # geometry) drift unchecked. OmniGibson 3.7.1 downgrades an asset-hash
    # mismatch to a warning, so an unchecked component fails SILENTLY.
    p = tmp_path / "partial.hdf5"
    _make_demo(p, og_version="3.7.1")
    problems = validate_processed_source(str(p), expected_versions={"omnigibson": "3.7.1"})
    assert problems, "an incomplete --expect-versions must never report VALID"
    joined = " ".join(problems)
    assert "behavior-1k-assets" in joined and "bddl" in joined
    # The operator must be able to read the full pin straight out of the message.
    assert "3.7.2rc1" in joined and "3.7.0" in joined


def test_partial_pin_reports_unchecked_even_when_named_ones_drift(tmp_path):
    # Both failures must be reported together, not one masking the other.
    p = tmp_path / "partial2.hdf5"
    _make_demo(p, og_version="3.9.0")
    problems = validate_processed_source(str(p), expected_versions={"omnigibson": "3.7.1"})
    joined = " ".join(problems)
    assert "3.9.0" in joined and "behavior-1k-assets" in joined


# --- declared-but-unverifiable components ------------------------------------
#
# read_versions only surfaces components whose entry carries a usable "version".
# A component declared with no readable version was therefore invisible to BOTH
# the comparison loop and the completeness set — a second silent-PASS door in the
# same class as the partial pin above. The two shapes are NOT equivalent and are
# treated differently on purpose; see the module docstring on _component_identity.

def test_component_declared_with_only_a_git_hash_is_rejected(tmp_path):
    # The dangerous shape: behavior-1k-assets is declared and pinned to a CONCRETE
    # build by its git_hash, but carries no "version" key. It can differ between
    # file and server, and nothing was checking it.
    p = tmp_path / "hashonly.hdf5"
    _make_demo(p, extra_versions={"behavior-1k-assets": {"git_hash": "cafebabe"}})
    problems = validate_processed_source(
        str(p), expected_versions={"omnigibson": "3.7.1", "bddl": "3.7.0"})
    assert problems, "a declared component with an unreadable version must not PASS"
    joined = " ".join(problems)
    assert "behavior-1k-assets" in joined
    assert "cafebabe" in joined, "the operator needs the identity that IS present"


def test_component_declared_with_only_a_git_hash_is_rejected_even_when_named(tmp_path):
    # Naming it does not help: there is no version in the file to compare against,
    # so the gate must still refuse rather than treat "absent" as "matching".
    p = tmp_path / "hashonly2.hdf5"
    _make_demo(p, extra_versions={"behavior-1k-assets": {"git_hash": "cafebabe"}})
    problems = validate_processed_source(str(p), expected_versions=FULL_PIN)
    assert any("behavior-1k-assets" in x for x in problems)


def test_fully_unidentified_component_does_not_break_a_full_pin(tmp_path):
    # OmniGibson writes {"version": null, "git_hash": null} for a component it
    # cannot identify at all — the SHIPPED file declares omnigibson-robot-assets
    # exactly this way. There is no fact to compare and no string that could ever
    # satisfy a pin, so making it fatal would render --expect-versions permanently
    # unsatisfiable on every file OmniGibson writes. See the advisory test below.
    p = tmp_path / "nullcomp.hdf5"
    _make_demo(p, extra_versions={"omnigibson-robot-assets": {"version": None,
                                                              "git_hash": None}})
    assert validate_processed_source(str(p), expected_versions=FULL_PIN) == []


def test_fully_unidentified_component_is_advised(tmp_path):
    # Not fatal, but never silent: the operator must be told the file declares
    # something --expect-versions can never verify.
    p = tmp_path / "nullcomp2.hdf5"
    _make_demo(p, extra_versions={"omnigibson-robot-assets": {"version": None,
                                                              "git_hash": None}})
    advisories = sync_advisories(str(p))
    assert any("omnigibson-robot-assets" in a for a in advisories)


def test_unidentified_component_advisory_absent_when_all_are_identified(tmp_path):
    p = tmp_path / "allident.hdf5"
    _make_demo(p)
    assert not any("verify" in a for a in sync_advisories(str(p)))


def test_read_versions_keeps_its_usable_only_contract(tmp_path):
    # read_versions is the "what can I compare?" view and must keep dropping
    # unusable entries; read_declared_components is the "what is declared?" view
    # and must drop nothing. The gate needs both.
    from momagen.utils.source_demo_validation import (
        read_declared_components,
        read_versions,
    )

    p = tmp_path / "views.hdf5"
    _make_demo(p, extra_versions={"omnigibson-robot-assets": {"version": None,
                                                              "git_hash": None}})
    with h5py.File(p, "r") as f:
        assert read_versions(f["data"].attrs) == FULL_PIN
        assert set(read_declared_components(f["data"].attrs)) == set(FULL_PIN) | {
            "omnigibson-robot-assets"}


def test_zero_length_demo_rejected(tmp_path):
    # A truncated prepare_src_dataset.py run leaves action (0, 11), eef_pose (0, 8, 4),
    # ... — every shape and length check agrees, so it used to validate clean. A
    # zero-step demo is not syncable.
    p = tmp_path / "zero.hdf5"
    _make_demo(p, T=0)
    problems = validate_processed_source(str(p))
    assert problems, "a zero-step demo must not validate clean"
    assert any("zero" in x.lower() or "0 step" in x.lower() for x in problems)


def test_empty_mask_use_rejected(tmp_path):
    # file_utils.py:57 builds demo_keys from mask/use; an empty one silently
    # generates over zero demos.
    p = tmp_path / "emptymask.hdf5"
    _make_demo(p)
    with h5py.File(p, "a") as f:
        del f["mask/use"]
        f["mask"].create_dataset("use", data=np.zeros((0,), dtype="S16"))
    assert any("mask/use" in x for x in validate_processed_source(str(p)))


def test_mask_use_naming_absent_demo_rejected(tmp_path):
    # mask/use naming demo_7 when only demo_0 exists raises KeyError on the SERVER,
    # after the sync — exactly the failure this validator must catch locally.
    p = tmp_path / "danglingmask.hdf5"
    _make_demo(p)
    with h5py.File(p, "a") as f:
        del f["mask/use"]
        f["mask"].create_dataset("use", data=np.array([b"demo_0", b"demo_7"]))
    problems = validate_processed_source(str(p))
    assert any("demo_7" in x for x in problems)


def test_object_poses_as_dataset_returns_problem_not_traceback(tmp_path):
    # object_poses must be a Group of per-object datasets. Stored as a Dataset it
    # used to raise TypeError ("Only 1D arrays allowed for fancy indexing") — this
    # module promises to return problems, never to raise.
    p = tmp_path / "opdataset.hdf5"
    _make_demo(p)
    with h5py.File(p, "a") as f:
        del f["data/demo_0/datagen_info/object_poses"]
        f["data/demo_0/datagen_info"].create_dataset(
            "object_poses", data=np.zeros((10, 4, 4), dtype=np.float32))
    problems = validate_processed_source(str(p))
    assert any("object_poses" in x for x in problems)


@pytest.mark.parametrize("group_path", [
    "data/demo_0/datagen_info/eef_pose",
    "data/demo_0/datagen_info/gripper_action",
])
def test_group_where_dataset_expected_returns_problem_not_traceback(tmp_path, group_path):
    # Same contract as object_poses above: wrong-kind nodes must be reported, not
    # raised. `.shape` on an h5py.Group is an AttributeError.
    p = tmp_path / "wrongkind.hdf5"
    _make_demo(p)
    with h5py.File(p, "a") as f:
        del f[group_path]
        f.create_group(group_path)
    name = group_path.rsplit("/", 1)[1]
    assert any(name in x for x in validate_processed_source(str(p)))


def test_state_size_only_advisory_does_not_claim_zero_mb(tmp_path):
    # With only state_size present the byte count is unknown; printing "~0.0 MB"
    # tells the operator there is nothing to strip, which is the opposite of true.
    p = tmp_path / "sizeonly.hdf5"
    _make_demo(p)
    with h5py.File(p, "a") as f:
        f["data/demo_0"].create_dataset("state_size", data=np.array([1270]))
    advisories = sync_advisories(str(p))
    assert advisories and "state_size" in advisories[0]
    assert "0.0 MB" not in advisories[0]


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
    # masked the original dead-code version check. The path must be derived from
    # __file__, not cwd — a relative path makes this drift guard silently skip
    # whenever pytest is invoked from anywhere but the repo root.
    import os

    from momagen.utils.source_demo_validation import read_versions

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real = os.path.join(
        repo_root, "momagen", "datasets", "processed_source_demos",
        "tidybot_picking_up_trash.hdf5")
    if not os.path.exists(real):
        pytest.skip("shipped demo not present")
    with h5py.File(real, "r") as f:
        versions = read_versions(f["data"].attrs)
    assert versions is not None and versions.get("omnigibson") == "3.7.1"
    # The fixture's FULL_PIN must stay in step with what the real file declares,
    # otherwise the partial-pin guard above is testing a shape that does not exist.
    assert versions == FULL_PIN


def _shipped_demo():
    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real = os.path.join(
        repo_root, "momagen", "datasets", "processed_source_demos",
        "tidybot_picking_up_trash.hdf5")
    if not os.path.exists(real):
        pytest.skip("shipped demo not present")
    return real


def test_shipped_demo_declares_a_null_version_component():
    # This is the boundary every rule above has to survive: the REAL file declares
    # omnigibson-robot-assets with version AND git_hash both null. Pinned here so
    # that if the shipped file ever stops having this shape, the two boundary tests
    # below are known to have stopped testing anything.
    from momagen.utils.source_demo_validation import read_declared_components

    with h5py.File(_shipped_demo(), "r") as f:
        declared = read_declared_components(f["data"].attrs)
    assert declared["omnigibson-robot-assets"] == {"version": None, "git_hash": None}
    assert set(declared) == set(FULL_PIN) | {"omnigibson-robot-assets"}


def test_shipped_demo_still_validates_with_no_flags():
    assert validate_processed_source(_shipped_demo()) == []


def test_shipped_demo_still_validates_under_the_full_pin():
    # The full pin names the three verifiable components and NOT the null one —
    # exactly the invocation the CLI help tells the operator to use. It must pass.
    assert validate_processed_source(_shipped_demo(), expected_versions=FULL_PIN) == []


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

    full = ",".join(f"{k}={v}" for k, v in FULL_PIN.items())
    match = subprocess.run(
        [sys.executable, script, str(p), "--expect-versions", full],
        capture_output=True, text=True, cwd=repo_root, env=env)
    assert match.returncode == 0 and "VALID" in match.stdout

    mismatch = subprocess.run(
        [sys.executable, script, str(p), "--expect-versions",
         full.replace("omnigibson=3.7.1", "omnigibson=3.9.0")],
        capture_output=True, text=True, cwd=repo_root, env=env)
    assert mismatch.returncode == 1 and "INVALID" in mismatch.stdout

    # An incomplete pin — the exact invocation the old help string suggested —
    # must exit 1, not 0.
    partial = subprocess.run(
        [sys.executable, script, str(p), "--expect-versions", "omnigibson=3.7.1"],
        capture_output=True, text=True, cwd=repo_root, env=env)
    assert partial.returncode == 1 and "INVALID" in partial.stdout


def test_cli_help_example_is_a_complete_pin(tmp_path):
    # The help string is the invocation operators copy. If it demonstrates an
    # incomplete pin it teaches the very mistake this validator now rejects, so
    # the example itself must validate the shipped file's full component set.
    import os
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(repo_root, "momagen", "scripts", "validate_processed_source.py")
    env = dict(os.environ, PYTHONPATH=repo_root)

    helptext = subprocess.run([sys.executable, script, "--help"],
                              capture_output=True, text=True, cwd=repo_root, env=env)
    assert helptext.returncode == 0
    for component in FULL_PIN:
        assert component in helptext.stdout, f"help example omits '{component}'"
