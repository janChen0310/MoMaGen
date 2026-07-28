import h5py
import numpy as np
import pytest

from momagen.utils.source_demo_validation import validate_processed_source


def _make_demo(path, T=10, with_state=False, eef_shape=(8, 4), objects=("can_of_soda_595",)):
    with h5py.File(path, "w") as f:
        d = f.create_group("data")
        d.attrs["env_args"] = '{"env_name": "tidybot_picking_up_trash_task_D0"}'
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


def test_raw_state_is_rejected(tmp_path):
    p = tmp_path / "raw.hdf5"
    _make_demo(p, with_state=True)
    problems = validate_processed_source(str(p))
    assert any("state" in x for x in problems)


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
    p = tmp_path / "ver.hdf5"
    _make_demo(p)
    with h5py.File(p, "a") as f:
        f["data"].attrs["versions"] = '{"omnigibson": "3.9.0"}'
    problems = validate_processed_source(str(p), expected_versions={"omnigibson": "3.7.1"})
    assert any("3.9.0" in x for x in problems)
