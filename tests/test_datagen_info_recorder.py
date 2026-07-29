import os

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


INTERFACE_NAME = "MG_TidyBotPickingUpTrash"
INTERFACE_TYPE = "omnigibson_tidybot"


def _shipped_demo():
    """Path to the real processed demo, or skip. Derived from __file__, not the cwd —
    a relative path would make this drift guard silently skip whenever pytest is
    invoked from anywhere but the repo root."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        repo_root, "momagen", "datasets", "processed_source_demos",
        "tidybot_picking_up_trash.hdf5")
    if not os.path.exists(path):
        pytest.skip("shipped demo not present")
    return path


def test_fake_datagen_info_matches_the_real_shipped_schema():
    # Everything else in this module is written against FakeDatagenInfo, so
    # `test_written_file_passes_the_sync_validator` only proves recorder<->validator
    # agreement FOR THE FAKE. Nothing catches the fake drifting away from what
    # OmniGibsonInterfaceTidyBot.get_datagen_info() actually returns — at which point
    # these tests would keep passing while the recorder writes an unusable file.
    # Pin the fake against the real shipped demo, same style as the validator's
    # test_real_shipped_demo_versions_are_readable.
    fake = FakeDatagenInfo(0).to_dict()
    with h5py.File(_shipped_demo(), "r") as f:
        demo = sorted(f["data"].keys())[-1]
        dg = f["data"][demo]["datagen_info"]

        assert set(dg.keys()) == set(fake), (
            f"fake datagen_info fields {sorted(fake)} != real {sorted(dg.keys())}")

        # Per-frame shape: the real datasets are (T, ...) stacks of exactly what
        # to_dict() returns for a single step, which is what recorder.write() builds
        # with np.stack. So real.shape[1:] must equal the fake's per-frame shape.
        for field in ("base_pose", "eef_pose", "gripper_action"):
            assert dg[field].shape[1:] == np.asarray(fake[field]).shape, (
                f"{field}: real per-frame shape {dg[field].shape[1:]} != fake "
                f"{np.asarray(fake[field]).shape}")
        # ...and the documented shapes themselves, so a change on BOTH sides at once
        # still trips something.
        assert dg["base_pose"].shape[1:] == (4, 4)
        assert dg["eef_pose"].shape[1:] == (8, 4)
        assert dg["gripper_action"].shape[1:] == (2,)

        assert isinstance(dg["object_poses"], h5py.Group)
        fake_obj = fake["object_poses"]
        for name in dg["object_poses"]:
            assert dg["object_poses"][name].shape[1:] == (4, 4)
        for name, pose in fake_obj.items():
            # The fake names a real object of this demo rather than an invented one.
            assert name in dg["object_poses"], (
                f"fake object_poses key {name!r} is not in the real demo "
                f"{sorted(dg['object_poses'].keys())}")
            assert dg["object_poses"][name].shape[1:] == np.asarray(pose).shape

        # The interface identity every other test in this module passes to the
        # recorder must be the identity the real file carries, too.
        assert dg.attrs["env_interface_name"] == INTERFACE_NAME
        assert dg.attrs["env_interface_type"] == INTERFACE_TYPE


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
    # Match the guard's own message, not just "any ValueError": np.stack([]) on the empty
    # self._infos list would ALSO raise a ValueError further down in write(), so a bare
    # pytest.raises(ValueError) here would still pass even if the explicit
    # "if not self._infos: raise ValueError(...)" guard were deleted. Pinning the message
    # is what actually proves the guard fires (and fires first).
    with pytest.raises(ValueError, match="no datagen_info recorded"):
        rec.write(str(p))


def test_write_selects_the_last_demo_when_the_file_already_has_one(tmp_path):
    # Both collectors tag mask/use with sorted(...)[-1] (the most recently appended demo)
    # after DataCollectionWrapper.save_data(). write()'s default selection must agree, or
    # a file with more than one demo group (overwrite=False, or a shared file) gets its
    # datagen_info silently attached to the wrong, OLDER demo whenever the action lengths
    # happen to coincide -- exactly what the old `sorted(...)[0]` default did.
    p = tmp_path / "demo.hdf5"
    with h5py.File(p, "w") as f:
        data = f.create_group("data")
        # demo_0: an older demo already in the file, same action length as the new one.
        data.create_group("demo_0").create_dataset(
            "action", data=np.ones((3, 11), dtype=np.float32))
        # demo_1: the demo this recorder's steps actually belong to.
        data.create_group("demo_1").create_dataset(
            "action", data=np.zeros((3, 11), dtype=np.float32))

    rec = DatagenInfoRecorder(FakeInterface(), "MG_TidyBotPickingUpTrash", "omnigibson_tidybot")
    for _ in range(3):
        rec.record(action=np.zeros(11, dtype=np.float32))
    key = rec.write(str(p))

    assert key == "demo_1"
    with h5py.File(p, "r") as f:
        assert "datagen_info" in f["data/demo_1"]
        assert "datagen_info" not in f["data/demo_0"], (
            "write() attached datagen_info to the older demo_0 instead of the new demo_1")


def test_write_with_no_demo_groups_raises_valueerror(tmp_path):
    # sorted(...)[-1] (or [0]) on an empty list raises IndexError, not this module's own
    # ValueError convention -- reachable in practice via only_successes=True discarding a
    # failed episode and leaving zero demo_* groups.
    p = tmp_path / "demo.hdf5"
    with h5py.File(p, "w") as f:
        f.create_group("data")

    rec = DatagenInfoRecorder(FakeInterface(), "MG_TidyBotPickingUpTrash", "omnigibson_tidybot")
    rec.record(action=np.zeros(11, dtype=np.float32))
    with pytest.raises(ValueError, match="no demo_\\* groups"):
        rec.write(str(p))
