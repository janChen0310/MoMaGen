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
