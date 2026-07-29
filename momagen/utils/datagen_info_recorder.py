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
            key = demo_key
            if key is None:
                demo_keys = sorted(k for k in data if k.startswith("demo"))
                if not demo_keys:
                    raise ValueError(
                        f"no demo_* groups found under 'data' in {hdf5_path} — "
                        "nothing to attach datagen_info to")
                # The LAST demo, not the first: both collectors tag mask/use with
                # sorted(...)[-1] after DataCollectionWrapper.save_data() (the most
                # recently appended demo). Defaulting to the first would silently
                # attach datagen_info to the wrong, older demo whenever the hdf5
                # already has prior demos (overwrite=False, or a shared file) and the
                # action lengths happen to coincide -- the length check below cannot
                # catch that, since it validates length, not identity.
                key = demo_keys[-1]
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
