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
