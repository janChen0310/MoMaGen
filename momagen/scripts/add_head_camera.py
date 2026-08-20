"""Add a `head_camera_link` (mast camera) to the TidyBot USD, additively and idempotently.

WHY THIS IS A SCRIPT AND NOT A COMMITTED ASSET
----------------------------------------------
`tidybot.usda` is a 133 MB ASCII file. Committing a copy of it per edit would be hostile to a
repository whose packfile is already 30 GB, and the edit itself is 1.5 KB of structured text. So the
edit travels as code: run this and the asset acquires the camera, run it twice and the second run is
a no-op.

WHAT IT ADDS
------------
A third camera at the rear-left mast pose, leaving the existing two untouched:

    base_camera_link   0.315 m up, pitched 45 deg down   (stock, unchanged)
    arm_camera_link    on the wrist                      (unchanged)
    head_camera_link   1.269 m up, pitched 30 deg down   <-- added here

Base and wrist are left exactly as they are so that nothing downstream shifts: the pi0.5 finetune
data and the generation fleet both consume those two streams.

WHY IT IS NEEDED
----------------
`base_pose_metric` takes max visibility over every camera on the robot, so "can any camera see the
target" is already the semantics. But a can on a counter sits at 0.944 m, and BOTH original cameras
are BELOW it pointing DOWN -- +0.629 m and +0.192 m above them respectively. No pitch or field of
view can recover a target that is behind the image plane. Measured over 1000 sampled base poses,
the can was visible from exactly zero of them. With the head camera: 362 of 1000, and the reward
went from a single tied value to a real 0.939-1.000 range.

THE 0.0688 m OFFSET
-------------------
The fixed joint's `localPos0` is the pose relative to the `base` link, but the top-level Xform's
authored `translate` is that value minus 68.8 mm in z. This is the same frame offset the collision
sphere work ran into. It is confirmed from two directions: base_camera_link has joint z 0.315 with
Xform z 0.2462, and an independently-authored mast variant used 1.2002 = 1.269 - 0.0688.

NOTE ON WHICH TREE
------------------
Assets load from wherever `get_dataset_path("custom_dataset")` resolves, which is NOT necessarily
the tree you are editing code in. Editing the wrong copy is silent and wastes a lot of time. With
no --usd argument this script asks OmniGibson for the real path rather than guessing.
"""
import argparse
import hashlib
import os
import shutil
import sys

MAST_POS = "(-0.28, 0.28, 1.269)"
MAST_ROT = "(0.95872596, 0.03154211, 0.25688985, -0.11771675)"   # R_z(-14 deg) . R_y(+30 deg)
XFORM_TRANSLATE = "(-0.28, 0.28, 1.20020105750858784)"           # joint z minus the 0.0688 offset

JOINT_ANCHOR = '''        def PhysicsFixedJoint "base_base_camera_link_joint" (
            prepend apiSchemas = ["PhysxJointAPI"]
        )
'''

JOINT_NEW = '''        def PhysicsFixedJoint "base_head_camera_link_joint" (
            prepend apiSchemas = ["PhysxJointAPI"]
        )
        {
            rel physics:body0 = </tidybot/base>
            rel physics:body1 = </tidybot/head_camera_link>
            uniform bool physics:excludeFromArticulation = 0
            bool physics:jointEnabled = 1
            point3f physics:localPos0 = %s
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot0 = %s
            quatf physics:localRot1 = (1, 0, 0, 0)
        }

''' % (MAST_POS, MAST_ROT)

LINK_ANCHOR = '''    def Xform "base_camera_link" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
    )
'''

LINK_NEW = '''    def Xform "head_camera_link" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
    )
    {
        quatd xformOp:orient = %s
        double3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = %s
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

        def Camera "Camera"
        {
            float2 clippingRange = (0.01, 1000000)
            float focalLength = 13
            float horizontalAperture = 20.955
            float verticalAperture = 15.2908
            quatd xformOp:orient = (0.5, 0.5, -0.5, -0.5)
            double3 xformOp:scale = (1, 1, 1)
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }
    }

''' % (MAST_ROT, XFORM_TRANSLATE)


def default_usd_path():
    """Ask OmniGibson where the asset actually is, rather than assuming a tree."""
    from omnigibson.utils.asset_utils import get_dataset_path

    return os.path.join(get_dataset_path("custom_dataset"),
                        "objects/robot/tidybot/usd/tidybot.usda")


def insert(text):
    """-> patched text. Raises if an anchor is missing or ambiguous."""
    for name, anchor in (("joint", JOINT_ANCHOR), ("link", LINK_ANCHOR)):
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError("%s anchor matched %d times, expected exactly 1 -- the asset is not "
                               "the layout this script was written against" % (name, n))
    out = text.replace(JOINT_ANCHOR, JOINT_NEW + JOINT_ANCHOR, 1)
    out = out.replace(LINK_ANCHOR, LINK_NEW + LINK_ANCHOR, 1)

    # Insertion only: every original byte must survive, and the existing cameras must not move.
    if len(out) <= len(text):
        raise RuntimeError("nothing was inserted")
    for tag in ('base_base_camera_link_joint', 'bracelet_link_arm_camera_link_joint'):
        if out.count(tag) != text.count(tag):
            raise RuntimeError("clobbered %s" % tag)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", default=None,
                    help="path to tidybot.usda; default asks OmniGibson for the loaded asset")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    usd = args.usd or default_usd_path()
    print("asset: %s" % usd, flush=True)
    if not os.path.exists(usd):
        print("ERROR: no such file", flush=True)
        return 1

    src = open(usd).read()
    if "head_camera_link" in src:
        print("head_camera_link already present -- nothing to do", flush=True)
        return 0

    out = insert(src)
    print("would insert %d bytes (%d -> %d)" % (len(out) - len(src), len(src), len(out)), flush=True)
    if args.dry_run:
        print("dry run, nothing written", flush=True)
        return 0

    backup = usd + ".jc_bak_prehead"
    if not os.path.exists(backup):
        shutil.copy2(usd, backup)
        print("backup: %s" % backup, flush=True)

    tmp = usd + ".tmp_head"
    with open(tmp, "w") as f:
        f.write(out)
    os.replace(tmp, usd)                      # atomic: never leave a half-written 133 MB asset
    print("done. md5 %s" % hashlib.md5(out.encode()).hexdigest(), flush=True)
    print("verify with: python momagen/scripts/verify_head_camera.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
