# momagen/utils/webxr_teleop.py
"""WebXR phone pose -> OmniGibson teleop deltas.

Ported from tidybot_ros src/tidybot_policy/tidybot_policy/phone_policy.py. The pose
conversion and the >2 debounce are reproduced from that proven implementation rather
than re-derived. Deltas are anchored to a live robot observation, so mount offsets and
tool-length differences (Hand-E here vs 2F-85 there) cancel and need no calibration.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# Offset from the device camera to the device centre, so rotations pivot about the
# device centre. Verbatim from phone_policy.py (measured for an iPad).
DEVICE_CAMERA_OFFSET = np.array([0.0, 0.085, -0.12])


def convert_webxr_pose(pos, quat):
    """WebXR (+x right, +y up, +z back) -> robot (+x fwd, +y left, +z up).

    Returns (position, scipy Rotation). Verbatim port of phone_policy.convert_webxr_pose.
    """
    pos = np.array([-pos[2], -pos[0], pos[1]], dtype=np.float64)
    rot = R.from_quat([-quat[2], -quat[0], quat[1], quat[3]])
    pos = pos + rot.apply(DEVICE_CAMERA_OFFSET)
    return pos, rot


class WebXRDeltaTracker:
    """Turns absolute phone poses into deltas expressed in the ANCHOR's frame.

    `enable_threshold` reproduces tidybot_ros's `enable_counts > 2` debounce, which
    prevents a jump on touch-down. Expressing the delta in the anchor frame (rather
    than the world frame) is what makes the mapping independent of how the operator
    happens to be holding the phone when they engage.
    """

    def __init__(self, enable_threshold=2):
        self.enable_threshold = enable_threshold
        self.count = 0
        self.ref_pos = None
        self.ref_rot = None

    def release(self):
        """Finger lifted / episode boundary: drop the anchor so the next touch re-anchors."""
        self.count = 0
        self.ref_pos = None
        self.ref_rot = None

    def update(self, pos, quat):
        """Return (dpos, drot) in the anchor frame, or None while debouncing/anchoring."""
        pos, rot = convert_webxr_pose(pos, quat)
        self.count += 1
        if self.count <= self.enable_threshold:
            return None
        if self.ref_pos is None:
            self.ref_pos, self.ref_rot = pos, rot
            return None
        dpos = self.ref_rot.inv().apply(pos - self.ref_pos)
        drot = self.ref_rot.inv() * rot
        return dpos, drot


def apply_webxr(msg, teleop, tracker):
    """Apply one WebXR message to a CartesianTeleop-like object.

    `msg` keys mirror tidybot_ros's TeleopMsg: state_update, teleop_mode,
    pos_x/pos_y/pos_z, or_x/or_y/or_z/or_w, gripper_delta.
    """
    # state_update messages are episode control (started/ended/reset), not motion.
    if msg.get("state_update"):
        tracker.release()
        return

    if not msg.get("teleop_mode"):
        tracker.release()
        return

    if msg.get("teleop_mode") == "arm":
        delta = tracker.update(
            np.array([msg["pos_x"], msg["pos_y"], msg["pos_z"]]),
            np.array([msg["or_x"], msg["or_y"], msg["or_z"], msg["or_w"]]),
        )
        if delta is not None:
            teleop.nudge(dpos=delta[0])

    if msg.get("gripper_delta"):
        teleop.gripper_closed = msg["gripper_delta"] > 0.5
