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
    """Turns absolute phone poses into deltas relative to a touch-down anchor.

    `enable_threshold` reproduces tidybot_ros's `enable_counts > 2` debounce, which
    prevents a jump on touch-down. The position delta is the plain XR-frame
    difference (arm mode, phone_policy.py:175) — NOT rotated into the anchor frame,
    since WebXR's `local` reference space is fixed for the whole session and rotating
    it would make the mapping depend on how the operator happened to be holding the
    phone at touch-down (that rotation is the BASE-mode formula, phone_policy.py:136-138,
    which this tracker does not implement).
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
        """Return (dpos, drot) relative to the anchor, or None while debouncing/anchoring."""
        pos, rot = convert_webxr_pose(pos, quat)
        self.count += 1
        if self.count <= self.enable_threshold:
            return None
        if self.ref_pos is None:
            self.ref_pos, self.ref_rot = pos, rot
            return None
        # ARM mode delta is the plain XR-frame difference — phone_policy.py:175
        # `pos_diff = xr_pos - self.arm_xr_ref_pos  # WebXR delta in XR/world frame`.
        # Do NOT rotate it into the anchor frame: that is the BASE-mode formula
        # (phone_policy.py:136-138) and applying it here would make the mapping depend
        # on how the operator happened to hold the phone at touch-down. WebXR's `local`
        # reference space is fixed for the whole session, so the plain difference is right.
        dpos = pos - self.ref_pos
        # World-frame composition order, matching `xr_quat * arm_xr_ref_rot_inv`
        # (phone_policy.py:180-182). Order matters — these do not commute.
        drot = rot * self.ref_rot.inv()
        return dpos, drot


def _xr_pose_from_msg(msg):
    """Read the WebXR client's wire format.

    index.html sends NESTED objects — `data.position = {x, y, z}` and
    `data.orientation = {x, y, z, w}` (index.html:247-257). The flat pos_x/or_x form
    existed only inside tidybot_ros's deleted rclpy publisher, which flattened these
    into a ROS TeleopMsg. Both spellings are accepted so either producer works.
    Returns (pos, quat) or None when the message carries no pose.
    """
    position, orientation = msg.get("position"), msg.get("orientation")
    if isinstance(position, dict) and isinstance(orientation, dict):
        try:
            pos = np.array([position["x"], position["y"], position["z"]], dtype=float)
            quat = np.array(
                [orientation["x"], orientation["y"], orientation["z"], orientation["w"]],
                dtype=float,
            )
        except KeyError:
            return None
        return pos, quat
    if "pos_x" in msg and "or_w" in msg:
        try:
            pos = np.array([msg["pos_x"], msg["pos_y"], msg["pos_z"]], dtype=float)
            quat = np.array([msg["or_x"], msg["or_y"], msg["or_z"], msg["or_w"]], dtype=float)
        except KeyError:
            return None
        return pos, quat
    return None


def apply_webxr(msg, teleop, tracker):
    """Apply one WebXR message to a CartesianTeleop-like object.

    Tolerates partial/malformed messages: anything without a usable pose simply does
    not move the robot.
    """
    # state_update messages are episode control (started/ended/reset), not motion.
    # NOTE: this is a deliberate deviation from phone_policy.py, which early-returns
    # on state_update without releasing; we release here so an episode boundary always
    # drops the anchor rather than leaving a stale one for the next episode.
    if msg.get("state_update"):
        tracker.release()
        return

    if not msg.get("teleop_mode"):
        tracker.release()
        return

    if msg.get("teleop_mode") == "arm":
        pose = _xr_pose_from_msg(msg)
        if pose is not None:
            delta = tracker.update(pose[0], pose[1])
            if delta is not None:
                teleop.nudge(dpos=delta[0])

    gripper_delta = msg.get("gripper_delta")
    if gripper_delta is not None:
        teleop.gripper_closed = gripper_delta > 0.5
