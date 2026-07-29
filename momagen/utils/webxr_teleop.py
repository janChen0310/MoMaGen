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


# Level at which the (continuous) gripper command counts as CLOSED. 0.5 is the
# midpoint of phone_policy.py's [0, 1] gripper command range.
GRIPPER_CLOSED_LEVEL = 0.5


def latch_gripper(gripper_closed, gripper_delta):
    """Fold one `gripper_delta` into a LATCHING boolean gripper state.

    This reproduces phone_policy.py's `clip(gripper_ref + gripper_delta, 0, 1)`
    (phone_policy.py:197-200) with `gripper_ref` taken from the state the gripper is
    already latched in, then thresholded back to the boolean `CartesianTeleop`
    exposes. Reference-plus-delta is the whole point, and dropping it is a real
    defect rather than a simplification:

        index.html's `handleTouch` sets `touchDeltaY = (touchStartY - clientY)/...`
        and `touchstart` sets `touchStartY = clientY`, so EVERY touch-down sends
        `gripper_delta = 0.0` (index.html:196-208, 260). Thresholding the raw delta
        (`gripper_delta > 0.5`) therefore reads 0.0 as "open" and DROPS whatever the
        operator was holding the moment they lift a finger and re-grip the phone.

    With the reference restored, a 0.0 delta re-asserts the current state instead of
    clearing it: closed stays closed, open stays open. Swiping up past the midpoint
    closes; swiping down past it opens (touchDeltaY is clipped to [-1, 1], so a full
    swipe in either direction always crosses the threshold).

    A non-finite delta (a JSON `null` coordinate becomes NaN under `float()`, and
    `nan > 0.5` is False) would otherwise silently re-open a closed gripper, so it is
    treated like an absent delta: leave the latch alone.
    """
    if gripper_delta is None:
        return gripper_closed
    try:
        delta = float(gripper_delta)
    except (TypeError, ValueError):
        return gripper_closed
    if not np.isfinite(delta):
        return gripper_closed
    ref = 1.0 if gripper_closed else 0.0
    return bool(min(1.0, max(0.0, ref + delta)) > GRIPPER_CLOSED_LEVEL)


def _finite_pose(pos, quat):
    """Return (pos, quat) only if every component is finite, else None.

    A missing key raises KeyError, but a JSON `null` does NOT — `np.array([None, 0.0],
    dtype=float)` yields NaN silently. That NaN flows into `nudge(dpos=[nan, ...])`,
    which NaNs the teleop target PERMANENTLY: every later delta is added to NaN, so
    the arm is dead for the rest of the episode with no exception and nothing in the
    logs. Rejecting the message instead costs one dropped frame at 30 Hz.
    """
    if not (np.isfinite(pos).all() and np.isfinite(quat).all()):
        return None
    return pos, quat


def _xr_pose_from_msg(msg):
    """Read the WebXR client's wire format.

    index.html sends NESTED objects — `data.position = {x, y, z}` and
    `data.orientation = {x, y, z, w}` (index.html:247-257). The flat pos_x/or_x form
    existed only inside tidybot_ros's deleted rclpy publisher, which flattened these
    into a ROS TeleopMsg. Both spellings are accepted so either producer works.
    Returns (pos, quat) or None when the message carries no USABLE pose — missing,
    non-numeric, or non-finite all collapse to None so the robot simply does not move.
    """
    position, orientation = msg.get("position"), msg.get("orientation")
    if isinstance(position, dict) and isinstance(orientation, dict):
        try:
            pos = np.array([position["x"], position["y"], position["z"]], dtype=float)
            quat = np.array(
                [orientation["x"], orientation["y"], orientation["z"], orientation["w"]],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError):
            return None
        return _finite_pose(pos, quat)
    if "pos_x" in msg and "or_w" in msg:
        try:
            pos = np.array([msg["pos_x"], msg["pos_y"], msg["pos_z"]], dtype=float)
            quat = np.array([msg["or_x"], msg["or_y"], msg["or_z"], msg["or_w"]], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
        return _finite_pose(pos, quat)
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
                # delta[1] (drot) is computed and correct (see the tracker) but not
                # applied yet: CartesianTeleop's orientation channel is not wired for
                # the collector in this branch. Kept, not deleted, because orientation
                # control is wanted next; its composition order is pinned by
                # test_drot_is_the_world_order_composition.
                teleop.nudge(dpos=delta[0])

    # The gripper command keeps phone_policy.py's reference-plus-delta semantics
    # (see `latch_gripper`) but collapses the [0, 1] continuum to the BOOLEAN
    # `gripper_closed` that CartesianTeleop exposes. The reference is the state the
    # gripper is already latched in, so a 0.0 delta — which is exactly what every
    # touch-down sends — re-asserts the grasp instead of dropping it.
    #
    # DELIBERATE DEVIATION from phone_policy.py (like the state_update release
    # above): this is applied outside the debounce/anchor gate. The original only
    # reaches its gripper publish after `enable_counts > 2` inside `case "arm"`.
    # Here the latch follows the phone's touch state on every message. That is safe
    # precisely BECAUSE it latches from the current state: there is no reference
    # pose to be captured at the wrong moment and no accumulated drift, so an early
    # message just re-asserts what the operator is already holding.
    teleop.gripper_closed = latch_gripper(teleop.gripper_closed, msg.get("gripper_delta"))
