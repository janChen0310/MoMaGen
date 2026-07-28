# tests/test_webxr_teleop.py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from momagen.utils.webxr_teleop import (
    DEVICE_CAMERA_OFFSET,
    WebXRDeltaTracker,
    convert_webxr_pose,
)


def test_axis_remap_and_camera_offset():
    # Ported verbatim from tidybot_ros phone_policy.convert_webxr_pose:
    # WebXR (+x right, +y up, +z back) -> robot (+x fwd, +y left, +z up) is
    # (x, y, z) -> (-z, -x, y), THEN the device-camera offset is applied in the
    # rotated frame so rotations pivot about the device centre, not its camera.
    pos, rot = convert_webxr_pose(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(pos, np.array([-3.0, -0.915, 1.88]))
    assert np.allclose(rot.as_quat(), np.array([0.0, 0.0, 0.0, 1.0]))


def test_camera_offset_value_matches_source():
    assert np.allclose(DEVICE_CAMERA_OFFSET, np.array([0.0, 0.085, -0.12]))


def test_quaternion_component_swap():
    # rot = R.from_quat([-qz, -qx, qy, qw]); a +90deg WebXR yaw about its +y (up)
    # must become a rotation about the robot +z (up).
    q_in = R.from_euler("y", 90, degrees=True).as_quat()
    _, rot = convert_webxr_pose(np.zeros(3), q_in)
    axis = rot.as_rotvec()
    assert np.linalg.norm(axis) > 1e-6
    assert abs(axis[2]) > 0.99 * np.linalg.norm(axis), f"expected robot-z axis, got {axis}"


def test_debounce_suppresses_first_samples():
    # tidybot_ros acts only when enable_counts > 2 — this prevents a jump on touch-down.
    t = WebXRDeltaTracker()
    q = np.array([0.0, 0.0, 0.0, 1.0])
    assert t.update(np.array([0.0, 0.0, 0.0]), q) is None
    assert t.update(np.array([0.5, 0.0, 0.0]), q) is None
    assert t.update(np.array([1.0, 0.0, 0.0]), q) is None   # 3rd: anchors
    assert t.update(np.array([1.0, 0.0, 0.0]), q) is not None


def test_delta_is_plain_xr_difference():
    # dpos = converted(pos) - converted(anchor), no frame rotation (arm mode).
    t = WebXRDeltaTracker(enable_threshold=0)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    t.update(np.array([1.0, 0.0, 0.0]), q)          # anchor
    dpos, _ = t.update(np.array([1.25, 0.0, 0.0]), q)
    assert np.allclose(dpos, np.array([0.0, -0.25, 0.0]))


def test_delta_independent_of_held_orientation():
    # ARM mode uses the plain XR-frame difference (phone_policy.py:175), so the SAME
    # hand motion must produce the SAME robot-frame delta no matter how the phone is
    # held. (The device-camera offset cancels in the difference.) Rotating the delta
    # into the anchor frame is the BASE-mode formula and would break this.
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = R.from_euler("y", 90, degrees=True).as_quat()

    def delta_with(q):
        t = WebXRDeltaTracker(enable_threshold=0)
        t.update(np.array([1.0, 0.0, 0.0]), q)
        return t.update(np.array([1.25, 0.0, 0.0]), q)[0]

    assert np.allclose(delta_with(q0), np.array([0.0, -0.25, 0.0]))
    assert np.allclose(delta_with(q0), delta_with(q1))


def test_drot_is_the_world_order_composition():
    # `drot` has no consumer yet (nudge() is called with dpos only) but orientation
    # control is wanted once the collector wiring lands, so its correctness is pinned
    # here rather than left dormant and unverified. phone_policy.py:180-182 composes
    # `xr_quat * arm_xr_ref_rot_inv` — WORLD order. These do NOT commute, so the
    # reversed order would silently produce a different rotation for any anchor that
    # is not identity, which is every real touch-down.
    t = WebXRDeltaTracker(enable_threshold=0)
    p = np.zeros(3)
    ref_q = R.from_euler("xyz", [10.0, -25.0, 40.0], degrees=True).as_quat()
    cur_q = R.from_euler("xyz", [-5.0, 15.0, 70.0], degrees=True).as_quat()

    t.update(p, ref_q)                       # anchors
    _, drot = t.update(p, cur_q)

    # Rebuild the expectation from convert_webxr_pose's outputs, so this pins the
    # composition ORDER and not the axis remap (which its own tests cover).
    _, ref_rot = convert_webxr_pose(p, ref_q)
    _, cur_rot = convert_webxr_pose(p, cur_q)
    assert np.allclose(drot.as_matrix(), (cur_rot * ref_rot.inv()).as_matrix())
    # The reversed (body-order) composition must be measurably different, otherwise
    # this test would pass for the wrong implementation too.
    assert not np.allclose(drot.as_matrix(), (ref_rot.inv() * cur_rot).as_matrix())


def test_drot_is_identity_when_orientation_is_unchanged():
    t = WebXRDeltaTracker(enable_threshold=0)
    q = R.from_euler("xyz", [10.0, -25.0, 40.0], degrees=True).as_quat()
    t.update(np.zeros(3), q)
    _, drot = t.update(np.array([0.3, 0.0, 0.0]), q)
    assert np.allclose(drot.magnitude(), 0.0, atol=1e-12)


def test_release_reanchors_instead_of_jumping():
    t = WebXRDeltaTracker(enable_threshold=0)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    t.update(np.array([1.0, 0.0, 0.0]), q)
    t.release()
    assert t.update(np.array([5.0, 0.0, 0.0]), q) is None   # re-anchors, no jump


def test_apply_webxr_moves_teleop_and_gripper():
    class StubTeleop:
        def __init__(self):
            self.dpos = None
            self.gripper_closed = False

        def nudge(self, dpos=None, dori=None):
            self.dpos = dpos

    from momagen.utils.webxr_teleop import apply_webxr

    def wire(x, gripper=None):
        # The real client wire format: nested position/orientation (index.html:247-257)
        msg = {
            "teleop_mode": "arm",
            "position": {"x": x, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
        if gripper is not None:
            msg["gripper_delta"] = gripper
        return msg

    t = WebXRDeltaTracker(enable_threshold=0)
    stub = StubTeleop()
    apply_webxr(wire(0.0), stub, t)      # anchors
    apply_webxr(wire(0.1), stub, t)
    assert stub.dpos is not None and np.linalg.norm(stub.dpos) > 0

    apply_webxr(wire(0.1, gripper=1.0), stub, t)
    assert stub.gripper_closed is True


def test_apply_webxr_accepts_flat_pos_keys_too():
    # The flat spelling is what tidybot_ros's ROS publisher produced; accept both.
    class StubTeleop:
        def __init__(self):
            self.dpos = None
            self.gripper_closed = False

        def nudge(self, dpos=None, dori=None):
            self.dpos = dpos

    from momagen.utils.webxr_teleop import apply_webxr

    def flat(x):
        return {"teleop_mode": "arm", "pos_x": x, "pos_y": 0.0, "pos_z": 0.0,
                "or_x": 0.0, "or_y": 0.0, "or_z": 0.0, "or_w": 1.0}

    t = WebXRDeltaTracker(enable_threshold=0)
    stub = StubTeleop()
    apply_webxr(flat(0.0), stub, t)
    apply_webxr(flat(0.1), stub, t)
    assert stub.dpos is not None


def test_apply_webxr_tolerates_malformed_message():
    # A partial message must not raise — it simply must not move the robot.
    class StubTeleop:
        def __init__(self):
            self.dpos = None
            self.gripper_closed = False

        def nudge(self, dpos=None, dori=None):
            self.dpos = dpos

    from momagen.utils.webxr_teleop import apply_webxr

    t = WebXRDeltaTracker(enable_threshold=0)
    stub = StubTeleop()
    apply_webxr({"teleop_mode": "arm"}, stub, t)                              # no pose
    apply_webxr({"teleop_mode": "arm", "position": {"x": 1.0}}, stub, t)      # partial
    assert stub.dpos is None


def test_state_update_message_releases_and_does_not_move():
    class StubTeleop:
        def __init__(self):
            self.dpos = None
            self.gripper_closed = False

        def nudge(self, dpos=None, dori=None):
            self.dpos = dpos

    from momagen.utils.webxr_teleop import apply_webxr

    t = WebXRDeltaTracker(enable_threshold=0)
    stub = StubTeleop()
    apply_webxr({"state_update": "episode_ended"}, stub, t)
    assert stub.dpos is None


class _StubTeleop:
    def __init__(self):
        self.dpos = None
        self.gripper_closed = False

    def nudge(self, dpos=None, dori=None):
        self.dpos = dpos


@pytest.mark.parametrize("bad", [None, "nan", float("nan"), float("inf")])
def test_non_finite_pose_component_is_rejected(bad):
    # A `None` (or non-numeric) coordinate becomes NaN under `dtype=float`, and NaN
    # propagates into nudge(dpos=[nan, ...]) which NaNs the target PERMANENTLY for
    # the rest of the episode — silent, unrecoverable, and invisible in logs. Only
    # KeyError used to be caught, so this slipped straight through.
    from momagen.utils.webxr_teleop import _xr_pose_from_msg, apply_webxr

    msg = {
        "teleop_mode": "arm",
        "position": {"x": 0.1, "y": bad, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    assert _xr_pose_from_msg(msg) is None

    t = WebXRDeltaTracker(enable_threshold=0)
    stub = _StubTeleop()
    good = {
        "teleop_mode": "arm",
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    apply_webxr(good, stub, t)                       # anchors
    apply_webxr(msg, stub, t)                        # poisoned message
    assert stub.dpos is None, "a non-finite pose must not reach the robot"

    # ...and the anchor must survive it, so the next good message still works.
    moved = dict(good, position={"x": 0.1, "y": 0.0, "z": 0.0})
    apply_webxr(moved, stub, t)
    assert stub.dpos is not None and np.all(np.isfinite(stub.dpos))


def test_non_finite_orientation_component_is_rejected():
    from momagen.utils.webxr_teleop import _xr_pose_from_msg

    msg = {
        "teleop_mode": "arm",
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": None, "z": 0.0, "w": 1.0},
    }
    assert _xr_pose_from_msg(msg) is None


def test_non_finite_flat_pose_is_rejected():
    from momagen.utils.webxr_teleop import _xr_pose_from_msg

    msg = {"teleop_mode": "arm", "pos_x": None, "pos_y": 0.0, "pos_z": 0.0,
           "or_x": 0.0, "or_y": 0.0, "or_z": 0.0, "or_w": 1.0}
    assert _xr_pose_from_msg(msg) is None


def test_gripper_latches_on_the_threshold():
    # Deliberate deviation from phone_policy.py's incremental clip(ref + delta, 0, 1):
    # the collector's CartesianTeleop exposes a BOOLEAN gripper_closed, so the
    # continuous command is latched at 0.5. Pinned here so the deviation is a
    # decision, not a drift.
    from momagen.utils.webxr_teleop import apply_webxr

    t = WebXRDeltaTracker(enable_threshold=0)
    stub = _StubTeleop()
    base = {"teleop_mode": "arm",
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}

    apply_webxr(dict(base, gripper_delta=1.0), stub, t)
    assert stub.gripper_closed is True
    apply_webxr(dict(base, gripper_delta=0.0), stub, t)
    assert stub.gripper_closed is False, "latching, not accumulating"
    apply_webxr(dict(base, gripper_delta=0.5), stub, t)
    assert stub.gripper_closed is False, "0.5 is not > 0.5"

    # Absent gripper_delta must leave the latch alone rather than opening it.
    apply_webxr(dict(base, gripper_delta=1.0), stub, t)
    apply_webxr(base, stub, t)
    assert stub.gripper_closed is True
