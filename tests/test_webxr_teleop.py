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


def test_delta_is_expressed_in_the_anchor_frame():
    t = WebXRDeltaTracker(enable_threshold=0)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    t.update(np.array([1.0, 0.0, 0.0]), q)          # anchor
    dpos, _ = t.update(np.array([1.25, 0.0, 0.0]), q)
    assert np.allclose(dpos, np.array([0.0, -0.25, 0.0]))


def test_rotated_anchor_rotates_the_delta():
    # The delta must be expressed in the ANCHOR's frame, so holding the phone at a
    # different yaw changes the robot-frame direction of the same hand motion.
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = R.from_euler("y", 90, degrees=True).as_quat()

    a = WebXRDeltaTracker(enable_threshold=0)
    a.update(np.array([0.0, 0.0, 0.0]), q0)
    da, _ = a.update(np.array([1.0, 0.0, 0.0]), q0)

    b = WebXRDeltaTracker(enable_threshold=0)
    b.update(np.array([0.0, 0.0, 0.0]), q1)
    db, _ = b.update(np.array([1.0, 0.0, 0.0]), q1)

    assert not np.allclose(da, db), "anchor orientation must affect the delta frame"


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

    t = WebXRDeltaTracker(enable_threshold=0)
    stub = StubTeleop()
    q = dict(or_x=0.0, or_y=0.0, or_z=0.0, or_w=1.0)
    from momagen.utils.webxr_teleop import apply_webxr

    apply_webxr({"teleop_mode": "arm", "pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0, **q}, stub, t)
    apply_webxr({"teleop_mode": "arm", "pos_x": 0.1, "pos_y": 0.0, "pos_z": 0.0, **q}, stub, t)
    assert stub.dpos is not None and np.linalg.norm(stub.dpos) > 0

    apply_webxr({"teleop_mode": "arm", "gripper_delta": 1.0, "pos_x": 0.1, "pos_y": 0.0,
                 "pos_z": 0.0, **q}, stub, t)
    assert stub.gripper_closed is True


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
