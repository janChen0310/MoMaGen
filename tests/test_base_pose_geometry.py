"""Offline tests for base_pose_metric.geometry -- no Isaac Sim, no GPU, milliseconds.

The projection convention is the thing worth testing hardest. USD cameras look down local -Z, and
using +Z instead mirrors the image and reports objects BEHIND the robot as visible -- which would
produce confident, plausible, entirely wrong visibility scores.
"""
import numpy as np
import pytest

from base_pose_metric.geometry import (
    aabb_sample_points,
    base_pose_to_matrix,
    distance_score,
    intrinsics_from_camera_params,
    look_at_rotation,
    matrix_to_pose,
    pose_to_matrix,
    project_points,
    quat_to_mat,
    score_from_components,
    visible_fraction,
)

W, H = 640, 480
K = np.array([[500.0, 0.0, W / 2.0], [0.0, 500.0, H / 2.0], [0.0, 0.0, 1.0]])


def _cam_at(pos, quat=(0.0, 0.0, 0.0, 1.0)):
    return pose_to_matrix(pos, quat)


def test_identity_camera_looks_down_negative_z():
    """An identity-oriented camera at the origin sees -Z, not +Z."""
    cam = _cam_at([0, 0, 0])
    ahead = project_points(K, cam, [[0.0, 0.0, -2.0]], W, H)
    behind = project_points(K, cam, [[0.0, 0.0, 2.0]], W, H)
    assert ahead["in_front"][0], "point down -Z must be in front"
    assert not behind["in_front"][0], "point down +Z must be behind"
    assert not behind["in_image"][0]


def test_point_straight_ahead_hits_principal_point():
    cam = _cam_at([0, 0, 0])
    res = project_points(K, cam, [[0.0, 0.0, -3.0]], W, H)
    assert res["in_image"][0]
    assert res["pixels"][0] == pytest.approx([W / 2.0, H / 2.0], abs=1e-6)


def test_pixel_axes_have_correct_sign():
    """Camera +X -> image right; camera +Y (up) -> image v decreases."""
    cam = _cam_at([0, 0, 0])
    right = project_points(K, cam, [[0.5, 0.0, -3.0]], W, H)["pixels"][0]
    up = project_points(K, cam, [[0.0, 0.5, -3.0]], W, H)["pixels"][0]
    assert right[0] > W / 2.0, "camera +X should move right in the image"
    assert up[1] < H / 2.0, "camera +Y is up, so image v must decrease"


def test_object_behind_scores_zero_visibility():
    """The case that matters: a base facing away from the object must score 0."""
    cam = _cam_at([0, 0, 0])
    pts = aabb_sample_points([-0.1, -0.1, 1.9], [0.1, 0.1, 2.1])  # squarely behind
    assert visible_fraction(K, cam, pts, W, H) == 0.0


def test_object_ahead_fully_framed_scores_one():
    cam = _cam_at([0, 0, 0])
    pts = aabb_sample_points([-0.1, -0.1, -2.1], [0.1, 0.1, -1.9])
    assert visible_fraction(K, cam, pts, W, H) == 1.0


def test_partial_framing_is_graded():
    """A clipped object scores strictly between 0 and 1 -- this is what makes ranking possible."""
    cam = _cam_at([0, 0, 0])
    # wide box straddling the left image edge at 2 m depth
    pts = aabb_sample_points([-3.0, -0.1, -2.1], [0.05, 0.1, -1.9])
    frac = visible_fraction(K, cam, pts, W, H)
    assert 0.0 < frac < 1.0, f"expected partial visibility, got {frac}"


def test_look_at_rotation_obeys_the_negative_z_convention():
    R = look_at_rotation([1.0, 0.0, 0.0])
    assert (R @ np.array([0.0, 0.0, -1.0])) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert (R @ np.array([0.0, 1.0, 0.0])) == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_base_yaw_rotates_the_view():
    """Rotating the base 180 deg must turn a visible object into an invisible one.

    This is the property the whole metric rests on for a rigidly-mounted camera: visibility is a
    function of base YAW, so yaw and camera aim are not independent.
    """
    mount = np.eye(4)
    mount[:3, :3] = look_at_rotation([1.0, 0.0, 0.0])   # camera looks along the base's +X
    target = aabb_sample_points([1.9, -0.1, -0.1], [2.1, 0.1, 0.1])  # 2 m along +X

    facing = base_pose_to_matrix(0.0, 0.0, 0.0) @ mount
    away = base_pose_to_matrix(0.0, 0.0, np.pi) @ mount
    assert visible_fraction(K, facing, target, W, H) == 1.0
    assert visible_fraction(K, away, target, W, H) == 0.0


def test_pose_matrix_roundtrip():
    pos = np.array([1.5, -2.0, 0.75])
    quat = np.array([0.03154, 0.25689, -0.11772, 0.95873])  # the real camera-C mount orientation
    quat = quat / np.linalg.norm(quat)
    p2, q2 = matrix_to_pose(pose_to_matrix(pos, quat))
    assert p2 == pytest.approx(pos, abs=1e-9)
    if np.dot(q2, quat) < 0:
        q2 = -q2
    assert q2 == pytest.approx(quat, abs=1e-6)


def test_quat_to_mat_is_orthonormal():
    q = np.array([0.03154, 0.25689, -0.11772, 0.95873])
    R = quat_to_mat(q / np.linalg.norm(q))
    assert (R @ R.T) == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_aabb_sample_points_shape_and_centre():
    pts = aabb_sample_points([0, 0, 0], [1, 2, 3])
    assert pts.shape == (9, 3)
    assert pts[-1] == pytest.approx([0.5, 1.0, 1.5])


def test_intrinsics_fallback_matches_expected_fov():
    """36 mm aperture, 24 mm focal, 640 px -> fx = 640*24/36."""
    K2 = intrinsics_from_camera_params(24.0, 36.0, 640, 480)
    assert K2[0, 0] == pytest.approx(640 * 24.0 / 36.0)
    assert K2[0, 2] == pytest.approx(320.0)
    assert K2[1, 2] == pytest.approx(240.0)


def test_distance_score_band():
    band = (0.4, 0.9)
    assert distance_score(0.6, band) == 1.0      # inside
    assert distance_score(0.4, band) == 1.0      # on the edge
    assert distance_score(0.9, band) == 1.0
    assert 0.0 < distance_score(1.1, band) < 1.0  # just outside decays
    assert distance_score(5.0, band) == 0.0      # far outside floors at 0


def test_infeasible_poses_score_zero_regardless_of_visibility():
    """A perfectly-framed pose that collides must not outrank a usable one."""
    for bad in ({"ik_ok": False}, {"collision_static": True}, {"collision_reach": True}):
        kw = {"ik_ok": True, "collision_static": False, "collision_reach": False}
        kw.update(bad)
        score, feasible = score_from_components(
            distance=0.6, visibility_any=1.0, band=(0.4, 0.9), **kw)
        assert score == 0.0 and feasible is False


def test_feasible_score_rewards_visibility():
    a, _ = score_from_components(0.6, 1.0, True, False, False, (0.4, 0.9))
    b, _ = score_from_components(0.6, 0.0, True, False, False, (0.4, 0.9))
    assert a > b
    assert 0.0 <= b < a <= 1.0


def test_batched_frustum_matches_the_scalar_one_exactly():
    """The vectorized frustum must agree with `project_points` bit-for-bit.

    It is a 2.8x optimization of the visibility term, and an optimization that changes the answer is
    a bug. The batched form uses the analytic rigid inverse instead of `np.linalg.inv`, so this also
    pins that the two inverses agree.
    """
    from base_pose_metric.geometry import (aabb_sample_points, base_poses_to_matrices,
                                           in_image_batched, project_points)

    rng = np.random.default_rng(0)
    K = np.array([[174.1, 0, 128.0], [0, 174.1, 128.0], [0, 0, 1.0]])
    W = H = 256
    pts = aabb_sample_points([0.4, -0.3, 0.9], [0.46, -0.24, 0.96])

    poses = np.stack([rng.uniform(-2, 2, 40), rng.uniform(-2, 2, 40),
                      rng.uniform(-np.pi, np.pi, 40)], axis=1)
    mount = np.eye(4)
    mount[:3, 3] = [-0.28, 0.28, 1.269]
    c, s = np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))
    mount[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    T_cams = base_poses_to_matrices(poses) @ mount
    batched = in_image_batched(K, T_cams, pts, W, H)

    for i in range(len(poses)):
        scalar = project_points(K, T_cams[i], pts, W, H)["in_image"]
        assert np.array_equal(batched[i], scalar), "batched frustum diverged at pose %d" % i


def test_base_poses_to_matrices_matches_the_scalar_builder():
    from base_pose_metric.geometry import base_pose_to_matrix, base_poses_to_matrices

    rng = np.random.default_rng(1)
    poses = np.stack([rng.uniform(-3, 3, 25), rng.uniform(-3, 3, 25),
                      rng.uniform(-np.pi, np.pi, 25)], axis=1)
    batched = base_poses_to_matrices(poses)
    for i, (x, y, yaw) in enumerate(poses):
        assert batched[i] == pytest.approx(base_pose_to_matrix(x, y, yaw), abs=1e-15)
