"""Offline tests for kineready.frames -- numpy only, milliseconds, no Isaac.

The base-frame transform is the load-bearing piece of the whole approach: if it is wrong, the
model learns a coherent function of the wrong quantity. That fails silently on IID data (the
labels come from the same wrong transform) and only shows up as garbage once the base moves. So
it is checked against an independent implementation (scipy / explicit inverse), not just against
itself.
"""
import numpy as np
import pytest

from kineready.frames import (
    base_pose_to_matrix,
    encode_features,
    mat_to_quat,
    matrix_to_pose,
    noisy_or,
    perturb_poses,
    pose_to_matrix,
    quat_to_mat,
    rotation_from_6d,
    rotation_to_6d,
    targets_in_base_frame,
)


def _rand_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_targets_in_base_frame_matches_explicit_inverse():
    """The fast analytic inverse must equal the general matrix inverse."""
    rng = np.random.default_rng(0)
    base = rng.uniform(-3, 3, size=(7, 3))
    base[:, 2] = rng.uniform(-np.pi, np.pi, 7)
    targets = np.stack([pose_to_matrix(rng.uniform(-2, 2, 3), _rand_quat(rng)) for _ in range(4)])

    got = targets_in_base_frame(base, targets)
    for i, (x, y, yaw) in enumerate(base):
        Tb_inv = np.linalg.inv(base_pose_to_matrix(x, y, yaw))
        for k in range(len(targets)):
            assert got[i, k] == pytest.approx(Tb_inv @ targets[k], abs=1e-9)


def test_base_frame_target_is_invariant_to_rigid_motion():
    """THE property the factorization rests on.

    Move the base and the target together by the same rigid transform and the target expressed in
    the base frame must not change -- that is exactly why reachability can be a function of this
    9-vector alone. If this fails, the model would have to memorize world coordinates.
    """
    base = np.array([[1.0, -2.0, 0.4]])
    target = pose_to_matrix([1.5, -1.7, 0.9], [0.1, 0.2, 0.3, 0.927])
    ref = targets_in_base_frame(base, target)[0, 0]

    for dx, dy, dyaw in ((3.0, -1.0, 0.0), (0.0, 0.0, 1.1), (-2.5, 4.0, -2.0)):
        c, s = np.cos(dyaw), np.sin(dyaw)
        G = np.eye(4)
        G[:2, :2] = [[c, -s], [s, c]]
        G[:2, 3] = [dx, dy]
        moved_base = np.array([[
            c * base[0, 0] - s * base[0, 1] + dx,
            s * base[0, 0] + c * base[0, 1] + dy,
            base[0, 2] + dyaw,
        ]])
        moved = targets_in_base_frame(moved_base, G @ target)[0, 0]
        assert moved == pytest.approx(ref, abs=1e-9)


def test_yaw_changes_the_base_frame_target():
    """Rotating the base alone must change the query -- the counterpart to the invariance test.

    Without this, a transform that ignored yaw entirely would pass the invariance test.
    """
    target = pose_to_matrix([1.0, 0.0, 0.5], [0, 0, 0, 1])
    a = targets_in_base_frame(np.array([[0.0, 0.0, 0.0]]), target)[0, 0]
    b = targets_in_base_frame(np.array([[0.0, 0.0, np.pi / 2]]), target)[0, 0]
    assert not np.allclose(a, b)
    # a target 1 m ahead becomes 1 m to the right after a +90 deg base yaw
    assert a[:3, 3] == pytest.approx([1.0, 0.0, 0.5], abs=1e-9)
    assert b[:3, 3] == pytest.approx([0.0, -1.0, 0.5], abs=1e-9)


def test_targets_in_base_frame_shapes():
    out = targets_in_base_frame(np.zeros((5, 3)), np.stack([np.eye(4)] * 3))
    assert out.shape == (5, 3, 4, 4)
    single = targets_in_base_frame([0.0, 0.0, 0.0], np.eye(4))
    assert single.shape == (1, 1, 4, 4)


def test_quat_mat_roundtrip_against_scipy():
    scipy_spatial = pytest.importorskip("scipy.spatial.transform")
    R_scipy = scipy_spatial.Rotation
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = _rand_quat(rng)
        assert quat_to_mat(q) == pytest.approx(R_scipy.from_quat(q).as_matrix(), abs=1e-9)
        back = mat_to_quat(quat_to_mat(q))
        if np.dot(back, q) < 0:
            back = -back
        assert back == pytest.approx(q, abs=1e-7)


def test_6d_rotation_roundtrip():
    rng = np.random.default_rng(4)
    for _ in range(20):
        R = quat_to_mat(_rand_quat(rng))
        assert rotation_from_6d(rotation_to_6d(R)) == pytest.approx(R, abs=1e-9)


def test_6d_rotation_is_continuous_across_the_quaternion_sign_flip():
    """q and -q are the same rotation; the 6D encoding must not jump between them."""
    q = _rand_quat(np.random.default_rng(5))
    assert rotation_to_6d(quat_to_mat(q)) == pytest.approx(rotation_to_6d(quat_to_mat(-q)), abs=1e-9)


def test_rotation_from_6d_orthonormalizes_noisy_input():
    R = quat_to_mat(_rand_quat(np.random.default_rng(6)))
    noisy = rotation_to_6d(R) + np.random.default_rng(7).normal(0, 0.05, 6)
    Rr = rotation_from_6d(noisy)
    assert Rr @ Rr.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(Rr) == pytest.approx(1.0, abs=1e-9)


def test_encode_features_layout():
    T = pose_to_matrix([0.3, -0.4, 0.9], [0, 0, 0, 1])
    f = encode_features(T)
    assert f.shape == (9,)
    assert f[:3] == pytest.approx([0.3, -0.4, 0.9])
    assert f[3:] == pytest.approx([1, 0, 0, 0, 1, 0])   # identity: first two columns
    assert encode_features(np.stack([T, T])).shape == (2, 9)
    assert encode_features(targets_in_base_frame(np.zeros((4, 3)), T)).shape == (4, 1, 9)


def test_noisy_or_semantics():
    assert noisy_or([0.0, 0.0]) == pytest.approx(0.0)
    assert noisy_or([1.0, 0.0]) == pytest.approx(1.0)
    assert noisy_or([0.5, 0.5]) == pytest.approx(0.75)
    # more chances can only help, and it stays a probability
    assert noisy_or([0.3, 0.3, 0.3]) > noisy_or([0.3, 0.3])
    assert 0.0 <= noisy_or(np.random.default_rng(8).uniform(size=8)) <= 1.0
    # priors down-weight candidates
    assert noisy_or([0.8], priors=[0.5]) == pytest.approx(0.4)


def test_noisy_or_batches_over_leading_axes():
    p = np.random.default_rng(9).uniform(size=(6, 4))
    out = noisy_or(p, axis=-1)
    assert out.shape == (6,)
    assert out[0] == pytest.approx(noisy_or(p[0]))


def test_perturb_poses_respects_sigma_and_stays_rigid():
    T = pose_to_matrix([0.5, 0.1, 0.8], _rand_quat(np.random.default_rng(10)))
    out = perturb_poses(T, 400, sigma_pos=0.02, sigma_rot=np.deg2rad(5),
                        rng=np.random.default_rng(11))
    assert out.shape == (400, 4, 4)
    disp = np.linalg.norm(out[:, :3, 3] - T[:3, 3], axis=1)
    assert 0.01 < disp.mean() < 0.06                       # ~sigma*sqrt(3)
    for R in out[:20, :3, :3]:
        assert R @ R.T == pytest.approx(np.eye(3), abs=1e-8)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)


def test_perturb_poses_zero_sigma_is_identity():
    T = pose_to_matrix([0.2, 0.2, 0.6], [0, 0, 0, 1])
    out = perturb_poses(T, 5, 0.0, 0.0, rng=np.random.default_rng(12))
    for o in out:
        assert o == pytest.approx(T, abs=1e-12)
