"""Frame maths for KineReady. numpy only -- no simulator, no torch.

The whole reason a 9-number MLP can answer "is this reachable" is the factorization

    T_E^B = T_B^W(b)^-1 @ T_O^W @ T_E^O

which converts *mobile base placement* into a *fixed-arm reachability query*. Everything the
model sees is a target eef pose expressed in the robot's base frame; where the base happens to
stand in the world is absorbed into that transform. Get this transform wrong and the model learns
a coherent function of the wrong quantity, which looks like a working model on IID data and fails
the moment the base moves -- so this file is tested against scipy independently of the simulator.

The model is object- and task-agnostic: it never sees an object identity, a task, or a scene.
Callers supply whatever eef target their planner produced.
"""
import numpy as np


def quat_to_mat(quat):
    """(x, y, z, w) -> (3, 3). Matches OmniGibson/scipy xyzw ordering."""
    x, y, z, w = np.asarray(quat, dtype=float)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def mat_to_quat(R):
    """(3, 3) -> (x, y, z, w)."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([(R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s, 0.25 / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[2, 1] - R[1, 2]) / s])
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s,
                         (R[0, 2] - R[2, 0]) / s])
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s,
                     (R[1, 0] - R[0, 1]) / s])


def pose_to_matrix(pos, quat):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(quat)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def matrix_to_pose(T):
    T = np.asarray(T, dtype=float)
    return T[:3, 3].copy(), mat_to_quat(T[:3, :3])


def base_pose_to_matrix(x, y, yaw, z=0.0):
    """Planar base pose (x, y, yaw) -> (4, 4) world transform."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = [x, y, z]
    return T


def targets_in_base_frame(base_poses, target_world):
    """The core factorization, batched over base poses AND targets.

    Args:
        base_poses: (N, 3) of (x, y, yaw) -- candidate base placements in the world.
        target_world: (4, 4) or (K, 4, 4) -- desired eef pose(s) in the world, from whatever
            upstream produced them (grasp sampler, demo re-anchoring, task planner).

    Returns:
        (N, K, 4, 4) targets expressed in each candidate's base frame.

    Inverting the base transform analytically rather than with `np.linalg.inv` is not premature
    optimization: this runs N*K times per RL step, and a rigid transform's inverse is exact in
    closed form (R^T, -R^T t) where the general inverse is not.
    """
    b = np.atleast_2d(np.asarray(base_poses, dtype=float))
    T = np.asarray(target_world, dtype=float)
    if T.ndim == 2:
        T = T[None]
    N, K = len(b), len(T)

    c, s = np.cos(b[:, 2]), np.sin(b[:, 2])
    # R_B^W is a yaw rotation, so its inverse is the transpose (yaw negated).
    Rinv = np.zeros((N, 3, 3))
    Rinv[:, 0, 0], Rinv[:, 0, 1] = c, s
    Rinv[:, 1, 0], Rinv[:, 1, 1] = -s, c
    Rinv[:, 2, 2] = 1.0
    t = np.stack([b[:, 0], b[:, 1], np.zeros(N)], axis=1)          # (N, 3)

    out = np.zeros((N, K, 4, 4))
    out[:, :, 3, 3] = 1.0
    out[:, :, :3, :3] = np.einsum("nij,kjl->nkil", Rinv, T[:, :3, :3])
    out[:, :, :3, 3] = np.einsum("nij,nkj->nki", Rinv, T[None, :, :3, 3] - t[:, None, :])
    return out


def rotation_to_6d(R):
    """(..., 3, 3) -> (..., 6): the first two columns, per Zhou et al. (proposal Sec 9).

    Quaternions and Euler angles are discontinuous as functions on SO(3), so a network regressing
    or consuming them has to represent a jump; the 6D form is continuous, which is why the
    proposal specifies it.
    """
    R = np.asarray(R, dtype=float)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rotation_from_6d(r6):
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt. Inverse of `rotation_to_6d` for valid rotations."""
    r6 = np.asarray(r6, dtype=float)
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2p = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2p / np.linalg.norm(a2p, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def encode_features(T_base_frame):
    """(..., 4, 4) target-in-base-frame -> (..., 9) model input [x, y, z, r1..r6]."""
    T = np.asarray(T_base_frame, dtype=float)
    return np.concatenate([T[..., :3, 3], rotation_to_6d(T[..., :3, :3])], axis=-1)


def noisy_or(probs, priors=None, axis=-1):
    """P(at least one candidate feasible) = 1 - prod(1 - pi_k * p_k)  (proposal Sec 11).

    Preferred over max() because K near-misses genuinely are better evidence than one near-miss,
    and the gradient is smooth for downstream reward shaping.
    """
    p = np.asarray(probs, dtype=float)
    if priors is not None:
        p = p * np.asarray(priors, dtype=float)
    return 1.0 - np.prod(np.clip(1.0 - p, 0.0, 1.0), axis=axis)


def perturb_poses(T, n, sigma_pos, sigma_rot, rng=None):
    """`n` perturbed copies of T for the robust label (proposal Sec 6.2).

    Robust reachability asks "is this reachable under the pose uncertainty we actually have",
    which is the useful navigation signal: a target deep in the workspace stays reachable under
    jitter, one on the boundary does not. sigma_pos/sigma_rot should come from the real pipeline's
    uncertainty, not be invented.
    """
    rng = rng or np.random.default_rng()
    T = np.asarray(T, dtype=float)
    out = np.repeat(T[None], n, axis=0)
    out[:, :3, 3] += rng.normal(0.0, sigma_pos, size=(n, 3))
    # Small-angle rotation perturbation via the exponential map, first order.
    w = rng.normal(0.0, sigma_rot, size=(n, 3))
    theta = np.linalg.norm(w, axis=1, keepdims=True)
    axis = np.where(theta > 1e-12, w / np.maximum(theta, 1e-12), np.array([1.0, 0.0, 0.0]))
    for i in range(n):
        a, th = axis[i], float(theta[i, 0])
        Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        dR = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)   # Rodrigues
        out[i, :3, :3] = dR @ out[i, :3, :3]
    return out
