"""Pure-geometry helpers for scoring a mobile base pose. numpy only -- no simulator imports.

Everything here is deliberately free of Isaac Sim / OmniGibson so it can be unit-tested in
milliseconds instead of booting a simulator. The camera projection in particular is the easiest
part of this whole module to get silently backwards (see the -Z note on `project_points`), and a
silently-backwards projection produces plausible-looking visibility numbers that are simply wrong.
"""
import numpy as np

# USD/Isaac cameras look down their local -Z axis with +Y up. A point is therefore IN FRONT of the
# camera when its z coordinate in the camera frame is NEGATIVE, and the perspective divide uses
# -z. Using +z here silently mirrors the image and reports objects behind the robot as visible.
CAMERA_FORWARD_AXIS = np.array([0.0, 0.0, -1.0])


def quat_to_mat(quat):
    """(x, y, z, w) quaternion -> (3, 3) rotation matrix. Matches OmniGibson's xyzw ordering."""
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


def pose_to_matrix(pos, quat):
    """(pos, xyzw quat) -> (4, 4) homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(quat)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def matrix_to_pose(T):
    """(4, 4) -> (pos, xyzw quat)."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x, y, z = (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return T[:3, 3].copy(), np.array([x, y, z, w])


def base_pose_to_matrix(x, y, yaw, z=0.0):
    """A planar base pose (x, y, yaw) -> (4, 4) world transform."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = [x, y, z]
    return T


def look_at_rotation(forward, up=(0.0, 0.0, 1.0)):
    """Camera-to-world rotation for a camera pointing along `forward`.

    Encapsulates the -Z convention so callers never hand-build the matrix: the third column is
    -forward, because the camera looks down its own -Z. Getting this backwards points the camera
    the wrong way while still producing a valid rotation, so it fails silently.
    """
    f = np.asarray(forward, dtype=float)
    f = f / np.linalg.norm(f)
    z_cam = -f
    x_cam = np.cross(np.asarray(up, dtype=float), z_cam)
    n = np.linalg.norm(x_cam)
    if n < 1e-9:                     # forward parallel to up; pick any perpendicular
        x_cam = np.cross(np.array([1.0, 0.0, 0.0]), z_cam)
        n = np.linalg.norm(x_cam)
        if n < 1e-9:
            x_cam = np.cross(np.array([0.0, 1.0, 0.0]), z_cam)
            n = np.linalg.norm(x_cam)
    x_cam /= n
    y_cam = np.cross(z_cam, x_cam)
    return np.stack([x_cam, y_cam, z_cam], axis=1)


def aabb_sample_points(aabb_lo, aabb_hi):
    """The 8 corners of an axis-aligned box plus its centre -> (9, 3).

    Sampling corners rather than just the centroid is what makes visibility graded: a pose that
    frames the whole object scores 1.0, one that clips it scores partially, and the ranking
    between them is meaningful.
    """
    lo = np.asarray(aabb_lo, dtype=float)
    hi = np.asarray(aabb_hi, dtype=float)
    corners = np.array([[xx, yy, zz] for xx in (lo[0], hi[0])
                        for yy in (lo[1], hi[1])
                        for zz in (lo[2], hi[2])])
    return np.vstack([corners, (lo + hi) / 2.0])


def intrinsics_from_camera_params(focal_length, horizontal_aperture, width, height):
    """Pinhole K from USD camera attributes, for when `intrinsic_matrix` is unavailable.

    `VisionSensor.intrinsic_matrix` derives K from the render product's projection matrix, which
    is only populated once the sensor has rendered. This is the analytic fallback so the metric
    can be constructed before any render has happened.
    """
    fx = float(width) * float(focal_length) / float(horizontal_aperture)
    fy = fx  # square pixels: vertical aperture scales with height by the same factor
    return np.array([[fx, 0.0, float(width) / 2.0],
                     [0.0, fy, float(height) / 2.0],
                     [0.0, 0.0, 1.0]])


def project_points(K, T_cam_world, points_world, width, height):
    """Project world points into a camera.

    Args:
        K: (3, 3) intrinsics.
        T_cam_world: (4, 4) camera pose IN THE WORLD (camera-to-world).
        points_world: (N, 3).
        width, height: image size in pixels.

    Returns:
        dict with `in_front` (N,) bool, `in_image` (N,) bool, `pixels` (N, 2) float.
        `in_image` already requires `in_front`, so it is the one to count.
    """
    pts = np.atleast_2d(np.asarray(points_world, dtype=float))
    T_world_cam = np.linalg.inv(np.asarray(T_cam_world, dtype=float))
    cam = (T_world_cam @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]

    # -Z is forward (see CAMERA_FORWARD_AXIS). Depth is therefore -z.
    depth = -cam[:, 2]
    in_front = depth > 1e-6
    safe = np.where(in_front, depth, 1.0)
    u = K[0, 0] * (cam[:, 0] / safe) + K[0, 2]
    v = K[1, 1] * (-cam[:, 1] / safe) + K[1, 2]   # image v grows downward, camera +Y is up
    pixels = np.stack([u, v], axis=1)
    in_image = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return {"in_front": in_front, "in_image": in_image, "pixels": pixels}


def in_image_batched(K, T_cams_world, points_world, width, height):
    """(N, 4, 4) camera poses x (M, 3) points -> (N, M) bool in_image.

    Same arithmetic as `project_points`, vectorized over cameras and without a matrix inverse. Two
    things this avoids, both measured as real costs in a 1000-pose sweep:

      * `np.linalg.inv` on a 4x4, 3000 times. A camera pose is rigid, so its inverse is exact in
        closed form -- cam = R^T (p - t) -- and a transpose costs nothing.
      * per-call `hstack` / `ones` / `stack` allocation churn, which cProfile showed as 9000 calls
        of pure overhead.

    Together with caching the prim-path sets, this took the visibility term from 213 ms to 77 ms
    for 1000 poses x 3 cameras, with bit-identical output.
    """
    T = np.asarray(T_cams_world, dtype=float)
    pts = np.atleast_2d(np.asarray(points_world, dtype=float))
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    d = pts[None, :, :] - t[:, None, :]                              # (N, M, 3)
    cam = np.einsum("nij,nkj->nki", np.swapaxes(R, 1, 2), d)          # R^T d
    depth = -cam[:, :, 2]                                             # -Z is forward
    in_front = depth > 1e-6
    safe = np.where(in_front, depth, 1.0)
    u = K[0, 0] * (cam[:, :, 0] / safe) + K[0, 2]
    v = K[1, 1] * (-cam[:, :, 1] / safe) + K[1, 2]
    return in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def base_poses_to_matrices(poses, z=0.0):
    """(N, 3) of (x, y, yaw) -> (N, 4, 4), built without a Python loop."""
    b = np.atleast_2d(np.asarray(poses, dtype=float))
    c, s = np.cos(b[:, 2]), np.sin(b[:, 2])
    T = np.zeros((len(b), 4, 4))
    T[:, 3, 3] = 1.0
    T[:, 0, 0], T[:, 0, 1] = c, -s
    T[:, 1, 0], T[:, 1, 1] = s, c
    T[:, 2, 2] = 1.0
    T[:, 0, 3], T[:, 1, 3], T[:, 2, 3] = b[:, 0], b[:, 1], z
    return T


def visible_fraction(K, T_cam_world, points_world, width, height):
    """Fraction of `points_world` landing inside the image. 0.0 -> not visible, 1.0 -> fully framed."""
    res = project_points(K, T_cam_world, points_world, width, height)
    return float(np.count_nonzero(res["in_image"])) / float(len(res["in_image"]))


def distance_score(distance, band):
    """1.0 inside the preferred [lo, hi] band, decaying linearly outside it.

    A band rather than "nearer is better": too close and the arm cannot fold to reach, too far and
    it cannot reach at all. The useful poses live in an annulus.
    """
    lo, hi = float(band[0]), float(band[1])
    d = float(distance)
    if lo <= d <= hi:
        return 1.0
    span = max(hi - lo, 1e-6)
    gap = (lo - d) if d < lo else (d - hi)
    return float(max(0.0, 1.0 - gap / span))


DEFAULT_WEIGHTS = {"distance": 0.3, "visibility": 0.7}


def score_from_components(distance, visibility_any, ik_ok, collision_static, collision_reach,
                          band, weights=None):
    """Blend the components into [0, 1].

    Feasibility is a HARD gate, not a weighted term: a pose whose IK fails or that stands inside
    the furniture is not "somewhat good", it is unusable, and letting a high visibility score
    compensate for it would rank unusable poses above usable ones.
    """
    feasible = bool(ik_ok) and not bool(collision_static) and not bool(collision_reach)
    if not feasible:
        return 0.0, False
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    total = sum(w.values()) or 1.0
    score = (w.get("distance", 0.0) * distance_score(distance, band)
             + w.get("visibility", 0.0) * float(visibility_any)) / total
    return float(np.clip(score, 0.0, 1.0)), True
