"""Scene-agnostic base-pose scoring for OmniGibson / Isaac Sim.

    from base_pose_metric import BasePoseMetric

    metric  = BasePoseMetric(robot)
    results = metric.evaluate(candidate_poses, target=some_object)

`geometry` is importable on its own (numpy only, no simulator), so the projection and scoring
maths can be unit-tested without booting Isaac.
"""
from .geometry import (  # noqa: F401
    aabb_sample_points,
    base_pose_to_matrix,
    distance_score,
    intrinsics_from_camera_params,
    look_at_rotation,
    matrix_to_pose,
    pose_to_matrix,
    project_points,
    score_from_components,
    visible_fraction,
)

__all__ = [
    "BasePoseMetric",
    "aabb_sample_points",
    "base_pose_to_matrix",
    "distance_score",
    "intrinsics_from_camera_params",
    "look_at_rotation",
    "matrix_to_pose",
    "pose_to_matrix",
    "project_points",
    "score_from_components",
    "visible_fraction",
]


def __getattr__(name):
    # Import the simulator-bound class lazily so `from base_pose_metric import geometry` (and the
    # offline tests) keep working in an environment without OmniGibson installed.
    if name == "BasePoseMetric":
        from .metric import BasePoseMetric
        return BasePoseMetric
    raise AttributeError(name)
