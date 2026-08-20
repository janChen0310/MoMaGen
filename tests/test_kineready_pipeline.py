"""Offline tests for the KineReady pipeline: datagen, model, dataset, reward.

No simulator, no GPU, seconds to run. These cover the parts where a bug would be invisible in the
output: a sampler that quietly produces invalid rotations, a split whose "held-out" set overlaps
training, a reward whose score depends on where the robot happens to stand in world coordinates.
"""
import numpy as np
import pytest

th = pytest.importorskip("torch")

from kineready.datagen import (
    _boundary_anchors,
    _downward_biased_rotations,
    _uniform_rotations,
    sample_uniform,
)
from kineready.dataset import geometric_splits
from kineready.frames import pose_to_matrix
from kineready.model import (
    KineReadyEnsemble,
    expected_calibration_error,
    fpr_at_recall,
)
from kineready.reward import ReadinessReward, potential_shaping, readiness


# ------------------------------------------------------------------ samplers

@pytest.mark.parametrize("fn", [_uniform_rotations, _downward_biased_rotations])
def test_rotation_samplers_produce_valid_rotations(fn):
    R = fn(300, np.random.default_rng(0))
    assert R.shape == (300, 3, 3)
    assert np.einsum("nij,nkj->nik", R, R) == pytest.approx(np.tile(np.eye(3), (300, 1, 1)), abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(np.ones(300), abs=1e-9)


def test_downward_bias_actually_points_down_and_uniform_does_not():
    """The two orientation slices must differ, or the 70/30 mix is decoration."""
    rng = np.random.default_rng(1)
    down_z = _downward_biased_rotations(2000, rng)[:, 2, 2]
    unif_z = _uniform_rotations(2000, rng)[:, 2, 2]
    assert down_z.mean() < -0.4          # tool axis genuinely downward
    assert abs(unif_z.mean()) < 0.1      # uniform on SO(3) has no preferred direction
    assert (down_z < 0).mean() > 0.99


def test_sample_uniform_respects_the_box():
    T = sample_uniform(500, np.random.default_rng(2), box=((-1, 1), (-2, 2), (0, 1.5)))
    assert T.shape == (500, 4, 4)
    assert (T[:, 0, 3] >= -1).all() and (T[:, 0, 3] <= 1).all()
    assert (T[:, 1, 3] >= -2).all() and (T[:, 1, 3] <= 2).all()
    assert (T[:, 2, 3] >= 0).all() and (T[:, 2, 3] <= 1.5).all()


def test_boundary_anchors_concentrate_near_the_decision_boundary():
    """The robust slice is only worth its 9x cost if its anchors land where the margin varies.

    Ground truth here is a spherical shell, so "near the boundary" is exactly measurable. Selected
    anchors must sit substantially closer to it than the candidate pool they came from.
    """
    centre, rmax = np.array([0.0, 0.0, 0.7]), 0.85

    class _Teacher:
        def label(self, T):
            return np.linalg.norm(np.asarray(T)[:, :3, 3] - centre, axis=1) < rmax

    sel, screen, labels = _boundary_anchors(_Teacher(), 1000, np.random.default_rng(3),
                                            oversample=4)
    # screen rows exclude the promoted anchors, so 4*1000 candidates minus the 1000 selected
    assert len(sel) == 1000 and len(screen) == 3000 and len(labels) == 3000

    d_sel = np.abs(np.linalg.norm(sel[:, :3, 3] - centre, axis=1) - rmax)
    d_all = np.abs(np.linalg.norm(screen[:, :3, 3] - centre, axis=1) - rmax)
    assert d_sel.mean() < 0.6 * d_all.mean()


# ------------------------------------------------------------------ splits

def test_geometric_splits_are_disjoint_and_the_region_is_truly_held_out():
    """A leaked test row makes every generalization number meaningless -- and looks like success."""
    rng = np.random.default_rng(4)
    n = 5000
    data = {"features": rng.uniform(-1, 1, size=(n, 9)),
            "exist": rng.integers(0, 2, n).astype(bool)}
    s = geometric_splits(data, seed=0, holdout_octant=(1, 1))

    all_idx = np.concatenate([s["train"], s["val"], s["test_iid"], s["test_region"]])
    assert len(np.unique(all_idx)) == len(all_idx), "splits overlap"
    assert len(np.unique(all_idx)) == n, "splits do not cover the data"

    xy = data["features"][:, :2]
    in_region = (np.sign(xy[:, 0]) == 1) & (np.sign(xy[:, 1]) == 1)
    assert not in_region[s["train"]].any(), "held-out region leaked into training"
    assert in_region[s["test_region"]].all()
    # the orientation slice is a subset of the region slice, never a separate leak
    assert np.isin(s["test_orientation"], s["test_region"]).all()


# ------------------------------------------------------------------ metrics

def test_fpr_at_recall_and_ece_on_known_cases():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    assert fpr_at_recall(np.array([.9, .8, .7, .6, .1, .2, .3, .4]), y, 1.0) == 0.0
    # a useless predictor flags everything at any recall
    assert fpr_at_recall(np.full(8, 0.5), y, 1.0) == 1.0
    # perfectly calibrated constant predictor on a balanced set
    assert expected_calibration_error(np.full(8, 0.5), y.astype(float)) == pytest.approx(0.0)
    assert expected_calibration_error(np.full(8, 1.0), y.astype(float)) == pytest.approx(0.5)


def test_ensemble_lcb_is_never_above_the_mean_and_widens_with_disagreement():
    """The conservative bound is the whole safety story; it must actually be conservative."""
    m = KineReadyEnsemble(m=4)
    x = th.randn(64, 9)
    out = m.predict(x, beta=1.0)
    assert (out["p_lcb"] <= out["p_exist"] + 1e-6).all()
    assert (out["p_lcb"] >= 0).all() and (out["p_lcb"] <= 1).all()
    wider = m.predict(x, beta=3.0)
    assert (wider["p_lcb"] <= out["p_lcb"] + 1e-6).all()


def test_temperature_fitting_moves_off_one_for_miscalibrated_logits():
    m = KineReadyEnsemble(m=2)
    with th.no_grad():          # force wildly over-confident logits
        for mem in m.members:
            mem.head_exist.bias.fill_(8.0)
            mem.head_exist.weight.zero_()
    x = th.randn(512, 9)
    y = (th.rand(512) < 0.5).float()
    t = m.fit_temperature(x, y)
    assert t > 1.5, "over-confident logits should be softened, got T=%.3f" % t


# ------------------------------------------------------------------ reward

def test_score_is_invariant_to_rigid_motion_of_base_and_target_together():
    """THE property the factorization rests on, tested end-to-end through the model.

    `frames.py` tests this for the transform alone. This tests it for the whole inference path --
    if the reward ever leaked a world coordinate into the model input, the score would change
    when the robot and object are moved together, and every learned value would be tied to the
    room the data was collected in.
    """
    r = ReadinessReward(KineReadyEnsemble(m=2), device="cpu")
    base = np.array([[0.7, -1.3, 0.4]])
    target = pose_to_matrix([1.1, -0.9, 0.85], [0.1, 0.2, 0.3, 0.927])
    ref = r.score(base, target)

    for dx, dy, dyaw in ((3.0, -1.0, 0.0), (0.0, 0.0, 1.1), (-2.5, 4.0, -2.0)):
        c, s = np.cos(dyaw), np.sin(dyaw)
        G = np.eye(4)
        G[:2, :2] = [[c, -s], [s, c]]
        G[:2, 3] = [dx, dy]
        moved_base = np.array([[c * base[0, 0] - s * base[0, 1] + dx,
                                s * base[0, 0] + c * base[0, 1] + dy,
                                base[0, 2] + dyaw]])
        assert r.score(moved_base, G @ target) == pytest.approx(ref, abs=1e-5)


def test_score_shapes_and_aggregation_over_multiple_targets():
    r = ReadinessReward(KineReadyEnsemble(m=2), device="cpu")
    base = np.random.default_rng(5).uniform(-1, 1, size=(7, 3))
    targets = np.stack([pose_to_matrix([0.5, 0.0, 0.8], [0, 0, 0, 1]),
                        pose_to_matrix([0.4, 0.2, 0.7], [0, 0, 0, 1])])
    s1 = r.score(base, targets[:1])
    s2, comp = r.score(base, targets, return_components=True)
    assert s1.shape == (7,) and s2.shape == (7,)
    assert comp["p_per_target"].shape == (7, 2)
    # noisy-OR: a second chance can only help
    assert (s2 >= s1 - 1e-9).all()


def test_best_pose_picks_the_argmax():
    r = ReadinessReward(KineReadyEnsemble(m=2), device="cpu")
    base = np.random.default_rng(6).uniform(-1, 1, size=(9, 3))
    target = pose_to_matrix([0.5, 0.0, 0.8], [0, 0, 0, 1])
    s = r.score(base, target)
    i, v = r.best_pose(base, target)
    assert i == int(np.argmax(s)) and v == pytest.approx(float(s.max()))


def test_potential_shaping_is_zero_on_a_constant_potential():
    """Ng et al.'s form: no shaping reward accrues where the potential does not change."""
    assert potential_shaping(0.5, 0.5, gamma=1.0) == pytest.approx(0.0)
    assert potential_shaping(0.9, 0.5, gamma=1.0) == pytest.approx(0.4)


def test_readiness_gates_hard_on_collision_and_cannot_be_outvoted():
    """A geometric mean, so one fatal term sinks the score instead of being averaged away."""
    assert readiness(1.0, 1.0, collision_free=0.0, p_kin=1.0) == pytest.approx(0.0)
    # perfect distance and visibility must not rescue a near-zero kinematic term
    good = readiness(1.0, 1.0, 1.0, 1.0)
    bad = readiness(1.0, 1.0, 1.0, 1e-6)
    assert good == pytest.approx(1.0, abs=1e-6)
    assert bad < 0.02
    # and it stays a bounded score
    vals = readiness(np.array([0.2, 0.9]), np.array([0.5, 1.0]),
                     np.array([1.0, 1.0]), np.array([0.4, 0.8]))
    assert ((vals >= 0) & (vals <= 1)).all()


# ------------------------------------------------------------------ worked example

def test_symmetry_orbit_preserves_the_grip_geometry():
    """Rotating about the OBJECT axis, not the gripper axis.

    Every orbit member must keep the same distance to the can's axis and the same height -- that
    is what makes it the same grasp seen from a different side. Rotating in the gripper frame
    instead would spin the tool in place and produce grasps that miss the can entirely, which is
    an easy mistake to make and impossible to see in an aggregate score.
    """
    from kineready.examples.trash_task import grasp_priors, symmetry_orbit

    centre = np.array([0.55, 0.10, 0.90])
    T = np.eye(4)
    T[:3, 3] = centre + np.array([0.20, 0.0, 0.075])
    orbit = symmetry_orbit(T, centre, k=8)

    assert orbit.shape == (8, 4, 4)
    assert orbit[0] == pytest.approx(T, abs=1e-9), "first member must be the original grasp"
    radii = np.linalg.norm(orbit[:, :2, 3] - centre[:2], axis=1)
    assert radii == pytest.approx(np.full(8, radii[0]), abs=1e-9)
    assert orbit[:, 2, 3] == pytest.approx(np.full(8, T[2, 3]), abs=1e-9)
    for R in orbit[:, :3, :3]:
        assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    # the members are genuinely distinct, or the augmentation does nothing
    assert not np.allclose(orbit[0], orbit[4])

    w = grasp_priors(8)
    assert w[0] == 1.0 and (w[1:] < 1.0).all(), "the observed grasp should outweigh inferred ones"


def test_boundary_anchors_do_not_also_appear_in_the_screen_rows():
    """The robust anchors are drawn FROM the screen pool; returning the pool whole would put every
    anchor in the dataset twice, and the two copies could land on opposite sides of a split."""
    centre, rmax = np.array([0.0, 0.0, 0.7]), 0.85

    class _Teacher:
        def label(self, T):
            return np.linalg.norm(np.asarray(T)[:, :3, 3] - centre, axis=1) < rmax

    sel, screen, labels = _boundary_anchors(_Teacher(), 500, np.random.default_rng(9), oversample=4)
    assert len(screen) == len(labels) == 2000 - 500
    key = lambda A: {tuple(np.round(t[:3, 3], 9)) for t in A}
    assert not (key(sel) & key(screen)), "anchors leaked back into the screen rows"


def test_geometric_splits_refuse_duplicated_rows():
    """The guard must fire, not silently tolerate a leak."""
    from kineready.dataset import _assert_no_duplicate_rows

    rng = np.random.default_rng(10)
    f = rng.uniform(-1, 1, size=(500, 9))
    _assert_no_duplicate_rows(f)                       # clean data passes
    dup = np.concatenate([f, f[:5]], axis=0)
    with pytest.raises(ValueError, match="duplicate"):
        _assert_no_duplicate_rows(dup)


def test_teacher_rejects_a_robot_whose_base_is_not_at_the_origin():
    """The world-frame trap, pinned as a test.

    For a holonomic base, curobo's `base_footprint_x` link sits at the WORLD ORIGIN regardless of
    where the robot is -- the world pose lives entirely in the base joints. So `is_local=True`
    targets are really world targets, and a teacher built while the robot stands elsewhere
    silently labels everything unreachable. Measured: with the robot at (4.79, -1.28), targets
    landed ~6.9 m from where they were meant to.
    """
    from kineready.teacher import IKTeacher

    class _Joint:
        pass

    class _Robot:
        base_joint_names = ["base_footprint_x_joint", "base_footprint_y_joint",
                            "base_footprint_rz_joint"]
        joints = {n: _Joint() for n in
                  ["base_footprint_x_joint", "base_footprint_y_joint", "base_footprint_rz_joint"]}

        def __init__(self, q):
            self._q = th.tensor(q, dtype=th.float32)

        def get_joint_positions(self):
            return self._q

    t = IKTeacher.__new__(IKTeacher)
    t.robot = _Robot([0.0, 0.0, 0.0])
    t._rest_q = t.robot.get_joint_positions()
    t._assert_base_at_origin()                       # at the origin: fine

    t.robot = _Robot([4.79, -1.28, -2.86])
    t._rest_q = t.robot.get_joint_positions()
    with pytest.raises(ValueError, match="base joints at zero"):
        t._assert_base_at_origin()
