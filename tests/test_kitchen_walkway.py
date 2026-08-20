import numpy as np
import pytest

from momagen.utils.kitchen_walkway import (
    RES,
    WalkwaySampler,
    default_dirs,
    map_to_world,
    world_to_map,
)

REPO = "/home/janchen/Documents/MoMaGen"


def test_map_world_roundtrip():
    size = 8630
    for xy in ([4.776, -0.151], [0.0, 0.0], [-3.2, 5.5]):
        rc = world_to_map(xy, size)
        back = map_to_world(rc, size)
        assert np.allclose(back, xy, atol=RES), (xy, back)


@pytest.fixture(scope="module")
def sampler():
    scene_dir, meta_dir = default_dirs(REPO)
    import os
    if not os.path.exists(os.path.join(scene_dir, "layout", "floor_trav_0.png")):
        pytest.skip("scene assets not present")
    return WalkwaySampler(scene_dir, meta_dir, clearance_m=0.28)


def test_room_is_the_kitchen(sampler):
    assert sampler.stats["room"] == "kitchen"


def test_usable_area_is_plausible(sampler):
    # A galley kitchen walkway ring: square metres, not hundreds and not zero.
    assert 1.0 < sampler.stats["usable_area_m2"] < 40.0, sampler.stats


def test_every_sample_is_on_the_usable_mask(sampler):
    rng = np.random.default_rng(1)
    for _ in range(200):
        xy = sampler.sample(rng)
        r, c = world_to_map(xy, sampler.size).astype(int)
        assert sampler.mask[r, c] > 0, xy


def test_min_dist_from_spawn_is_respected(sampler):
    spawn = np.array([4.79, -1.28])
    rng = np.random.default_rng(2)
    for _ in range(200):
        xy = sampler.sample(rng, exclude_xy=spawn, min_dist=2.0)
        assert np.linalg.norm(xy - spawn) >= 2.0


def test_samples_move_far_beyond_the_old_fixed_position(sampler):
    # The whole point: the old can sat 1.13 m from spawn and pi0.5 memorised it.
    spawn = np.array([4.79, -1.28])
    rng = np.random.default_rng(3)
    d = [np.linalg.norm(sampler.sample(rng, exclude_xy=spawn, min_dist=2.0) - spawn)
         for _ in range(300)]
    assert min(d) >= 2.0 and max(d) > 3.5, (min(d), max(d))


def test_samples_span_the_whole_kitchen_not_one_corner(sampler):
    rng = np.random.default_rng(4)
    P = np.array([sampler.sample(rng, exclude_xy=[4.79, -1.28], min_dist=2.0) for _ in range(400)])
    # np.ptp(): ndarray.ptp() was removed in numpy 2.x
    assert np.ptp(P[:, 0]) > 3.0, np.ptp(P[:, 0])   # spread in x
    assert np.ptp(P[:, 1]) > 1.5, np.ptp(P[:, 1])   # spread in y


def test_impossible_constraint_raises_loudly(sampler):
    with pytest.raises(ValueError, match="min_dist|constraints"):
        sampler.sample(exclude_xy=[4.79, -1.28], min_dist=500.0)


def test_min_sep_enforced_in_sample_many(sampler):
    pts = sampler.sample_many(15, rng=np.random.default_rng(5), min_sep=0.5)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert np.linalg.norm(pts[i] - pts[j]) >= 0.5


def _fresh_sampler():
    """A private instance: restrict_to_points MUTATES, so it must never touch the shared
    module-scoped `sampler` fixture that the other tests rely on."""
    import os
    scene_dir, meta_dir = default_dirs(REPO)
    if not os.path.exists(os.path.join(scene_dir, "layout", "floor_trav_0.png")):
        pytest.skip("scene assets not present")
    return WalkwaySampler(scene_dir, meta_dir, clearance_m=0.28)


def test_restrict_to_points_narrows_pool_and_confines_samples():
    s = _fresh_sampler()
    before = len(s._world)
    pts = [s._world[0].copy(), s._world[len(s._world) // 2].copy()]
    s.restrict_to_points(pts, radius_m=0.25)
    assert len(s._world) < before, "restriction did not shrink the pool"
    assert s.stats["navigable_points"] == 2
    rng = np.random.default_rng(0)
    for _ in range(50):
        xy = s.sample(rng=rng)
        assert min(np.linalg.norm(xy - p) for p in pts) <= 0.25 + 1e-9


def test_restrict_to_points_raises_when_nothing_survives():
    """Loud failure beats silently falling back to the unrestricted (colliding) pool."""
    s = _fresh_sampler()
    with pytest.raises(ValueError):
        s.restrict_to_points([[999.0, 999.0]], radius_m=0.1)


def test_restrict_to_points_raises_on_empty_set():
    s = _fresh_sampler()
    with pytest.raises(ValueError):
        s.restrict_to_points([], radius_m=0.5)
