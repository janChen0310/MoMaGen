"""Sample trash-can positions over the kitchen walkway.

Shared by `momagen/scripts/analyze_kitchen_walkway.py` (offline visualisation) and
`robomimic/robomimic/envs/env_omnigibson.py` (runtime randomisation) so both use identical
geometry -- a divergence between "what we showed the human" and "what generation places"
would be invisible and corrupting.

"Walkway" = pixels inside the kitchen room AND traversable in the with-furniture trav map,
eroded by the object's clearance so a sampled CENTRE keeps the whole object clear of
counters and walls. Both layers ship with the scene, so no simulator is needed.

Coordinate conventions mirror omnigibson/maps/map_base.py exactly:
    world_to_map([x, y]) -> [row, col] = [y/res + size/2, x/res + size/2]
    map_to_world([row, col]) -> [x, y] = [(col - size/2)*res, (row - size/2)*res]
"""
import os

import numpy as np

RES = 0.01  # m/px — SegmentationMap.map_default_resolution (layout PNGs are 1 cm)
KITCHEN_LINE = 18  # 1-indexed line of "kitchen" in metadata/room_categories.txt == sem id


def map_to_world(rc, size):
    return np.array([(rc[1] - size / 2.0) * RES, (rc[0] - size / 2.0) * RES])


def world_to_map(xy, size):
    return np.array([xy[1] / RES + size / 2.0, xy[0] / RES + size / 2.0])


def build_walkway(scene_dir, meta_dir, clearance_m=0.28, room_line=KITCHEN_LINE):
    """Return (eroded_mask, size, stats) for the walkable part of the room."""
    import cv2

    trav = cv2.imread(os.path.join(scene_dir, "layout", "floor_trav_0.png"), cv2.IMREAD_GRAYSCALE)
    sem = cv2.imread(os.path.join(scene_dir, "layout", "floor_semseg_0.png"), cv2.IMREAD_GRAYSCALE)
    if trav is None or sem is None:
        raise FileNotFoundError("missing layout PNGs under %s/layout" % scene_dir)
    if trav.shape != sem.shape:
        sem = cv2.resize(sem, trav.shape[::-1], interpolation=cv2.INTER_NEAREST)

    cats = [l.rstrip() for l in open(os.path.join(meta_dir, "room_categories.txt"))]
    room_name = cats[room_line - 1]

    room = (sem == room_line).astype(np.uint8)
    walk = ((trav == 255) & (room > 0)).astype(np.uint8)
    k = max(1, int(round(2 * clearance_m / RES)) | 1)
    eroded = cv2.erode(walk, np.ones((k, k), np.uint8))
    stats = {
        "room": room_name,
        "room_px": int(room.sum()),
        "walkway_px": int(walk.sum()),
        "usable_px": int(eroded.sum()),
        "usable_area_m2": float(eroded.sum()) * RES * RES,
        "clearance_m": clearance_m,
    }
    return eroded, trav.shape[0], stats, room, walk


def load_navigable(path):
    """Read nav_sweep.py output -> list of xy the robot could actually plan a path to.

    The 2D traversability map says far more of the kitchen is usable than CuRobo can really
    navigate with a correctly-sized base (it omits stools and other 3D geometry, and says
    nothing about whether a *trajectory* exists). So the sampling pool has to come from the
    planner itself, not from the map.
    """
    import json

    recs = json.load(open(path))
    xy = [r["xy"] for r in recs if r.get("plan_ok")]
    if not xy:
        raise ValueError("%s contains no plan_ok positions -- the sweep found nowhere "
                         "navigable, so the task is infeasible as configured" % path)
    return xy


class WalkwaySampler:
    """Draws xy positions uniformly over the usable walkway, excluding a radius around a
    reference point (the robot spawn) so the object cannot land on top of the robot.

    Use `restrict_to_points(load_navigable(...))` to narrow the pool to neighbourhoods a
    correctly-sized base was measured to reach.
    """

    def __init__(self, scene_dir, meta_dir, clearance_m=0.28, room_line=KITCHEN_LINE):
        self.mask, self.size, self.stats, _, _ = build_walkway(
            scene_dir, meta_dir, clearance_m=clearance_m, room_line=room_line)
        rows, cols = np.nonzero(self.mask)
        if len(rows) == 0:
            raise ValueError("no usable walkway pixels (clearance %.2f m too large?)" % clearance_m)
        self._cells = np.stack([rows, cols], axis=1)
        self._world = np.stack([map_to_world(rc, self.size) for rc in self._cells])

    def restrict_to_points(self, points, radius_m):
        """Keep only cells within `radius_m` of one of `points`.

        The walkway mask says where the CAN fits; it says nothing about whether a
        correctly-sized base can navigate there. `momagen/scripts/nav_sweep.py` measures that
        empirically, and this narrows the pool to the validated neighbourhoods -- keeping
        sampling continuous rather than collapsing onto a handful of discrete spots.
        Raises if nothing survives, because silently sampling the old pool would reintroduce
        exactly the furniture collisions this is meant to remove.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            raise ValueError("navigable set is empty -- nothing to restrict to")
        keep = np.zeros(len(self._world), dtype=bool)
        for p in pts:
            keep |= np.linalg.norm(self._world - p, axis=1) <= radius_m
        if not keep.any():
            raise ValueError("no walkway cell lies within %.2f m of the %d navigable points"
                             % (radius_m, len(pts)))
        self._cells = self._cells[keep]
        self._world = self._world[keep]
        self.stats = dict(self.stats)
        self.stats["restricted_cells"] = int(keep.sum())
        self.stats["restricted_area_m2"] = float(keep.sum()) * RES * RES
        self.stats["navigable_points"] = int(len(pts))
        return self

    def candidates(self, exclude_xy=None, min_dist=0.0):
        if exclude_xy is None or min_dist <= 0:
            return self._world
        d = np.linalg.norm(self._world - np.asarray(exclude_xy, dtype=float), axis=1)
        return self._world[d >= min_dist]

    def sample(self, rng=None, exclude_xy=None, min_dist=0.0):
        """Return one xy. Raises if the constraints leave nothing (loud beats silent)."""
        pool = self.candidates(exclude_xy=exclude_xy, min_dist=min_dist)
        if len(pool) == 0:
            raise ValueError(
                "no walkway cell is >= %.2f m from %s — loosen min_dist or clearance"
                % (min_dist, np.round(np.asarray(exclude_xy), 2).tolist()))
        rng = rng or np.random.default_rng()
        return pool[rng.integers(len(pool))].copy()

    def sample_many(self, n, rng=None, exclude_xy=None, min_dist=0.0, min_sep=0.0):
        rng = rng or np.random.default_rng()
        pool = self.candidates(exclude_xy=exclude_xy, min_dist=min_dist)
        if len(pool) == 0:
            raise ValueError("no walkway cell satisfies the constraints")
        picked = []
        for idx in rng.permutation(len(pool)):
            w = pool[idx]
            if min_sep > 0 and any(np.linalg.norm(w - p) < min_sep for p in picked):
                continue
            picked.append(w.copy())
            if len(picked) >= n:
                break
        return picked


def default_dirs(repo):
    """(scene_dir, meta_dir) for house_single_floor inside a MoMaGen checkout."""
    assets = os.path.join(repo, "BEHAVIOR-1K", "datasets", "behavior-1k-assets")
    return os.path.join(assets, "scenes", "house_single_floor"), os.path.join(assets, "metadata")
