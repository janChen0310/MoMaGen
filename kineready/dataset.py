"""Shard reader -> torch Dataset. Imports no simulator, so training runs anywhere.

Splits are deliberately NOT random row shuffles. A random split over a dense sample of a smooth
function is nearly free to fit -- every test point sits between two training points -- so IID
accuracy would look excellent while saying nothing about the question that matters: does the model
generalize to targets it has not seen? The splits here are geometric.
"""
import glob
import json
import os

import numpy as np


def load_shards(paths_or_dir, samplers=None):
    """-> dict of concatenated arrays: features, T, exist, robust, has_robust, sampler."""
    import h5py

    if isinstance(paths_or_dir, str):
        paths = sorted(glob.glob(os.path.join(paths_or_dir, "*.hdf5")))
    else:
        paths = list(paths_or_dir)
    if not paths:
        raise FileNotFoundError("no shards found in %r" % (paths_or_dir,))

    acc = {k: [] for k in ("features", "T", "exist", "robust", "has_robust")}
    tags, meta = [], None
    for p in paths:
        with h5py.File(p, "r") as f:
            s = str(f.attrs["sampler"])
            if samplers is not None and s not in samplers:
                continue
            for k in acc:
                acc[k].append(np.asarray(f[k]))
            tags.append(np.full(len(f["exist"]), s, dtype=object))
            if meta is None and "meta" in f.attrs:
                meta = json.loads(f.attrs["meta"])
    if not tags:
        raise ValueError("no shards matched samplers=%r" % (samplers,))

    out = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    out["sampler"] = np.concatenate(tags, axis=0)
    out["meta"] = meta
    return out


def geometric_splits(data, seed=0, holdout_octant=(1, 1), val_frac=0.1):
    """-> dict of index arrays: train, val, test_iid, test_region, test_orientation.

    - test_iid: a random holdout. This is the DEPLOYMENT-RELEVANT number: the shipped model
      trains on the whole workspace, so at query time every target is in-distribution. Read this
      one when asking "will it work".
    - test_region: every target whose base-frame (x, y) falls in one held-out quadrant. Nothing
      from that quadrant is trained on. This is a STRUCTURE PROBE, not a deployment scenario --
      it will look worse by construction, and that is the point: a model that memorized will
      collapse here while one that learned the arm's smooth geometry degrades gently. Do not read
      it as an expected failure rate.
    - test_orientation: the region slice restricted to near-uniform-SO(3) orientations, so the
      30% downward-biased sampling cannot hide an orientation coverage gap behind an easy average.
    """
    rng = np.random.default_rng(seed)
    n = len(data["exist"])
    _assert_no_duplicate_rows(data["features"])
    xy = data["features"][:, :2]
    sx, sy = holdout_octant
    in_region = (np.sign(xy[:, 0]) == sx) & (np.sign(xy[:, 1]) == sy)

    # "Near-uniform orientation" = tool axis not concentrated downward. Column 3 of the 6D
    # encoding's implied frame is the cross product; use the encoded columns directly.
    r1, r2 = data["features"][:, 3:6], data["features"][:, 6:9]
    tool_z = np.cross(r1, r2)[:, 2]
    upright_ish = tool_z > -0.2

    rest = np.where(~in_region)[0]
    rng.shuffle(rest)
    n_val = int(len(rest) * val_frac)
    n_iid = int(len(rest) * val_frac)
    return {
        "val": rest[:n_val],
        "test_iid": rest[n_val:n_val + n_iid],
        "train": rest[n_val + n_iid:],
        "test_region": np.where(in_region)[0],
        "test_orientation": np.where(in_region & upright_ish)[0],
    }


def _assert_no_duplicate_rows(features, sample=200000, seed=0):
    """Fail loudly if the same target pose appears twice.

    A duplicated row can be assigned to train and to test independently, which turns a test row
    into a verbatim training row. The resulting metric looks like excellent generalization and is
    measuring memorization -- the single most expensive mistake this pipeline could make silently,
    since nothing downstream re-checks it.
    """
    f = np.asarray(features)
    if len(f) > sample:
        f = f[np.random.default_rng(seed).choice(len(f), sample, replace=False)]
    uniq = np.unique(np.round(f, 6), axis=0)
    if len(uniq) != len(f):
        raise ValueError(
            "dataset contains %d duplicate target poses out of %d sampled rows; splits would leak "
            "training rows into the test sets" % (len(f) - len(uniq), len(f)))


class KineReadyDataset:
    """Minimal torch Dataset over a preloaded split (the arrays fit in RAM comfortably)."""

    def __init__(self, data, idx):
        import torch as th

        self.x = th.tensor(data["features"][idx], dtype=th.float32)
        self.y = th.tensor(data["exist"][idx].astype(np.float32))
        robust = data["robust"][idx]
        self.r = th.tensor(np.nan_to_num(robust, nan=0.0), dtype=th.float32)
        self.r_mask = th.tensor(np.isfinite(robust).astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.r[i], self.r_mask[i]
