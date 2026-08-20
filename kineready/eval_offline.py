"""Stage 3: offline metrics for the model AND the baselines it has to beat.

A learned model is only worth its complexity if it beats the cheap thing. These baselines are all
fit on the SAME labels from the SAME training split, so the comparison isolates the model rather
than the data:

  distance_band     Is the target within [r_lo, r_hi] of the base? The rule the metric used before
                    any of this, and the one a reviewer will ask about first.
  cylinder          Radial distance AND height both in range -- the standard hand-drawn workspace
                    approximation for a shoulder-mounted arm.
  rm4d_histogram    An RM4D-style reachability map: bin the labels by (radius, height, tool-axis
                    tilt, tool azimuth relative to the radial direction) and look up the empirical
                    positive rate. This is the real competition -- it uses the same data, needs no
                    training, and captures genuine orientation dependence. If the MLP does not
                    beat it, the MLP is not earning its keep.
  knn               k-nearest-neighbour vote in feature space. Not a deployable baseline (it needs
                    the whole dataset at query time) but it bounds how much signal the features
                    carry at all, which tells you whether a gap is the model's fault or the
                    representation's.
"""
import argparse
import json

import numpy as np

from .dataset import geometric_splits, load_shards
from .model import expected_calibration_error, fpr_at_recall
from .train import _auprc, _auroc


def _cyl_coords(features):
    """(N, 9) -> (radius, height, tool tilt, tool azimuth relative to the radial direction).

    The azimuth is taken RELATIVE to the radial direction, not to the world x-axis. An arm on a
    turret is symmetric about its own vertical axis, so what matters is how the tool is oriented
    with respect to the direction it is reaching, not its absolute compass heading. Using absolute
    azimuth would force the histogram to relearn that symmetry from data and blur every bin.
    """
    x, y, z = features[:, 0], features[:, 1], features[:, 2]
    r = np.hypot(x, y)
    r1, r2 = features[:, 3:6], features[:, 6:9]
    tool = np.cross(r1, r2)                        # third frame axis = tool direction
    tilt = np.arccos(np.clip(tool[:, 2], -1, 1))

    radial = np.stack([x, y, np.zeros_like(x)], axis=1)
    nr = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = radial / np.maximum(nr, 1e-9)
    tangent = np.stack([-radial[:, 1], radial[:, 0], np.zeros_like(x)], axis=1)
    az = np.arctan2((tool * tangent).sum(1), (tool * radial).sum(1))
    return r, z, tilt, az


class DistanceBand:
    def fit(self, f, y):
        d = np.linalg.norm(f[:, :3], axis=1)
        self.lo, self.hi = np.quantile(d[y], [0.02, 0.98])
        return self

    def score(self, f):
        d = np.linalg.norm(f[:, :3], axis=1)
        # Graded rather than a hard 0/1 so AUROC can rank within the band; a step function would
        # produce a mass of ties and understate the baseline.
        return np.exp(-np.maximum(0, np.maximum(self.lo - d, d - self.hi)) / 0.15)


class Cylinder:
    def fit(self, f, y):
        r, z, _, _ = _cyl_coords(f)
        self.rlo, self.rhi = np.quantile(r[y], [0.02, 0.98])
        self.zlo, self.zhi = np.quantile(z[y], [0.02, 0.98])
        return self

    def score(self, f):
        r, z, _, _ = _cyl_coords(f)
        pr = np.exp(-np.maximum(0, np.maximum(self.rlo - r, r - self.rhi)) / 0.15)
        pz = np.exp(-np.maximum(0, np.maximum(self.zlo - z, z - self.zhi)) / 0.15)
        return pr * pz


class RM4DHistogram:
    """4D reachability map: P(reachable | radius, height, tilt, relative azimuth)."""

    def __init__(self, bins=(24, 24, 12, 12), prior_strength=5.0):
        self.bins = bins
        # Laplace-style smoothing toward the global rate, so a bin with two samples does not
        # announce 0.0 or 1.0 with total confidence.
        self.prior_strength = prior_strength

    def fit(self, f, y):
        r, z, tilt, az = _cyl_coords(f)
        self.edges = [
            np.linspace(0, np.quantile(r, 0.999), self.bins[0] + 1),
            np.linspace(z.min(), z.max(), self.bins[1] + 1),
            np.linspace(0, np.pi, self.bins[2] + 1),
            np.linspace(-np.pi, np.pi, self.bins[3] + 1),
        ]
        idx = self._index(r, z, tilt, az)
        shape = tuple(self.bins)
        cnt = np.zeros(shape)
        pos = np.zeros(shape)
        np.add.at(cnt, idx, 1.0)
        np.add.at(pos, idx, y.astype(float))
        self.global_rate = float(y.mean())
        self.table = (pos + self.prior_strength * self.global_rate) / (cnt + self.prior_strength)
        self.count = cnt
        return self

    def _index(self, r, z, tilt, az):
        out = []
        for v, e, nb in zip((r, z, tilt, az), self.edges, self.bins):
            i = np.clip(np.digitize(v, e) - 1, 0, nb - 1)
            out.append(i)
        return tuple(out)

    def score(self, f):
        r, z, tilt, az = _cyl_coords(f)
        return self.table[self._index(r, z, tilt, az)]


class KNN:
    def __init__(self, k=16, max_train=200000):
        self.k, self.max_train = k, max_train

    def fit(self, f, y):
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(0)
        if len(f) > self.max_train:
            sub = rng.choice(len(f), self.max_train, replace=False)
            f, y = f[sub], y[sub]
        self.f = f.copy()
        self.f[:, 3:] *= 0.3        # same scaling rationale as the boundary sampler
        self.y = y.astype(float)
        self.tree = cKDTree(self.f)
        return self

    def score(self, f):
        q = f.copy()
        q[:, 3:] *= 0.3
        _, idx = self.tree.query(q, k=self.k)
        return self.y[idx].mean(axis=1)


def metrics(p, y):
    return {"auroc": _auroc(p, y), "auprc": _auprc(p, y),
            "brier": float(np.mean((p - y) ** 2)),
            "ece": expected_calibration_error(p, y),
            "fpr_at_95_recall": fpr_at_recall(p, y, 0.95)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None, help="trained checkpoint; baselines run without it")
    ap.add_argument("--out", default="kineready_eval_offline.json")
    args = ap.parse_args()

    data = load_shards(args.data)
    splits = geometric_splits(data, seed=0)
    f_tr, y_tr = data["features"][splits["train"]], data["exist"][splits["train"]]
    print("train %d rows, %.1f%% positive" % (len(y_tr), 100 * y_tr.mean()), flush=True)

    models = {"distance_band": DistanceBand().fit(f_tr, y_tr),
              "cylinder": Cylinder().fit(f_tr, y_tr),
              "rm4d_histogram": RM4DHistogram().fit(f_tr, y_tr),
              "knn": KNN().fit(f_tr, y_tr)}

    learned = None
    if args.model:
        import torch as th

        from .model import KineReadyEnsemble

        ckpt = th.load(args.model, map_location="cpu", weights_only=False)
        net = KineReadyEnsemble(m=ckpt.get("m", 5))
        net.load_state_dict(ckpt["state_dict"])
        net.eval()

        @th.no_grad()
        def learned(f):
            return net.predict(th.tensor(f, dtype=th.float32))["p_exist"].numpy()

    results = {}
    for slice_name in ("test_iid", "test_region", "test_orientation"):
        idx = splits[slice_name]
        if len(idx) == 0:
            continue
        f, y = data["features"][idx], data["exist"][idx]
        row = {name: metrics(m.score(f), y) for name, m in models.items()}
        if learned is not None:
            row["kineready_mlp"] = metrics(learned(f), y)
        results[slice_name] = row

        print("\n--- %s (n=%d, %.1f%% positive) ---" % (slice_name, len(y), 100 * y.mean()),
              flush=True)
        for name, mm in sorted(row.items(), key=lambda kv: -kv[1]["auroc"]):
            print("  %-16s AUROC %.4f  AUPRC %.4f  Brier %.4f  ECE %.4f  FPR@95%%rec %.4f"
                  % (name, mm["auroc"], mm["auprc"], mm["brier"], mm["ece"],
                     mm["fpr_at_95_recall"]), flush=True)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
