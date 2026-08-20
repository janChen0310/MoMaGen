"""Stage 1: generate the (target-pose-in-base-frame -> reachable) dataset.

The model is object- and task-agnostic: its input is an arbitrary eef target expressed in the
robot's base frame, and the upstream planner supplies whatever target it wants at query time. So
the training distribution must cover the arm's task space GENERALLY -- not the query distribution
of any one task. Three samplers, each covering something the others miss:

  fk       Forward kinematics of random valid configurations. Concentrated on the reachable
           manifold, and -- crucially -- carrying its true ORIENTATION distribution, which
           uniform task-space sampling gets wrong: at a given position only a small, strongly
           structured set of orientations is achievable, and a model trained only on uniform
           samples sees almost none of them as positives.
  uniform  Uniform over a box around the base with mixed orientations. Supplies the negatives
           and the decision boundary. Without it the model never sees the outside of the
           workspace and cannot answer the question it exists to answer.
  robust   Nominal poses plus J perturbations each, giving a graded margin instead of a bit
           (proposal Sec 6.2). A pose deep in the workspace and one on its boundary are both
           "reachable"; only the margin distinguishes them, and that difference is the whole
           navigation signal.

FK poses are LABELED, not assumed positive. The proposal (Sec 12.1) treats them as free positives
since a configuration reaching them exists by construction -- but that configuration may be in
self-collision, which the solver enforces and forward kinematics does not. An unverifiable label
in the largest slice of the dataset is a bad trade for solve time we now know we have.

Shards are written atomically and skipped if already complete, because box sessions get torn down
mid-run and a 2 M-solve job that cannot resume is a job that never finishes.
"""
import hashlib
import json
import os
import time
import zlib

import numpy as np

from .frames import encode_features

SAMPLERS = ("fk", "uniform", "robust")


def _downward_biased_rotations(n, rng, max_tilt_deg=75.0):
    """Rotations whose tool axis points broadly downward.

    Mobile-manipulation targets cluster here (reaching down to a table, a floor, a bin), so this
    slice buys density where queries actually land. It is deliberately a MINORITY of the data --
    a straight-down assumption already produced a wrong answer once in this project (the real
    trash grasp is tilted ~23 deg), and over-concentrating here would rebuild that mistake as a
    property of the model.
    """
    # Sample the tool axis in a cone about -Z, then a uniform roll about it.
    cos_max = np.cos(np.deg2rad(max_tilt_deg))
    cz = rng.uniform(cos_max, 1.0, n)
    s = np.sqrt(np.clip(1 - cz**2, 0, 1))
    phi = rng.uniform(0, 2 * np.pi, n)
    axis = np.stack([s * np.cos(phi), s * np.sin(phi), -cz], axis=1)   # points down

    # Complete to a frame with a uniformly random roll.
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    degenerate = np.abs(axis @ np.array([1.0, 0.0, 0.0])) > 0.9
    ref[degenerate] = np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, axis)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(axis, x)
    roll = rng.uniform(0, 2 * np.pi, n)
    c, sn = np.cos(roll)[:, None], np.sin(roll)[:, None]
    x_r = c * x + sn * y
    y_r = -sn * x + c * y
    return np.stack([x_r, y_r, axis], axis=-1)                          # columns = frame axes


def _uniform_rotations(n, rng):
    """Uniform on SO(3) via QR of a Gaussian matrix (Haar measure)."""
    A = rng.normal(size=(n, 3, 3))
    out = np.empty_like(A)
    for i in range(n):
        Q, R = np.linalg.qr(A[i])
        Q = Q * np.sign(np.diag(R))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        out[i] = Q
    return out


def sample_uniform(n, rng, box=((-1.1, 1.1), (-1.1, 1.1), (0.0, 1.5)), downward_frac=0.30):
    """Uniform positions in a box around the base, mixed orientations -> (n, 4, 4)."""
    T = np.repeat(np.eye(4)[None], n, axis=0)
    T[:, 0, 3] = rng.uniform(*box[0], n)
    T[:, 1, 3] = rng.uniform(*box[1], n)
    T[:, 2, 3] = rng.uniform(*box[2], n)

    n_down = int(round(n * downward_frac))
    idx = rng.permutation(n)
    T[idx[:n_down], :3, :3] = _downward_biased_rotations(n_down, rng)
    T[idx[n_down:], :3, :3] = _uniform_rotations(n - n_down, rng)
    return T


def _script_provenance():
    """md5 of this file. The box has no version control, so the code identity travels with the
    data or it is lost."""
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _shard_path(out_dir, sampler, index):
    return os.path.join(out_dir, "%s_%05d.hdf5" % (sampler, index))


def generate(teacher, out_dir, plan=None, shard_size=20000, seed=0, n_perturb=8,
             sigma_pos=0.02, sigma_rot=np.deg2rad(5.0), verbose=True):
    """Run the plan, writing resumable shards. Returns a manifest dict.

    `plan` maps sampler name -> number of ANCHOR poses (for 'robust', each anchor costs 1+J
    solves). Defaults to a pilot-sized plan; the caller sizes the real run from the Stage-0
    throughput measurement rather than from a guess.
    """
    plan = plan or {"fk": 20000, "uniform": 50000, "robust": 5000}
    unknown = set(plan) - set(SAMPLERS)
    if unknown:
        raise ValueError("unknown samplers %s (known: %s)" % (sorted(unknown), list(SAMPLERS)))
    os.makedirs(out_dir, exist_ok=True)

    ok, controls = teacher.run_controls(n_fk=64, rng=np.random.default_rng(seed + 991))
    if not ok:
        # The dataset is the one artifact whose errors are invisible downstream: a model trained
        # on wrong labels fits them perfectly and reports excellent validation metrics.
        raise RuntimeError("teacher controls FAILED, refusing to generate data: %s" % controls)

    meta = {
        "script_md5": _script_provenance(),
        "seed": seed,
        "embodiment": str(teacher._emb),
        "robot": type(teacher.robot).__name__,
        "position_threshold_m": 0.01,       # curobo.py motion_kwargs
        "rotation_threshold_rad": 0.12,
        "num_ik_seeds": 512,
        "scene": "empty (labels are scene-free: ik_world_collision_check=False)",
        "n_perturb": n_perturb,
        "sigma_pos": sigma_pos,
        "sigma_rot": sigma_rot,
        "controls": controls,
    }

    manifest = {"meta": meta, "shards": [], "counts": {}}
    for sampler, n_total in plan.items():
        # crc32, not hash(): Python's hash() is randomized per process for strings, so a resumed
        # run would draw a different stream and the recorded seed would not reproduce the dataset.
        rng = np.random.default_rng((seed * 1000003 + zlib.crc32(sampler.encode())) % (2**32))
        n_shards = int(np.ceil(n_total / shard_size))
        written = 0
        for si in range(n_shards):
            path = _shard_path(out_dir, sampler, si)
            n_this = min(shard_size, n_total - si * shard_size)
            if _shard_is_complete(path, n_this):
                if verbose:
                    print("[datagen] skip %s (already complete)" % os.path.basename(path),
                          flush=True)
                manifest["shards"].append(path)
                sp = _shard_path(out_dir, "screen", si)
                if os.path.exists(sp):
                    manifest["shards"].append(sp)
                written += n_this
                continue

            t0 = time.perf_counter()
            T, exist, robust, extra = _generate_shard(teacher, sampler, n_this, rng,
                                                      n_perturb, sigma_pos, sigma_rot)
            _write_shard(path, sampler, T, exist, robust, meta)
            if extra is not None:
                ep = _shard_path(out_dir, "screen", si)
                _write_shard(ep, "screen", extra[0], extra[1], extra[2], meta)
                manifest["shards"].append(ep)
                manifest["counts"]["screen"] = manifest["counts"].get("screen", 0) + len(extra[0])
            written += n_this
            if verbose:
                dt = time.perf_counter() - t0
                print("[datagen] %s shard %d/%d  n=%d  %.1f s  (%.0f/s, %.1f%% positive)"
                      % (sampler, si + 1, n_shards, n_this, dt, n_this / dt,
                         100 * exist.mean()), flush=True)
            manifest["shards"].append(path)
        manifest["counts"][sampler] = written

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def _write_shard(path, sampler, T, exist, robust, meta):
    """Write atomically: a shard file exists only once it is complete.

    A session killed mid-write would otherwise leave a truncated file that the resume logic reads
    as legitimate, and a short shard looks exactly like a distribution shift during training.
    """
    import h5py

    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("T", data=np.asarray(T, dtype=np.float32), compression="gzip")
        f.create_dataset("features", data=encode_features(T).astype(np.float32),
                         compression="gzip")
        f.create_dataset("exist", data=np.asarray(exist).astype(np.bool_))
        f.create_dataset("robust", data=np.asarray(robust, dtype=np.float32))
        f.create_dataset("has_robust", data=np.isfinite(np.asarray(robust, dtype=float)))
        f.attrs["sampler"] = sampler
        f.attrs["n"] = len(exist)
        f.attrs["meta"] = json.dumps(meta)
    os.replace(tmp, path)


def _shard_is_complete(path, expected_n):
    """A shard counts as done only if it opens and holds the expected row count.

    Checking existence alone would silently accept a file truncated by a killed session -- and
    the resulting short shard would look like a legitimate distribution shift in training.
    """
    if not os.path.exists(path):
        return False
    try:
        import h5py

        with h5py.File(path, "r") as f:
            return int(f["exist"].shape[0]) == int(expected_n)
    except Exception:
        return False


def _generate_shard(teacher, sampler, n, rng, n_perturb, sigma_pos, sigma_rot):
    """-> (T (n,4,4), exist (n,) bool, robust (n,) float NaN where unmeasured, extra or None).

    `extra` is an optional second bundle in the same format, written as its own shard.
    """
    if sampler == "fk":
        T = teacher.sample_fk_poses(n, rng=rng)
        return T, teacher.label(T), np.full(n, np.nan), None
    if sampler == "uniform":
        T = sample_uniform(n, rng)
        return T, teacher.label(T), np.full(n, np.nan), None
    if sampler == "robust":
        T, screen_T, screen_exist = _boundary_anchors(teacher, n, rng, oversample=4)
        exist, robust = teacher.label_robust(T, n_perturb=n_perturb, sigma_pos=sigma_pos,
                                             sigma_rot=sigma_rot, rng=rng)
        # The screen's labels are ordinary `exist` rows and cost nothing extra to keep.
        extra = (screen_T, screen_exist, np.full(len(screen_T), np.nan))
        return T, exist, robust, extra
    raise ValueError("unknown sampler %r" % sampler)


def _boundary_anchors(teacher, n, rng, oversample=4, k=16, random_floor=0.2):
    """Pick `n` anchors concentrated where the reachability margin actually varies.

    Measured on the pilot: with physically-grounded perturbation scales (2 cm / 5 deg, taken from
    the pipeline's real object jitter and base coupling noise), 92.5% of uniformly-chosen anchors
    return a robust score of exactly 0 or 1 -- all perturbations agree with the nominal. Those
    rows teach the robust head nothing that `exist` does not already say, while costing 1+J solves
    each. Only 7.5% land in the graded middle, which is the entire signal.

    Widening the perturbation would manufacture a gradient by measuring uncertainty the pipeline
    does not have, so the fix is WHERE anchors are drawn, not how hard they are shaken. A cheap
    one-solve screen labels `oversample`*n candidates; an anchor whose k nearest neighbours in
    feature space disagree about reachability sits near the decision boundary. `random_floor`
    keeps a fraction drawn at random so the slice is not exclusively hard cases.

    The screen's own labels are not thrown away -- they are returned by `screen_labels` for the
    caller to keep as ordinary `exist` rows, which makes the extra solves nearly free.
    """
    from scipy.spatial import cKDTree

    # Candidates come entirely from the uniform box, NOT half from FK. FK poses are reachable by
    # construction, so they are interior positives that can never sit near the boundary -- mixing
    # them in spends half the screen budget on candidates that are ineligible by definition, and
    # measurably dilutes the selection (1.6x enrichment instead of 3.3x on a synthetic shell).
    # The manifold's orientation distribution is already covered by the dedicated 'fk' slice.
    m = int(n * oversample)
    cand = sample_uniform(m, rng)
    labels = teacher.label(cand)

    feats = encode_features(cand)
    # Scale rotation columns down: positions span ~2 m while the 6D rotation entries span [-1, 1],
    # and an unscaled metric would let orientation dominate the neighbourhood definition.
    scaled = feats.copy()
    scaled[:, 3:] *= 0.3
    tree = cKDTree(scaled)
    dist, idx = tree.query(scaled, k=min(k, len(scaled)))
    disagreement = (labels[idx] != labels[:, None]).mean(axis=1)

    # Disagreement is quantized to multiples of 1/k, so the top of the ranking is a large block of
    # ties that argsort would break arbitrarily. Break them toward the tightest neighbourhoods:
    # among two equally-mixed points, the one whose k neighbours are closer pins the boundary more
    # sharply. Measured on a synthetic shell, this tie-break alone lifts boundary enrichment from
    # ~2x to ~3.3x -- the ranking was mostly ties before it.
    tightness = dist.mean(axis=1)
    score = disagreement - 1e-3 * tightness / max(float(tightness.mean()), 1e-9)

    n_rand = int(n * random_floor)
    order = np.argsort(-score)
    chosen = list(order[:n - n_rand])
    remaining = np.setdiff1d(np.arange(m), np.array(chosen, dtype=int), assume_unique=False)
    chosen += list(rng.choice(remaining, size=min(n_rand, len(remaining)), replace=False))

    chosen = np.array(chosen, dtype=int)
    # The screen rows handed back EXCLUDE the promoted anchors. The robust slice is drawn FROM the
    # screen pool, so returning the pool whole would put every anchor in the dataset twice -- and
    # the two copies could land on opposite sides of a train/test split, making a test row a
    # verbatim training row. That inflates the headline IID number and is invisible in the data.
    keep = np.ones(len(cand), dtype=bool)
    keep[chosen] = False
    return cand[chosen], cand[keep], labels[keep]
