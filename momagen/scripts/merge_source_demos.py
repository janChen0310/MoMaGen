"""Merge several single-demo source HDF5 files into one MoMaGen source dataset.

`DataCollectionWrapper` recreates its output file on every run, so collecting N source demos means
N separate files -- a second collection into the same path silently destroys the first. This
stitches them back together and writes the `mask/use` filter key that the base config's
`experiment.source.filter_key` selects on.

Boundary check: MoMaGen reads `MP_end_step` / `subtask_term_step` as GLOBAL values
(`data_generator.parse_MP_end_step_local`), applying one number to whichever source demo it picked.
Demos of differing length therefore desynchronise the motion-planned / replayed split. This refuses
to merge demos whose lengths differ by more than `--length-tol` unless `--allow-ragged` is passed.
"""
import argparse
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length-tol", type=int, default=0,
                    help="max allowed spread in demo length, in steps")
    ap.add_argument("--allow-ragged", action="store_true")
    args = ap.parse_args()

    for p in args.inputs:
        if not os.path.exists(p):
            raise SystemExit("missing input: %s" % p)

    lengths = []
    for p in args.inputs:
        with h5py.File(p, "r") as f:
            for d in sorted(f["data"], key=_demo_key):
                lengths.append((p, d, int(f["data"][d].attrs["num_samples"])))
    spread = max(l for _, _, l in lengths) - min(l for _, _, l in lengths)
    print("demo lengths:")
    for p, d, l in lengths:
        print("  %-6d %s/%s" % (l, os.path.basename(p), d))
    print("spread: %d steps" % spread)
    if spread > args.length_tol and not args.allow_ragged:
        raise SystemExit(
            "demo lengths differ by %d steps (> --length-tol %d). MoMaGen applies ONE global "
            "MP_end_step/subtask_term_step to every source demo, so ragged demos put the "
            "plan/replay boundary in the wrong place for all but one of them. Re-collect with a "
            "fixed NAV pad (JC_NAV_PAD), or pass --allow-ragged if you accept the misalignment."
            % (spread, args.length_tol))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    n_steps = 0
    names = []
    with h5py.File(args.out, "w") as out:
        og = out.create_group("data")
        for src_i, p in enumerate(args.inputs):
            with h5py.File(p, "r") as f:
                if src_i == 0:
                    for k, v in f["data"].attrs.items():
                        og.attrs[k] = v
                else:
                    for k in ("config", "env_args"):
                        if k in f["data"].attrs and f["data"].attrs[k] != og.attrs.get(k):
                            print("WARNING: %s differs in %s; keeping the first file's value.\n"
                                  "         Differing values: %s"
                                  % (k, os.path.basename(p),
                                     _diff_summary(og.attrs.get(k), f["data"].attrs[k])))
                for d in sorted(f["data"], key=_demo_key):
                    new = "demo_%d" % len(names)
                    f.copy(f["data"][d], og, name=new)
                    names.append(new)
                    n_steps += int(f["data"][d].attrs["num_samples"])
                    print("  %s/%s -> data/%s" % (os.path.basename(p), d, new))
        og.attrs["n_episodes"] = len(names)
        og.attrs["n_steps"] = n_steps
        mg = out.create_group("mask")
        mg.create_dataset("use", data=np.array(names, dtype="S"))

    print("\nwrote %s: %d demos, %d steps, mask/use = %s"
          % (args.out, len(names), n_steps, names))


def _diff_summary(a, b, limit=6):
    """Name what actually differs, so a benign mismatch cannot mask a real one.

    A blanket "env_args differs" is easy to wave off; the collector's boot spawn pose legitimately
    differs per source demo (generation re-randomises it at reset), while a differing scene model or
    robot config would invalidate the merge. Printing the changed leaves separates the two.
    """
    import json as _json

    def _flat(x, pre=""):
        if isinstance(x, (bytes, bytearray)):
            x = x.decode()
        if isinstance(x, str):
            try:
                x = _json.loads(x)
            except Exception:
                return {pre or "value": x}
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                out.update(_flat(v, "%s.%s" % (pre, k) if pre else str(k)))
            return out
        if isinstance(x, (list, tuple)):
            out = {}
            for i, v in enumerate(x):
                out.update(_flat(v, "%s[%d]" % (pre, i)))
            return out
        return {pre or "value": x}

    fa, fb = _flat(a), _flat(b)
    keys = [k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)]
    if not keys:
        return "(no leaf differences -- formatting only)"
    keys.sort()
    shown = ", ".join("%s: %r -> %r" % (k, fa.get(k), fb.get(k)) for k in keys[:limit])
    return shown + ("" if len(keys) <= limit else "  (+%d more)" % (len(keys) - limit))


def _demo_key(name):
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return name


if __name__ == "__main__":
    main()
