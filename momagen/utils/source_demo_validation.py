"""Validate a PROCESSED source demo before syncing it to the generation server.

Generation reads only world-frame SE(3) geometry (`datagen_info`), so a processed
demo is portable across OmniGibson versions once that group exists. A RAW demo is
one with no `datagen_info` at all — its only content is the version-specific `state`
blob, and OmniGibson 3.7.1 has no version gate, so a newer file fails SILENTLY if
someone tries to replay that state on the server. This module is that missing gate.

Leftover `state`/`state_size` alongside a complete `datagen_info` is legal (generation
never reads it) but bulky — `sync_advisories` flags that as non-fatal hygiene advice.

Version gating, in one place, because a false PASS here is the worst defect this
module can have and every rule below exists to prevent one:

* Without `--expect-versions` there is no version gate at all — just schema checks.
* With it, EVERY component the file declares must be accounted for. Comparing only
  the names the caller passed lets the rest drift unchecked.
* A component declared with no readable `version` but with some other identity (a
  `git_hash`) is FATAL: it names a concrete build that can differ from the server's
  and nothing can compare it.
* A component declared with no identity at all (`{"version": null, "git_hash": null}`,
  which is how OmniGibson marks a component it cannot identify, and which the shipped
  demo carries for `omnigibson-robot-assets`) is an ADVISORY, not fatal. There is no
  fact to compare and no pin string that could ever satisfy it, so failing on it would
  make `--expect-versions` unsatisfiable on every file OmniGibson writes — and an
  unsatisfiable guardrail gets switched off, which is how the hole reopens. It is
  reported on every run instead, so it is never silent.
"""
import json

import h5py


def read_declared_components(data_attrs):
    """Every component the file's `scene_file` declares, mapped to its raw entry.

    Unlike `read_versions` this drops NOTHING. That matters because a component the
    reader silently omits is a component the version gate can neither compare nor
    report as unpinned — it becomes invisible, which is the same silent-PASS class
    the gate exists to close. Scalar entries are normalised to {"version": value}.
    Returns None when the scene_file/versions blob is unreadable at all.
    """
    raw = data_attrs.get("scene_file")
    if raw is None:
        return None
    try:
        scene_file = json.loads(raw if isinstance(raw, str) else raw.decode())
    except Exception:
        return None
    versions = scene_file.get("versions")
    if not isinstance(versions, dict):
        return None
    return {key: (val if isinstance(val, dict) else {"version": val})
            for key, val in versions.items()}


def _usable_version(entry):
    """The comparable version string in a declared entry, or None if there is none."""
    version = entry.get("version")
    return None if version is None else str(version)


def _component_identity(entry):
    """Any field that ties this declared component to a CONCRETE build, or None.

    This is the distinction the version gate turns on, and the two shapes are not
    equivalent:

    * An entry with a version, or with a git_hash but no version, names a specific
      build. It CAN differ between the file and the server, so leaving it unchecked
      is a real silent-drift path and must be fatal under --expect-versions.
    * An entry with neither ({"version": null, "git_hash": null}) carries no fact to
      compare and no string that could ever satisfy a pin. OmniGibson writes exactly
      this for components it cannot identify, and the shipped demo declares
      `omnigibson-robot-assets` that way, so treating it as fatal would make
      --expect-versions permanently unsatisfiable on every file OmniGibson produces.
      An unsatisfiable guardrail gets switched off, which would restore the very hole
      the completeness check closes — so it is surfaced as an advisory instead, never
      dropped in silence.

    Only "version" and "git_hash" are consulted: those are the fields OmniGibson
    actually writes, and inventing others would guess at a format we have not seen.
    """
    for key in ("version", "git_hash"):
        value = entry.get(key)
        if value is not None and str(value) != "":
            return f"{key}={value}"
    return None


def read_versions(data_attrs):
    """Extract {component: version} from a demo's attrs — the COMPARABLE view.

    Versions are NOT a flat `data.attrs["versions"]` key (verified against the real
    files). They live nested inside the `scene_file` JSON blob that OmniGibson writes:
    scene_file["versions"]["omnigibson"]["version"]. Returns None when unreadable.

    Components declared without a usable version are omitted, deliberately: this is
    the "what can I compare?" view. Use `read_declared_components` for the "what does
    the file declare?" view — the gate needs both, and conflating them is what let a
    hash-only component slip through unchecked.
    """
    declared = read_declared_components(data_attrs)
    if declared is None:
        return None
    return {key: _usable_version(entry) for key, entry in declared.items()
            if _usable_version(entry) is not None}


def open_or_error(path):
    """Open @path as HDF5, or return (None, [problem]) instead of raising.

    Factored out so the CLI's main() can open the file once and reuse the handle for
    both the fatal-problem scan and the advisory scan, instead of each of
    validate_processed_source/sync_advisories opening the file independently.
    """
    try:
        return h5py.File(path, "r"), []
    except OSError as exc:
        return None, [f"cannot open as HDF5 ({exc.__class__.__name__}: {exc})"]


def _mask_use_names(f):
    """Return the demo names listed in mask/use, or None if it cannot be read."""
    try:
        raw = f["mask"]["use"][:]
    except Exception:
        return None
    return [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in raw]


def _scan_versions(data_attrs, expected_versions):
    """Version-gate problems for a file whose attrs are @data_attrs.

    A partial pin is itself a problem. Comparing only the components the CALLER
    names means every component the FILE declares but the caller omitted goes
    unchecked, and OmniGibson 3.7.1 downgrades an asset-hash mismatch to a warning
    — so an unchecked behavior-1k-assets drift displaces grasps with no exception
    anywhere. A false PASS is the worst defect this module can have, so anything
    the file declares must be explicitly pinned before the file is certified.

    "Anything the file declares" means the DECLARED set, not the comparable set: a
    component whose version is unreadable used to be dropped by read_versions and so
    escaped both the comparison and the completeness check. See `_component_identity`
    for why an unreadable-but-identified component is fatal while a wholly
    unidentified one is only advised.
    """
    problems = []
    declared = read_declared_components(data_attrs)
    if declared is None:
        # Fail CLOSED: a guardrail that cannot read the versions must not
        # report VALID, because a false PASS is the failure mode that silently
        # corrupts generation.
        return ["cannot determine file versions (no readable scene_file/versions) — "
                "refusing to certify against --expect-versions"]

    got = {k: _usable_version(e) for k, e in declared.items()
           if _usable_version(e) is not None}

    for key, want in expected_versions.items():
        have = got.get(key)
        if have is None:
            problems.append(f"file declares no version for '{key}' (server expects {want})")
        elif str(have) != str(want):
            problems.append(f"version mismatch {key}: file has {have}, server expects {want}")

    # Declared, not comparable, but still tied to a concrete build (e.g. a git_hash
    # with no "version" key). Nothing can verify it and the operator has no way to
    # pin it, so refuse rather than certify around it. Names the caller already
    # passed are skipped: the loop above reported those with a better message.
    opaque = sorted(k for k, e in declared.items()
                    if k not in got and k not in expected_versions
                    and _component_identity(e) is not None)
    if opaque:
        problems.append(
            "unverifiable component: the file declares "
            + ", ".join(f"'{k}' ({_component_identity(declared[k])})" for k in opaque)
            + " with no readable 'version', so --expect-versions cannot check "
            + ("it" if len(opaque) == 1 else "them")
            + " — refusing to certify a file whose declared build cannot be verified")

    unchecked = sorted(k for k in got if k not in expected_versions)
    if unchecked:
        pin = ",".join(f"{k}={got[k]}" for k in sorted(got))
        problems.append(
            "incomplete version pin: the file declares "
            + ", ".join(f"'{k}'" for k in unchecked)
            + " but --expect-versions does not cover "
            + ("it" if len(unchecked) == 1 else "them")
            + " — an unchecked component drifts SILENTLY (asset-hash mismatch is only "
            "a warning on the server). Pin every declared component: " + pin)
    return problems


def scan_problems(f, expected_versions=None):
    """Core fatal-problem scan against an already-open HDF5 file handle."""
    problems = []
    if "data" not in f:
        return ["missing top-level 'data' group"]
    data = f["data"]
    if not isinstance(data, h5py.Group):
        # Everything below iterates `data` as a group of demos; a Dataset here would
        # yield row arrays and blow up. Return a problem instead of raising.
        return [f"top-level 'data' is a {type(data).__name__}, expected a group of demo_* groups"]

    if "env_args" not in data.attrs:
        problems.append("data.attrs['env_args'] missing (generation needs it to build the env)")

    if "mask" not in f or "use" not in f["mask"]:
        problems.append("missing mask/use (generation selects demos through it)")
    else:
        # MG_FileUtils.get_demos_from_dataset builds demo_keys straight out of
        # mask/use (file_utils.py:57) without checking it against data/. An EMPTY
        # mask/use therefore generates over zero demos in silence, and a name with
        # no matching group raises KeyError on the server AFTER the sync.
        names = _mask_use_names(f)
        if names is None:
            problems.append("mask/use is unreadable (expected a 1-D list of demo names)")
        elif not names:
            problems.append(
                "mask/use is empty — generation would select zero demos and produce nothing")
        else:
            dangling = [n for n in names if n not in data]
            if dangling:
                problems.append(
                    f"mask/use names {dangling} with no matching group under data/ "
                    "(this raises KeyError on the generation server, after the sync)")

    if expected_versions:
        problems.extend(_scan_versions(data.attrs, expected_versions))

    demos = [k for k in data if k.startswith("demo")]
    if not demos:
        problems.append("no demo_* groups found")

    for demo in demos:
        g = data[demo]
        if "action" not in g:
            problems.append(f"{demo}: missing 'action' (its length defines the trajectory)")
            continue
        action_shape = getattr(g["action"], "shape", None)
        if not action_shape:
            problems.append(
                f"{demo}: 'action' is not a 2-D (T, action_dim) dataset "
                f"(shape {action_shape!r}) — its length defines the trajectory")
            continue
        T = action_shape[0]
        if T == 0:
            # Every shape and length check below agrees with T == 0 (a truncated
            # prepare_src_dataset.py run leaves action (0, 11), eef_pose (0, 8, 4), ...),
            # so without this the whole file validates clean and syncs a demo that
            # generation cannot step even once.
            problems.append(
                f"{demo}: zero-length trajectory (action shape {tuple(action_shape)}) — "
                "not syncable; re-run prepare_src_dataset.py, it was truncated")

        if "datagen_info" not in g:
            problems.append(f"{demo}: missing 'datagen_info' — run prepare_src_dataset.py")
            continue
        dg = g["datagen_info"]

        # MG_FileUtils.get_env_interface_info_from_dataset reads these off the
        # datagen_info group (file_utils.py:92); generation cannot start without them.
        for attr in ("env_interface_name", "env_interface_type"):
            if attr not in dg.attrs:
                problems.append(f"{demo}: datagen_info.attrs['{attr}'] missing")

        if "eef_pose" not in dg:
            problems.append(f"{demo}: datagen_info/eef_pose missing")
        else:
            # getattr, not `.shape`: a Group here would raise AttributeError, and this
            # module's contract is to return problems rather than raise.
            shape = getattr(dg["eef_pose"], "shape", None)
            if shape is None or len(shape) != 3 or shape[1:] != (8, 4):
                problems.append(
                    f"{demo}: eef_pose shape {shape}, expected (T, 8, 4) "
                    "(bimanual layout; TidyBot duplicates its single arm)")
            elif shape[0] != T:
                problems.append(f"{demo}: eef_pose length {shape[0]} != action length {T}")

        if "gripper_action" not in dg:
            problems.append(f"{demo}: datagen_info/gripper_action missing")
        else:
            shape = getattr(dg["gripper_action"], "shape", None)
            if shape is None or len(shape) != 2 or shape[1] != 2:
                problems.append(f"{demo}: gripper_action shape {shape}, expected (T, 2)")
            elif shape[0] != T:
                problems.append(f"{demo}: gripper_action length {shape[0]} != action length {T}")

        if "object_poses" not in dg:
            problems.append(
                f"{demo}: datagen_info/object_poses missing — "
                "generation re-anchors every subtask against these")
        elif not isinstance(dg["object_poses"], h5py.Group):
            # Iterating a Dataset here yields row arrays, and indexing the Dataset
            # with one raises TypeError ("Only 1D arrays allowed for fancy indexing").
            # This module promises to RETURN problems, never to raise.
            problems.append(
                f"{demo}: datagen_info/object_poses is a "
                f"{type(dg['object_poses']).__name__}, expected a group of per-object "
                "(T, 4, 4) datasets")
        elif len(dg["object_poses"]) == 0:
            problems.append(
                f"{demo}: datagen_info/object_poses is empty — "
                "generation re-anchors every subtask against these")
        else:
            for obj in dg["object_poses"]:
                shape = dg["object_poses"][obj].shape
                if len(shape) != 3 or shape[1:] != (4, 4):
                    problems.append(f"{demo}: object_poses/{obj} shape {shape}, expected (T, 4, 4)")
                elif shape[0] != T:
                    problems.append(
                        f"{demo}: object_poses/{obj} length {shape[0]} != action length {T}")
    return problems


def _unidentified_component_advisory(f):
    """Name components the file declares but does not identify at all.

    These are NOT fatal (see `_component_identity` for why), but they must never be
    silent: the operator is entitled to know that part of what the file declares sits
    outside what --expect-versions can ever verify. Reported whether or not a pin was
    supplied, because it is a property of the file rather than of the invocation.
    """
    try:
        declared = read_declared_components(f["data"].attrs)
    except Exception:
        return []
    if not declared:
        return []
    blind = sorted(k for k, e in declared.items() if _component_identity(e) is None)
    if not blind:
        return []
    them = "it" if len(blind) == 1 else "them"
    return ["declares " + ", ".join(f"'{k}'" for k in blind)
            + " with neither a version nor a git_hash — nothing in the file identifies "
            + them + ", so --expect-versions can never verify " + them
            + "; if " + ("it drifts" if len(blind) == 1 else "they drift")
            + " on the server this gate will not see it"]


def scan_advisories(f):
    """Core non-fatal advisory scan against an already-open HDF5 file handle."""
    advisories = []
    if "data" in f:
        advisories.extend(_unidentified_component_advisory(f))
    for demo in [k for k in f.get("data", {}) if k.startswith("demo")]:
        g = f["data"][demo]
        present = [k for k in ("state", "state_size") if k in g]
        if not present:
            continue
        # Only quote a size when 'state' itself is there to measure. Saying
        # "~0.0 MB" for a state_size-only demo reads as "nothing to strip", which
        # is the opposite of the truth.
        state = g["state"] if "state" in g else None
        if isinstance(state, h5py.Dataset):
            size = f" (~{state.size * state.dtype.itemsize / 1e6:.1f} MB)"
        else:
            size = " (size not measurable here)"
        advisories.append(
            f"{demo}: carries leftover {'/'.join(present)}{size} that generation "
            "never reads — strip it with make_minimal_source.py before syncing")
    return advisories


def validate_processed_source(path, expected_versions=None):
    """Return a list of problems; empty list means the file is safe to sync."""
    f, open_problems = open_or_error(path)
    if open_problems:
        return open_problems
    with f:
        return scan_problems(f, expected_versions)


def sync_advisories(path):
    """Non-fatal hygiene advice. Leftover `state` is legal in a processed demo (generation
    never reads it) but it is ~95% of the file size, so advise stripping before syncing."""
    f, open_problems = open_or_error(path)
    if open_problems:
        return []
    with f:
        return scan_advisories(f)
