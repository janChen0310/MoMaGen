"""Compute the kitchen walkway for trash-can randomization, and render top-down views.

Why offline: the walkway is fully determined by the scene's shipped layout PNGs
(floor_trav_0.png = traversable-with-furniture, floor_semseg_0.png = room semantics),
so no simulator is needed to define or visualise it. Coordinate conventions mirror
omnigibson/maps/map_base.py exactly:

    world_to_map([x, y]) -> [row, col] = [y/res + size/2, x/res + size/2]
    map_to_world([row, col]) -> [x, y] = [(col - size/2)*res, (row - size/2)*res]

"Walkway" = pixels that are BOTH inside the kitchen room AND traversable in the
with-furniture trav map, then eroded by the object's footprint so a sampled centre
cannot clip a counter.
"""
import argparse
import json
import os

import cv2
import numpy as np

RES = 0.01  # m/px — matches SegmentationMap.map_default_resolution (layout PNGs are 1cm)
KITCHEN_LINE = 18  # 1-indexed line of "kitchen" in metadata/room_categories.txt => sem_id


def load_layers(scene_dir, meta_dir):
    trav = cv2.imread(os.path.join(scene_dir, "layout", "floor_trav_0.png"), cv2.IMREAD_GRAYSCALE)
    sem = cv2.imread(os.path.join(scene_dir, "layout", "floor_semseg_0.png"), cv2.IMREAD_GRAYSCALE)
    assert trav is not None and sem is not None, "missing layout PNGs"
    if trav.shape != sem.shape:
        sem = cv2.resize(sem, trav.shape[::-1], interpolation=cv2.INTER_NEAREST)
    cats = [l.rstrip() for l in open(os.path.join(meta_dir, "room_categories.txt"))]
    assert cats[KITCHEN_LINE - 1] == "kitchen", cats[KITCHEN_LINE - 1]
    return trav, sem, KITCHEN_LINE


def world_to_map(xy, size):
    return np.array([xy[1] / RES + size / 2.0, xy[0] / RES + size / 2.0])


def map_to_world(rc, size):
    return np.array([(rc[1] - size / 2.0) * RES, (rc[0] - size / 2.0) * RES])


def build_walkway(trav, sem, kitchen_id, clearance_m):
    size = trav.shape[0]
    kitchen = (sem == kitchen_id).astype(np.uint8)
    walk = ((trav == 255) & (kitchen > 0)).astype(np.uint8)
    # Erode by the object's half-footprint + margin so a sampled CENTRE keeps the whole
    # object clear of counters and walls.
    k = max(1, int(round(2 * clearance_m / RES)) | 1)
    eroded = cv2.erode(walk, np.ones((k, k), np.uint8))
    return kitchen, walk, eroded, size


def sample_positions(eroded, size, n, spawn_xy, min_dist, rng, min_sep):
    rows, cols = np.nonzero(eroded)
    if len(rows) == 0:
        return []
    order = rng.permutation(len(rows))
    picked = []
    for idx in order:
        w = map_to_world((rows[idx], cols[idx]), size)
        if np.linalg.norm(w - np.asarray(spawn_xy)) < min_dist:
            continue
        if any(np.linalg.norm(w - p) < min_sep for p in picked):
            continue
        picked.append(w)
        if len(picked) >= n:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--clearance", type=float, default=0.28,
                    help="m from the can centre that must stay walkable (can radius + margin)")
    ap.add_argument("--min-dist", type=float, default=1.8, help="m minimum distance from the robot spawn")
    ap.add_argument("--min-sep", type=float, default=0.25, help="m minimum separation between samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assets = os.path.join(args.repo, "BEHAVIOR-1K", "datasets", "behavior-1k-assets")
    scene_dir = os.path.join(assets, "scenes", "house_single_floor")
    trav, sem, kid = load_layers(scene_dir, os.path.join(assets, "metadata"))
    kitchen, walk, eroded, size = build_walkway(trav, sem, kid, args.clearance)

    tmpl = os.path.join(args.repo, "momagen", "scene_instances", "house_single_floor",
                        "house_single_floor_task_datagen_picking_up_trash_0_0_template.json")
    reg = json.load(open(tmpl))["state"]["registry"]["object_registry"]
    can_xy = np.array(reg["can_of_soda_595"]["root_link"]["pos"][:2])
    trash_xy = np.array(reg["trash_can_596"]["root_link"]["pos"][:2])
    # The collector places the base at can + the proven standoff (see JC_BASE_COUPLED).
    spawn_xy = can_xy + np.array([0.35, 0.10])

    # --- sanity: the known points must fall where we expect ---
    def probe(name, xy):
        r, c = world_to_map(xy, size).astype(int)
        inside = bool(kitchen[r, c] > 0)
        walkable = bool(walk[r, c] > 0)
        print(f"  {name:16s} world={np.round(xy,3).tolist()} map=({r},{c}) "
              f"in_kitchen={inside} walkable={walkable}")
        return inside

    print("kitchen walkway analysis (offline, from shipped layout PNGs)")
    print(f"  map {size}x{size} px @ {RES} m/px | kitchen px={int(kitchen.sum())} "
          f"walkway px={int(walk.sum())} usable(after {args.clearance}m erosion)={int(eroded.sum())}")
    ok_spawn = probe("robot spawn", spawn_xy)
    probe("can (on counter)", can_xy)
    ok_trash = probe("trash can (old)", trash_xy)
    area = int(eroded.sum()) * RES * RES
    print(f"  usable walkway area = {area:.2f} m^2")

    rng = np.random.default_rng(args.seed)
    picked = sample_positions(eroded, size, args.n, spawn_xy, args.min_dist, rng, args.min_sep)
    d = [float(np.linalg.norm(p - spawn_xy)) for p in picked]
    print(f"  sampled {len(picked)} positions | dist from spawn "
          f"min={min(d):.2f} max={max(d):.2f} mean={np.mean(d):.2f} m" if picked else "  NO SAMPLES")

    out = args.out or os.path.join(args.repo, "kitchen_walkway.png")
    render(kitchen, walk, eroded, size, picked, spawn_xy, can_xy, trash_xy, reg, out, args)
    js = os.path.splitext(out)[0] + ".json"
    json.dump({"positions": [p.tolist() for p in picked],
               "spawn_xy": spawn_xy.tolist(),
               "usable_area_m2": area,
               "clearance_m": args.clearance,
               "min_dist_from_spawn_m": args.min_dist}, open(js, "w"), indent=1)
    print(f"  wrote {out}\n  wrote {js}")
    if not (ok_spawn and ok_trash):
        print("  WARNING: a known reference point is NOT in the kitchen mask — check the "
              "room id / transform before trusting these samples.")


def render(kitchen, walk, eroded, size, picked, spawn_xy, can_xy, trash_xy, reg, out, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ys, xs = np.nonzero(kitchen)
    pad = 40
    r0, r1 = max(0, ys.min() - pad), min(size, ys.max() + pad)
    c0, c1 = max(0, xs.min() - pad), min(size, xs.max() + pad)

    def ext(r0, r1, c0, c1):
        a = map_to_world((r0, c0), size)
        b = map_to_world((r1, c1), size)
        return [a[0], b[0], a[1], b[1]]

    extent = ext(r0, r1, c0, c1)
    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    for ax, (layer, title) in zip(axes, [
        (kitchen[r0:r1, c0:c1], "kitchen room (semseg) + traversable overlay"),
        (eroded[r0:r1, c0:c1], f"USABLE walkway (eroded {args.clearance} m) + sampled trash-can positions"),
    ]):
        ax.imshow(layer, origin="lower", extent=extent, cmap="Greys_r", alpha=0.35, aspect="equal")
        ax.imshow(np.ma.masked_where(walk[r0:r1, c0:c1] == 0, walk[r0:r1, c0:c1]),
                  origin="lower", extent=extent, cmap="summer", alpha=0.45, aspect="equal")
        for name, o in reg.items():
            p = o["root_link"]["pos"]
            if extent[0] < p[0] < extent[1] and extent[2] < p[1] < extent[3] and p[2] < 2.0:
                ax.plot(p[0], p[1], "x", color="0.45", ms=4, zorder=2)
        ax.plot(*spawn_xy, "*", color="tab:blue", ms=22, label="TidyBot spawn", zorder=6)
        ax.plot(*can_xy, "o", color="tab:red", ms=9, label="soda can (on counter)", zorder=6)
        ax.plot(*trash_xy, "s", color="tab:orange", ms=12, label="trash can OLD (fixed)", zorder=6)
        ax.set_title(title)
        ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)")
        ax.grid(alpha=0.25)

    if picked:
        P = np.array(picked)
        sc = axes[1].scatter(P[:, 0], P[:, 1], c=np.linalg.norm(P - spawn_xy, axis=1),
                             cmap="viridis", s=90, edgecolor="k", linewidth=0.6, zorder=7,
                             label=f"sampled trash can (n={len(picked)})")
        plt.colorbar(sc, ax=axes[1], label="distance from spawn (m)")
        circ = plt.Circle(spawn_xy, args.min_dist, fill=False, color="tab:blue",
                          ls="--", lw=1.6, zorder=5)
        axes[1].add_patch(circ)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=9)
    fig.suptitle("Trash-can randomization over the kitchen walkway — house_single_floor "
                 f"(green = traversable, dashed circle = {args.min_dist} m exclusion around spawn)")
    fig.tight_layout()
    fig.savefig(out, dpi=115)


if __name__ == "__main__":
    main()
