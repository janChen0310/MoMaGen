"""Annotate can_region.json with the nearest standable base pose, and filter on it.

WHY
---
Reachability is not the only constraint on where the can may go. MoMaGen replays the contact-rich
segment of the source demo with a LOCAL eef tracker, and a local tracker needs the target near the
arm's natural configuration -- not merely inside its kinematic envelope. Measured while collecting
the source demo: the scripted DLS tracker reaches a can 0.43 m away cleanly (fingers within 3 mm)
and plateaus 0.16 m short at 0.47 m. The learned IK head says both are reachable, and it is right;
a global solver finds them. The tracker cannot.

How far the base must stand is set by how deep the can sits on the slab: the base cannot come
closer than (counter edge + half the chassis depth), so a can 0.12 m inside the edge forces a
0.46 m standoff while one 0.05 m inside allows 0.43 m.

So each cell is annotated with the distance to the nearest pose the chassis actually fits in, and
cells beyond the tracker's envelope are dropped. Pure geometry -- the walkway mask is built from
PNGs, so this needs no simulator.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="can_region.json")
    ap.add_argument("--out", default=None, help="default: overwrite --region")
    ap.add_argument("--standoff-clearance", type=float, default=0.34,
                    help="half the chassis DEPTH; the base faces the counter, so the "
                         "circumscribed radius is needlessly conservative here")
    ap.add_argument("--max-standoff", type=float, default=0.50,
                    help="the local replay tracker's envelope, measured at 0.43 ok / 0.47 short")
    args = ap.parse_args()

    repo = os.environ.get("MOMAGEN_REPO", os.getcwd())
    from momagen.utils.kitchen_walkway import WalkwaySampler, default_dirs

    sd, md = default_dirs(repo)
    ws = WalkwaySampler(sd, md, clearance_m=args.standoff_clearance)
    stand = np.asarray(ws.candidates(), dtype=float)
    print("standable poses at %.2f m clearance: %d" % (args.standoff_clearance, len(stand)))

    blob = json.load(open(args.region))
    rows = blob["rows"]
    kept = 0
    for r in rows:
        if not r.get("on_surface"):
            r["min_standoff"] = None
            continue
        d = float(np.min(np.linalg.norm(stand - np.array(r["xy"])[None], axis=1)))
        r["min_standoff"] = round(d, 4)
        r["servable"] = bool(d <= args.max_standoff)
        kept += r["servable"]

    on = [r for r in rows if r.get("on_surface")]
    viable = [r for r in on if r.get("viable", 0) >= blob["summary"].get("min_viable", 0.05)]
    both = [r for r in viable if r.get("servable")]
    ms = [r["min_standoff"] for r in on]
    blob["summary"].update(standoff_clearance=args.standoff_clearance,
                           max_standoff=args.max_standoff,
                           servable_cells=int(kept), usable_and_servable=len(both))
    json.dump(blob, open(args.out or args.region, "w"), indent=2)

    print("\non-surface cells        : %d" % len(on))
    print("  viable (reachable)    : %d" % len(viable))
    print("  servable (<= %.2f m)   : %d" % (args.max_standoff, kept))
    print("  BOTH                  : %d   <-- the task's effective can region" % len(both))
    print("\nmin_standoff over on-surface cells: min %.2f  median %.2f  max %.2f"
          % (min(ms), float(np.median(ms)), max(ms)))
    if both:
        b = sorted(both, key=lambda r: -r["viable"])
        print("\nbest cells (high viability AND close standoff) -- good source-demo positions:")
        for r in b[:8]:
            print("   [%.2f, %+.2f]  viable %.2f  min_standoff %.2f m"
                  % (r["xy"][0], r["xy"][1], r["viable"], r["min_standoff"]))
    print("\nwrote %s" % (args.out or args.region))


if __name__ == "__main__":
    main()
