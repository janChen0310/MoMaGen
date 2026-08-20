"""Plot the total base-pose reward distribution from the dumped npz. matplotlib, no simulator.

Three panels, each answering a different question:

  1. the distribution itself, log-binned, with the exactly-zero mass DETACHED -- it is a hard-gate
     outcome, not a small reward, and drawing it as just another bin would imply a continuum that
     does not exist.
  2. how many poses each gate passes. Drawn as independent counts, NOT a funnel: visibility passes
     362 poses while collision passes 228, so the gates are not nested and a funnel would imply a
     nesting that is not there.
  3. where the usable poses sit, relative to the can.

Counts are direct-labelled on every non-empty bar. The y axis is log because a 772-pose mode beside
31-pose bars cannot share a linear scale, and direct labels remove any ambiguity a log axis
introduces about relative magnitude.

Colors are the validated palette (see the dataviz skill's reference instance): one blue hue for the
graded signal, gray for anything excluded from it. Both themes are rendered, because a PNG cannot
adapt to the viewer's editor.
"""
import argparse
import os

import numpy as np

# --- the validated palette, one dict per theme -------------------------------------------------
THEMES = {
    "light": dict(surface="#fcfcfb", plane="#f9f9f7", ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  use="#2a78d6", out="#898781", gate="#86b6ef",
                  ramp=["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf"]),
    "dark": dict(surface="#1a1a19", plane="#0d0d0d", ink="#ffffff", ink2="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", axis="#383835",
                 use="#3987e5", out="#898781", gate="#184f95",
                 ramp=["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf"]),
}
MONO = ["DejaVu Sans Mono", "monospace"]


def build(npz, theme, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    C = THEMES[theme]
    d = np.load(npz, allow_pickle=True)
    R4, free, Sv, pk, dist = d["R4"], d["free"], d["S_v"], d["p_kin"], d["distance"]
    poses, centre = d["poses"], d["can_centre"]

    n = len(R4)
    n_zero = int((R4 == 0).sum())
    n_floor = int(((R4 > 0) & (R4 < 1e-2)).sum())
    USABLE = 0.1                 # the data-driven break, and a bin edge
    n_use = int((R4 > USABLE).sum())
    n_half = int((R4 > 0.5).sum())
    g = R4 > USABLE

    fig = plt.figure(figsize=(13.2, 9.4), dpi=190, facecolor=C["plane"])
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.30, 1.05, 0.95],
                  hspace=0.42, wspace=0.20,
                  left=0.062, right=0.978, top=0.955, bottom=0.075)

    # ---------------- header ------------------------------------------------------------------
    fig.text(0.062, 0.975, "Base-pose reward distribution",
             fontsize=21, color=C["ink"], va="top", weight="semibold")
    fig.text(0.062, 0.942,
             "1000 poses sampled around the soda can, scored on all four terms "
             "(distance x visibility x IK x collision gate)",
             fontsize=10.5, color=C["ink2"], va="top")
    kpis = [(f"{n_zero}", "score exactly 0\nbase in collision"),
            (f"{n_floor}", "in 1e-6 - 1e-2\na clip artifact"),
            (f"{n_use}", "usable cluster\nreward > 0.1"),
            (f"{100.0 * n_use / n:.1f}%", "of samples are\nactually usable")]
    for i, (v, k) in enumerate(kpis):
        x = 0.062 + i * 0.235
        fig.text(x, 0.905, v, fontsize=19, color=C["use"], family=MONO, va="top")
        fig.text(x, 0.868, k, fontsize=9, color=C["ink2"], va="top", linespacing=1.5)

    def style(ax, grid_axis="y"):
        ax.set_facecolor(C["surface"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C["axis"])
            ax.spines[s].set_linewidth(1.0)
        ax.tick_params(colors=C["muted"], labelsize=9, length=3, width=1.0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_family(MONO)
        if grid_axis:
            ax.grid(True, axis=grid_axis, color=C["grid"], linewidth=1.0, zorder=0)
            ax.set_axisbelow(True)

    # ---------------- panel 1: the distribution -----------------------------------------------
    ax = fig.add_subplot(gs[1, :]); style(ax)
    edges = np.array([10 ** (-6 + 0.5 * i) for i in range(13)])
    nz = R4[R4 > 0]
    counts = [int(((nz >= edges[i]) & (nz < edges[i + 1])).sum()) for i in range(12)]
    counts[-1] += int((nz >= edges[-1]).sum())

    GAP = 1.15                                   # the physical gap that detaches the zero mass
    xs = np.arange(12) + 1 + GAP
    ax.bar(0, n_zero, width=0.9, color=C["out"], zorder=3)
    for i, (x, c) in enumerate(zip(xs, counts)):
        if c == 0:
            continue
        usable = edges[i] >= USABLE
        ax.bar(x, c, width=0.9, color=C["use"] if usable else C["out"], zorder=3)
        ax.text(x, c * 1.16, str(c), ha="center", va="bottom", fontsize=9.5,
                family=MONO, color=C["ink"], zorder=4)
    ax.text(0, n_zero * 1.16, str(n_zero), ha="center", va="bottom", fontsize=10.5,
            family=MONO, color=C["ink"], weight="semibold", zorder=4)
    ax.axvline(0.5 + GAP / 2, color=C["axis"], linewidth=1.0, linestyle=(0, (2, 3)), zorder=2)

    ax.set_yscale("log")
    ax.set_ylim(0.7, n_zero * 4.2)
    ax.set_xlim(-0.85, xs[-1] + 0.85)
    # decade labels belong at bin EDGES, not centres: bin i covers [edges[i], edges[i+1]), so the
    # decade 10^(-6+k) sits at the left edge of bin 2k, and 10^0 at the right edge of the last bin.
    dec_pos = [xs[2 * k] - 0.5 for k in range(6)] + [xs[-1] + 0.5]
    ax.set_xticks([0] + dec_pos)
    ax.set_xticklabels(["0"] + [r"$10^{%d}$" % e for e in range(-6, 1)])
    ax.set_ylabel("poses  (log scale)", fontsize=10, color=C["ink2"])
    ax.set_xlabel("total reward", fontsize=10, color=C["ink2"], labelpad=6)
    ax.text(0, 0.55, "gated", ha="center", va="top", fontsize=8.5, color=C["muted"],
            family=MONO, transform=ax.get_xaxis_transform())

    # bracket over the clip-floor band
    fb = [i for i in range(12) if edges[i] < 1e-2]
    x0, x1 = xs[fb[0]] - 0.45, xs[fb[-1]] + 0.45
    yb = n_zero * 1.9
    ax.plot([x0, x0, x1, x1], [yb / 1.35, yb, yb, yb / 1.35], color=C["muted"],
            linewidth=1.0, zorder=4)
    ax.text((x0 + x1) / 2, yb * 1.12, f"{n_floor} poses below the $10^{{-6}}$ clip floor "
            "— an artifact, not \"slightly good\"", ha="center", va="bottom",
            fontsize=9.5, color=C["muted"], zorder=4)
    ax.annotate("usable cluster", xy=(xs[-1], counts[-1] * 1.05),
                xytext=(xs[-1] - 2.6, counts[-1] * 9), fontsize=9.5, color=C["use"],
                ha="center", zorder=4,
                arrowprops=dict(arrowstyle="-", color=C["use"], linewidth=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_title("The distribution is bimodal, with a three-decade dead band",
                 fontsize=13, color=C["ink"], loc="left", pad=10, weight="semibold")

    # ---------------- panel 2: gate pass counts -----------------------------------------------
    ax2 = fig.add_subplot(gs[2, 0]); style(ax2, grid_axis="x")
    gates = [("Collision-free", int(free.sum()), False),
             ("Can visible", int((Sv > 0).sum()), False),
             (r"$p_{kin} \geq 0.5$", int((pk >= 0.5).sum()), False),
             ("All three", int(((free > 0) & (Sv > 0) & (pk >= 0.5)).sum()), True)]
    ypos = np.arange(len(gates))[::-1]
    for y, (lab, cnt, emph) in zip(ypos, gates):
        ax2.barh(y, cnt, height=0.62, color=C["use"] if emph else C["gate"], zorder=3)
        ax2.text(cnt + n * 0.012, y, str(cnt), va="center", fontsize=10, family=MONO,
                 color=C["ink"], zorder=4)
    ax2.set_yticks(ypos); ax2.set_yticklabels([g[0] for g in gates], fontsize=10, color=C["ink2"])
    for lbl in ax2.get_yticklabels():
        lbl.set_family("DejaVu Sans")
    ax2.set_xlim(0, n * 0.46)
    ax2.set_xlabel(f"poses passing, of {n}", fontsize=10, color=C["ink2"])
    ax2.set_title("Which gate rejects what", fontsize=13, color=C["ink"], loc="left",
                  pad=30, weight="semibold")
    ax2.text(0.0, 1.022, "independent, not nested — visibility passes more poses than collision",
             transform=ax2.transAxes, fontsize=9, color=C["muted"], va="bottom")

    # ---------------- panel 3: where the usable poses are -------------------------------------
    ax3 = fig.add_subplot(gs[2, 1]); style(ax3, grid_axis=None)
    rel = poses[:, :2] - centre[:2][None]
    bad = R4 <= USABLE
    ax3.scatter(rel[bad, 0], rel[bad, 1], s=11, c=C["out"], alpha=0.45,
                linewidths=0, zorder=2, label=f"unusable ({int(bad.sum())})")
    ok = ~bad
    if ok.any():
        t = np.clip((np.log10(np.maximum(R4[ok], 1e-2)) + 2) / 2.0, 0, 1)
        idx = np.clip((t * len(C["ramp"])).astype(int), 0, len(C["ramp"]) - 1)
        ax3.scatter(rel[ok, 0], rel[ok, 1], s=46,
                    c=[C["ramp"][i] for i in idx], edgecolors=C["surface"],
                    linewidths=1.4, zorder=4, label=f"usable ({int(ok.sum())})")
    for m in (0.5, 1.0, 1.5):
        ax3.add_patch(plt.Circle((0, 0), m, fill=False, color=C["grid"], linewidth=1.0, zorder=1))
        ax3.text(-m * 0.7071 + 0.02, m * 0.7071 + 0.02, f"{m:g} m", fontsize=8,
                 color=C["muted"], family=MONO, zorder=6, ha="center", va="bottom",
                 bbox=dict(facecolor=C["surface"], edgecolor="none", pad=1.2, alpha=0.92))
    ax3.plot([-0.07, 0.07], [0, 0], color=C["ink"], linewidth=1.8, zorder=5)
    ax3.plot([0, 0], [-0.07, 0.07], color=C["ink"], linewidth=1.8, zorder=5)
    ax3.set_aspect("equal")
    lim = 1.95
    ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim)
    ax3.set_xlabel("metres east of the can", fontsize=10, color=C["ink2"])
    ax3.set_ylabel("metres north of the can", fontsize=10, color=C["ink2"])
    ax3.set_title("Where the usable poses are", fontsize=13, color=C["ink"], loc="left",
                  pad=30, weight="semibold")
    ax3.text(0.0, 1.022,
             f"all {n_use} sit {dist[g].min():.2f}–{dist[g].max():.2f} m out, in one arc",
             transform=ax3.transAxes, fontsize=9, color=C["muted"], va="bottom")
    leg = ax3.legend(loc="lower left", frameon=True, fontsize=9, borderpad=0.6,
                     handletextpad=0.5, labelspacing=0.35)
    leg.get_frame().set_facecolor(C["surface"])
    leg.get_frame().set_edgecolor(C["grid"])
    for txt in leg.get_texts():
        txt.set_color(C["ink2"])

    fig.text(0.062, 0.017,
             f"median non-zero reward {np.median(nz):.2e}  ·  max {R4.max():.6f}  ·  "
             f"{n_half} poses score above 0.5  ·  "
             f"of {int(free.sum())} collision-free poses, {int((Sv[free > 0] == 0).sum())} cannot see "
             f"the can and {int((pk[free > 0] < 0.5).sum())} are out of reach; "
             f"{int(((Sv[free > 0] > 0) & (pk[free > 0] >= 0.5)).sum())} clear both",
             fontsize=8.5, color=C["muted"], family=MONO)

    fig.savefig(out_path, facecolor=C["plane"], bbox_inches="tight", pad_inches=0.32)
    print("wrote %s" % out_path, flush=True)
    return dict(zero=n_zero, floor=n_floor, usable=n_use, counts=counts)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--npz", default=os.path.join(here, "kineready/figures/reward_distribution.npz"))
    ap.add_argument("--out-dir", default=os.path.join(here, "kineready/figures"))
    args = ap.parse_args()
    for theme in ("light", "dark"):
        s = build(args.npz, theme,
                  os.path.join(args.out_dir, "reward_distribution_%s.png" % theme))
    print("bins:", s["counts"], flush=True)


if __name__ == "__main__":
    main()
