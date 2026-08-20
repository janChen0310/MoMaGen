"""KineReadyMLP: 9 numbers -> P(an IK solution exists) + a reachability margin.

Architecture per proposal Sec 10 (9 -> 128 -> 128 -> 128 residual -> 64 -> heads). Tiny on
purpose: the input is already the right coordinates. The factorization in `frames.py` absorbs
where the base stands into the target-in-base-frame transform, so the function to learn is a
fixed property of the arm -- one smooth-ish region of SE(3) -- not a function of the world.

Two heads, because the binary label cannot express what navigation needs:
  exist   P(IK solution exists) -- matches the teacher's label.
  robust  Fraction of nearby perturbations that stay reachable. A target deep in the workspace
          and one on its edge both have exist=1; only this separates them, and choosing between
          those two base poses is the entire point of the metric.

Ensembles give the uncertainty that makes a conservative filter possible (Sec 14). The dangerous
error is a FALSE POSITIVE -- a base pose the model calls reachable and the planner then fails on,
wasting a real navigation episode -- so downstream consumers get a lower confidence bound rather
than the mean.
"""
import numpy as np
import torch as th
import torch.nn as nn


class KineReadyMLP(nn.Module):
    def __init__(self, in_dim=9, width=128, head_dim=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU())
        self.res = nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width))
        self.act = nn.SiLU()
        self.trunk = nn.Sequential(nn.Linear(width, head_dim), nn.SiLU())
        self.head_exist = nn.Linear(head_dim, 1)
        self.head_robust = nn.Linear(head_dim, 1)

    def forward(self, x):
        h = self.inp(x)
        h = self.act(h + self.res(h))
        h = self.trunk(h)
        return self.head_exist(h).squeeze(-1), self.head_robust(h).squeeze(-1)


class KineReadyEnsemble(nn.Module):
    """M independently-seeded MLPs. Disagreement between them is the uncertainty estimate."""

    def __init__(self, m=5, **kw):
        super().__init__()
        self.members = nn.ModuleList([KineReadyMLP(**kw) for _ in range(m)])
        # Temperature scaling, fitted on validation after training (Sec 14). Starts at 1 so an
        # uncalibrated ensemble behaves exactly like a plain one rather than silently distorting.
        self.register_buffer("temperature", th.ones(1))

    def forward(self, x):
        # One pass per member, both heads taken from it. Indexing `m(x)[0]` and `m(x)[1]` in two
        # separate comprehensions would run the whole trunk twice for every member -- double the
        # inference cost of the thing whose entire purpose is being cheap.
        outs = [m(x) for m in self.members]
        e = th.stack([o[0] for o in outs])                     # (M, N) exist logits
        r = th.stack([o[1] for o in outs])
        return e, r

    @th.no_grad()
    def predict(self, x, beta=1.0):
        """-> dict with mean/std/lcb of P(exist) and the mean robust score.

        `lcb = clip(mean - beta*std)` is what a filter should threshold on: where the members
        disagree it moves toward 0, so uncertainty is treated as "may not be reachable" rather
        than averaged away into a confident-looking number.
        """
        e, r = self(x)
        p = th.sigmoid(e / self.temperature)
        mean, std = p.mean(0), p.std(0)
        return {
            "p_exist": mean,
            "p_std": std,
            "p_lcb": (mean - beta * std).clamp(0.0, 1.0),
            "robust": th.sigmoid(r).mean(0),
        }

    @th.no_grad()
    def fit_temperature(self, x_val, y_val, grid=None):
        """Pick the temperature minimizing validation NLL.

        Deep ensembles trained with BCE are typically over-confident; without this the calibration
        metric (ECE) reports a number nobody can act on.
        """
        grid = grid if grid is not None else th.logspace(-0.7, 0.7, 41)
        logits = self(x_val)[0].mean(0)
        best_t, best_nll = 1.0, float("inf")
        for t in grid:
            nll = nn.functional.binary_cross_entropy_with_logits(logits / t, y_val).item()
            if nll < best_nll:
                best_t, best_nll = float(t), nll
        self.temperature.fill_(best_t)
        return best_t


def expected_calibration_error(p, y, n_bins=15):
    """Mean |confidence - accuracy| over equal-width confidence bins."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def fpr_at_recall(p, y, target_recall=0.95):
    """False-positive rate at the threshold achieving `target_recall` on positives.

    The headline safety number: how often the model green-lights an unreachable target when tuned
    to rarely miss a reachable one. AUROC can look excellent while this is unusable.
    """
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    thr = np.quantile(p[y], 1.0 - target_recall)
    return float((p[~y] >= thr).mean())
