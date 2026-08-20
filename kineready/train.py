"""Stage 2: train the KineReady ensemble. No simulator import -- runs anywhere the data is.

Loss (proposal Sec 13):
  BCE on `exist`                          -- the primary label.
  BCE + Brier on `robust`, MASKED         -- only the 'robust' sampler carries this target; the
                                             mask keeps the other slices from being trained
                                             toward a fabricated 0.
The proposal also lists a pairwise ranking term over same-target/different-base groups. That term
belongs to a task-conditioned dataset; this model is task-agnostic and its rows are independent
targets, so there are no natural ranking groups to form. Ranking quality is measured instead where
it is actually defined -- Stage 3, against the exact oracle over candidate base poses.
"""
import argparse
import json
import os
import time

import numpy as np
import torch as th
import torch.nn as nn

from .dataset import KineReadyDataset, geometric_splits, load_shards
from .model import KineReadyEnsemble, expected_calibration_error, fpr_at_recall


def train(data_dir, out_path, m=5, epochs=30, batch=4096, lr=2e-3, seed=0, device=None,
          verbose=True):
    device = device or ("cuda" if th.cuda.is_available() else "cpu")
    data = load_shards(data_dir)
    splits = geometric_splits(data, seed=seed)
    if verbose:
        print("rows: %d | %s" % (len(data["exist"]),
                                 {k: len(v) for k, v in splits.items()}), flush=True)
        print("overall positive rate: %.1f%%" % (100 * data["exist"].mean()), flush=True)

    tr = KineReadyDataset(data, splits["train"])
    va = KineReadyDataset(data, splits["val"])
    loader = th.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, drop_last=False)

    model = KineReadyEnsemble(m=m).to(device)
    opt = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(loader)))
    bce = nn.functional.binary_cross_entropy_with_logits

    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y, r, rm in loader:
            x, y, r, rm = x.to(device), y.to(device), r.to(device), rm.to(device)
            e_log, r_log = model(x)
            # Each member sees the same batch; summing their losses trains them jointly while
            # keeping their parameters independent, so the only thing coupling them is the data.
            loss = bce(e_log, y.expand_as(e_log))
            if rm.sum() > 0:
                r_prob = th.sigmoid(r_log)
                w = rm.expand_as(r_log)
                loss = loss + (bce(r_log, r.expand_as(r_log), reduction="none") * w).sum() / w.sum()
                loss = loss + ((r_prob - r.expand_as(r_prob)) ** 2 * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.detach().item() * len(x)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print("epoch %2d  loss %.4f" % (ep, tot / max(1, len(tr))), flush=True)

    model.eval()
    t_fit = model.fit_temperature(va.x.to(device), va.y.to(device))
    if verbose:
        print("calibration temperature: %.3f | train wall %.1f s"
              % (t_fit, time.perf_counter() - t0), flush=True)

    metrics = {"temperature": t_fit,
               "n_train": len(tr),
               "positive_rate": float(data["exist"].mean())}
    for name in ("test_iid", "test_region", "test_orientation"):
        idx = splits[name]
        if len(idx) == 0:
            continue
        ds = KineReadyDataset(data, idx)
        metrics[name] = evaluate(model, ds, device)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    th.save({"state_dict": model.state_dict(), "m": m, "metrics": metrics}, out_path)
    with open(os.path.splitext(out_path)[0] + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        note = {"test_iid": "deployment-relevant",
                "test_region": "structure probe, worse by construction",
                "test_orientation": "orientation coverage check"}
        for name in ("test_iid", "test_region", "test_orientation"):
            if name in metrics:
                d = metrics[name]
                print("%-17s n=%-7d AUROC %.4f  Brier %.4f  ECE %.4f  FPR@95%%recall %.4f   (%s)"
                      % (name, d["n"], d["auroc"], d["brier"], d["ece"], d["fpr_at_95_recall"],
                         note[name]), flush=True)
    return model, metrics


@th.no_grad()
def evaluate(model, ds, device):
    out = model.predict(ds.x.to(device))
    p = out["p_exist"].cpu().numpy()
    y = ds.y.numpy()
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "auroc": _auroc(p, y),
        "auprc": _auprc(p, y),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(p, y),
        "fpr_at_95_recall": fpr_at_recall(p, y, 0.95),
        "mean_std": float(out["p_std"].mean()),
    }


def _auroc(p, y):
    """Rank-based AUROC (ties averaged), so no sklearn dependency on the box."""
    y = np.asarray(y, dtype=bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks within ties
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _auprc(p, y):
    """Average precision via the step-wise precision-recall sum."""
    y = np.asarray(y, dtype=bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    return float((precision * ys).sum() / ys.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="kineready_models/kineready.pt")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.data, args.out, m=args.members, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
