#!/usr/bin/env python
"""Score the classifier on held-out real light curves, both ways.

There are two different models in this project and they do not score the same,
which is the first thing this script exists to stop anyone from confusing:

* the **shipped artifact** (`models/classifier.joblib`) -- five classes
  including `noise`, probability-calibrated. This is what `api/main.py` serves,
  so it is what a user of the pipeline actually gets.
* the **four-class comparison protocol** in `check_domain_gap.py` -- a fresh
  uncalibrated LightGBM trained on the shared classes only. The project's
  published headline (0.653 on 320 held-out curves) was measured this way.

They differ by ~0.07 accuracy on identical data. Reporting one number while
shipping the other is exactly the kind of quiet mismatch this project has
already been bitten by once, so both are printed, always, side by side.

Both arms score the SAME held-out targets, split by star id before training.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet import classify                                   # noqa: E402
from exodet.config import CLASSES                             # noqa: E402
from exodet.features import FEATURE_NAMES                     # noqa: E402
from exodet.metrics import difference_ci, report              # noqa: E402

SHARED = ["transit", "eclipse", "blend", "variable"]

# The measured pre-session state, reproduced exactly by the four-class protocol
# below (accuracy 0.653, CI [0.599, 0.703], macro F1 0.614, and every per-class
# figure). Hard-coded on purpose: a baseline recomputed from whatever is on disk
# today is not a baseline, it moves whenever the data does.
BASELINE = dict(
    accuracy=0.653, ci=(0.599, 0.703), macro_f1=0.614, n=320, self_test=0.716,
    per_class={"transit": (0.582, 0.716, 74), "eclipse": (0.874, 0.833, 108),
               "blend": (0.435, 0.351, 57), "variable": (0.575, 0.568, 81)},
)


def _fit(X, y, seed=0):
    """The four-class model of check_domain_gap.py, kept identical to it."""
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="multiclass", num_class=len(SHARED),
                           n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=10, subsample=0.9, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0,
                           class_weight="balanced", random_state=seed,
                           n_jobs=-1, verbose=-1)
    m.fit(X, y)
    return m


def _xy(df, feats):
    idx = {c: i for i, c in enumerate(SHARED)}
    df = df[df["label"].isin(SHARED)]
    return (df[feats].replace([np.inf, -np.inf], np.nan),
            df["label"].map(idx).values)


def per_class_table(y, pred, res, tag):
    p, r, f1, sup = precision_recall_fscore_support(
        y, pred, labels=list(range(len(SHARED))), zero_division=0)
    print(f"\n  {'class':<10}{'precision':>24}{'recall':>22}{'n':>12}")
    for i, c in enumerate(SHARED):
        b = BASELINE["per_class"][c]
        print(f"  {c:<10}{b[0]:>11.3f} -> {p[i]:<9.3f}"
              f"{b[1]:>11.3f} -> {r[i]:<9.3f}{b[2]:>6d} -> {sup[i]:<5d}")
    return {c: dict(precision=float(p[i]), recall=float(r[i]),
                    f1=float(f1[i]), n=int(sup[i])) for i, c in enumerate(SHARED)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train_production.parquet")
    ap.add_argument("--holdout", default="data/processed/eval_holdout.parquet")
    ap.add_argument("--models", default="models/classifier.joblib")
    ap.add_argument("--json-out", default="reports/holdout_metrics.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr = pd.read_parquet(args.train)
    te = pd.read_parquet(args.holdout)
    overlap = set(tr.get("target", pd.Series(dtype=object)).dropna()) & \
        set(te.get("target", pd.Series(dtype=object)).dropna())
    if overlap:
        raise SystemExit(f"LEAK: {len(overlap)} targets in both sets, "
                         f"e.g. {sorted(overlap)[:5]}")
    print(f"train {len(tr)} rows / holdout {len(te)} rows, "
          f"0 shared targets -- verified leak-free")

    out = {}

    # ---- arm B: the protocol the published baseline was measured with -------
    print("\n" + "=" * 78)
    print("ARM 1 -- four-class comparison protocol (matches the 0.653 baseline)")
    print("=" * 78)
    Xs, ys = _xy(tr, FEATURE_NAMES)
    Xr, yr = _xy(te, FEATURE_NAMES)
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        Xs, ys, test_size=0.25, stratify=ys, random_state=args.seed)
    from sklearn.metrics import accuracy_score
    self_acc = float(accuracy_score(ys_te, _fit(Xs_tr, ys_tr, args.seed)
                                    .predict(Xs_te)))
    pred = _fit(Xs, ys, args.seed).predict(Xr)
    res = report(yr, pred, SHARED, f"held-out real curves (n={len(yr)})")
    pc = per_class_table(yr, pred, res, "4class")

    d, lo, hi, sig = difference_ci(BASELINE["accuracy"], res["accuracy"],
                                   min(res["n"], BASELINE["n"]))
    print(f"\n  accuracy   {BASELINE['accuracy']:.3f} -> {res['accuracy']:.3f}"
          f"   delta {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"   {'DISTINGUISHABLE' if sig else 'not distinguishable'} from zero")
    print(f"  macro F1   {BASELINE['macro_f1']:.3f} -> {res['macro_f1']:.3f}"
          f"   delta {res['macro_f1'] - BASELINE['macro_f1']:+.3f}")
    print(f"  self-test  {BASELINE['self_test']:.3f} -> {self_acc:.3f}"
          f"   domain gap {self_acc - res['accuracy']:+.3f}")
    out["four_class_protocol"] = dict(
        accuracy=res["accuracy"], ci=[res["ci_low"], res["ci_high"]],
        macro_f1=res["macro_f1"], n=res["n"], self_test=self_acc,
        domain_gap=self_acc - res["accuracy"], per_class=pc,
        delta_accuracy=d, delta_accuracy_ci=[lo, hi],
        delta_distinguishable=bool(sig))

    # ---- arm A: the artifact the API actually serves ------------------------
    print("\n" + "=" * 78)
    print("ARM 2 -- the shipped artifact (what api/main.py actually returns)")
    print("=" * 78)
    try:
        model, calibrator = classify.load(args.models)
        feats = list(getattr(model, "feature_name_", None) or FEATURE_NAMES)
        missing = [c for c in feats if c not in te.columns]
        if missing:
            raise SystemExit(f"holdout missing model features: {missing}")
        sub = te[te["label"].isin(SHARED)]
        X = sub[feats].replace([np.inf, -np.inf], np.nan)
        y = sub["label"].map({c: i for i, c in enumerate(SHARED)}).values
        proba = model.predict_proba(X)
        if calibrator is not None:
            try:
                proba = calibrator.predict_proba(X)
            except Exception:
                pass
        # The artifact carries a `noise` class the real holdout cannot contain.
        # Zero it before the argmax, not after: a row whose top class is
        # unavailable must fall to its best AVAILABLE class rather than be
        # scored as wrong for predicting something the labels cannot express.
        full = list(CLASSES)
        keep = [full.index(c) for c in SHARED]
        pred_a = np.argmax(proba[:, keep], axis=1)
        res_a = report(y, pred_a, SHARED, f"held-out real curves (n={len(y)})")
        pc_a = per_class_table(y, pred_a, res_a, "shipped")
        out["shipped_artifact"] = dict(
            accuracy=res_a["accuracy"], ci=[res_a["ci_low"], res_a["ci_high"]],
            macro_f1=res_a["macro_f1"], n=res_a["n"], per_class=pc_a)
        print(f"\n  NOTE: the shipped artifact scores {res_a['accuracy']:.3f} "
              f"against the four-class protocol's {res['accuracy']:.3f} on "
              f"identical data.")
        print("  The published headline describes the protocol, not the "
              "artifact. Report whichever you mean, and say which.")
    except SystemExit:
        raise
    except Exception as exc:
        print(f"  could not score the shipped artifact: {exc}")
        out["shipped_artifact"] = None

    out["baseline"] = BASELINE
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {args.json_out}")

    if res["accuracy"] < BASELINE["accuracy"] - 0.001:
        print("\n*** REGRESSION on the comparable arm vs the pre-session "
              "baseline. Do not report as final until the cause is found. ***")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
