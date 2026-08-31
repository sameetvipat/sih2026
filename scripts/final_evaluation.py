#!/usr/bin/env python
"""Phase 4 evaluation: every number the directive asks to freeze, in one run.

Reports against BOTH baselines, because they measure different things and
conflating them is what produced the discrepancy this project already had to
correct:

  protocol number (0.653)  a fresh 4-class uncalibrated LightGBM on the shared
                           classes, built by check_domain_gap's comparison. The
                           like-for-like number -- same construction before and
                           after -- so the directive's regression rule applies
                           to this one.
  shipped number  (0.584)  models/classifier.joblib itself: 5 classes including
                           noise, probability-calibrated, the artefact api/main.py
                           actually serves. This is what a user gets, so it is
                           what the report should headline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.features import FEATURE_NAMES            # noqa: E402
from exodet.metrics import report, wilson_interval    # noqa: E402

SHARED = ["transit", "eclipse", "blend", "variable"]

# Section 0 baselines, stated exactly as the directive gives them.
BASE_PROTOCOL = dict(acc=0.653, macro_f1=0.614,
                     per_class={"transit": (0.58, 0.72), "eclipse": (0.87, 0.83),
                                "blend": (0.44, 0.35), "variable": (0.58, 0.57)})
BASE_SHIPPED = dict(acc=0.584)


def clean(X):
    return X.replace([np.inf, -np.inf], np.nan)


def protocol_number(train_path, eval_path, seed=0):
    """Reproduce check_domain_gap's construction exactly: 4-class, uncalibrated."""
    import lightgbm as lgb
    idx = {c: i for i, c in enumerate(SHARED)}
    tr = pd.read_parquet(train_path)
    te = pd.read_parquet(eval_path)
    for d in (tr, te):
        if "error" in d:
            d.drop(d[d.error.notna()].index, inplace=True)
    tr, te = tr[tr.label.isin(SHARED)], te[te.label.isin(SHARED)]

    shared_targets = set(tr.get("target", pd.Series(dtype=str)).dropna()) & \
                     set(te.get("target", pd.Series(dtype=str)).dropna())
    if shared_targets:
        raise SystemExit(f"LEAKAGE: {len(shared_targets)} targets in both sets")

    m = lgb.LGBMClassifier(objective="multiclass", num_class=len(SHARED),
                           n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=10, subsample=0.9, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0,
                           class_weight="balanced", random_state=seed,
                           n_jobs=-1, verbose=-1)
    m.fit(clean(tr[FEATURE_NAMES]), tr.label.map(idx).values)
    y = te.label.map(idx).values
    pred = m.predict(clean(te[FEATURE_NAMES]))
    return y, pred, SHARED, len(tr)


def shipped_number(eval_path, model_path="models/classifier.joblib"):
    """Evaluate the artefact the API actually serves, on its own class set."""
    import joblib
    d = joblib.load(model_path)
    est = d["calibrator"] if d["calibrator"] is not None else d["model"]
    classes = d["classes"]
    idx = {c: i for i, c in enumerate(classes)}

    te = pd.read_parquet(eval_path)
    if "error" in te:
        te = te[te.error.isna()]
    te = te[te.label.isin(classes)]
    X = clean(te[d["features"]])
    y = te.label.map(idx).values
    return y, est.predict(X), classes, len(te)


def delta(new, old, label, better="higher"):
    d = new - old
    arrow = "improved" if (d > 0) == (better == "higher") else "REGRESSED"
    if abs(d) < 0.005:
        arrow = "flat"
    return f"  {label:<34} {old:.3f} -> {new:.3f}   {d:+.3f}   {arrow}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train_production.parquet")
    ap.add_argument("--eval", default="data/processed/eval_holdout.parquet")
    ap.add_argument("--out", default="reports/final_evaluation.json")
    args = ap.parse_args()

    out = {}

    print("=" * 68)
    print("PROTOCOL NUMBER — 4-class uncalibrated, like-for-like vs 0.653")
    print("=" * 68)
    y, pred, classes, n_tr = protocol_number(args.train, args.eval)
    r = report(y, pred, classes, f"trained on {n_tr} rows")
    out["protocol"] = r
    print()
    print(delta(r["accuracy"], BASE_PROTOCOL["acc"], "accuracy"))
    print(delta(r["macro_f1"], BASE_PROTOCOL["macro_f1"], "macro F1"))
    from sklearn.metrics import precision_recall_fscore_support
    p, rc, _, _ = precision_recall_fscore_support(
        y, pred, labels=list(range(len(classes))), zero_division=0)
    for i, c in enumerate(classes):
        bp, br = BASE_PROTOCOL["per_class"][c]
        print(delta(p[i], bp, f"{c} precision"))
        print(delta(rc[i], br, f"{c} recall"))
    out["protocol"]["per_class"] = {c: {"precision": float(p[i]), "recall": float(rc[i])}
                                    for i, c in enumerate(classes)}

    print()
    print("=" * 68)
    print("SHIPPED NUMBER — models/classifier.joblib, what the API serves")
    print("=" * 68)
    try:
        y2, pred2, cls2, n2 = shipped_number(args.eval)
        r2 = report(y2, pred2, cls2, "shipped artefact")
        out["shipped"] = r2
        print()
        print(delta(r2["accuracy"], BASE_SHIPPED["acc"], "accuracy"))
    except Exception as exc:
        print(f"  could not evaluate shipped model: {exc}")
        out["shipped"] = {"error": str(exc)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nfrozen -> {args.out}")


if __name__ == "__main__":
    main()
