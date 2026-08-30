#!/usr/bin/env python
"""Guard against the synthetic-to-real generalisation gap silently returning.

The original failure was reporting 0.984 held-out accuracy as if it were the
project's real-world performance, when accuracy on genuinely real data was
~0.42.  Nothing in the test suite caught that -- it took a human running a
controlled experiment by hand.  This script is the mechanical safeguard.

It measures:

    gap = self_test_accuracy - cross_domain_accuracy

and exits non-zero when the gap exceeds a threshold, so a retrain that quietly
destroys real-world generalisation while keeping a pretty self-test number
fails loudly instead of shipping.

Wired into `train.py` so it cannot be skipped by forgetting to run it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.features import FEATURE_NAMES          # noqa: E402
from exodet.metrics import report as metric_report  # noqa: E402

SHARED = ["transit", "eclipse", "blend", "variable"]

# Chosen with deliberate headroom over the gap actually measured after the
# real-background injection fix, so ordinary run-to-run variance does not trip
# it -- but far below the original 0.568 regression, which it must catch.
# Re-tune only alongside a recorded measurement, never to silence a failure.
DEFAULT_THRESHOLD = 0.30


def _xy(df: pd.DataFrame):
    idx = {c: i for i, c in enumerate(SHARED)}
    df = df[df["label"].isin(SHARED)]
    X = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    y = df["label"].map(idx).values
    return X, y, df


def _fit(X, y, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="multiclass", num_class=len(SHARED),
                           n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=10, subsample=0.9, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0,
                           class_weight="balanced", random_state=seed,
                           n_jobs=-1, verbose=-1)
    m.fit(X, y)
    return m


def measure(train_path: str, real_path: str, seed: int = 0):
    """Returns (self_test_acc, cross_domain_acc, gap, detail dict)."""
    tr = pd.read_parquet(train_path)
    real = pd.read_parquet(real_path)
    if "error" in real:
        real = real[real["error"].isna()]

    Xs, ys, _ = _xy(tr)
    Xr, yr, real_df = _xy(real)
    if len(np.unique(yr)) < 2:
        raise SystemExit("real set has fewer than two classes; cannot evaluate")

    # self-test: held out from the training distribution
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        Xs, ys, test_size=0.25, stratify=ys, random_state=seed)
    m_self = _fit(Xs_tr, ys_tr, seed)
    self_acc = accuracy_score(ys_te, m_self.predict(Xs_te))
    self_f1 = f1_score(ys_te, m_self.predict(Xs_te), average="macro")

    # cross-domain: the whole training set, tested on real data.
    # Per-class metrics are mandatory here, not optional: aggregate accuracy on
    # a class-imbalanced real set can look healthy while one class is never
    # predicted at all.
    m_full = _fit(Xs, ys, seed)
    pred_r = m_full.predict(Xr)
    r = metric_report(yr, pred_r, SHARED, "REAL-DATA EVALUATION")
    cross_acc, cross_f1 = r["accuracy"], r["macro_f1"]

    return self_acc, cross_acc, self_acc - cross_acc, {
        "self_f1": self_f1, "cross_f1": cross_f1,
        "ci": (r["ci_low"], r["ci_high"]),
        "n_train": len(ys), "n_real": len(yr),
        "real_counts": real_df["label"].value_counts().to_dict(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.parquet")
    ap.add_argument("--real", default="data/processed/real.parquet")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warn-only", action="store_true",
                    help="report but do not fail (for exploratory runs)")
    args = ap.parse_args()

    for p in (args.train, args.real):
        if not os.path.exists(p):
            print(f"[skip] {p} not found; cannot check the domain gap")
            return 0

    self_acc, cross_acc, gap, d = measure(args.train, args.real, args.seed)

    print("=" * 64)
    print("DOMAIN GAP CHECK")
    print("=" * 64)
    print(f"  training set        : {args.train}  (n={d['n_train']})")
    print(f"  real evaluation set : {args.real}  (n={d['n_real']})")
    print(f"  real class counts   : {d['real_counts']}")
    print(f"  self-test accuracy  : {self_acc:.3f}   macro F1 {d['self_f1']:.3f}")
    print(f"  real-data accuracy  : {cross_acc:.3f}   macro F1 {d['cross_f1']:.3f}"
          f"   95% CI [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]")
    print(f"  gap                 : {gap:+.3f}   (threshold {args.threshold:.3f})")
    print("=" * 64)

    if gap > args.threshold:
        print(f"\nFAIL: the model scores {self_acc:.3f} on its own held-out data "
              f"but {cross_acc:.3f} on real observations.\n"
              f"A gap of {gap:.3f} exceeds the {args.threshold:.3f} threshold, "
              "which means the reported accuracy does not reflect real-world\n"
              "performance. Do not publish the self-test number. Investigate the "
              "training distribution before shipping this model.", file=sys.stderr)
        return 1 if not args.warn_only else 0

    print(f"\nPASS: gap {gap:.3f} is within the {args.threshold:.3f} threshold.")
    print(f"Report {cross_acc:.3f} (real data) as the model's accuracy, "
          f"not {self_acc:.3f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
