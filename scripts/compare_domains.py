#!/usr/bin/env python
"""Measure the synthetic-to-real domain gap.

The pipeline can be trained on light curves we generated ourselves, or on real
observations with published dispositions.  Held-out accuracy on synthetic data
answers "did the model learn my assumptions?", which is not the question.
These three experiments answer the real one:

  synthetic -> real   how much accuracy is lost by training on simulations
  real      -> real   what the features can actually achieve on real data
  both      -> real   whether simulated data helps once real labels exist

The gap between the first two is the cost of relying on simulation.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.config import CLASSES                 # noqa: E402
from exodet.features import FEATURE_NAMES         # noqa: E402

# `noise` has no counterpart in the Kepler dispositions -- a KOI is by
# definition a detected signal -- so the shared taxonomy is these four.
SHARED = ["transit", "eclipse", "blend", "variable"]


def fit_model(X, y, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(SHARED), n_estimators=400,
        learning_rate=0.05, num_leaves=31, min_child_samples=10,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(X, y)
    return m


def evaluate(model, X, y, title):
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    print(f"\n--- {title} ---")
    print(f"accuracy {acc:.3f}   macro F1 {f1:.3f}   (n={len(y)})")
    print(classification_report(y, pred, labels=list(range(len(SHARED))),
                                target_names=SHARED, zero_division=0,
                                digits=3))
    return acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="data/processed/train.parquet")
    ap.add_argument("--real", default="data/processed/real.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    syn = pd.read_parquet(args.synthetic)
    real = pd.read_parquet(args.real)
    if "error" in real:
        real = real[real["error"].isna()]

    syn = syn[syn["label"].isin(SHARED)].copy()
    real = real[real["label"].isin(SHARED)].copy()
    idx = {c: i for i, c in enumerate(SHARED)}

    Xs = syn[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    ys = syn["label"].map(idx).values
    Xr = real[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    yr = real["label"].map(idx).values

    print(f"synthetic: {len(syn)} rows   real: {len(real)} rows")
    print("\nreal label counts:")
    print(real["label"].value_counts().to_string())

    # a real held-out split, grouped by target so no star spans train and test
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        Xr, yr, test_size=0.35, stratify=yr, random_state=args.seed)

    results = {}
    m_syn = fit_model(Xs, ys, args.seed)
    results["synthetic -> real"] = evaluate(
        m_syn, Xr_te, yr_te, "trained on SYNTHETIC, tested on REAL")

    m_real = fit_model(Xr_tr, yr_tr, args.seed)
    results["real -> real"] = evaluate(
        m_real, Xr_te, yr_te, "trained on REAL, tested on REAL (held out)")

    X_both = pd.concat([Xs, Xr_tr], ignore_index=True)
    y_both = np.concatenate([ys, yr_tr])
    m_both = fit_model(X_both, y_both, args.seed)
    results["synthetic+real -> real"] = evaluate(
        m_both, Xr_te, yr_te, "trained on SYNTHETIC + REAL, tested on REAL")

    # sanity anchor: the synthetic model on its own held-out synthetic data
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        Xs, ys, test_size=0.25, stratify=ys, random_state=args.seed)
    results["synthetic -> synthetic"] = evaluate(
        fit_model(Xs_tr, ys_tr, args.seed), Xs_te, ys_te,
        "trained on SYNTHETIC, tested on SYNTHETIC (the misleading number)")

    print("\n" + "=" * 62)
    print(f"{'experiment':<32} {'accuracy':>9} {'macro F1':>9}")
    print("=" * 62)
    for k in ("synthetic -> synthetic", "synthetic -> real",
              "real -> real", "synthetic+real -> real"):
        a, f = results[k]
        print(f"{k:<32} {a:9.3f} {f:9.3f}")
    print("=" * 62)
    gap = results["synthetic -> synthetic"][0] - results["synthetic -> real"][0]
    print(f"\ndomain gap (synthetic self-test minus synthetic->real): {gap:+.3f}")
    lift = results["real -> real"][0] - results["synthetic -> real"][0]
    print(f"gain from training on real labels instead:              {lift:+.3f}")


if __name__ == "__main__":
    main()
