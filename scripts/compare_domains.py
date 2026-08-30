#!/usr/bin/env python
"""Measure the synthetic-to-real domain gap, before and after real-background injection.

Held-out accuracy on the training distribution answers "did the model learn my
assumptions?", which is not the question.  These experiments answer the real one
by always testing on the same held-out set of REAL catalogued objects:

    <training regime>  ->  real held-out

Regimes compared:
    pure synthetic        the original generator: white noise + a few sinusoids
    real-injected         known signals injected onto real quiet stars
    real only             catalogued dispositions, no synthetic component
    real-injected + real  both

Leakage discipline, which the numbers are worthless without:
  * real light curves are split by TARGET (star id), so no star spans train/test
  * VARIABLE has no injected counterpart (a real variable's structure is exactly
    what the simulator fails at), so it is sourced from the real TRAIN half only
    -- never from the half used for evaluation
  * injected examples come only from baselines in the train split, assigned by
    scripts/split_baselines.py before any example was generated
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

from exodet.features import FEATURE_NAMES         # noqa: E402

# `noise` is excluded: a Kepler KOI is by definition a detected signal, so the
# real catalogue has no counterpart and the class cannot be compared fairly.
SHARED = ["transit", "eclipse", "blend", "variable"]
IDX: dict[str, int] = {}


def set_classes(classes):
    """Restrict the comparison to a subset of the taxonomy.

    The four-class comparison is confounded: VARIABLE has no injected
    counterpart, so it is sourced from the real train half (32 rows against
    ~600 injected). With class_weight="balanced" each of those rows carries
    ~6x the weight of an injected one, and the model over-predicts VARIABLE
    (measured: recall 0.889 at precision 0.229). Restricting to the three
    injectable classes compares like with like.
    """
    global SHARED, IDX
    SHARED = list(classes)
    IDX = {c: i for i, c in enumerate(SHARED)}


set_classes(["transit", "eclipse", "blend", "variable"])


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "error" in df:
        df = df[df["error"].isna()]
    return df[df["label"].isin(SHARED)].copy()


def xy(df: pd.DataFrame):
    X = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    return X, df["label"].map(IDX).values


def fit(X, y, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="multiclass", num_class=len(SHARED),
                           n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=10, subsample=0.9, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0,
                           class_weight="balanced", random_state=seed,
                           n_jobs=-1, verbose=-1)
    m.fit(X, y)
    return m


def evaluate(model, X, y, title, show_report=False):
    pred = model.predict(X)
    acc, f1 = accuracy_score(y, pred), f1_score(y, pred, average="macro")
    print(f"\n--- {title} ---")
    print(f"accuracy {acc:.3f}   macro F1 {f1:.3f}   (n={len(y)})")
    if show_report:
        print(classification_report(y, pred, labels=list(range(len(SHARED))),
                                    target_names=SHARED, zero_division=0,
                                    digits=3))
    return acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="data/processed/train.parquet",
                    help="pure-synthetic training set (the original regime)")
    ap.add_argument("--injected", default="data/processed/train_injected.parquet",
                    help="real-background-injected training set")
    ap.add_argument("--real", default="data/processed/real.parquet")
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", nargs="+", default=None,
                    help="restrict the taxonomy, e.g. --classes transit eclipse "
                         "blend for the confound-free three-class comparison")
    args = ap.parse_args()
    if args.classes:
        set_classes(args.classes)
        print(f"restricted to {len(SHARED)} classes: {', '.join(SHARED)}\n")

    real = load(args.real)
    syn = load(args.synthetic) if os.path.exists(args.synthetic) else None
    inj = load(args.injected) if os.path.exists(args.injected) else None

    # --- split real data by star, not by row -------------------------------
    # .unique() on an Arrow-backed column is not a plain ndarray
    targets = np.asarray(real["target"].astype(str).unique(), dtype=object)
    tr_t, te_t = train_test_split(targets, test_size=args.test_frac,
                                  random_state=args.seed)
    real_tr = real[real["target"].isin(tr_t)]
    real_te = real[real["target"].isin(te_t)]
    assert not (set(real_tr["target"]) & set(real_te["target"])), "star-id leak"

    print(f"real: {len(real)} rows / {len(targets)} stars "
          f"-> {len(real_tr)} train, {len(real_te)} test")
    print(f"held-out class counts: {real_te['label'].value_counts().to_dict()}")
    if inj is not None:
        print(f"injected: {len(inj)} rows over "
              f"{inj['baseline_target'].nunique()} baselines")

    Xte, yte = xy(real_te)
    results: dict[str, tuple[float, float]] = {}

    # VARIABLE has no injected counterpart; take it from the real TRAIN half
    var_tr = (real_tr[real_tr["label"] == "variable"]
              if "variable" in SHARED else real_tr.iloc[0:0])

    if syn is not None:
        Xs, ys = xy(syn)
        results["pure synthetic -> real"] = evaluate(
            fit(Xs, ys, args.seed), Xte, yte, "PURE SYNTHETIC -> real held-out")
        Xs_a, Xs_b, ys_a, ys_b = train_test_split(
            Xs, ys, test_size=0.25, stratify=ys, random_state=args.seed)
        results["pure synthetic self-test"] = evaluate(
            fit(Xs_a, ys_a, args.seed), Xs_b, ys_b,
            "PURE SYNTHETIC self-test (the misleading number)")

    if inj is not None:
        inj_full = pd.concat([inj, var_tr], ignore_index=True)
        Xi, yi = xy(inj_full)
        results["real-injected -> real"] = evaluate(
            fit(Xi, yi, args.seed), Xte, yte,
            "REAL-INJECTED (+real variables) -> real held-out",
            show_report=True)
        Xi_a, Xi_b, yi_a, yi_b = train_test_split(
            Xi, yi, test_size=0.25, stratify=yi, random_state=args.seed)
        results["real-injected self-test"] = evaluate(
            fit(Xi_a, yi_a, args.seed), Xi_b, yi_b,
            "REAL-INJECTED self-test")

    Xr, yr = xy(real_tr)
    if len(np.unique(yr)) == len(SHARED):
        results["real only -> real"] = evaluate(
            fit(Xr, yr, args.seed), Xte, yte, "REAL ONLY -> real held-out")

    if inj is not None:
        Xb = pd.concat([xy(inj_full)[0], Xr], ignore_index=True)
        yb = np.concatenate([yi, yr])
        results["real-injected + real -> real"] = evaluate(
            fit(Xb, yb, args.seed), Xte, yte,
            "REAL-INJECTED + REAL -> real held-out")

    # --- the table ---------------------------------------------------------
    print("\n" + "=" * 66)
    print(f"{'regime':<38} {'accuracy':>10} {'macro F1':>10}")
    print("=" * 66)
    for k in ["pure synthetic self-test", "pure synthetic -> real",
              "real-injected self-test", "real-injected -> real",
              "real only -> real", "real-injected + real -> real"]:
        if k in results:
            a, f = results[k]
            print(f"{k:<38} {a:10.3f} {f:10.3f}")
    print("=" * 66)

    def gap(self_k, cross_k):
        if self_k in results and cross_k in results:
            return results[self_k][0] - results[cross_k][0]
        return None

    g_syn = gap("pure synthetic self-test", "pure synthetic -> real")
    g_inj = gap("real-injected self-test", "real-injected -> real")
    if g_syn is not None:
        print(f"\ndomain gap, pure synthetic : {g_syn:+.3f}")
    if g_inj is not None:
        print(f"domain gap, real-injected  : {g_inj:+.3f}")
    if g_syn is not None and g_inj is not None:
        print(f"gap closed by injection    : {g_syn - g_inj:+.3f}")
    if "pure synthetic -> real" in results and "real-injected -> real" in results:
        d = results["real-injected -> real"][0] - results["pure synthetic -> real"][0]
        print(f"real-world accuracy change : {d:+.3f}")


if __name__ == "__main__":
    main()
