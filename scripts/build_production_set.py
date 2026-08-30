#!/usr/bin/env python
"""Assemble the shipped model's training set, with the evaluation half held out.

The obvious assembly -- concatenate every injected and every real row -- scored
1.000 on "real data" with all four classes perfect, because the evaluation rows
were in the training set. An earlier version of check_domain_gap PASSED that and
printed "report 1.000 as the model's accuracy".

So real targets are split by star id here, before training: the train half joins
the injected rows, the test half is written separately and never trained on.
Both files carry a `target` column so the leakage check can verify it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.features import FEATURE_NAMES        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--injected", default="data/processed/train_injected.parquet")
    ap.add_argument("--real", default="data/processed/real.parquet")
    ap.add_argument("--out-train", default="data/processed/train_production.parquet")
    ap.add_argument("--out-test", default="data/processed/eval_holdout.parquet")
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    inj = pd.read_parquet(args.injected)
    real = pd.read_parquet(args.real)
    if "error" in real:
        real = real[real["error"].isna()]

    targets = np.asarray(real["target"].astype(str).unique(), dtype=object)
    tr_t, te_t = train_test_split(targets, test_size=args.test_frac,
                                  random_state=args.seed)
    real_tr = real[real["target"].astype(str).isin(set(tr_t))]
    real_te = real[real["target"].astype(str).isin(set(te_t))]

    keep = FEATURE_NAMES + ["label", "target"]
    if "target" not in inj.columns:
        inj = inj.assign(target=inj.get("baseline_target"))
    train = pd.concat([inj[keep], real_tr[keep]], ignore_index=True)
    test = real_te[keep].copy()

    overlap = set(train["target"].dropna()) & set(test["target"].dropna())
    assert not overlap, f"leak: {sorted(overlap)[:5]}"

    os.makedirs(os.path.dirname(args.out_train) or ".", exist_ok=True)
    train.to_parquet(args.out_train, index=False)
    test.to_parquet(args.out_test, index=False)

    print(f"train -> {args.out_train}  ({len(train)} rows: "
          f"{len(inj)} injected + {len(real_tr)} real)")
    print(train["label"].value_counts().to_string())
    print(f"\nheld-out eval -> {args.out_test}  ({len(test)} real rows)")
    print(test["label"].value_counts().to_string())
    print(f"\nverified: 0 targets shared between train and eval")


if __name__ == "__main__":
    main()
