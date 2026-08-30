#!/usr/bin/env python
"""Assign disjoint train/test splits to the quiet-baseline bank.

Without this, the same real star can supply the noise floor for a training
example and a test example, differing only in which synthetic signal was
injected on top.  The model could then learn to recognise *that star's*
systematics rather than generalisable signal-versus-noise structure, and the
test score would be inflated by exactly the kind of leakage this project has
already been burned by once.

Splitting is done on baseline identity, before any examples are generated, and
stratified by brightness so both pools span the full noise regime.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--manifest", default="data/baselines/manifest.csv")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    rng = np.random.default_rng(args.seed)

    # stratify by brightness so neither pool is all bright or all faint
    df["split"] = "train"
    for b, grp in df.groupby("brightness_bin"):
        idx = grp.index.to_numpy().copy()   # to_numpy() can be read-only
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * args.test_frac)))
        df.loc[idx[:n_test], "split"] = "test"

    df.to_csv(args.manifest, index=False)

    train_ids = set(df[df.split == "train"]["target"])
    test_ids = set(df[df.split == "test"]["target"])
    overlap = train_ids & test_ids
    assert not overlap, f"baseline leaked across splits: {sorted(overlap)[:5]}"

    print(f"{len(df)} baselines split -> {len(train_ids)} train, {len(test_ids)} test")
    print(df.groupby(["brightness_bin", "split"]).size().to_string())
    print("\nverified: zero baseline overlap between train and test")


if __name__ == "__main__":
    main()
