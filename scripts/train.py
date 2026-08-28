#!/usr/bin/env python
"""Train the light-curve classifier and report held-out performance."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet import classify                      # noqa: E402
from exodet.config import CLASSES                # noqa: E402


def reliability(y_true, proba, n_bins=10):
    """Calibration check: does a stated confidence of p come true p of the time?"""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 5:
            rows.append((0.5 * (lo + hi), float(conf[m].mean()),
                         float(correct[m].mean()), int(m.sum())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="data/processed/train.parquet")
    ap.add_argument("-o", "--outdir", default="models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", default="data/processed/real.parquet",
                    help="real-disposition set used for the domain-gap check")
    ap.add_argument("--gap-threshold", type=float, default=0.30)
    ap.add_argument("--skip-gap-check", action="store_true",
                    help="only for exploratory runs; never for a model you report")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    print(f"loaded {len(df)} rows from {args.data}")
    print(df["label"].value_counts().to_string(), "\n")

    model, calibrator, rep, (X_te, y_te, proba) = classify.train(df, seed=args.seed)

    print("=" * 62)
    print(f"held-out accuracy : {rep.accuracy:.3f}")
    print(f"macro F1          : {rep.macro_f1:.3f}")
    print(f"log loss (raw)    : {rep.log_loss_raw:.4f}")
    print(f"log loss (calib.) : {rep.log_loss_calibrated:.4f}")
    print("=" * 62)
    print(rep.report_text)

    print("confusion matrix (rows = true, cols = predicted)")
    hdr = "".join(f"{c[:8]:>9}" for c in CLASSES)
    print(f"{'':>9}{hdr}")
    for name, row in zip(CLASSES, rep.confusion):
        print(f"{name:>9}" + "".join(f"{v:>9d}" for v in row))

    print("\ntop feature importances")
    for k, v in rep.importances.head(10).items():
        print(f"  {k:<24s} {v:6.0f}")

    print("\ncalibration (stated confidence vs actual accuracy)")
    print(f"  {'bin':>6} {'stated':>8} {'actual':>8} {'n':>5}")
    for centre, stated, actual, n in reliability(y_te, proba):
        print(f"  {centre:6.2f} {stated:8.3f} {actual:8.3f} {n:5d}")

    classify.save(model, calibrator, rep, args.outdir)
    print(f"\nsaved -> {args.outdir}/classifier.joblib")

    # The domain-gap check runs as part of training, not as a separate command
    # somebody has to remember. The original failure -- publishing a 0.984
    # self-test score while real-world accuracy was ~0.42 -- was caught by a
    # human running an experiment by hand, and nothing automated would have
    # flagged it.
    if not args.skip_gap_check:
        print()
        import subprocess
        rc = subprocess.call([sys.executable,
                              os.path.join(os.path.dirname(__file__),
                                           "check_domain_gap.py"),
                              "--train", args.data,
                              "--real", args.real,
                              "--threshold", str(args.gap_threshold)])
        if rc != 0:
            print("\n[train] domain-gap check FAILED -- the saved model's "
                  "self-test score is not a trustworthy accuracy figure.",
                  file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
