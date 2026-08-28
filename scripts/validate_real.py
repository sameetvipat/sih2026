#!/usr/bin/env python
"""Run the pipeline on real TESS observations with known dispositions.

This is the end-to-end reality check: everything else in the project is
validated against signals we injected ourselves, which cannot expose a wrong
assumption shared by the simulator and the pipeline.  These are real photons.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet import classify                  # noqa: E402
from exodet.pipeline import analyze          # noqa: E402

# published values, for comparison only -- never fed to the pipeline
PUBLISHED = {
    "TIC 261136679": dict(name="Pi Men c",   period=6.2679, rp_rs=0.01703,
                          note="shallow ~300 ppm, quiet host"),
    "TIC 22529346":  dict(name="WASP-121 b", period=1.27493, rp_rs=0.12355,
                          note="deep hot Jupiter"),
    "TIC 441420236": dict(name="AU Mic b",   period=8.46321, rp_rs=0.0514,
                          note="young, heavily spotted host"),
    "TIC 100100827": dict(name="WASP-18 b",  period=0.94145, rp_rs=0.09716,
                          note="very deep, short period"),
}


def harmonic_ratio(rec, pub):
    r = rec / pub
    for mult in (1.0, 2.0, 0.5, 3.0, 1 / 3):
        if abs(r / mult - 1) < 0.02:
            return mult
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="models/classifier.joblib")
    ap.add_argument("--no-mcmc", action="store_true")
    ap.add_argument("--detrend", nargs="+", default=["biweight", "lowess"],
                    help="wotan filters to try; the best SDE wins")
    args = ap.parse_args()

    try:
        model, calibrator = classify.load(args.model)
        print(f"loaded classifier from {args.model}\n")
    except Exception as exc:
        model = calibrator = None
        print(f"[warn] no classifier ({exc}); detection + fitting only\n")

    files = sorted(glob.glob("data/cache/TIC_*.npz"))
    if not files:
        print("no cached light curves; run scripts/fetch_real.py first")
        return

    for path in files:
        d = np.load(path, allow_pickle=True)
        tic = str(d["tic"])
        pub = PUBLISHED.get(tic, {})
        name = pub.get("name", tic)
        print("=" * 70)
        print(f"{name}  ({tic})   {pub.get('note', '')}")
        print("=" * 70)

        res = analyze(d["time"], d["flux"], d["flux_err"], model, calibrator,
                      run_mcmc=not args.no_mcmc,
                      detrend_methods=args.detrend)
        for line in res.summary_lines():
            print("  " + line)
        print(f"  Detrender used : {res.detrend_method}")

        if res.detected and pub:
            mult = harmonic_ratio(res.detection.period, pub["period"])
            tag = ("exact" if mult == 1.0 else
                   f"{mult:g}x harmonic" if mult else "MISMATCH")
            print(f"\n  published period : {pub['period']:.5f} d   -> {tag}")
            if res.fit is not None and res.fit.converged:
                print(f"  published Rp/R*  : {pub['rp_rs']:.5f}   "
                      f"fitted {res.fit.rp:.5f} +/- {res.fit.rp_err:.5f}"
                      f"   ({(res.fit.rp/pub['rp_rs']-1)*100:+.1f} %)")
        print()


if __name__ == "__main__":
    main()
