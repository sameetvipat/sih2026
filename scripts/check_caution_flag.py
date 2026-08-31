#!/usr/bin/env python
"""Does the classifier/fit cross-check actually fire on the case it exists for?

AU Mic b is the reason this check exists: a young star whose starspot amplitude
is ~19x its transit depth, which the classifier confidently calls "eclipse"
while the fit reports reduced chi-square ~17. Asserting the logic by reading it
is not enough -- the failure mode being guarded against is precisely a
threshold or field-name mismatch that looks correct in source.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet import classify                        # noqa: E402
from exodet.pipeline import analyze                # noqa: E402

CASES = {
    "TIC 441420236": dict(name="AU Mic b", expect_caution=True,
                          why="spot amplitude 19x the transit depth"),
    "TIC 261136679": dict(name="Pi Men c", expect_caution=False,
                          why="quiet host, well-described by the model"),
    "TIC 22529346":  dict(name="WASP-121 b", expect_caution=False,
                          why="deep, high-S/N, clean fit"),
}


def main():
    model, calibrator = None, None
    try:
        model, calibrator = classify.load()
    except Exception as exc:
        print(f"[warn] no classifier loaded ({exc}); labels will be absent "
              f"and the cross-check cannot fire")

    failures = []
    for target, meta in CASES.items():
        path = os.path.join("data", "cache", target.replace(" ", "_") + ".npz")
        if not os.path.exists(path):
            print(f"{meta['name']:<12} SKIP  (no cached light curve)")
            continue
        d = np.load(path, allow_pickle=True)
        e = d["flux_err"] if "flux_err" in d and d["flux_err"].size else None
        # Multi-detrend, matching validate_real.py and the production batch
        # path. It matters here: AU Mic b reaches SDE 4.6 under biweight alone
        # (below the threshold of 7, so no detection and nothing to cross-check)
        # and SDE 11.0 under lowess. The target only exists as a classification
        # failure *because* the fallback detrender promoted it -- which is the
        # multi-detrend significance inflation this project documents
        # separately, showing up here as a concrete consequence.
        res = analyze(d["time"], d["flux"], e, model, calibrator,
                      do_fit=True, run_mcmc=False,
                      detrend_methods=("biweight", "lowess"))

        chi2 = (f"{res.fit.chi2_red:.1f}" if res.fit is not None
                and np.isfinite(res.fit.chi2_red) else "n/a")
        ok = res.caution_flag == meta["expect_caution"]
        detr = res.detrend_method
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures.append(meta["name"])
        print(f"{meta['name']:<12} {status}  label={str(res.label):<9} "
              f"chi2_red={chi2:<7} detrend={detr:<9} "
              f"caution={res.caution_flag}  "
              f"(expected {meta['expect_caution']}: {meta['why']})")
        if res.caution_reason:
            print(f"             reason: {res.caution_reason}")

    if failures:
        print(f"\nFAILED on: {', '.join(failures)}")
        return 1
    print("\nall cross-check cases behaved as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
