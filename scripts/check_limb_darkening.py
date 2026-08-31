#!/usr/bin/env python
"""Did freeing u2 actually widen the depth uncertainties, or just move them?

Quoted depth errors measured 1.5x too narrow with u2 held fixed. The claim is
that sampling it recovers the missing width. That claim is only worth anything
if it is measured against the *actual* previous code rather than a remembered
number, so the pre-fix `fit.py` is loaded straight out of git and run on the
same cached photons in the same process. Nothing differs between the two arms
except the limb-darkening treatment.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.fit import fit_transit                     # noqa: E402
from exodet.pipeline import analyze                    # noqa: E402

TARGETS = {
    "TIC 261136679": dict(name="Pi Men c", depth_ppm=290.0),
    "TIC 22529346":  dict(name="WASP-121 b", depth_ppm=15300.0),
}


def load_legacy_fit(ref: str = "HEAD"):
    """Import the committed pre-fix fit.py as a standalone module.

    It uses package-relative imports, which will not resolve outside the
    package, so they are rewritten to absolute ones. The rest is byte-identical
    to what shipped -- this is the real old code, not a reconstruction.
    """
    src = subprocess.run(["git", "show", f"{ref}:src/exodet/fit.py"],
                         capture_output=True, text=True, check=True).stdout
    src = src.replace("from .simulate import", "from exodet.simulate import")
    mod = types.ModuleType("fit_legacy")
    mod.__dict__["__file__"] = "<git:fit.py>"
    # @dataclass resolves annotations via sys.modules[cls.__module__], so the
    # module must be registered before the class body executes, not after.
    sys.modules["fit_legacy"] = mod
    exec(compile(src, "<git:fit.py>", "exec"), mod.__dict__)
    return mod


def run(target: str, legacy_mod, seed: int = 0):
    path = os.path.join("data", "cache", target.replace(" ", "_") + ".npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    t, f = d["time"], d["flux"]
    e = d["flux_err"] if "flux_err" in d and d["flux_err"].size else None

    # Detrend/search once and hand the SAME detection to both fitters, so a
    # difference in the fit cannot be a difference in what was fitted.
    res = analyze(t, f, e, model=None, do_fit=False)
    if res.detection is None or not res.detection.detected:
        return None

    out = {}
    for tag, fn in (("before (u2 fixed)", legacy_mod.fit_transit),
                    ("after  (u2 sampled)", fit_transit)):
        # Result keeps the detrended arrays for plotting but not the errors;
        # fit_transit falls back to the flux scatter when passed None, which is
        # what the pipeline itself does here.
        fr = fn(res.time, res.flux, None, res.detection,
                run_mcmc=True, seed=seed)
        out[tag] = fr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD", help="git ref holding the old fit.py")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    legacy = load_legacy_fit(args.ref)
    print(f"legacy fit.py loaded from {args.ref}: "
          f"params {legacy.PARAM_NAMES}")
    print(f"current fit.py params        : "
          f"{__import__('exodet.fit', fromlist=['x']).PARAM_NAMES}\n")

    for target, meta in TARGETS.items():
        got = run(target, legacy, args.seed)
        if got is None:
            print(f"{meta['name']}: no cached data / no detection -- skipped")
            continue
        print(f"=== {meta['name']} ({target}) ===")
        print(f"  {'arm':<22}{'depth_obs ppm':>16}{'+/- ppm':>10}"
              f"{'rel err':>10}{'chi2_red':>10}")
        widths = {}
        for tag, fr in got.items():
            dppm, eppm = fr.depth_obs * 1e6, fr.depth_obs_err * 1e6
            widths[tag] = eppm
            rel = eppm / dppm if dppm else float("nan")
            print(f"  {tag:<22}{dppm:>16.1f}{eppm:>10.1f}{rel:>10.4f}"
                  f"{fr.chi2_red:>10.2f}")
        a, b = list(widths.values())
        if a and np.isfinite(a) and np.isfinite(b):
            print(f"  -> uncertainty width ratio after/before = {b / a:.2f}x")
        fr_new = got["after  (u2 sampled)"]
        print(f"  -> fitted limb darkening u1={fr_new.u1:.3f} u2={fr_new.u2:.3f}"
              f"  (sum {fr_new.u1 + fr_new.u2:.3f})\n")


if __name__ == "__main__":
    main()
