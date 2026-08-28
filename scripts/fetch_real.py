#!/usr/bin/env python
"""Download a few TESS light curves with known dispositions, for validation.

These are real observations of objects whose nature is already established, so
they act as an end-to-end check that the pipeline works outside simulation.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# target -> (TIC, known disposition, published period in days)
TARGETS = {
    "Pi Men c":   ("TIC 261136679", "transit", 6.2679),   # TESS first-light planet, ~300 ppm
    "WASP-18 b":  ("TIC 100100827", "transit", 0.9415),   # deep hot Jupiter
    "WASP-121 b": ("TIC 22529346",  "transit", 1.2749),   # deep hot Jupiter
    "AU Mic b":   ("TIC 441420236", "transit", 8.4630),   # young, very spotty host
}

OUT = "data/cache"


def main():
    import lightkurve as lk
    os.makedirs(OUT, exist_ok=True)
    for name, (tic, disp, period) in TARGETS.items():
        path = os.path.join(OUT, f"{tic.replace(' ', '_')}.npz")
        if os.path.exists(path):
            print(f"[skip] {name} already cached")
            continue
        try:
            print(f"[get ] {name} ({tic}) ...", flush=True)
            q = lk.search_lightcurve(tic, mission="TESS", author="SPOC",
                                     exptime=120)
            if len(q) == 0:
                print(f"[miss] no SPOC 2-min data for {name}")
                continue
            lc = q[0].download().remove_nans()
            np.savez_compressed(
                path,
                time=np.asarray(lc.time.value, dtype=float),
                flux=np.asarray(lc.flux.value, dtype=float),
                flux_err=np.asarray(lc.flux_err.value, dtype=float),
                name=name, tic=tic, disposition=disp, period=period,
                sector=str(getattr(lc.meta, "get", lambda *_: "")("SECTOR", "")),
            )
            print(f"[ok  ] {name}: {len(lc)} points -> {path}")
        except Exception as exc:
            print(f"[fail] {name}: {exc}")


if __name__ == "__main__":
    main()
