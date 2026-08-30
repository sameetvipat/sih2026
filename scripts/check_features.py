#!/usr/bin/env python
"""Spot-check the two features most likely to misbehave on real backgrounds.

Both were designed and tuned against fully synthetic red noise:

  harmonic_test / sec_ratio_2p  BLS locks onto half an eclipsing binary's true
      period, so folding at 2P should show two EQUAL events for a planet and a
      shallower secondary for an EB. Real stellar variability has structure the
      synthetic noise model never produced, and could plausibly fake a harmonic
      signature.

  implied_density  the stellar density implied by period, duration and depth.
      It exists to catch physically impossible geometries -- wrong harmonics and
      systematics. On messier real backgrounds the duration estimate degrades,
      and this feature is computed FROM the duration, so it degrades with it.

Physical reference points, not opinions:
  * a real main-sequence star has mean density ~0.1-10 g/cm^3 (the Sun is 1.41);
    values far outside that band mean the geometry is not a transit
  * a genuine planet folded at 2P shows two equal events, so sec_ratio_2p_dev
    (|ratio - 1|) sits near 0; an EB caught at half period shows a shallower
    secondary and a clearly non-zero deviation
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CLASSES = ["transit", "eclipse", "blend", "variable"]
COLOR = {"transit": "#2E86DE", "eclipse": "#EE5A24",
         "blend": "#F79F1F", "variable": "#A55EEA"}

# main-sequence stars live here; outside is unphysical for a transit geometry
RHO_LO, RHO_HI = 0.1, 10.0


def load(path):
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path)
    if "error" in d:
        d = d[d["error"].isna()]
    return d[d["label"].isin(CLASSES)]


def summarise(name, df):
    print(f"\n{'='*70}\n{name}  (n={len(df)})\n{'='*70}")

    if "log_rho_implied" in df:
        rho = 10 ** df["log_rho_implied"]
        frac = float(((rho >= RHO_LO) & (rho <= RHO_HI)).mean())
        print(f"implied stellar density -- physical band {RHO_LO}-{RHO_HI} g/cm^3")
        print(f"  in band: {frac:.1%}")
        print(f"  median by class (g/cm^3):")
        for c in CLASSES:
            sub = rho[df["label"] == c]
            if len(sub):
                inb = float(((sub >= RHO_LO) & (sub <= RHO_HI)).mean())
                print(f"    {c:<9} {sub.median():8.2f}   in band {inb:5.1%}  (n={len(sub)})")

    if "sec_ratio_2p_dev" in df:
        print(f"\n2P harmonic test -- |depth ratio - 1|, near 0 means "
              f"two equal events (planet-like)")
        for c in CLASSES:
            sub = df.loc[df["label"] == c, "sec_ratio_2p_dev"]
            if len(sub):
                print(f"    {c:<9} median {sub.median():6.3f}   "
                      f"IQR [{sub.quantile(.25):.3f}, {sub.quantile(.75):.3f}]")
        t = df.loc[df["label"] == "transit", "sec_ratio_2p_dev"]
        e = df.loc[df["label"] == "eclipse", "sec_ratio_2p_dev"]
        if len(t) > 3 and len(e) > 3:
            sep = (e.median() - t.median())
            print(f"    eclipse minus transit separation: {sep:+.3f}  "
                  f"({'discriminating' if sep > 0.05 else 'NOT discriminating'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--injected", default="data/processed/train_injected.parquet")
    ap.add_argument("--synthetic", default="data/processed/train.parquet")
    ap.add_argument("--real", default="data/processed/real.parquet")
    ap.add_argument("-o", "--out", default="reports/feature_check.png")
    args = ap.parse_args()

    sets = [(n, load(p)) for n, p in
            [("PURE SYNTHETIC", args.synthetic),
             ("REAL-BACKGROUND INJECTED", args.injected),
             ("REAL CATALOGUED", args.real)]]
    sets = [(n, d) for n, d in sets if d is not None and len(d)]
    for n, d in sets:
        summarise(n, d)

    fig, axes = plt.subplots(2, len(sets), figsize=(5 * len(sets), 8))
    if len(sets) == 1:
        axes = axes.reshape(2, 1)
    for j, (name, df) in enumerate(sets):
        ax = axes[0, j]
        for c in CLASSES:
            sub = df.loc[df["label"] == c, "log_rho_implied"].dropna()
            if len(sub) > 3:
                ax.hist(sub, bins=25, alpha=0.5, label=c, color=COLOR[c])
        ax.axvspan(np.log10(RHO_LO), np.log10(RHO_HI), color="green", alpha=0.08)
        ax.axvline(np.log10(1.41), color="k", ls="--", lw=1)
        ax.set_title(f"{name}\nimplied density", fontsize=10)
        ax.set_xlabel("log10 rho (g/cm^3)  — shaded = physical, dashed = solar")
        if j == 0:
            ax.legend(fontsize=7)

        ax = axes[1, j]
        for c in CLASSES:
            sub = df.loc[df["label"] == c, "sec_ratio_2p_dev"].dropna()
            if len(sub) > 3:
                ax.hist(np.clip(sub, 0, 2), bins=25, alpha=0.5, label=c,
                        color=COLOR[c])
        ax.set_xlabel("|2P depth ratio - 1|   (0 = two equal events)")
        ax.set_title("2P harmonic test", fontsize=10)
    fig.suptitle("Feature validity across data regimes "
                 "(sanity check, not a performance result)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nplot -> {args.out}")


if __name__ == "__main__":
    main()
