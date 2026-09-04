#!/usr/bin/env python
"""Injection-recovery analysis: how accurate are the recovered parameters?

Because every synthetic light curve has known injected parameters, we can
measure -- not assume -- the pipeline's accuracy:

  * detection completeness as a function of transit depth and S/N,
  * bias and scatter in recovered period, depth and duration,
  * whether the quoted uncertainties are honest, via the pull distribution
    (fitted - true) / sigma, which should be a unit Gaussian.

The last one is what justifies the error bars in the report.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.config import TRANSIT                    # noqa: E402
from exodet.fit import fit_transit                   # noqa: E402
from exodet.preprocess import prepare                # noqa: E402
from exodet.search import run_bls                    # noqa: E402
from exodet.simulate import generate_sample          # noqa: E402


def harmonic_err(recovered, true):
    """Fractional period error, tolerating the 2x / 0.5x harmonic ambiguity."""
    r = recovered / true
    return min(abs(r - 1), abs(r - 2) / 2, abs(r - 0.5) * 2)


def recover_one(seed: int) -> dict | None:
    """Inject a transit, run detection + fitting, compare against truth."""
    rng = np.random.default_rng(seed)
    try:
        s = generate_sample(TRANSIT, rng)
        t, f, e, _ = prepare(s["time"], s["flux"], s["flux_err"])
        det, _, _ = run_bls(t, f, e)
        tr = s["truth"]
        row = dict(seed=seed, detected=det.detected,
                   true_period=tr["period"], true_depth=tr["depth"],
                   true_duration=tr["duration"], white_ppm=s["white_ppm"],
                   sde=det.sde, snr=det.snr,
                   bls_period=det.period, bls_depth=det.depth)
        if not det.detected:
            return row
        # only fit when BLS found the right period -- otherwise we would be
        # measuring the search failure, not the fit quality
        row["period_ok"] = harmonic_err(det.period, tr["period"]) < 0.01
        if row["period_ok"]:
            fr = fit_transit(t, f, e, det, run_mcmc=True, n_steps=1200, n_burn=400)
            if fr.converged:
                row.update(fit_period=fr.period, fit_period_err=fr.period_err,
                           fit_depth=fr.depth, fit_depth_err=fr.depth_err,
                           fit_duration=fr.duration,
                           fit_duration_err=fr.duration_err,
                           chi2_red=fr.chi2_red)
        return row
    except Exception:
        return None


def pull_stats(df, param):
    """Pull = (fitted - true) / sigma. Honest errors give mean 0, stdev 1."""
    col, err, true = f"fit_{param}", f"fit_{param}_err", f"true_{param}"
    m = df[[col, err, true]].dropna()
    m = m[m[err] > 0]
    if len(m) < 5:
        return None
    pull = (m[col] - m[true]) / m[err]
    pull = pull[np.abs(pull) < 50]                 # drop catastrophic outliers
    return dict(n=len(pull), mean=float(pull.mean()), std=float(pull.std()),
                frac_within_1sig=float((np.abs(pull) < 1).mean()),
                frac_within_2sig=float((np.abs(pull) < 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", type=int, default=200)
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument("-o", "--outdir", default="reports")
    ap.add_argument("--seed", type=int, default=50_000)
    args = ap.parse_args()

    seeds = [args.seed + i for i in range(args.n)]
    print(f"injection-recovery on {len(seeds)} transits, {args.jobs} workers")
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for r in tqdm(ex.map(recover_one, seeds, chunksize=2), total=len(seeds)):
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "injection_recovery.csv"), index=False)

    print("\n" + "=" * 64)
    print(f"detection completeness      : {df['detected'].mean():.3f}")
    if "period_ok" in df:
        det = df[df["detected"]]
        print(f"correct period | detected   : {det['period_ok'].mean():.3f}")

    good = df[df.get("fit_depth").notna()] if "fit_depth" in df else pd.DataFrame()
    if len(good):
        for p, scale, unit in (("period", 1.0, "d"),
                               ("depth", 1e6, "ppm"),
                               ("duration", 24.0, "h")):
            resid = (good[f"fit_{p}"] - good[f"true_{p}"]) * scale
            frac = (good[f"fit_{p}"] / good[f"true_{p}"] - 1) * 100
            print(f"\n{p}:")
            print(f"  median bias   : {np.median(resid):+.4g} {unit} "
                  f"({np.median(frac):+.2f} %)")
            print(f"  scatter (MAD) : {1.4826*np.median(np.abs(resid-np.median(resid))):.4g} {unit}")
            st = pull_stats(good, p)
            if st:
                print(f"  pull mean/std : {st['mean']:+.2f} / {st['std']:.2f} "
                      f"(n={st['n']})")
                print(f"  within 1s/2s  : {st['frac_within_1sig']:.2f} / "
                      f"{st['frac_within_2sig']:.2f}   (ideal 0.68 / 0.95)")

    # --- plots --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    d = df.dropna(subset=["true_depth"])
    bins = np.logspace(np.log10(max(d["true_depth"].min(), 1e-5)),
                       np.log10(d["true_depth"].max()), 9)
    idx = np.digitize(d["true_depth"], bins) - 1
    xs, ys = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= 3:
            xs.append(np.sqrt(bins[b] * bins[b + 1]) * 1e6)
            ys.append(d["detected"].values[m].mean())
    ax.plot(xs, ys, "o-", color="#2E86DE")
    ax.set_xscale("log"); ax.set_xlabel("injected depth (ppm)")
    ax.set_ylabel("detection fraction"); ax.set_ylim(0, 1.05)
    ax.set_title("Completeness vs depth"); ax.grid(alpha=0.3)

    if len(good):
        ax = axes[1]
        ax.errorbar(good["true_depth"] * 1e6, good["fit_depth"] * 1e6,
                    yerr=good["fit_depth_err"] * 1e6, fmt="o", ms=3,
                    alpha=0.6, color="#2E86DE", elinewidth=0.7)
        lim = [good["true_depth"].min() * 1e6 * 0.7, good["true_depth"].max() * 1e6 * 1.3]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("injected depth (ppm)"); ax.set_ylabel("recovered depth (ppm)")
        ax.set_title("Depth recovery"); ax.grid(alpha=0.3)

        ax = axes[2]
        st = pull_stats(good, "depth")
        pull = ((good["fit_depth"] - good["true_depth"]) / good["fit_depth_err"])
        pull = pull[np.abs(pull) < 8]
        ax.hist(pull, bins=25, density=True, color="#2E86DE", alpha=0.75)
        x = np.linspace(-5, 5, 200)
        ax.plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), "k--", lw=1.5,
                label="unit Gaussian")
        ax.set_xlabel("(fitted - true) / sigma"); ax.set_ylabel("density")
        ax.set_title(f"Depth pull  (std = {st['std']:.2f})" if st else "Depth pull")
        ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(args.outdir, "injection_recovery.png")
    fig.savefig(out, dpi=130)
    print(f"\nplots -> {out}")


if __name__ == "__main__":
    main()
