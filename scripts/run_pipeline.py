#!/usr/bin/env python
"""Batch-run the pipeline over many light curves and emit a candidate catalogue.

Sources:
  --fits DIR        every *.fits under DIR (TESS SPOC light curve files)
  --tic-list FILE   one TIC ID per line, downloaded from MAST via lightkurve
  --simulate N      N synthetic light curves per class (for validation)

Output is a CSV, one row per light curve, ranked by detection significance.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet import classify                          # noqa: E402
from exodet.config import CLASSES                    # noqa: E402
from exodet.pipeline import analyze                  # noqa: E402
from exodet.simulate import generate_sample          # noqa: E402

_MODEL = None


def _get_model(path):
    """Load the classifier once per worker process."""
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = classify.load(path)
        except Exception:
            _MODEL = (None, None)
    return _MODEL


def load_fits(path):
    """Read a TESS SPOC light curve file; prefer PDCSAP over SAP flux."""
    import lightkurve as lk
    lc = lk.read(path).remove_nans()
    return (np.asarray(lc.time.value, dtype=float),
            np.asarray(lc.flux.value, dtype=float),
            np.asarray(lc.flux_err.value, dtype=float))


def process(task):
    kind, ident, model_path, run_mcmc = task
    model, calibrator = _get_model(model_path)
    try:
        if kind == "fits":
            t, f, e = load_fits(ident)
            name = os.path.basename(ident)
        elif kind == "tic":
            import lightkurve as lk
            q = lk.search_lightcurve(ident, mission="TESS")
            if len(q) == 0:
                return None
            lc = q[0].download().remove_nans()
            t = np.asarray(lc.time.value, dtype=float)
            f = np.asarray(lc.flux.value, dtype=float)
            e = np.asarray(lc.flux_err.value, dtype=float)
            name = ident
        else:  # simulated
            label, seed = ident
            s = generate_sample(label, np.random.default_rng(seed))
            t, f, e = s["time"], s["flux"], s["flux_err"]
            name = f"sim_{label}_{seed}"

        res = analyze(t, f, e, model, calibrator, run_mcmc=run_mcmc)
        row = {"target": name, "detected": res.detected}
        if kind == "sim":
            row["true_label"] = ident[0]
        if res.detection is not None:
            d = res.detection
            row.update(bls_period=d.period, bls_depth_ppm=d.depth * 1e6,
                       bls_duration_hr=d.duration * 24, sde=d.sde, snr=d.snr,
                       n_transits=d.n_transits)
        if res.label:
            row.update(label=res.label, confidence=res.confidence)
            row.update({f"p_{c}": res.probabilities[c] for c in CLASSES})
        if res.fit is not None and res.fit.converged:
            fr = res.fit
            row.update(fit_period=fr.period, fit_period_err=fr.period_err,
                       fit_depth_ppm=fr.depth * 1e6,
                       fit_depth_err_ppm=fr.depth_err * 1e6,
                       fit_duration_hr=fr.duration * 24,
                       fit_duration_err_hr=fr.duration_err * 24,
                       fit_rp_rs=fr.rp, chi2_red=fr.chi2_red)
        return row
    except Exception as exc:
        return {"target": str(ident), "detected": False, "error": str(exc)}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fits", help="directory of TESS FITS light curves")
    src.add_argument("--tic-list", help="file with one TIC ID per line")
    src.add_argument("--simulate", type=int, help="N synthetic curves per class")
    ap.add_argument("-o", "--out", default="reports/catalog.csv")
    ap.add_argument("-m", "--model", default="models/classifier.joblib")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--mcmc", action="store_true",
                    help="run MCMC uncertainties (much slower)")
    args = ap.parse_args()

    if args.fits:
        files = sorted(glob.glob(os.path.join(args.fits, "**", "*.fits"),
                                 recursive=True))
        tasks = [("fits", f, args.model, args.mcmc) for f in files]
    elif args.tic_list:
        with open(args.tic_list) as fh:
            ids = [ln.strip() for ln in fh if ln.strip()]
        tasks = [("tic", i, args.model, args.mcmc) for i in ids]
    else:
        tasks = [("sim", (c, 90_000 + i * 7 + k), args.model, args.mcmc)
                 for k, c in enumerate(CLASSES) for i in range(args.simulate)]

    print(f"processing {len(tasks)} light curves on {args.jobs} workers")
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for r in tqdm(ex.map(process, tasks, chunksize=2), total=len(tasks)):
            if r:
                rows.append(r)

    df = pd.DataFrame(rows)
    if "sde" in df:
        df = df.sort_values("sde", ascending=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nwrote {len(df)} rows -> {args.out}")
    print(f"detected: {int(df['detected'].sum())} / {len(df)}")
    if "label" in df:
        print("\nclassification counts:")
        print(df[df["detected"]]["label"].value_counts().to_string())
    if "true_label" in df and "label" in df:
        ok = df.dropna(subset=["label", "true_label"])
        if len(ok):
            acc = (ok["label"] == ok["true_label"]).mean()
            print(f"\naccuracy on detected simulated curves: {acc:.3f}")


if __name__ == "__main__":
    main()
