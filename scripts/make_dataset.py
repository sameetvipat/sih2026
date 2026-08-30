#!/usr/bin/env python
"""Build a labelled feature table by simulating light curves and running the
full detection pipeline over them.

Every row is one light curve that has been simulated -> detrended -> searched
with BLS -> reduced to vetting features, alongside its true label and the
injected ground-truth parameters (for injection-recovery analysis).

Results are flushed to shard files as they complete, and re-running skips work
already on disk.  A run of several thousand light curves takes the better part
of an hour, and writing only at the end means an interruption costs all of it.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.config import (BLEND, CLASSES, ECLIPSE, NOISE,   # noqa: E402
                            TRANSIT, VARIABLE)
from exodet.pipeline import analyze                  # noqa: E402
from exodet.simulate import generate_sample, inject_into_real   # noqa: E402

# Classes that can be created by injecting a known signal. VARIABLE and NOISE
# are deliberately not here -- see `_load_baseline` and the module docstring.
INJECTABLE = (TRANSIT, ECLIPSE, BLEND)

_BASELINES: pd.DataFrame | None = None


def _baseline_pool(manifest: str) -> pd.DataFrame:
    """Load the quiet-baseline manifest once per worker process."""
    global _BASELINES
    if _BASELINES is None:
        _BASELINES = pd.read_csv(manifest)
        if "split" not in _BASELINES:
            _BASELINES["split"] = "train"
    return _BASELINES


# Trim baselines to the length of the real evaluation set. Two reasons: a
# 44-day window makes the BLS period grid much denser than the 33.5-day Kepler
# Q1 curves we test against (25 s per example vs ~11 s), and training on a
# longer baseline than we evaluate on is itself a domain mismatch -- transit
# counts and SDE both scale with observing span.
BASELINE_MAX_DAYS = 33.5


def _load_baseline(manifest: str, seed: int, split: str | None):
    """Draw a raw baseline, stratified across brightness bins.

    Stratification matters: always drawing from the brightest, cleanest stars
    would reintroduce an unrealistically easy noise floor from a different
    angle, which is the exact failure this pipeline exists to fix.
    """
    pool = _baseline_pool(manifest)
    if split:
        pool = pool[pool["split"] == split]
    if pool.empty:
        return None
    rng = np.random.default_rng(seed)
    # pick a brightness stratum first, then a star within it, so faint stars
    # are not swamped by however many bright ones happen to be in the bank
    bins = sorted(pool["brightness_bin"].dropna().unique())
    if bins:
        b = bins[rng.integers(0, len(bins))]
        sub = pool[pool["brightness_bin"] == b]
        pool = sub if len(sub) else pool
    row = pool.iloc[int(rng.integers(0, len(pool)))]
    d = np.load(row["path"], allow_pickle=True)
    t = np.asarray(d["time"], float)
    f = np.asarray(d["flux"], float)
    err = (np.asarray(d["flux_err"], float)
           if "flux_err" in d and d["flux_err"].size else None)
    if t.size and (t.max() - t.min()) > BASELINE_MAX_DAYS:
        keep = t <= (t.min() + BASELINE_MAX_DAYS)
        t, f = t[keep], f[keep]
        err = err[keep] if err is not None else None
    return t, f, err, str(row["target"])

FLUSH_EVERY = 100
# Flush on elapsed time as well as row count. Count alone is not enough: at
# ~13 s per injected example, a 100-row threshold means ~20 minutes of
# completed work sits unwritten, and a stop discards all of it (measured: an
# interrupted injection run banked 0 rows while the download job, which already
# had time-based flushing, preserved 368).
FLUSH_SECONDS = 60.0


def one_sample(args) -> dict | None:
    """Build + process a single light curve. Returns a feature row."""
    label, seed, cadence, days, source, manifest, split = args
    rng = np.random.default_rng(seed)
    baseline_target = None
    try:
        if source == "real_injected":
            got = _load_baseline(manifest, seed, split)
            if got is None:
                return None
            bt, bf, be, baseline_target = got
            if label in INJECTABLE:
                s = inject_into_real(bt, bf, be, label, rng)
            else:
                # VARIABLE and NOISE are not injected. A real quiet star with
                # nothing added is already a genuine NOISE example whenever the
                # full pipeline false-alarms on it; injecting a fake wobble
                # would just re-import the synthetic realism problem.
                s = dict(time=bt, flux=bf, flux_err=be, label=label,
                         white_ppm=np.nan, red_ppm=np.nan, truth={})
        else:
            s = generate_sample(label, rng, cadence_min=cadence, n_days=days)
        # Go through the same entry point inference uses, so training features
        # and serving features come from identical preprocessing.  (The
        # two-pass masked detrend recovers ~20% of transit depth, so computing
        # training features without it would skew every depth-based feature.)
        res = analyze(s["time"], s["flux"], s["flux_err"],
                      model=None, do_fit=False)
        if res.features is None:
            return None
        det = res.detection
        row = dict(res.features)
        row["label"] = label
        row["seed"] = seed
        row["detected"] = det.detected
        # ground truth, for injection-recovery scoring
        tr = s["truth"]
        row["true_period"] = tr.get("period", np.nan)
        row["true_depth"] = tr.get("depth", np.nan)
        row["true_duration"] = tr.get("duration", np.nan)
        row["bls_period"] = det.period
        row["bls_depth"] = det.depth
        row["bls_duration"] = det.duration
        row["white_ppm"] = s["white_ppm"]
        row["cadence_min"] = cadence
        row["source"] = source
        row["baseline_target"] = baseline_target
        # `target` lets downstream leakage checks verify that a
        # training row and an evaluation row are not the same star
        row["target"] = baseline_target
        return row
    except Exception as exc:                      # keep the batch alive
        print(f"[warn] seed={seed} label={label}: {exc}", file=sys.stderr)
        return None


def load_shards(shard_dir: str) -> pd.DataFrame:
    """Concatenate whatever has already been computed."""
    files = sorted(glob.glob(os.path.join(shard_dir, "part_*.parquet")))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as exc:                  # a shard truncated mid-write
            print(f"[warn] unreadable shard {f}: {exc}", file=sys.stderr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-per-class", type=int, default=250)
    ap.add_argument("-o", "--out", default="data/processed/train.parquet")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--source", choices=["synthetic", "real_injected"],
                    default="real_injected",
                    help="real_injected puts known signals on real quiet stars; "
                         "synthetic uses the purely simulated noise model, which "
                         "is retained for validating the fitting code, not for "
                         "training the classifier")
    ap.add_argument("--manifest", default="data/baselines/manifest.csv")
    ap.add_argument("--split", default=None,
                    help="restrict to baselines in this split (train/test), so "
                         "the same star's noise floor never spans both")
    ap.add_argument("--cadence", type=float, default=2.0,
                    help="sampling cadence in minutes (2 = TESS, 29 = Kepler long)")
    ap.add_argument("--days", type=float, default=27.4,
                    help="baseline in days (27.4 = TESS sector, 33.5 = Kepler Q1)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing shards and start over")
    ap.add_argument("--merge-only", action="store_true",
                    help="just merge existing shards into the output file")
    args = ap.parse_args()

    shard_dir = os.path.join(os.path.dirname(args.out) or ".", "shards")
    os.makedirs(shard_dir, exist_ok=True)

    if args.fresh:
        for f in glob.glob(os.path.join(shard_dir, "part_*.parquet")):
            os.remove(f)

    done = load_shards(shard_dir)
    if args.merge_only:
        if done.empty:
            print("no shards to merge")
            return
        finish(done, args.out)
        return

    # Only curves that pass the detection threshold produce a feature row --
    # the classifier never sees anything else.  `noise` and `variable` are
    # detected far less often (they mostly aren't periodic dips), so oversample
    # them at generation time to keep the surviving rows roughly balanced.
    yield_factor = {NOISE: 3, VARIABLE: 2}
    if args.source == "real_injected" and not os.path.exists(args.manifest):
        sys.exit(
            f"no baseline manifest at {args.manifest}.\n"
            "real_injected mode needs a bank of real quiet stars to inject into:\n"
            "  python scripts/build_baseline_bank.py --mission kepler\n"
            "  python scripts/split_baselines.py\n"
            "Or pass --source synthetic to use the simulated noise model "
            "(valid for fitting validation, not for training the classifier).")

    classes = CLASSES
    if args.source == "real_injected":
        # VARIABLE comes from the catalogued real variables in real.parquet --
        # a real star with real rotation/pulsation structure, which is exactly
        # what the synthetic generator fails to reproduce. Merge it at train
        # time rather than faking it here.
        classes = [c for c in CLASSES if c != VARIABLE]
        print("[note] VARIABLE is not generated in real_injected mode; "
              "merge those rows from data/processed/real.parquet at train time")

    tasks = [(label, args.seed + i * 1000 + k, args.cadence, args.days,
              args.source, args.manifest, args.split)
             for k, label in enumerate(classes)
             for i in range(args.n_per_class * yield_factor.get(label, 1))]

    if not done.empty:
        seen = set(zip(done["label"], done["seed"]))
        if "cadence_min" in done and (done["cadence_min"] != args.cadence).any():
            print("[warn] shards contain a different cadence; use --fresh",
                  file=sys.stderr)
        before = len(tasks)
        tasks = [t for t in tasks if t not in seen]
        print(f"resuming: {len(done)} rows already on disk, "
              f"{before - len(tasks)} tasks skipped")

    if not tasks:
        print("nothing left to compute")
        finish(done, args.out)
        return

    print(f"simulating {len(tasks)} light curves on {args.jobs} workers "
          f"(flushing every {FLUSH_EVERY} rows)")

    buf: list[dict] = []
    shard_i = len(glob.glob(os.path.join(shard_dir, "part_*.parquet")))
    last_flush = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for row in tqdm(ex.map(one_sample, tasks, chunksize=4),
                            total=len(tasks)):
                if row is not None:
                    buf.append(row)
                due = (len(buf) >= FLUSH_EVERY
                       or (buf and time.monotonic() - last_flush >= FLUSH_SECONDS))
                if due:
                    shard_i = flush(buf, shard_dir, shard_i)
                    last_flush = time.monotonic()
    except KeyboardInterrupt:
        print("\ninterrupted -- flushing what is complete", file=sys.stderr)
    finally:
        if buf:
            flush(buf, shard_dir, shard_i)

    finish(load_shards(shard_dir), args.out)


def flush(buf: list[dict], shard_dir: str, i: int) -> int:
    """Write the buffer to a new shard and clear it. Returns the next index."""
    path = os.path.join(shard_dir, f"part_{i:04d}.parquet")
    tmp = path + ".tmp"
    pd.DataFrame(buf).to_parquet(tmp, index=False)
    os.replace(tmp, path)          # atomic, so a kill cannot leave a half file
    buf.clear()
    return i + 1


def finish(df: pd.DataFrame, out: str):
    if df.empty:
        print("no rows produced")
        return
    df = df.drop_duplicates(subset=["label", "seed"], keep="first")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nwrote {len(df)} rows -> {out}")
    print(df["label"].value_counts().to_string())
    print("\ndetection rate by class (SDE >= 7):")
    print(df.groupby("label")["detected"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
