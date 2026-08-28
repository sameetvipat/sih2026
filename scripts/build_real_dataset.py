#!/usr/bin/env python
"""Download real light curves for catalogued targets and extract vetting features.

The counterpart to `make_dataset.py`: identical feature vectors, but computed on
real observations with published dispositions instead of injected signals.  This
is what the domain-gap comparison in `compare_domains.py` tests against.

Download, retry and checkpoint logic lives in `exodet.fetch` and is shared with
`build_baseline_bank.py` -- two divergent download harnesses would drift.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.fetch import (download_lightcurve, load_shards,      # noqa: E402
                          merge_preferring_success,
                          run_resumable_pool, with_backoff)
from exodet.pipeline import analyze                              # noqa: E402


def process(task) -> dict | None:
    """Fetch (or reuse a cached copy) and reduce to vetting features."""
    row, cache_dir = task
    target, mission = row["target"], row["mission"]
    stem = target.replace(" ", "_")
    path = os.path.join(cache_dir, f"{stem}.npz")
    base = dict(target=target, mission=mission, label=row["label"])

    try:
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            t, f = d["time"], d["flux"]
            e = d["flux_err"] if "flux_err" in d and d["flux_err"].size else None
        else:
            got = with_backoff(download_lightcurve, target, mission)
            if got is None:
                return {**base, "error": "no data"}
            t, f, e = got
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(path, time=t, flux=f,
                                flux_err=e if e is not None else np.array([]))

        res = analyze(t, f, e, model=None, do_fit=False,
                      detrend_methods=("biweight", "lowess"))
        if res.features is None:
            sde = f"{res.detection.sde:.1f}" if res.detection else "n/a"
            return {**base, "error": f"no detection (SDE {sde})"}

        det = res.detection
        out = dict(res.features)
        out.update(**base, detected=det.detected, bls_period=det.period,
                   bls_depth=det.depth, bls_duration=det.duration,
                   sde_val=det.sde, snr_val=det.snr,
                   detrend_method=res.detrend_method,
                   published_period=row.get("published_period"),
                   published_depth_ppm=row.get("published_depth_ppm"),
                   mag=row.get("mag"), error=None)
        return out
    except Exception as exc:
        return {**base, "error": f"{type(exc).__name__}: {exc}"[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--labels", default="data/labels/targets.csv")
    ap.add_argument("-o", "--out", default="data/processed/real.parquet")
    ap.add_argument("-n", "--per-class", type=int, default=400)
    ap.add_argument("--mission", choices=["kepler", "tess", "both"], default="kepler")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--cache", default="data/cache/bulk")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    shard_dir = os.path.join(os.path.dirname(args.out) or ".", "real_shards")
    if args.fresh:
        import glob as _g
        for f in _g.glob(os.path.join(shard_dir, "part_*.parquet")):
            os.remove(f)

    lab = pd.read_csv(args.labels)
    if args.mission != "both":
        lab = lab[lab["mission"] == args.mission]
    # TESS dispositions do not say *why* an object is a false positive, so they
    # cannot be mapped onto the four-class taxonomy.
    lab = lab[lab["label"] != "false_positive"]

    # Per-class targets follow catalogue availability rather than a naive even
    # split -- padding a scarce class with ambiguous-flag objects would bake in
    # label noise, which is exactly what this dataset exists to avoid.
    avail = lab["label"].value_counts()
    print("catalogue availability per class:")
    for c, n in avail.items():
        want = min(args.per_class, n)
        note = "" if n >= args.per_class else f"  <- capped by catalogue ({n})"
        print(f"  {c:<10} available {n:>5}   requesting {want}{note}")

    sample = (lab.sort_values("mag")             # brightest = best S/N per download
                 .groupby("label", group_keys=False)
                 .head(args.per_class))

    done = load_shards(shard_dir)
    if not done.empty:
        ok_rows = done[done["error"].isna()] if "error" in done else done
        seen = set(ok_rows["target"])
        before = len(sample)
        sample = sample[~sample["target"].isin(seen)]
        print(f"\nresuming: {len(ok_rows)} good rows on disk, "
              f"{before - len(sample)} skipped, "
              f"{len(done) - len(ok_rows)} failures will retry")

    tasks = [(r, args.cache) for r in sample.to_dict("records")]
    if tasks:
        print(f"\nfetching {len(tasks)} targets on {args.jobs} threads")
        run_resumable_pool(tasks, process, shard_dir, jobs=args.jobs,
                           desc="real labels",
                           failures_path="data/processed/failures.csv")

    df = merge_preferring_success(load_shards(shard_dir), "target")
    if df.empty:
        print("no rows produced")
        return
    df.to_parquet(args.out, index=False)

    ok = df[df["error"].isna()] if "error" in df else df
    print(f"\nwrote {len(df)} rows -> {args.out}")
    print(f"usable (detected): {len(ok)} / {len(df)}")
    if len(ok):
        print(ok["label"].value_counts().to_string())
    if "error" in df and df["error"].notna().any():
        print("\nfailure reasons:")
        print(df["error"].dropna().str.slice(0, 45).value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
