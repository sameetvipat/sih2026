#!/usr/bin/env python
"""Recompute the feature vectors for already-downloaded light curves.

Adding a feature invalidates every stored row -- not the photons, which are
cached, just the reduction of them. Re-running the downloader would work but
would also re-open the download queue and start fetching new targets, which is
not what is wanted here: the target list must stay exactly as it is so that
before/after model comparisons are not confounded by a changed dataset.

This reads the existing shards for the target list, re-analyses each one from
the local cache, and writes fresh shards. Network is never touched -- a target
whose cache file is missing is reported and skipped rather than fetched.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.fetch import (load_shards, merge_preferring_success,   # noqa: E402
                          run_resumable_pool)
from exodet.pipeline import analyze                                # noqa: E402


def _cache_path(target: str, cache_dirs) -> str | None:
    stem = target.replace(" ", "_") + ".npz"
    for d in cache_dirs:
        p = os.path.join(d, stem)
        if os.path.exists(p):
            return p
    return None


def process(task) -> dict:
    row, cache_dirs = task
    target = row["target"]
    base = dict(target=target, mission=row.get("mission"), label=row["label"])
    path = _cache_path(target, cache_dirs)
    if path is None:
        return {**base, "error": "not cached"}
    try:
        d = np.load(path, allow_pickle=True)
        t, f = d["time"], d["flux"]
        e = d["flux_err"] if "flux_err" in d and d["flux_err"].size else None
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
                   sde_corrected=res.sde_corrected,
                   detrend_is_fallback=res.detrend_is_fallback,
                   published_period=row.get("published_period"),
                   published_depth_ppm=row.get("published_depth_ppm"),
                   mag=row.get("mag"), error=None)
        return out
    except Exception as exc:
        return {**base, "error": f"{type(exc).__name__}: {exc}"[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/processed/real_shards")
    ap.add_argument("--out", default="data/processed/real.parquet")
    ap.add_argument("--out-shards", default="data/processed/real_shards_v2")
    ap.add_argument("--cache", nargs="+",
                    default=["data/cache/bulk", "data/cache"])
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    src = load_shards(args.shards)
    if src.empty:
        print(f"no shards under {args.shards}")
        return
    # One row per target: a target that once failed and later succeeded should
    # be recomputed once, from its successful identity.
    src = merge_preferring_success(src, "target")
    print(f"{len(src)} unique targets in {args.shards}")

    if args.fresh:
        for f in glob.glob(os.path.join(args.out_shards, "part_*.parquet")):
            os.remove(f)
    done = load_shards(args.out_shards)
    if not done.empty:
        seen = set(done["target"])
        src = src[~src["target"].isin(seen)]
        print(f"resuming: {len(seen)} already recomputed, {len(src)} to go")

    tasks = [(r, args.cache) for r in src.to_dict("records")]
    if tasks:
        run_resumable_pool(tasks, process, args.out_shards, jobs=args.jobs,
                           desc="refresh features",
                           failures_path="data/processed/refresh_failures.csv")

    df = merge_preferring_success(load_shards(args.out_shards), "target")
    df.to_parquet(args.out, index=False)
    ok = df[df["error"].isna()]
    print(f"\nwrote {len(df)} rows -> {args.out}")
    print(f"usable (detected): {len(ok)} / {len(df)}")
    print(ok["label"].value_counts().to_string())
    if df["error"].notna().any():
        print("\nfailure reasons:")
        print(df["error"].dropna().str.slice(0, 45).value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
