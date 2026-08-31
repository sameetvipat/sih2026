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

    done = load_shards(shard_dir)
    attempted: set[str] = set()
    usable = pd.Series(dtype=int)
    rates: dict[str, float] = {}
    if not done.empty:
        ok_rows = done[done["error"].isna()] if "error" in done else done
        # "attempted" means *settled*, not merely tried. A no-detection is
        # deterministic -- the light curve is cached, so re-running BLS on it
        # burns time to reach the same verdict. A network failure is not
        # settled, and must stay in the queue or a transient outage silently
        # becomes a permanent hole in the dataset.
        errs = done["error"] if "error" in done else pd.Series(dtype=object)
        settled = done["error"].isna() | done["error"].fillna("").str.startswith(
            "no detection") if "error" in done else pd.Series(True, index=done.index)
        attempted = set(done.loc[settled, "target"])
        retryable = sorted(set(done.loc[~settled, "target"]))
        if retryable:
            print(f"[info] {len(retryable)} targets failed for non-deterministic "
                  f"reasons and stay in the queue")
        usable = ok_rows["label"].value_counts()
        # Detection rate is measured, not assumed. It differs sharply by class
        # (blend detects far worse than eclipse), and using one global rate
        # would under-request exactly the class that needs the most attempts.
        for c in done["label"].unique():
            tried = int((done["label"] == c).sum())
            got = int(usable.get(c, 0))
            rates[c] = (got / tried) if tried >= 20 else 0.6

    # The 400/class target counts USABLE rows -- ones where BLS actually found
    # something to extract features from. Capping *attempts* at 400, as this
    # script used to, therefore cannot reach it for any class detecting below
    # 100%: blend detects at ~57%, so 400 attempts asymptotes at ~230 usable.
    # Attempts are now sized from the measured rate to cover the deficit.
    avail = lab["label"].value_counts()
    plan = []
    for c in avail.index:
        have = int(usable.get(c, 0))
        deficit = max(0, args.per_class - have)
        rate = max(rates.get(c, 0.6), 0.15)      # floor: never demand infinity
        need = int(np.ceil(deficit / rate)) if deficit else 0
        pool = lab[(lab["label"] == c) & (~lab["target"].isin(attempted))]
        take = min(need, len(pool))
        plan.append(dict(label=c, have=have, deficit=deficit, rate=rate,
                         need=need, pool=len(pool), take=take))

    # Most-deficient class first, and strictly in blocks rather than
    # interleaved. Under a hard time ceiling an even spread would leave every
    # class equally short; a block order means the ceiling truncates the
    # classes that need the fetch least.
    plan.sort(key=lambda d: d["deficit"], reverse=True)

    print("download plan (usable-count deficit drives attempt count):")
    print(f"  {'class':<10}{'usable':>7}{'deficit':>9}{'det.rate':>10}"
          f"{'attempts':>10}{'unfetched':>11}")
    for d in plan:
        note = "" if d["take"] >= d["need"] else "  <- catalogue exhausted"
        print(f"  {d['label']:<10}{d['have']:>7}{d['deficit']:>9}"
              f"{d['rate']:>10.2f}{d['take']:>10}{d['pool']:>11}{note}")

    frames = []
    for d in plan:
        if d["take"] <= 0:
            continue
        pool = lab[(lab["label"] == d["label"]) & (~lab["target"].isin(attempted))]
        frames.append(pool.sort_values("mag").head(d["take"]))   # brightest first
    sample = pd.concat(frames) if frames else lab.iloc[0:0]

    if not done.empty:
        print(f"\nresuming: {len(done)} rows on disk "
              f"({int(usable.sum())} usable), {len(attempted)} targets skipped")
    if sample.empty:
        print("\nevery class is at target; nothing to fetch")

    # sample is already in deficit-priority order; ThreadPoolExecutor consumes
    # submissions in order, so this is what makes the ceiling truncate the
    # least-needed class rather than a random one.
    tasks = [(r, args.cache) for r in sample.to_dict("records")]
    if tasks:
        print(f"\nfetching {len(tasks)} targets on {args.jobs} threads, "
              f"most-deficient class first")
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
