#!/usr/bin/env python
"""Download real light curves for labelled targets and extract vetting features.

This produces the counterpart to `make_dataset.py`: identical feature vectors,
but computed on real observations with published dispositions instead of
signals we injected ourselves.  Comparing models trained on each is how the
synthetic-to-real domain gap gets measured rather than assumed.

Downloads dominate the runtime and are network-bound, so they run on a thread
pool while feature extraction runs on the CPU.  Results are flushed to shards
so an interrupted run resumes instead of starting over.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.pipeline import analyze          # noqa: E402

FLUSH_EVERY = 25


def download(target: str, mission: str, max_quarters: int = 1,
             max_days: float = 45.0):
    """Fetch a light curve, preferring the pipeline-corrected PDCSAP flux.

    Kepler data is 30-minute cadence in quarters of ~90 days, against TESS's
    2-minute cadence over 27 days.  We stitch at most `max_quarters` so the
    baseline stays roughly comparable and the download stays cheap.

    We also truncate to `max_days`.  Two stitched Kepler quarters span ~180
    days, which makes the BLS period grid ~7x denser than for a TESS sector and
    the lowess detrend far more expensive -- one target sat compute-bound for
    over eight minutes.  Truncating also narrows the domain gap, since the model
    is ultimately aimed at 27-day TESS sectors.

    lightkurve prints a progress bar per file straight to stdout.  Those writes
    race across download threads and end up closing the shared stream, which
    poisons unrelated targets with "I/O operation on closed file".  stdout is
    therefore redirected ONCE in the main thread, around the whole pool -- see
    `main`.  Doing it per worker with contextlib.redirect_stdout does not work:
    it mutates the global sys.stdout and is explicitly not thread-safe, so
    concurrent enter/exit pairs restore each other's already-closed buffers.
    """
    import lightkurve as lk

    if mission == "kepler":
        q = lk.search_lightcurve(target, mission="Kepler", author="Kepler",
                                 cadence="long")
        # Quarter 0 is a ~10-day commissioning run (469 points on one test
        # target) -- far too short to search for periods up to 13 days.
        # Quarter 1 is ~33 days, conveniently close to a TESS sector; later
        # quarters are ~90 days and get truncated anyway.
        missions = [str(m) for m in q.table["mission"]]
        keep = [i for i, m in enumerate(missions) if "Quarter 00" not in m]
        if not keep:
            return None
        q = q[keep]
    else:
        q = lk.search_lightcurve(target, mission="TESS", author="SPOC",
                                 exptime=120)
    if len(q) == 0:
        return None

    coll = q[:max_quarters].download_all()
    if coll is None or len(coll) == 0:
        return None
    lc = coll.stitch().remove_nans()

    t = np.asarray(lc.time.value, dtype=float)
    f = np.asarray(lc.flux.value, dtype=float)
    e = np.asarray(lc.flux_err.value, dtype=float)
    good = np.isfinite(t) & np.isfinite(f)
    t, f, e = t[good], f[good], e[good]

    if t.size and (t.max() - t.min()) > max_days:
        keep = t <= (t.min() + max_days)
        t, f, e = t[keep], f[keep], e[keep]

    if t.size < 500:
        return None
    return t, f, (e if np.isfinite(e).all() else None)


def process(row: dict, cache_dir: str) -> dict | None:
    """Download (or reuse a cached copy) and reduce to vetting features."""
    target, mission = row["target"], row["mission"]
    stem = target.replace(" ", "_")
    path = os.path.join(cache_dir, f"{stem}.npz")

    try:
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            t, f = d["time"], d["flux"]
            e = d["flux_err"] if "flux_err" in d and d["flux_err"].size else None
        else:
            got = download(target, mission)
            if got is None:
                return {"target": target, "label": row["label"],
                        "mission": mission, "error": "no data"}
            t, f, e = got
            np.savez_compressed(path, time=t, flux=f,
                                flux_err=e if e is not None else np.array([]))

        res = analyze(t, f, e, model=None, do_fit=False,
                      detrend_methods=("biweight", "lowess"))
        if res.features is None:
            return {"target": target, "label": row["label"], "mission": mission,
                    "error": f"no detection (SDE {res.detection.sde:.1f})"
                             if res.detection else "no detection"}

        det = res.detection
        out = dict(res.features)
        out.update(target=target, mission=mission, label=row["label"],
                   detected=det.detected, bls_period=det.period,
                   bls_depth=det.depth, bls_duration=det.duration,
                   sde_val=det.sde, snr_val=det.snr,
                   detrend_method=res.detrend_method,
                   published_period=row.get("published_period"),
                   published_depth_ppm=row.get("published_depth_ppm"),
                   mag=row.get("mag"), error=None)
        return out
    except Exception as exc:
        return {"target": target, "label": row["label"], "mission": mission,
                "error": f"{type(exc).__name__}: {exc}"[:200]}


def load_shards(d: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(d, "part_*.parquet")))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def flush(buf, shard_dir, i):
    path = os.path.join(shard_dir, f"part_{i:04d}.parquet")
    tmp = path + ".tmp"
    pd.DataFrame(buf).to_parquet(tmp, index=False)
    os.replace(tmp, path)
    buf.clear()
    return i + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--labels", default="data/labels/targets.csv")
    ap.add_argument("-o", "--out", default="data/processed/real.parquet")
    ap.add_argument("-n", "--per-class", type=int, default=60)
    ap.add_argument("--mission", choices=["kepler", "tess", "both"], default="kepler")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--cache", default="data/cache/bulk")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    shard_dir = os.path.join(os.path.dirname(args.out) or ".", "real_shards")
    os.makedirs(shard_dir, exist_ok=True)
    if args.fresh:
        for f in glob.glob(os.path.join(shard_dir, "part_*.parquet")):
            os.remove(f)

    lab = pd.read_csv(args.labels)
    if args.mission != "both":
        lab = lab[lab["mission"] == args.mission]
    lab = lab[lab["label"] != "false_positive"]     # too coarse to train on

    # brightest targets first: best signal-to-noise per download
    sample = (lab.sort_values("mag")
                 .groupby("label", group_keys=False)
                 .head(args.per_class))

    done = load_shards(shard_dir)
    if not done.empty:
        # Only successful rows count as done. A row that errored (a transient
        # network failure, say) must be retried rather than cached as a defeat.
        ok_rows = done[done["error"].isna()] if "error" in done else done
        seen = set(ok_rows["target"])
        before = len(sample)
        sample = sample[~sample["target"].isin(seen)]
        n_retry = (len(done) - len(ok_rows))
        print(f"resuming: {len(ok_rows)} good rows on disk, "
              f"{before - len(sample)} skipped, {n_retry} failures will retry")

    rows = sample.to_dict("records")
    if not rows:
        print("nothing left to fetch")
    else:
        print(f"fetching {len(rows)} targets on {args.jobs} threads")
        print(sample["label"].value_counts().to_string())
        buf, shard_i = [], len(glob.glob(os.path.join(shard_dir, "part_*.parquet")))
        try:
            # One redirect, in this thread only, for the lifetime of the pool.
            # tqdm writes to stderr, so progress still shows.
            with contextlib.redirect_stdout(io.StringIO()):
                with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                    futs = {ex.submit(process, r, args.cache): r for r in rows}
                    for fut in tqdm(as_completed(futs), total=len(futs)):
                        r = fut.result()
                        if r:
                            buf.append(r)
                        if len(buf) >= FLUSH_EVERY:
                            shard_i = flush(buf, shard_dir, shard_i)
        except KeyboardInterrupt:
            print("\ninterrupted -- flushing completed rows", file=sys.stderr)
        finally:
            if buf:
                flush(buf, shard_dir, shard_i)

    df = load_shards(shard_dir)
    if df.empty:
        print("no rows produced")
        return
    # prefer a successful row over an earlier failed attempt for the same target
    if "error" in df:
        df = (df.assign(_ok=df["error"].isna())
                .sort_values("_ok", ascending=False)
                .drop_duplicates(subset=["target"], keep="first")
                .drop(columns="_ok"))
    else:
        df = df.drop_duplicates(subset=["target"], keep="first")
    df.to_parquet(args.out, index=False)

    ok = df[df["error"].isna()] if "error" in df else df
    print(f"\nwrote {len(df)} rows -> {args.out}")
    print(f"usable (detected): {len(ok)} / {len(df)}")
    if len(ok):
        print(ok["label"].value_counts().to_string())
    if "error" in df and df["error"].notna().any():
        print("\nfailure reasons:")
        print(df["error"].dropna().str.slice(0, 40).value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
