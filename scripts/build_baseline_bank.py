#!/usr/bin/env python
"""Build a bank of real, signal-free light curves to inject synthetic signals into.

Why this exists: a controlled experiment showed the classifier's synthetic-to-real
accuracy gap (0.984 -> 0.42) is NOT explained by cadence mismatch -- regenerating
synthetic data at the target cadence left the gap unchanged, and 82% of features
stayed cadence-robust on synthetic data while still collapsing on real data.  The
dominant cause is that the synthetic noise model (white noise plus a handful of
low-frequency sinusoids) is categorically too clean next to real instrumental
systematics and real stellar variability.

You cannot close a 20x feature-separation gap by tuning sinusoid amplitudes.  So
instead of trying to *simulate* a realistic noise floor, we borrow one: inject
known-truth signals onto real quiet stars.  This is standard practice for
injection-recovery work in the exoplanet literature, and it gives a genuinely
real noise floor while keeping exact ground truth.

A baseline is accepted only if BLS finds nothing above the detection threshold.
Targets where BLS *does* find something are logged and discarded, never silently
kept: such a star may carry its own real, undocumented signal, and injecting on
top of it would mislabel a mixed signal as purely ours.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.config import SDE_THRESHOLD                       # noqa: E402
from exodet.fetch import (download_lightcurve, load_shards,    # noqa: E402
                          merge_preferring_success,
                          run_resumable_pool, with_backoff)
from exodet.preprocess import prepare                          # noqa: E402
from exodet.search import run_bls                              # noqa: E402

# Brightness strata. Photon noise scales steeply with magnitude, so a bank drawn
# only from bright stars would present an unrealistically clean noise floor --
# the exact failure this whole exercise is meant to fix, reintroduced from a
# different angle.
KEPLER_BINS = [("bright", 9.0, 12.5), ("medium", 12.5, 13.8),
               ("faint", 13.8, 15.0)]
TESS_BINS = [("bright", 6.0, 9.5), ("medium", 9.5, 11.5),
             ("faint", 11.5, 13.5)]


def candidate_targets(mission: str, n_per_bin: int, seed: int,
                      exclude: set[str]) -> pd.DataFrame:
    """Sample stars stratified by brightness, excluding known-signal targets."""
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive as NEA

    if mission == "kepler":
        df = NEA.query_criteria(table="q1_q17_dr25_stellar",
                                select="kepid,kepmag",
                                where="kepmag > 9 and kepmag < 15").to_pandas()
        df["target"] = "KIC " + df["kepid"].astype(int).astype(str)
        df["mag"] = df["kepmag"]
        bins = KEPLER_BINS
    else:
        # TOI hosts are the only broad TESS target list reachable without a
        # bulk TIC download; we drop the ones with dispositions below.
        df = NEA.query_criteria(table="toi", select="tid,st_tmag").to_pandas()
        df["target"] = "TIC " + df["tid"].astype(int).astype(str)
        df["mag"] = df["st_tmag"]
        bins = TESS_BINS

    df = df[df["mag"].notna()]
    df = df[~df["target"].isin(exclude)]         # never use a known-signal star
    df = df.drop_duplicates(subset=["target"])

    out = []
    for name, lo, hi in bins:
        sub = df[(df["mag"] >= lo) & (df["mag"] < hi)]
        if sub.empty:
            print(f"[warn] no candidates in {mission}/{name} bin", file=sys.stderr)
            continue
        take = sub.sample(min(n_per_bin, len(sub)), random_state=seed)
        out.append(take.assign(brightness_bin=name))
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    res["mission"] = mission
    return res


def vet_one(task) -> dict | None:
    """Download a candidate and accept it only if BLS finds nothing."""
    row, cache_dir = task
    target, mission = row["target"], row["mission"]
    stem = target.replace(" ", "_")
    path = os.path.join(cache_dir, f"{mission}_{stem}.npz")

    base = dict(target=target, mission=mission, mag=float(row["mag"]),
                brightness_bin=row["brightness_bin"], segment=int(row["segment"]))
    try:
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            t, f = d["time"], d["flux"]
            e = d["flux_err"] if d["flux_err"].size else None
        else:
            got = with_backoff(download_lightcurve, target, mission,
                               segment=int(row["segment"]))
            if got is None:
                return {**base, "accepted": False, "error": "no data"}
            t, f, e = got

        # Vet with the SAME cleaning + search path used at inference time.
        ct, cf, ce, _ = prepare(t, f, e)
        if ct.size < 500:
            return {**base, "accepted": False, "error": "too few points"}
        det, _, _ = run_bls(ct, cf, ce)

        if det.sde >= SDE_THRESHOLD:
            # Discard, loudly. This star may carry its own real signal.
            return {**base, "accepted": False, "sde": float(det.sde),
                    "error": f"rejected: BLS found a signal (SDE {det.sde:.1f})"}

        # Cache RAW arrays. Injection must happen before prepare(), so that a
        # training example and a live query traverse identical preprocessing.
        if not os.path.exists(path):
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(path, time=t, flux=f,
                                flux_err=e if e is not None else np.array([]))

        oot_scatter = float(1.4826 * np.median(np.abs(cf - np.median(cf))))
        return {**base, "accepted": True, "sde": float(det.sde),
                "path": path, "n_points": int(t.size),
                "baseline_days": float(t.max() - t.min()),
                "cadence_min": float(np.median(np.diff(t)) * 1440),
                "oot_scatter_ppm": oot_scatter * 1e6, "error": None}
    except Exception as exc:
        return {**base, "accepted": False,
                "error": f"{type(exc).__name__}: {exc}"[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", choices=["kepler", "tess"], default="kepler")
    ap.add_argument("-n", "--per-bin", type=int, default=200,
                    help="candidates to TRY per brightness bin (many are rejected)")
    ap.add_argument("-j", "--jobs", type=int, default=12)
    ap.add_argument("--cache", default="data/baselines")
    ap.add_argument("--manifest", default="data/baselines/manifest.csv")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    shard_dir = os.path.join(args.cache, f"_shards_{args.mission}")
    os.makedirs(args.cache, exist_ok=True)
    if args.fresh:
        import glob as _g
        for f in _g.glob(os.path.join(shard_dir, "part_*.parquet")):
            os.remove(f)

    # never draw a baseline from a star with a catalogued signal
    exclude: set[str] = set()
    lab_path = "data/labels/targets.csv"
    if os.path.exists(lab_path):
        exclude = set(pd.read_csv(lab_path)["target"])
        print(f"excluding {len(exclude)} catalogued signal-bearing targets")

    cand = candidate_targets(args.mission, args.per_bin, args.seed, exclude)
    # vary the observing window: systematics are epoch-specific
    rng = np.random.default_rng(args.seed)
    cand["segment"] = rng.integers(0, 6, size=len(cand))
    print(f"{len(cand)} candidates:")
    print(cand.groupby("brightness_bin").size().to_string())

    done = load_shards(shard_dir)
    if not done.empty:
        seen = set(done[done["error"].isna()]["target"]) if "error" in done else set(done["target"])
        before = len(cand)
        cand = cand[~cand["target"].isin(seen)]
        print(f"resuming: {len(seen)} accepted already, {before - len(cand)} skipped")

    tasks = [(r, args.cache) for r in cand.to_dict("records")]
    if tasks:
        run_resumable_pool(tasks, vet_one, shard_dir, jobs=args.jobs,
                           desc=f"vetting {args.mission} baselines",
                           failures_path=os.path.join(args.cache, "failures.csv"))

    df = merge_preferring_success(load_shards(shard_dir), "target")
    acc = df[df["accepted"] == True] if "accepted" in df else df   # noqa: E712

    if len(acc):
        prev = (pd.read_csv(args.manifest)
                if os.path.exists(args.manifest) else pd.DataFrame())
        man = pd.concat([prev, acc], ignore_index=True) if len(prev) else acc
        man = man.drop_duplicates(subset=["target"], keep="last")
        man.to_csv(args.manifest, index=False)
        print(f"\nmanifest -> {args.manifest}  ({len(man)} baselines)")
        print(man.groupby(["mission", "brightness_bin"]).size().to_string())
        print(f"\nnoise floor by bin (ppm):")
        print(man.groupby("brightness_bin")["oot_scatter_ppm"]
                 .describe()[["count", "50%", "max"]].round(0).to_string())

    print(f"\ntried {len(df)}  accepted {len(acc)}  "
          f"rejected {len(df) - len(acc)}")
    if "error" in df and df["error"].notna().any():
        print("\nrejection reasons:")
        print(df["error"].dropna().str.slice(0, 45).value_counts().head(6).to_string())


if __name__ == "__main__":
    main()
