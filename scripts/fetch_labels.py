#!/usr/bin/env python
"""Build a labelled target list from published dispositions.

Two public catalogues carry the ground truth this project needs, and neither
depends on anyone providing us data:

  Kepler KOI cumulative table -- 9564 objects with a disposition AND, for the
    false positives, diagnostic flags that say *why* they are false. Those
    flags map almost one-to-one onto our taxonomy, which is what makes real
    four-class labels possible at all.

  TESS TOI table -- 8136 objects in the same photometric domain we actually
    target, but with a single coarse disposition and no sub-classification of
    the false positives.

We take fine-grained labels from Kepler and same-domain labels from TESS, and
keep them separate so the domain shift stays measurable rather than hidden.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.config import BLEND, ECLIPSE, TRANSIT, VARIABLE   # noqa: E402

# --- Kepler KOI false-positive flags ---------------------------------------
#   nt = not transit-like        (instrumental artefact or stellar variability)
#   ss = significant secondary   (an eclipsing binary)
#   co = centroid offset         (the signal comes from a neighbouring star)
#   ec = ephemeris match         (contamination from a known variable nearby)
#
# We only accept *unambiguous* flag combinations. An object flagged both ss and
# co could be an EB or a blended EB; including it would teach the model noise.


def label_koi(row) -> str | None:
    disp = row["koi_disposition"]
    if disp == "CONFIRMED":
        return TRANSIT
    if disp != "FALSE POSITIVE":
        return None                      # CANDIDATE: not adjudicated, skip

    nt, ss, co, ec = (int(row[f"koi_fpflag_{f}"]) for f in ("nt", "ss", "co", "ec"))

    if ss and not (nt or co or ec):
        return ECLIPSE                   # secondary eclipse, nothing else
    if nt and not (ss or co or ec):
        return VARIABLE                  # not transit-like at all
    if (co or ec) and not (ss or nt):
        return BLEND                     # signal is off-target
    return None                          # ambiguous combination


def fetch_koi() -> pd.DataFrame:
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive as NEA
    cols = ("kepid,kepoi_name,koi_disposition,koi_fpflag_nt,koi_fpflag_ss,"
            "koi_fpflag_co,koi_fpflag_ec,koi_period,koi_depth,koi_duration,"
            "koi_model_snr,koi_kepmag")
    df = NEA.query_criteria(table="cumulative", select=cols).to_pandas()
    df["label"] = df.apply(label_koi, axis=1)
    df = df[df["label"].notna()].copy()
    df["mission"] = "kepler"
    df["target"] = "KIC " + df["kepid"].astype(int).astype(str)
    df["published_period"] = df["koi_period"]
    df["published_depth_ppm"] = df["koi_depth"]
    df["published_duration_hr"] = df["koi_duration"]
    df["mag"] = df["koi_kepmag"]
    return df


def fetch_toi() -> pd.DataFrame:
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive as NEA
    cols = ("toi,tid,tfopwg_disp,pl_orbper,pl_trandep,pl_trandurh,st_tmag")
    df = NEA.query_criteria(table="toi", select=cols).to_pandas()
    # TESS gives no reason for a false positive, so only the planets are
    # usable as fine-grained labels; FP/FA are kept for a binary sanity check.
    mapping = {"CP": TRANSIT, "KP": TRANSIT, "FP": "false_positive",
               "FA": "false_positive"}
    df["label"] = df["tfopwg_disp"].map(mapping)
    df = df[df["label"].notna()].copy()
    df["mission"] = "tess"
    df["target"] = "TIC " + df["tid"].astype(int).astype(str)
    df["published_period"] = df["pl_orbper"]
    df["published_depth_ppm"] = df["pl_trandep"]
    df["published_duration_hr"] = df["pl_trandurh"]
    df["mag"] = df["st_tmag"]
    return df


KEEP = ["target", "mission", "label", "published_period",
        "published_depth_ppm", "published_duration_hr", "mag"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", choices=["kepler", "tess", "both"], default="both")
    ap.add_argument("-o", "--out", default="data/labels/targets.csv")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="cap each class, brightest targets first")
    args = ap.parse_args()

    frames = []
    if args.mission in ("kepler", "both"):
        print("querying Kepler KOI cumulative table ...")
        k = fetch_koi()
        print(f"  {len(k)} usable labels")
        print(k["label"].value_counts().to_string())
        frames.append(k[KEEP])
    if args.mission in ("tess", "both"):
        print("\nquerying TESS TOI table ...")
        t = fetch_toi()
        print(f"  {len(t)} usable labels")
        print(t["label"].value_counts().to_string())
        frames.append(t[KEEP])

    df = pd.concat(frames, ignore_index=True)
    # one row per target: a star with several KOIs would otherwise be
    # downloaded repeatedly and leak between train and test splits
    df = df.sort_values("mag").drop_duplicates(subset=["target"], keep="first")

    if args.max_per_class:
        # brightest first -- the best signal-to-noise for a fixed download budget
        df = (df.groupby(["mission", "label"], group_keys=False)
                .apply(lambda g: g.nsmallest(args.max_per_class, "mag")))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} unique targets -> {args.out}")
    print(df.groupby(["mission", "label"]).size().to_string())


if __name__ == "__main__":
    main()
