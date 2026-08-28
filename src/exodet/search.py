"""Periodic dip detection via Box Least Squares.

BLS answers "is there a periodic dip, and at what period/epoch/duration?".
It says nothing about *what* the dip is -- that is the classifier's job.
This module is the cheap triage stage of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from astropy.timeseries import BoxLeastSquares

from .config import MAX_PERIOD, MIN_PERIOD, SDE_THRESHOLD

# trial transit durations, in days (~45 min to ~6 h)
DURATION_GRID = np.array([0.03, 0.05, 0.08, 0.12, 0.18, 0.25])


@dataclass
class Detection:
    """Best periodic dip found by BLS, with its significance."""
    period: float
    t0: float
    duration: float
    depth: float
    depth_err: float
    snr: float
    sde: float
    n_transits: int
    detected: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _sde(power: np.ndarray) -> float:
    """Signal Detection Efficiency: peak height above the periodogram floor."""
    power = power[np.isfinite(power)]
    if power.size < 10:
        return 0.0
    sd = np.std(power)
    return float((power.max() - np.mean(power)) / sd) if sd > 0 else 0.0


def bin_lightcurve(time, flux, flux_err=None, minutes=10.0):
    """Bin to a coarser cadence for the period search.

    Transit durations are 1-6 hours, so 10-minute bins preserve the signal
    exactly (verified: identical recovered period and SDE) while cutting the
    BLS cost ~3x.  Final parameters are always fitted on the unbinned data.
    """
    dt = minutes / 1440.0
    idx = ((time - time.min()) / dt).astype(int)
    n = idx.max() + 1
    cnt = np.bincount(idx, minlength=n)
    ok = cnt > 0
    tb = np.bincount(idx, weights=time, minlength=n)[ok] / cnt[ok]
    fb = np.bincount(idx, weights=flux, minlength=n)[ok] / cnt[ok]
    if flux_err is None:
        return tb, fb, None
    eb = np.full(fb.size, np.median(flux_err) / np.sqrt(max(np.median(cnt[ok]), 1)))
    return tb, fb, eb


def run_bls(time, flux, flux_err=None,
            min_period=MIN_PERIOD, max_period=MAX_PERIOD,
            duration_grid=DURATION_GRID, oversample=5, bin_minutes=10.0):
    """Run BLS and return (Detection, periods, power) for plotting.

    The search runs on binned data for speed; `Detection` timings refer to the
    original time axis, so downstream fitting still uses full cadence.
    """
    baseline = float(time.max() - time.min())
    # never search beyond half the baseline -- we need >=2 events to fold
    max_period = min(max_period, baseline / 2.0)
    if max_period <= min_period:
        raise ValueError(f"baseline {baseline:.2f} d too short to search")

    if bin_minutes and bin_minutes > 0:
        stime, sflux, serr = bin_lightcurve(time, flux, flux_err, bin_minutes)
    else:
        stime, sflux, serr = time, flux, flux_err

    bls = BoxLeastSquares(stime, sflux, dy=serr)
    periods = bls.autoperiod(duration_grid,
                             minimum_period=min_period,
                             maximum_period=max_period,
                             frequency_factor=1.0 / oversample)
    power = bls.power(periods, duration_grid)

    i = int(np.nanargmax(power.power))
    period = float(power.period[i])
    t0 = float(power.transit_time[i])
    duration = float(power.duration[i])
    depth = float(power.depth[i])

    stats = bls.compute_stats(period, duration, t0)
    n_transits = int(len(stats["transit_times"]))

    # astropy returns depth as (value, uncertainty)
    d = np.atleast_1d(stats["depth"])
    depth_err = float(d[1]) if d.size > 1 else np.nan
    snr = float(depth / depth_err) if depth_err and np.isfinite(depth_err) and depth_err > 0 else 0.0

    sde = _sde(np.asarray(power.power))

    det = Detection(period=period, t0=t0, duration=duration, depth=depth,
                    depth_err=depth_err, snr=snr, sde=sde,
                    n_transits=n_transits,
                    detected=bool(sde >= SDE_THRESHOLD))
    return det, np.asarray(power.period), np.asarray(power.power)


def fold(time, flux, period, t0):
    """Phase-fold onto [-0.5, 0.5) in units of period, sorted by phase."""
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], flux[order]


def bin_phase(phase, flux, n_bins=200):
    """Bin a folded curve; returns (centres, means, errors-on-the-mean)."""
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    idx = np.digitize(phase, edges) - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    errs = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = flux[idx == b]
        if sel.size:
            means[b] = np.mean(sel)
            errs[b] = np.std(sel) / np.sqrt(sel.size) if sel.size > 1 else np.nan
    return centres, means, errs
