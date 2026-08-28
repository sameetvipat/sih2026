"""Light curve cleaning and transit-safe detrending.

The single biggest failure mode in a transit pipeline is a detrending filter
that absorbs the transit itself.  We guard against that two ways:
  1. the biweight window is kept well above any plausible transit duration,
  2. detrending is iterated with in-transit points masked out.
"""
from __future__ import annotations

import numpy as np
from wotan import flatten

from .config import DETREND_WINDOW_DAYS


def normalize(flux: np.ndarray, flux_err: np.ndarray | None = None):
    """Divide by the median so flux sits about 1.0."""
    med = np.nanmedian(flux)
    if med == 0 or not np.isfinite(med):
        med = 1.0
    return flux / med, (flux_err / med if flux_err is not None else None)


def clip_outliers(time, flux, flux_err=None, upper=4.0, lower=None):
    """Clip positive outliers only.

    Flares and cosmic rays are upward excursions and are safe to remove.
    Downward excursions are NOT clipped by default, because a transit *is* a
    downward excursion: for a bright star a 1% transit sits ~50 sigma below the
    median, so any finite lower threshold deletes exactly the signal we are
    looking for.  (Measured: a 20-sigma lower clip removed 496 of 496
    in-transit points from a 1%-deep transit at 200 ppm noise.)

    Genuinely bad low points are handled downstream by the robust biweight
    filter, which is designed to tolerate them.  Pass `lower` explicitly only
    if you know the data has a specific downward artefact.
    """
    med = np.nanmedian(flux)
    sigma = 1.4826 * np.nanmedian(np.abs(flux - med))  # robust MAD-based sigma
    if sigma <= 0 or not np.isfinite(sigma):
        keep = np.isfinite(flux)
    else:
        keep = flux < med + upper * sigma
        if lower is not None:
            keep &= flux > med - lower * sigma
    keep &= np.isfinite(flux)
    out = (time[keep], flux[keep])
    return out + ((flux_err[keep],) if flux_err is not None else (None,))


def detrend(time, flux, window_days=DETREND_WINDOW_DAYS, mask=None,
            method="biweight"):
    """Remove stellar variability / systematics with a sliding filter.

    `method` is any wotan filter.  No single choice wins everywhere: biweight
    is the robust default, but on heavily spotted stars it fails badly -- on
    AU Mic (5% spot modulation against a 0.26% transit) biweight locks BLS onto
    the 4.86 d rotation period, while lowess recovers the true 8.46 d transit
    period at SDE 10.3.  `pipeline.analyze` can therefore try several.

    `mask` marks points to *exclude* from trend estimation (i.e. in-transit
    points), so the trend is interpolated across the transit rather than
    bending down into it.
    """
    if mask is not None and mask.any():
        work = flux.copy()
        work[mask] = np.nan
    else:
        work = flux

    kw = dict(window_length=window_days, method=method, return_trend=True)
    if method == "cosine":
        kw["robust"] = True
    flat, trend = flatten(time, work, **kw)

    # wotan does NOT interpolate across NaN input -- it propagates NaN straight
    # into the trend.  Dividing by that would delete exactly the in-transit
    # points we masked, which silently destroys the signal (measured: SDE 21 ->
    # 6.7 on Pi Men c).  So bridge the gaps ourselves before dividing.
    bad = ~np.isfinite(trend)
    if bad.any() and (~bad).sum() > 2:
        trend = trend.copy()
        trend[bad] = np.interp(time[bad], time[~bad], trend[~bad])

    # apply the trend to the *original* flux so masked points survive
    with np.errstate(invalid="ignore", divide="ignore"):
        flat_full = flux / trend
    return flat_full, trend


def in_transit_mask(time, period, t0, duration, n_durations=1.5):
    """Boolean mask of points within +/- n_durations/2 of any transit centre."""
    if not np.isfinite(period) or period <= 0 or not np.isfinite(duration):
        return np.zeros_like(time, dtype=bool)
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    return np.abs(phase) < (n_durations * duration / 2.0)


def prepare(time, flux, flux_err=None, window_days=DETREND_WINDOW_DAYS,
            method="biweight"):
    """Full clean: finite -> normalise -> clip -> detrend. Returns arrays."""
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    flux_err = flux_err[good] if flux_err is not None else None

    flux, flux_err = normalize(flux, flux_err)
    time, flux, flux_err = clip_outliers(time, flux, flux_err)
    flat, trend = detrend(time, flux, window_days, method=method)

    good = np.isfinite(flat)
    result = (time[good], flat[good])
    result += (flux_err[good] if flux_err is not None else None,)
    return result + (trend[good],)
