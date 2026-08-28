"""Physically-motivated vetting features.

Each feature encodes a discriminator a human vetter would actually use:

  odd/even depth difference  -> EB caught at half its true period
  secondary eclipse depth    -> EB (a planet's secondary is unmeasurable here)
  trapezoid T23/T14 ratio    -> U-shape (planet) vs V-shape (EB / grazing)
  implied stellar density    -> is the duration physically consistent with the
                                period, or is this a systematic / wrong harmonic?
  Lomb-Scargle power ratio   -> smooth sinusoidal variability (starspots)

These are what a gradient-boosted tree can act on directly, and -- unlike a raw
CNN -- what lets the pipeline explain *why* it made a call.
"""
from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import least_squares

from .config import G_CGS
from .search import fold

DAY_S = 86400.0

FEATURE_NAMES = [
    "log_depth", "log_duration_hr", "log_period", "duration_phase_frac",
    "sde", "log_snr", "n_transits",
    "odd_even_sigma", "odd_even_frac",
    "secondary_sigma", "secondary_depth_ratio",
    "sec_sigma_2p", "sec_ratio_2p", "sec_ratio_2p_dev", "odd_even_sigma_2p",
    "trap_t23_t14", "trap_depth_ratio", "shape_resid_ratio",
    "log_rho_implied", "depth_consistency", "ls_power_ratio",
    "oot_skew",
]


# --------------------------------------------------------------------------- #
# Trapezoid shape fit
# --------------------------------------------------------------------------- #
def trapezoid(dt, depth, t14, t23, base):
    """Symmetric trapezoid in time-from-mid-transit `dt` (days)."""
    x = np.abs(dt)
    h14, h23 = t14 / 2.0, t23 / 2.0
    f = np.full_like(x, base)
    f[x <= h23] = base - depth
    ramp = (x > h23) & (x < h14)
    if h14 > h23:
        f[ramp] = base - depth * (h14 - x[ramp]) / (h14 - h23)
    return f


def fit_trapezoid(dt, flux, depth0, dur0):
    """Fit a trapezoid; T23/T14 near 1 is box/U-shaped, near 0 is V-shaped."""
    base0 = float(np.median(flux[np.abs(dt) > dur0]))if np.any(np.abs(dt) > dur0) else 1.0
    p0 = [max(depth0, 1e-6), dur0, dur0 * 0.6, base0]

    def resid(p):
        return trapezoid(dt, *p) - flux

    try:
        r = least_squares(
            resid, p0, method="trf",
            bounds=([0.0, dur0 * 0.2, 0.0, base0 - 0.1],
                    [1.0, dur0 * 3.0, dur0 * 3.0, base0 + 0.1]),
            max_nfev=2000,
        )
        depth, t14, t23, base = r.x
        t23 = min(t23, t14)                       # flat part cannot exceed total
        ratio = t23 / t14 if t14 > 0 else 0.0
        # how much better is the trapezoid than a flat line?
        ss_model = float(np.sum(r.fun ** 2))
        ss_flat = float(np.sum((flux - base) ** 2))
        resid_ratio = ss_model / ss_flat if ss_flat > 0 else 1.0
        return dict(depth=float(depth), t14=float(t14), t23=float(t23),
                    ratio=float(np.clip(ratio, 0, 1)), resid_ratio=float(resid_ratio))
    except Exception:
        return dict(depth=depth0, t14=dur0, t23=dur0 * 0.5,
                    ratio=0.5, resid_ratio=1.0)


# --------------------------------------------------------------------------- #
# Individual diagnostics
# --------------------------------------------------------------------------- #
def odd_even_test(time, flux, period, t0, duration):
    """Compare depths of alternating transits.

    An eclipsing binary detected at half its true period alternates between
    primary and secondary eclipse, so odd and even events differ in depth.
    Returns (significance in sigma, fractional difference).
    """
    epoch = np.round((time - t0) / period).astype(int)
    phase = (time - t0) - epoch * period
    intr = np.abs(phase) < duration / 2.0
    oot = (np.abs(phase) > duration) & (np.abs(phase) < 2.5 * duration)
    if oot.sum() < 10:
        return 0.0, 0.0
    base = np.median(flux[oot])

    depths, errs = {}, {}
    for parity in (0, 1):
        sel = intr & (epoch % 2 == parity)
        if sel.sum() < 5:
            return 0.0, 0.0
        depths[parity] = base - np.mean(flux[sel])
        errs[parity] = np.std(flux[sel]) / np.sqrt(sel.sum())

    diff = abs(depths[0] - depths[1])
    err = np.hypot(errs[0], errs[1])
    sigma = diff / err if err > 0 else 0.0
    mean_depth = 0.5 * (depths[0] + depths[1])
    frac = diff / abs(mean_depth) if mean_depth != 0 else 0.0
    return float(np.clip(sigma, 0, 200)), float(np.clip(frac, 0, 10))


def secondary_test(time, flux, period, t0, duration, primary_depth):
    """Search for a secondary eclipse at phase 0.5.

    A real planet produces no detectable secondary in TESS optical data; an
    eclipsing binary (or a blend of one) usually does.
    """
    phase, f = fold(time, flux, period, t0)
    dur_phase = duration / period
    at_sec = np.abs(np.abs(phase) - 0.5) < dur_phase / 2.0
    oot = (np.abs(phase) > 2 * dur_phase) & (np.abs(np.abs(phase) - 0.5) > 2 * dur_phase)
    if at_sec.sum() < 5 or oot.sum() < 20:
        return 0.0, 0.0
    base = np.median(f[oot])
    sec_depth = base - np.mean(f[at_sec])
    err = np.std(f[oot]) / np.sqrt(at_sec.sum())
    sigma = sec_depth / err if err > 0 else 0.0
    ratio = sec_depth / primary_depth if primary_depth > 0 else 0.0
    return float(np.clip(sigma, -50, 200)), float(np.clip(ratio, -2, 5))


def implied_density(period, duration, depth):
    """Mean stellar density implied by the transit geometry (g/cm^3).

    For a central transit, a/R* ~ P / (pi * T14), and
    rho* = 3*pi/(G P^2) * (a/R*)^3.
    Values far outside ~0.1-10 g/cm^3 mean the event is not a transit around a
    normal star -- often a systematic or the wrong period harmonic.
    """
    if duration <= 0 or period <= 0:
        return np.nan
    rp = np.sqrt(max(depth, 0.0))
    aRs = (period / (np.pi * duration)) * (1.0 + rp)
    p_s = period * DAY_S
    return float(3.0 * np.pi / (G_CGS * p_s ** 2) * aRs ** 3)


def per_transit_consistency(time, flux, period, t0, duration):
    """Scatter of individual transit depths, normalised by the mean depth.

    Real transits repeat at constant depth; systematics and blends often do not.
    """
    epoch = np.round((time - t0) / period).astype(int)
    phase = (time - t0) - epoch * period
    intr = np.abs(phase) < duration / 2.0
    oot = (np.abs(phase) > duration) & (np.abs(phase) < 3 * duration)
    if oot.sum() < 10:
        return 0.0
    base = np.median(flux[oot])
    depths = [base - np.mean(flux[intr & (epoch == e)])
              for e in np.unique(epoch[intr]) if (intr & (epoch == e)).sum() >= 3]
    if len(depths) < 2:
        return 0.0
    depths = np.array(depths)
    m = np.mean(depths)
    return float(np.clip(np.std(depths) / abs(m), 0, 10)) if m != 0 else 0.0


def ls_power_ratio(time, flux, period):
    """Lomb-Scargle power at the detected period, relative to the mean.

    Smooth sinusoidal modulation (starspots) produces a large ratio; a boxy
    transit spreads its power across harmonics and produces a small one.
    """
    try:
        freq = np.linspace(1.0 / 20.0, 1.0 / 0.2, 4000)
        power = LombScargle(time, flux).power(freq)
        at = np.interp(1.0 / period, freq, power)
        mean_p = np.mean(power)
        return float(np.clip(at / mean_p, 0, 500)) if mean_p > 0 else 0.0
    except Exception:
        return 0.0


def harmonic_test(time, flux, period, t0, duration):
    """Re-examine the signal at twice the detected period.

    BLS almost always locks onto *half* an eclipsing binary's true period,
    because primary and secondary eclipses then stack at the same phase.  This
    is the single most important vetting check in the pipeline:

      * fold at 2P and a genuine planet shows two *equal* depth events (its own
        alternating transits), giving a depth ratio near 1.0;
      * an eclipsing binary shows a primary at phase 0 and a *shallower*
        secondary at phase 0.5, giving a depth ratio well below 1.0.

    So the depth ratio at 2P separates "BLS found the right period" from
    "BLS found half the true period", which is exactly the transit/EB question.
    """
    p2 = 2.0 * period
    if (time.max() - time.min()) < 1.5 * p2:
        return 0.0, 1.0, 0.0     # baseline too short to test the harmonic
    sec_sigma, sec_ratio = secondary_test(time, flux, p2, t0, duration,
                                          primary_depth=_fold_depth(time, flux, p2, t0, duration))
    oe_sigma, _ = odd_even_test(time, flux, p2, t0, duration)
    return sec_sigma, sec_ratio, oe_sigma


def _fold_depth(time, flux, period, t0, duration):
    """Mean depth of the events at phase 0 when folded on `period`."""
    phase, f = fold(time, flux, period, t0)
    dur_phase = duration / period
    intr = np.abs(phase) < dur_phase / 2.0
    oot = (np.abs(phase) > 2 * dur_phase) & (np.abs(np.abs(phase) - 0.5) > 2 * dur_phase)
    if intr.sum() < 3 or oot.sum() < 10:
        return 1e-7
    return max(float(np.median(f[oot]) - np.mean(f[intr])), 1e-7)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def extract_features(time, flux, det) -> dict:
    """Compute the full vetting feature vector for one detection."""
    P, t0, dur, depth = det.period, det.t0, det.duration, max(det.depth, 1e-7)

    oe_sigma, oe_frac = odd_even_test(time, flux, P, t0, dur)
    sec_sigma, sec_ratio = secondary_test(time, flux, P, t0, dur, depth)
    sec_sig2, sec_rat2, oe_sig2 = harmonic_test(time, flux, P, t0, dur)

    # local view around mid-transit for the shape fit
    epoch = np.round((time - t0) / P).astype(int)
    dt = (time - t0) - epoch * P
    near = np.abs(dt) < 2.5 * dur
    trap = (fit_trapezoid(dt[near], flux[near], depth, dur)
            if near.sum() > 20 else
            dict(depth=depth, t14=dur, t23=dur * 0.5, ratio=0.5, resid_ratio=1.0))

    rho = implied_density(P, dur, depth)
    oot = flux[np.abs(dt) > 2.5 * dur]
    skew = (float(np.mean(((oot - oot.mean()) / oot.std()) ** 3))
            if oot.size > 30 and oot.std() > 0 else 0.0)

    return {
        "log_depth": float(np.log10(depth)),
        "log_duration_hr": float(np.log10(max(dur * 24.0, 1e-3))),
        "log_period": float(np.log10(max(P, 1e-3))),
        "duration_phase_frac": float(dur / P),
        "sde": float(det.sde),
        "log_snr": float(np.log10(max(det.snr, 1e-3))),
        "n_transits": float(det.n_transits),
        "odd_even_sigma": oe_sigma,
        "odd_even_frac": oe_frac,
        "secondary_sigma": sec_sigma,
        "secondary_depth_ratio": sec_ratio,
        "sec_sigma_2p": sec_sig2,
        "sec_ratio_2p": sec_rat2,
        # a genuine planet folded at 2P shows two equal-depth events, so this
        # sits near 0; any deviation -- in either direction -- means the two
        # alternating events differ, i.e. BLS found half an EB's true period.
        "sec_ratio_2p_dev": float(abs(sec_rat2 - 1.0)),
        "odd_even_sigma_2p": oe_sig2,
        "trap_t23_t14": trap["ratio"],
        "trap_depth_ratio": float(np.clip(trap["depth"] / depth, 0, 10)),
        "shape_resid_ratio": float(np.clip(trap["resid_ratio"], 0, 5)),
        "log_rho_implied": float(np.log10(np.clip(rho, 1e-4, 1e6))) if np.isfinite(rho) else 0.0,
        "depth_consistency": per_transit_consistency(time, flux, P, t0, dur),
        "ls_power_ratio": ls_power_ratio(time, flux, P),
        "oot_skew": float(np.clip(skew, -10, 10)),
    }
