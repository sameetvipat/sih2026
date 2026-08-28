"""Physically-motivated synthetic light curve generator.

Produces labelled TESS-like light curves by injecting analytic signals into
realistic (white + red) noise.  This gives us a balanced training set with
*known ground-truth parameters*, which is what lets us quantify parameter
recovery accuracy later (injection-recovery testing).

The four signal classes are built so that the physical confusers are real:
  * an eclipsing binary detected at half its true period shows alternating
    odd/even depths,
  * a blend is literally a deep EB diluted by third light, so it lands in the
    planet depth regime while keeping its V-shape and secondary eclipse.
"""
from __future__ import annotations

import numpy as np
import batman

from .config import (
    CADENCE_MIN, DOWNLINK_GAP_DAYS, G_CGS, SECTOR_DAYS,
    TRANSIT, ECLIPSE, BLEND, VARIABLE, NOISE,
)

DAY_S = 86400.0


# --------------------------------------------------------------------------- #
# Observing cadence
# --------------------------------------------------------------------------- #
def make_time_grid(n_days: float = SECTOR_DAYS,
                   cadence_min: float = CADENCE_MIN,
                   gap_days: float = DOWNLINK_GAP_DAYS) -> np.ndarray:
    """Uniform cadence grid with a mid-sector downlink gap, as TESS really is."""
    dt = cadence_min / (24.0 * 60.0)
    t = np.arange(0.0, n_days, dt)
    if gap_days > 0:
        mid = n_days / 2.0
        keep = (t < mid - gap_days / 2.0) | (t > mid + gap_days / 2.0)
        t = t[keep]
    return t


def make_noise(t: np.ndarray, rng: np.random.Generator,
               white_ppm: float, red_ppm: float) -> np.ndarray:
    """White (photon) noise plus correlated red noise from instrument drift.

    Red noise is built from a handful of low-frequency sinusoids -- cheap, and
    it reproduces the slow wander that makes naive detrending dangerous.
    """
    white = rng.normal(0.0, white_ppm * 1e-6, size=t.size)

    red = np.zeros_like(t)
    baseline = t.max() - t.min()
    for _ in range(4):
        # periods from ~1/2 sector down to ~1/10 sector
        period = rng.uniform(baseline / 10.0, baseline / 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        red += rng.normal(0.0, 1.0) * np.sin(2 * np.pi * t / period + phase)
    if np.std(red) > 0:
        red *= (red_ppm * 1e-6) / np.std(red)
    return white + red


# --------------------------------------------------------------------------- #
# Transit geometry
# --------------------------------------------------------------------------- #
def a_over_rs(period_d: float, rho_star: float) -> float:
    """Scaled semi-major axis from Kepler's third law.

    (a/R*)^3 = G * rho_star * P^2 / (3*pi)
    """
    p_s = period_d * DAY_S
    return float((G_CGS * rho_star * p_s ** 2 / (3.0 * np.pi)) ** (1.0 / 3.0))


def transit_duration(period_d: float, rp: float, aRs: float, b: float) -> float:
    """Total (first-to-fourth contact) transit duration T14 in days."""
    inc = np.arccos(np.clip(b / aRs, -1.0, 1.0))
    arg = np.sqrt(max((1.0 + rp) ** 2 - b ** 2, 0.0)) / (aRs * np.sin(inc))
    if arg >= 1.0:
        return period_d / np.pi
    return float(period_d / np.pi * np.arcsin(arg))


def transit_model(t: np.ndarray, period: float, t0: float, rp: float,
                  aRs: float, b: float, u: tuple[float, float]) -> np.ndarray:
    """Mandel-Agol limb-darkened transit via batman. Returns normalised flux."""
    p = batman.TransitParams()
    p.t0 = t0
    p.per = period
    p.rp = rp
    p.a = aRs
    p.inc = float(np.degrees(np.arccos(np.clip(b / aRs, -1.0, 1.0))))
    p.ecc = 0.0
    p.w = 90.0
    p.u = list(u)
    p.limb_dark = "quadratic"
    return batman.TransitModel(p, t).light_curve(p)


# --------------------------------------------------------------------------- #
# Per-class signal builders
# --------------------------------------------------------------------------- #
def _draw_common(rng: np.random.Generator, t: np.ndarray,
                 period_range: tuple[float, float]):
    """Draw the geometry shared by transit / eclipse / blend signals."""
    baseline = t.max() - t.min()
    # cap the period so at least two events land inside the sector
    hi = min(period_range[1], baseline / 2.2)
    period = rng.uniform(period_range[0], max(hi, period_range[0] * 1.5))
    t0 = t.min() + rng.uniform(0.0, period)
    rho_star = rng.uniform(0.3, 3.0) * 1.408          # g/cm^3, FGKM-ish
    aRs = a_over_rs(period, rho_star)
    u = (rng.uniform(0.2, 0.6), rng.uniform(0.1, 0.4))
    return period, t0, aRs, u


def sim_transit(t, rng):
    """Planetary transit: shallow, U-shaped, no odd/even split.

    Hot planets DO show a secondary eclipse.  An earlier version of this
    generator assumed planetary secondaries were unmeasurable, so the
    classifier learned "secondary eclipse => not a planet" and misclassified
    the real ultra-hot Jupiter WASP-121 b (secondary ~500 ppm against a
    14000 ppm transit, i.e. a depth ratio of ~0.04) as a blend.

    Short-period planets therefore get a secondary whose depth is a few percent
    of the transit at most.  That stays well clear of the 0.15-0.9 surface
    brightness ratios used for eclipsing binaries, so the two remain separable
    -- but the model now learns that a *small* secondary is compatible with a
    planet, rather than ruling one out.
    """
    period, t0, aRs, u = _draw_common(rng, t, (0.7, 12.0))
    rp = rng.uniform(0.02, 0.16)                      # 0.04% - 2.6% deep
    b = rng.uniform(0.0, 0.85)
    if b > aRs:                                       # geometry must transit
        b = rng.uniform(0.0, 0.8) * aRs
    flux = transit_model(t, period, t0, rp, aRs, b, u)

    # thermal emission from a hot, close-in planet
    sec_ratio = 0.0
    if period < 4.0:
        sec_ratio = rng.uniform(0.0, 0.06) * (4.0 - period) / 4.0
        if sec_ratio > 1e-4:
            secondary = transit_model(t, period, t0 + period / 2.0,
                                      rp, aRs, b, u)
            flux = flux - (1.0 - secondary) * sec_ratio

    truth = dict(period=period, t0=t0, rp=rp, aRs=aRs, b=b,
                 depth=rp ** 2, sec_ratio=sec_ratio,
                 duration=transit_duration(period, rp, aRs, b))
    return flux, truth


def sim_eclipse(t, rng, dilution: float = 0.0):
    """Eclipsing binary: deep, V-shaped, with a secondary eclipse at phase 0.5.

    `dilution` is the third-light fraction used to build the BLEND class:
        F_obs = (F_eb + c) / (1 + c)
    which shrinks the depth into planet territory while leaving the V-shape and
    secondary eclipse intact -- exactly the confuser a real pipeline must catch.
    """
    period, t0, aRs, u = _draw_common(rng, t, (0.5, 10.0))
    rp = rng.uniform(0.15, 0.45)                      # stellar companion
    b = rng.uniform(0.0, 0.7)
    if b > aRs:
        b = rng.uniform(0.0, 0.7) * aRs

    primary = transit_model(t, period, t0, rp, aRs, b, u)
    d1 = 1.0 - primary.min()

    # secondary eclipse half a period later, shallower by the surface
    # brightness ratio of the two stars
    sb_ratio = rng.uniform(0.15, 0.9)
    secondary = transit_model(t, period, t0 + period / 2.0, rp, aRs, b, u)
    flux = primary - (1.0 - secondary) * sb_ratio

    if dilution > 0:
        flux = (flux + dilution) / (1.0 + dilution)
        d1 = d1 / (1.0 + dilution)

    truth = dict(period=period, t0=t0, rp=rp, aRs=aRs, b=b,
                 depth=d1, sb_ratio=sb_ratio, dilution=dilution,
                 duration=transit_duration(period, rp, aRs, b))
    return flux, truth


def sim_variable(t, rng):
    """Starspot rotation / pulsation: smooth, sinusoidal, no sharp ingress."""
    period = rng.uniform(0.4, 12.0)
    amp = rng.uniform(2e-3, 3e-2)
    phase = rng.uniform(0, 2 * np.pi)
    flux = 1.0 + amp * np.sin(2 * np.pi * t / period + phase)
    # a harmonic makes the shape non-sinusoidal, as real spot patterns are
    flux += amp * rng.uniform(0.1, 0.5) * np.sin(4 * np.pi * t / period + phase)
    truth = dict(period=period, amplitude=amp, depth=np.nan, duration=np.nan)
    return flux, truth


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_sample(label: str, rng: np.random.Generator,
                    cadence_min: float = CADENCE_MIN,
                    n_days: float = SECTOR_DAYS) -> dict:
    """Generate one labelled light curve with its ground-truth parameters.

    `cadence_min` and `n_days` exist so the synthetic set can be matched to
    whatever real data it will be compared against.  This matters more than it
    sounds: the vetting features are strongly cadence-dependent.  At TESS's
    2-minute cadence a 3-hour transit is sampled 90 times and its ingress is
    well resolved; at Kepler's 29-minute long cadence it is sampled 6 times and
    the ingress is a single point.  Comparing a model trained at one cadence
    against data taken at another measures the sampling mismatch, not the
    model.
    """
    t = make_time_grid(n_days=n_days, cadence_min=cadence_min)

    # noise level: bright stars ~200 ppm, faint ones ~3000 ppm per 2-min cadence
    white_ppm = float(10 ** rng.uniform(np.log10(200), np.log10(3000)))
    red_ppm = white_ppm * rng.uniform(0.3, 1.5)

    if label == TRANSIT:
        signal, truth = sim_transit(t, rng)
    elif label == ECLIPSE:
        signal, truth = sim_eclipse(t, rng, dilution=0.0)
    elif label == BLEND:
        # third light 5x-60x the EB's own flux -> deep eclipse becomes shallow
        signal, truth = sim_eclipse(t, rng, dilution=rng.uniform(5.0, 60.0))
    elif label == VARIABLE:
        signal, truth = sim_variable(t, rng)
    elif label == NOISE:
        signal, truth = np.ones_like(t), dict(depth=np.nan, duration=np.nan)
    else:
        raise ValueError(f"unknown label: {label!r}")

    flux = signal + make_noise(t, rng, white_ppm, red_ppm)
    flux_err = np.full_like(t, white_ppm * 1e-6)

    return dict(time=t, flux=flux, flux_err=flux_err, label=label,
                white_ppm=white_ppm, red_ppm=red_ppm, truth=truth)
