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
# Limb darkening
# --------------------------------------------------------------------------- #
def draw_limb_darkening(rng: np.random.Generator) -> tuple[float, float]:
    """Draw a physically valid quadratic limb-darkening pair.

    u1 and u2 used to be drawn from independent uniform ranges
    (u1 ~ U(0.2, 0.6), u2 ~ U(0.1, 0.4)). That is wrong in two ways that both
    push injected signals away from real ones:

    * It treats the two coefficients as independent. They are not -- across the
      FGKM range real stars occupy a narrow correlated band, and u2 falls as u1
      rises. Sampling the product space fills in combinations no star occupies.
    * Its support is a rectangle, but the physically valid region is the
      triangle u1 + u2 < 1, u1 > 0, u1 + 2*u2 > 0. A rectangle both admits
      unphysical corners and excludes legitimate ones.

    Drawn instead in the Kipping (2013) (q1, q2) basis that `fit.py` samples in,
    so injected signals and the model fitted to them share one set of
    assumptions rather than disagreeing by construction. The sum u1 + u2
    (= sqrt(q1)) is the well-constrained direction, so it is drawn from the
    range real FGKM hosts span in the optical; q2 spreads the pair across the
    valid triangle.
    """
    from .fit import q_to_u

    u_sum = rng.uniform(0.35, 0.85)          # FGKM optical, Claret-grid range
    q1 = u_sum ** 2
    q2 = rng.uniform(0.25, 0.75)             # avoids the degenerate corners
    return q_to_u(q1, q2)


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
    u = draw_limb_darkening(rng)
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

# --------------------------------------------------------------------------- #
# Injection into real stellar backgrounds
# --------------------------------------------------------------------------- #
# A controlled experiment showed the classifier's synthetic-to-real accuracy gap
# is not explained by cadence: regenerating synthetic data at the target cadence
# left the gap unchanged (0.565 vs 0.568), 82% of features stayed cadence-robust
# on synthetic data, and the median real/synthetic feature-separation ratio at
# matched cadence was 0.047.  The noise model above -- white noise plus four
# low-frequency sinusoids -- is simply too clean next to real instrumental
# systematics and real stellar variability, and no amount of amplitude tuning
# closes a 20x gap.
#
# So rather than simulate a realistic noise floor, borrow one: inject
# known-truth signals onto real quiet stars.  This is how injection-recovery
# testing is done in the exoplanet literature, and it keeps exact ground truth.

MIN_TRANSIT_COVERAGE = 0.6      # fraction of the in-transit window that must
                                # actually contain data points
MAX_T0_REDRAWS = 12


def _event_coverage(t: np.ndarray, truth: dict) -> float:
    """Fraction of the injected in-transit windows that actually holds samples.

    Real light curves have downlink gaps and flare-clipped stretches.  If an
    injected transit lands inside one, the example carries a label for a signal
    largely absent from the data -- a malformed training row.

    Coverage is computed from the known ephemeris rather than from the model
    array: every predicted event window is compared against how many samples
    really fall inside it.  (Thresholding the model array instead looks
    reasonable but silently measures the span from the first event to the last,
    which reports ~0.01 for a perfectly sampled signal.)
    """
    period = truth.get("period")
    duration = truth.get("duration")
    t0 = truth.get("t0")
    if not all(np.isfinite(v) for v in (period, duration, t0) if v is not None):
        return 0.0
    if period is None or duration is None or t0 is None or period <= 0 or duration <= 0:
        return 0.0

    cadence = float(np.median(np.diff(t)))
    if cadence <= 0:
        return 0.0

    lo, hi = float(t.min()), float(t.max())
    k0 = int(np.floor((lo - t0) / period)) - 1
    k1 = int(np.ceil((hi - t0) / period)) + 1

    got = expected = 0.0
    for k in range(k0, k1 + 1):
        centre = t0 + k * period
        a, b = centre - duration / 2.0, centre + duration / 2.0
        if b < lo or a > hi:
            continue                      # event falls outside the observation
        # clip to the observed span so partial events at the edges are fair
        a_c, b_c = max(a, lo), min(b, hi)
        expected += (b_c - a_c) / cadence
        got += float(np.count_nonzero((t >= a_c) & (t <= b_c)))

    if expected < 1.0:
        return 0.0
    return float(min(got / expected, 1.0))


def inject_into_real(baseline_time, baseline_flux, baseline_flux_err,
                     label: str, rng: np.random.Generator,
                     min_coverage: float = MIN_TRANSIT_COVERAGE) -> dict:
    """Inject a known-truth signal onto a real, signal-free light curve.

    Drop-in alternative to `generate_sample` for TRANSIT / ECLIPSE / BLEND,
    returning the same dict schema.

    `baseline_flux` MUST be raw -- neither normalised nor detrended.  The
    combined array is returned raw too, so the caller passes it through the same
    `prepare()` that live queries use.  Injecting onto already-cleaned flux
    would train the model on data whose messy parts were removed by a process
    inference does not replicate, which silently inflates accuracy.

    The signal is applied multiplicatively: a transit blocks a fraction of the
    star's light, so it scales with the real flux including its variations,
    rather than being added as a fixed offset.
    """
    t = np.asarray(baseline_time, dtype=float)
    f = np.asarray(baseline_flux, dtype=float)
    e = (np.asarray(baseline_flux_err, dtype=float)
         if baseline_flux_err is not None and np.size(baseline_flux_err) else None)

    dilution = 0.0
    for attempt in range(MAX_T0_REDRAWS):
        if label == TRANSIT:
            model, truth = sim_transit(t, rng)
        elif label == ECLIPSE:
            model, truth = sim_eclipse(t, rng, dilution=0.0)
        elif label == BLEND:
            # Third light from a neighbour in the aperture. Diluting the
            # multiplicative model is algebraically identical to diluting the
            # combined array -- (F*M + c*F)/(1+c) == F*(M+c)/(1+c) -- but stays
            # dimensionally sound, since c is a flux ratio and F is raw counts.
            dilution = float(rng.uniform(5.0, 60.0))
            model, truth = sim_eclipse(t, rng, dilution=dilution)
        else:
            raise ValueError(
                f"inject_into_real handles TRANSIT/ECLIPSE/BLEND, not {label!r}; "
                "VARIABLE and NOISE are sourced from real data directly")

        if _event_coverage(t, truth) >= min_coverage:
            break
        # otherwise the event fell in a gap -- redraw the ephemeris
    else:
        truth["low_coverage"] = True

    flux = f * model

    return dict(time=t, flux=flux, flux_err=e, label=label,
                white_ppm=np.nan, red_ppm=np.nan,
                truth={**truth, "injected_on_real": True,
                       "dilution": dilution,
                       "coverage": _event_coverage(t, truth)})
