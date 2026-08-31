"""Invariants the pipeline must hold. Run with: .venv/bin/python -m pytest -q"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.config import BLEND, ECLIPSE, NOISE, TRANSIT, VARIABLE
from exodet.features import fit_trapezoid, implied_density, odd_even_test
from exodet.preprocess import in_transit_mask, prepare
from exodet.search import bin_lightcurve, fold, run_bls
from exodet.simulate import (a_over_rs, generate_sample, make_time_grid,
                             transit_duration, transit_model)


# --- simulator --------------------------------------------------------------
@pytest.mark.parametrize("label", [TRANSIT, ECLIPSE, BLEND, VARIABLE, NOISE])
def test_sample_is_finite(label):
    s = generate_sample(label, np.random.default_rng(0))
    assert np.all(np.isfinite(s["flux"]))
    assert np.all(np.isfinite(s["time"]))
    assert s["time"].size > 1000


def test_depth_regimes_are_physical():
    """Planets stay shallow, eclipsing binaries go deep, blends land between."""
    rng = np.random.default_rng(4)
    d = {k: [generate_sample(k, rng)["truth"]["depth"] for _ in range(25)]
         for k in (TRANSIT, ECLIPSE, BLEND)}
    assert max(d[TRANSIT]) < 0.03          # < 3%: planet regime
    assert min(d[ECLIPSE]) > 0.02          # > 2%: stellar companion
    assert np.median(d[BLEND]) < np.median(d[ECLIPSE])   # dilution shrinks it


def test_time_grid_has_downlink_gap():
    t = make_time_grid()
    gaps = np.diff(t)
    assert gaps.max() > 0.5                # the mid-sector gap is present


# --- geometry ---------------------------------------------------------------
def test_a_over_rs_matches_solar_analogue():
    """Earth at 1 AU around the Sun: a/R* should be ~215."""
    aRs = a_over_rs(365.25, 1.408)
    assert 200 < aRs < 230


def test_duration_shrinks_with_impact_parameter():
    aRs = a_over_rs(5.0, 1.408)
    central = transit_duration(5.0, 0.1, aRs, 0.0)
    grazing = transit_duration(5.0, 0.1, aRs, 0.9)
    assert grazing < central


# --- search -----------------------------------------------------------------
def test_bls_recovers_injected_period():
    rng = np.random.default_rng(7)
    s = generate_sample(TRANSIT, rng)
    t, f, fe, _ = prepare(s["time"], s["flux"], s["flux_err"])
    det, _, _ = run_bls(t, f, fe)
    ratio = det.period / s["truth"]["period"]
    # accept the true period or its 2x / 0.5x harmonic
    err = min(abs(ratio - 1), abs(ratio - 2) / 2, abs(ratio - 0.5) * 2)
    assert err < 0.02, f"period off by {err:.1%} (got {det.period:.4f})"


def test_binning_preserves_the_signal():
    """The speed optimisation must not change what we detect."""
    rng = np.random.default_rng(7)
    s = generate_sample(TRANSIT, rng)
    t, f, fe, _ = prepare(s["time"], s["flux"], s["flux_err"])
    full, _, _ = run_bls(t, f, fe, bin_minutes=0)
    fast, _, _ = run_bls(t, f, fe, bin_minutes=10)
    assert abs(fast.period / full.period - 1) < 0.01
    assert abs(fast.sde - full.sde) / full.sde < 0.15


def test_pure_noise_rarely_triggers():
    """False alarm rate on signal-free data must stay low."""
    rng = np.random.default_rng(99)
    hits = 0
    for _ in range(8):
        s = generate_sample(NOISE, rng)
        t, f, fe, _ = prepare(s["time"], s["flux"], s["flux_err"])
        det, _, _ = run_bls(t, f, fe)
        hits += det.detected
    assert hits <= 4, f"{hits}/8 false alarms on pure noise"


def test_fold_covers_unit_phase():
    t = np.linspace(0, 10, 1000)
    ph, _ = fold(t, np.ones_like(t), 2.0, 0.0)
    assert -0.5 <= ph.min() and ph.max() < 0.5
    assert np.all(np.diff(ph) >= 0)        # sorted by phase


# --- vetting features -------------------------------------------------------
def test_trapezoid_separates_box_from_v_shape():
    dt = np.linspace(-0.2, 0.2, 500)
    box = np.where(np.abs(dt) < 0.05, 0.99, 1.0)
    v = 1.0 - 0.01 * np.clip(1 - np.abs(dt) / 0.05, 0, 1)
    r_box = fit_trapezoid(dt, box, 0.01, 0.1)["ratio"]
    r_v = fit_trapezoid(dt, v, 0.01, 0.1)["ratio"]
    assert r_box > r_v, f"box ratio {r_box:.2f} should exceed V ratio {r_v:.2f}"


def test_odd_even_flags_alternating_depths():
    """Alternating depths -- an EB at half its period -- must be detected."""
    t = np.arange(0, 20, 2 / 1440)
    P, t0, dur = 2.0, 0.5, 0.1
    f = np.ones_like(t)
    epoch = np.round((t - t0) / P).astype(int)
    ph = (t - t0) - epoch * P
    intr = np.abs(ph) < dur / 2
    f[intr & (epoch % 2 == 0)] -= 0.010
    f[intr & (epoch % 2 == 1)] -= 0.005    # every other eclipse is shallower
    sigma, frac = odd_even_test(t, f, P, t0, dur)
    assert sigma > 10 and frac > 0.3


def test_implied_density_is_sane_for_a_real_transit():
    """A physical transit implies a stellar density near solar."""
    P, aRs = 5.0, a_over_rs(5.0, 1.408)
    dur = transit_duration(P, 0.1, aRs, 0.0)
    rho = implied_density(P, dur, 0.01)
    assert 0.2 < rho < 10.0, f"rho = {rho:.3f} g/cm3 is unphysical"


def test_in_transit_mask_selects_the_right_fraction():
    t = np.arange(0, 20, 2 / 1440)
    m = in_transit_mask(t, period=2.0, t0=0.5, duration=0.1, n_durations=1.0)
    assert 0.02 < m.mean() < 0.08          # ~0.1/2.0 = 5% of the time


# --- preprocessing ----------------------------------------------------------
def test_detrending_preserves_transit_depth():
    """The filter must not eat the signal it is meant to reveal."""
    t = make_time_grid()
    aRs = a_over_rs(4.0, 1.408)
    truth_depth = 0.01
    f = transit_model(t, 4.0, 2.0, np.sqrt(truth_depth), aRs, 0.2, (0.4, 0.2))
    f = f + np.random.default_rng(0).normal(0, 2e-4, t.size)
    tt, ff, _, _ = prepare(t, f, None)
    observed = 1.0 - np.percentile(ff, 0.4)
    assert observed > 0.6 * truth_depth, "detrending suppressed the transit"


def test_masked_detrend_keeps_in_transit_points():
    """Regression: wotan propagates NaN into the trend instead of interpolating.

    Dividing by that NaN trend deleted every masked (in-transit) point, which
    dropped Pi Men c's detection from SDE 21 to 6.7.  The trend must be bridged
    across masked gaps.
    """
    from exodet.preprocess import detrend
    t = np.linspace(0, 10, 2000)
    f = np.ones_like(t) + 1e-3 * np.sin(2 * np.pi * t / 5)
    mask = (t > 4.0) & (t < 4.3)
    flat, trend = detrend(t, f, mask=mask)
    assert np.isfinite(trend).all(), "trend still has NaN gaps"
    assert np.isfinite(flat[mask]).all(), "masked points were destroyed"


def test_pipeline_second_pass_preserves_detection():
    """The transit-masked re-detrend must not weaken the signal it refines."""
    from exodet.pipeline import analyze
    rng = np.random.default_rng(7)
    s = generate_sample(TRANSIT, rng)
    res = analyze(s["time"], s["flux"], s["flux_err"], model=None,
                  do_fit=False, iterate_detrend=True)
    assert res.detected, f"second pass killed the detection (SDE {res.detection.sde:.1f})"


def test_fit_reliability_flag_catches_bad_fits():
    """chi2_red must separate trustworthy fits from contaminated ones."""
    from exodet.fit import FitResult
    good = FitResult(period=1, period_err=0, t0=0, t0_err=0, depth=0,
                     depth_err=0, depth_obs=0, depth_obs_err=0, duration=0,
                     duration_err=0, rp=0, rp_err=0, aRs=10, b=0,
                     u1=0.4, u2=0.2,
                     chi2_red=1.7, beta_red_noise=1.0, converged=True)
    bad = FitResult(period=1, period_err=0, t0=0, t0_err=0, depth=0,
                    depth_err=0, depth_obs=0, depth_obs_err=0, duration=0,
                    duration_err=0, rp=0, rp_err=0, aRs=10, b=0,
                    u1=0.4, u2=0.2,
                    chi2_red=16.9, beta_red_noise=1.0, converged=True)
    assert good.reliable and not bad.reliable


def test_multi_detrend_picks_the_better_filter():
    """Given several filters, analyze() must keep the highest-SDE result."""
    from exodet.pipeline import analyze
    rng = np.random.default_rng(7)
    s = generate_sample(TRANSIT, rng)
    one = analyze(s["time"], s["flux"], s["flux_err"], do_fit=False,
                  detrend_methods=("biweight",))
    both = analyze(s["time"], s["flux"], s["flux_err"], do_fit=False,
                   detrend_methods=("biweight", "lowess"))
    assert both.detection.sde >= one.detection.sde - 1e-6
    assert both.detrend_method in ("biweight", "lowess")


def test_hot_planets_get_a_small_secondary_eclipse():
    """Regression: real ultra-hot Jupiters show a secondary eclipse.

    Assuming they do not made the classifier read WASP-121 b's genuine ~500 ppm
    secondary as evidence of a blend. Planetary secondaries must exist but stay
    far shallower than the 0.15-0.9 ratios used for eclipsing binaries.
    """
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(60):
        _, truth = __import__("exodet.simulate", fromlist=["sim_transit"]).sim_transit(
            make_time_grid(), rng)
        if truth["period"] < 4.0:
            ratios.append(truth["sec_ratio"])
    assert ratios, "no short-period planets drawn"
    assert max(ratios) <= 0.06, f"planetary secondary too deep: {max(ratios):.3f}"
    assert max(ratios) > 0.0, "no planetary secondary eclipses generated at all"
    # must stay clearly below the EB regime
    assert max(ratios) < 0.15


def test_transit_depths_cover_real_hot_jupiters():
    """WASP-121 b is 14000 ppm; the transit class must reach that."""
    rng = np.random.default_rng(1)
    depths = [generate_sample(TRANSIT, rng)["truth"]["depth"] for _ in range(80)]
    assert max(depths) > 0.014, (
        f"deepest simulated planet is {max(depths)*1e6:.0f} ppm, "
        "below real hot Jupiters")


# --------------------------------------------------------------------------- #
# classifier / fit cross-check
# --------------------------------------------------------------------------- #
def _result(label, chi2, converged=True):
    """A Result carrying just enough of a fit for the cross-check to judge."""
    from exodet.pipeline import Result, _cross_check

    class _Fit:
        CHI2_RELIABLE_MAX = 3.0

        def __init__(self, chi2, converged):
            self.chi2_red, self.converged = chi2, converged

        @property
        def reliable(self):
            return bool(self.converged and np.isfinite(self.chi2_red)
                        and self.chi2_red < self.CHI2_RELIABLE_MAX)

    r = Result(detected=True, detection=None, label=label, confidence=0.54,
               probabilities=None, features=None, fit=_Fit(chi2, converged))
    r.caution_flag, r.caution_reason = _cross_check(r)
    return r


def test_caution_fires_when_a_geometric_label_meets_an_unreliable_fit():
    """The AU Mic b shape: confident 'eclipse', reduced chi-square ~17."""
    r = _result("eclipse", 16.7)
    assert r.caution_flag
    assert "16.7" in r.caution_reason and "eclipse" in r.caution_reason


def test_caution_stays_quiet_on_a_good_fit():
    assert not _result("transit", 1.2).caution_flag


def test_caution_ignores_variable_which_makes_no_geometric_claim():
    """A poor *transit* fit is consistent with 'variable', not evidence
    against it, so flagging it would be pure noise."""
    assert not _result("variable", 16.7).caution_flag


def test_caution_stays_quiet_when_the_fit_never_converged():
    """A failed fit is missing evidence, not contradicting evidence."""
    assert not _result("eclipse", float("nan"), converged=False).caution_flag


def test_limb_darkening_reparameterisation_is_physical_by_construction():
    """Kipping (2013): the unit square in (q1, q2) IS the valid triangle.

    This is the property the whole reparameterisation rests on -- if it does
    not hold, the sampler is free to wander into unphysical limb darkening
    exactly as it was before.
    """
    from exodet.fit import q_to_u, u_to_q

    rng = np.random.default_rng(0)
    q = rng.uniform(0, 1, size=(4000, 2))
    u1, u2 = np.array([q_to_u(a, b) for a, b in q]).T
    assert np.all(u1 >= 0), "negative u1 escaped the parameterisation"
    assert np.all(u1 + u2 <= 1.0 + 1e-12), "u1 + u2 exceeded 1"
    assert np.all(u1 + 2 * u2 >= -1e-12), "brightness rose toward the limb"

    # round-trip, so seeding the sampler from a (u1, u2) guess lands where meant
    for a, b in [(0.4, 0.2), (0.6, 0.1), (0.3, 0.35)]:
        got = q_to_u(*u_to_q(a, b))
        assert np.allclose(got, (a, b), atol=1e-9)


def test_injected_limb_darkening_is_physical():
    """The injection generator and the fitter must not disagree by
    construction -- the old independent-uniform draws admitted pairs the
    fitter's prior gives essentially zero weight."""
    from exodet.simulate import draw_limb_darkening

    rng = np.random.default_rng(1)
    us = np.array([draw_limb_darkening(rng) for _ in range(2000)])
    u1, u2 = us[:, 0], us[:, 1]
    assert np.all(u1 > 0) and np.all(u1 + u2 < 1) and np.all(u1 + 2 * u2 > 0)
