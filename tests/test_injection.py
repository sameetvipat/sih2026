"""Tests for real-background signal injection.

These cover the contract that `inject_into_real` must satisfy to be a safe
drop-in for `generate_sample`, plus the specific edge case where an injected
event lands on a real data gap or flare-clipped stretch.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.config import BLEND, ECLIPSE, NOISE, TRANSIT, VARIABLE
from exodet.preprocess import prepare
from exodet.simulate import _event_coverage, generate_sample, inject_into_real

CAD = 29.0 / 1440.0          # Kepler long cadence, in days


def continuous_baseline(days=33.0, level=50000.0, seed=0, noise=25.0):
    """A raw, un-normalised stand-in for a real quiet light curve."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, days, CAD)
    f = level * (1 + 3e-4 * np.sin(2 * np.pi * t / 7.0)) + rng.normal(0, noise, t.size)
    return t, f


def gapped_baseline(seed=0):
    """Baseline with a multi-day downlink gap in the middle."""
    rng = np.random.default_rng(seed)
    t = np.concatenate([np.arange(0, 14, CAD), np.arange(18, 33, CAD)])
    f = 50000.0 + rng.normal(0, 25.0, t.size)
    return t, f


# --- contract ---------------------------------------------------------------
@pytest.mark.parametrize("label", [TRANSIT, ECLIPSE, BLEND])
def test_schema_matches_generate_sample(label):
    """Must be a drop-in: same keys, so make_dataset can swap them freely."""
    t, f = continuous_baseline()
    got = inject_into_real(t, f, None, label, np.random.default_rng(1))
    ref = generate_sample(label, np.random.default_rng(1))
    assert set(ref).issubset(set(got)), f"missing keys: {set(ref) - set(got)}"
    assert got["label"] == label
    assert got["time"].size == got["flux"].size


def test_rejects_labels_it_cannot_inject():
    """VARIABLE and NOISE come from real data directly, not from injection."""
    t, f = continuous_baseline()
    for label in (VARIABLE, NOISE):
        with pytest.raises(ValueError, match="TRANSIT/ECLIPSE/BLEND"):
            inject_into_real(t, f, None, label, np.random.default_rng(0))


def test_output_stays_raw_and_unnormalised():
    """prepare() must run on the combined array, so injection must not clean it.

    Injecting onto already-detrended flux would train the model on data whose
    messy parts were removed by a process inference never repeats.
    """
    t, f = continuous_baseline(level=50000.0)
    got = inject_into_real(t, f, None, TRANSIT, np.random.default_rng(2))
    assert np.median(got["flux"]) > 1000, (
        "flux was normalised during injection; it must stay in raw counts")


def test_signal_is_multiplicative_not_additive():
    """A transit blocks a fraction of the light, so it scales with the flux."""
    t, f = continuous_baseline(level=50000.0, noise=0.0)
    rng_state = 7
    a = inject_into_real(t, f, None, TRANSIT, np.random.default_rng(rng_state))
    b = inject_into_real(t, f * 3.0, None, TRANSIT, np.random.default_rng(rng_state))
    # tripling the baseline must triple the injected curve exactly
    assert np.allclose(b["flux"], a["flux"] * 3.0, rtol=1e-9)


# --- depth regimes ----------------------------------------------------------
def test_injected_depths_land_in_the_right_regimes():
    t, f = continuous_baseline()
    rng = np.random.default_rng(3)
    depths = {}
    for label in (TRANSIT, ECLIPSE, BLEND):
        ds = []
        for _ in range(12):
            g = inject_into_real(t, f, None, label, rng)
            ds.append(1.0 - g["flux"].min() / np.median(g["flux"]))
        depths[label] = np.array(ds)
    assert depths[TRANSIT].max() < 0.05, "planets should stay shallow"
    assert depths[ECLIPSE].min() > 0.02, "eclipsing binaries should be deep"
    # dilution must push blends below their undiluted eclipse depth
    assert np.median(depths[BLEND]) < np.median(depths[ECLIPSE])


def test_blend_dilution_is_recorded():
    t, f = continuous_baseline()
    g = inject_into_real(t, f, None, BLEND, np.random.default_rng(4))
    assert g["truth"]["dilution"] > 0
    assert g["truth"]["injected_on_real"] is True


# --- the gap / flare edge case ---------------------------------------------
def test_coverage_is_one_on_a_continuous_baseline():
    t, f = continuous_baseline()
    g = inject_into_real(t, f, None, TRANSIT, np.random.default_rng(5))
    assert g["truth"]["coverage"] > 0.95


def test_events_landing_in_a_gap_are_redrawn():
    """Regression: an event inside a downlink gap is a malformed example.

    The ephemeris is redrawn until the in-transit windows are adequately
    sampled, so no training row claims a signal the data does not contain.
    """
    t, f = gapped_baseline()
    covs = [inject_into_real(t, f, None, TRANSIT,
                             np.random.default_rng(i))["truth"]["coverage"]
            for i in range(15)]
    assert min(covs) >= 0.6, f"worst coverage {min(covs):.2f} below threshold"


def test_coverage_uses_the_ephemeris_not_the_model_array():
    """Regression: thresholding the model array measured first-to-last event
    span and reported ~0.01 for a perfectly sampled signal."""
    t, f = continuous_baseline()
    truth = dict(period=5.0, t0=2.0, duration=0.15)
    assert _event_coverage(t, truth) > 0.95


def test_coverage_detects_a_genuinely_missing_event():
    """A transit sitting entirely inside a gap must score low."""
    t = np.concatenate([np.arange(0, 10, CAD), np.arange(20, 33, CAD)])
    truth = dict(period=100.0, t0=15.0, duration=0.15)   # single event, in gap
    assert _event_coverage(t, truth) < 0.2


def test_injected_curve_survives_the_real_preprocessing_path():
    """The whole point: the combined array must flow through prepare() intact."""
    t, f = gapped_baseline()
    g = inject_into_real(t, f, None, TRANSIT, np.random.default_rng(6))
    ct, cf, ce, trend = prepare(g["time"], g["flux"], g["flux_err"])
    assert ct.size > 500
    assert np.all(np.isfinite(cf))
    assert 0.9 < np.median(cf) < 1.1, "prepare() should normalise to ~1"


# --- leakage guards ---------------------------------------------------------
def test_baseline_splits_are_disjoint():
    """Regression: a baseline in both pools leaks the noise floor itself.

    The model could learn to recognise a specific star's systematics rather
    than generalisable signal structure, inflating the test score.
    """
    import pandas as pd
    path = "data/baselines/manifest.csv"
    if not os.path.exists(path):
        pytest.skip("baseline manifest not built yet")
    df = pd.read_csv(path)
    if "split" not in df:
        pytest.skip("splits not assigned yet (run scripts/split_baselines.py)")
    train = set(df[df.split == "train"]["target"])
    test = set(df[df.split == "test"]["target"])
    assert not (train & test), f"baselines in both splits: {sorted(train & test)[:5]}"
    assert len(test) > 0 and len(train) > 0
