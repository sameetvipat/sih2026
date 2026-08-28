"""End-to-end analysis of a single light curve.

    raw flux -> clean/detrend -> BLS search -> vetting features
             -> classification (+ confidence) -> transit fit (+ uncertainties)

This is the one entry point the app and the batch runner both call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .classify import explain, predict_one
from .config import SDE_THRESHOLD, TRANSIT, BLEND
from .features import extract_features
from .fit import fit_transit
from .preprocess import in_transit_mask, prepare
from .search import Detection, bin_lightcurve, fold, run_bls


@dataclass
class Result:
    """Everything the pipeline learned about one light curve."""
    detected: bool
    detection: Detection | None
    label: str | None
    confidence: float | None
    probabilities: dict[str, float] | None
    features: dict[str, float] | None
    fit: Any = None
    drivers: list = field(default_factory=list)
    # arrays kept for plotting
    time: np.ndarray | None = None
    flux: np.ndarray | None = None
    trend: np.ndarray | None = None
    raw_time: np.ndarray | None = None
    raw_flux: np.ndarray | None = None
    detrend_method: str = "biweight"
    periods: np.ndarray | None = None
    power: np.ndarray | None = None
    message: str = ""

    def summary_lines(self) -> list[str]:
        if not self.detected:
            return [f"No significant periodic dip (SDE {self.detection.sde:.1f} "
                    f"< {SDE_THRESHOLD}). {self.message}".strip()]
        d = self.detection
        out = []
        if self.label is not None and self.confidence is not None:
            out.append(f"Classification : {self.label}  "
                       f"(confidence {self.confidence:.1%})")
        else:
            out.append("Classification : (no classifier loaded)")
        out += [
            f"Period         : {d.period:.5f} d",
            f"Depth (BLS)    : {d.depth * 1e6:.0f} ppm",
            f"Duration       : {d.duration * 24:.2f} h",
            f"SDE / SNR      : {d.sde:.1f} / {d.snr:.1f}",
            f"Transits seen  : {d.n_transits}",
        ]
        if self.fit is not None and self.fit.converged:
            f = self.fit
            out += [
                "",
                "Fitted transit model (median +/- 1 sigma):",
                f"  Period    : {f.period:.5f} +/- {f.period_err:.5f} d",
                f"  Depth     : {f.depth * 1e6:.0f} +/- {f.depth_err * 1e6:.0f} ppm",
                f"  Duration  : {f.duration * 24:.3f} +/- {f.duration_err * 24:.3f} h",
                f"  Rp/R*     : {f.rp:.4f} +/- {f.rp_err:.4f}",
                f"  chi2_red  : {f.chi2_red:.2f}",
            ]
            if not f.reliable:
                out.append(
                    f"  WARNING: chi2_red = {f.chi2_red:.1f} exceeds "
                    f"{f.CHI2_RELIABLE_MAX}; the transit model does not "
                    "describe the data (residual stellar variability or a "
                    "wrong period). Treat these parameters as unreliable.")
        return out


def analyze(time, flux, flux_err=None, model=None, calibrator=None,
            do_fit=True, run_mcmc=True, iterate_detrend=True,
            detrend_methods=("biweight",)) -> Result:
    """Run the full pipeline on one light curve.

    `detrend_methods` may name several wotan filters; each is tried and the one
    giving the highest SDE is kept.  That rescues heavily spotted stars, where
    the default biweight filter fails (see `preprocess.detrend`).  Note this
    costs one BLS run per method, and trying N filters and keeping the best
    inflates the effective false-alarm rate -- treat SDE from a multi-method
    run as slightly optimistic.
    """
    raw_time, raw_flux = np.asarray(time), np.asarray(flux)

    best = None
    for method in detrend_methods:
        try:
            t_, f_, e_, trend_ = prepare(raw_time, raw_flux, flux_err,
                                         method=method)
            if t_.size < 100:
                continue
            det_, periods_, power_ = run_bls(t_, f_, e_)
        except Exception:
            continue
        if best is None or det_.sde > best[0].sde:
            best = (det_, periods_, power_, t_, f_, e_, trend_, method)

    if best is None:
        return Result(False, None, None, None, None, None,
                      message="too few valid points after cleaning")

    det, periods, power, t, f, e, trend, used_method = best

    # Second pass: mask the transits we just found and re-detrend, so the
    # filter cannot bend down into the signal and suppress its depth.
    if iterate_detrend and det.detected:
        from .preprocess import detrend
        mask = in_transit_mask(t, det.period, det.t0, det.duration)
        if mask.any() and mask.mean() < 0.4:
            # f * trend reconstructs the pre-detrend flux on the surviving points
            f2, trend2 = detrend(t, f * trend, mask=mask,
                                 method=used_method)
            good = np.isfinite(f2)
            if good.sum() > 100:
                t, f, trend = t[good], f2[good], trend2[good]
                e = e[good] if e is not None else None
                det, periods, power = run_bls(t, f, e)

    result = Result(detected=det.detected, detection=det, label=None,
                    confidence=None, probabilities=None, features=None,
                    time=t, flux=f, trend=trend,
                    raw_time=raw_time, raw_flux=raw_flux,
                    detrend_method=used_method,
                    periods=periods, power=power)

    if not det.detected:
        result.message = "below detection threshold"
        return result

    result.features = extract_features(t, f, det)

    if model is not None:
        pred = predict_one(model, calibrator, result.features)
        result.label = pred["label"]
        result.confidence = pred["confidence"]
        result.probabilities = pred["probabilities"]
        result.drivers = explain(model, result.features)

    # Fit a transit model when the signal is transit-like.  Blends get fitted
    # too: the depth is still measurable, it just refers to a diluted source.
    if do_fit and (model is None or result.label in (TRANSIT, BLEND)):
        result.fit = fit_transit(t, f, e, det, run_mcmc=run_mcmc)

    return result


def phase_view(result: Result, n_bins: int = 200):
    """Folded light curve plus binned means, for plotting."""
    from .search import bin_phase
    d = result.detection
    phase, f = fold(result.time, result.flux, d.period, d.t0)
    centres, means, errs = bin_phase(phase, f, n_bins)
    return phase, f, centres, means, errs
