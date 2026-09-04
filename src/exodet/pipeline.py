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
from .config import SDE_THRESHOLD, TRANSIT, ECLIPSE, BLEND
from .features import extract_features
from .fit import fit_transit
from .preprocess import in_transit_mask, prepare
from .search import Detection, run_bls


@dataclass
# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
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
    # Set by _cross_check: the classifier and the fit disagree about whether
    # there is really an eclipsing signal here.
    caution_flag: bool = False
    caution_reason: str | None = None
    # Multi-detrend bookkeeping. `sde` is the raw statistic of the winning
    # trial; `sde_corrected` is what it is worth once the search over
    # detrenders is accounted for.
    n_detrend_trials: int = 1
    detrend_is_fallback: bool = False
    sde_corrected: float | None = None

    def summary_lines(self) -> list[str]:
        if not self.detected:
            return [f"No significant periodic dip (SDE {self.detection.sde:.1f} "
                    f"< {SDE_THRESHOLD}). {self.message}".strip()]
        d = self.detection
        out = []
        if self.label is not None and self.confidence is not None:
            out.append(f"Classification : {self.label}  "
                       f"(confidence {self.confidence:.1%})")
            if self.caution_flag:
                out.append(f"  CAUTION: {self.caution_reason}")
        else:
            out.append("Classification : (no classifier loaded)")
        out += [
            f"Period         : {d.period:.5f} d",
            f"Depth (BLS)    : {d.depth * 1e6:.0f} ppm",
            f"Duration       : {d.duration * 24:.2f} h",
            f"SDE / SNR      : {d.sde:.1f} / {d.snr:.1f}",
            f"Transits seen  : {d.n_transits}",
        ]
        if self.detrend_is_fallback and self.sde_corrected is not None:
            out.append(
                f"  note: this detection came from the fallback '"
                f"{self.detrend_method}' detrender, not the primary one. "
                f"Across {self.n_detrend_trials} trials the SDE is worth "
                f"{self.sde_corrected:.2f} after a look-elsewhere discount.")
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


# --------------------------------------------------------------------------- #
# End-to-end analysis
# --------------------------------------------------------------------------- #
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
    n_trials = 0
    for method in detrend_methods:
        try:
            t_, f_, e_, trend_ = prepare(raw_time, raw_flux, flux_err,
                                         method=method)
            if t_.size < 100:
                continue
            det_, periods_, power_ = run_bls(t_, f_, e_)
        except Exception:
            continue
        n_trials += 1                    # count only trials that actually ran
        if best is None or det_.sde > best[0].sde:
            best = (det_, periods_, power_, t_, f_, e_, trend_, method)

    if best is None:
        return Result(False, None, None, None, None, None,
                      message="too few valid points after cleaning")

    det, periods, power, t, f, e, trend, used_method = best
    primary_method = detrend_methods[0] if detrend_methods else used_method

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
                    n_detrend_trials=n_trials,
                    detrend_is_fallback=(used_method != primary_method),
                    sde_corrected=look_elsewhere_sde(det.sde, n_trials),
                    periods=periods, power=power)

    if not det.detected:
        result.message = "below detection threshold"
        return result

    result.features = extract_features(t, f, det, trend=trend)

    if model is not None:
        pred = predict_one(model, calibrator, result.features)
        result.label = pred["label"]
        result.confidence = pred["confidence"]
        result.probabilities = pred["probabilities"]
        result.drivers = explain(model, result.features)

    # Fit a model whenever the signal is eclipse-like in the broad sense.
    # Blends get fitted because the depth is still measurable, it just refers
    # to a diluted source; eclipses get fitted because their depth and duration
    # are real measurements too, and -- less obviously -- because the fit's
    # reduced chi-square is the ONLY evidence the pipeline has that a confident
    # classification might be wrong. Skipping the fit for eclipses meant the
    # class most often assigned to spotted stars was also the one class with no
    # reliability check attached, which is exactly how AU Mic b shipped a
    # confident "eclipse" label with nothing contradicting it.
    if do_fit and (model is None or result.label in (TRANSIT, ECLIPSE, BLEND)):
        result.fit = fit_transit(t, f, e, det, run_mcmc=run_mcmc)

    result.caution_flag, result.caution_reason = _cross_check(result)
    return result


# --------------------------------------------------------------------------- #
# Multiple-trials correction
# --------------------------------------------------------------------------- #
def look_elsewhere_sde(sde: float, n_trials: int) -> float:
    """Discount an SDE for the number of detrenders the search chose between.

    Trying two filters and keeping whichever gave the higher SDE is a search,
    not a measurement. The retained value is then the maximum of two draws, and
    comparing a maximum against a threshold calibrated for a single draw
    inflates the effective false-alarm rate -- roughly two-fold per extra trial
    in the tail. The pipeline documented this and did not correct it; this is
    the correction.

    The standard bounded form (Sidak) treats the trials as independent, maps the
    SDE to a Gaussian-equivalent single-trial tail probability, multiplies the
    false-alarm rate by the trial count, and maps back:

        p1  = 1 - Phi(SDE)
        pN  = 1 - (1 - p1)^N        ~= N * p1  in the tail
        SDE_corrected = Phi^-1(1 - pN)

    Two properties worth stating plainly, because the number is easy to
    over-read:

    * This is deliberately CONSERVATIVE -- an upper bound on the inflation, not
      an estimate of it. Independence is assumed and is false here: biweight and
      lowess are different filters run over the same photons and their SDEs are
      strongly correlated, so the true effective trial count is between 1 and 2,
      nearer 1. The real correction is therefore SMALLER than this one.
    * The Gaussian mapping is itself an approximation. BLS SDE is not exactly
      Gaussian-distributed under the null, so this is a calibrated-scale
      discount rather than an exact p-value.

    It is reported alongside the raw SDE rather than replacing it, and does not
    change the detection decision -- moving the threshold onto a corrected
    statistic would silently redefine every count in the existing dataset.
    """
    from scipy.stats import norm

    if not np.isfinite(sde) or n_trials <= 1:
        return float(sde)
    p1 = norm.sf(sde)
    if p1 <= 0:                       # SDE beyond float resolution of the tail
        return float(sde)
    p_n = -np.expm1(n_trials * np.log1p(-p1))    # 1-(1-p1)^N, stable in the tail
    return float(np.clip(norm.isf(p_n), 0.0, sde))


# Classes whose label asserts a specific geometric event, and so can be
# contradicted by a fit that does not describe the data. "variable" makes no
# such claim -- a poor transit fit is consistent with it, not evidence against
# it -- so flagging it would be noise.
_GEOMETRIC = (TRANSIT, ECLIPSE, BLEND)


# --------------------------------------------------------------------------- #
# Classifier / fit cross-check
# --------------------------------------------------------------------------- #
def _cross_check(result: "Result") -> tuple[bool, str | None]:
    """Reconcile the classifier's verdict against the fit's reliability.

    classify.py and fit.py reach their conclusions independently, which is
    deliberate -- a fit that agreed with the classifier by construction could
    not contradict it. But independent verdicts were also being *reported*
    independently, so a confident label and a fit saying "this model does not
    describe the data" would sit in the same response with nothing connecting
    them, and a reader would see only the label.

    AU Mic b is the worked example: a young star whose starspot amplitude is
    19x its transit depth. The classifier calls it "eclipse" with high
    confidence, the fit correctly reports reduced chi-square 16.9, and the
    output showed the label. The disagreement between the two subsystems is
    itself the finding, so it is now surfaced rather than left for the reader
    to notice.
    """
    if result.label not in _GEOMETRIC:
        return False, None
    fit = result.fit
    if fit is None or not fit.converged or not np.isfinite(fit.chi2_red):
        return False, None
    if fit.reliable:
        return False, None
    return True, (
        f"fit quality poor (reduced chi-square = {fit.chi2_red:.1f}, above the "
        f"{fit.CHI2_RELIABLE_MAX:.0f} reliability threshold) while the "
        f"classifier reports '{result.label}'"
        + (f" at {result.confidence:.0%} confidence" if result.confidence else "")
        + " -- the classification may be driven by non-transit stellar "
          "variability rather than a genuine eclipsing signal, and the fitted "
          "parameters should not be trusted")

