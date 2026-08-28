"""Request and response models for the analysis API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """One of `tic`, `cached` or `simulate` selects the data source."""
    tic: str | None = Field(None, description="TIC ID to download from MAST, e.g. '261136679'")
    cached: str | None = Field(None, description="Identifier of a locally cached real target")
    simulate: Literal["transit", "eclipse", "blend", "variable", "noise"] | None = Field(
        None, description="Generate a synthetic light curve of this class")
    seed: int = Field(42, description="Random seed, simulated sources only")
    run_mcmc: bool = Field(True, description="Sample posteriors for uncertainties (slower)")
    multi_detrend: bool = Field(True, description="Try biweight and lowess, keep the higher SDE")


class TargetInfo(BaseModel):
    id: str
    name: str
    source: Literal["cached", "mast", "simulated"]
    n_points: int
    baseline_days: float
    note: str | None = None
    published_period: float | None = None
    published_rp_rs: float | None = None
    truth_label: str | None = None


class Detection(BaseModel):
    period_days: float
    t0: float
    duration_hours: float
    depth_ppm: float
    sde: float
    snr: float
    n_transits: int


class Classification(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    drivers: list[tuple[str, float]] = []


class Fit(BaseModel):
    period_days: float
    period_err: float | None
    depth_geometric_ppm: float
    depth_geometric_err_ppm: float | None
    depth_observed_ppm: float
    depth_observed_err_ppm: float | None
    duration_hours: float
    duration_err_hours: float | None
    rp_over_rs: float
    rp_over_rs_err: float | None
    impact_param: float
    a_over_rs: float
    chi2_reduced: float
    reliable: bool
    warning: str | None = None


class XY(BaseModel):
    x: list[float]
    y: list[float]
    e: list[float] | None = None


class Series(BaseModel):
    raw: XY
    trend: XY
    detrended: XY
    periodogram: XY
    fold: XY
    fold_binned: XY
    model: XY | None = None
    fold_2p: XY
    odd: XY | None = None
    even: XY | None = None


class AnalyzeResponse(BaseModel):
    target: TargetInfo
    detected: bool
    message: str = ""
    detection: Detection | None = None
    classification: Classification | None = None
    fit: Fit | None = None
    features: dict[str, float] | None = None
    series: Series | None = None
    detrend_method: str = "biweight"
    elapsed_seconds: float = 0.0
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    classes: list[str]
    n_features: int
    cached_targets: int
    warm_entries: int
