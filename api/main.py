"""FastAPI service exposing the exoplanet detection pipeline.

    GET  /api/health    service + model status
    GET  /api/targets   what can be analysed
    POST /api/analyze   run the pipeline on one light curve
    POST /api/batch     run it on several, ranked by significance

The single-page frontend in web/ is served from /.
"""
from __future__ import annotations

import asyncio
import glob
from contextlib import asynccontextmanager
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet import classify                                    # noqa: E402
from exodet.config import CLASSES                              # noqa: E402
from exodet.features import FEATURE_NAMES                       # noqa: E402
from exodet.pipeline import analyze                             # noqa: E402
from exodet.search import bin_phase, fold                       # noqa: E402
from exodet.simulate import generate_sample, transit_model      # noqa: E402

from .schemas import (AnalyzeRequest, AnalyzeResponse, Classification,  # noqa: E402
                      Detection, Fit, HealthResponse, Series, TargetInfo, XY)

ROOT = os.path.join(os.path.dirname(__file__), "..")
WEB = os.path.join(ROOT, "web")
CACHE_DIR = os.path.join(ROOT, "data", "cache")

# published values for the cached real targets -- display only, never used by
# the pipeline itself
PUBLISHED = {
    "TIC 261136679": dict(name="Pi Men c", period=6.26790, rp_rs=0.01703,
                          note="shallow ~300 ppm transit, quiet host"),
    "TIC 22529346":  dict(name="WASP-121 b", period=1.27493, rp_rs=0.12355,
                          note="deep hot Jupiter"),
    "TIC 441420236": dict(name="AU Mic b", period=8.46321, rp_rs=0.05140,
                          note="young, heavily spotted host"),
    "TIC 100100827": dict(name="WASP-18 b", period=0.94145, rp_rs=0.09716,
                          note="very deep, short period"),
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the classifier, then warm the result cache in the background."""
    global _model, _calibrator
    try:
        _model, _calibrator = classify.load(os.path.join(ROOT, "models",
                                                         "classifier.joblib"))
        print("[api] classifier loaded")
    except Exception as exc:
        print(f"[api] no classifier ({exc}); detection + fitting only")

    # Warming means the first demo click is instant and does not depend on
    # MAST being reachable.
    asyncio.get_event_loop().run_in_executor(_pool, _warm)
    yield
    _pool.shutdown(wait=False)


app = FastAPI(
    lifespan=lifespan,
    title="Exoplanet Transit Detection API",
    description="Detects periodic dips in TESS light curves, classifies them "
                "into transit / eclipse / blend / variable / noise, and fits "
                "transit parameters with MCMC uncertainties.",
    version="0.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_model = _calibrator = None
_cache: dict[str, AnalyzeResponse] = {}
_pool = ThreadPoolExecutor(max_workers=4)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
# Full float64 repr costs ~17 characters per number and the browser cannot
# draw anything like that precision.  8 significant digits still resolves a
# 300 ppm transit in normalised flux and roughly halves the payload.
def _sig(a, digits=8):
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.round(a, digits).tolist()


def _decimate(x, y, n=2500, e=None):
    """Bin down to ~n points for display. Plots cannot resolve more anyway."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if e is not None:
        e = np.asarray(e, float)[good]
    if x.size <= n:
        return XY(x=_sig(x), y=_sig(y), e=_sig(e) if e is not None else None)
    edges = np.linspace(x.min(), x.max(), n + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n - 1)
    cnt = np.bincount(idx, minlength=n)
    ok = cnt > 0
    bx = np.bincount(idx, weights=x, minlength=n)[ok] / cnt[ok]
    by = np.bincount(idx, weights=y, minlength=n)[ok] / cnt[ok]
    return XY(x=_sig(bx), y=_sig(by), e=None)


def _load_cached(ident: str):
    """Read a locally cached real light curve by TIC id or filename stem."""
    stem = ident.replace(" ", "_")
    if not stem.startswith("TIC"):
        stem = f"TIC_{stem}"
    path = os.path.join(CACHE_DIR, f"{stem}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return (np.asarray(d["time"], float), np.asarray(d["flux"], float),
            np.asarray(d["flux_err"], float), str(d["tic"]))


def _series_from(res) -> Series:
    """Everything the frontend needs to draw, at display resolution."""
    d = res.detection
    raw = _decimate(res.raw_time, res.raw_flux / np.nanmedian(res.raw_flux))
    trend = _decimate(res.time, res.trend) if res.trend is not None else XY(x=[], y=[])
    det = _decimate(res.time, res.flux)
    pg = _decimate(res.periods, res.power, n=2000)

    phase, f = fold(res.time, res.flux, d.period, d.t0)
    centres, means, errs = bin_phase(phase, f, 200)
    bm = np.isfinite(means)
    fold_binned = XY(x=_sig(centres[bm]), y=_sig(means[bm]),
                     e=_sig(np.nan_to_num(errs[bm], nan=0.0)))

    # model overlay on the folded view
    model = None
    if res.fit is not None and res.fit.converged:
        fr = res.fit
        pm = np.linspace(-0.5, 0.5, 900)
        tm = fr.t0 + pm * fr.period
        # Limb darkening is now fitted, so the overlay uses the fit's own
        # coefficients rather than a Sun-like stand-in. Plotting a curve with
        # different limb darkening than the fit that produced its depth makes
        # the overlay disagree with the number printed beside it.
        fm = transit_model(tm, fr.period, fr.t0, fr.rp, fr.aRs, fr.b,
                           (fr.u1, fr.u2))
        model = XY(x=_sig(pm), y=_sig(fm))

    p2, f2 = fold(res.time, res.flux, 2 * d.period, d.t0)
    c2, m2, _ = bin_phase(p2, f2, 300)
    ok2 = np.isfinite(m2)

    # odd vs even transits, in hours from mid-transit
    epoch = np.round((res.time - d.t0) / d.period).astype(int)
    dt = (res.time - d.t0) - epoch * d.period
    near = np.abs(dt) < 3 * d.duration
    odd = even = None
    for parity, slot in ((0, "even"), (1, "odd")):
        m = near & (epoch % 2 == parity)
        if m.sum() > 5:
            xs, ys, _ = bin_phase(dt[m] / (6 * d.duration), res.flux[m], 40)
            keep = np.isfinite(ys)
            xy = XY(x=_sig(xs[keep] * 6 * d.duration * 24),
                    y=_sig(ys[keep]))
            if slot == "odd":
                odd = xy
            else:
                even = xy

    return Series(
        raw=raw, trend=trend, detrended=det, periodogram=pg,
        fold=_decimate(phase, f, n=2500), fold_binned=fold_binned,
        model=model,
        fold_2p=XY(x=_sig(c2[ok2]), y=_sig(m2[ok2])),
        odd=odd, even=even,
    )


def _to_response(res, target: TargetInfo, elapsed: float) -> AnalyzeResponse:
    if not res.detected:
        msg = res.message or "no significant periodic dip"
        if res.detection is not None:
            msg = (f"No significant periodic dip: SDE {res.detection.sde:.1f} "
                   f"is below the detection threshold of 7.0.")
        return AnalyzeResponse(target=target, detected=False, message=msg,
                               detrend_method=res.detrend_method,
                               elapsed_seconds=elapsed)

    d = res.detection
    detection = Detection(period_days=d.period, t0=d.t0,
                          duration_hours=d.duration * 24, depth_ppm=d.depth * 1e6,
                          sde=d.sde, snr=d.snr, n_transits=d.n_transits)

    classification = None
    if res.label is not None:
        classification = Classification(
            label=res.label, confidence=res.confidence or 0.0,
            probabilities=res.probabilities or {},
            drivers=[(k, float(v)) for k, v in (res.drivers or [])])

    fit = None
    if res.fit is not None and res.fit.converged:
        fr = res.fit
        warn = None
        if not fr.reliable:
            warn = (f"Reduced chi-square is {fr.chi2_red:.1f}, above the "
                    f"{fr.CHI2_RELIABLE_MAX} threshold: the transit model does "
                    "not describe the data. This usually means residual "
                    "stellar variability or a wrong period. Treat these "
                    "parameters as unreliable.")
        nn = lambda v: None if v is None or not np.isfinite(v) else float(v)
        fit = Fit(
            period_days=fr.period, period_err=nn(fr.period_err),
            depth_geometric_ppm=fr.depth * 1e6,
            depth_geometric_err_ppm=nn(fr.depth_err * 1e6 if np.isfinite(fr.depth_err) else None),
            depth_observed_ppm=fr.depth_obs * 1e6,
            depth_observed_err_ppm=nn(fr.depth_obs_err * 1e6 if np.isfinite(fr.depth_obs_err) else None),
            duration_hours=fr.duration * 24,
            duration_err_hours=nn(fr.duration_err * 24 if np.isfinite(fr.duration_err) else None),
            rp_over_rs=fr.rp, rp_over_rs_err=nn(fr.rp_err),
            impact_param=fr.b, a_over_rs=fr.aRs,
            chi2_reduced=fr.chi2_red, reliable=fr.reliable, warning=warn)

    feats = None
    if res.features:
        feats = {k: (float(v) if np.isfinite(v) else 0.0)
                 for k, v in res.features.items()}

    return AnalyzeResponse(
        target=target, detected=True, detection=detection,
        classification=classification, fit=fit, features=feats,
        series=_series_from(res), detrend_method=res.detrend_method,
        n_detrend_trials=res.n_detrend_trials,
        detrend_is_fallback=res.detrend_is_fallback,
        sde_corrected=res.sde_corrected,
        caution_flag=res.caution_flag, caution_reason=res.caution_reason,
        elapsed_seconds=elapsed)


def _run(req: AnalyzeRequest) -> AnalyzeResponse:
    """Resolve the data source, run the pipeline, serialise the result."""
    t_start = time.time()
    truth_label = None

    if req.simulate:
        s = generate_sample(req.simulate, np.random.default_rng(req.seed))
        t, f, e = s["time"], s["flux"], s["flux_err"]
        ident, name, source = f"sim:{req.simulate}:{req.seed}", \
            f"Simulated {req.simulate} (seed {req.seed})", "simulated"
        truth_label = req.simulate
        pub = {}
    elif req.cached:
        got = _load_cached(req.cached)
        if got is None:
            raise HTTPException(404, f"no cached light curve for {req.cached!r}")
        t, f, e, tic = got
        pub = PUBLISHED.get(tic, {})
        ident, name, source = tic, pub.get("name", tic), "cached"
    elif req.tic:
        import lightkurve as lk
        query = req.tic if req.tic.upper().startswith("TIC") else f"TIC {req.tic}"
        try:
            q = lk.search_lightcurve(query, mission="TESS", author="SPOC",
                                     exptime=120)
            if len(q) == 0:
                raise HTTPException(404, f"no SPOC 2-minute TESS data for {query}")
            lc = q[0].download().remove_nans()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f"MAST download failed: {exc}")
        t = np.asarray(lc.time.value, float)
        f = np.asarray(lc.flux.value, float)
        e = np.asarray(lc.flux_err.value, float)
        pub = PUBLISHED.get(query, {})
        ident, name, source = query, pub.get("name", query), "mast"
    else:
        raise HTTPException(400, "provide one of: tic, cached, simulate")

    methods = ("biweight", "lowess") if req.multi_detrend else ("biweight",)
    res = analyze(t, f, e, _model, _calibrator, run_mcmc=req.run_mcmc,
                  detrend_methods=methods)

    target = TargetInfo(
        id=ident, name=name, source=source, n_points=int(np.size(t)),
        baseline_days=float(np.nanmax(t) - np.nanmin(t)),
        note=pub.get("note"), published_period=pub.get("period"),
        published_rp_rs=pub.get("rp_rs"), truth_label=truth_label)

    return _to_response(res, target, time.time() - t_start)


def _key(req: AnalyzeRequest) -> str:
    src = req.simulate or req.cached or req.tic or "?"
    return f"{src}|{req.seed}|{int(req.run_mcmc)}|{int(req.multi_detrend)}"


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #
def _warm():
    reqs = [AnalyzeRequest(cached=os.path.basename(p)[:-4].replace("TIC_", ""))
            for p in sorted(glob.glob(os.path.join(CACHE_DIR, "TIC_*.npz")))]
    reqs += [AnalyzeRequest(simulate=c, seed=42) for c in CLASSES]
    for r in reqs:
        try:
            _cache[_key(r)] = _run(r)
            print(f"[api] warmed {_key(r)}")
        except Exception as exc:
            print(f"[api] warm failed for {_key(r)}: {exc}")


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/api/health", response_model=HealthResponse)
def health():
    # The failure that broke this today: a model trained on N features while the
    # code emits N+1. It loads fine, so classifier_loaded stays true, and every
    # predict call then raises inside the request -- HTTP 500 with a healthy
    # health check. Compare the two counts here, where it is cheap to catch.
    usable = _model is not None
    if _model is not None:
        try:
            n_model = getattr(_model, "n_features_in_", None)
            if n_model is not None and int(n_model) != len(FEATURE_NAMES):
                usable = False
                print(f"[api] MODEL MISMATCH: trained on {n_model} features, "
                      f"code produces {len(FEATURE_NAMES)}. Classification "
                      f"disabled -- retrain, or restore "
                      f"models/demo_frozen/classifier.joblib")
        except Exception:
            pass

    return HealthResponse(
        status="ok", classifier_loaded=usable, classes=CLASSES,
        n_features=len(FEATURE_NAMES),
        cached_targets=len(glob.glob(os.path.join(CACHE_DIR, "TIC_*.npz"))),
        warm_entries=len(_cache))


@app.get("/api/targets")
def targets():
    """Everything analysable: cached real targets and simulated classes."""
    real = []
    for p in sorted(glob.glob(os.path.join(CACHE_DIR, "TIC_*.npz"))):
        d = np.load(p, allow_pickle=True)
        tic = str(d["tic"])
        pub = PUBLISHED.get(tic, {})
        real.append(dict(id=tic.replace("TIC ", ""), tic=tic,
                         name=pub.get("name", tic), note=pub.get("note"),
                         published_period=pub.get("period"),
                         published_rp_rs=pub.get("rp_rs"),
                         n_points=int(d["time"].size)))
    return {"real": real, "simulated": CLASSES,
            "classifier_loaded": _model is not None}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(req: AnalyzeRequest):
    k = _key(req)
    if k in _cache:
        out = _cache[k].model_copy(deep=True)
        out.cached = True
        return out
    loop = asyncio.get_event_loop()
    out = await loop.run_in_executor(_pool, _run, req)
    _cache[k] = out
    return out


@app.post("/api/batch")
async def batch(reqs: list[AnalyzeRequest]):
    """Analyse several light curves; returns a summary ranked by SDE."""
    if len(reqs) > 50:
        raise HTTPException(400, "batch limited to 50 light curves")
    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        *[loop.run_in_executor(_pool, _run, r) for r in reqs],
        return_exceptions=True)
    rows = []
    for r, res in zip(reqs, results):
        if isinstance(res, Exception):
            rows.append(dict(target=_key(r), error=str(res)))
            continue
        rows.append(dict(
            target=res.target.name, detected=res.detected,
            label=res.classification.label if res.classification else None,
            confidence=res.classification.confidence if res.classification else None,
            period_days=res.detection.period_days if res.detection else None,
            depth_ppm=res.detection.depth_ppm if res.detection else None,
            sde=res.detection.sde if res.detection else None))
    rows.sort(key=lambda r: (r.get("sde") or -1), reverse=True)
    return {"n": len(rows), "results": rows}


if os.path.isdir(WEB):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB, "index.html"))

    app.mount("/", StaticFiles(directory=WEB), name="web")
