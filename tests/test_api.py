"""API contract tests. Run with: .venv/bin/python -m pytest tests/test_api.py -q

These use FastAPI's TestClient, so no server needs to be running.
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # context manager form runs lifespan; we skip warming by not waiting on it
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["n_features"] == 22
    assert len(d["classes"]) == 5


def test_targets_lists_simulated_classes(client):
    d = client.get("/api/targets").json()
    assert set(d["simulated"]) == {"transit", "eclipse", "blend", "variable", "noise"}
    for t in d["real"]:
        assert t["n_points"] > 1000


def test_analyze_requires_a_source(client):
    r = client.post("/api/analyze", json={})
    assert r.status_code == 400


def test_analyze_rejects_unknown_simulate_class(client):
    r = client.post("/api/analyze", json={"simulate": "banana"})
    assert r.status_code == 422          # pydantic Literal rejects it


def test_analyze_missing_cached_target_is_404(client):
    r = client.post("/api/analyze", json={"cached": "999999999"})
    assert r.status_code == 404


def test_analyze_simulated_transit_round_trip(client):
    """A strong injected transit must be detected and fully serialised."""
    r = client.post("/api/analyze", json={
        "simulate": "transit", "seed": 7, "run_mcmc": False,
        "multi_detrend": False})
    assert r.status_code == 200
    d = r.json()
    assert d["target"]["truth_label"] == "transit"
    assert d["detected"] is True

    det = d["detection"]
    assert det["period_days"] > 0
    assert det["sde"] >= 7.0
    assert det["depth_ppm"] > 0

    # every series the frontend draws must be present and non-empty
    s = d["series"]
    for key in ("raw", "detrended", "periodogram", "fold", "fold_binned", "fold_2p"):
        assert len(s[key]["x"]) > 0, f"series.{key} is empty"
        assert len(s[key]["x"]) == len(s[key]["y"]), f"series.{key} x/y length mismatch"

    assert len(d["features"]) == 22
    assert all(isinstance(v, (int, float)) for v in d["features"].values())


def test_analyze_is_cached_on_repeat(client):
    body = {"simulate": "transit", "seed": 11, "run_mcmc": False,
            "multi_detrend": False}
    first = client.post("/api/analyze", json=body).json()
    second = client.post("/api/analyze", json=body).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["detection"]["period_days"] == first["detection"]["period_days"]


def test_batch_ranks_by_significance(client):
    r = client.post("/api/batch", json=[
        {"simulate": "transit", "seed": 3, "run_mcmc": False, "multi_detrend": False},
        {"simulate": "noise", "seed": 4, "run_mcmc": False, "multi_detrend": False},
    ])
    assert r.status_code == 200
    rows = r.json()["results"]
    assert len(rows) == 2
    sdes = [row.get("sde") or -1 for row in rows]
    assert sdes == sorted(sdes, reverse=True), "batch results not ranked by SDE"


def test_batch_size_is_capped(client):
    r = client.post("/api/batch", json=[{"simulate": "noise"}] * 51)
    assert r.status_code == 400


def test_web_assets_are_served(client):
    assert client.get("/").status_code == 200
    for path in ("/app.js", "/style.css"):
        assert client.get(path).status_code == 200, f"{path} not served"


# Every field path the frontend reads out of /api/analyze. Kept explicit so a
# schema rename fails here rather than silently breaking a chart in the demo.
# (This caught web/app.js reading `detection.period` when the API returns
# `period_days` -- undefined propagated to NaN and hung the browser in an
# infinite marker loop.)
FRONTEND_CONTRACT = [
    "target.name", "target.source", "target.n_points", "target.baseline_days",
    "target.note", "target.published_period", "target.published_rp_rs",
    "target.truth_label",
    "detected", "message", "detrend_method", "elapsed_seconds", "cached",
    "detection.period_days", "detection.t0", "detection.duration_hours",
    "detection.depth_ppm", "detection.sde", "detection.snr",
    "detection.n_transits",
    "classification.label", "classification.confidence",
    "classification.probabilities", "classification.drivers",
    "fit.period_days", "fit.period_err",
    "fit.depth_observed_ppm", "fit.depth_observed_err_ppm",
    "fit.depth_geometric_ppm", "fit.depth_geometric_err_ppm",
    "fit.duration_hours", "fit.duration_err_hours",
    "fit.rp_over_rs", "fit.rp_over_rs_err", "fit.impact_param",
    "fit.chi2_reduced", "fit.reliable", "fit.warning",
    "series.raw.x", "series.raw.y", "series.trend.x", "series.trend.y",
    "series.detrended.x", "series.detrended.y",
    "series.periodogram.x", "series.periodogram.y",
    "series.fold.x", "series.fold.y",
    "series.fold_binned.x", "series.fold_binned.y", "series.fold_binned.e",
    "series.fold_2p.x", "series.fold_2p.y",
    "series.model.x", "series.model.y",
    "series.odd.x", "series.odd.y", "series.even.x", "series.even.y",
    "features",
]


def _dig(obj, path):
    """Walk a dotted path; raises KeyError naming the first missing segment."""
    cur = obj
    for i, part in enumerate(path.split(".")):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(".".join(path.split(".")[:i + 1]))
        cur = cur[part]
    return cur


def test_response_satisfies_the_frontend_contract(client):
    """Every field web/app.js reads must exist on a fully-populated response."""
    r = client.post("/api/analyze", json={
        "simulate": "transit", "seed": 7, "run_mcmc": True,
        "multi_detrend": False})
    assert r.status_code == 200
    d = r.json()
    assert d["detected"], "need a detected result to exercise the full contract"

    # The classification block is null until a model has been trained, so only
    # require it when one is actually loaded.
    has_classifier = client.get("/api/health").json()["classifier_loaded"]
    paths = [p for p in FRONTEND_CONTRACT
             if has_classifier or not p.startswith("classification.")]

    missing = []
    for path in paths:
        try:
            _dig(d, path)
        except KeyError as exc:
            missing.append(str(exc))
    assert not missing, f"frontend reads fields absent from the response: {missing}"
    if not has_classifier:
        assert d["classification"] is None, \
            "classification must be null, not partial, when no model is loaded"


def test_app_js_uses_only_real_detection_fields(client):
    """Guard the specific mismatch that hung the browser."""
    import os
    import re
    js = open(os.path.join(os.path.dirname(__file__), "..", "web", "app.js")).read()
    valid = set(client.post("/api/analyze", json={
        "simulate": "transit", "seed": 7, "run_mcmc": False,
        "multi_detrend": False}).json()["detection"])
    used = set(re.findall(r"\bdet\.([a-zA-Z_][a-zA-Z0-9_]*)", js))
    assert used <= valid, f"app.js reads unknown detection fields: {used - valid}"


def test_shap_drivers_reach_the_api_response(client):
    """P5.4: the per-candidate explanation must survive to the JSON payload.

    Confirming classify.explain() works in isolation is not enough -- the demo
    shows whatever /api/analyze returns.
    """
    if not client.get("/api/health").json()["classifier_loaded"]:
        pytest.skip("no classifier loaded")
    r = client.post("/api/analyze", json={
        "simulate": "transit", "seed": 7, "run_mcmc": False,
        "multi_detrend": False})
    assert r.status_code == 200
    cls = r.json()["classification"]
    assert cls is not None
    assert cls["drivers"], "no feature drivers in the response"
    for name, value in cls["drivers"]:
        assert isinstance(name, str) and isinstance(value, (int, float))


def test_drivers_are_per_candidate_not_global_importances(client):
    """classify.explain() falls back to global feature importances when SHAP
    fails. That fallback returns the SAME features for every candidate, which
    would look fine in a demo while explaining nothing about the specific
    object. Different classes must yield different drivers.
    """
    if not client.get("/api/health").json()["classifier_loaded"]:
        pytest.skip("no classifier loaded")

    def drivers(label):
        r = client.post("/api/analyze", json={
            "simulate": label, "seed": 7, "run_mcmc": False,
            "multi_detrend": False}).json()
        cls = r.get("classification")
        return [k for k, _ in (cls["drivers"] if cls else [])]

    a, b = drivers("transit"), drivers("eclipse")
    assert a and b
    assert a != b, (
        f"identical drivers for transit and eclipse ({a}) -- explain() has "
        "silently fallen back to global importances")


def test_frontend_has_a_visible_fallback_banner():
    """P4.2: degraded mode must be visible in the page, not only in the API.

    Without the classifier the app still renders detections and fits, so a
    demo can silently answer none of the classification questions it appears
    to answer. A status chip is too easy to miss.
    """
    import os
    js = open(os.path.join(os.path.dirname(__file__), "..", "web", "app.js")).read()
    assert "setFallbackBanner" in js
    assert "fallbackBanner" in js
    # must be driven by the health endpoint, not a one-time page-load guess
    assert js.index("setFallbackBanner") < js.index("pollHealth")
