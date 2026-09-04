---
title: Transit Console
emoji: "🪐"
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Detect and classify exoplanet transits in TESS light curves
---

# Exoplanet transit detection from noisy light curves

Finds periodic dips in TESS/Kepler photometry, decides what physically caused
each dip, and fits a transit model with calibrated uncertainties.

Detection itself is classical signal processing (BLS) and is not the
contribution. The contribution is everything after it: deciding *what* a dip is,
and putting honest error bars on it.

```
raw flux
   |
   +-- clean & detrend ......... wotan biweight, transit-masked second pass
   |
   +-- period search ........... BLS on 10-min bins -> period, t0, duration, depth, SDE
   |
   +-- vetting features ........ 23 physically-motivated diagnostics
   |
   +-- classification .......... LightGBM + probability calibration
   |                             -> transit | eclipse | blend | variable | noise
   |
   +-- parameter fitting ....... batman (Mandel-Agol) + emcee
                                 -> period, depth, duration, each +/- 1 sigma
```

## The five classes

| class | physical origin | signature the pipeline keys on |
|---|---|---|
| `transit` | planet crossing its host star | shallow, U-shaped, no secondary, equal depths at 2P |
| `eclipse` | eclipsing binary | deep, V-shaped, secondary eclipse, odd/even split |
| `blend` | deep EB diluted by a neighbour in the aperture | planet *depth*, binary *shape* |
| `variable` | starspot rotation / pulsation | smooth, sinusoidal, no sharp ingress |
| `noise` | no coherent periodic signal | low SDE, inconsistent per-transit depths |

`blend` is the hard case by construction: dilution puts it in the planet depth
range, so depth alone cannot separate it from `transit`. Only shape and the
harmonic tests can.

## Headline result

**0.653 accuracy on 320 held-out real Kepler light curves the model never saw**
(95% CI [0.599, 0.703], macro F1 0.614, four classes, chance 0.25).

The same model scores 0.716 on held-out data from its own training
distribution. That number is not this pipeline's accuracy and is quoted only
for contrast — the gap between them is the point, and
[docs/RESULTS.md](docs/RESULTS.md) works through why.

## Install

Requires Python 3.12. LightGBM needs OpenMP, which is the one non-obvious
dependency.

```bash
# macOS
brew install uv libomp

# Debian/Ubuntu
sudo apt-get install -y libgomp1 && pip install uv
```

```bash
git clone <repo> && cd sih2026
uv venv --python 3.12 .venv
uv pip install -r requirements.txt setuptools   # setuptools shims distutils for batman
```

Verify:

```bash
.venv/bin/python -m pytest tests/ -q            # 77 tests, ~2.5 min
```

The trained model, the feature tables and three real light curves are
committed, so nothing needs downloading or regenerating before the demo runs.

## Run the demo

```bash
./start.sh                  # then open http://localhost:8000
```

`start.sh` checks the things that have actually broken before — the virtualenv,
the classifier's feature count against the code's, the offline assets — and
refuses to claim it is healthy when it is not. `./stop.sh` shuts it down.

Presenting? Read [docs/RUNBOOK.md](docs/RUNBOOK.md) first.

The API warms its result cache in a background thread at startup, so the first
click is instant and does not depend on the network.

## Reproduce the pipeline

Each step is optional — all of their outputs are committed.

```bash
# build a labelled training set (~35 min on 10 cores, resumable)
.venv/bin/python scripts/make_dataset.py --jobs 10

# train the classifier
.venv/bin/python scripts/train.py

# measure parameter accuracy against injected truth
.venv/bin/python scripts/evaluate.py

# batch-run over real data
.venv/bin/python scripts/run_pipeline.py --fits data/raw
```

`scripts/` holds the data-building, training, evaluation and verification
tools; see [scripts/README.md](scripts/README.md) for what each one is for.

## API

The pipeline is a REST service and the web UI is a static page that calls it,
so the same endpoints drive the demo and any batch tooling.

| endpoint | purpose |
|---|---|
| `GET /api/health` | service status, whether the classifier is loaded |
| `GET /api/targets` | cached real targets and simulated classes |
| `POST /api/analyze` | run the pipeline on one light curve |
| `POST /api/batch` | run it on up to 50, ranked by significance |
| `GET /docs` | auto-generated OpenAPI documentation |

```bash
curl -X POST localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"cached": "261136679", "run_mcmc": true}'
```

A request picks its source with exactly one of `tic` (download from MAST),
`cached` (a locally stored light curve) or `simulate` (generate a labelled
synthetic curve). The response carries the detection, the calibrated
classification, the fitted parameters with uncertainties, the full 23-feature
vetting vector, and every series the frontend plots — about 300 KB, decimated
to display resolution.

## Layout

```
src/exodet/       the pipeline
  config.py         taxonomy, cadence, search and detrend constants
  simulate.py       synthetic light curve generator (batman injection)
  preprocess.py     cleaning, transit-safe biweight detrending
  search.py         BLS period search, folding, binning
  features.py       23 vetting diagnostics
  fit.py            batman least-squares + emcee posteriors
  classify.py       LightGBM training, calibration, SHAP explanations
  pipeline.py       end-to-end analyze() -- the single entry point
  fetch.py          MAST downloads, resumable job pools
  metrics.py        confidence intervals and score reporting
api/              FastAPI service; also serves the web UI
web/              single page, no build step; Plotly and fonts vendored
scripts/          data building, training, evaluation, verification
tests/            77 tests: pipeline invariants, API contract,
                  dataset resumability, real-background injection
data/             cached light curves, feature tables, labels
models/           the trained classifier, plus a frozen demo fallback
docs/             runbook, results, session notes
```

## Key design decisions

**Synthetic training data with known ground truth.** Every training light curve
injects an analytic signal (`batman`) into realistic white + red noise. That
gives balanced classes, unlimited data, and — critically — *known injected
parameters*, so parameter accuracy can be measured rather than assumed.

**The 2x harmonic test.** BLS locks onto *half* an eclipsing binary's true
period in essentially every case, because primary and secondary eclipses then
stack at the same phase. Folding at 2P separates them: a real planet shows two
*equal*-depth events, an EB a shallower secondary. The deviation of that depth
ratio from 1.0 is among the strongest features in the model.

**Features, not pixels.** A tree ensemble over interpretable diagnostics beats a
CNN at this data scale, trains in seconds, and can name the evidence behind
every call via SHAP. A CNN on folded views is the natural Tier-2 extension.

## Limitations

- **Blend identification is signal-shape only.** The decisive test is
  per-pixel centroid analysis on target pixel files, which needs TPF downloads.
- **Circular orbits assumed throughout.**
- **Displayed confidences are directional, not literal.** The real labelled set
  is small enough that the calibration split is in the low tens; read a "92%"
  as a rank ordering, not as a frequency claim. See
  [docs/RESULTS.md](docs/RESULTS.md).
- **No single detrender works everywhere** — the pipeline tries two and reports
  which one produced the retained detection, with a look-elsewhere discount on
  its significance.

## Further reading

- [docs/RUNBOOK.md](docs/RUNBOOK.md) — how to present the demo, and what to do when it breaks
- [docs/RESULTS.md](docs/RESULTS.md) — every measurement, the five bugs worth knowing about, and the synthetic-to-real domain gap
- [docs/DASHBOARD_OUTPUTS_EXPLAINED.md](docs/DASHBOARD_OUTPUTS_EXPLAINED.md) — what every number and plot in the UI means
- [docs/SESSION_NOTES.md](docs/SESSION_NOTES.md) — chronological working notes
