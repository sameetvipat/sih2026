Against the Problem Statement
The five required capabilities
PS requirement status where

1. Identify periodic dips ✅ search.py — BLS, SDE ≥ 7 threshold
2. Classify into transit / eclipse / blend / other ✅ features.py + classify.py — 22 features → LightGBM
3. Apply to science datasets ✅ run_pipeline.py batch, build_real_dataset.py — 1521 real Kepler targets processed
4. Provide SNR / significance ✅ SDE + depth-SNR per detection; Wilson CIs on all accuracy
5. Estimate depth, period, duration ✅ fit.py — batman + emcee, with validated error bars
   Expected outcomes: pipeline ✅, parameter fitting ✅, visualisation ✅ (web/ + FastAPI), confidence levels ✅ (calibrated probabilities), 3-page report ✅ (reports/report.md, 1124 words).

How the modules connect

                    ┌─────────────── DATA SOURCES ───────────────┐

fetch.py ────────┤ MAST download (retry/backoff/checkpoint) │
simulate.py ─────┤ synthetic signals + inject_into_real() │
└────────────────────┬───────────────────────┘
│ raw (time, flux, flux_err)
▼
preprocess.py normalize → clip positive outliers → wotan detrend
│
▼
search.py BLS on 10-min bins → period, t0, duration, depth, SDE, SNR
│
┌──────────────┴─── if detected ───┐
▼ ▼
preprocess.py mask transits, re-detrend features.py
search.py re-run BLS (2nd pass) 22 vetting diagnostics
│ │
└──────────────┬────────────────────┘
▼
classify.py LightGBM → label + calibrated probability + SHAP drivers
│
┌──────────────┴── if transit/blend ──┐
▼
fit.py batman least-squares → emcee posterior → P, depth, duration ± σ
│
▼
pipeline.py analyze() — the single entry point everything calls
│
┌────────────────────┴────────────────────┐
▼ ▼
api/main.py REST JSON scripts/\*.py batch + training
web/app.js Plotly charts
The critical invariant: make_dataset.py and api/main.py both call pipeline.analyze(). Training data and live queries traverse identical preprocessing. That's deliberate — the alternative silently inflates accuracy.

The logic flow, in order

1. Detrend (preprocess.py) — wotan biweight, window well above transit duration. Only positive outliers clipped: a transit is a downward excursion, and clipping those deleted 496/496 in-transit points before I fixed it.

2. Search (search.py) — BLS on 10-min bins (verified lossless, 3× faster). Outputs where a dip is, never what it is.

3. Second pass — mask the found transits, re-detrend so the trend interpolates across them, re-run BLS. This moved Pi Men c's Rp/R\* from 8σ discrepant to within 1σ.

4. Features (features.py) — 22 diagnostics. The load-bearing ones:

odd_even_sigma — BLS finds half an eclipsing binary's true period in 10/10 cases; alternating depths expose it
sec_ratio_2p_dev — fold at 2P: a planet shows two equal events, an EB a shallower secondary
trap_t23_t14 — U-shape (planet) vs V-shape (grazing/EB)
log_rho_implied — implied stellar density; physically impossible values flag wrong harmonics 5. Classify (classify.py) — LightGBM, calibrated, kept only if calibration improves held-out log loss. SHAP gives per-candidate evidence.

6. Fit (fit.py) — batman + emcee. u1 sampled under a prior (fixing it was the dominant depth error). Reduced-χ² flags fits the model doesn't describe — reported independently of classification confidence.

The output
Per light curve, as JSON:

detection period, t0, duration, depth_ppm, SDE, SNR, n_transits
classification label, calibrated confidence, all 4 probabilities, SHAP drivers
fit period ± σ, depth ± σ (geometric AND observed), duration ± σ,
Rp/R\*, impact param, reduced χ², reliable flag
features all 22 values
series 6 plot-ready arrays (~300 KB, decimated)
Real example — Pi Men c: transit (87%), P = 6.26787 ± 0.00026 d vs published 6.26790, depth 271 ± 13 ppm, drivers sde, trap_t23_t14, trap_depth_ratio, secondary_depth_ratio.

Headline accuracy: 0.653 on 320 held-out real Kepler curves (95% CI [0.599, 0.703], macro F1 0.614, chance 0.25).

The errors — what's actually wrong
Classification is mediocre and blend is near-useless.

class precision recall
eclipse 0.87 0.83
transit 0.58 0.72
variable 0.58 0.57
blend 0.44 0.35
Blend is both the worst-performing and the most under-sampled (183 vs 400 target). Those are likely related.

Real-background injection did not work. +0.025, CI [−0.055, +0.105] — indistinguishable from zero. The directive predicted this would close the gap; it didn't. Real labelled data did.

Simulated features don't transfer. Implied density lands in the physical band for 90.2% of injected rows but only 49.4% of real ones. Injection fixed the noise floor, not signal realism.

AU Mic b is still misclassified — its starspots are 19× its transit depth. The fit is correctly flagged unreliable (χ² 16.9), but the classifier calls it eclipse.

Depth uncertainties remain 1.5× too narrow (u2 still fixed).

Multi-detrend inflates significance — two tests, best kept. Documented, not fixed.

Download harness has no socket timeout — with_backoff catches exceptions, not hangs. This stalled the last run at 95%.

Two Definition-of-Done items are genuinely open: 400+/class (short by 112–217 per class), and Colab verification, which only your team can do.
