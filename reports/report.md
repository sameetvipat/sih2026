<!--
  SIH 2026 — three-page report. STRUCTURE ONLY.

  Every {{PLACEHOLDER}} is a number that must be filled from a fresh run AFTER
  data/processed/real.parquet finishes regenerating. They are deliberately
  left as loud markers rather than filled with current values, because the
  current values were measured with a feature-encoding bug (unmeasurable
  secondary tests were being recorded as 1.0 -- a confident "not a planet" --
  on 32.6% of real rows). A stale number that looks plausible is the single
  easiest way for a wrong figure to reach a submission.

  Fill from:
    scripts/compare_domains.py --classes transit eclipse blend
    scripts/check_domain_gap.py
    scripts/evaluate.py            (injection-recovery, fitting validation)
    scripts/check_features.py
  Do not transcribe from this README or from chat history.
-->

# AI-enabled Detection of Exoplanets from Noisy Light Curves

## 1. Methodology

Pipeline: `detrend → BLS period search → 22 vetting features → gradient-boosted
classifier → transit fit with MCMC uncertainties`.

- **Detrending.** `wotan` biweight filter, window kept well above any plausible
  transit duration, with a second pass that masks known transits so the trend
  interpolates across them rather than bending into them. Multiple filters are
  tried (biweight, lowess) and the higher-SDE result kept.
- **Detection.** Box Least Squares on 10-minute bins. BLS locates a periodic
  dip; it cannot say what caused one. That disambiguation is the project's
  actual contribution.
- **Vetting features (22).** Physically motivated diagnostics: odd/even depth
  difference, secondary eclipse significance, the 2×-period harmonic test,
  trapezoid U-vs-V shape ratio, implied stellar density, per-transit depth
  consistency, Lomb–Scargle power ratio.
- **Classification.** LightGBM over those features rather than a CNN on folded
  views — at this data scale (hundreds to low thousands of examples) trees
  train in seconds and, critically, explain each verdict via per-candidate SHAP
  contributions.
- **Parameter estimation.** Mandel–Agol model (`batman`) fitted by least
  squares, then `emcee` for posteriors.

## 2. Assumptions

- Circular orbits throughout.
- Quadratic limb-darkening coefficients held fixed; they are strongly
  degenerate with impact parameter at this signal-to-noise.
- A baseline is treated as signal-free when BLS finds nothing above SDE 7.
- Kepler false-positive flags map onto the taxonomy only for unambiguous
  single-flag objects; mixed-flag objects are excluded rather than guessed.

## 3. The domain-gap finding

The headline result of this project is a negative one, found by testing rather
than assumed.

A classifier trained on purely synthetic light curves scored **{{SYN_SELF}}**
on held-out synthetic data and **{{SYN_REAL}}** on real catalogued objects — a
gap of **{{GAP_SYN}}**.

The obvious explanation was cadence: Kepler samples every 29 minutes against
TESS's 2, so a 3-hour transit gets 6 points instead of 90. A controlled
experiment ruled this out as the dominant cause — regenerating synthetic data
at Kepler's exact cadence left the gap at **{{GAP_MATCHED}}** versus
**{{GAP_MISMATCHED}}** mismatched, {{PCT_CADENCE_ROBUST}}% of features stayed
cadence-robust on synthetic data, and the median real/synthetic
class-separation ratio at *matched* cadence was **{{F_RATIO}}**.

The dominant cause is that the simulated noise model is too clean. The fix —
injecting known-truth signals onto {{N_BASELINES}} real quiet stars — moved
real-world accuracy to **{{INJ_REAL}}** and the gap to **{{GAP_INJ}}**.
[STATE PLAINLY whether that change is distinguishable from zero at the
measured sample size; if not, say so.]

## 4. Results

| regime | accuracy | macro F1 |
|---|---|---|
| pure synthetic, self-test | {{SYN_SELF}} | {{SYN_SELF_F1}} |
| pure synthetic → real | {{SYN_REAL}} | {{SYN_REAL_F1}} |
| real-injected → real | {{INJ_REAL}} | {{INJ_REAL_F1}} |
| real only → real | {{REAL_REAL}} | {{REAL_REAL_F1}} |

Headline accuracy is **{{HEADLINE}}** — the real-world number. The synthetic
self-test figure is reported only for contrast and is not this pipeline's
accuracy.

Per-class precision/recall/F1: {{PER_CLASS_TABLE}}

**Parameter recovery** is validated separately, on pure-synthetic data where
ground truth is exact (a catalogued period carries its own uncertainty and
would only add noise to a test of our own fitting code). Recovered vs injected
depth: residual scatter **{{DEPTH_SCATTER}}** ppm; period recovered to
**{{PERIOD_ERR}}**. Pull distributions confirm the quoted error bars:
{{PULL_STATS}}. **This validates the fitting code, not the classifier.**

Real-target checks: {{WASP121}}, {{PIMEN}}, {{AUMIC}}.

## 5. Uncertainty estimation

Parameter uncertainties are the standard deviation of an `emcee` posterior (32
walkers), propagated to derived quantities by evaluating them per sample.
Honesty of those bars is tested via the pull distribution `(fitted − true)/σ`,
which should be a unit Gaussian. A reduced-χ² threshold flags fits the model
does not describe; classification confidence and fit reliability are reported
as independent claims.

Accuracy figures carry Wilson intervals, and differences between regimes are
reported with a CI on the difference — at these sample sizes a point estimate
alone can imply a distinction the data cannot support.

## 6. Limitations

- **Real test-set size.** {{N_REAL_TEST}} held-out light curves; 95% CI on
  accuracy spans {{CI_WIDTH}}. Differences smaller than that are not resolvable.
- **Calibration.** The calibration split is in the low tens of rows. Displayed
  confidences are a rank ordering, not frequency claims.
- **Multi-detrend inflates significance.** Trying two filters and keeping the
  higher SDE is running two tests and reporting the better one, which biases
  the significance distribution upward. Exercised more often on real spotted
  stars than on synthetic backgrounds.
- **Baseline bank coverage.** {{N_BASELINES}} Kepler stars across three
  brightness strata ({{STRATA_COUNTS}}). Kepler-only: cross-mission
  generalisation is not claimed.
- **Injection fixes noise, not signal realism.** Both features spot-checked
  survive injection but collapse on real catalogued objects
  ({{FEATURE_CHECK}}), which bounds what real backgrounds alone can achieve.

## 7. Tools

`lightkurve`, `astropy` (BLS), `astroquery` (KOI/TOI dispositions), `wotan`,
`batman`, `emcee`, `scikit-learn`, `LightGBM`, `FastAPI`, `Plotly`.
