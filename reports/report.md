# AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves

**SIH 2026** · Pipeline, results and limitations · All figures measured, none estimated.

## 1. Methodology

`detrend → BLS period search → 22 vetting features → gradient-boosted classifier → transit fit with MCMC`

**Detrending.** `wotan` biweight filter with a window well above any plausible
transit duration, plus a second pass that masks known transits so the trend
interpolates across them rather than bending into them. Two filters are tried
(biweight, lowess) and the higher-SDE result kept — this rescues heavily
spotted stars where one filter destroys the signal.

**Detection.** Box Least Squares on 10-minute bins (verified lossless: identical
recovered period and SDE, 3× faster). BLS locates a periodic dip; it cannot say
what caused one. That disambiguation is this project's contribution.

**Vetting features (22).** Physically motivated: odd/even depth difference,
secondary eclipse significance, the 2×-period harmonic test, trapezoid U-vs-V
shape ratio, implied stellar density, per-transit depth consistency,
Lomb–Scargle power ratio, out-of-transit skew.

**Classification.** LightGBM over those features rather than a CNN on folded
views. At this data scale (thousands, not the hundreds of thousands AstroNet
used) trees train in seconds and — decisively for vetting — explain each verdict
through per-candidate SHAP contributions.

**Parameters.** Mandel–Agol model (`batman`) fitted by least squares, then
`emcee` for posteriors.

## 2. Assumptions

Circular orbits. Quadratic limb darkening with `u1` sampled under a
N(0.4, 0.15) prior and `u2` fixed (§5). A baseline is treated as signal-free
when BLS finds nothing above SDE 7. Kepler false-positive flags map onto the
taxonomy only for unambiguous single-flag objects; mixed-flag objects are
excluded rather than guessed.

## 3. Headline result

**Accuracy on real data: 0.653** (95% CI [0.599, 0.703], macro F1 0.614), measured
on **320 held-out real Kepler light curves the model never saw**, four classes,
chance = 0.25.

The same model scores 0.716 on held-out data from its own training
distribution. That 0.716 is *not* this pipeline's accuracy and is reported only
for contrast.

| class | precision | recall | n |
|---|---|---|---|
| transit | 0.58 | 0.72 | 74 |
| eclipse | 0.87 | 0.83 | 108 |
| blend | 0.44 | 0.35 | 57 |
| variable | 0.58 | 0.57 | 81 |

`blend` is the weakest class. Aggregate accuracy would have hidden that, which
is why per-class reporting is mandatory throughout this project.

## 4. The domain-gap investigation

A classifier trained purely on simulated light curves scored **0.984** on
held-out synthetic data and **0.458** on real catalogued objects — a **0.526**
gap. Reporting the first number would have been indefensible.

**Cadence was ruled out, not assumed.** Kepler samples every 29 min against
TESS's 2, so a 3-hour transit gets 6 points instead of 90. Regenerating the
synthetic set at Kepler's exact cadence left the gap at **0.565** versus
**0.568** mismatched — indistinguishable. 82% of features kept >30% of their
class-separating power at coarse cadence, and the median real/synthetic
separation ratio at *matched* cadence was **0.047**. Cadence does damage the
three duration- and shape-derived features specifically, but it is not the
cause.

**Diagnosis:** the simulated noise model (white noise plus four sinusoids) is
categorically too clean.

**Fix attempted:** inject known-truth signals onto **463 real quiet Kepler
stars** (154 bright / 150 medium / 159 faint), vetted so BLS finds nothing in
them, with train/test baselines split before any example was generated.

**Outcome — reported honestly:** injection did *not* measurably help.

| regime (3-class) | accuracy | macro F1 |
|---|---|---|
| pure synthetic, self-test | 0.979 | 0.978 |
| pure synthetic → real | 0.600 | 0.572 |
| real-injected → real | 0.625 | 0.594 |
| **real only → real** | **0.681** | **0.636** |

Injection's contribution is **+0.025, 95% CI [−0.055, +0.105]** — not
distinguishable from zero at n=285. What actually closed the gap was **real
labelled training data**: 1065 real light curves from Kepler KOI dispositions
brought the production gap to **+0.063**.

Feature-level evidence explains why: both spot-checked features survive
injection but collapse on real catalogued objects (implied density in the
physical band: 90.2% injected vs 49.4% real). Real backgrounds fixed the noise
floor, not signal realism.

## 5. Parameter recovery and uncertainties — *validates the fitting code, not the classifier*

Measured on pure-synthetic injections where truth is exact (a catalogued
period carries its own error and would only add noise to a test of our own
fitter). n = 250.

- Detection completeness **0.996**; correct period given detection **0.996**
- Period bias **+1.4×10⁻⁷ d**, scatter **9.6×10⁻⁵ d**
- Depth bias **−0.37%**, scatter **256 ppm**
- Duration bias **−0.38%**

**Error bars are validated, not asserted.** The pull `(fitted − true)/σ` should
be a unit Gaussian. It initially was not — depth was 4× too narrow. Two
hypotheses were tested: red noise (implemented the Winn et al. 2008 β factor;
it did *not* close the gap, so correlated noise was not the cause) and fixed
limb darkening (**decisive** — refitting with the true coefficients cut median
depth error from 6.19% to 0.70%). Sampling `u1` under a prior fixed it:

| robust pull σ | period | depth | duration |
|---|---|---|---|
| before | 0.99 | 4.01 | 1.94 |
| **after** | **0.84** | **1.52** | **1.33** |

A reduced-χ² threshold separately flags fits the model does not describe;
classification confidence and fit reliability are reported as independent
claims.

**Real targets** (published values never fed to the pipeline): WASP-121 b
period 1.27493 ± 0.00002 d (exact), Rp/R\* −2.1%; Pi Men c period 6.26787 ±
0.00026 d (exact), Rp/R\* −3.4%. AU Mic b is detected at the correct period but
misclassified and its fit flagged unreliable (χ² 16.9) — its starspots modulate
flux 19× deeper than its transit.

## 6. Limitations

- **Test-set size.** 320 held-out real curves; the 95% CI spans 0.104. Differences
  below ~0.10 are not resolvable, which is why injection's +0.025 is reported as
  inconclusive rather than as a gain.
- **Calibration.** The calibration split is in the low hundreds. Displayed
  confidences are a rank ordering, not frequency claims.
- **Multi-detrend inflates significance.** Trying two filters and keeping the
  higher SDE is running two tests and reporting the better one, biasing the
  significance distribution upward. Exercised more on real spotted stars than on
  synthetic ones.
- **Baseline bank.** 463 Kepler stars, three brightness strata, Kepler-only.
  Cross-mission generalisation is not claimed.
- **Residual 1.5× on depth uncertainty.** `u2` is still fixed and real photometry
  has structure the model does not capture.
- **Class coverage.** `blend` reached 183 usable examples against 400 requested —
  the weakest class in both volume and recall (0.35).

## 7. Tools

`lightkurve`, `astropy` (BLS), `astroquery` (KOI/TOI dispositions), `wotan`,
`batman`, `emcee`, `scikit-learn`, `LightGBM`, `FastAPI`, `Plotly`. 62 tests
cover pipeline invariants, the API contract, dataset resumability and injection.
