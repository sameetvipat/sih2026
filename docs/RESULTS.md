# Results and findings

Everything the project measured, and what each measurement changed. The
README carries the headline numbers; this is the working out behind them.

## Five bugs worth knowing about

All were found by testing against known truth, and all would silently corrupt
results rather than crash:

1. **Sigma-clipping deleted the transits.** A symmetric outlier clip removes
   downward excursions -- but a transit *is* a downward excursion. For a bright
   star a 1% transit sits ~50σ below the median, so a 20σ lower clip removed
   *496 of 496* in-transit points. Fixed: only positive outliers are clipped.
   See `preprocess.clip_outliers`.

2. **The masked re-detrend deleted the transit.** `wotan` propagates NaN
   straight into the returned trend rather than interpolating across it — so
   masking in-transit points and dividing by that trend removed exactly those
   points. This dropped Pi Men c from SDE 21 to 6.7, turning a solid detection
   into a non-detection. The trend is now bridged across masked gaps
   (`preprocess.detrend`). Fixing it moved Pi Men c's Rp/R* from 8 sigma
   discrepant to within 1 sigma of the published value.

3. **Train/serve skew in the feature pipeline.** Training features were built
   with a single detrending pass while inference used the two-pass masked
   version, which recovers ~20% more transit depth — so every depth-based
   feature would have been systematically shifted between training and use.
   `scripts/make_dataset.py` now calls `pipeline.analyze()`, the same entry
   point inference uses.

4. **Dataset generation wrote only at the end.** A ~40 minute run that
   serialises nothing until the final line loses everything to an interrupt —
   which happened twice. Rows are now flushed to shards every 100 results via
   write-to-temp-then-`os.replace`, so a kill cannot leave a half-written file,
   a corrupt shard is skipped rather than fatal, and re-running resumes.

5. **Geometric vs observed transit depth.** `rp^2` is not what the literature
   means by "transit depth": limb darkening makes the observed flux decrement
   ~18% deeper. Reporting only `rp^2` looked like a 25% error against published
   values for Pi Men c when it was a definitional mismatch. `FitResult` now
   reports both.

## Results

**Accuracy on real data: 0.653** (95% CI [0.599, 0.703], macro F1 0.614), on
**320 held-out real Kepler light curves the model never saw**. Four classes,
chance 0.25. The same model scores 0.716 on held-out data from its own training
distribution; that number is not this pipeline's accuracy and appears only for
contrast.

| class | precision | recall | n |
|---|---|---|---|
| transit | 0.58 | 0.72 | 74 |
| eclipse | 0.87 | 0.83 | 108 |
| blend | 0.44 | 0.35 | 57 |
| variable | 0.58 | 0.57 | 81 |

### Did real-background injection help? No, not measurably.

The diagnosed cause of the synthetic-to-real collapse was that the simulated
noise model is too clean, so we stopped simulating a noise floor and borrowed
one: inject known-truth signals onto 463 real quiet Kepler stars
(`scripts/build_baseline_bank.py`), vetted so BLS finds nothing in them,
stratified across three brightness bins (154 bright / 150 medium / 159 faint),
with train and test baselines split before any example was generated.

**Mission-matching:** baselines and evaluation targets are both Kepler — the
mission-matched experiment. Cross-mission generalisation is not claimed.

| regime (3-class, n=285) | accuracy | macro F1 |
|---|---|---|
| pure synthetic, self-test | 0.979 | 0.978 |
| pure synthetic -> real | 0.600 | 0.572 |
| real-injected -> real | 0.625 | 0.594 |
| **real only -> real** | **0.681** | **0.636** |
| real-injected + real -> real | 0.677 | 0.644 |

Injection's contribution is **+0.025, 95% CI [-0.055, +0.105]** — not
distinguishable from zero. What closed the gap was **real labelled data**: 1065
real light curves brought the production domain gap from +0.526 to **+0.063**.

Feature-level evidence explains why. Both features spot-checked
(`scripts/check_features.py`) survive injection but collapse on real catalogued
objects — implied stellar density lands in the physical band for 90.2% of
injected rows against 49.4% of real ones. Real backgrounds fixed the noise
floor, not signal realism.

### The four-class comparison is confounded — read the three-class one

`VARIABLE` has no injected counterpart, so in the injected regime it can only
come from real data. Under `class_weight="balanced"` that imbalance drives the
model to over-predict it. `compare_domains.py --classes transit eclipse blend`
compares like with like.

## What real data caught that simulation could not

The classifier scored **97.4% held-out accuracy on synthetic light curves** and
then got **one of three real planets right**. That gap is the single most
important result in this project, and it is why validating on real
observations is not optional.

| target | truth | before fix | after fix |
|---|---|---|---|
| Pi Men c | planet | `transit` (100%) ✓ | `transit` (99.9%) ✓ |
| WASP-121 b | planet | `blend` (66%) ✗ | **`transit` (90.4%) ✓** |
| AU Mic b | planet | `variable` (83%) ✗ | `variable` (75%) ✗ |

Fixing the simulator flipped WASP-121 b from a confident wrong answer to a
confident right one, without hurting Pi Men c. Held-out synthetic accuracy also
rose from 97.4% to 98.7% — but that number was never the point; the real-target
result is.

**WASP-121 b exposed a wrong assumption baked into the training data.** The
generator's docstring stated that "a planet's secondary is unmeasurable here",
so every synthetic planet had a perfectly flat phase 0.5. The classifier
learned the rule *secondary eclipse implies not-a-planet*. But WASP-121 b is an
ultra-hot Jupiter with a genuine ~500 ppm secondary from thermal emission,
detected in the same TESS data — so the pipeline measured a real secondary and
the model confidently called a confirmed planet a blend. The SHAP evidence said
so directly: `secondary_sigma` was the second-strongest driver toward `blend`.

`simulate.sim_transit` now gives short-period planets a secondary whose depth is
at most 6% of the transit, scaled by irradiation. That stays well clear of the
0.15-0.9 surface-brightness ratios used for eclipsing binaries, so the classes
remain separable — but the model now learns that a *small* secondary is
compatible with a planet rather than ruling one out. The transit radius range
was widened to Rp/R* <= 0.16 at the same time, since real hot Jupiters are
deeper than the original ceiling allowed.

**AU Mic b still fails, and that is the honest state of it.** The failure is
upstream of the classifier, in detrending a star whose spots are 19x its
transit depth. It is a different story and arguably self-consistent. After
detrending, its light curve genuinely still looks variable: the trapezoid shape
ratio is 0.244 (V-shaped, where a planet should be U-shaped) and the 2P harmonic
test is maximally against a planet. The pipeline had already flagged the fit as
untrustworthy via reduced chi2 = 16.9. Calling it `variable` is the honest read
of the data the detrender produced; the failure is upstream, in detrending a
star whose spots are 19x its transit depth.

**The general lesson.** Accuracy measured on data you generated yourself
measures how well the model learned your assumptions, not how well it works.
Every headline number in this project from synthetic data should be read that
way.

## Training on real dispositions, and where the model actually fails

`scripts/fetch_labels.py` pulls **8071 labelled targets** from two public
catalogues, neither of which depends on anyone providing us data:

| mission | transit | eclipse | blend | variable |
|---|---|---|---|---|
| Kepler KOI | 1961 | 1435 | 1009 | 1075 |
| TESS TOI | 1220 | — (not sub-classified) | — | — |

Kepler is the useful one, because its false-positive flags record *why* an
object is false, and those map almost one-to-one onto our taxonomy:
`ss` (secondary eclipse) to `eclipse`, `co`/`ec` (centroid offset, ephemeris
match) to `blend`, `nt` (not transit-like) to `variable`. Only unambiguous flag
combinations are accepted — an object flagged both `ss` and `co` could be
either, and including it would teach the model noise.

`scripts/compare_domains.py` trains on each source and tests on real data:

| experiment | accuracy | macro F1 |
|---|---|---|
| synthetic -> synthetic | **0.984** | 0.984 |
| synthetic -> real | **0.417** | 0.404 |
| real -> real | 0.433 | 0.418 |
| synthetic + real -> real | 0.450 | 0.456 |

A 57-point collapse, and training on real labels recovers almost none of it
(+1.7 points). **0.984 is not this pipeline's accuracy. 0.42 is closer to the
truth, and even that is measured on only 170 real light curves.**

### Ruling out the obvious explanation

The natural suspect was sampling. Kepler long cadence is 29 minutes against
TESS's 2, so a 3-hour transit is sampled 6 times instead of 90 and its ingress
is a single point. Since the model was trained at 2 minutes and tested at 29,
that looked like the whole story.

It is not. `generate_sample` takes `cadence_min` and `n_days`, so the
experiment can be run properly — synthetic data generated at Kepler's exact
cadence and baseline, then tested on Kepler:

| training set | synthetic self-test | -> real | gap |
|---|---|---|---|
| 2-min cadence (mismatched) | 0.984 | 0.417 | **0.568** |
| 29-min cadence (matched) | 0.948 | 0.383 | **0.565** |

Matching the cadence closed **none** of the gap. Comparing class separation
(one-way F statistic) across all three sets shows why: 82% of features keep
more than 30% of their separating power at 29-minute cadence on synthetic data,
and the median real-to-synthetic F ratio *at identical cadence* is **0.047**.

| feature | syn 2-min | syn 29-min | real 29-min |
|---|---|---|---|
| `log_depth` | 727.5 | 662.2 | 24.8 |
| `sde` | 685.7 | 374.6 | 7.1 |
| `log_snr` | 371.3 | 307.5 | 31.9 |
| `log_rho_implied` | 699.1 | 115.5 | 0.9 |
| `log_duration_hr` | 653.2 | 95.7 | 0.8 |
| `trap_t23_t14` | 175.2 | 31.6 | 6.8 |

Coarse sampling does specifically damage the three features derived from
transit duration and shape — `log_rho_implied`, `log_duration_hr` and
`trap_t23_t14` lose most of their power, exactly as the ingress argument
predicts. But that is a secondary effect. Features like `log_depth` and `sde`
sail through the cadence change (727 -> 662, 686 -> 375) and still collapse on
real data (24.8, 7.1).

**So the gap is simulator realism, not sampling.** Real light curves carry
correlated instrumental systematics, stellar variability with structure our
four-sinusoid red-noise model does not reproduce, and genuine astrophysical
diversity within each class. The generator produces light curves that are too
clean, and a model trained on them learns a problem that is easier than the
real one.

### What this means for the project

- Report 0.42, not 0.98. The synthetic number measures how well the model
  learned our assumptions.
- The honest path to a better classifier is more real labelled data, not a
  better simulator. 8071 labelled targets are available; only 240 have been
  processed so far, and `real -> real` trains on just 110 rows.
- Simulation still earns its place for *parameter recovery*, where the injected
  truth is known exactly and the physics (Mandel-Agol) is not an approximation.
  It is classification transfer that it fails at.

## Validation on real TESS data

Everything else is validated against signals we injected ourselves, which
cannot expose an assumption shared by the simulator and the pipeline. These are
real observations of objects with published dispositions:

| target | published period | recovered period | published Rp/R* | fitted Rp/R* | reduced chi2 |
|---|---|---|---|---|---|
| WASP-121 b (deep hot Jupiter) | 1.27493 d | **1.27493 ± 0.00001** | 0.12355 | 0.1181 ± 0.0001 (−4.4%) | 1.27 |
| Pi Men c (shallow, ~300 ppm)  | 6.26790 d | **6.26791 ± 0.00030** | 0.01703 | 0.01665 ± 0.00069 (−2.2%) | 1.71 |
| AU Mic b (heavily spotted)    | 8.46321 d | **8.4727** (detected)   | 0.05140 | 0.1115 (+117%) — **flagged unreliable** | 16.85 |

Periods are recovered essentially exactly. AU Mic b is the instructive case: it
is *detected* at the right period, but its parameters are wrong, and the
pipeline says so — reduced chi2 of 16.9 against 1.3–1.7 for the good fits
trips the `FitResult.reliable` flag. Detecting a signal and trusting its
parameters are separate claims, and the pipeline makes them separately.

### No single detrender works everywhere

AU Mic's starspots modulate its flux by 4.97% peak-to-peak against a 0.26%
transit — 19x larger than the signal. With the default biweight filter, BLS
locks onto the 4.86 d **rotation** period and the planet is missed entirely
(SDE 4.6). LOWESS recovers the true 8.46 d period at SDE 10.3. So
`analyze(detrend_methods=("biweight", "lowess"))` tries both and keeps the
higher SDE. Caveat: trying N filters and keeping the best inflates the
effective false-alarm rate, so multi-method SDE is slightly optimistic.

### A caveat on reported confidences

Every probability the API and UI display is calibrated on a held-out split, and
`classify.train` keeps the calibrator only when it actually improves held-out
log loss. That guards against calibration making things worse, but it does not
make the calibration *reliable* at the current data scale.

The real labelled set is a few hundred rows split three ways, which leaves a
calibration split in the low tens. Isotonic regression needs far more than that
to fit a trustworthy mapping, and even Platt scaling is noisy there — which is
why `_best_calibrator` picks between them on an inner split rather than
assuming isotonic. Treat a displayed "92% confidence" as a rank ordering, not
as a claim that 92 of 100 such cases are correct.

Recalibration is warranted once the real labelled set grows; until then the
number is directional.
