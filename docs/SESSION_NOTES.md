# Session progress — Implementation Directive v2

## Section 0 state check (VERIFIED against live repo, not docs/ref.md)

Catalogue (`data/labels/targets.csv`, Kepler only after dropping
false_positive): transit 3181, eclipse 1435, variable 1075, blend 1009.

`real.parquet` at session start — 1521 rows attempted, **usable (detected)**:
  eclipse 341, transit 288, variable 253, blend 183.
Shards held slightly more (blend 199) from the run that stalled at 95%.
Shortfall vs the 400/class target: blend -201, variable -147, transit -112,
eclipse -59.

**Correction to the directive's framing:** the 400/class target counts *usable*
rows, but `build_real_dataset.py` capped *attempts* at 400. Blend detects at
57%, so that cap asymptotes near 230 usable and could never reach 400 however
long it ran. Fixed: attempts are now sized from each class's measured detection
rate.

Domain-gap regression check: **already implemented** (`check_domain_gap.py`,
threshold 0.30, wired into `train.py`). Phase 4's item was already satisfied.

Baseline to beat: accuracy 0.653 [0.599, 0.703], macro F1 0.614,
blend precision 0.44 / recall 0.35, on 320 held-out real curves.

---

## Phase 0 — download harness hang fix ✅

Cause confirmed by reading: `with_backoff` catches raised exceptions only;
`download_lightcurve` called `lk.search_lightcurve` and `q.download_all()` with
no timeout on either.

Two layers, because neither suffices alone:
- `socket.setdefaulttimeout(90s)` — bounds a stalled *read* inside astroquery /
  lightkurve, which expose no timeout parameter to pass down.
- `call_with_deadline()` — a daemon-thread wall-clock deadline (420 s total,
  120 s of it for the catalogue query). Bounds *total* time, which a per-read
  timeout cannot: a server dribbling one byte per second resets the socket
  timeout forever. Daemon rather than a pool worker so an abandoned hung thread
  cannot deadlock interpreter exit.
Both raise ordinary exceptions, so they feed the existing backoff path.

Verified against a deliberately reproduced hang (`tests/test_fetch_timeout.py`,
6 tests): TEST-NET-1 blackhole connect, bounded by the socket layer when it is
tighter and by the deadline when the socket has none; a hang retried through
the same path as an exception.

Batch watchdog added (warns after 900 s with no completed target).

🔍 CHECKPOINT — small-batch test: **PASS**. Plan printed correctly, first
~60 targets completed with zero hangs, no mass timeouts.

## Phase 1 — data collection (running)

Queue reprioritised: attempts sized by measured per-class detection rate,
ordered most-deficient first (blend → variable → transit → eclipse), 813 total.
Resume now distinguishes *settled* failures (no detection — deterministic,
skip) from retryable network failures (stay queued).

🔍 CHECKPOINT at 13 min: **PASS decisively, continued to ceiling.**
  throughput 15.2 targets/min = 912/hour
  blend usable 199 → 321 in 13 min (61% of the 201 deficit closed)
  extrapolation: full queue in ~41 min, blend reaching ~400

At ~35 min: **blend reached 405 usable — the 400/class target is MET**, up from
183 in `real.parquet` at session start (199 counting the stalled run's shards).
This was the session's highest-leverage goal: blend was both the worst class
(precision 0.44 / recall 0.35) and the most under-sampled.

A note on a false alarm worth not repeating: the tqdm progress line stopped
advancing in the log for ~5 minutes and looked like the hang bug returning. It
was not — shard mtimes and row counts showed steady 16-20 rows/min throughout.
Progress was verified from the shards, which is the durable signal; the tqdm
line is not.

## Phase 2 — limb darkening ✅

`u2` was fixed while `u1` was sampled. Both now sampled in the Kipping (2013)
(q1, q2) basis: the unit square maps exactly onto the physically valid triangle
(u1+u2<1, u1>0, u1+2u2>0), so the degeneracy is handled by construction rather
than by rejection. Moderately-informative Gaussian on the sum u1+u2 (the
well-constrained direction), uniform on q2.

Measured against the pre-fix code loaded straight from git, same photons, same
detection handed to both arms (`scripts/check_limb_darkening.py`):

| target     | depth_obs before | after   | width ratio |
|------------|------------------|---------|-------------|
| Pi Men c   | 302.8 ± 9.4 ppm  | 299.9 ± 11.7 ppm | **1.24×** |
| WASP-121 b | 16308 ± 510 ppm  | 16286 ± 611 ppm  | **1.20×** |

Direction is correct. Errors were diagnosed as 1.5× too narrow; this recovers
~1.2× of that, so it is a partial close, not a full one.

`simulate.py` draws from the same distribution — previously independent uniform
ranges u1~U(0.2,0.6), u2~U(0.1,0.4), a rectangle that both admits unphysical
corners and excludes legitimate ones. Now 100% physically valid by construction.

🔍 CHECKPOINT — `implied_density` in the physical band (0.1–10 g/cm³,
`log_rho_implied` ∈ [-1,1]), classes transit/eclipse/blend:

| set | in-band |
|---|---|
| injected, BEFORE (independent uniform u1,u2) | **90.2%** (n=594) |
| injected, AFTER (Kipping q1,q2) | **90.3%** (n=590) |
| real (fixed reference) | **55.5%** (n=1188) |

**Movement: +0.1 pp. NEGATIVE RESULT — the fix did not close the shape gap.**
Followed the checkpoint's redirect: skipped the secondary Rp/R*, a/R*, b prior
extension entirely, kept the `u2` fix (independently justified, validated at
1.24× on Pi Men c), moved on. No further time spent on shape realism.

The pre-fix 90.2% reproduces docs/ref.md exactly. The real-side figure is 55.5%
rather than docs/ref.md's 49.4% because the real set has grown from 1065 to 1560
usable rows since that measurement; it is a moving reference, not a discrepancy.

**Why it did not move, mechanistically.** `log_rho_implied` is derived from
a/R*, and the injection generator draws a/R* from `rho_star ~ U(0.3, 3.0) ×
1.408 g/cm³` — inside the physical band *by construction*. Limb darkening can
only perturb it second-hand, through the recovered duration. So this feature was
never going to respond much to a limb-darkening change, whatever its realism.

That sharpens the diagnosis rather than just recording a null. The 44.5% of REAL
detections landing outside the band are not there because real *signals* have
strange shapes — they are there because BLS locked onto the wrong harmonic or
mismeasured the duration. The gap is in **detection and measurement realism, not
signal-shape realism**. Injected signals are recovered cleanly because they were
planted cleanly. Recorded, deliberately not chased, per the directive.

## Phase 3 — AU Mic b cross-check ✅

`_cross_check()` in `pipeline.py`; `caution_flag` + `caution_reason` in the
Result, the API schema, and the UI (banner above the classification card).

**Also required a real fix:** eclipses were never fitted
(`result.label in (TRANSIT, BLEND)`), so the class most often assigned to
spotted stars was the one class with no reliability evidence attached at all.
ECLIPSE added to the fit condition.

🔍 CHECKPOINT — AU Mic b: **PASS on all three cases.**
  AU Mic b    label=eclipse chi2_red=16.7 caution=True  ✅
  Pi Men c    label=transit chi2_red=1.7  caution=False ✅
  WASP-121 b  label=transit chi2_red=1.2  caution=False ✅

Non-obvious finding: AU Mic b reaches SDE 4.6 under biweight (below the
threshold of 7 — no detection at all) and SDE 11.0 under lowess. It exists as a
classification failure *only because* the fallback detrender promoted it. The
multi-detrend inflation and the AU Mic b failure are the same story.

Tier (b) — `log_oot_var_to_depth` feature added. First version measured the
detrended flux and ranked AU Mic b *quieter* than Pi Men c, because detrending
is precisely what removes stellar variability. Measured on the trend instead:
  AU Mic b 13.3× its depth · Pi Men c 0.5× · WASP-121 b 0.2×

## Phase 5 — multi-detrend correction ✅ (implemented, not just documented)

Šidák look-elsewhere discount, reported beside the raw SDE; `detrend_is_fallback`
and `n_detrend_trials` surfaced in API + UI. Does not change the detection
decision (that would silently redefine every count in the dataset).

  cost at the detection threshold: **0.10 SDE** (7.00 → 6.90)
  median cost across fallback detections: **0.066 SDE**
  fallback-retained detections: **657/1210 = 54.3%**
  fallback AND within 0.5 SDE of threshold: **20 = 1.7%** ← the bound on how
  much of the dataset the inflation could actually be deciding

Deliberately conservative: biweight and lowess run over the same photons and
are far from independent, so the true effective trial count is nearer 1 than 2
and the real discount is smaller than quoted.

## Phase 6 — Colab/Linux ⚠️ partially verified

No Docker and no Linux environment available in this session.

What WAS verifiable from macOS, because it is a property of the published
artifact rather than the running machine — and the answer contradicts the
README:

  lightgbm 4.7.0 manylinux_2_27_x86_64 wheel, ELF dynamic section of
  lightgbm/lib/lib_lightgbm.so:
    DT_NEEDED includes **libgomp.so.1**
    DT_RUNPATH: (none)
    shared objects in wheel: lib_lightgbm.so only — no vendored libgomp

  → `libgomp1` is REQUIRED from the host. The README claimed recent wheels
    "usually bundle their own OpenMP, in which case libgomp1 is already
    satisfied". Wrong, and wrong in the direction that fails silently: it works
    on Colab only because those images already carry libgomp1 for unrelated
    reasons. Corrected, with `scripts/check_linux_wheel.py` to re-run it.

  **Still genuinely open:** no full training run executed inside a live
  Colab/Kaggle session. Nothing here substitutes for that.

## Outstanding

- Phase 1 running; Phase 2 injected-batch regeneration running.
- All existing real shards predate `log_oot_var_to_depth` →
  `scripts/refresh_features.py` must run over the cache before the retrain.
- Phase 4 retrain not yet started (single consolidated run, as directed).

## Phase 2 — 🔍 CHECKPOINT 3/5 (implied_density) — RUN, RESULT: FLAT

Was missing from this file; run and recorded now.

`implied_density` in the physical band (0.1–10 g/cm³), injectable classes only:

| set | in-band | n |
|---|---|---|
| real catalogued (reference) | **55.5%** | 1188 |
| injected, BEFORE fix (uniform u1/u2) | **90.2%** | 594 |
| injected, AFTER fix (Kipping q1/q2) | **90.3%** | 590 |

Movement toward real: **−0.1 pp**. Gap 34.8 → 34.9 pp. Nothing moved.

The directive's real-side reference was 49.4% at n=1065; it is 55.5% at n=1188
after Phase 1 grew the dataset. The synthetic side is unchanged at 90.2/90.3
either way, so the conclusion does not depend on which reference is used.

**Decision, per the checkpoint's stated rule for a flat result:** the
limb-darkening hypothesis for shape realism is **falsified**. Skipping the
secondary Rp/R*, a/R*, b prior extension entirely. Keeping the q1/q2 fix — it is
independently justified by the depth-uncertainty result (width ratio 1.24×
Pi Men c, 1.20× WASP-121 b) and that justification does not depend on this
checkpoint.

**The per-class split is where the real information is:**

| class | injected before | injected after | real |
|---|---|---|---|
| transit | 86.7% | 86.3% | 53.2% |
| eclipse | 88.5% | 88.5% | 80.7% |
| **blend** | **95.5%** | **96.4%** | **32.8%** |

Blend is the outlier by a wide margin: injected blends land in-band 96.4% of the
time, real ones 32.8% — a **63.6 pp** gap, against 33 pp for transit and 8 pp for
eclipse. That is the largest single discrepancy measured in this project, and it
sits on the worst-performing class.

It also has a physical reading. `implied_density` is derived from period,
duration and depth assuming the signal comes from the target star. For a real
blend that assumption is false by construction — the eclipse is on a
neighbouring star and the depth is diluted — so the inferred density *should*
come out unphysical. Injected blends apply dilution to a signal whose geometry
is still self-consistent, so they keep looking physical. The injector reproduces
the dilution but not the geometric inconsistency that makes a real blend
detectable as one.

That is a sharper diagnosis than "signal shapes are unrealistic", and it is
specific to blend rather than general to injection. Not actioned this session
(the checkpoint says stop), but it is the concrete next hypothesis.
