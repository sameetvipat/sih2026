# Scripts

Nothing here is needed to run the demo — the trained model, feature tables and
cached light curves are all committed. These are the tools that produced them,
and the guards that keep them honest.

Run everything from the repository root, e.g. `.venv/bin/python
scripts/train.py`.

## Building data

| script | what it does |
|---|---|
| `fetch_labels.py` | Build a labelled target list from published KOI/TOI dispositions. |
| `fetch_real.py` | Download a few TESS light curves with known dispositions, for validation. |
| `build_baseline_bank.py` | Collect real, signal-free light curves to inject synthetic signals into. |
| `split_baselines.py` | Assign disjoint train/test splits to the baseline bank, so one star cannot supply noise to both sides. |
| `make_dataset.py` | Simulate light curves, run the full pipeline over them, emit a labelled feature table. The long one (~35 min on 10 cores), resumable. |
| `build_real_dataset.py` | The counterpart to `make_dataset.py` for catalogued real targets. |
| `build_production_set.py` | Assemble the shipped model's training set with the evaluation half held out. |
| `refresh_features.py` | Recompute feature vectors for already-downloaded light curves, after a feature is added. |

## Training and evaluation

| script | what it does |
|---|---|
| `train.py` | Train the classifier and report held-out performance. |
| `evaluate.py` | Injection-recovery: how accurate are the recovered parameters against known injected truth? |
| `evaluate_holdout.py` | Score the classifier on held-out real light curves, both ways. |
| `final_evaluation.py` | Every number frozen for the report, in one run, against both baselines. |
| `compare_domains.py` | Measure the synthetic-to-real domain gap. |
| `validate_real.py` | End-to-end reality check on real TESS observations with known dispositions. |
| `run_pipeline.py` | Batch-run the pipeline over many light curves and emit a candidate catalogue. |

## Verification guards

These exist because each one caught something real. They are cheap to run and
fail loudly.

| script | what it guards against |
|---|---|
| `check_domain_gap.py` | The synthetic-to-real generalisation gap silently returning. |
| `check_features.py` | The two features most likely to misbehave on real backgrounds. |
| `check_caution_flag.py` | The classifier/fit cross-check failing to fire on the case it exists for (AU Mic b). |
| `check_limb_darkening.py` | Freeing `u2` moving the depth uncertainties rather than widening them. |
| `check_linux_wheel.py` | The LightGBM Linux wheel's OpenMP dependency, without needing a Linux box. |
| `check_report.py` | The report still containing unfilled `{{MARKER}}` placeholders. |

## Job control

For the long dataset builds only.

| script | what it does |
|---|---|
| `jobs.sh` | List running pipeline jobs, their worker pools and progress. |
| `stop_jobs.sh` | Stop them cleanly, including the pool children that `pkill` leaves orphaned. |
