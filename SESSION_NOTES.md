# Session progress — Implementation Directive v2

## Section 0 state check (VERIFIED against live repo, not ref.md)

Catalogue (`data/labels/targets.csv`, 8071 rows, Kepler only after
false_positive drop): transit 3181, eclipse 1435, variable 1075, blend 1009.

`data/processed/real.parquet` — 1521 rows ATTEMPTED:
  transit 400, eclipse 400, variable 400, blend 321
`detected == True` (USABLE) per class — this is what the 400/class target means:
  eclipse 341, transit 288, variable 253, blend 183
Shortfall vs 400 usable: blend -217, variable -147, transit -112, eclipse -59.

NOTE: `build_real_dataset.py` caps ATTEMPTS at `--per-class` (400). Blend
detects at 183/321 = 57%, so a 400-attempt cap can never reach 400 usable.
The cap must be made deficit-aware. Catalogue has 1009 blend => ~688 unfetched.

Domain-gap regression check: ALREADY IMPLEMENTED (`scripts/check_domain_gap.py`,
threshold 0.30, wired into train.py). Phase 4 item already satisfied.

Baseline to beat (from README/report): accuracy 0.653 [0.599, 0.703],
macro F1 0.614, blend precision 0.44 / recall 0.35, on 320 held-out real curves.

## Phase status
- [ ] Phase 0 — download harness hang fix
- [ ] Phase 1 — blend-prioritised data collection
- [ ] Phase 2 — limb darkening u2
- [ ] Phase 3 — caution_flag cross-check
- [ ] Phase 4 — consolidated retrain
- [ ] Phase 5 — multi-detrend
- [ ] Phase 6 — freeze numbers
