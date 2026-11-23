# SWBMCM2611888 — Reproduction

Source code for all three problems. Python 3.10+, dependencies:
`numpy pandas scipy scikit-learn matplotlib`.

## Data prerequisites

- Problems 1/2: raw first-person data folders `EgoDemo/` and `EgoX/`
  from the problem archive (path set at the top of
  `src_p12/load_records.py`).
- Problem 3: `prefix_features.csv` at the workspace root
  (referenced in `src/common.py`).

## Layout

| Directory | Content |
|---|---|
| `src/` | Problem 3: simulator, duration model, strategies, statistics, figures |
| `src_p12/` | Problem 1 (data audit, reliability) and Problem 2 (masking replay, re-observation) |
| `results/` | Problem 3 result tables and figures |
| `results_p12/` | Problem 1/2 result tables and figures |

## Run

Tests (no data required):

```
python src/test_simulator3.py
python src/test_strategy.py
python src_p12/test_p12.py
```

Full pipeline, in order:

```
python src_p12/load_records.py --force     # cache records (once)
python src_p12/p1_pipeline.py              # audit + KW/ICC + reliability
python src_p12/p2_pipeline.py              # masking replay + strategies + stats
python src_p12/stats_p12_extra.py
python src_p12/figures_p12.py
python src/run_test.py                     # P3 simulator on validation cases
python src/run_ablation.py                 # P3 ablation
python src/analyze_duration3.py            # P3 duration model stats
python src/stats_audit.py                  # P3 paired Wilcoxon + Holm
python src/make_figures3.py                # P3 figures
```

Every pipeline writes CSV/JSON to `results/` or `results_p12/`.
All randomness is seeded; reruns reproduce the paper numbers.
