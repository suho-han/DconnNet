# Testing Notes (Condensed)

Last updated: 2026-04-17

## Scope

This file keeps only currently relevant validation results.
Historical step-by-step smoke logs were removed for brevity.

## Current Validation Matrix

### A. Aggregation (`scripts/aggregate_results.py`)

- Syntax check
  - `.venv/bin/python -m py_compile scripts/aggregate_results.py` passed
- Single-experiment flow (CHASE root)
  - command:
    - `.venv/bin/python scripts/aggregate_results.py --input-root output/chase --output-dir /tmp/agg_single_verify --sample-vis-count 0`
  - result: passed
  - key artifacts:
    - `/tmp/agg_single_verify/summary.pdf`
    - `/tmp/agg_single_verify/dump/summary.csv`
    - `/tmp/agg_single_verify/dump/summary_experiment_means.csv`
    - `/tmp/agg_single_verify/summary_experiment_means_datasets.pdf`
- Single-experiment + sample visualization
  - command:
    - `.venv/bin/python scripts/aggregate_results.py --input-root output/chase --output-dir /tmp/agg_single_verify_vis`
  - result: passed
  - key artifacts:
    - `/tmp/agg_single_verify_vis/summary_sample_visualization.png`
    - `/tmp/agg_single_verify_vis/dump/summary_sample_visualization.csv`
- Multi-dataset synthetic fixture smoke (`chase`, `isic`)
  - result: passed
  - key artifacts:
    - `.../summary_experiment_means_chase.csv`
    - `.../summary_experiment_means_isic.csv`
    - `.../summary_experiment_means_datasets.pdf`

### B. Training/Eval Core

- `train.py`, `solver.py` syntax check
  - `.venv/bin/python -m py_compile train.py solver.py` passed
- Seed reproducibility wiring check
  - `train.py` seed path compile/usage check passed

### C. Launchers and Shell Scripts (latest relevant)

- CHASE/ISIC launcher syntax checks passed in recent runs.
- Telegram notifier script path is operational (`scripts/telegram_alert.py`).

## Current Confidence

- Aggregation pipeline: good for current single-run layout.
- Dataset-split summary generation: verified.
- Sample visualization export: verified on CHASE output layout.

## Remaining Gaps

1. Add automated regression tests for aggregation (currently manual/smoke-driven).
2. Run periodic full-root smoke when new datasets/experiments are added.
3. Keep output naming conventions stable to avoid parser drift.
