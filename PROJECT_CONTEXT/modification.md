# Project Modification Snapshot

Last updated: 2026-04-17

## Purpose

- Keep upstream baseline behavior reproducible.
- Add fork-specific distance-map extensions in an additive way.
- Keep baseline and extension paths clearly separable by config/CLI.

## Current Valid State

- Training entrypoint: `train.py`
- Main aggregation utility: `scripts/aggregate_results.py`
- Core outputs from solver:
  - epoch log: `results.csv`
  - final summary: `final_results.csv`
  - final split summary files: `<split>_results.csv`
- Main model output naming:
  - binary: `binary_<conn_num>_bce`
  - dist: `<label_mode>_<conn_num>_<dist_aux_loss>`

## Active Architecture Decisions

1. Baseline compatibility first

- Upstream binary path remains intact.
- Fork extensions are controlled by `--label_mode` and related flags.

1. Dist path supervision design

- Final segmentation output is supervised with binary-mask objective.
- Distance information is used as auxiliary supervision (not as final output target).

1. Reproducibility

- Seed control is explicit (`--seed`).
- DataLoader worker seeding and fold-level deterministic behavior are enabled.

1. Aggregation policy

- Aggregation accepts both indexed and single-run result layouts.
- Dataset-grouped summary artifacts are supported.
- Single-experiment input is supported.

## Recent High-Impact Changes (Kept)

- `train.py`
  - Seed hardening (`--seed`, deterministic setup, worker seeding, fold-level generator seeding).
- `solver.py`
  - ISIC precision/accuracy persistence in saved metrics.
  - Unified CSV writing around `results.csv` and `<split>_results.csv`.
- `scripts/aggregate_results.py`
  - Single-run file support (`final_results.csv`, legacy `results.csv`).
  - Fold/indexed compatibility (`final_results_<fold>.csv`, optional legacy `results_<fold>.csv`).
  - Single-experiment aggregation path enabled.
  - Dataset-split summary artifacts and dataset-bundle PDF support.
  - Sample-visualization model-dir fallback (`models/<fold>` -> `models`).

## Deprecated / Removed From Active Context

- Old per-day exhaustive changelog entries are removed from this file.
- Historical implementation details remain in Git history.
- Legacy references to `scripts/aggregate_kfold_results.py` are no longer treated as primary workflow.

## Known Constraints

- `scripts/aggregate_results.py` currently logs generation of experiment-mean LaTeX/CSV; dataset-bundle PDF is generated, while standalone experiment-mean PDF generation depends on explicit `build_pdf` call path.
- Runtime metrics/warnings from third-party metric utilities (for example clDice edge cases) may still appear during smoke runs.

## Next Practical Steps

1. Keep one canonical aggregation script (`aggregate_results.py`) and retire legacy script paths from docs/scripts.
2. Add a small regression test fixture for aggregation input layouts:
   - single-run (`final_results.csv`)
   - indexed (`final_results_1.csv`)
   - mixed dataset roots.
3. Keep PROJECT_CONTEXT files concise; append only currently actionable state.
