# Project Modification Snapshot

Last updated: 2026-04-19

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
  - Added additive `octa500` dataset branch using `MyDataset_OCTA500` (train/test split loading).
- `solver.py`
  - ISIC precision/accuracy persistence in saved metrics.
  - Unified CSV writing around `results.csv` and `<split>_results.csv`.
- `scripts/aggregate_results.py`
  - Single-run file support (`final_results.csv`, legacy `results.csv`).
  - Fold/indexed compatibility (`final_results_<fold>.csv`, optional legacy `results_<fold>.csv`).
  - Single-experiment aggregation path enabled.
  - Dataset-split summary artifacts and dataset-bundle PDF support.
  - Sample-visualization model-dir fallback (`models/<fold>` -> `models`).
- Unified dataset launcher
  - `scripts/train_launcher.sh` now runs in config-first mode:
    - `scripts/train_launcher.sh --config <yaml> [--device N] [--dry_run]`
    - launcher-side `while/case` CLI parsing was removed and moved into a Python config helper.
  - Added `scripts/train_launcher_from_config.py` to parse YAML, build schedules, and run/dry-run generated `train.py` commands.
  - Added launcher config templates under `scripts/configs/` for all dataset single/multi wrappers.
  - Grouped-direction options (`direction_grouping`, `direction_fusion`) are now configured in YAML.
  - `coarse24to8` conn policy remains enforced:
    - single mode: `conn_num=24` is rejected
    - multi mode: conn sweep is normalized to `[8]`
  - Existing dataset-specific launchers remain deprecated wrappers for one-version compatibility and now forward with mapped `--config` paths.
  - Added `scripts/gpu_train_process_summary.sh` to inspect active GPU compute PIDs and summarize live training args:
    - dataset / conn_num / label_mode / dist_aux_loss / direction_grouping / direction_fusion / device
    - process rows are sourced from `nvidia-smi` and parsed from `/proc/<pid>/cmdline`.
    - ETA fields are now included per process by inferring `results.csv` path from training args and parsing `epoch`/`elapsed_hms`:
      - `EPOCH_PROGRESS`, `ETA_DURATION`, `ETA_FINISH`, `ETA_STATUS`
  - Added `scripts/eta_monitor.py` for CSV-based ETA estimation:
    - reads `results.csv` (`epoch`, `elapsed_hms`) and computes ETA from full-epoch average
    - supports one-shot output and periodic watch mode (`--watch --interval`)
    - default total epochs is `500` with override via `--total-epochs`
  - Added `scripts/eta_monitor.md` usage guide for ETA monitoring commands and interpretation.
- Telegram notifier policy alignment
  - `scripts/telegram_alert.py` now allows `session` job alerts by default to match repository policy in `AGENTS.md`.
  - Optional explicit skip is available via `--skip-session-alert` while keeping `--allow-session-alert` as compatibility flag.
  - Alert payload now always includes explicit execution metadata lines:
    - `Server` (hostname)
    - `Folder` (basename of current working directory)
    - `Summary` (from `--summary` or fallback to `--job`)
  - Added optional `--summary` flag for explicit completed-work summary text.
- Coarse directional grouping extension
  - Added fork-specific `coarse24to8` path that builds 24 proto-direction outputs internally, fuses them into 8 coarse directional clusters, and feeds the canonical 8-direction DconnNet branch.
  - Group definition/angle metadata/fusion blocks are isolated in `model/coarse_direction_grouping.py`.
  - Coarse groups are now defined directly in canonical-8 order (`SE,S,SW,E,W,NE,N,NW`) instead of legacy `G1..G8` order.
  - Canonical output reorder stage was removed; grouped outputs are canonical by construction.
  - Supported fusion modes: `mean`, `weighted_sum`, `conv1x1`, `attention_gating`.
  - Default grouped-mode fusion is `weighted_sum`.
  - CLI gating is explicit via `--direction_grouping` and `--direction_fusion`.

## Deprecated / Removed From Active Context

- Old per-day exhaustive changelog entries are removed from this file.
- Historical implementation details remain in Git history.
- Legacy references to `scripts/aggregate_kfold_results.py` are no longer treated as primary workflow.

## Known Constraints

- `scripts/aggregate_results.py` currently logs generation of experiment-mean LaTeX/CSV; dataset-bundle PDF is generated, while standalone experiment-mean PDF generation depends on explicit `build_pdf` call path.
- Runtime metrics/warnings from third-party metric utilities (for example clDice edge cases) may still appear during smoke runs.
- Grouped coarse-direction mode no longer preserves legacy `G1..G8` compatibility order; metadata and indices follow canonical-8 direction order directly.

## Next Practical Steps

1. Keep one canonical aggregation script (`aggregate_results.py`) and retire legacy script paths from docs/scripts.
2. Add a small regression test fixture for aggregation input layouts:
   - single-run (`final_results.csv`)
   - indexed (`final_results_1.csv`)
   - mixed dataset roots.
3. Keep PROJECT_CONTEXT files concise; append only currently actionable state.
