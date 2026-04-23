# Project Modification Snapshot

Last updated: 2026-04-24

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
  - Added additive `cremi` dataset branch using `getdataset_cremi` (`data/CREMI/{train,test}`).
  - Extended `--dist_aux_loss` choices with `cl_dice` for distance-mode auxiliary supervision selection.
- `connect_loss.py`
  - Added differentiable soft-clDice (soft skeletonization) branch in `dist_aux_regression_loss`:
    - new supported value: `dist_aux_loss='cl_dice'`
    - compatibility fix: previously empty `cl_dice` branch now returns a valid loss.
  - Refactored dist auxiliary loss internals into `src/losses/dist_aux.py` and kept
    `connect_loss.dist_aux_regression_loss(...)` as a backward-compatible wrapper.
  - Reverted `cl_dice` CPU offload and restored GPU-path computation.
  - Aligned `soft_cldice_loss` core formula with upstream `jocpae/clDice` style
    (`tprec/tsens` and `1 - 2*(tprec*tsens)/(tprec+tsens)`).
  - Stabilized dist+clDice training path by applying `cl_dice` to vote-map supervision
    and using SmoothL1 for dense affinity/bicon regression terms when `dist_aux_loss=cl_dice`
    (prevents high-resolution affinity-map OOM/instability without changing CLI/experiment name).
- `solver.py`
  - ISIC precision/accuracy persistence in saved metrics.
  - Extended precision/accuracy logging condition to include `cremi` (single/multi eval branches).
  - Unified CSV writing around `results.csv` and `<split>_results.csv`.
  - Epoch-level `results.csv` rows are now written even when `val_loader` is `None`, so `elapsed_hms`-based ETA tracking remains available in no-validation runs (val metrics stay `NaN`).
  - Added validation-metric monitoring controls with Dice-first defaults:
    - `monitor_metric=val_dice` (default), optional `val_loss`
    - patience-based early stopping with boundary-stop interval (`10` by default)
    - optional Dice tie-break with lower validation loss (`tie_break_with_loss`, `early_stopping_tie_eps`)
  - Best checkpoint flow now saves:
    - backward-compatible `models/best_model.pth` (raw `state_dict`)
    - rich `models/checkpoint_best.pth` (`model_state_dict`, `optimizer_state_dict`, epoch, monitor metadata)
    - expanded `best_model_meta.txt` (`best_val_dice`, `best_val_loss`, monitor metric, tie-break flag)
  - Added optional `save_best_only` behavior to skip periodic epoch checkpoints.
- `train.py`
  - Added configurable training-control CLI args:
    - `--monitor_metric`
    - `--early_stopping_patience`
    - `--early_stopping_min_delta`
    - `--early_stopping_tie_eps`
    - `--early_stopping_stop_interval`
    - `--tie_break_with_loss` / `--no_tie_break_with_loss`
    - `--save_best_only` / `--no_save_best_only`
- `scripts/aggregate_results.py`
  - Single-run file support (`final_results.csv`, legacy `results.csv`).
  - Fold/indexed compatibility (`final_results_<fold>.csv`, optional legacy `results_<fold>.csv`).
  - Fixed legacy-path consistency: fold completion checks and fold aggregation now both allow
    legacy `results.csv` fallback (for example single-run roots without `final_results_1.csv`),
    preventing `FileNotFoundError` during aggregation.
  - Single-experiment aggregation path enabled.
  - Added `cremi` dataset canonicalization in dataset-name parsing.
  - CREMI metric tables now use `Dice/IoU/Accuracy/Precision` (same policy as ISIC).
  - Dataset-split summary artifacts and dataset-bundle PDF support.
  - Multi-root aggregation now skips per-root `{dataset}_*.pdf` exports; only single-root summary PDF is emitted.
  - Sample-visualization model-dir fallback (`models/<fold>` -> `models`).
  - Experiment-name metadata parser now supports grouped-direction suffixes
    (for example `binary_8_bce_24to8_weighted_sum`).
  - Experiment-mean CSV/LaTeX and sample-visualization CSV now include
    `direction_grouping` / `direction_fusion` metadata fields.
- Unified dataset launcher
  - `scripts/train_launcher.sh` now runs in config-first mode:
    - `scripts/train_launcher.sh --config <yaml> [--device N] [--dry_run]`
    - launcher-side `while/case` CLI parsing was removed and moved into a Python config helper.
  - Added `scripts/train_launcher_from_config.py` to parse YAML, build schedules, and run/dry-run generated `train.py` commands.
  - Mode-aware dataset schema support was added:
    - `single` mode requires `dataset` as a string.
    - `multi` mode accepts `dataset` as a string or list of strings, and list inputs run each dataset with the existing per-dataset sweep policy.
  - Multi-dataset schedule support now allows config such as:
    - `dataset: [chase, drive, octa500]` with `mode: multi`
    - non-OCTA datasets use standard multi combinations, while `octa500` additionally expands `octa_variants`.
  - Added launcher config templates under `scripts/configs/` for all dataset single/multi wrappers.
  - Added CREMI launcher support:
    - `scripts/train_launcher_from_config.py` now accepts `dataset: cremi`
      with preset (`data_root=data/CREMI`, `resize=256x256`, `batch_size=4`).
    - Added single-run config template: `scripts/configs/cremi_train.yaml`.
  - Grouped-direction options (`direction_grouping`, `direction_fusion`) are now configured in YAML.
  - `direction_fusion` now supports both scalar and list forms in launcher YAML.
    - list form (for example `[mean, conv1x1, attention_gating]`) expands schedule runs per fusion option.
    - scalar form remains backward-compatible with previous configs.
  - `24to8` conn policy remains enforced:
    - single mode: `conn_num=24` is rejected
    - multi mode: conn sweep is normalized to `[8]`
  - `scripts/train_launcher.sh` preflight config existence check now resolves relative `--config` paths against repo root as a fallback, so invoking the launcher from outside repo root remains consistent.
  - Existing dataset-specific launchers remain deprecated wrappers for one-version compatibility and now forward with mapped `--config` paths.
  - Added `scripts/gpu_train_process_summary.sh` to inspect active GPU compute PIDs and summarize live training args:
    - dataset / conn_num / label_mode / dist_aux_loss / direction_grouping / direction_fusion / device
    - process rows are sourced from `nvidia-smi` and parsed from `/proc/<pid>/cmdline`.
    - ETA fields are now included per process by inferring `results.csv` path from training args and parsing `epoch`/`elapsed_hms`:
      - `EPOCH_PROGRESS`, `ETA_DURATION`, `ETA_FINISH`, `ETA_STATUS`
    - `results.csv` lookup now supports `--output_dir` passed as output root, dataset root, or experiment root
      (for example `output/chase/binary_8_bce_24to8_weighted_sum`).
    - Added schedule-progress auto-detection mode (no extra argument):
      - infers launcher config path from active `train.py` process ancestry (`/proc/<pid>/status`, `/proc/<pid>/cmdline`).
      - prints `Remaining/Total` counts and a remaining experiment table (`DATASET`, `EXPERIMENT`, `TARGET_FOLD`).
      - remaining-state check uses `final_results.csv` / `final_results_*.csv` and ignores currently running experiments.
  - Added `scripts/eta_monitor.py` for CSV-based ETA estimation:
    - reads `results.csv` (`epoch`, `elapsed_hms`) and computes ETA from full-epoch average
    - supports one-shot output and periodic watch mode (`--watch --interval`)
    - default total epochs is `500` with override via `--total-epochs`
  - Added `scripts/eta_monitor.md` usage guide for ETA monitoring commands and interpretation.
  - Added passthrough support for optional early-stopping/checkpoint controls in YAML (`single`/`multi` blocks):
    - `monitor_metric`
    - `early_stopping_patience`
    - `early_stopping_min_delta`
    - `early_stopping_tie_eps`
    - `early_stopping_stop_interval`
    - `tie_break_with_loss`
    - `save_best_only`
  - Added test-only passthrough support in YAML (`single`/`multi` and optional top-level defaults):
    - `test_only` (boolean) to append `--test_only`
    - `pretrained` (optional string) for explicit checkpoint path
    - when `pretrained` is omitted in test-only mode, launcher now auto-resolves:
      - `<output_dir>/<dataset>/<experiment_name>/models/best_model.pth`
    - optional `output_dir` is now passed through to `train.py` and also used for auto-resolving test-only checkpoint paths
    - `pretrained` supports format placeholders:
      - `{dataset}`, `{conn_num}`, `{label_mode}`, `{dist_aux_loss}`, `{direction_grouping}`, `{direction_fusion}`, `{experiment_name}`
  - Added launcher CLI passthrough for test-only runs:
    - `scripts/train_launcher.sh --config ... --test_only [--pretrained [PATH]]`
    - CLI `--test_only` / `--pretrained` now override top-level config defaults before schedule build
    - CLI `--pretrained` without `PATH` enables auto checkpoint resolution
    - CLI/YAML `pretrained` also supports models-relative shorthand:
      - `best_model.pth`, `checkpoint_best.pth`, `subdir/file.pth`
      - resolved as `<output_dir>/<dataset>/<experiment_name>/models/<value>`
- Telegram notifier policy alignment
  - `scripts/telegram_alert.py` now allows `session` job alerts by default to match repository policy in `AGENTS.md`.
  - Optional explicit skip is available via `--skip-session-alert` while keeping `--allow-session-alert` as compatibility flag.
  - `.env` resolution now falls back to repository-root relative to `scripts/telegram_alert.py` when current working directory differs.
  - `scripts/train_launcher_from_config.py` now passes explicit `--env-file <repo_root>/.env` and includes both `--summary` and `--message` payload fields.
  - Alert payload now always includes explicit execution metadata lines:
    - `Server` (hostname)
    - `Folder` (basename of current working directory)
    - `Summary` (from `--summary` or fallback to `--job`)
  - Added optional `--summary` flag for explicit completed-work summary text.
- Coarse directional grouping extension
  - Added fork-specific `24to8` path that builds 24 proto-direction outputs internally, fuses them into 8 coarse directional clusters, and feeds the canonical 8-direction DconnNet branch.
  - Group definition/angle metadata/fusion blocks are isolated in `model/coarse_direction_grouping.py`.
  - Coarse groups are now defined directly in canonical-8 order (`SE,S,SW,E,W,NE,N,NW`) instead of legacy `G1..G8` order.
  - Canonical output reorder stage was removed; grouped outputs are canonical by construction.
  - Supported fusion modes: `mean`, `weighted_sum`, `conv1x1`, `attention_gating`.
  - Default grouped-mode fusion is `weighted_sum`.
  - CLI gating is explicit via `--direction_grouping` and `--direction_fusion`.
  - Added inference-time fusion map export (additive, backward-compatible):
    - `CoarseDirectionReducer.forward(..., return_maps=True)` now returns `(fused, fusion_maps)`.
    - `weighted_sum`: returns `logits` and `weight_map`.
    - `conv1x1`: returns `conv_map`.
    - `attention_gating`: returns `attention_map`.
- Runtime shape-alignment fix for launcher resize path
  - `data_loader/GetDataset_CHASE.py` now uses runtime `args.resize[:2]` in `MyDataset_CHASE`, `MyDataset_DRIVE`, and `MyDataset_OCTA500` (`__getitem__`).
  - `scripts/train_launcher_from_config.py` presets continue to provide dataset defaults (for example DRIVE `512x512`) and can be overridden by runtime args.
  - impact:
    - keeps dataloader sample shape aligned with solver translation matrices (`--resize`).
    - resolves DRIVE 24to8 runtime crash (`RuntimeError: shape '[-1, 512, 512]' is invalid for input of size 3686400`).
- CREMI dataloader + offline distance-map prep
  - Added `data_loader/GetDataset_CREMI.py`:
    - reads `data/CREMI/{train,test}/images/*.npy`
    - supports `label_mode`:
      - `binary`: `labels/<stem>.npy`
      - `dist`: `labels_dist/<stem>_dist.npy`
      - `dist_inverted`: `labels_dist_inverted/<stem>_dist_inverted.npy`
    - returns training samples as `(image, mask)` and eval samples as `(image, mask, name)` to stay compatible with `solver.py`.
    - normalizes grayscale input into 3-channel DconnNet input and resizes by runtime `args.resize`.
  - Added CREMI offline prep notebook:
    - `notebooks/distance_map_cremi.ipynb`
    - generates `labels_dist` / `labels_dist_inverted` via `distance_transform_edt` and 0-1 normalization.
- RETOUCH dataset conversion utility (training volumes only)
  - Added `data_loader/prepare_retouch_dataset.py` to convert RETOUCH raw training volumes
    (`oct.mhd`, `reference.mhd`) into PNG slices.
  - Output structure is K-fold friendly and unified by device:
    - `data/retouch/<Device>/all/TRAINxxx/orig/*.png`
    - `data/retouch/<Device>/all/TRAINxxx/mask/*.png`
  - Device mapping preserves existing training naming compatibility:
    - output `Spectrailis` maps from source `TrainingSpectralis/RETOUCH-TrainingSet-Spectralis`.
  - Mask values are preserved (`0/1/2/3`), and non-uint8 OCT volumes are converted to uint8
    by per-volume min-max scaling for consistent PNG export.
  - Utility supports `--dry-run`, `--overwrite`, and `--max-volumes-per-device` for safe smoke checks.
- RETOUCH paper-aligned launcher/profile updates
  - Added RETOUCH launcher config files:
    - `scripts/configs/retouch_train.yaml`
    - `scripts/configs/retouch_multi_train.yaml`
    - `scripts/configs/retouch_{cir,spe,top}_train.yaml`
  - Updated RETOUCH launcher preset in `scripts/train_launcher_from_config.py`:
    - default `lr_update`: `poly` (was `step`)
    - default `use_SDL`: `False` (was `True`)
    - kept paper-sized input preset (`256x256`) and 3-fold policy.
  - Migrated legacy `scripts/retouch_{cir,spe,top}.sh` to deprecated wrappers over config-first launcher.
  - Disabled RETOUCH train-time random flip augmentation in `data_loader/GetDataset_Retouch.py`
    to match paper statement (`no data augmentation` for RETOUCH).

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
