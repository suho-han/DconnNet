# Testing Notes (Condensed)

Last updated: 2026-04-19

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

- Config-first launcher syntax checks passed:
  - `bash -n scripts/train_launcher.sh`
  - `bash -n scripts/chasedb1_train.sh scripts/chasedb1_multi_train.sh scripts/drive_train.sh scripts/drive_multi_train.sh`
  - `bash -n scripts/isic2018_train.sh scripts/isic2018_multi_train.sh scripts/octa500_train.sh scripts/octa500_multi_train.sh`
  - `bash -n scripts/gpu_train_process_summary.sh`
  - `.venv/bin/python -m py_compile scripts/train_launcher_from_config.py scripts/eta_monitor.py`
- Config-first launcher CLI checks passed:
  - `scripts/train_launcher.sh --config scripts/configs/drive_multi_train.yaml --dry_run` produced 10 runs
  - `scripts/train_launcher.sh --config scripts/configs/drive_multi_train.yaml --dry_run --device 3` reflected device override
  - wrapper compatibility check passed:
    - `scripts/drive_multi_train.sh --dry_run` shows deprecation warning and forwards to mapped config launcher
  - invalid/failure scenario checks passed:
    - missing config path fails with non-zero exit
    - malformed YAML config fails with non-zero exit
    - `single + coarse24to8 + conn_num=24` config fails with non-zero exit
    - `multi + coarse24to8 + conn_nums=[8,24]` config normalizes to conn sweep `[8]`
- Telegram notifier script path is operational (`scripts/telegram_alert.py`).
- GPU train-process summary helper:
  - `bash -n scripts/gpu_train_process_summary.sh` passed
  - `scripts/gpu_train_process_summary.sh --help` passed
  - runtime query check in current Codex sandbox returns:
    - `[ERROR] Failed to query GPU metadata via nvidia-smi.`
  - note: this is environment-limited (no accessible NVIDIA driver in sandbox), so live GPU-process table output is not verified here.
  - ETA integration note:
    - output schema now includes `EPOCH_PROGRESS`, `ETA_DURATION`, `ETA_FINISH`, `ETA_STATUS` columns from `results.csv` parsing.
- ETA monitor helper:
  - one-shot ETA check passed:
    - `.venv/bin/python scripts/eta_monitor.py --csv output/octa500-3M/dist_inverted_24_gjml_sf_l1/results.csv`
  - watch mode check passed (time-limited run):
    - `timeout 3s .venv/bin/python -u scripts/eta_monitor.py --csv output/octa500-3M/dist_inverted_24_gjml_sf_l1/results.csv --watch --interval 1`
  - failure scenario checks passed:
    - missing csv path returns non-zero
    - invalid `--total-epochs`/`--interval` returns non-zero
    - missing required CSV column (`elapsed_hms`) returns non-zero
- Telegram notifier session policy check passed:
  - `.venv/bin/python scripts/telegram_alert.py --job "codex_session_test" --status DONE --dry-run` prints message (no default skip).
  - `.venv/bin/python scripts/telegram_alert.py --job "codex_session_test" --status DONE --dry-run --skip-session-alert` prints policy skip.
- Telegram notifier metadata formatting check passed:
  - default dry-run output includes `Server`, `Folder`, `Summary` lines.
  - custom-message dry-run with explicit summary:
    - `.venv/bin/python scripts/telegram_alert.py --job "chasedb1_train(conn=8,label=binary,epochs=130)" --summary "CHASE DB1 train fold1 완료" --status DONE --message "학습 종료" --dry-run`
    - output includes custom body plus `Server`, `Folder`, `Summary`, `Status`, `Time`, `Path`.

### D. Coarse Direction Grouping (`coarse24to8`)

- Syntax check
  - `.venv/bin/python -m py_compile model/coarse_direction_grouping.py model/DconnNet.py tests/test_coarse_direction_grouping.py` passed
- Canonical-first conversion smoke script
  - result: passed
  - covered:
    - 24-channel repo ordering -> canonical coarse group map (`SE,S,SW,E,W,NE,N,NW`)
    - mean-vector representative angle calculation in image coordinates
    - fusion blocks: `mean`, `weighted_sum`, `conv1x1`, `attention_gating`
    - grouped reducer output is canonical by construction (no reorder step)
    - grouped `DconnNet` forward path: internal 24 proto-directions -> final 8 canonical directions
    - baseline `direction_grouping='none'` path shape smoke remained valid
- Pytest availability
  - `.venv/bin/python -m pytest ...` could not run because `pytest` is not installed in `.venv`
- Step-by-step visualization notebook
  - Added `tests/test_coarse_direction_grouping.ipynb`
  - purpose: interactive, visual reproduction of checks in `tests/test_coarse_direction_grouping.py`
  - includes: group-index mapping table/plot, mean-vector angle plots, fusion output comparison, canonical-order check, grouped `DconnNet` smoke test

## Current Confidence

- Aggregation pipeline: good for current single-run layout.
- Dataset-split summary generation: verified.
- Sample visualization export: verified on CHASE output layout.
- Coarse directional grouping path: manually smoke-verified for ordering, fusion, and model forward compatibility.

## Remaining Gaps

1. Add automated regression tests for aggregation (currently manual/smoke-driven).
2. Run periodic full-root smoke when new datasets/experiments are added.
3. Install `pytest` in `.venv` or provide a test runner path so the new coarse-grouping tests can run directly.
4. Keep output naming conventions stable to avoid parser drift.
