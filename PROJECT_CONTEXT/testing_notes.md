# Testing Notes (Condensed)

Last updated: 2026-04-24

## Scope

This file keeps only currently relevant validation results.
Historical step-by-step smoke logs were removed for brevity.

## Current Validation Matrix

### A. Aggregation (`scripts/aggregate_results.py`)

- Syntax check
  - `.venv/bin/python -m py_compile scripts/aggregate_results.py` passed
- Legacy-only final summary fallback regression check
  - command:
    - `uv run scripts/aggregate_results.py`
  - result: passed
  - note:
    - verified no `FileNotFoundError` for roots that only have legacy `results.csv`
      (for example `output/cremi/dist_8_gjml_sf_l1`)
- CREMI dataset metric-spec smoke
  - command:
    - `.venv/bin/python - <<'PY' ... canonical_dataset_name('cremi'), dataset_metric_specs('cremi'), dataset_fold_metric_specs('cremi') ... PY`
  - result: passed
  - checked fields:
    - `canonical_dataset_name('cremi') == 'cremi'`
    - `dataset_metric_specs('cremi') == [('best_dice','Dice',True), ('best_jac','IoU',True), ('best_accuracy','Accuracy',True), ('best_precision','Precision',True)]`
    - `dataset_fold_metric_specs('cremi') == [('best_dice','Dice'), ('best_jac','IoU'), ('best_accuracy','Accuracy'), ('best_precision','Precision')]`
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
- Direction-grouping metadata parse smoke
  - command:
    - `.venv/bin/python - <<'PY' ... parse_experiment_metadata('binary_8_bce_24to8_weighted_sum') ... PY`
  - result: passed
  - checked fields:
    - `experiment='binary'`
    - `conn_num=8`
    - `loss='bce'`
    - `direction_grouping='24to8'`
    - `direction_fusion='weighted_sum'`
- Direction-grouping aggregate fixture smoke
  - command:
    - `cp output/chase/binary_8_bce/final_results_1.csv /tmp/agg_direction_fixture/chase/binary_8_bce_24to8_weighted_sum/final_results_1.csv`
    - `.venv/bin/python scripts/aggregate_results.py --input-root /tmp/agg_direction_fixture/chase --output-dir /tmp/agg_direction_fixture_out --sample-vis-count 0`
  - result: passed
  - key check:
    - `/tmp/agg_direction_fixture_out/dump/summary_experiment_means.csv` includes `direction_grouping,direction_fusion` columns with `24to8,weighted_sum`

### B. Training/Eval Core

- Coarse-direction fusion map export smoke
  - syntax check:
    - `.venv/bin/python -m py_compile model/coarse_direction_grouping.py tests/test_coarse_direction_grouping.py` passed
  - runtime smoke:
    - `.venv/bin/python - <<'PY' ... CoarseDirectionReducer(..., return_maps=True) ... PY` passed
  - key checks:
    - default call path unchanged: `reducer(x)` returns fused tensor only
    - `weighted_sum` maps include `logits`, `weight_map`
    - `conv1x1` maps include `conv_map`
    - `attention_gating` maps include `attention_map`
    - `mean` maps are empty dict (`{}`)

- `train.py`, `solver.py` syntax check
  - `.venv/bin/python -m py_compile train.py solver.py` passed
- CREMI integration smoke (`getdataset_cremi` + `train.py/solver.py`)
  - syntax/import check:
    - `.venv/bin/python -m py_compile train.py solver.py data_loader/GetDataset_CREMI.py` passed
  - offline dist artifact generation (CREMI):
    - generated `data/CREMI/{train,test}/labels_dist/*_dist.npy`
    - generated `data/CREMI/{train,test}/labels_dist_inverted/*_dist_inverted.npy`
    - count check passed:
      - `train`: labels=94, dist=94, dist_inverted=94
      - `test`: labels=31, dist=31, dist_inverted=31
  - dataloader sample smoke:
    - `label_mode in {binary, dist, dist_inverted}` for both train/test splits loads successfully
    - sample tensor shapes:
      - image: `(3, 256, 256)`
      - mask: `(1, 256, 256)`
  - end-to-end train-loop smoke:
    - command:
      - `timeout 240s .venv/bin/python train.py --dataset cremi --data_root data/CREMI --resize 256 256 --num-class 1 --batch-size 1 --epochs 1 --lr 0.0038 --lr-update poly --folds 1 --conn_num 8 --label_mode binary --dist_aux_loss smooth_l1 --dist_sf_l1_gamma 1.0 --direction_grouping none --direction_fusion weighted_sum --device 3 --output_dir output`
    - result:
      - completed epoch 1 + final test eval (`FINISH.`)
      - no shape mismatch or batch-unpack/runtime errors in `solver.py`
- Seed reproducibility wiring check
  - `train.py` seed path compile/usage check passed
- Dist auxiliary `cl_dice` integration smoke
  - syntax/import check:
    - `.venv/bin/python -m py_compile connect_loss.py train.py` passed
    - `.venv/bin/python -m py_compile src/losses/dist_aux.py tests/test_dist_aux_loss_selection.py` passed
  - launcher dry-run check:
    - `uv run bash scripts/train_launcher.sh --config scripts/configs/direction_grouping_multi_train.yaml --dry_run` passed
    - generated `dist`/`dist_inverted` commands for `chase` and `drive` with `--dist_aux_loss cl_dice`
  - runtime check:
    - inline python smoke with `connect_loss(dist_aux_loss='cl_dice')` passed
    - finite-loss check on dense tensors: `True`
    - identical-input near-zero and mismatch-separation check on binary tensors passed (`same_loss=0.0`, `diff_loss>0`)
    - wrapper/direct parity check passed:
      - `connect_loss.dist_aux_regression_loss(...)` equals `src.losses.dist_aux_regression_loss(...)` for `cl_dice`
    - backward check passed for modular `cl_dice` path (`loss.backward()` + finite grad)
    - CUDA tensor check passed:
      - input on `cuda:0` -> returned loss on `cuda:0`
      - backward gradient finite on CUDA input tensor
    - clDice stabilization check passed:
      - `dist_aux_loss=cl_dice` with non-CHASE dist mode yields finite `affinity`, `bicon`, and `total` loss terms.
  - OOM regression check:
    - command:
      - `timeout 240s uv run bash scripts/train_launcher.sh --config scripts/configs/direction_grouping_multi_train.yaml`
    - result:
      - after stabilization patch, run progressed through multiple epochs (`epoch 0` to `epoch 3`) without prior `cl_dice` CUDA OOM
      - training loss trended down (`~3.54` -> `~2.90`) with no NaN/Inf tracebacks in the timed run window
  - overflow/finite check (short run):
    - command:
      - `timeout 180s uv run /home/suhohan/DconnNet/.venv/bin/python /home/suhohan/DconnNet/train.py --dataset chase --data_root data/chase --resize 960 960 --num-class 1 --batch-size 4 --epochs 1 --lr 0.0038 --lr-update poly --folds 1 --conn_num 8 --label_mode dist --dist_aux_loss cl_dice --dist_sf_l1_gamma 1.0 --direction_grouping none --direction_fusion weighted_sum --device 3 --output_dir output`
    - result:
      - completed epoch 0 + final test eval (`FINISH.`)
      - no overflow/NaN/Inf traceback observed in this run
  - note:
    - `.venv` currently does not include `pytest` (`No module named pytest`), so `tests/test_dist_aux_loss_selection.py` was not executed via pytest in this pass.
- Dice-based early stopping / best-checkpoint control checks
  - syntax/import check:
    - `.venv/bin/python -m py_compile train.py solver.py scripts/train_launcher_from_config.py` passed
  - CLI surface check:
    - `.venv/bin/python train.py --help | rg "monitor_metric|early_stopping|tie_break|save_best_only"` shows new options and aliases
  - launcher default passthrough check (no overrides set):
    - `.venv/bin/python scripts/train_launcher_from_config.py --config scripts/configs/isic2018_train.yaml --dry_run`
    - generated command does not append early-stopping flags (defaults are handled in `train.py`)
  - launcher explicit passthrough check (overrides set):
    - `/tmp/isic_train_es.yaml` (single-run config with `monitor_metric`, `early_stopping_*`, `tie_break_with_loss`, `save_best_only`)
    - `.venv/bin/python scripts/train_launcher_from_config.py --config /tmp/isic_train_es.yaml --dry_run`
    - generated command includes expected flags:
      - `--monitor_metric val_dice`
      - `--early_stopping_patience 3`
      - `--early_stopping_min_delta 0.001`
      - `--early_stopping_tie_eps 0.0001`
      - `--early_stopping_stop_interval 10`
      - `--tie_break_with_loss`
      - `--save_best_only`
  - EarlyStopping helper behavior unit smoke:
    - inline python snippet (`from solver import EarlyStopping`) passed
    - boundary-stop behavior verified:
      - patience exceeded before epoch 10 -> `waiting_for_boundary=True`
      - stop signal emitted only at epoch 10 (`should_stop=True`)
    - Dice tie-break behavior verified:
      - near-equal Dice within `tie_eps` and lower `val_loss` triggers best update with `used_tie_break=True`
  - note:
    - full end-to-end train-loop smoke for real dataset + GPU run was not executed in this pass.
- OCTA500 launcher-based end-to-end smoke (single-run reduced multi config)
  - config:
    - `/tmp/octa500_multi_smoke.yaml` (derived from `scripts/configs/octa500_multi_train.yaml`)
    - reduced scope: `epochs=1`, `conn_nums=[8]`, `label_modes=[binary]`, `dist_aux_losses=[smooth_l1]`, `octa_variants=[6M]`
  - command:
    - `bash scripts/train_launcher.sh --config /tmp/octa500_multi_smoke.yaml`
  - result: passed (`FINISH.`)
  - key runtime logs:
    - `Train batch number: 12`, `Validation batch number: 20`, `Test batch number: 100`
    - epoch/validation/final-test evaluation completed
  - outputs:
    - `output/octa500-6M/binary_8_bce/` (including `results.csv`, `final_results.csv`, `models/best_model.pth`)
  - observed warnings:
    - `metrics/cldice.py` divide warning on empty skeleton cases
    - `np.nanmean` warnings for precision/accuracy (non-ISIC path keeps these metrics as NaN)

### C. Launchers and Shell Scripts (latest relevant)

- Config-first launcher syntax checks passed:
  - `bash -n scripts/train_launcher.sh`
  - `bash -n scripts/chasedb1_train.sh scripts/chasedb1_multi_train.sh scripts/drive_train.sh scripts/drive_multi_train.sh`
  - `bash -n scripts/isic2018_train.sh scripts/isic2018_multi_train.sh scripts/octa500_train.sh scripts/octa500_multi_train.sh`
  - `bash -n scripts/gpu_train_process_summary.sh`
  - `.venv/bin/python -m py_compile scripts/train_launcher_from_config.py scripts/eta_monitor.py`
  - test-only passthrough update syntax check:
    - `.venv/bin/python -m py_compile scripts/train_launcher_from_config.py` passed
  - test-only dry-run checks passed:
    - single (auto checkpoint path):
      - `.venv/bin/python scripts/train_launcher_from_config.py --config /tmp/octa500_single_test_only.yaml --dry_run`
      - generated command includes `--test_only`, `--pretrained`, `--output_dir`
    - multi (explicit checkpoint format string):
      - `.venv/bin/python scripts/train_launcher_from_config.py --config /tmp/isic_multi_test_only.yaml --dry_run`
      - generated command includes `--test_only` and formatted `--pretrained` path
    - launcher CLI passthrough (`train_launcher.sh`) check:
      - `uv run bash scripts/train_launcher.sh --config scripts/configs/octa500_train.yaml --test_only --pretrained --dry_run`
      - parser accepts CLI flags and generated command includes auto-resolved `--pretrained .../models/best_model.pth` with `--test_only`
    - models-relative pretrained shorthand check:
      - `uv run bash scripts/train_launcher.sh --config scripts/configs/octa500_train.yaml --test_only --pretrained best_model.pth --dry_run`
      - generated command resolves to `.../models/best_model.pth`
      - `uv run bash scripts/train_launcher.sh --config scripts/configs/octa500_train.yaml --test_only --pretrained checkpoint_best.pth --dry_run`
      - generated command resolves to `.../models/checkpoint_best.pth`
- Config-first launcher CLI checks passed:
  - `scripts/train_launcher.sh --config scripts/configs/drive_multi_train.yaml --dry_run` produced 10 runs
  - `scripts/train_launcher.sh --config scripts/configs/drive_multi_train.yaml --dry_run --device 3` reflected device override
  - direction_fusion list sweep check passed:
    - `scripts/train_launcher.sh --config scripts/configs/direction_grouping_multi_train.yaml --dry_run`
    - with `direction_fusion: [mean, conv1x1, attention_gating]`, generated 9 runs (3 label modes x 3 fusion modes)
    - each generated command contained the expected `--direction_fusion` value
  - direction-grouping runtime debug (DRIVE + 24to8):
    - failing smoke (pre-fix):
      - `uv run python train.py --dataset drive --data_root data/DRIVE --resize 960 960 --num-class 1 --batch-size 4 --epochs 1 --lr 0.0038 --lr-update poly --folds 1 --conn_num 8 --label_mode binary --dist_aux_loss smooth_l1 --dist_sf_l1_gamma 1.0 --direction_grouping 24to8 --direction_fusion weighted_sum --device 1`
      - reproduced traceback:
        - `RuntimeError: shape '[-1, 512, 512]' is invalid for input of size 3686400`
      - failure location:
        - `connect_loss.py::Bilateral_voting` via `solver.py::train`
    - root cause:
      - translation matrices in `solver.py` follow `--resize` (960x960), while DRIVE/OCTA dataset loaders were hardcoded to resize samples at 512x512.
    - post-fix validation (args.resize unification approach):
      - `data_loader/GetDataset_CHASE.py` loaders (`CHASE/DRIVE/OCTA500`) now resize via `args.resize[:2]`.
      - 1-epoch smoke run passed:
        - `uv run /home/suhohan/DconnNet/.venv/bin/python train.py --dataset drive --data_root data/DRIVE --resize 512 512 --num-class 1 --batch-size 4 --epochs 1 --lr 0.0038 --lr-update poly --folds 1 --conn_num 8 --label_mode binary --dist_aux_loss smooth_l1 --dist_sf_l1_gamma 1.0 --direction_grouping 24to8 --direction_fusion weighted_sum --device 1`
        - training + final test eval completed (`FINISH.`) without shape mismatch.
  - launcher from non-repo cwd check passed:
    - `(cd /tmp && /home/suhohan/DconnNet/scripts/train_launcher.sh --config scripts/configs/drive_multi_train.yaml --dry_run)` succeeded via repo-root fallback config resolution
  - wrapper compatibility check passed:
    - `scripts/drive_multi_train.sh --dry_run` shows deprecation warning and forwards to mapped config launcher
  - CREMI single-config dry-run check passed:
    - `.venv/bin/python scripts/train_launcher_from_config.py --config scripts/configs/cremi_train.yaml --dry_run`
    - generated command contains:
      - `--dataset cremi`
      - `--data_root data/CREMI`
      - `--resize 256 256`
  - invalid/failure scenario checks passed:
    - missing config path fails with non-zero exit
    - malformed YAML config fails with non-zero exit
    - `single + 24to8 + conn_num=24` config fails with non-zero exit
    - `multi + 24to8 + conn_nums=[8,24]` config normalizes to conn sweep `[8]`
    - `multi + dataset=[chase,unknown_dataset]` fails with explicit unsupported-dataset error
    - `single + dataset=[...]` fails with explicit schema error (`single mode requires dataset to be a string`)
    - ISIC readiness check remains enforced for multi dataset-list path:
      - temporary launcher-root simulation with `dataset: [isic2018]` fails when `data/ISIC2018/{image,label}` is absent
- Telegram notifier script path is operational (`scripts/telegram_alert.py`).
- GPU train-process summary helper:
  - `bash -n scripts/gpu_train_process_summary.sh` passed
  - `scripts/gpu_train_process_summary.sh --help` passed
  - schedule progress auto-inference check passed with mocked `nvidia-smi` + fake launcher parent:
    - launcher: `.venv/bin/python /tmp/fake_launcher.py --config scripts/configs/drive_multi_train.yaml`
    - summary: `FAKE_GPU_PID=<pid> PATH="/tmp/codex_fakebin:$PATH" bash scripts/gpu_train_process_summary.sh`
    - key check: output includes `Remaining/Total` and remaining experiment table.
  - command-path compatibility update (results.csv inference):
    - script now resolves all of the following `--output_dir` forms:
      - output root: `output`
      - dataset root: `output/chase`
      - experiment root: `output/chase/binary_8_bce_24to8_weighted_sum`
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
- Launcher notifier argument compatibility fixed and smoke-checked:
  - `scripts/train_launcher_from_config.py` now sends `--env-file <repo_root>/.env` and forwards both `--summary` and `--message`.
  - `.venv/bin/python -m py_compile scripts/telegram_alert.py scripts/train_launcher_from_config.py` passed.
- `.env` lookup fallback check passed from non-repo cwd:
  - `(cd /tmp && /home/suhohan/DconnNet/.venv/bin/python /home/suhohan/DconnNet/scripts/telegram_alert.py --job "codex_session_test" --status DONE --dry-run)` prints message.
- Real Telegram API send check:
  - `.venv/bin/python scripts/telegram_alert.py --job "codex_session_done_fix" --status DONE --summary "codex telegram fix" --message "codex telegram fix"` returned `Telegram alert sent.` (network-enabled execution).
- Telegram notifier metadata formatting check passed:
  - default dry-run output includes `Server`, `Folder`, `Summary` lines.
  - custom-message dry-run with explicit summary:
    - `.venv/bin/python scripts/telegram_alert.py --job "chasedb1_train(conn=8,label=binary,epochs=130)" --summary "CHASE DB1 train fold1 완료" --status DONE --message "학습 종료" --dry-run`
    - output includes custom body plus `Server`, `Folder`, `Summary`, `Status`, `Time`, `Path`.

### D. Coarse Direction Grouping (`24to8`)

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

### E. RETOUCH conversion utility (`data_loader/prepare_retouch_dataset.py`)

- Syntax check
  - `.venv/bin/python -m py_compile data_loader/prepare_retouch_dataset.py` passed
- Multi-device dry-run smoke (1 volume per device)
  - command:
    - `.venv/bin/python data_loader/prepare_retouch_dataset.py --dry-run --max-volumes-per-device 1`
  - result: passed
  - summary:
    - `Cirrus: volumes=1, slices=128`
    - `Spectrailis: volumes=1, slices=49`
    - `Topcon: volumes=1, slices=128`
- Write-path smoke (Cirrus, 1 volume)
  - command:
    - `.venv/bin/python data_loader/prepare_retouch_dataset.py --devices Cirrus --max-volumes-per-device 1 --overwrite`
  - result: passed
  - key checks:
    - output tree created:
      - `data/retouch/Cirrus/all/TRAIN001/{orig,mask}`
    - pair count check:
      - `orig=128`, `mask=128` (`256` files total)
    - mask value sanity:
      - observed labels in converted PNG masks: `[0, 1, 2]` for `TRAIN001`

### F. RETOUCH paper-alignment + cross-dataset launcher checks

- Syntax checks
  - `.venv/bin/python -m py_compile scripts/train_launcher_from_config.py data_loader/GetDataset_Retouch.py` passed
- RETOUCH launcher config dry-run checks
  - `scripts/train_launcher.sh --config scripts/configs/retouch_train.yaml --dry_run` passed
    - generated `retouch-Spectrailis` fold commands with:
      - `--resize 256 256`
      - `--lr-update poly`
      - `--folds 3 --target_fold {1,2,3}`
      - no `--use_SDL` flag
  - `scripts/train_launcher.sh --config scripts/configs/retouch_multi_train.yaml --dry_run` passed
    - generated 3 devices x 3 folds = 9 runs
    - Topcon LR remained device-specific (`--lr 0.0008`)
- Legacy RETOUCH wrappers dry-run checks
  - `bash scripts/retouch_cir.sh --dry_run` passed
  - `bash scripts/retouch_spe.sh --dry_run` passed
  - `bash scripts/retouch_top.sh --dry_run` passed
  - wrappers correctly forward to:
    - `scripts/configs/retouch_cir_train.yaml`
    - `scripts/configs/retouch_spe_train.yaml`
    - `scripts/configs/retouch_top_train.yaml`
- Other dataset one-shot launcher checks (no mutation)
  - `scripts/train_launcher.sh --config scripts/configs/chasedb1_train.yaml --dry_run` passed
  - `scripts/train_launcher.sh --config scripts/configs/isic2018_train.yaml --dry_run` passed
  - `scripts/train_launcher.sh --config scripts/configs/drive_train.yaml --dry_run` passed
  - `scripts/train_launcher.sh --config scripts/configs/octa500_train.yaml --dry_run` passed
- Paper-text alignment notes (from CVPR 2023 PDF excerpt)
  - RETOUCH:
    - resized to `256x256`
    - no data augmentation
    - `poly` LR strategy
  - ISIC:
    - resized to `224x320`
  - CHASEDB1:
    - resized to `960x960`

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
