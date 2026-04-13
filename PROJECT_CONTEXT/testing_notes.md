# Testing Notes

## Updated on 2026-04-13 (sample visualization RGB/BGR handling smoke)

- Validation target:
  - verify explicit RGB/grayscale rendering path after panel-loader change
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_sample_vis_test_rgbbgr --sample-vis-count 2` passed
- Verified output:
  - `output/summary_sample_vis_test_rgbbgr/kfold_summary_sample_visualization.png`
  - `output/summary_sample_vis_test_rgbbgr/kfold_summary_sample_visualization.csv`

## Updated on 2026-04-13 (aggregate_kfold_results sample visualization smoke)

- Validation target:
  - verify that k-fold aggregation now exports top/bottom sample comparison artifacts
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_sample_vis_test` passed
- Verified output:
  - `output/summary_sample_vis_test/kfold_summary_sample_visualization.csv`
  - `output/summary_sample_vis_test/kfold_summary_sample_visualization.png`
  - current sample-visualization selection on this repo state:
    - `Model 1`: `dist_inverted_8_gjml_sf_l1`, fold `2`, best epoch `220`
    - `Model 2`: `dist_inverted_8`, fold `2`, best epoch `259`
    - selected samples: top `04L`, `05R`; bottom `06R`, `06L`
- Observed fallback behavior:
  - `Model 1` best epoch `220` had no saved checkpoint PNG, so visualization used nearest saved epoch `225`
  - `Model 2` best epoch `259` had no saved checkpoint PNG, so visualization used nearest saved epoch `255`

## Updated on 2026-04-13 (`chasedb1_train.sh` epoch policy restore smoke)

- Validation target:
  - verify epoch default policy is restored while keeping `--epochs` override
- Executed checks:
  - `sh -n scripts/chasedb1_train.sh` passed
  - `sh scripts/chasedb1_train.sh --help` passed
- Verified output:
  - help includes restored default policy:
    - `binary + conn_num=8 -> 130`
    - `binary + conn_num=25 -> 390`
    - `dist_* -> 260`
  - `--epochs` is documented as optional override

## Updated on 2026-04-13 (`chasedb1_train.sh` condition simplification smoke)

- Note:
  - this temporary fixed-130 behavior was later superseded by:
    - `Updated on 2026-04-13 (chasedb1_train.sh epoch policy restore smoke)`

- Validation target:
  - verify simplified launcher script remains syntactically valid and prints updated help
- Executed checks:
  - `sh -n scripts/chasedb1_train.sh` passed
  - `sh scripts/chasedb1_train.sh --help` passed
- Verified output (historical at that time):
  - help showed fixed epoch default:
    - `--epochs <int> (default: 130)`
  - primary option list remains:
    - `--conn_num`
    - `--label_mode`
    - `--dist_aux_loss`
    - `--dist_sf_l1_gamma`

## Updated on 2026-04-13 (CHASE unified train launcher + wrapper compatibility smoke)

- Validation target:
  - verify consolidated script interface (`conn_num`, `label_mode`, `dist_aux_loss`, `dist_sf_l1_gamma`) and wrapper compatibility
- Executed checks:
  - `sh -n scripts/chasedb1_train.sh` passed
  - `sh -n scripts/chasedb1_train_5x5.sh` passed
  - `sh -n scripts/chasedb1_train_dist_signed.sh` passed
  - `sh -n scripts/chasedb1_train_dist_inverted.sh` passed
  - `sh scripts/chasedb1_train.sh --help` passed
  - `sh scripts/chasedb1_train_5x5.sh --help` passed
  - `sh scripts/chasedb1_train_dist_signed.sh --help` passed
  - `sh scripts/chasedb1_train_dist_inverted.sh --help` passed
- Verified output:
  - unified help text shows primary options:
    - `--conn_num`
    - `--label_mode`
    - `--dist_aux_loss`
    - `--dist_sf_l1_gamma`
    - `--epochs` (optional override)
  - wrapper scripts print the same unified help, confirming delegation path is active

## Updated on 2026-04-13 (experiment metadata split + custom ordering smoke)

- Validation target:
  - verify that cross-experiment mean table separates `experiment`, `conn_num`, `loss` and applies `dist_signed` before `dist_inverted`
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_mean_merge_test` passed
- Verified output:
  - `output/summary_mean_merge_test/kfold_summary_experiment_means.csv`
  - header: `experiment,conn_num,loss,num_folds,best_dice_mean,best_jac_mean,best_cldice_mean`
  - sample row order:
    - `binary,8,...`
    - `binary,25,...`
    - `dist_signed,8,...`
    - `dist_inverted,8,...`
  - experiment column excludes loss/conn tags (e.g., no `_gjml_sf_l1`, no trailing `_8`/`_25`)

## Updated on 2026-04-13 (experiment-mean natural-sort order smoke)

- Validation target:
  - verify `natsorted`-style experiment ordering in cross-experiment mean CSV/TEX/PDF
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_mean_merge_test` passed
- Verified output:
  - `output/summary_mean_merge_test/kfold_summary_experiment_means.csv`
  - order starts with `binary_8`, then `binary_25` (natural sort)

## Updated on 2026-04-13 (experiment-mean top-1/top-2 highlight smoke)

- Validation target:
  - verify rank highlight formatting in cross-experiment mean LaTeX/PDF output
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_mean_merge_test` passed
- Verified output:
  - `output/summary_mean_merge_test/kfold_summary_experiment_means.tex` contains rank markup:
    - best values: `\\textbf{...}`
    - second-best values: `\\underline{...}`
  - corresponding PDF regenerated successfully:
    - `output/summary_mean_merge_test/kfold_summary_experiment_means.pdf`

## Updated on 2026-04-13 (experiment-mean loss column + best-epoch removal smoke)

- Validation target:
  - verify cross-experiment mean bundle format change (`loss` column added, `best_epoch_mean` removed)
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_mean_merge_test` passed
- Verified output:
  - `output/summary_mean_merge_test/kfold_summary_experiment_means.csv`
  - header: `experiment,loss,num_folds,best_dice_mean,best_jac_mean,best_cldice_mean`
  - sample labels:
    - `binary_*` -> `BCE`
    - `dist_*` -> `Smooth-L1`
    - `*_gjml_sf_l1` -> `GJML+SF-L1`

## Updated on 2026-04-12 (aggregate_kfold_results combined mean bundle smoke)

- Validation target:
  - verify new cross-experiment mean aggregation output (`csv/tex/pdf`) from `scripts/aggregate_kfold_results.py`
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_mean_merge_test` passed
- Verified outputs:
  - per-experiment summaries remained generated as before
  - added combined mean bundle files:
    - `output/summary_mean_merge_test/kfold_summary_experiment_means.csv`
    - `output/summary_mean_merge_test/kfold_summary_experiment_means.tex`
    - `output/summary_mean_merge_test/kfold_summary_experiment_means.pdf`

## Updated on 2026-04-12 (binary edge target notebook check vs distance-derived mask)

- Added focused validation notebook:
  - `notebooks/edge_target_binary_vs_distance.ipynb`
- Validation target:
  - verify that `connect_loss.binary_edge_target_from_affinity()` exactly matches its intended formula on real CHASE samples
  - compare edge targets built from `binary` GT vs edge targets built from `dist_target > 0` masks
  - show why raw `distance_affinity_matrix(dist_target)` should not be fed directly into the binary edge-target helper
- Executed check:
  - replayed the notebook code with `.venv` and confirmed successful execution (`NOTEBOOK_CODE_OK`)
- Observations on 6 CHASE validation samples:
  - function-vs-manual comparison: mismatch ratio was `0.0` for both `binary_gt` and `dist_mask` inputs on every checked sample
  - `binary` GT vs `dist_target > 0` mask were close but not identical:
    - mean mask mismatch ratio: about `0.007736`
    - mean edge mismatch ratio: about `0.013478`
  - raw distance affinity path is clearly different from the intended binary edge target:
    - mean `binary_edge_ratio`: about `0.029219`
    - mean `dist_mask_edge_ratio`: about `0.031198`
    - mean `dist_raw_edge_ratio`: about `0.084187`
    - mean edge mismatch `binary_edge` vs `dist_raw_edge`: about `0.054967`
- Interpretation:
  - `binary_edge_target_from_affinity()` itself is behaving correctly
  - small binary-vs-dist edge differences come from GT mismatch between the stored binary mask and the stored distance-derived mask, not from the helper implementation
  - current `dist_edge_loss()` design is justified: derive a binary mask first, then build binary connectivity, then compute the edge target

## Updated on 2026-04-10 (dist_signed final-output loss/eval fix smoke validation)

- Validation target:
  - verify that the new dist loss/eval path no longer prefers the trivial zero solution and no longer uses the invalid `pred_score > 0` all-foreground conversion
- Static checks:
  - `.venv/bin/python -m py_compile connect_loss.py solver.py` passed
- Real-sample loss comparison:
  - on a CHASE sample after the patch:
    - `dist_signed` zero-logit total loss: about `4.066`
    - `dist_signed` mid-logit total loss: about `2.535`
  - this confirms the old near-zero collapse regime is no longer the lowest-cost trivial solution
- Eval threshold sanity:
  - using the new `solver.dist_score_to_binary()`:
    - constant logits `-6, -4, -2, 0` all produced mask foreground ratio `0.0`
    - this replaced the previous `> 0` behavior that turned all non-negative score maps into all-foreground masks
- Smoke run A:
  - command:
    - `.venv/bin/python train.py --dataset chase --data_root data/chase --resize 960 960 --num-class 1 --batch-size 4 --epochs 1 --lr 0.0038 --lr-update poly --folds 1 --label_mode dist_signed --output_dir output/dist_signed_lossfix_smoke`
  - result:
    - completed successfully
    - wrote `output/dist_signed_lossfix_smoke/dist_signed_8/results_1.csv`
- Smoke run B:
  - command:
    - `.venv/bin/python train.py --dataset chase --data_root data/chase --resize 960 960 --num-class 1 --batch-size 4 --epochs 3 --lr 0.0038 --lr-update poly --folds 1 --label_mode dist_signed --output_dir output/dist_signed_lossfix_smoke3`
  - observed in `output/dist_signed_lossfix_smoke3/dist_signed_8/results_1.csv`:
    - epoch1: `dice=0.159463`
    - epoch2: `dice=0.284554`
    - epoch3: `dice=0.444370`
  - best-model metadata:
    - `best_epoch=3`
    - `best_val_loss=3.043943`
  - direct checkpoint inspection on the best model:
    - example prediction foreground ratios: about `0.0567 ~ 0.0666`
    - corresponding GT foreground ratios: about `0.0766 ~ 0.0897`
  - interpretation:
    - model predictions moved away from the previous all-one eval collapse and started tracking vessel-area scale
- Smoke run C:
  - command:
    - `.venv/bin/python train.py --dataset chase --data_root data/chase --resize 960 960 --num-class 1 --batch-size 4 --epochs 30 --lr 0.0038 --lr-update poly --folds 1 --label_mode dist_signed --save-per-epochs 15 --output_dir output/dist_signed_lossfix_smoke30`
  - observed in `output/dist_signed_lossfix_smoke30/dist_signed_8/results_1.csv`:
    - epoch5: `dice=0.648412`
    - epoch12: `dice=0.741771`
    - epoch18: `dice=0.756515`
    - epoch24: `dice=0.772392`
    - epoch30: `dice=0.788337`
  - best-model metadata:
    - `best_epoch=30`
    - `best_dice=0.788337`
    - `best_val_loss=1.932942`
    - `best_jac=0.650835`
    - `best_clDice=0.801648`
  - saved prediction/mask batch snapshots at epoch 30 remained on the correct vessel-area scale:
    - `batch_0000.png`: pred foreground `0.082055`, GT `0.076635`
    - `batch_0001.png`: pred foreground `0.083303`, GT `0.083031`
    - `batch_0002.png`: pred foreground `0.085894`, GT `0.089678`
  - interpretation:
    - the revised `dist_signed` path now converges to a stable binary segmentation solution instead of the old near-zero / all-foreground failure mode

## Added on 2026-04-07

- Added CHASE-specific data-loading test for signed distance maps:
  - `tests/test_chase_dataloader_gt_dist_signed.py`
  - `tests/test_chase_dataloader_gt_dist_signed.ipynb`
- Test behavior:
  - Python test (`.py`) creates a minimal CHASE-like mock structure and validates image/GT/signed-distance loading shapes.
  - Notebook (`.ipynb`) now uses only real files from `data/chase/{img,gt,gt_dist_signed}` (mock cells removed).
  - `gt_dist_signed` file naming follows: `Image_{name}_1stHO_dist_signed.npy`
  - Added notebook cell in `notebooks/distance_map_chasedb1.ipynb` to save positive-inverted npy maps to `data/chase/gt_dist_inverted`.
- Scope:
  - Fork-specific test utility only.
  - No upstream baseline training/evaluation path changes.

## Updated on 2026-04-07 (distance_map_chasedb1 notebook)

- Added mode-validation cells to `notebooks/distance_map_chasedb1.ipynb` for `MyDataset_CHASE`:
  - User-facing modes: `binary`, `dist`, `dist_inv`
  - Internal mapping: `dist -> dist_signed`, `dist_inv -> dist_inverted`
  - Checks dataset length, tensor shapes, and per-mode value behavior.
- Validation result from `.venv` execution:
  - `binary` mode: pass (0/1 mask).
  - `dist` and `dist_inv`: warning; loaded masks become binary due to current loader thresholding path.
- Real CHASE folder compatibility check:
  - `MyDataset_CHASE` expects `data/chase/images`, but repository currently uses `data/chase/img`.
  - This mismatch is now surfaced in notebook output for explicit verification.

## Updated on 2026-04-07 (real-data validation flow)

- Revised `notebooks/distance_map_chasedb1.ipynb` validation cells to remove mock-sample generation.
- Validation now follows `train.py` CHASE split logic:
  - `overall_id = ['01' ... '14']`
  - `exp_id = 0`
  - `test_id = overall_id[0:3]`
- Removed dictionary key-mapping usage and switched to explicit mode branching:
  - requested `binary` -> `label_mode='binary'`
  - requested `dist` -> `label_mode='dist_signed'`
  - requested `dist_inv` -> `label_mode='dist_inverted'`
- For compatibility with current repository layout, notebook creates a temporary real-data bridge:
  - `img/` is symlinked as `images/`
  - `gt/`, `gt_dist_signed/` are symlinked directly
  - `gt_dist_inverted/` `.npy` files are converted to `*_dist_inverted.png` in a temp folder when PNG files are absent.
- Verified with `.venv` execution:
  - `binary`, `dist`, `dist_inv` all load successfully.
  - `dist` and `dist_inv` remain binarized in output under current loader thresholding behavior.
  - `trainset`/`validset` lengths matched expected counts (`22`, `6`) for `exp_id=0`.

## Updated on 2026-04-07 (final dataset-test visualization)

- Updated final dataset-test cell in `notebooks/distance_map_chasedb1.ipynb`:
  - Keeps `train.py`-style CHASE split and per-mode dataset length checks.
  - Added per-mode sample visualization from `validset[0]`:
    - input image
    - label map (`binary` / `dist_signed` / `dist_inverted`)
    - image-label overlay
- Adjusted temporary compatibility-root cleanup scope:
  - Build once, iterate all modes, then cleanup in outer `finally` to avoid mid-loop path removal.

## Updated on 2026-04-07 (RGB visualization fix)

- Reintroduced RGB channel reversal in the final dataset visualization cell of `notebooks/distance_map_chasedb1.ipynb`.
- Verified after re-run:
  - the preview image displays with natural colors again
  - `binary`, `dist_signed`, and `dist_inverted` checks still execute successfully

## Updated on 2026-04-07 (duplicate save fix)

- Removed the second inverted-distance save pass from the final preview cell in `notebooks/distance_map_chasedb1.ipynb`.
- The notebook now writes `gt_dist_inverted/` only in the distance-generation cell; the final cell is read-only visualization.

## Updated on 2026-04-07 (training runtime: Apex import path)

- Attempted local environment fix to install NVIDIA Apex for upstream `from apex import amp` usage in `solver.py`.
- Findings on this machine (`Python 3.13.9`, `torch 2.11.0+cu130`):
  - PyPI `apex` package is a Pyramid package and causes import failure.
  - Current NVIDIA Apex `master` installs but no longer exposes `apex.amp`.
  - Legacy NVIDIA Apex `amp` branch fails to build/install against this modern torch/python stack.
- Implemented additive compatibility fallback in `solver.py`:
  - keep Apex path when importable
  - if Apex AMP is unavailable, use a no-op AMP shim preserving existing callsites (`initialize`, `scale_loss`)
  - no baseline training-loop rewrite.
- Validation with `.venv`:
  - `train.py` now passes previous Apex import stage and starts training setup.
  - Next blocker is CHASE path/layout mismatch at runtime (`data/chase/images/...` unreadable in current repo layout), not AMP import.

## Updated on 2026-04-07 (Python 3.13 dataloader overflow fix)

- Resolved CHASE dataloader crash during training augmentation:
  - Error: `OverflowError: Python integer -16 out of bounds for uint8`
  - Location: `data_loader/GetDataset_CHASE.py`, `randomHueSaturationValue`
- Root cause:
  - Python 3.13 raises on direct `np.uint8(negative_int)` conversion in this path.
- Implemented minimal compatibility fix:
  - Replaced `np.uint8(hue_shift)` with `np.uint8(hue_shift % 256)`.
  - This preserves prior uint8 wraparound semantics while avoiding the runtime exception.
- Scope:
  - Upstream-touching, minimal, localized bug fix in existing augmentation utility.

## Updated on 2026-04-07 (output_dir and label_mode-coupled save path)

- Added `--output_dir` to `train.py`.
- When `--output_dir` is set, training outputs are now saved under:
  - `<output_dir>/<label_mode>/`
- Updated `solver.py` checkpoint save path to follow `args.save` instead of fixed `models/` at repository root:
  - checkpoints now go to `<save>/models/<exp_id>/`
- Scope:
  - Upstream-touching, minimal path-handling change.
  - No dataset loading or model/loss behavior changes.

## Updated on 2026-04-07 (CSV logging: train_loss and val_loss)

- Updated `solver.py` CSV logging format to include per-epoch losses:
  - Header changed to: `epoch,train_loss,val_loss,dice,Jac,clDice`
  - `train_loss`: epoch mean of training loss (`loss_main + 0.3 * loss_aux`)
  - `val_loss`: epoch mean of validation loss computed with the same loss formula in `test_epoch`
- Existing metric columns (`dice`, `Jac`, `clDice`) and best-model selection logic were kept unchanged.
- Updated `scripts/aggregate_kfold_results.py` parser for compatibility:
  - supports both legacy epoch rows (`epoch,dice,Jac,clDice`) and extended rows (`epoch,train_loss,val_loss,dice,Jac,clDice`)
  - avoids misreading `train_loss` as `dice` when aggregating folds.
- Scope:
  - Upstream-touching, minimal logging change in existing training/evaluation flow.
  - No hyperparameter, dataloader, or metric-definition changes.

## Updated on 2026-04-07 (batch-size 1 directional-affinity shape fix)

- While validating new `val_loss` logging, CHASE validation crashed at `connect_loss` with:
  - BCE target/input size mismatch when validation batch size is 1.
- Root cause:
  - `connect_loss.connectivity_matrix()` used unconditional `squeeze()`, which dropped the batch axis for batch-size 1 in directional affinity targets.
- Implemented minimal fix:
  - keep existing behavior, but re-add batch axis when `conn.dim() == 3` via `unsqueeze(0)`.
- Validation:
  - `.venv` smoke test confirms expected output shapes:
    - batch1 -> `(1, 8, H, W)`
    - batch2 -> `(2, 8, H, W)`

## Updated on 2026-04-07 (checkpoint-time batch image export)

- Initial implementation added checkpoint-time batch image export.
- This behavior was later superseded on the same date by per-batch `image/pred/mask` triplet export (see next section).
- Scope:
  - Upstream-touching, additive logging/visualization behavior only.
  - No training loss, optimizer, dataloader, or metric logic changes.

## Updated on 2026-04-07 (checkpoint-time image/pred/mask batch export)

- Refined checkpoint visualization in `solver.py` from single image-grid export to per-batch triplet export.
- When periodic checkpoint condition is met (`(epoch + 1) % save_per_epochs == 0`), `test_epoch` now saves each validation batch as separate files:
  - `<save>/models/<exp_id>/checkpoint_batches/epoch_XXX/image/batch_YYYY.png`
  - `<save>/models/<exp_id>/checkpoint_batches/epoch_XXX/pred/batch_YYYY.png`
  - `<save>/models/<exp_id>/checkpoint_batches/epoch_XXX/mask/batch_YYYY.png`
- Scope:
  - Upstream-touching, additive artifact-saving change only.
  - No model update rule, loss, metric computation, or dataloader behavior changes.

## Updated on 2026-04-08 (distance_map notebook: pre-normalization mean)

- Updated `notebooks/distance_map_chasedb1.ipynb` distance-generation cell to accumulate pre-normalized distance-map statistics.
- Added final prints for:
  - overall pre-normalization mean across all samples (`signed` + `inverted`)
  - per-mode pre-normalization mean (`signed`, `inverted`)
- This change is fork-specific notebook instrumentation only; no upstream training/evaluation path changed.

## Updated on 2026-04-08 (dist_signed post-fix smoke run)

- Run:
  - `.venv/bin/python train.py --dataset chase --data_root data/chase --resize 960 960 --num-class 1 --batch-size 4 --epochs 5 --lr 0.0038 --lr-update poly --folds 1 --label_mode dist_signed --output_dir output/dist_signed_smoke_after_fix`
- Purpose:
  - verify dist-only loader scaling fix and dist-only clDice non-finite guard.
- Observed in `output/dist_signed_smoke_after_fix/dist_signed/results_1.csv`:
  - epoch1/2: non-zero metrics (`dice` up to `0.157044`)
  - epoch3~5: `dice/jac` collapse to `0.0`
  - `clDice` remained finite (`0.0` after collapse), no CSV `nan`.
- Note:
  - `metrics/cldice.py` still emits runtime divide warning internally during empty-skeleton cases, but solver-side guard now prevents non-finite values from being recorded in results CSV for `dist_*`.

## Updated on 2026-04-09 (30-epoch dist_signed smoke and metric-path verification)

- Environment gate:
  - installed `pytest` into `.venv`
  - `.venv/bin/python -m pytest -q tests/test_chase_dataloader_gt_dist_signed.py` passed
- Smoke run A:
  - command:
    - `.venv/bin/python train.py --dataset chase --data_root data/chase --resize 960 960 --num-class 1 --batch-size 4 --epochs 30 --lr 0.0038 --lr-update poly --folds 1 --label_mode dist_signed --save-per-epochs 15 --output_dir output/chase_dist_signed_smoke_20260409`
  - runtime result:
    - completed 30 epochs
    - generated `15_model.pth`, `30_model.pth`, checkpoint batch triplets, `best_model_meta.txt`
  - observed metric issue:
    - `results_1.csv` stayed constant at `dice=0.155274`, `jac=0.084187`, `clDice=0.500000`
    - per-sample metrics were also epoch-invariant
- Metric-path diagnosis:
  - direct checkpoint inspection showed:
    - `best_model.pth`: solver-path prediction ratio `1.0`
    - `30_model.pth`: solver-path prediction ratio `1.0`
    - `30_model.pth` simultaneously had `logit>0` ratio `0.0`
  - conclusion:
    - dist validation path `pred_score > 0` in `solver.py` was invalid for sigmoid-voted score maps and produced all-one masks.
- Smoke run B after eval fix:
  - command:
    - `.venv/bin/python train.py --dataset chase --data_root data/chase --resize 960 960 --num-class 1 --batch-size 4 --epochs 30 --lr 0.0038 --lr-update poly --folds 1 --label_mode dist_signed --save-per-epochs 15 --output_dir output/chase_dist_signed_smoke_metricfix_20260409`
  - observed in `results_1.csv`:
    - all 30 epochs recorded `dice=0.000000`, `jac=0.000000`
    - `clDice` became `nan` under empty-pred cases
  - debug artifact result:
    - `models/1/debug_report.md` marked `15_model.pth`, `30_model.pth`, and `best_model.pth` as `collapsed=1`
- Practical outcome:
  - previous non-zero constant metric was a false signal from the evaluation threshold
  - after removing that false signal, current CHASE `dist_signed` smoke run still shows real output collapse
  - full 5-fold training should remain blocked until the dist training path itself is improved

## Updated on 2026-04-13 (`test_getdataset_isic2018.ipynb` smoke parity check)

- Validation target:
  - verify notebook test logic for `data_loader/GetDataset_ISIC2018.py` runs successfully on real ISIC files via temporary `.npy` fixture/list artifacts
- Executed check:
  - `.venv/bin/python` smoke script mirroring notebook assertions (non-interactive) passed
- Verified items:
  - temporary fixture/list generation from `data/ISIC2018/train` succeeded
  - `ISIC2018_dataset` lengths matched split lists for `train/validation/test`
  - returned tensor shapes/types and label binarization constraints were valid
  - validation transform output matched expected deterministic preprocessing
  - `with_name=True` output contract matched list entry naming
  - `DataLoader` batch tensor dimensions were valid
  - `connectivity_matrix` toy-mask output shape was `(8, 3, 3)`
  - temporary artifacts were removed cleanly
