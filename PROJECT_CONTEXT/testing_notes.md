# Testing Notes

## Updated on 2026-04-16 (dataset-specific table-header verification)

- Validation target:
  - confirm fold-level and experiment-mean LaTeX tables use dataset-specific columns
  - confirm direct dataset roots (`output/chase`, `output/isic`) resolve dataset names correctly
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output/chase --output-dir output/summary_metric_check/chase --sample-vis-count 0` completed (with unfinished-experiment warning)
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output/isic --output-dir output/summary_metric_check/isic --sample-vis-count 0` completed (with unfinished-experiment warning)
  - header grep over generated TeX files passed
- Verified output:
  - CHASE fold summary header:
    - `Fold & Best Epoch & Dice & IoU & clDice & B0 error & B1 error & Train Time \\`
  - CHASE experiment-mean header:
    - `Experiment & Conn & Loss & #Folds & Dice & IoU & clDice & B0 error & B1 error \\`
  - ISIC fold summary header:
    - `Fold & Best Epoch & Dice & IoU & Accuracy & Precision & Train Time \\`

## Updated on 2026-04-16 (`solver.py` ISIC precision/accuracy output update)

- Validation target:
  - verify edited `solver.py` remains syntactically valid after adding precision/accuracy calculations and CSV columns
- Executed checks:
  - `.venv/bin/python -m py_compile solver.py` passed

## Updated on 2026-04-16 (suppress top-level experiment-means CSV/PDF smoke)

- Validation target:
  - confirm `output/summary/kfold_summary_experiment_means.csv` and `.pdf` are no longer produced
  - confirm dataset-specific outputs still render
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_no_overall_check --sample-vis-count 0` passed
- Verified output:
  - `output/summary_no_overall_check/kfold_summary_experiment_means.csv` absent
  - `output/summary_no_overall_check/kfold_summary_experiment_means.pdf` absent
  - dataset bundle still produced:
    - `output/summary_no_overall_check/dump/kfold_summary_experiment_means_datasets.tex`
    - `output/summary_no_overall_check/kfold_summary_experiment_means_datasets.pdf`

## Updated on 2026-04-16 (`aggregate_kfold_results.py` DRIVE/ISIC legacy results fallback smoke)

- Validation target:
  - verify aggregation works from dataset roots that only have legacy `results_<fold>.csv`
  - keep existing CHASE aggregation behavior unchanged
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output/chase --output-dir output/summary_tmp2 --output-stem chase_test --sample-vis-count 0` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output/drive --output-dir output/summary_tmp2 --output-stem drive_test --sample-vis-count 0` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output/isic --output-dir output/summary_tmp2 --output-stem isic_test --sample-vis-count 0` passed
- Verified output:
  - CHASE: `output/summary_tmp2/chase_test_experiment_means.pdf` and related dump CSV/TEX generated
  - DRIVE: `output/summary_tmp2/drive_test.pdf` and `output/summary_tmp2/dump/drive_test.csv` generated
  - ISIC: `output/summary_tmp2/isic_test.pdf` and `output/summary_tmp2/dump/isic_test.csv` generated
- Notes:
  - all three dataset roots completed with expected missing-fold warnings (`2,3,4,5`) and auto-detected fold `1`

## Updated on 2026-04-16 (launcher all-exit Telegram alert path)

- Validation target:
  - verify CHASE/ISIC training launchers remain syntactically valid after introducing `EXIT`-trap notifier logic
- Executed checks:
  - `bash -n scripts/chasedb1_train.sh scripts/isic2018_train.sh` passed
- Notes:
  - notifier now runs on shell exit for both success and failure paths; this validation only covers shell syntax

## Updated on 2026-04-16 (`multi_train.sh` 1-fold 10-epoch full-combination smoke)

- Validation target:
  - verify every experiment combination in `scripts/multi_train.sh` completes under the current `1fold` policy
  - verify the batch launcher no longer depends on `5folds`
  - verify CHASE training still runs after restoring the held-out split as `validset`
- Executed checks:
  - `bash -n scripts/multi_train.sh scripts/chasedb1_train.sh` passed
  - `.venv/bin/python -m py_compile train.py solver.py` passed
  - `MULTI_TRAIN_EPOCHS=10 PYTHONUNBUFFERED=1 MPLBACKEND=Agg bash scripts/multi_train.sh --output_dir output_multi_smoke_1fold_10epoch --save-per-epochs 10` passed
- Verified output:
  - run log:
    - `output_multi_smoke_1fold_10epoch/run.log`
  - successful completion count:
    - `FINISH.` lines in log: `10`
    - `val_results_1.csv` files under `output_multi_smoke_1fold_10epoch/1fold/*/`: `10`
  - completed experiment roots:
    - `binary_8_bce`
    - `binary_24_bce`
    - `dist_signed_8_gjml_sf_l1`
    - `dist_signed_8_smooth_l1`
    - `dist_inverted_8_gjml_sf_l1`
    - `dist_inverted_8_smooth_l1`
    - `dist_signed_24_gjml_sf_l1`
    - `dist_signed_24_smooth_l1`
    - `dist_inverted_24_gjml_sf_l1`
    - `dist_inverted_24_smooth_l1`
  - per-combination final validation summaries (`val_results_1.csv`) include:
    - `binary_24_bce`: `dice=0.675209`, `jac=0.510123`
    - `binary_8_bce`: `dice=0.674290`, `jac=0.509033`
    - `dist_inverted_24_gjml_sf_l1`: `dice=0.715970`, `jac=0.558031`
    - `dist_inverted_24_smooth_l1`: `dice=0.690617`, `jac=0.527915`
    - `dist_inverted_8_gjml_sf_l1`: `dice=0.713433`, `jac=0.554967`
    - `dist_inverted_8_smooth_l1`: `dice=0.709243`, `jac=0.549838`
    - `dist_signed_24_gjml_sf_l1`: `dice=0.708597`, `jac=0.549128`
    - `dist_signed_24_smooth_l1`: `dice=0.684347`, `jac=0.520496`
    - `dist_signed_8_gjml_sf_l1`: `dice=0.665237`, `jac=0.498813`
    - `dist_signed_8_smooth_l1`: `dice=0.695516`, `jac=0.533546`
- Notes:
  - the earlier `5fold` sweep attempt was intentionally abandoned after the project direction changed to `1fold`
  - CHASE still emits existing `clDice` runtime warnings in some validation batches; the smoke run itself completed successfully

## Updated on 2026-04-16 (ISIC validation-vs-test split behavior check)

- Validation target:
  - verify ISIC training epochs evaluate on the validation split, not the held-out test split
  - verify the held-out ISIC test split is evaluated only after training finishes
  - verify final held-out evaluation reloads `best_model.pth`
- Executed checks:
  - `.venv/bin/python -m py_compile solver.py train.py` passed
  - `PYTHONUNBUFFERED=1 MPLBACKEND=Agg bash scripts/isic2018_train.sh --epochs 1 --save-per-epochs 1 --output_dir output_script_smoke_isic_eval_split_check` passed
- Verified runtime behavior:
  - log printed:
    - `RUN VALIDATION ON validation split.`
  - after epoch validation finished, log printed:
    - `LOAD BEST MODEL FOR FINAL TEST EVAL: .../best_model.pth`
    - `RUN FINAL TEST EVAL AFTER TRAINING.`
- Verified output:
  - validation epoch history remained in:
    - `output_script_smoke_isic_eval_split_check/1fold/binary_8_bce/results_1.csv`
  - final held-out test summary was written separately:
    - `output_script_smoke_isic_eval_split_check/1fold/binary_8_bce/test_results_1.csv`
  - split-specific sample metrics were separated:
    - `.../models/1/val_sample_metrics.csv`
    - `.../models/1/test_sample_metrics.csv`
  - observed metric separation:
    - validation best row in `results_1.csv`: `dice=0.817109`, `jac=0.714064`
    - held-out final test row in `test_results_1.csv`: `checkpoint=best_model.pth`, `dice=0.803035`, `jac=0.695233`
- Notes:
  - this confirms ISIC `folds=1` now follows the intended protocol:
    - validation split for epoch-by-epoch model selection
    - held-out test split only after training completes

## Updated on 2026-04-15 (CHASE/ISIC launcher smoke + ISIC one-time prep decoupling)

- Validation target:
  - verify `scripts/chasedb1_train.sh` still completes a minimal training smoke
  - verify `scripts/isic2018_train.sh` runs without invoking dataset conversion during training
  - verify `scripts/prepare_isic2018_npy.py` behaves as a standalone one-time prep utility
- Executed checks:
  - `bash -n scripts/chasedb1_train.sh scripts/isic2018_train.sh` passed
  - `.venv/bin/python -m py_compile scripts/prepare_isic2018_npy.py train.py data_loader/GetDataset_ISIC2018.py` passed
  - `MPLBACKEND=Agg bash scripts/chasedb1_train.sh --epochs 1 --target_fold 1 --batch-size 2 --save-per-epochs 1 --output_dir output_script_smoke_chase` passed
  - `.venv/bin/python scripts/prepare_isic2018_npy.py --migrate-from-root data/ISIC2018 --raw-root data/ISIC2018_img --output-root data/ISIC2018 --height 224 --width 320` passed
    - observed runtime: about `0.8s`
    - observed summary: `converted=0, skipped=3694`
  - `PYTHONUNBUFFERED=1 MPLBACKEND=Agg bash scripts/isic2018_train.sh --epochs 1 --save-per-epochs 1 --output_dir output_script_smoke_isic_no_prepare` passed
- Verified output:
  - CHASE smoke wrote:
    - `output_script_smoke_chase/5folds/binary_8_bce/results_1.csv`
    - `output_script_smoke_chase/5folds/binary_8_bce/models/1/best_model_meta.txt`
  - ISIC standalone prep no longer rebuilt existing flat artifacts during the validation run
  - ISIC smoke wrote:
    - `output_script_smoke_isic_no_prepare/1fold/binary_8_bce/results_1.csv`
    - `output_script_smoke_isic_no_prepare/1fold/binary_8_bce/models/1/{1_model.pth,best_model.pth,best_model_meta.txt}`
  - ISIC smoke `results_1.csv` epoch row recorded:
    - `train_loss=2.918797`
    - `val_loss=2.811608`
    - `dice=0.737385`
    - `elapsed_hms=00:00:24`
- Notes:
  - ISIC validation/test still emits existing `clDice` runtime warnings in empty-skeleton cases; the launcher change did not modify metric behavior

## Updated on 2026-04-15 (ISIC2018 raw-image migration + `.npy` preparation check)

- Validation target:
  - verify raw `data/ISIC2018` can be migrated to `data/ISIC2018_img`
  - verify resized `.npy` artifacts are created under `data/ISIC2018/{image,label}`
  - verify `folder0` lists still resolve correctly against the rebuilt flat `.npy` dataset
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/prepare_isic2018_npy.py train.py data_loader/GetDataset_ISIC2018.py` passed
  - `bash -n scripts/isic2018_train.sh` passed
  - `.venv/bin/python scripts/prepare_isic2018_npy.py --migrate-from-root data/ISIC2018 --raw-root data/ISIC2018_img --output-root data/ISIC2018 --height 224 --width 320` passed
  - `.venv/bin/python - <<'PY' ... load`ISIC2018_dataset(..., folder=0, train/validation/test)`and check counts/shapes ... PY` passed
- Verified output:
  - raw tree exists at `data/ISIC2018_img/{train,val,test}/{images,labels}`
  - rebuilt flat dataset exists at `data/ISIC2018/image` and `data/ISIC2018/label`
  - `folder0` lengths remain `2594 / 100 / 1000`
  - rebuilt `.npy` sample shapes match `(224, 320, 3)` for images and `(224, 320)` for masks

## Updated on 2026-04-15 (`solver.py`, `aggregate_kfold_results.py` elapsed-time reporting smoke)

- Validation target:
  - verify raw training result CSVs accept the new `elapsed_hms` epoch column
  - verify `aggregate_kfold_results.py` reads both new and legacy result CSV formats
  - verify fold summary / experiment mean outputs expose train-time columns without breaking existing aggregation
- Executed checks:
  - `.venv/bin/python -m py_compile solver.py scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python - <<'PY' ... Solver.create_exp_directory(...) + _write_epoch_result_row(...) ... PY` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root <tmp fixture root> --output-dir <tmp fixture out> --sample-vis-count 0` passed
    - fixture mix:
      - new format `results_<fold>.csv` with `elapsed_hms`
      - legacy format `results_<fold>.csv` without `elapsed_hms`
  - `MPLBACKEND=Agg .venv/bin/python train.py --dataset chase --data_root data/chase --resize 960 960 --num-class 1 --batch-size 2 --epochs 1 --lr 0.0038 --lr-update poly --folds 1 --conn_num 8 --label_mode binary --output_dir output_chase_time_smoke --save-per-epochs 1` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output_chase_time_smoke --output-dir output_chase_time_smoke/summary --sample-vis-count 0` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_time_check --sample-vis-count 0` passed
- Verified output:
  - raw solver smoke wrote header:
    - `epoch,train_loss,val_loss,dice,Jac,clDice,betti_error_0,betti_error_1,elapsed_hms`
  - raw solver smoke wrote epoch row ending with:
    - `00:01:23`
  - fixture aggregation outputs:
    - per-root summary CSV header includes `train_elapsed_hms`
    - experiment mean CSV header includes `train_elapsed_mean_hms`
    - new-format mean duration example rendered as `00:03:15`
    - legacy-format duration remains blank in CSV and `N/A` in TeX
  - CHASE 1-epoch smoke wrote:
    - `output_chase_time_smoke/1fold/binary_8_bce/results_1.csv`
    - epoch row with `elapsed_hms=00:00:06`
    - summary tail row `001,0.000000`
    - `models/1/best_model_meta.txt` with `best_epoch=1`
  - CHASE smoke summary wrote:
    - `output_chase_time_smoke/summary/dump/kfold_summary.csv`
    - header includes `train_elapsed_hms`
    - row includes `best_epoch=1` and `train_elapsed_hms=00:00:06`
    - companion TeX includes `Train Time` column with `00:00:06`
  - real-output aggregation on existing repository outputs completed successfully with empty train-time cells for legacy runs
- Notes:
  - attempted ISIC 1-epoch smoke first, but current flat `.npy` dataset was incomplete relative to `folder0` split lists; switched to CHASE for end-to-end runtime verification.

## Updated on 2026-04-15 (`folder0` official ISIC2018 split-list generation check)

- Validation target:
  - verify `data_loader/isic_datalist/folder0` matches official raw `data/ISIC2018/{train,val,test}` split boundaries
- Executed checks:
  - `.venv/bin/python - <<'PY' ... compare`folder0_{train,validation,test}.list`stems against raw split image stems ... PY` passed
  - `.venv/bin/python - <<'PY' ... compare raw image stems vs label stems for`train/val/test`... PY` passed
- Verified output:
  - `folder0_train.list`: `2594` entries, exact stem match with `data/ISIC2018/train/images`
  - `folder0_validation.list`: `100` entries, exact stem match with `data/ISIC2018/val/images`
  - `folder0_test.list`: `1000` entries, exact stem match with `data/ISIC2018/test/images`
  - raw image/label stem sets matched for all three splits before list generation

## Updated on 2026-04-15 (`notebooks/result.ipynb` experiment-comparison update smoke)

- Validation target:
  - verify fold-comparison removal and new experiment-comparison section execute correctly
  - verify notebook runs using scope-split summary CSV inputs
- Executed checks:
  - `MPLBACKEND=Agg .venv/bin/python - <<'PY' ... execute all code cells in notebooks/result.ipynb ... PY` passed
- Verified output:
  - notebook loaded:
    - `output/summary/kfold_summary_experiment_means_5folds.csv` (`6` rows)
  - figures saved:
    - `output/summary/analysis_figures/abs_performance_by_config.png`
    - `output/summary/analysis_figures/experiment_comparison_mean_vs_best.png`
    - `output/summary/analysis_figures/delta_conn_24_minus_8.png`
    - `output/summary/analysis_figures/delta_loss_gjml_minus_smooth.png`
  - summary printout includes experiment mean ranking and no fold-delta summary

## Updated on 2026-04-15 (`aggregate_kfold_results.py` scope-split experiment-means CSV smoke)

- Validation target:
  - verify `1fold` and `5folds` cross-experiment mean CSV files are emitted separately
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary --sample-vis-count 0` passed
- Verified output:
  - `output/summary/kfold_summary_experiment_means_1fold.csv` exists
  - `output/summary/kfold_summary_experiment_means_5folds.csv` exists
  - `output/summary/dump/` also contains both scope-split CSV files

## Updated on 2026-04-15 (`notebooks/result.ipynb` analysis notebook smoke execution)

- Validation target:
  - verify newly created notebook code cells execute end-to-end
  - verify requested setting-change visualizations are generated from summary CSV
- Executed checks:
  - `MPLBACKEND=Agg .venv/bin/python - <<'PY' ... execute all code cells in notebooks/result.ipynb ... PY` passed
- Verified output:
  - notebook load message:
    - `Loaded output/summary/kfold_summary_experiment_means.csv with 16 rows`
  - figures saved:
    - `output/summary/analysis_figures/abs_performance_by_config.png`
    - `output/summary/analysis_figures/delta_fold_5_minus_1.png`
    - `output/summary/analysis_figures/delta_conn_24_minus_8.png`
    - `output/summary/analysis_figures/delta_loss_gjml_minus_smooth.png`
  - summary printout includes:
    - best-setting rows for Dice/Jac/clDice
    - mean delta overview for fold/conn/loss changes

## Updated on 2026-04-15 (`aggregate_kfold_results.py` summary-root experiment-means CSV smoke)

- Validation target:
  - verify cross-experiment mean CSV is emitted to `output/summary/` in addition to `dump/`
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary --sample-vis-count 0` passed
  - `cmp -s output/summary/kfold_summary_experiment_means.csv output/summary/dump/kfold_summary_experiment_means.csv` returned `0`
- Verified output:
  - `output/summary/kfold_summary_experiment_means.csv` exists
  - file contents are identical to `output/summary/dump/kfold_summary_experiment_means.csv`

## Updated on 2026-04-15 (conn_num=24 5x5 normalization + dist-mode support)

- Validation target:
  - verify global 5x5 contract is now 24 directional channels with center excluded
  - verify distance label modes run with `conn_num=24`
  - verify 24-channel first 8 offsets stay aligned with canonical 8-neighbor voting order
- Executed checks:
  - `.venv/bin/python -m py_compile connect_loss.py solver.py train.py scripts/rebuild_dist_signed_artifacts.py` passed
  - `.venv/bin/python -m pytest -q tests/test_conn24_support.py` passed
  - `.venv/bin/python -m pytest -q tests/test_dist_aux_loss_selection.py` passed
  - `bash -n scripts/chasedb1_train.sh scripts/chasedb1_test_best.sh scripts/multi_train.sh` passed
  - `.venv/bin/python - <<'PY' ... Solver.apply_connectivity_voting(conn_num=24) ... PY` passed
- targeted smoke script confirmed:
  - `shift_n_directions(..., 24)` returns 24 channels
  - `distance_affinity_matrix(..., conn_num=24, ...)` shape is `(B, 24, H, W)`
  - `connect_loss(..., label_mode='binary', conn_num=24)` forward passes
  - `connect_loss(..., label_mode='dist_signed', conn_num=24)` forward passes
  - `Solver.apply_connectivity_voting(...)` uses `Bilateral_voting_kxk(..., conn_num=5)` for 24-channel maps

## Updated on 2026-04-15 (`aggregate_kfold_results.py` all-model sample visualization smoke)

- Validation target:
  - verify sample visualization PNG renders all models in one figure
  - verify sample visualization CSV includes dynamic `model1..modelN` columns
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --output-dir output/summary_all_models_vis_check --sample-vis-count 1` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --output-dir output/summary_all_models_vis_check2 --sample-vis-count 1` passed
- Verified output:
  - `output/summary_all_models_vis_check/kfold_summary_sample_visualization.png` generated with all-model comparison columns
  - CSV header includes dynamic columns through `model12_*` in current dataset
    - file: `output/summary_all_models_vis_check/dump/kfold_summary_sample_visualization.csv`

## Updated on 2026-04-15 (`aggregate_kfold_results.py` PNG routed to summary root smoke)

- Validation target:
  - verify sample visualization PNG is saved in summary root (with PDFs)
  - verify sample visualization CSV remains in `dump/`
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --output-dir output/summary_png_with_pdf_check --sample-vis-count 1` passed
- Verified output:
  - top-level includes:
    - `output/summary_png_with_pdf_check/kfold_summary_sample_visualization.png`
    - `*.pdf` reports
  - dump includes:
    - `output/summary_png_with_pdf_check/dump/kfold_summary_sample_visualization.csv`
    - report `.csv/.tex/.aux/.log` artifacts

## Updated on 2026-04-15 (`aggregate_kfold_results.py` PDF-only root + dump artifact routing smoke)

- Validation target:
  - verify summary root keeps only PDF files
  - verify TeX/CSV/log/aux/png artifacts are written or moved into `dump/`
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --output-dir output/summary_pdf_only_check --sample-vis-count 1` passed
  - added a synthetic top-level non-PDF file and reran:
    - `echo stale > output/summary_pdf_only_check/legacy_note.txt`
    - `.venv/bin/python scripts/aggregate_kfold_results.py --output-dir output/summary_pdf_only_check --sample-vis-count 0` passed
- Verified output:
  - top-level files under `output/summary_pdf_only_check` are all `.pdf`
  - `output/summary_pdf_only_check/dump/` contains:
    - `.csv`, `.tex`, `.aux`, `.log`
    - sample visualization files (`.csv`, `.png`)
    - migrated legacy file `legacy_note.txt`

## Updated on 2026-04-15 (`aggregate_kfold_results.py` single-TeX two-table scope report smoke)

- Validation target:
  - verify `1fold` and `5folds` are rendered as two tables in one TeX document
  - verify consolidated scope PDF builds successfully
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --output-dir output/summary_scope_two_tables_check --sample-vis-count 0` passed
- Verified output:
  - generated consolidated scope report:
    - `output/summary_scope_two_tables_check/kfold_summary_experiment_means_scopes.tex`
    - `output/summary_scope_two_tables_check/kfold_summary_experiment_means_scopes.pdf`
  - TeX content check:
    - contains `\subsection*{Scope: 1fold}` + one `table`
    - contains `\subsection*{Scope: 5folds}` + one `table`

## Updated on 2026-04-15 (`aggregate_kfold_results.py` 1fold/5fold split-table smoke)

- Validation target:
  - verify cross-experiment mean outputs are generated separately for `1fold` and `5folds`
  - verify existing combined mean output remains generated
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --output-dir output/summary_scope_split_check --sample-vis-count 0` passed
- Verified output:
  - combined:
    - `output/summary_scope_split_check/kfold_summary_experiment_means.csv`
    - `output/summary_scope_split_check/kfold_summary_experiment_means.tex`
    - `output/summary_scope_split_check/kfold_summary_experiment_means.pdf`
  - split by scope:
    - `output/summary_scope_split_check/kfold_summary_experiment_means_1fold.csv`
    - `output/summary_scope_split_check/kfold_summary_experiment_means_1fold.tex`
    - `output/summary_scope_split_check/kfold_summary_experiment_means_1fold.pdf`
    - `output/summary_scope_split_check/kfold_summary_experiment_means_5folds.csv`
    - `output/summary_scope_split_check/kfold_summary_experiment_means_5folds.tex`
    - `output/summary_scope_split_check/kfold_summary_experiment_means_5folds.pdf`

## Updated on 2026-04-14 (best_model-only CHASE test launcher smoke)

- Validation target:
  - verify the new best-model test launcher is syntactically valid
  - verify `train.py` exposes single-fold execution needed for fold-specific `best_model.pth` evaluation
- Executed checks:
  - `.venv/bin/python -m py_compile train.py` passed
  - `bash -n scripts/chasedb1_test_best.sh` passed
  - `.venv/bin/python train.py --help` passed and showed `--target_fold`
  - `bash scripts/chasedb1_test_best.sh --help` passed
- Notes:
  - `train.py --help` emitted an environment warning from `pyramid/pkg_resources`, but CLI output completed successfully.
  - launcher includes Telegram alert handling, but alert send itself was not end-to-end exercised in this turn.

## Updated on 2026-04-14 (inference/test Betti-0/1 metric smoke)

- Validation target:
  - verify the evaluation path compiles after adding separate `betti_error_0` / `betti_error_1` reporting
  - verify the Betti helper returns expected component/hole errors on toy masks
- Executed checks:
  - `.venv/bin/python -m py_compile solver.py metrics/cal_betti.py metrics/betti_error.py metrics/betti_compute.py` passed
  - `.venv/bin/python` toy-mask smoke passed:
    - identical solid mask -> `betti_error_0=0`, `betti_error_1=0`
    - two-component prediction vs one-component GT -> `betti_error_0=1`, `betti_error_1=0`
    - ring prediction vs solid GT -> `betti_error_0=0`, `betti_error_1=1`
  - temporary-CSV parser smoke passed:
    - `scripts.aggregate_kfold_results.parse_fold_csv(...)` kept reading `dice/Jac/clDice` from the first 6 columns of extended `results_<fold>.csv`
    - `parse_sample_metrics_csv(...)` accepted extended `test_sample_metrics.csv` rows and preserved legacy fields

## Updated on 2026-04-14 (sample visualization config-title smoke)

- Validation target:
  - verify sample-visualization labels summarize model config instead of generic `Model 1` / `Model 2`
  - verify companion CSV exports the same config metadata explicitly
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --output-dir output/summary_vis_config_label_check --sample-vis-count 1` passed
- Verified output:
  - `output/summary_vis_config_label_check/kfold_summary_sample_visualization.csv` header now includes:
    - `model1_config`, `model1_experiment`, `model1_conn_num`, `model1_loss`, `model1_folds`
    - `model2_config`, `model2_experiment`, `model2_conn_num`, `model2_loss`, `model2_folds`
  - sample row values include config summaries such as:
    - `Exp=dist_inverted | Conn=8 | Loss=gjml_sf_l1 | Folds=5`

## Updated on 2026-04-14 (binary `_bce` naming + 5-fold batch script smoke)

- Validation target:
  - verify binary runs are named with `_bce` instead of inheriting `smooth_l1`
  - verify `scripts/multi_train.sh` active commands now target 5-fold runs
- Executed checks:
  - `.venv/bin/python -m py_compile train.py scripts/aggregate_kfold_results.py` passed
  - `sh -n scripts/chasedb1_train.sh` passed
  - `bash -n scripts/multi_train.sh` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --output-dir output/summary_binary_bce_name_check --sample-vis-count 0` passed
- Verified output:
  - `output/summary_binary_bce_name_check/kfold_summary_experiment_means.csv` contains binary rows as:
    - `binary,8,bce,1fold,...`
    - `binary,8,bce,5folds,...`
  - active lines in `scripts/multi_train.sh` now all use `--folds 5`

## Updated on 2026-04-14 (`aggregate_kfold_results.py` default all-scope aggregation smoke)

- Validation target:
  - verify default execution aggregates both `output/1fold` and `output/5folds`
  - verify folder-name parsing splits `binary_8_smooth_l1` into `binary / 8 / smooth_l1`
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --output-dir output/summary_default_all_scope_check --sample-vis-count 0` passed
- Verified output:
  - run prints multi-scope discovery info and writes both:
    - `output/summary_default_all_scope_check/1fold_binary_8_smooth_l1_kfold_summary.csv`
    - `output/summary_default_all_scope_check/5folds_binary_8_smooth_l1_kfold_summary.csv`
  - `output/summary_default_all_scope_check/kfold_summary_experiment_means.csv` includes both rows:
    - `binary,8,smooth_l1,1fold,...`
    - `binary,8,smooth_l1,5folds,...`

## Updated on 2026-04-14 (fold-aware output script/help smoke + legacy 5-fold directory move)

- Validation target:
  - verify launcher help and 5-fold batch script align with the `output/{1fold,5folds}` layout
  - verify CHASE 5-fold result directories are placed under `output/5folds/`
- Executed checks:
  - `sh -n scripts/chasedb1_train.sh` passed
  - `bash -n scripts/multi_train.sh` passed
  - `sh scripts/chasedb1_train.sh --help` passed
  - directory checks confirmed:
    - `output/5folds/binary_8_smooth_l1`
    - `output/5folds/binary_25_smooth_l1`
    - `output/5folds/dist_signed_8_smooth_l1`
    - `output/5folds/dist_inverted_8_smooth_l1`
    - `output/5folds/dist_signed_8_gjml_sf_l1`
    - `output/5folds/dist_inverted_8_gjml_sf_l1`
- Notes:
  - `output/summary*` directories were left in place because they are aggregation artifacts, not training run roots.

## Updated on 2026-04-14 (`aggregate_kfold_results.py` default 5-fold root smoke)

- Validation target:
  - verify aggregation defaults now target `output/5folds`
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --output-dir output/summary_5fold_default_root_check` passed
- Verified output:
  - `output/summary_5fold_default_root_check/kfold_summary_experiment_means.csv`
  - default `--input-root` resolved current 5-fold experiment directories under `output/5folds/`

## Updated on 2026-04-13 (sample visualization RGB/BGR handling smoke)

- Validation target:
  - verify explicit RGB/grayscale rendering path after panel-loader change
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_sample_vis_test_rgbbgr --sample-vis-count 2` passed
- Verified output:
  - `output/summary_sample_vis_test_rgbbgr/kfold_summary_sample_visualization.png`
  - `output/summary_sample_vis_test_rgbbgr/kfold_summary_sample_visualization.csv`

## Updated on 2026-04-13 (sample visualization blue-tint fix smoke)

- Validation target:
  - verify blue-tinted image-panel correction via automatic BGR->RGB swap heuristic
- Executed checks:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_sample_vis_test_rgbfix --sample-vis-count 2` passed
- Verified output:
  - `output/summary_sample_vis_test_rgbfix/kfold_summary_sample_visualization.png`
  - image-panel color changed from blue-dominant to expected orange/red fundus tone

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
  - `<output_dir>/<fold_scope>/<label_mode>_<conn_num>_<dist_aux_loss>/`
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
