# Modification Status (as of 2026-04-08)

## Updated on 2026-04-13 (sample visualization RGB/BGR handling hardening)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (result aggregation utility only)
- Implemented in `scripts/aggregate_kfold_results.py`:
  - sample-visualization panel image loading now uses `PIL` with explicit mode conversion
  - `Image` column is forced to `RGB`
  - `GT` and model prediction columns are forced to single-channel grayscale (`L`)
  - this prevents accidental BGR-vs-RGB channel-order confusion in saved comparison PNGs

## Updated on 2026-04-13 (aggregate_kfold_results sample visualization export)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (result aggregation utility only)
- Implemented in `scripts/aggregate_kfold_results.py`:
  - added `--sample-vis-count` (default `2`) to export qualitative sample-comparison artifacts during k-fold aggregation
  - selects the reference model as:
    - multi-root mode: experiment with highest mean Dice, then its best fold
    - single-root mode: best fold in the root
  - selects `top-N` and `bottom-N` validation samples from the reference model's `best_epoch` using `models/<fold>/test_sample_metrics.csv`
  - saves a comparison PNG with row layout:
    - `Sample 1..K`
    - columns: `Image / GT / Model 1 / Model 2`
  - saves a companion CSV with sample names, Dice values, fold/epoch metadata, and resolved image paths
  - comparison model policy:
    - multi-root mode: second-best experiment on the same fold when available
    - single-root mode: second-best fold in the same experiment
  - if `best_epoch` has no saved `checkpoint_batches/epoch_*` PNGs, the script falls back to the nearest saved checkpoint epoch and prints a warning
- Output artifacts:
  - `<output-dir>/<output-stem>_sample_visualization.png`
  - `<output-dir>/<output-stem>_sample_visualization.csv`

## Updated on 2026-04-13 (`chasedb1_train.sh` epoch policy restored)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (script entrypoint only)
- Implemented in `scripts/chasedb1_train.sh`:
  - restored automatic epoch defaults when `--epochs` is not provided:
    - `binary + conn_num=8` -> `130`
    - `binary + conn_num=25` -> `390`
    - `dist_signed` / `dist_inverted` -> `260`
  - `--epochs` remains explicit override.
- Validation:
  - `sh -n scripts/chasedb1_train.sh` passed
  - `sh scripts/chasedb1_train.sh --help` shows restored policy text

## Updated on 2026-04-13 (`chasedb1_train.sh` condition-branch simplification)

- Note:
  - the temporary fixed-130 epoch behavior in this entry was later superseded by the same-day restore in:
    - `Updated on 2026-04-13 (chasedb1_train.sh epoch policy restored)`

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (script entrypoint only)
- Implemented in `scripts/chasedb1_train.sh`:
  - removed multi-branch python interpreter fallback (`.venv/python`, `python`, `python3`) and now requires `.venv/bin/python` only
  - removed label-mode validation branch
  - temporarily simplified epoch behavior to fixed default `--epochs 130` (later restored)
  - current repo state: CHASE wrapper scripts were removed; this launcher is the only CHASE entrypoint
  - kept key launcher args:
    - `--conn_num`
    - `--label_mode`
    - `--dist_aux_loss`
    - `--dist_sf_l1_gamma`
    - `--epochs`
- Validation:
  - `sh -n scripts/chasedb1_train.sh` passed
  - `sh scripts/chasedb1_train.sh --help` passed

## Updated on 2026-04-13 (CHASE training scripts unified into one launcher)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (script entrypoints only)
- Implemented in `scripts/chasedb1_train.sh`:
  - merged duplicated logic from:
    - `scripts/chasedb1_train.sh` (binary 8-neighbor)
    - `scripts/chasedb1_train_5x5.sh` (binary 25-neighbor)
    - `scripts/chasedb1_train_dist_signed.sh`
    - `scripts/chasedb1_train_dist_inverted.sh`
  - unified configurable arguments:
    - `--conn_num`
    - `--label_mode`
    - `--dist_aux_loss`
    - `--dist_sf_l1_gamma`
  - kept backward-compatible default epoch policy:
    - binary + `conn_num=8` -> `130`
    - binary + `conn_num=25` -> `390`
    - `dist_signed` / `dist_inverted` -> `260`
  - kept `.venv/bin/python -> python -> python3` interpreter fallback.
- Compatibility wrappers:
  - `scripts/chasedb1_train_5x5.sh` now delegates to unified launcher with `--conn_num 25 --epochs 390`
  - `scripts/chasedb1_train_dist_signed.sh` now delegates with `--label_mode dist_signed --epochs 260`
  - `scripts/chasedb1_train_dist_inverted.sh` now delegates with `--label_mode dist_inverted --epochs 260`
- Validation:
  - `sh -n scripts/chasedb1_train.sh scripts/chasedb1_train_5x5.sh scripts/chasedb1_train_dist_signed.sh scripts/chasedb1_train_dist_inverted.sh` passed
  - `sh scripts/chasedb1_train.sh --help` passed
  - wrapper help passthrough (`*_5x5.sh`, `*_dist_signed.sh`, `*_dist_inverted.sh`) passed

## Updated on 2026-04-13 (experiment mean table: split experiment/loss/conn and enforce signed-before-inverted order)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (result aggregation utility only)
- Implemented in `scripts/aggregate_kfold_results.py`:
  - cross-experiment mean bundle now parses experiment metadata from folder names:
    - `experiment`: base experiment only (e.g., `dist_signed`, `dist_inverted`, `binary`)
    - `conn_num`: extracted numeric connectivity (e.g., `8`, `25`) as a dedicated column
    - `loss`: extracted as dedicated column (`BCE`, `Smooth-L1`, `GJML+SF-L1`, and support for `GJML+SJ-L1` tag parsing)
  - loss suffix tokens and conn number are removed from `experiment` column values
  - row ordering switched to custom experiment order:
    - `binary` -> `dist_signed` -> `dist_inverted`
    - then by `conn_num`, then `loss`
- Validation:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_mean_merge_test` passed
  - verified output schema/order in:
    - `output/summary_mean_merge_test/kfold_summary_experiment_means.csv`

## Updated on 2026-04-13 (experiment order switched to natural sort in mean bundle)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (result aggregation utility only)
- Implemented in `scripts/aggregate_kfold_results.py`:
  - cross-experiment mean bundle rows are now sorted by natural order (`natsorted`-style) on experiment name
  - example order change: `binary_8` now appears before `binary_25`
- Validation:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - regenerated summary via `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_mean_merge_test`
  - verified row order in `output/summary_mean_merge_test/kfold_summary_experiment_means.csv`

## Updated on 2026-04-13 (experiment-mean ranking highlight in LaTeX/PDF)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (result aggregation utility only)
- Implemented in `scripts/aggregate_kfold_results.py`:
  - for cross-experiment mean LaTeX table (`*_experiment_means.tex/.pdf`), metric columns now apply rank-based emphasis:
    - best (top-1) value per metric column (`Dice/Jac/clDice`): `\\textbf{...}`
    - second-best (top-2) value per metric column: `\\underline{...}`
  - ranking is computed independently for each metric column across experiment rows
  - CSV output remains numeric-only (no markup)
- Validation:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_mean_merge_test` passed
  - verified markup in `output/summary_mean_merge_test/kfold_summary_experiment_means.tex`:
    - includes both `\\textbf{...}` and `\\underline{...}` entries

## Updated on 2026-04-13 (experiment-mean table: drop best-epoch column, add loss label)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (result aggregation utility only)
- Implemented in `scripts/aggregate_kfold_results.py`:
  - cross-experiment mean bundle (`*_experiment_means.csv/.tex/.pdf`) no longer includes `best_epoch_mean`
  - added `loss` column in cross-experiment mean bundle
  - loss-label rule:
    - experiment name containing `_gjml_sf_l1` -> `GJML+SF-L1`
    - otherwise, containing `dist` -> `Smooth-L1`
    - containing `binary` -> `BCE`
- Validation:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_mean_merge_test` passed
  - verified `output/summary_mean_merge_test/kfold_summary_experiment_means.csv` header:
    - `experiment,loss,num_folds,best_dice_mean,best_jac_mean,best_cldice_mean`

## Updated on 2026-04-12 (selectable `GJML + SF-L1` added for distance auxiliary regression)

- Affected path:
  - fork-specific distance auxiliary loss path in `connect_loss.py`
  - CLI surface in `train.py`
- Change:
  - added `--dist_aux_loss` with choices `smooth_l1` and `gjml_sf_l1`; default remains `smooth_l1`
  - added `--dist_sf_l1_gamma` with default `1.0`
  - distance-label auxiliary regression now routes through a selector:
    - `smooth_l1`: existing `SmoothL1` behavior
    - `gjml_sf_l1`: paper-inspired `GJML + SF-L1` on the current `[0,1]` distance-affinity targets
  - selector is applied to `affinity_l` and to non-CHASE `bicon_l`
  - CHASE dist policy is unchanged: `bicon_l` stays zero and excluded from total loss
- Rationale:
  - make the newer regression objective available without rewriting the current distance pipeline into the full SAUNA / `[-1,1]` formulation from Dang et al. (2024)
- Validation:
  - targeted unit test added for loss selection and CHASE/non-CHASE `bicon_l` behavior

## Updated on 2026-04-12 (aggregate_kfold_results: cross-experiment mean bundle outputs)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (utility aggregation script only)
- Implemented in `scripts/aggregate_kfold_results.py`:
  - while aggregating each detected experiment root, collect the per-experiment `mean` row into an in-memory list
  - added `write_experiment_mean_csv(...)` and `write_experiment_mean_latex(...)`
  - in multi-root mode (`len(target_roots) > 1`), emit one combined bundle:
    - `<output-dir>/<output-stem>_experiment_means.csv`
    - `<output-dir>/<output-stem>_experiment_means.tex`
    - `<output-dir>/<output-stem>_experiment_means.pdf`
  - combined CSV schema:
    - `experiment,num_folds,best_epoch_mean,best_dice_mean,best_jac_mean,best_cldice_mean`
  - existing per-experiment outputs (`<experiment>_<output-stem>.*`) are preserved for backward compatibility
- Validation:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary_mean_merge_test` passed
  - confirmed combined artifacts:
    - `output/summary_mean_merge_test/kfold_summary_experiment_means.csv`
    - `output/summary_mean_merge_test/kfold_summary_experiment_means.tex`
    - `output/summary_mean_merge_test/kfold_summary_experiment_means.pdf`

## Updated on 2026-04-12 (solver voting path now uses dynamic `conn_num` instead of hardcoded `8` in eval)

- Affected path:
  - upstream-touching evaluation path in `solver.py`
- Change:
  - added `Solver.apply_connectivity_voting()` to centralize 8-neighborhood vs kxk voting dispatch
  - `test_epoch()` dist evaluation now reshapes prediction tensors with `self.args.conn_num` instead of hardcoded `8`
  - `test_epoch()` multi-class evaluation now also uses the same dynamic `conn_num` voting path
  - `connectivity_to_mask()` now reuses the shared voting helper instead of duplicating the dispatch logic
- Rationale:
  - remove brittle hardcoding from evaluation so solver behavior stays aligned with configured connectivity size
  - keep 8-neighbor behavior unchanged while making the solver safe for existing `conn_num=25` support and future extensions
- Validation:
  - `.venv/bin/python -m py_compile solver.py` passed

## Updated on 2026-04-12 (CHASE dist path excludes bicon auxiliary loss, aligned with binary CHASE policy)

- Affected path:
  - upstream-touching single-class CHASE loss path in `connect_loss.py`
- Change:
  - in `connect_loss.single_class_forward()` distance-label branch (`dist_signed`, `dist_inverted`), when `args.dataset == 'chase'`, `bicon_l` is now set to zero and excluded from total loss
  - non-CHASE datasets keep previous behavior (`bicon_l = SmoothL1(bicon_map, affinity_target)` with `0.2` weight)
- Rationale:
  - align CHASE handling between `binary` and `dist_*` branches to avoid branch-specific auxiliary-term asymmetry
- Validation:
  - `.venv/bin/python -m py_compile connect_loss.py` passed

## Updated on 2026-04-10 (dist_signed loss/output path aligned to binary final mask supervision)

- Affected path:
  - upstream-touching single-class CHASE loss/eval path in `connect_loss.py` and `solver.py`
  - binary baseline path kept intact
- Problem comparison:
  - `binary` path already supervised the final voted output as a binary vessel mask via BCE/Dice and evaluated with a binary thresholding path
  - previous `dist_signed` path supervised the final voted output against the raw distance map with SmoothL1 and evaluated via `pred_score > 0`
  - this created two concrete failure modes:
    1. trivial zero collapse in training
       - with `dist_signed`, a near-zero affinity prediction produced very small total loss (`~0.0044` on a real CHASE sample)
       - the analogous `binary` near-zero case remained heavily penalized (`~4.996`)
    2. all-foreground collapse in evaluation
       - `pred_score` from sigmoid affinity voting is non-negative almost everywhere
       - therefore `pred_score > 0` produced nearly all-one masks
- Implemented fix:
  - `connect_loss.py`
    - added `dist_target_to_mask()` to derive the binary vessel mask from a distance label
    - added `binary_edge_target_from_affinity()` and `dist_edge_loss()` for the dist path
    - changed `dist_signed` / `dist_inverted` single-class loss from:
      - voted-output regression to raw distance map (`SmoothL1(pred, target)`)
      - continuous edge target `4n(1-n)` with `soft_edge_loss()`
    - to:
      - voted-output BCE to binary vessel mask
      - Dice supervision on the same binary vessel mask
      - binary edge supervision built from the derived vessel mask
      - continuous distance affinity regression retained as auxiliary supervision (`affinity_l`, `bicon_l`)
  - `solver.py`
    - kept GT conversion for dist labels as `(y > 0)`
    - added `dist_score_to_binary()` and changed dist evaluation to threshold predicted score maps at `> 0.5` instead of `> 0`
- Rationale:
  - the final model output is a segmentation mask, so distance mode now matches binary mode at the final-output supervision level
  - distance labels are still used, but only where they are informative: directional affinity regression
- Validation:
  - `.venv/bin/python -m py_compile connect_loss.py solver.py` passed
  - real-CHASE zero/mid/high-logit comparison after the patch:
    - `dist_signed` zero-logit total loss increased from the previous collapse regime to `~4.066`
    - no longer lower than the mid-logit case by orders of magnitude
  - evaluation-threshold sanity:
    - with constant logits `<= 0`, `dist_score_to_binary(pred_score)` now yields foreground ratio `0.0` instead of all-ones
  - smoke runs:
    - 1 epoch: `output/dist_signed_lossfix_smoke/dist_signed_8/results_1.csv`
      - run completed without crashes
      - metrics no longer come from the old `pred_score > 0` all-foreground bug path
    - 3 epochs: `output/dist_signed_lossfix_smoke3/dist_signed_8/results_1.csv`
      - Dice improved across epochs: `0.159463 -> 0.284554 -> 0.444370`
      - best-model sample foreground ratios moved off the previous all-one collapse:
        - examples around `0.0567 ~ 0.0666` vs GT `0.0766 ~ 0.0897`
    - 30 epochs: `output/dist_signed_lossfix_smoke30/dist_signed_8/results_1.csv`
      - Dice improved to `0.788337`
      - best-model metadata: `best_epoch=30`, `best_val_loss=1.932942`, `best_clDice=0.801648`
      - saved epoch-30 prediction batches stayed aligned with GT vessel area:
        - examples around `0.0821 ~ 0.0859` vs GT `0.0766 ~ 0.0897`
- Current decision:
  - keep binary loss/eval path unchanged
  - keep distance affinity as auxiliary supervision
  - supervise the final dist output as a binary mask, because that is the quantity used for the final segmentation result

## Updated on 2026-04-10 (dist_signed y-path diagnosis notebook aligned and conclusion recorded, pre-fix diagnosis)

- Affected path:
  - fork-specific analysis/documentation only
  - updated `notebooks/distance_map_chasedb1.ipynb`
- Notebook updates:
  - corrected stale distance-mode checks to match current `connect_loss.py`
    - distance edge supervision now uses `soft_edge_loss()` in the notebook examples
    - helper translation matrices now match `solver.py` / `connect_loss.py` one-step shift matrices
  - added a new `dist_signed` diagnostic section for the real CHASE loader path:
    - `y` statistics / histogram / heatmap
    - `affinity_target`, `sum_conn`, `norm_conn`, `edge`, `pred_min` diagnostics
    - ideal-target structural mismatch table
    - random-prediction loss-scale comparison table
  - lowered prior `solver.py` notebook wording so it is clearly a mechanics/runtime sanity check, not a convergence claim
- Recorded conclusion:
  - keep `edge_loss()` for binary path only
  - do not reuse `edge_loss()` for distance path; add a distance-specific edge loss or redesign the distance edge target/aggregation
- Representative ranges observed in the notebook:
  - `y_mean` roughly `0.014 ~ 0.019`
  - `edge_mean` roughly `0.012 ~ 0.019`
  - `edge_nonzero_ratio` roughly `0.08`
  - random-prediction distance loss terms: `edge ~= 0.005 ~ 0.007`, `vote ~= 0.124`, `affinity ~= 0.145`
  - ideal-target `soft_edge_loss(min(bicon_map), edge)` floor roughly `0.002 ~ 0.004`
- Interpretation:
  - current distance edge target is sparse/weak on CHASE and is not fully compatible with `min(bicon_map)` even under ideal affinity input, so the edge term is both small and structurally limited in what it can optimize
  - this diagnosis explains why the old raw-distance final-output path failed
  - later 2026-04-10 fix changed the final dist supervision target to the binary vessel mask, and this diagnosis is now treated as historical context

## Updated on 2026-04-09 (solver CSV module shadow fix for CHASE 5x5 run)

- Affected path:
  - upstream-touching validation/logging path in `solver.py`
  - reproduced during `uv run bash scripts/chasedb1_train_5x5.sh`
- Runtime failure:
  - `AttributeError: 'str' object has no attribute 'DictWriter'`
  - cause: local variables named `csv` in `Solver.create_exp_directory()`, `Solver.train()`, and `Solver.test_epoch()` shadowed the imported `csv` module
- Implemented fix:
  - renamed those local result-file variables from `csv` to `results_csv`
  - kept training, metric, and output behavior unchanged
- Validation:
  - `.venv/bin/python -m py_compile solver.py` passed
  - confirmed `csv` is no longer a local variable in `Solver.test_epoch()` code object
  - 1-epoch smoke run passed:
    - `.venv/bin/python train.py --dataset chase --data_root data/chase --resize 960 960 --num-class 1 --batch-size 4 --epochs 1 --lr 0.0038 --lr-update poly --folds 1 --conn_num 25 --output_dir output/csv_shadow_smoke`
    - generated `output/csv_shadow_smoke/binary_25/results_1.csv`
    - generated `output/csv_shadow_smoke/binary_25/models/1/test_sample_metrics.csv`

## 목적

- CHASE 기준으로 binary GT 외에 distance-map GT(`dist_signed`, `dist_inverted`) 학습 경로를 추가.
- upstream baseline 경로는 유지하고 fork 확장은 additive하게 적용.

## 현재 상태 요약

### 데이터/입력

- `data_loader/GetDataset_CHASE.py`
  - `label_mode` 지원: `binary`, `dist_signed`, `dist_inverted`
  - `binary`는 기존 이진화, distance mode는 `.npy` 연속값 로딩
  - Python 3.13 hue shift overflow 수정 반영

### 학습 인자/출력

- `train.py`
  - `--label_mode`, `--tau`, `--sigma`, `--output_dir` 추가
  - 저장 경로: `<output_dir>/<label_mode>_<conn_num>/`
  - CHASE dataloader/solver에 `label_mode` 전달

### 손실 경로

- `connect_loss.py`
  - `label_mode`, `tau`, `sigma`, `dist_aux_loss`, `dist_sf_l1_gamma` 기반 distance 분기 유지
  - single-class
    - `binary`: 기존 경로(`connectivity_matrix`, BCE/Dice 중심)
    - `dist_*`: 최종 출력은 binary mask 기준 BCE+Dice로 supervision, directional affinity에서 `affinity_l`은 `SmoothL1` 또는 선택적 `GJML + SF-L1` 보조 supervision이며 `bicon_l`은 CHASE에서는 제외(그 외 데이터셋은 동일 selector 적용)
    - `dist_*` edge supervision은 `dist_edge_loss()`를 통해 binary mask 기반 edge target 사용
  - batch size 1에서 `connectivity_matrix` 배치축 보정 반영
  - 변수명 통일 반영: `affinity_map`, `affinity_target`, `affinity_l`
  - 주석 용어 통일 반영: `directional affinity`, `distance score map`

### 학습/로그/스크립트

- `solver.py`: AMP fallback, CSV 확장(`train/val loss` 포함), validation 이미지 저장 반영
- 실행 스크립트:
  - `scripts/chasedb1_train.sh`
  - `scripts/chasedb1_train_dist_signed.sh`
  - `scripts/chasedb1_train_dist_inverted.sh`
- `scripts/aggregate_kfold_results.py`: 구/신 CSV 포맷 모두 파싱 가능

## 남은 이슈

1. multi-class distance 확장
   - 현재는 single-class 중심 구현.
2. distance 하이퍼파라미터/가중치 튜닝
   - `sigma`, dist 보조 손실 가중치(`affinity`, `bicon`)의 기본값 고정 기준 필요.
3. 문서 동기화 운영
   - 손실/평가 정책 변경 시 `PROJECT_CONTEXT` 문서와 CLI help를 같은 변경 단위로 갱신 필요.

## 다음 우선 작업

1. distance mode 추가 smoke 실행으로 현재 기본 하이퍼파라미터 안정성 재확인.
2. dist 보조 손실 가중치/`sigma` 튜닝 실험 설계 및 기준값 확정.
3. multi-class distance 확장 필요 여부 결정(필요 시 별도 fork 모듈로 추가).
## 확인된 검증

- `python3 -m py_compile connect_loss.py` 통과(문법 기준).
- CHASE signed-distance dataloader 테스트 파일 존재:
  - `tests/test_chase_dataloader_gt_dist_signed.py`
- 성능 개선/수렴 안정성은 추가 실험으로 확인 필요.

## Updated on 2026-04-08 (dist_signed script runtime + pred-empty analysis)

- `scripts/chasedb1_train_dist_signed.sh`
  - `python` 미존재 환경에서도 실행되도록 인터프리터 탐색 추가:
    - 우선순위: `.venv/bin/python` -> `python` -> `python3`
  - 스크립트 실행 위치와 무관하게 동작하도록 repo root로 `cd` 후 `train.py` 실행.
  - `DIST_AUX_LOSS`, `DIST_SF_L1_GAMMA` 환경변수와 추가 CLI 인자(`"$@"`)를 통해 distance auxiliary loss 선택 가능.
- `connect_loss.py`
  - `dist_map.ndim()` 오탈자 수정 (`ndim` 속성 사용)으로 dist 경로 런타임 크래시 제거.
- 실구동 확인:
  - `sh scripts/chasedb1_train_dist_signed.sh` 기준 학습 루프가 epoch 반복까지 정상 진입.
  - `pred` 저장 파일이 전부 0(검은 화면)으로 보이는 현상 확인.
- 원인 분석(코드 경로 기준):
  1. `solver.py` 테스트 경로(single-class)에서 affinity를 먼저 `>0.5` 이진화 후 `Bilateral_voting` 적용.
     - 이로 인해 vote 결과 `pred`가 쉽게 0으로 붕괴.
  2. 같은 경로에서 distance GT(`y_test_raw`)를 `long()`으로 변환하여 평가에 사용.
     - CHASE `dist_signed` 값이 매우 작은 연속값(예: max≈0.0039)이므로 `long()` 시 거의 전부 0.
     - 결과적으로 Dice가 과대평가(1.0 고정)될 수 있음.

## Updated on 2026-04-08 (debug script for output distribution)

- Added `scripts/debug_dist_output_distribution.py` for quick diagnosis of CHASE model-output/GT distributions.
- Script capabilities:
  - load CHASE split (`exp_id`, `train/test`) in same style as `train.py`
  - load optional checkpoint
  - print per-batch and aggregate stats for:
    - logits/sigmoid distributions
    - threshold ratios (`logit>0`, `sigmoid>0.5`)
    - vote output positive ratio from two binarization paths
    - GT distribution and positive ratios (`>0`, `>0.5`)
- Local run (1 batch, dist_signed checkpoint) confirmed:
  - `logit>0` ratio not zero (~0.374)
  - but vote outputs remain zero (`vote(sig>0.5)=0`, `vote(logit>0)=0`)
  - GT positive ratio under `>0` is non-zero (~0.0766), while `>0.5` is zero
  - indicates current zero-pred issue is downstream of voting behavior, not only threshold choice.

## Updated on 2026-04-08 (k-fold aggregation path compatibility)

- `scripts/aggregate_kfold_results.py` 입력 경로 해석 확장:
  - 기존: `<input_root>/<fold>/<input_name>`
  - 추가 지원: `<input_root>/results_<fold>.csv` (예: `output/chase_binary/results_1.csv`)
  - `input_name`에 `{fold}`가 포함된 경우도 포맷팅 후 경로 탐색 지원
- 목적:
  - 현재 CHASE 결과 저장 구조(`results_1.csv` ... `results_5.csv`)와 집계 스크립트 기본 실행 경로를 정합화.
- 검증:
  - `python3 -m py_compile scripts/aggregate_kfold_results.py` 통과
  - `python3 scripts/aggregate_kfold_results.py --output-dir /tmp/dconn_summary_test` 실행 성공(CSV/TEX/PDF 생성)

## Updated on 2026-04-08 (enhanced voting-collapse diagnostics)

- Expanded `scripts/debug_dist_output_distribution.py` with:
  - 8-direction positive ratios per batch and aggregate
  - pair-consistency proxies for opposite direction pairs: `[0-7, 1-6, 2-5, 3-4]`
  - bilateral voting pre/post stats (`min/max/mean`) for both `sig>0.5` and `logit>0` binarization inputs
- Verification on `dist_signed` checkpoint:
  - direction activations are strongly imbalanced (example: some directions near 1.0 while others exactly 0.0)
  - opposite-pair consistency is 0 across all four pairs
  - bilateral voting raw output remains all-zero, so post-threshold pred is all-zero
- Conclusion:
  - current empty-pred issue is consistent with directional-pair mismatch/collapse before voting, not with GT thresholding policy.

## Updated on 2026-04-08 (dist target scale diagnostics added)

- `scripts/debug_dist_output_distribution.py` now also reports:
  - `score_target` stats from `1 - exp(-dist/tau)`
  - `affinity_target = distance_connectivity_matrix(dist, tau, sigma)` stats and per-direction means
- Observed with `models/1/best_model.pth` + `dist_signed`:
  - voting path is alive (`vote_raw mean ~= 0.069`, non-zero)
  - but `affinity_target` scale is extremely small:
    - max around `1.66e-06`, mean around `6.17e-09`
    - `affinity_target > 1e-6` ratio around `1.69e-4`
- Implication:
  - dist target magnitude is near-zero under current `(dist/255, tau=3.0, sigma=2.0)` setting, which can weaken affinity-supervision signal significantly.

## Updated on 2026-04-08 (epoch-wise voting debug utility)

- Added `scripts/debug_epochwise_voting_analysis.py` for checkpoint-by-checkpoint diagnostics.
- Purpose:
  - iterate `*_model.pth` (and optional `best_model.pth`) in a checkpoint directory
  - compute per-checkpoint aggregates on fixed CHASE split batches:
    - vote positive ratio
    - opposite-direction pair consistency
    - direction-wise positive ratios
    - collapsed-voting flag
- Verified run example:
  - `--checkpoint_dir output/dist_signed/models/1 --label_mode dist_signed --split test`
  - sampled early checkpoints (`15/30/45/60_model.pth`) showed `collapsed_voting=1` with near-zero direction activations.

## Updated on 2026-04-08 (best-model metric metadata persistence)

- Updated `solver.py` so `best_model_meta.txt` stores full metrics of the exact epoch that triggered best-model update:
  - `best_epoch`, `best_dice`, `best_train_loss`, `best_val_loss`, `best_jac`, `best_clDice`
- Implementation detail:
  - `test_epoch()` now returns a metric dict (`dice/jac/cldice/val_loss/train_loss`), and `train()` writes those values when `best_model.pth` is updated.
- Validation:
  - 1-epoch smoke run (`output/dist_signed_meta_check/dist_signed`) confirmed `best_model_meta.txt` and `results_1.csv` metrics match on the same epoch row.

## Updated on 2026-04-08 (artifact rebuild automation script)

- Added `scripts/rebuild_dist_signed_artifacts.py` to automate artifact checks/rebuild for a run root (e.g., `output/dist_signed`):
  - regenerates `models/<exp>/epochwise_voting_debug.csv` by iterating checkpoints
  - writes/refreshes `models/<exp>/best_model_meta.txt` by combining:
    - `results_<exp>.csv` best-epoch metrics
    - debug stats from `best_model.pth`
- Verified on `output/dist_signed`:
  - generated both files successfully
  - detected mismatch condition where `best_epoch_from_results=1` but `1_model.pth` is absent (`best_epoch_checkpoint_exists=False`)

## Updated on 2026-04-08 (debug script unification + markdown report)

- Consolidated three debug utilities into one script:
  - kept: `scripts/rebuild_dist_signed_artifacts.py`
  - removed: `scripts/debug_dist_output_distribution.py`, `scripts/debug_epochwise_voting_analysis.py`
- Unified script now performs in one run:
  - epochwise checkpoint diagnostics and CSV generation
  - best-model metadata generation
  - detailed single-checkpoint distribution/voting diagnostics
  - markdown report generation: `models/<exp>/debug_report.md`
- Verified by running unified script on `output/dist_signed`:
  - created/updated `epochwise_voting_debug.csv`
  - created `debug_report.md`

## Updated on 2026-04-08 (meta ownership rule)

- Updated unified script policy:
  - `best_model_meta.txt` is no longer generated/overwritten by debug script.
  - Ownership is now solver-only (`solver.py` writes it when best model updates).
  - unified script only reads/existence-checks that meta file and reports status.

## Updated on 2026-04-08 (dist metric path alignment with distance-affinity training)

- `solver.py:test_epoch()` single-class `dist_*` validation path adjusted to align with training-side distance affinity flow:
  - before: `logit -> hard threshold -> bilateral voting -> mask`
  - now: `logit -> sigmoid (continuous affinity) -> bilateral voting -> mask threshold`
- Binary (`label_mode='binary'`) and multi-class evaluation paths are unchanged.
- Purpose:
  - reduce train/eval mismatch for distance-affinity experiments by evaluating from continuous directional affinity maps as in `connect_loss.single_class_forward()`.
- Verification:
  - `.venv/bin/python -m py_compile solver.py` passed.

## Updated on 2026-04-08 (5x5 conn_num runtime channel mismatch fix)

- `model/DconnNet.py` `SDE_module` fixed for non-divisible directional splits (e.g., `conn_num=25` with backbone `in_channels=512`):
  - `inter_channels = 512 // 25 = 20`, so concatenated directional feature has `20 * 25 = 500` channels.
  - `final_conv` input channels changed from fixed `in_channels`(512) to `self.inter_channels * self.conn_num`(500).
- Effect:
  - removes runtime error in 5x5 training path:
    - `expected input ... to have 512 channels, but got 500 channels`.
- Validation:
  - `.venv/bin/python -m py_compile model/DconnNet.py` passed.
  - focused smoke test passed: `SDE_module(512,512,num_class=25,conn_num=25)` forward on `(4,512,30,30)` with `d_prior(4,25,1,1)` returns `(4,512,30,30)`.
- `solver.py` now logs (when enabled):
  - train per-batch: total/main/aux loss, lr, target positive ratio, output/aux logit stats, sigmoid activation ratios
  - val per-batch: val loss, dice/jac/clDice, pred/gt positive ratios, output logits stats, affinity/vote-related ratios
- CSV outputs (when `--debug_save_csv` is set):
  - `<save>/models/<exp_id>/debug_train.csv`
  - `<save>/models/<exp_id>/debug_val.csv`
- Baseline behavior remains unchanged when debug flags are not set.

## Updated on 2026-04-08 (5x5 binary baseline-compat runtime fixes + validation)

- Scope classification:
  - upstream baseline path affected: YES (5x5 binary path runtime compatibility)
  - fork extension path affected: NO (no dist-specific behavior change in this patch)
- Implemented fixes:
  1. `model/DconnNet.py`
     - `SDE_module.final_conv` 입력 채널을 고정 `in_channels(512)`에서
       `self.inter_channels * self.conn_num`로 변경.
     - 배경: `conn_num=25`일 때 `512//25=20`, concat 결과 채널이 `500`이므로
       기존 512 고정 conv와 shape mismatch 발생.
  2. `connect_loss.py`
     - `single_class_forward()`에서 binary 경로 변수 정합성 수정:
       - `affinity_target`를 binary에서도 `con_target`로 연결.
     - 8-channel 하드코딩 제거:
       - `view(..., 8, ...)` -> `view(..., self.conn_num, ...)`
       - edge 계산의 `8` -> `self.conn_num`
     - `Bilateral_voting_kxk` 5x5 호환 보강:
       - `conn_num=25`(채널수) 입력 시 내부에서 `kxk_size=5`로 해석.
       - offset 채널 순서를 `connectivity_matrix_5x5` 생성 순서와 정렬.
       - 중심 채널 포함 케이스(`K=(2r+1)^2`) 지원.
     - `shift_map()`에서 translation 텐서의 3D/4D 및 배치 반복 확장 처리.
  3. `solver.py`
     - binary validation 경로 reshape를 `8` 고정에서 `self.args.conn_num` 기반으로 변경.
     - `connectivity_to_mask()`가 `conn_num=25`일 때 `Bilateral_voting_kxk`를 사용하도록 확장.
- Validation executed:
  - command:
    - `.venv/bin/python train.py --dataset chase --data_root data/chase --resize 960 960 --num-class 1 --batch-size 4 --epochs 1 --lr 0.0038 --lr-update poly --folds 1 --conn_num 25 --label_mode binary --output_dir output/chase_5x5_smoke`
  - observed result:
    - train/val loop completed without runtime crash
    - artifact files generated:
      - `output/chase_5x5_smoke/binary/results_1.csv`
      - `output/chase_5x5_smoke/binary/models/1/best_model.pth`
      - `output/chase_5x5_smoke/binary/models/1/best_model_meta.txt`
    - `results_1.csv` contains epoch metrics row (`001`) and best summary row (`000`).

## Updated on 2026-04-08 (Telegram completion alert script)

- Added `scripts/telegram_alert.py` (fork-specific utility script).
- Purpose:
  - send Telegram alerts when training/inference job or Codex session completes, following `AGENTS.md` rule.
- Env-variable support:
  - token: prefers `SERVER_ALERT_TELEGRAM_TOKEN` (also supports `TELEGRAM_BOT_TOKEN`, `BOT_TOKEN`)
  - chat id: prefers `TELEGRAM_ID` (also supports `TELEGRAM_CHAT_ID`, `CHAT_ID`)
- Usage examples:
  - dry run: `.venv/bin/python scripts/telegram_alert.py --job train_fold1 --status DONE --dry-run`
  - send custom: `.venv/bin/python scripts/telegram_alert.py --message "train fold1 finished"`
- Scope:
  - fork extension only; no upstream baseline training/eval logic changed.

## Updated on 2026-04-08 (aggregate_kfold_results: aggregate all folders under output)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (utility aggregation script only)
- Implemented in `scripts/aggregate_kfold_results.py`:
  - default `--input-root` changed from `output/binary` to `output`
  - added automatic target-root discovery:
    - if `<input-root>` itself has fold CSVs, aggregate that root only (backward-compatible behavior)
    - otherwise, aggregate every direct subfolder under `<input-root>` that has fold CSVs
  - added per-root fold resolution:
    - tries requested `--folds` first
    - if some folds are missing, auto-detects available folds from `results_<fold>.csv` and uses those with warning
  - refactored repeated aggregation math into helper function `aggregate_root(...)`
  - in multi-root mode, outputs are split by folder name:
    - `<output-dir>/<folder>_<output-stem>.csv|.tex|.pdf`
  - added LaTeX title escaping helper to avoid PDF build failures when folder names include `_`.
- Validation:
  - `.venv/bin/python -m py_compile scripts/aggregate_kfold_results.py` passed
  - `.venv/bin/python scripts/aggregate_kfold_results.py --input-root output --output-dir output/summary` passed
  - generated summaries for:
    - `output/binary_25` (auto-detected folds 1,2,3)
    - `output/binary_8` (folds 1,2,3,4,5)

## Updated on 2026-04-08 (session alert skip policy)

- Updated `scripts/telegram_alert.py` policy:
  - if `--job` contains `session`, alert is skipped by default.
  - override option added: `--allow-session-alert`.
- Purpose:
  - reflect user preference to remove session-end alerts while keeping train/inference alerts.
- Scope:
  - fork extension only; no upstream baseline path changed.

## Updated on 2026-04-08 (test_epoch per-sample metric logging)

- Scope classification:
  - upstream baseline path affected: YES (`solver.py` validation logging path)
  - fork extension path affected: NO
- Implemented in `solver.py`:
  - `test_epoch()` now logs per-sample validation metrics to:
    - `<save>/models/<exp_id>/test_sample_metrics.csv`
  - Added robust test-batch unpacking helper to support different dataset return formats:
    - `(image, label)`
    - `(image, label, name)`
    - `(name, image, label)`
    - additional metadata fields are tolerated.
  - Added sample-name normalization helper:
    - uses dataset-provided names when available
    - falls back to deterministic IDs (`batch_xxxx_idx_xx`) when unavailable.
  - Per-sample CSV columns:
    - `epoch,batch,sample_in_batch,sample_name,val_loss,dice,jac,cldice`
  - Existing epoch-level aggregate logging (`results_<exp>.csv`) remains unchanged.
- Validation:
  - `.venv/bin/python -m py_compile solver.py` passed.

## Updated on 2026-04-08 (remove debug-related runtime features)

- Scope classification:
  - upstream baseline path affected: YES (`train.py`, `solver.py` runtime/CLI surface)
  - fork extension path affected: NO
- Implemented:
  - `train.py`
    - removed debug CLI options:
      - `--debug_verbose`
      - `--debug_interval`
      - `--debug_save_csv`
  - `solver.py`
    - removed debug runtime state fields:
      - `self.debug_verbose`
      - `self.debug_interval`
      - `self.debug_save_csv`
    - removed debug helper methods:
      - `_should_debug_batch`
      - `_tensor_stats`
      - `_append_debug_row`
    - removed train/val debug print blocks and per-batch debug CSV writing.
  - kept non-debug outputs unchanged:
    - `results_<exp>.csv` epoch-level logging
    - `test_sample_metrics.csv` per-sample metric logging
- Validation:
  - `.venv/bin/python -m py_compile train.py solver.py` passed.

## Updated on 2026-04-08 (train.py CLI help-text consistency fix)

- Scope classification:
  - upstream baseline path affected: YES (`train.py` CLI help text only)
  - fork extension path affected: NO
- Implemented in `train.py` (`parse_args()`):
  - `--label_mode` help updated to current CHASE-supported values:
    - `binary`, `dist_signed`, `dist_inverted`
  - `--epochs` help default corrected:
    - `default: 50` -> `default: 45`
  - `--lr` help default corrected:
    - `default: 1e-4` -> `default: 8.5e-4`
  - `--output_dir` help updated to actual save-path behavior:
    - `output_dir/label_mode` -> `output_dir/label_mode_conn_num`
- Validation:
  - `.venv/bin/python -m py_compile train.py` passed.

## Updated on 2026-04-09 (dist smoke rerun: metric bug confirmed, eval threshold fix applied)

- Scope classification:
  - upstream baseline path affected: YES (`solver.py` validation metric path)
  - fork extension path affected: YES (`scripts/rebuild_dist_signed_artifacts.py`)
- Preflight / environment:
  - installed `pytest` into repo `.venv`
  - `.venv/bin/python -m pytest -q tests/test_chase_dataloader_gt_dist_signed.py` passed
- Initial 30-epoch smoke run:
  - command output root: `output/chase_dist_signed_smoke_20260409/dist_signed_8`
  - runtime completed without traceback
  - `results_1.csv` recorded 30 epoch rows, but `dice/jac/clDice` were effectively constant:
    - `dice=0.155274`, `jac=0.084187`, `clDice=0.500000`
- Root-cause finding:
  - current dist eval path in `solver.py` used:
    - `sigmoid(output) -> Bilateral_voting -> (pred_score > 0)`
  - because sigmoid outputs are strictly positive, later collapsed checkpoints still produced an all-positive mask after voting.
  - confirmed by direct inspection:
    - `30_model.pth`: `logit>0` ratio `0.0`, but solver-path `pred_ratio_mean` was `1.0`
  - implication:
    - the constant `0.155274` was a validation-metric artifact, not evidence of stable segmentation quality.
- Implemented fix in `solver.py`:
  - added `dist_score_to_mask()`
  - dist validation now computes metric mask with:
    - `sigmoid(output) -> Bilateral_voting -> (pred_score > 0.5)`
  - GT mask for dist modes remains `(y_test_raw > 0)`
- Re-run after eval fix:
  - command output root: `output/chase_dist_signed_smoke_metricfix_20260409/dist_signed_8`
  - 30-epoch smoke completed without traceback
  - `results_1.csv` still had 30 epoch rows, but `dice/jac` stayed at `0.0`
  - debug report showed checkpoint-level collapse:
    - `15_model.pth`: `collapsed=1`
    - `30_model.pth`: `collapsed=1`
    - `best_model.pth`: `collapsed=1`
- Current interpretation:
  - two separate issues were disentangled:
    1. validation metric bug masked predictions as all-ones
    2. after fixing that bug, the dist-trained model still collapses to unusable segmentation on this smoke run
- Full 5-fold run status:
  - intentionally not started after the fix
  - blocked until the dist training collapse is addressed
- Debug utility maintenance:
  - `scripts/rebuild_dist_signed_artifacts.py` fixed to:
    - insert repo root into `sys.path` before local imports
    - match current `distance_affinity_matrix(..., sigma=...)` signature

## Updated on 2026-04-09 (dist edge supervision switched to continuous target loss)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (`connect_loss.py` distance-label branch)
- Implemented in `connect_loss.py` (`single_class_forward`):
  - for `label_mode in ['dist_signed', 'dist_inverted']`, changed edge supervision from:
    - `edge_l = self.edge_loss(bicon_map, edge)` (BCE-style min-vote suppression)
  - to:
    - `edge_l = self.soft_edge_loss(bicon_map, edge)` (SmoothL1 against continuous edge target)
  - edge target definition remains:
    - `norm_conn = clamp(sum_conn / conn_num, 0, 1)`
    - `edge = 4 * norm_conn * (1 - norm_conn)`
- Rationale:
  - distance modes use continuous-valued targets, so edge supervision now matches target type (regression-style).
- Validation:
  - `.venv/bin/python -m py_compile connect_loss.py` passed.

## Updated on 2026-04-09 (dist edge target epoch stats logging added)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (`connect_loss.py`, `solver.py` dist-only logging path)
- Implemented:
  - `connect_loss.py`
    - added dist-edge stat helpers:
      - `set_dist_edge_stat_collection(enabled)`
      - `reset_dist_edge_stats()`
      - `get_dist_edge_stats()`
    - in `single_class_forward()` dist branch, when collection is enabled:
      - accumulate per-call `edge.mean()`
      - accumulate per-call non-zero ratio `(edge > 1e-6).mean()`
  - `solver.py`
    - for `label_mode in ['dist_signed', 'dist_inverted']`:
      - reset edge stats at epoch start
      - collect stats only from main loss forward (not aux)
      - print epoch summary with:
        - `edge_mean`
        - `edge_nonzero`
- Rationale:
  - make distance-edge target sparsity/scale observable per epoch to support loss-scale tuning.
- Validation:
  - `.venv/bin/python -m py_compile connect_loss.py solver.py` passed.

## Updated on 2026-04-09 (connect_loss detailed return + tensorboard loss logging)

- Scope classification:
  - upstream baseline path affected: YES (`connect_loss.py`, `solver.py` training/validation loss I/O)
  - fork extension path affected: YES
- Implemented:
  - `connect_loss.py`
    - `forward(..., return_details=False)` added.
    - when `return_details=True`, returns `(total_loss, loss_dict)`.
    - `multi_class_forward` / `single_class_forward` now build and return `loss_dict` with component terms.
    - existing call sites remain backward-compatible because default is still scalar-only return.
  - `solver.py`
    - training/validation now call `self.loss_func(..., return_details=True)` and unpack `(loss, dict)`.
    - per-step TensorBoard logging added under:
      - `train/loss_total`
      - `train/main/*`
      - `train/aux/*`
    - per-epoch TensorBoard logging added under:
      - `epoch/train_loss`, `epoch/val_loss`, `epoch/dice`, `epoch/jac`, `epoch/cldice`
      - `val/*` (validation loss components)
    - TensorBoard writer path:
      - `output_dir/.../tensorboard/exp_<exp_id>`
    - added safe fallback no-op `SummaryWriter` if tensorboard backend import is unavailable.
- Validation:
  - `.venv/bin/python -m py_compile connect_loss.py solver.py train.py` passed.

## Updated on 2026-04-09 (tensorboard package missing in .venv diagnosed and fixed)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: NO (environment + warning message)
- Finding:
  - `scripts/chasedb1_train_dist_signed.sh` uses `.venv/bin/python` first.
  - in that environment, `from torch.utils.tensorboard import SummaryWriter` failed with:
    - `ModuleNotFoundError: No module named 'tensorboard'`
  - this made TensorBoard logging silently no-op under existing fallback.
- Action:
  - installed `tensorboard` into repo `.venv`.
  - added explicit warning print in `solver.py` import fallback so missing backend is visible at runtime.
- Validation:
  - `.venv/bin/python -c "from torch.utils.tensorboard import SummaryWriter"` now succeeds.
  - smoke write created event file under:
    - `output/_tb_smoke/events.out.tfevents.*`
  - `.venv/bin/python -m py_compile solver.py` passed.

## Updated on 2026-04-13 (ISIC2018 dataloader notebook test added)

- Scope classification:
  - upstream baseline path affected: NO
  - fork extension path affected: YES (test artifact only)
- Implemented:
  - added `tests/test_getdataset_isic2018.ipynb`
  - notebook behavior:
    - builds a small temporary `.npy` fixture from real `data/ISIC2018/train/{images,labels}` files
    - creates a temporary split list folder under `data_loader/isic_datalist/foldertmp_<uuid>`
    - validates `ISIC2018_dataset` for `train` / `validation` / `test` paths
    - checks `with_name=True` return format
    - checks batch shape through `DataLoader`
    - checks `connectivity_matrix` output shape on a toy mask
    - removes temporary fixture/list artifacts at the end
- Notes:
  - this is additive validation coverage only; no training/inference or loader implementation logic was modified.
