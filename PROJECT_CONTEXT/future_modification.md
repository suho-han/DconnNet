# 앞으로 고쳐야 할 것

Last updated: 2026-04-29

## 원칙

- 기존 upstream baseline 동작에 영향이 갈 수 있는 변경은 바로 코드에 반영하지 않는다.
- 우선 구조 분리/책임 분리처럼 동작 보존이 가능한 작업부터 진행한다.
- dataset split, k-fold 정책, multi-class segmentation 정책처럼 실험 결과나 재현성에 영향을 줄 수 있는 항목은 `ambiguous_modification_candidates`에 남긴다.
- 수정은 가능한 한 additive 하게 진행하고, 기존 public import/API/output CSV schema는 유지한다.

## 2차 수정 대상

solver.py
`create_exp_directory`, `_format_elapsed_hms`, `_write_epoch_result_row`, `_write_eval_summary`, `_compute_binary_precision_accuracy`, `_compute_multiclass_precision_accuracy`, `save_checkpoint_batch_triplet`, `_normalize_sample_names`, `_unpack_test_batch`, `dist_to_binary`, `dist_score_to_binary`, `apply_connectivity_voting`, `connectivity_to_mask`, `_unpack_model_outputs`, `_compute_fusion_profile_loss`, `_resolve_training_runtime_configs

## 2026-04-29 1차 수정 진행 상태

- 완료:
  - `train.py` dataset/dataloader 분기를 `src/data/builders.py`로 분리.
  - `solver.py`의 fusion profile 계산을 `src/losses/fusion.py`로 이동(기존 import 계약 유지).
  - `solver.py` loss 초기화를 `src/losses/factory.py`로 분리.
  - `EarlyStopping`/NaN/postfix/elapsed 및 results writer를 `src/utils/`로 분리하고 `solver.py`는 wrapper로 연결.
  - metric 계산 helper를 `src/metrics/segmentation.py`로 분리하고 `solver.py`에서 위임.
  - `args.test_only` 분기와 validation epoch 처리를 `src/scripts/test.py`, `src/scripts/val.py`로 분리.
  - RETOUCH active 지원 제거:
    - `train.py`에서 RETOUCH dataset 입력 차단.
    - launcher에서 RETOUCH dataset 입력 차단.
    - `scripts/configs/retouch_*.yaml` 제거.
  - 문서/런처 정리:
    - `README.md`, `scripts/train_launcher.md`에서 RETOUCH active 사용 안내 제거/정리.
  - 검증:
    - `tests/test_conn_layout_out8.py`, `tests/test_conn_fusion.py`, `tests/test_dist_aux_loss_selection.py` 통과.
    - launcher dry-run(`drive_train.yaml`, `cremi_train.yaml`) 정상.
- 보류:
  - `solver.py`를 더 얇게 만들기 위한 추가 경계 정리(동작 변경 없는 리팩터링만).

## 1차 수정 대상

### [train.py](../train.py)

- `args.dataset` 기반 dataset 생성 조건문을 `src/data/`로 이동한다.
- dataloader 생성 로직도 `src/data/` helper로 분리한다.
- `train.py`에는 다음 책임만 남긴다.
  - CLI argument parsing
  - seed 설정
  - fold 실행 범위 계산
  - model 생성 및 pretrained load
  - `Solver` 호출
- RETOUCH helper(`_resolve_retouch_case_roots`, `_get_retouch_fold_indices`) 삭제 (완료)

### [solver.py](../solver.py)

- `self.loss_func` / `self.loss_func_outer` 초기화 조건문을 `src/losses/`로 이동한다. (완료)
  - fusion / baseline loss 생성 규칙은 변경하지 않는다.
  - 기존 테스트가 import하는 loss 관련 함수는 wrapper 또는 re-export로 호환성을 유지한다.
- monitor / early stopper / checkpoint 관련 코드를 `src/utils/`로 이동한다. (완료)
  - `EarlyStopping`
  - best checkpoint 저장 helper
  - `results.csv`, `final_results.csv` 작성 helper
  - elapsed time formatting helper
- `args.test_only` 분기는 `src/scripts/test.py`로 이동한다. (완료)
  - `Solver.train()`에서는 test-only runner를 호출하는 형태로 유지한다.
- validation 실행 코드는 `src/scripts/val.py`로 이동한다. (완료)
  - 현재 `if val_loader is not None:` 아래 best model 갱신, early stopping, metric logging 동작을 유지한다.
- epoch/sample metric 계산 코드는 `src/metrics/`로 이동한다. (완료)
  - dice / Jac
  - clDice
  - precision / accuracy
  - Betti error
  - epoch metric 기본값 생성
- `math.isnan` 반복 조건은 small helper로 함수화한다. (완료)
- `epoch_postfix` 생성 로직은 helper로 함수화한다. (완료)
- `Solver.test_epoch(...)`의 외부 호출 계약은 우선 유지하고 내부 구현만 새 모듈에 위임한다.
- RETOUCH/SDL 관련 경로 정리:
  - `--use_SDL`, `--weights` 옵션 제거와 함께 solver의 SDL 경로 제거 완료.

## 기타 수정 필요 사항

- `src/` 하위 구조 후보:
  - `src/data/`
  - `src/losses/`
  - `src/metrics/`
  - `src/scripts/`
  - `src/utils/`
- 문서 정리 후 실제 리팩터링 시 `PROJECT_CONTEXT/modification.md`와 `PROJECT_CONTEXT/testing_notes.md`를 함께 갱신한다.
- 리팩터링 검증은 작은 순서로 진행한다.
  - syntax / import check
  - 관련 unit test
  - 가능하면 1-epoch smoke run
- 위 항목 중 syntax/import check 및 관련 unit test는 2026-04-29 기준 완료.

## ambiguous_modification_candidates

- RETOUCH 관련
  - active train/launcher 지원은 제거 완료.
  - legacy 파일(`data_loader/GetDataset_Retouch.py`, `data_loader/prepare_retouch_dataset.py`, `data_loader/retouch_weights`) 정리 여부는 별도 결정 필요.
- k-fold 실행 정책 변경 여부
  - `args.target_fold`
  - 전체 fold 실행과 단일 fold 실행의 output path / aggregation 호환성을 먼저 확인해야 한다.
- multi-class segmentation
  - 앞으로 진행하지 않을 예정.
- `solver.py` 분리 경계
  - `Solver`를 얇은 orchestrator로 만들지, 기존 class method wrapper를 많이 남길지 결정 필요.
  - 1차 리팩터링에서는 public method 계약 보존을 우선한다.
