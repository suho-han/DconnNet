# Distance Map 변경 한눈에 보기 (CHASE 기준)

> 기준 문서: `PROJECT_CONTEXT/modification.md`  
> 정리 기준일: 2026-04-15  
> 범위: distance-map(`dist`, `dist_inverted`) 학습/평가 경로 중심

## 1) 변경 목적

- upstream binary baseline 재현성은 유지
- fork 확장으로 distance-map GT 학습 경로를 additive하게 추가
- baseline 경로와 확장 경로를 설정(`label_mode`)으로 명확히 분리

## 2) 핵심 변경 요약 (최신 유효 상태)

| 구분 | 파일 | 핵심 변경 | baseline 영향 |
|---|---|---|---|
| 데이터 로딩 | `data_loader/GetDataset_CHASE.py` | `label_mode` 지원(`binary`, `dist`, `dist_inverted`), distance GT `.npy` 로딩 | 없음(기존 binary 유지) |
| 실행 인자/출력 | `train.py` | `--label_mode`, `--tau`, `--sigma`, `--dist_aux_loss`, `--dist_sf_l1_gamma`, `--output_dir` 추가, 저장 경로를 `<output_dir>/<fold_scope>/<experiment_name>`로 정규화. 현재 `conn_num` 유효값은 `8`, `24` | 경미(CLI/저장경로 인터페이스) |
| 거리 기반 타깃/손실 | `connect_loss.py` | distance 전용 경로 추가(`distance_connectivity_matrix`, affinity regression). **최신 설계(2026-04-15): 5x5는 center-excluding `24` directional channels로 통일**, final 출력은 binary mask로 BCE+Dice supervision, distance affinity는 보조(supervision)로 유지 | 있음(single-class CHASE 손실 경로) |
| 평가/메트릭 | `solver.py` | dist 평가를 `pred_score > 0`에서 분리/개선하여 **`dist_score_to_binary(..., >0.5)`** 적용, GT는 `(y>0)` 기준으로 이진화 | 있음(single-class CHASE eval 경로) |
| 실행 스크립트 | `scripts/chasedb1_train.sh` (single launcher) | CHASE 학습 진입점을 단일 스크립트로 유지 (`conn_num`, `label_mode`, `dist_aux_loss`, `dist_sf_l1_gamma`, `epochs`) | 없음(추가 스크립트) |
| 디버그/아티팩트 | `scripts/rebuild_dist_signed_artifacts.py` 등 | collapse/투표/체크포인트 진단 및 리포트 생성 흐름 정리 | 없음(유틸리티) |

## 3) dist 경로 설계 변화 포인트 (중요)

1. 초기 dist 경로는 final output을 raw distance에 직접 회귀(`SmoothL1`)하던 구조였음.
2. 이 구조에서 zero-collapse(학습) + all-foreground 성향(평가) 문제가 관찰됨.
3. 최신 fix(2026-04-10)에서 final output supervision을 binary mask(BCE+Dice)로 정렬:
   - segmentation 출력 목적과 손실 정의를 일치
   - distance 정보는 directional affinity 회귀 보조 항으로 유지
4. 결과적으로 binary 경로와 dist 경로의 “최종 마스크 학습 목표”는 맞추고, dist 경로의 추가 이점은 affinity supervision으로 분리함.

## 4) 현재 실험/검증 상태 요약

- 문법 검증:
  - `connect_loss.py`, `solver.py`, `train.py` py_compile 통과 기록 존재
- smoke 결과:
  - dist 경로에서 과거 metric artifact(`>0` 임계) 문제를 분리/수정한 기록 존재
  - 최신 loss/eval 정렬 후 1/3/30 epoch smoke에서 지표 개선 추세 기록 존재
- 해석:
  - “거리맵 자체를 final 출력으로 직접 맞추는 방식”보다
  - “final은 binary mask, dist는 affinity 보조”가 현재 채택된 안정 경로

## 5) 아직 남아있는 논점

- multi-class distance 확장 여부/범위 미확정(현재 dist는 single-class 중심)
- dist 전용 하이퍼파라미터(`sigma`, loss weight) 튜닝 기준 정리 필요

## 6) 문서/도움말 정합성 (2026-04-12 반영)

- `train.py --label_mode` 도움말은 실제 지원값(`binary`, `dist`, `dist_inverted`)과 일치
- 저장 경로 설명은 실제 동작과 일치:
  - `binary` -> `output_dir/<fold_scope>/binary_<conn_num>_bce`
  - `dist_*` -> `output_dir/<fold_scope>/<label_mode>_<conn_num>_<dist_aux_loss>`
- `scripts/chasedb1_train.sh --help`는 `output/5folds/...`와 `output/1fold/...` 예시를 포함
- `scripts/multi_train.sh`는 5-fold batch 실행을 명시적으로 `--folds 5`로 고정
- `scripts/aggregate_kfold_results.py` 기본 `--input-root`는 `output`
- `scripts/aggregate_kfold_results.py`는 `output/{1fold,5folds}/...`를 함께 감지해 동일 실행에서 둘 다 집계
- 본 문서와 `modification.md`의 dist 손실 설명은 최신 구현(최종 출력 BCE+Dice, affinity/bicon은 `SmoothL1` 또는 선택적 `GJML + SF-L1` 보조, CHASE dist에서는 bicon 제외) 기준으로 동기화

## 7) 기존 모델 대비 변경점 (파일/함수 단위)

### train.py

- `parse_args()`
  - `--label_mode` 추가: `binary`, `dist`, `dist_inverted`
  - `--tau`, `--sigma` 추가: distance affinity 관련 하이퍼파라미터 전달
  - `--dist_aux_loss`, `--dist_sf_l1_gamma` 추가: distance auxiliary regression loss 선택
  - `--output_dir` 추가 및 저장 경로 규칙 적용:
    - `binary` -> `output_dir/<fold_scope>/binary_<conn_num>_bce`
    - `dist_*` -> `output_dir/<fold_scope>/<label_mode>_<conn_num>_<dist_aux_loss>`
- `main(args)`
  - CHASE dataloader 생성 시 `label_mode` 전달
  - `Solver.train(...)` 호출 시 `label_mode` 전달

### data_loader/GetDataset_CHASE.py

- `default_DRIVE_loader(..., label_mode='binary')`
  - `binary`와 `dist_*` 분기 로딩 지원
  - `dist`/`dist_inverted`에서 `.npy` distance GT 로딩
- `MyDataset_CHASE.__init__(..., label_mode='binary')`
  - `label_mode`에 따라 GT 경로/타입 선택

### connect_loss.py

- `distance_affinity_matrix(dist_map, conn_num, sigma=2.0)`
  - distance map으로부터 directional affinity target 생성
- `connect_loss.__init__(..., label_mode=None, conn_num=8, sigma=2.0)`
  - distance 분기 제어를 위한 설정값 보관
- `gjml_loss(pred, target)`, `stable_focal_l1_loss(pred, target, gamma)`, `dist_aux_regression_loss(pred, target)`
  - dist auxiliary regression을 `SmoothL1` 또는 `GJML + SF-L1`로 선택
- `dist_target_to_mask(target)`
  - distance GT를 최종 segmentation supervision용 binary mask로 변환
- `binary_edge_target_from_affinity(affinity_target)`
  - binary affinity로부터 edge target 생성
- `dist_edge_loss(vote_out, mask_target)`
  - dist 경로 edge supervision 계산
- `single_class_forward(c_map, target)`
  - `binary`: 기존 BCE/Dice + connectivity 경로 유지
  - `dist_*`: 최종 출력은 binary mask(BCE+Dice)로 supervision
  - `dist_*`: `affinity_l`은 `SmoothL1` 또는 선택적 `GJML + SF-L1` 보조 supervision, `bicon_l`은 CHASE에서는 제외(그 외 데이터셋은 동일 selector 적용)
- `set_dist_edge_stat_collection()`, `reset_dist_edge_stats()`, `get_dist_edge_stats()`
  - dist 학습 중 edge target 통계 수집용

### solver.py

- `dist_to_binary(x, label_mode)`
  - dist GT를 메트릭 계산용 binary mask로 변환(`dist_*`는 `x>0`)
- `dist_score_to_binary(x)`
  - voting 결과 score map을 `>0.5` 임계값으로 binary화
- `train(...)`
  - `connect_loss` 생성 시 `label_mode`, `sigma` 전달
  - dist 모드에서 edge 통계 수집/로깅 연동
- `test_epoch(...)`
  - dist 평가 경로를 `sigmoid -> Bilateral_voting -> >0.5`로 고정
  - GT는 `(y_test_raw > 0)` 기준 binary화

### scripts/chasedb1_train.sh (unified launcher)

- `.venv/bin/python` 고정 인터프리터 사용(없으면 종료)
- repo root 기준 실행 보장
- 통합 인자 지원:
  - `--conn_num`
  - `--label_mode` (`binary`, `dist`, `dist_inverted`)
  - `--dist_aux_loss` (`smooth_l1`, `gjml_sf_l1`)
  - `--dist_sf_l1_gamma`
  - `--epochs` (미지정 시 정책값 자동 적용, 지정 시 직접 override)
- 기본 epoch 정책:
  - `binary + conn_num=8` -> `130`
  - `binary + conn_num=24` -> `390`
  - `dist/dist_inverted` -> `260`

### Wrapper scripts

- 현재 저장소 상태 기준으로 `chasedb1` 전용 래퍼 스크립트들은 제거됨
- CHASE 학습 실행은 `scripts/chasedb1_train.sh` 단일 엔트리포인트 사용
