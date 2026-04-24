# Project Modification Snapshot

Last updated: 2026-04-24

## Purpose

- Upstream baseline 재현성을 유지한다.
- Fork 확장은 additive 방식으로 유지한다.
- 실험/집계 인터페이스는 명확하고 단순하게 관리한다.

## Current Valid State

- Training entrypoint: `train.py`
- Main aggregation utility: `scripts/aggregate_results.py`
- Output naming:
  - binary: `binary_<conn_num>_bce`
  - dist: `<label_mode>_<conn_num>_<dist_aux_loss>`

## High-Impact Decisions

1. Baseline compatibility first

- baseline 경로를 우선 보존하고, fork 기능은 분리 가능한 옵션만 유지한다.

2. Dist auxiliary supervision

- 최종 분할 출력은 binary mask objective로 감독한다.
- distance 정보는 auxiliary supervision으로 사용한다.

3. Reproducibility

- `--seed` 기반 deterministic 설정과 worker seeding을 유지한다.

4. Aggregation policy

- single-run / indexed run / mixed root를 모두 수용한다.
- dataset별 평균표/요약 PDF 산출을 유지한다.

## 2026-04-24 Update: 24to8 Removal

- `24to8/coarse24to8` 기능 제거:
  - `train.py` CLI에서 `--direction_grouping`, `--direction_fusion` 제거
  - `model/DconnNet.py`에서 coarse reducer 경로 제거
  - `model/coarse_direction_grouping.py` 삭제
- 런처/모니터링/집계 정리:
  - `scripts/train_launcher_from_config.py`에서 direction 키/스케줄/명령 인자 제거
  - `scripts/gpu_train_process_summary.sh`에서 direction 컬럼 및 레거시 이름 매핑 제거
  - `scripts/aggregate_results.py`에서 direction 메타 파싱/정렬/출력 제거
- 설정/테스트 정리:
  - `scripts/configs/*.yaml`에서 direction 관련 키 제거
  - `tests/test_coarse_direction_grouping.py` 및 관련 notebook 제거
- 산출물 정리:
  - `output/**`의 `*24to8*`, `*coarse24to8*` 디렉터리 삭제

## Known Constraints

- `.venv`에 `pytest`가 없을 수 있어 일부 테스트는 smoke 수준 검증으로 대체될 수 있다.
- third-party metric 유틸(clDice 등)에서 경고가 출력될 수 있다.

## Next Practical Steps

1. `scripts/aggregate_results.py`에 대한 소형 fixture 회귀 테스트를 추가한다.
2. `scripts/train_launcher_from_config.py`의 schema validation 테스트를 추가한다.
3. 문서 예시 커맨드를 주기적으로 정리해 현재 인터페이스와 맞춘다.
