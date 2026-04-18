# PROJECT_CONTEXT README

Last updated: 2026-04-19

## 목적

- `PROJECT_CONTEXT/`는 이 포크의 현재 상태, 주요 의사결정, 검증 결과를 빠르게 파악하기 위한 프로젝트 메모리입니다.
- 기준 원칙은 **upstream 재현성 유지 + fork 확장 기능의 명확한 분리**입니다.

## 현재까지 반영된 핵심 변경

### 1) Direction Grouping (`coarse24to8`)

- 24방향 proto 출력을 8방향 canonical 출력으로 축약하는 fork 경로 추가
- 그룹 순서를 canonical-8 기준(`SE,S,SW,E,W,NE,N,NW`)으로 정렬
- 후처리 재정렬 의존 제거(출력은 canonical by construction)
- `mean`, `weighted_sum`, `conv1x1`, `attention_gating` fusion 지원
- 관련 반영:
  - `model/coarse_direction_grouping.py`
  - `model/DconnNet.py`
  - `train.py` (`--direction_grouping`, `--direction_fusion`)
  - `tests/test_coarse_direction_grouping.py`, notebook

### 2) 학습 런처 통합 (config-first)

- `scripts/train_launcher.sh --config <yaml> [--device] [--dry_run]`로 단일화
- 실행 스케줄/검증 로직은 `scripts/train_launcher_from_config.py`로 분리
- dataset별 single/multi YAML 스펙 추가:
  - `scripts/configs/*.yaml`
- 기존 `{dataset}_train.sh`, `{dataset}_multi_train.sh`는 deprecate wrapper로 유지
- `coarse24to8` 시 `conn_num` 정책 강제:
  - single: `conn_num=8`만 허용
  - multi: conn sweep을 `[8]`로 정규화

### 3) 실험 모니터링/알림

- `scripts/gpu_train_process_summary.sh`:
  - GPU 프로세스 요약 + train 인자 파싱
  - `results.csv` 기반 ETA 컬럼 통합:
    - `EPOCH_PROGRESS`, `ETA_DURATION`, `ETA_FINISH`, `ETA_STATUS`
- `scripts/eta_monitor.py` + `scripts/eta_monitor.md`:
  - CSV 직접 지정 기반 ETA 계산/감시(`--watch`)
- `scripts/telegram_alert.py`:
  - session alert 정책/메타데이터 표준화

### 4) 데이터셋/학습 경로 확장

- `octa500-3M`, `octa500-6M` 학습 경로 추가
- `MyDataset_OCTA500` 데이터셋 로더 반영

## 참고 문서

- 변경 히스토리/결정: `PROJECT_CONTEXT/modification.md`
- 검증 결과/명령: `PROJECT_CONTEXT/testing_notes.md`
- 배경 참고: `PROJECT_CONTEXT/distance_map_change_overview.md`
- 논문/레퍼런스: `PROJECT_CONTEXT/references/`

## 운영 원칙 요약

- upstream baseline 동작을 깨지 않도록 최소 침습으로 확장
- fork 기능은 CLI/config로 on/off 가능하게 유지
- 기능 변경 시 `modification.md`, 검증 시 `testing_notes.md`를 동기화
