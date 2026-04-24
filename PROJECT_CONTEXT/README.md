# PROJECT_CONTEXT README

Last updated: 2026-04-24

## 목적

- `PROJECT_CONTEXT/`는 포크의 현재 상태, 핵심 결정, 검증 결과를 빠르게 파악하기 위한 프로젝트 메모리입니다.
- 기본 원칙은 **upstream baseline 재현성 유지 + fork 확장 기능의 분리**입니다.

## 현재 상태 요약

### 1) Direction Grouping 제거 완료

- `24to8/coarse24to8` 관련 학습/모델/런처/집계 경로를 제거했습니다.
- 현재 실험 경로는 baseline 형식(`binary_<conn>_bce`, `<label_mode>_<conn>_<dist_aux_loss>`)만 사용합니다.

### 2) 학습 런처(config-first)

- `scripts/train_launcher.sh --config <yaml>` 단일 진입점 유지
- 실행 스케줄 생성은 `scripts/train_launcher_from_config.py` 담당
- 방향 그룹핑 관련 YAML 키(`direction_grouping`, `direction_fusion`)는 미지원

### 3) 주요 확장 경로(유지)

- CREMI 데이터셋 경로 (`data_loader/GetDataset_CREMI.py`, `train.py --dataset cremi`)
- Dist auxiliary(`cl_dice`) 경로 (`src/losses/dist_aux.py`)
- ETA/모니터링 유틸 (`scripts/eta_monitor.py`, `scripts/gpu_train_process_summary.sh`)

## 참고 문서

- 변경 결정/상태: `PROJECT_CONTEXT/modification.md`
- 검증 기록: `PROJECT_CONTEXT/testing_notes.md`
- 배경 참고: `PROJECT_CONTEXT/distance_map_change_overview.md`
- 논문 자료: `PROJECT_CONTEXT/references/`
