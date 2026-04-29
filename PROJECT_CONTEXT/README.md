# PROJECT_CONTEXT README

Last updated: 2026-04-25

## 목적

- `PROJECT_CONTEXT/`는 포크의 현재 상태, 핵심 결정, 검증 결과를 빠르게 파악하기 위한 프로젝트 메모리입니다.
- 기본 원칙은 **upstream baseline 재현성 유지 + fork 확장 기능의 분리**입니다.

## 현재 상태 요약

### 1) Direction Grouping 제거 완료

- `24to8/coarse24to8` 관련 학습/모델/런처/집계 경로를 제거했습니다.
- 현재 실험 경로는 baseline 형식과 fork-specific fusion 형식을 함께 지원합니다.
  - baseline: `binary_<conn>_bce`, `<label_mode>_<conn>_<dist_aux_loss>`
  - fusion: `binary_<fusion>_<profile>_<conn>_bce`, `<label_mode>_<fusion>_<profile>_<conn>_<dist_aux_loss>`

### 2) 학습 런처(config-first)

- `scripts/train_launcher.sh --config <yaml>` 단일 진입점 유지
- 실행 스케줄 생성은 `scripts/train_launcher_from_config.py` 담당
- 공식 `label_mode`는 `binary`, `dist`, `dist_inverted` 세 가지로 통일
- 방향 그룹핑 관련 YAML 키(`direction_grouping`, `direction_fusion`)는 미지원

### 3) 주요 확장 경로(유지)

- CREMI 데이터셋 경로 (`data_loader/GetDataset_CREMI.py`, `train.py --dataset cremi`)
- Dist auxiliary(`cl_dice`) 경로 (`src/losses/dist_aux.py`)
- ETA/모니터링 유틸 (`scripts/eta_monitor.py`, `scripts/gpu_train_process_summary.sh`)
- Fork-specific connectivity layout:
  - `--conn_layout out8` 추가
  - scope: single-class only
  - offsets: `(-2,-2), (-2,0), (-2,2), (0,-2), (0,2), (2,-2), (2,0), (2,2)`
- launcher helper import은 stdin 기반 실행(`uv run bash ...`)에서도 안전하게 동작하도록 fallback 처리됨

### 4) 집계 출력 그룹화

- `scripts/aggregate_results.py`의 cross-experiment aggregate 출력 기본 스코프는 다음 dataset만 포함합니다.
  - `DRIVE`
  - `CHASE`
  - `OCTA3M&6M`
- 전체 counted dataset을 종합하려면 `--all`을 명시해야 합니다.
- 기본/`--all` 공통으로 dataset summary LaTeX / CSV 출력은 다음 순서로 그룹화됩니다.
  - `CREMI`
  - `DRIVE`
  - `CHASE`
  - `ISIC`
  - `OCTA3M&6M`
  - `Other datasets`
- 각 dataset table은 여전히 개별 표로 유지됩니다.
- aggregate PDF에서는 dataset/table 경계마다 `\clearpage`를 넣어 서로 다른 dataset이 같은 페이지에 섞이지 않도록 합니다.
- `out8` connectivity는 table의 `Conn` 열에서 `8'`로 축약 표기됩니다.
  - launcher configs:
    - `scripts/configs/drive_out8_multi_train.yaml`
    - `scripts/configs/cremi_out8_multi_train.yaml`

### 5) Fork-specific `inner8/out8` fusion 실험 경로 추가

- 기본값 `--conn_fusion none`은 legacy 경로를 그대로 유지합니다.
- fusion 활성화(`gate`, `scaled_sum`, `conv_residual`) 시:
  - `C3`: `standard8(inner8)` logits head
  - `C5`: `out8(outer8)` logits head
  - `C_fused`: outer logits를 standard 방향 순서로 정렬 후 logits 수준에서 fusion
- outer→standard 정렬 인덱스는 고정:
  - `OUTER_8_TO_STANDARD8_INDEX = [7, 6, 5, 4, 3, 2, 1, 0]`
- fusion 제약:
  - single-class only
  - `conn_num=8`, `conn_layout=standard8` only
- fusion 실험명은 legacy와 분리:
  - 예: `binary_gate_A_8_bce`, `dist_scaled_sum_C_8_smooth_l1`

### 6) Explicit Decoder Fusion Removal

- 명시적 `decoder_fusion` 옵션은 제거되었습니다.
- 더 이상 지원하지 않는 키:
  - `--decoder_fusion`
  - `decoder_fusions`
  - `--lambda_vote_aux`
- 유지되는 관련 경로:
  - `conn_fusion`
  - `conn_fusion=decoder_guided`
  - `SegAux`

## 참고 문서

- 변경 결정/상태: `PROJECT_CONTEXT/modification.md`
- 검증 기록: `PROJECT_CONTEXT/testing_notes.md`
- 아키텍처 상세: `PROJECT_CONTEXT/change_summary.md`
- 논문 자료: `PROJECT_CONTEXT/references/`
