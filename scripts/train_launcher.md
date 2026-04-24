# `scripts/train_launcher.sh` 사용 방법

## 1) 기본 실행 형식

```bash
scripts/train_launcher.sh --config <yaml 경로> [--device GPU_ID] [--batch-size N] [--dry_run] [--test_only] [--pretrained [PATH]]
```

- `--config`: 실행 스펙 YAML 파일 경로 (필수)
- `--device`: YAML의 `device` 값을 CLI에서 덮어쓰기 (선택)
- `--batch-size`: YAML의 `batch_size` 값을 CLI에서 덮어쓰기 (선택)
- `--dry_run`: 실제 실행 없이 생성될 `train.py` 명령만 출력 (선택)
- `--test_only`: YAML과 무관하게 test-only 실행으로 강제 (선택)
- `--pretrained [PATH]`: test-only 체크포인트 경로 오버라이드 (선택)

## 2) 사전 조건

- 프로젝트 루트에서 실행
- `.venv` 환경 사용
- `PyYAML` 설치 필요

## 3) 핵심 예시

```bash
scripts/train_launcher.sh --config scripts/configs/drive_multi_train.yaml --dry_run
scripts/train_launcher.sh --config scripts/configs/chasedb1_train.yaml
scripts/train_launcher.sh --config scripts/configs/octa500_multi_train.yaml --device 1 --dry_run
```

## 4) YAML 요약

### single

```yaml
dataset: chase
mode: single
device: 0
single:
  conn_num: 8
  label_mode: binary
  dist_aux_loss: smooth_l1
  folds: 1
  epochs: 130
```

### multi

```yaml
dataset: [chase, drive]
mode: multi
device: 0
multi:
  epochs: 500
  folds: 1
  conn_nums: [8, 24]
  label_modes: [binary, dist, dist_inverted]
  dist_aux_losses: [smooth_l1, gjml_sf_l1]
```

## 5) test-only

- `single.test_only: true` 또는 `multi.test_only: true` 지원
- `pretrained` 미지정 시 자동 경로:
  - `<output_dir>/<dataset>/<experiment_name>/models/best_model.pth`
- `pretrained` 포맷 문자열 지원 키:
  - `{dataset}`, `{conn_num}`, `{label_mode}`, `{dist_aux_loss}`, `{experiment_name}`

## 6) 주의 사항

- `direction_grouping`, `direction_fusion` 키는 제거되어 더 이상 지원되지 않습니다.
- RETOUCH 계열은 현재 `folds: 3` 정책과 `conn_num: 8` 정책을 유지합니다.
