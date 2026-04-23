# `scripts/train_launcher.sh` 사용 방법

## 1) 기본 실행 형식

```bash
scripts/train_launcher.sh --config <yaml 경로> [--device GPU_ID] [--batch-size N] [--dry_run] [--test_only] [--pretrained [PATH]]
```

- `--config`: 실행 스펙 YAML 파일 경로 (필수)
- `--device`: YAML의 `device` 값을 CLI에서 덮어쓰기 (선택)
- `--batch-size`: YAML의 `batch_size` 값을 CLI에서 덮어쓰기 (선택)
- `--dry_run`: 학습은 실행하지 않고 생성될 `train.py` 명령만 출력 (선택)
- `--test_only`: YAML 설정과 무관하게 test-only 실행으로 강제 (선택)
- `--pretrained [PATH]`: test-only용 체크포인트 경로 오버라이드 (선택)
  - `PATH`를 생략하고 `--pretrained`만 주면 자동 경로 추론 사용
  - `PATH`에 `best_model.pth`처럼 `models/` 이후 상대 경로만 넣으면,
    런처가 자동으로 `<output_dir>/<dataset>/<experiment_name>/models/<PATH>`로 확장

## 2) 사전 조건

- 프로젝트 루트에서 실행
- `.venv` 환경 사용
- `PyYAML` 설치 필요 (`pyproject.toml`에 `pyyaml` 포함)

## 3) 자주 쓰는 예시

### DRIVE multi 학습 dry-run

```bash
scripts/train_launcher.sh --config scripts/configs/drive_multi_train.yaml --dry_run
```

### CHASE single 학습 실제 실행

```bash
scripts/train_launcher.sh --config scripts/configs/chasedb1_train.yaml
```

### OCTA500 multi를 GPU 1에서 dry-run

```bash
scripts/train_launcher.sh --config scripts/configs/octa500_multi_train.yaml --device 1 --dry_run
```

## 4) 제공되는 config 파일

- `scripts/configs/chasedb1_train.yaml`
- `scripts/configs/chasedb1_multi_train.yaml`
- `scripts/configs/drive_train.yaml`
- `scripts/configs/drive_multi_train.yaml`
- `scripts/configs/isic2018_train.yaml`
- `scripts/configs/isic2018_multi_train.yaml`
- `scripts/configs/octa500_train.yaml`
- `scripts/configs/octa500_multi_train.yaml`
- `scripts/configs/retouch_train.yaml`
- `scripts/configs/retouch_multi_train.yaml`
- `scripts/configs/retouch_cir_train.yaml`
- `scripts/configs/retouch_spe_train.yaml`
- `scripts/configs/retouch_top_train.yaml`

RETOUCH는 위 YAML 템플릿을 그대로 사용하면 됩니다.

### RETOUCH 단일 디바이스 실행 예시

```bash
scripts/train_launcher.sh --config scripts/configs/retouch_train.yaml
scripts/train_launcher.sh --config scripts/configs/retouch_cir_train.yaml
scripts/train_launcher.sh --config scripts/configs/retouch_spe_train.yaml
scripts/train_launcher.sh --config scripts/configs/retouch_top_train.yaml
```

### RETOUCH multi dry-run 예시

```bash
scripts/train_launcher.sh --config scripts/configs/retouch_multi_train.yaml --dry_run
```

### RETOUCH 사용자 정의 single 예시 (Spectrailis, 3-fold 모두 실행)

```yaml
dataset: retouch
mode: single
device: 0
direction_grouping: none
direction_fusion: weighted_sum
dist_sf_l1_gamma: 1.0
single:
  retouch_device: Spectrailis
  retouch_data_root: data/retouch
  conn_num: 8
  label_mode: binary
  dist_aux_loss: smooth_l1
  folds: 3
  target_folds: [1, 2, 3]
  epochs: 50
```

### RETOUCH 사용자 정의 multi 예시 (3개 디바이스, fold 1만 dry-run)

```yaml
dataset: retouch
mode: multi
device: 0
direction_grouping: none
direction_fusion: weighted_sum
dist_sf_l1_gamma: 1.0
multi:
  retouch_devices: [Cirrus, Spectrailis, Topcon]
  retouch_data_root: data/retouch
  retouch_target_folds: [1]
  epochs: 50
  folds: 3
  conn_nums: [8]
  label_modes: [binary]
  dist_aux_losses: [smooth_l1]
```

## 5) 출력 해석

- `--dry_run` 사용 시:
  - 각 실행 명령이 `[DRY_RUN] ...` 형태로 출력됨
  - 마지막에 `[INFO] Completed ... schedule with N run(s).` 출력
- 에러 발생 시:
  - `[ERROR] ...` 메시지 출력
  - 종료 코드 `2` 반환

## 6) test-only 사용법

`train.py --test_only`를 launcher YAML에서도 사용할 수 있습니다.

- 설정 위치:
  - `single.test_only: true` 또는 `multi.test_only: true`
  - 공통 기본값으로 top-level `test_only: true`도 사용 가능
- 체크포인트 지정:
  - `pretrained`를 지정하지 않으면 자동 경로를 사용:
    - `<output_dir>/<dataset>/<experiment_name>/models/best_model.pth`
  - `pretrained`를 지정하면 해당 경로를 사용
    - 상대 경로(`best_model.pth`, `checkpoint_best.pth`, `subdir/file.pth`)는
      자동으로 실험별 `models/` 하위 경로로 해석
  - `pretrained`는 포맷 문자열 지원:
    - 사용 가능 키: `{dataset}`, `{conn_num}`, `{label_mode}`, `{dist_aux_loss}`, `{direction_grouping}`, `{direction_fusion}`, `{experiment_name}`
- `batch_size` 오버라이드:
  - top-level 또는 `single`/`multi` 블록 내에서 지정 가능
- `test_only=true` + 실제 실행(`--dry_run` 아님) 시 체크포인트 파일이 없으면 즉시 에러로 종료

### single test-only 예시

```yaml
dataset: octa500
mode: single
device: 0
direction_grouping: none
direction_fusion: weighted_sum
single:
  octa_variant: 6M
  conn_num: 24
  label_mode: dist_inverted
  dist_aux_loss: smooth_l1
  folds: 1
  test_only: true
```

### multi test-only 예시 (자동 경로)

```yaml
dataset: [isic2018]
mode: multi
device: 0
direction_grouping: 24to8
direction_fusion: weighted_sum
multi:
  epochs: 500
  folds: 1
  conn_nums: [8]
  label_modes: [binary]
  test_only: true
```

## 7) 동작 규칙 참고

- `direction_fusion`은 문자열 또는 문자열 리스트를 지원:
  - 예: `direction_fusion: weighted_sum`
  - 예: `direction_fusion: [mean, conv1x1, attention_gating]`
  - 리스트를 주면 동일 스케줄을 fusion 값별로 확장해 순차 실행
- `direction_grouping: 24to8`인 경우:
  - `single` 모드에서 `conn_num: 24`는 허용되지 않음
  - `multi` 모드에서 `conn_nums`는 자동으로 `[8]`로 정규화됨
- `retouch` 계열(`retouch`, `retouch-Cirrus`, `retouch-Spectrailis`, `retouch-Topcon`)의 경우:
  - 현재 런처는 `folds: 3`만 허용
  - `target_fold`는 런처가 `target_folds`/`retouch_target_folds` 기준으로 자동 생성
  - `single`에서는 `conn_num: 8`만 허용
  - `multi`에서는 `conn_nums`가 자동으로 `[8]`로 정규화됨
  - paper-aligned preset은 `lr-update=poly`, `use_SDL` 미사용을 기본으로 생성됨
