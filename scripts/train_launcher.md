# `scripts/train_launcher.sh` 사용 방법

## 1) 기본 실행 형식

```bash
scripts/train_launcher.sh --config <yaml 경로> [--device GPU_ID] [--dry_run]
```

- `--config`: 실행 스펙 YAML 파일 경로 (필수)
- `--device`: YAML의 `device` 값을 CLI에서 덮어쓰기 (선택)
- `--dry_run`: 학습은 실행하지 않고 생성될 `train.py` 명령만 출력 (선택)

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

## 5) 출력 해석

- `--dry_run` 사용 시:
  - 각 실행 명령이 `[DRY_RUN] ...` 형태로 출력됨
  - 마지막에 `[INFO] Completed ... schedule with N run(s).` 출력
- 에러 발생 시:
  - `[ERROR] ...` 메시지 출력
  - 종료 코드 `2` 반환

## 6) 동작 규칙 참고

- `direction_grouping: coarse24to8`인 경우:
  - `single` 모드에서 `conn_num: 24`는 허용되지 않음
  - `multi` 모드에서 `conn_nums`는 자동으로 `[8]`로 정규화됨
