# `scripts/train_launcher.sh` 사용 방법

## 1) 기본 실행 형식

```bash
scripts/train_launcher.sh --config <yaml 경로> [--device GPU_ID] [--batch-size N] [--dry_run] [--test_only] [--pretrained [PATH]] [--smoke] [--smoke_limit N]
```

- `--config`: 실행 스펙 YAML 파일 경로 (필수)
- `--device`: YAML의 `device` 값을 CLI에서 덮어쓰기 (선택)
- `--batch-size`: YAML의 `batch_size` 값을 CLI에서 덮어쓰기 (선택)
- `--dry_run`: 실제 실행 없이 생성될 `train.py` 명령만 출력 (선택)
- `--smoke`: 최소 설정으로 smoke run 실행(기본: 1 run, 1 epoch, batch_size=1, target_fold=1, output_dir=output_smoke) (선택)
- `--smoke_limit N`: `--smoke` 시 첫 N개 run만 실행 (선택)
- `--test_only`: YAML과 무관하게 test-only 실행으로 강제 (선택)
- `--pretrained [PATH]`: test-only 체크포인트 경로 오버라이드 (선택)

## 2) 사전 조건

- 프로젝트 루트에서 실행
- `.venv` 환경 사용
- `PyYAML` 설치 필요

## 3) 핵심 예시

```bash
scripts/train_launcher.sh --config scripts/configs/drive_multi_train.yaml --dry_run
scripts/train_launcher.sh --config scripts/configs/drive_multi_train.yaml --smoke
scripts/train_launcher.sh --config scripts/configs/chasedb1_train.yaml
scripts/train_launcher.sh --config scripts/configs/octa500_multi_train.yaml --device 1 --dry_run
scripts/train_launcher.sh --config scripts/configs/drive_out8_multi_train.yaml --dry_run
scripts/train_launcher.sh --config scripts/configs/cremi_out8_multi_train.yaml --dry_run
scripts/train_launcher.sh --config scripts/configs/drive_fusion_gate_multi_train.yaml --dry_run
scripts/train_launcher.sh --config scripts/configs/drive_fusion_scaled_sum_multi_train.yaml --dry_run
scripts/train_launcher.sh --config scripts/configs/drive_cremi_dgrf_binary.yaml --dry_run
scripts/train_launcher.sh --config scripts/configs/cremi_dgrf_binary.yaml --dry_run
scripts/train_launcher.sh --config scripts/configs/cremi_dgrf_dist_smooth_l1.yaml --dry_run
scripts/train_launcher.sh --config scripts/configs/drive_dgrf_binary_segaux_w0.5.yaml --dry_run
scripts/train_launcher.sh --config scripts/configs/other_datasets_dgrf_binary_segaux_w0.5.yaml --dry_run
```

## 4) YAML 요약

### single

```yaml
dataset: chase
mode: single
device: 3
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
device: 3
multi:
  epochs: 500
  folds: 1
  conn_nums: [8, 24]
  label_modes: [binary, dist, dist_inverted]
  dist_aux_losses: [smooth_l1, gjml_sf_l1]
  # explicit run list (optional)
  experiments_only: false
  experiments:
    - datasets: [drive]   # optional override; defaults to top-level dataset list
      conn_num: 8
      conn_layout: standard8
      label_mode: dist
      dist_aux_loss: smooth_l1
      conn_fusion: none
  # fork-specific fusion ablations
  conn_fusions: [none, gate, scaled_sum, conv_residual]
  fusion_loss_profiles: [A, B, C]
  fusion_lambda_inner: 0.2
  fusion_lambda_outer: 0.05
  fusion_lambda_fused: 0.3
  fusion_residual_scales: [0.1, 0.2, 0.3, 0.5]
```

## 5) test-only

- `single.test_only: true` 또는 `multi.test_only: true` 지원
- `pretrained` 미지정 시 자동 경로:
  - `<output_dir>/<dataset>/<experiment_name>/models/best_model.pth`
- `pretrained` 포맷 문자열 지원 키:
  - `{dataset}`, `{conn_num}`, `{label_mode}`, `{dist_aux_loss}`, `{conn_layout}`, `{conn_fusion}`, `{fusion_loss_profile}`, `{experiment_name}`

## 6) 주의 사항

- `direction_grouping`, `direction_fusion` 키는 제거되어 더 이상 지원되지 않습니다.
- `decoder_fusion`, `decoder_fusions`, `lambda_vote_aux` 키는 제거되어 더 이상 지원되지 않습니다.
- `conn_fusion != none`은 현재 single-class/`conn_num=8`/`conn_layout=standard8` 조합만 지원합니다.
- RETOUCH 계열은 현재 `folds: 3` 정책과 `conn_num: 8` 정책을 유지합니다.
- `scripts/configs/drive_cremi_dgrf_binary.yaml`은 `binary + decoder_guided/A`를 `drive, cremi`에만 별도 스케줄링하는 fork-specific config입니다.
- `scripts/configs/cremi_dgrf_binary.yaml`은 `cremi` 전용 `binary + decoder_guided/A` 단일 run config입니다.
- `scripts/configs/cremi_dgrf_dist_smooth_l1.yaml`은 `cremi` 전용 `dist + decoder_guided/A + smooth_l1` 단일 run config입니다.
- `scripts/configs/drive_dgrf_binary_segaux_w0.5.yaml`은 `drive` 전용 `binary + decoder_guided/A + segaux_w0.5` 단일 run config입니다.
- `scripts/configs/other_datasets_dgrf_binary_segaux_w0.5.yaml`은 `chase`, `cremi`, `isic2018`, `octa500-3M`, `octa500-6M`에 같은 `binary + decoder_guided/A + segaux_w0.5` 규칙을 적용하는 config입니다.
