# Testing Notes (Condensed)

Last updated: 2026-04-29

## Scope

- 이 문서는 현재 유효한 핵심 검증 결과만 기록합니다.
- 상세 이력 로그는 Git 기록을 기준으로 합니다.

## Current Test Suite

- `tests/test_conn_layout_out8.py`: out8 layout, offsets, voting shape, CLI/launcher validation, stdin import fallback
- `tests/test_conn_fusion.py`: gate/scaled_sum/conv_residual/decoder_guided fusion formulas, loss profile terms, decoder fusion shape/formula, experiment naming
- `tests/test_aggregate_results.py`: dataset grouping order, Fusion/Dec column parsing, drop-one delta logic, trash exclusion, drop-one default-axis rules, CHASE/ISIC section split

2026-04-28 기준 전체 suite: **47 passed**.

**주의**: 2026-04-29 변경(dataset PDF loss-split, ablation best-run 방식 변경) 이후 `test_aggregate_results.py` 일부 기대값이 stale 상태. 해당 변경에 대한 타겟 테스트는 통과; 전체 suite 기준 약 7건 실패.

## Residual Risks

- `.venv`에 `pytest`가 없을 수 있어 일부 테스트는 smoke 수준 검증으로 대체될 수 있다.
- `test_aggregate_results.py` 일부 기대값이 현재 코드와 불일치 → 후속 정리 필요.
- 현재 워크트리에 기존 변경사항이 포함되어 있어, 후속 머지 시 충돌 가능성이 남아 있음.
- smoke run에서 `cldice/precision/accuracy`가 `nan`을 반환하는 RuntimeWarning이 있으나, 학습 자체는 정상 종료함.

## Key Validations

### 2026-04-27: DRIVE multi-train smoke run

- `train.py` 1-epoch run (`binary_8_bce`) 정상 종료, `output/_smoke/drive/binary_8_bce/` 생성.
- `--smoke` launcher 기능: 실제 smoke 실행 완료 (`output_smoke/drive/binary_8_bce/`).

### 2026-04-28: `scaled_sum` residual-scale output path

- 실험명 `dist_scaled_sum_A_rs0.1_8_smooth_l1` 형식 분리 확인.
- `test_conn_fusion.py -k scaled_sum`, `test_aggregate_results.py -k scaled_sum` 통과.

### 2026-04-29: Config dry-run validations

| Config | 결과 |
|---|---|
| `cremi_drive_missing_combos.yaml` | `21 run(s)` (CREMI scaled_sum grid + DRIVE dist baseline) |
| `drive_dgrf_binary_segaux_w0.5.yaml` | `1 run(s)` + smoke 1-epoch 완료 (train_loss=2.68, dice=0.08) |
| `cremi_dgrf_binary.yaml` | `1 run(s)` |
| `cremi_dgrf_dist_smooth_l1.yaml` | `1 run(s)` |
| `other_datasets_dgrf_binary_segaux_w0.5.yaml` | `5 run(s)` (chase/cremi/isic/octa×2) |
| `cremi_dgrf_ablations.yaml` | `7 run(s)`, DGRF binary 복구 확인 |

### 2026-04-29: Aggregate report validations

- `scaled_sum/A/rs0.x` Fusion 컬럼 분리 표기 확인.
- `decoder_guided` 표기 `Fusion=decoder_guided`, `SegAux=none/segaux/w0.x`로 정리 확인.
- drop-one delta 표에 `clDice`, `Err (β0)`, `Err (β1)` 추가 및 Betti error의 lower-is-better delta coloring 확인.
- drop-one delta 표 variant 순서 `-SegAux` -> `-Fusion` 조정 확인.
- drop-one delta 표의 `Other datasets` 버킷도 dataset별 개별 table(`isic`, `chase` 등)로 분리 확인.
- cross-experiment / dataset / ablation / drop-one LaTeX table에 왼쪽 `No.` 열 추가 확인.
- `trash` dataset 제외, CHASE/ISIC 섹션 독립 분리 확인.
- dataset PDF loss-split 테이블 (loss별 별도 캡션) 확인.
- ambiguous `scaled_sum/A` (rs 미지정) 13개 폴더 `output/trash/ambiguous_scaled_sum_no_rs_2026-04-29/`로 이동 및 집계 제외 확인.
- `gpu_train_process_summary.sh` bad array subscript 오류 수정 확인.
