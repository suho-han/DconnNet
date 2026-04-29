# Project Modification Snapshot

Last updated: 2026-04-27

## Purpose

- Upstream baseline 재현성을 유지한다.
- Fork 확장은 additive 방식으로 유지한다.
- 실험/집계 인터페이스는 명확하고 단순하게 관리한다.

## Current Valid State

- Training entrypoint: `train.py`
- Main aggregation utility: `scripts/aggregate_results.py`
- Output naming:
  - baseline binary: `binary_<conn_num>_bce`
  - baseline dist: `<label_mode>_<conn_num>_<dist_aux_loss>`
  - fusion binary: `binary_<conn_fusion>_<fusion_loss_profile>_<conn_num>_bce`
  - fusion dist: `<label_mode>_<conn_fusion>_<fusion_loss_profile>_<conn_num>_<dist_aux_loss>`

## High-Impact Decisions

1. Baseline compatibility first

- baseline 경로를 우선 보존하고, fork 기능은 분리 가능한 옵션만 유지한다.

2. Dist auxiliary supervision

- 최종 분할 출력은 binary mask objective로 감독한다.
- distance 정보는 auxiliary supervision으로 사용한다.
- 공식 `label_mode`는 `binary`, `dist`, `dist_inverted`만 유지한다.

3. Reproducibility

- `--seed` 기반 deterministic 설정과 worker seeding을 유지한다.

4. Aggregation policy

- single-run / indexed run / mixed root를 모두 수용한다.
- dataset별 평균표/요약 PDF 산출을 유지한다.

## 2026-04-29 Update: Explicit `decoder_fusion` Removal

- 제거 범위:
  - `train.py` CLI의 `--decoder_fusion`, `--lambda_vote_aux`
  - `model/DconnNet.py`의 explicit decoder-fusion segmentation head
  - `solver.py`의 decoder-fusion 전용 main segmentation loss 조합
  - launcher / gpu summary / aggregate metadata의 explicit `decoder_fusion` 경로
- 유지 범위:
  - baseline 경로
  - `conn_fusion`
  - `conn_fusion=decoder_guided`
  - `SegAux`
- 현재 정책:
  - legacy decoder-fusion YAML 키(`decoder_fusion`, `decoder_fusions`, `lambda_vote_aux`)는 launcher에서 unsupported로 실패한다.
  - 실험명은 더 이상 `..._dec_<mode>_...` suffix를 생성하지 않는다.
  - aggregate summary는 `decoder_fusion` 컬럼 없이 `Fusion`/`SegAux` 기준으로만 정리한다.

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

## 2026-04-24 Update: `out8` Connectivity Layout

- New fork-specific single-class layout:
  - CLI/config key: `conn_layout`
  - defaults:
    - `conn_num=8 -> standard8`
    - `conn_num=24 -> full24`
  - new option:
    - `conn_num=8 + conn_layout=out8`
- `out8` definition:
  - 5x5 neighborhood에서 outer offsets 8개만 사용
  - order: `(-2,-2), (-2,0), (-2,2), (0,-2), (0,2), (2,-2), (2,0), (2,2)`
- Compatibility policy:
  - 기존 `8` / `24` 경로는 명시적으로 `out8`을 선택하지 않으면 유지
  - multi-class는 계속 `standard8` only
- Runtime coverage:
  - train / solver voting / launcher / debug reconstruction script 지원

## 2026-04-24 Update: Aggregation Table Grouping

- `scripts/aggregate_results.py`의 dataset summary LaTeX 출력에 그룹 순서를 추가했다.
- 그룹 순서:
  - `CREMI`
  - `DRIVE`
  - `OCTA3M&6M`
  - `Other datasets`
- 같은 그룹 안의 개별 dataset table은 기존처럼 분리된 채 유지된다.
- `scripts/aggregate_results.py`의 cross-experiment dataset CSV 출력 순서도 같은 그룹 기준으로 맞췄다.
- `out8` connectivity는 table의 `Conn` 열에서 `8'`로 축약 표기한다.

## 2026-04-24 Update: Launcher Import Robustness

- `scripts/train_launcher_from_config.py`가 `uv run bash scripts/gpu_train_process_summary.sh`처럼 stdin 기반 helper import에서도 실패하지 않도록 했다.
- `autorootcwd` import 실패 시 repository root를 수동으로 `sys.path`에 추가하는 fallback을 넣었다.

## 2026-04-25 Update: `inner8/out8` Fusion Ablation

- New fork-specific CLI path:
  - `--conn_fusion {none,gate,scaled_sum,conv_residual}`
  - `--fusion_loss_profile {A,B,C}`
  - `--fusion_lambda_inner`, `--fusion_lambda_outer`, `--fusion_lambda_fused`
  - `--fusion_residual_scale` (default `0.2`)
- Compatibility policy:
  - `conn_fusion=none`은 모델 반환 타입/학습 경로/실험명을 legacy와 동일하게 유지
  - fusion은 `num_class=1`, `conn_num=8`, `conn_layout=standard8` 조합에서만 허용
- Model path:
  - final decoder에서 `inner8`/`outer8` 두 logits head 생성
  - `outer8` native logits는 branch loss에 그대로 사용
  - fusion 전 방향 정렬:
    - native order: `(-2,-2),(-2,0),(-2,2),(0,-2),(0,2),(2,-2),(2,0),(2,2)`
    - standard-aligned order: `(2,2),(2,0),(2,-2),(0,2),(0,-2),(-2,2),(-2,0),(-2,-2)`
    - index mapping: `[7,6,5,4,3,2,1,0]`
- Loss policy (1차):
  - `L_seg = vote(C_fused) + dice(C_fused)`
  - `L_Cfused = affinity(C_fused)`
  - `L_C3 = affinity(C3, standard8)`
  - `L_C5 = affinity(C5_native, out8)`
  - profile A/B/C 가중합만 사용하고 `edge/bicon/total`은 fusion objective에서 제외
- Launcher/aggregation/monitoring:
  - fusion-aware experiment naming 지원
  - config에서 `conn_fusions`, `fusion_loss_profiles`, `fusion_residual_scales` sweep 지원

## 2026-04-27 Update: Aggregation `label_mode` Column Cleanup

- `scripts/aggregate_results.py`의 cross-experiment mean LaTeX/CSV 출력에서 첫 컬럼(`label_mode`)에 Conn/Fusion 정보가 중복으로 포함되던 문제를 정리했다.
- 이제 `label_mode` 컬럼은 `binary/dist/dist_inverted`만 표시하고, 연결/퓨전 정보는 각각 `Conn`/`Fusion`/`Dec` 컬럼에만 표시된다.

## 2026-04-27 Update: Ordered Experiment Name Parsing (`label_mode/fusion/conn/loss`)

- `scripts/aggregate_results.py`의 `parse_experiment_metadata`에 fork naming 우선 파서를 추가했다.
- 순서가 `label_mode / fusion / conn / loss`인 실험명(예: `dist_inverted_decoder_guided_A_8_gjml_sf_l1`)을 우선 해석한다.
- loss 뒤 suffix(`_segaux`, `_segaux_w...`)가 있어도 `conn_num`, `loss`, `fusion` 메타가 유지되도록 했다.

## 2026-04-27 Update: Table Sort/Display Policy for Fusion vs Dec

- `scripts/aggregate_results.py`의 cross-experiment summary 정렬 키를 표 기준 컬럼 순서로 조정했다:
  - `label_mode -> conn -> fusion -> Dec -> loss`
- `decoder_guided`은 표 표시에서 `Fusion` 컬럼이 아니라 `Dec` 컬럼으로 이동하도록 변경했다.
  - 표시 규칙: `Fusion=none`, `Dec=decoder_guided/<profile>`

## 2026-04-27 Update: Ablation PDF Dataset Split

- `scripts/aggregate_results.py`의 `write_ablation_latex`를 변경하여 각 ablation 섹션을 단일 테이블 대신 최대 3개 테이블로 분할했다.
  - `CREMI` 전용
  - `DRIVE` 전용
  - `Other datasets` 전용 (`cremi`, `drive` 제외)
- 각 테이블은 해당 버킷 내부 dataset들만 컬럼으로 포함하며, `Dice/IoU` best/second 강조도 버킷 내부 기준으로 계산한다.
- 캡션을 `Ablation on <name> (<bucket>)` 형태로 변경했다.

## 2026-04-27 Update: Experiment Means Dedup Fix

- `scripts/aggregate_results.py`의 cross-experiment dedup key에 `experiment_source`(원본 실험 디렉토리명)를 포함했다.
- `*_segaux`, `*_segaux_w...` 같은 suffix 변형 실험이 이전처럼 하나로 합쳐지지 않고 테이블/CSV에 개별 행으로 유지된다.

## 2026-04-27 Update: Exclude `_smoke` from Aggregation

- `scripts/aggregate_results.py`의 target root 탐색에서 경로에 `_smoke`가 포함된 디렉토리를 제외하도록 했다.
- 집계 CSV/LaTeX/PDF 결과에서 smoke 실험이 반영되지 않는다.

## 2026-04-27 Update: Show SegAux Weight in `Dec` Column

- `scripts/aggregate_results.py`에서 실험명 suffix(`_segaux`, `_segaux_w...`)를 파싱해 `seg_aux_weight`를 메타로 보존한다.
- 요약 테이블의 `Dec` 컬럼에 segaux 실험은 텍스트 대신 숫자만 표시한다.
  - `_segaux` -> `0.3` (default)
  - `_segaux_w0.1` -> `0.1`
- 규칙 일관성을 위해 `decoder_guided + segaux` 조합도 `Fusion=none`, `Dec=<숫자>`로 통일했다.
- 후속으로 `decoder_guided` 문자열을 Dec 표시에서 생략했다.
  - base: `A`
  - segaux: `A-segaux`
  - weighted segaux: `A-w0.1` (LaTeX 렌더링 상 `A--w0.1`)
  - fusion 예시 config 추가:
    - `scripts/configs/drive_fusion_gate_multi_train.yaml`
    - `scripts/configs/drive_fusion_conv_residual_multi_train.yaml`
    - `scripts/configs/drive_fusion_scaled_sum_multi_train.yaml`
- `scripts/gpu_train_process_summary.sh`에서도 `--use_seg_aux/--seg_aux_weight` 및 config의 `use_seg_aux/seg_aux_weight`를 반영해 실험명 suffix가 누락되지 않도록 수정했다.

## 2026-04-27 Update: CREMI/DRIVE Missing-Combo Configs

- CREMI/DRIVE 요약표(`summary_experiment_means_datasets.tex`) 기준으로 누락된 조합을 실행하기 위한 통합 YAML을 추가했다(기존 설정/베이스라인 변경 없음).
  - `scripts/configs/cremi_drive_missing_combos.yaml`
- 초기에는 목적별 YAML 6개로 분리되어 있었지만, 유지보수를 위해 통합 YAML로 합치고 개별 파일은 제거했다.

## 2026-04-29 Update: CREMI/DRIVE Summary Combo Audit

- 현재 summary dataset CSV 기준 확인 결과:
  - 이전 missing-combo config가 겨냥했던 `DRIVE` baseline `conn=24 + gjml_sf_l1` 두 조합은 이제 집계에 포함되어 있다.
    - `dist + 24 + gjml_sf_l1`
    - `dist_inverted + 24 + gjml_sf_l1`
- 현재 `CREMI`/`DRIVE` 간 summary 조합 parity 관점에서 보이는 차이:
  - `CREMI`에는 explicit residual-scale suffix(`rs0.1/0.2/0.3/0.5`)가 붙은 `scaled_sum/A` 조합이 없다.
  - `DRIVE`에는 plain baseline `dist + conn=8 + gjml_sf_l1` 조합이 없다.
- 주의:
  - `CREMI`의 legacy `scaled_sum/A` 실험(`rs` 미표기)은 raw 산출물이 존재하지만,
    `scripts/aggregate_results.py`에서 ambiguous run으로 제외된다.
  - 따라서 summary 기준 누락과 raw 디렉터리 존재 여부는 다를 수 있다.

## 2026-04-29 Update: `cremi_drive_missing_combos.yaml` 갱신

- `scripts/configs/cremi_drive_missing_combos.yaml`를 현재 summary parity 기준 누락 조합에 맞게 갱신했다.
- 포함 조합:
  - `CREMI`: explicit `scaled_sum/A + rs{0.1,0.2,0.3,0.5}` grid
    - `binary` 4개
    - `dist` (`smooth_l1`, `gjml_sf_l1`) 8개
    - `dist_inverted` (`smooth_l1`, `gjml_sf_l1`) 8개
  - `DRIVE`: baseline `dist + 8 + gjml_sf_l1` 1개
- config policy:
  - top-level `dataset: [cremi, drive]`
  - `multi.experiments_only: true`
  - 각 experiment에서 `datasets: [...]`를 명시해 교차 확장을 방지
  - `multi.skip_completed: true`

## 2026-04-29 Update: LaTeX Display Cleanup for `decoder_guided`

- `scripts/aggregate_results.py`의 LaTeX 출력에서 `decoder_guided`는 profile `A`를 표기하지 않도록 정리했다.
- 적용 범위:
  - cross-experiment / dataset summary table
  - ablation table
  - best-run drop-one delta table
- 표시 정책:
  - summary 계열:
    - `Fusion`: `decoder_guided/A` -> `decoder_guided`
    - `Dec`: base `none`, segaux `segaux`, weighted segaux `segaux_w0.1`
  - ablation의 `Fusion Spec` 컬럼:
    - `decoder_guided + A`는 빈 칸으로 렌더링
- 메타데이터/정렬/CSV는 그대로 유지하고, LaTeX 렌더링만 변경했다.

## 2026-04-29 Update: LaTeX Display Cleanup for `scaled_sum/A`

- `scripts/aggregate_results.py`의 LaTeX 출력에서 `scaled_sum`도 profile `A`를 표기하지 않도록 정리했다.
- 표시 정책:
  - summary 계열 `Fusion` 컬럼:
    - `scaled_sum/A` -> `scaled_sum`
    - `scaled_sum/A/rs0.1` -> `scaled_sum/rs0.1`
  - ablation의 `Fusion Spec` 컬럼:
    - `scaled_sum + A`는 빈 칸
    - `scaled_sum + A + rs0.1`은 `rs0.1`
- 메타데이터/정렬/CSV는 그대로 유지하고, LaTeX 렌더링만 변경했다.

## 2026-04-29 Update: Drop-one Delta Table Metric Expansion

- `scripts/aggregate_results.py`의 `Best-run drop-one deltas` 테이블에 metric 컬럼을 확장했다.
- 기존:
  - `Dice`, `IoU`
- 변경:
  - `Dice`, `IoU`, `clDice`, `Err (β0)`, `Err (β1)`
- delta 해석:
  - `Dice/IoU/clDice`는 높을수록 좋으므로 `+`를 improvement로 표시
  - `Betti error`는 낮을수록 좋으므로 `-`를 improvement로 표시
- `BEST` row는 절대값, 나머지 row는 best 대비 delta를 계속 표시한다.

## 2026-04-29 Update: `SegAux` Column Label + Drop-one Variant Order

- LaTeX summary / dataset / drop-one table의 `Dec` 컬럼명을 `SegAux`로 변경했다.
- `decoder_guided`는 `Fusion` 컬럼에 유지하고, `SegAux` 값만 따로 표시한다.
  - base: `none`
  - default segaux: `segaux`
  - weighted segaux: `w0.1`
- `Best-run drop-one deltas`의 variant 순서를 조정했다.
  - 기존: `-Label Mode`, `-Loss`, `-Fusion`, `-Dec`
  - 변경: `-Label Mode`, `-Loss`, `-SegAux`, `-Fusion`

## 2026-04-29 Update: Drop-one Other-dataset Table Split

- `scripts/aggregate_results.py`의 `Best-run drop-one deltas`는 이제 `Other datasets`를 합쳐서 한 table로 내지 않는다.
- `CREMI`, `DRIVE`와 동일하게 `isic`, `chase` 등도 dataset별 개별 table로 분리한다.
- 캡션 예시:
  - `Best-run drop-one deltas (isic)`
  - `Best-run drop-one deltas (chase)`

## 2026-04-29 Update: Leftmost Experiment Numbering

- `scripts/aggregate_results.py`의 비교용 LaTeX 테이블들에 맨 왼쪽 `No.` 열을 추가했다.
- 적용 범위:
  - cross-experiment mean summary
  - dataset mean summary
  - ablation option tables
  - best-run drop-one delta tables
- 번호는 각 table 내부에서 `1`부터 다시 시작한다.
- fold별 raw summary table은 그대로 유지한다.

## 2026-04-25 Follow-up: Review-driven Refinements

- `OUTER_8_TO_STANDARD8_INDEX`를 하드코딩 대신 order 리스트에서 자동 유도하도록 변경
  - `model/DconnNet.py`에서 native/standard order 불일치 시 즉시 드러나도록 구성
- fusion profile loss term 반환 정리
  - profile A에서는 `inner_affinity`, `outer_affinity` 키를 반환하지 않음
  - 불필요한 zero tensor 로깅/집계 노이즈 감소
- `fusion_loss_profile` 검증 중복 제거
  - `solver.py`에서 초기화 시 중복 검증 제거, compose 함수 단일 검증 경로 유지
- `conn_num` 의미 명확화
  - `train.py`, `scripts/rebuild_dist_signed_artifacts.py`에 `conn_channels` 명시 필드 추가
  - CLI 입력 `conn_num`은 layout 선택 의미로 유지
- `connect_loss.py`의 `Bilateral_voting_kxk(offsets=None)` 경로에 레거시 호환 경로임을 주석으로 명시

## 2026-04-25 Update: Fusion Multi Config Consolidation

- 런처(`scripts/train_launcher_from_config.py`)에 `multi.fusion_matrix`를 추가했다.
  - 목적: `conn_fusion × profile × residual_scale` 전체 Cartesian product 대신, 의도한 조합만 명시적으로 실행
  - backward-compatible:
    - `fusion_matrix` 미사용 시 기존 `conn_fusions`/`fusion_loss_profiles`/`fusion_residual_scales` 동작 유지
- DRIVE fusion 멀티 실험 YAML 3개를 1개로 통합:
  - 추가: `scripts/configs/fusion_multi_train.yaml`
  - 삭제:
    - `scripts/configs/drive_fusion_gate_multi_train.yaml`
    - `scripts/configs/drive_fusion_conv_residual_multi_train.yaml`
    - `scripts/configs/drive_fusion_scaled_sum_multi_train.yaml`
- 통합 YAML의 실험 조합:
  - `gate`: profiles `A/B/C`
  - `conv_residual`: profiles `A/C`
  - `scaled_sum`: profile `A`, residual scale `0.1/0.2/0.3/0.5`

## 2026-04-27 Update: Dump CSV Path + Per-Experiment TEX Skip

- `scripts/aggregate_results.py`에서 fold별(실험별) summary는 CSV만 생성하도록 변경했다.
  - 기존: `dump/<stem>.csv`, `dump/<stem>.tex` (+ single-root 시 PDF)
  - 변경: `dump/csv/<stem>.csv`만 생성
- cross-experiment / dataset CSV도 `dump/csv/`로 통일했다.
- `output/summary` 루트에 dataset CSV를 복제 저장하던 동작은 제거했다.
- dataset/ablation 등 cross-experiment LaTeX/PDF 산출 경로는 기존처럼 유지한다.

## 2026-04-25 Update: GPU Summary Fusion Columns

- `scripts/gpu_train_process_summary.sh`의 프로세스 요약 테이블에 fusion 식별 컬럼을 추가했다.
  - 추가 컬럼: `CONN_FUSION`, `FUSION_LOSS_PROFILE`
- experiment name 역파싱(`parse_experiment_name_fields`)도 fusion 정보를 반환하도록 확장했다.
  - fusion 실험명(`binary_<fusion>_<profile>_...`, `dist_<fusion>_<profile>_...`)에서 `conn_fusion`, `fusion_loss_profile` 복원
  - baseline 실험명은 `conn_fusion=none`, `fusion_loss_profile=A`로 유지

## 2026-04-25 Update: Decoder Fusion 옵션

- New fork-specific CLI/config path:
  - `--decoder_fusion {none,concat,residual_gate}` (default `none`)
  - `--lambda_vote_aux` (default `0.2`)
- Compatibility policy:
  - `decoder_fusion=none`은 baseline 및 기존 `conn_fusion` 동작/실험명을 유지
  - decoder fusion은 `conn_fusion != none`, `num_class=1`, `conn_num=8`, `conn_layout=standard8`에서만 허용
  - 기존 encoder/connectivity fusion 수식(`C3`, `C5`, `C_fused`)은 수정하지 않음
- Model path:
  - `C_fused`를 만든 뒤 decoder feature 해상도로 정렬해 `DecoderFusion(D_final, C_fused)` 수행
  - `concat`: `Conv([D_final, C_fused])`
  - `residual_gate`: `D_final + sigmoid(Conv([D_final, C_fused])) * Conv(C_fused)`
  - `SegHead(D_fused)` 출력 `seg`를 main segmentation output으로 반환
- Loss/eval policy:
  - main segmentation: `BCEWithLogits(seg, mask) + Dice(sigmoid(seg), mask)`
  - `vote(C_fused) + dice(vote(C_fused))`는 `lambda_vote_aux`로 약하게 보조
  - `lambda_fused`, `lambda_inner`, `lambda_outer` affinity terms는 기존 profile A/B/C 정책 유지
  - validation/test metric은 decoder fusion 활성화 시 `seg` 기준으로 계산
- Config:
  - DRIVE decoder fusion 테스트 전용 config 추가: `scripts/configs/drive_decoder_fusion_multi_train.yaml`
  - 조합: `conn_fusion=gate`, `fusion_loss_profile=C`, `decoder_fusion=residual_gate`, label modes `binary/dist/dist_inverted`
- Reporting/monitoring:
  - `scripts/aggregate_results.py` experiment summary CSV/LaTeX 및 sample summary가 `conn_fusion`, `fusion_loss_profile`, `decoder_fusion`을 명시적으로 표시
  - ablation PDF에 `conn_fusion × fusion_loss_profile`만 비교하는 별도 `Fusion Objective` 테이블 추가
- `scripts/gpu_train_process_summary.sh` 프로세스 표에 `DECODER_FUSION` 컬럼 추가

## 2026-04-26 Update: LaTeX `#Folds` Column Removal

- `scripts/aggregate_results.py`의 LaTeX 실험 요약 표에서 `\#Folds` 컬럼을 제거했다.
- 적용 범위:
  - cross-experiment mean summary table
  - dataset별 mean summary table(표준편차 포함/미포함 출력 모두)
- 비적용 범위:
  - CSV 출력의 `num_folds` 컬럼은 유지(기존 분석 파이프라인 호환)
  - per-fold 상세 LaTeX 표(`write_latex`)는 기존 구조 유지

## 2026-04-26 Update: DGRF Fusion Loss KeyError Fix

- 증상:
  - `conn_fusion=decoder_guided` 학습 경로에서 `KeyError: 'fused'`가 발생
  - 원인: `solver.py::_compute_fusion_profile_loss`의 DGRF 분기에서 제거된 legacy key(`terms['fused']`, `terms['inner']`, `terms['outer']`)를 참조
- 수정 정책(compat additive):
  - 기존 fused profile objective(`total`)는 그대로 main term으로 유지
  - DGRF에서만 `C3/C5` auxiliary branch loss를 가중합으로 추가
  - gate regularization은 기존처럼 추가
  - 최종 반환 직전에 `terms['total']`을 최신 `total`과 동기화
- 구현 범위:
  - fork-specific DGRF 분기(`decoder_guided`)에만 적용
  - baseline 및 non-DGRF fusion compose 동작은 유지
- 모니터링 키:
  - `dgrf_fused_main`, `dgrf_c3_aux`, `dgrf_c5_aux`를 추가해 DGRF 구성항 추적 가능

## Known Constraints

- `.venv`에 `pytest`가 없을 수 있어 일부 테스트는 smoke 수준 검증으로 대체될 수 있다.
- third-party metric 유틸(clDice 등)에서 경고가 출력될 수 있다.

## Next Practical Steps

1. `tests/test_aggregate_results.py` 기대값 정리: 2026-04-29 변경(dataset PDF loss-split, ablation best-run 방식) 이후 stale한 기대값 약 7건을 현재 코드 기준으로 갱신한다.
2. `scripts/train_launcher_from_config.py`의 schema validation 테스트를 추가한다.
3. 문서 예시 커맨드를 주기적으로 정리해 현재 인터페이스와 맞춘다.
4. `PROJECT_CONTEXT/change_summary.md`의 `conn_fusion` 설명에서 `C5 -> C5_aligned` 세부 과정(현재 생략)을 ASCII 그림 또는 보충 단락으로 추가할지 검토한다.

## 2026-04-28 Update: `scaled_sum` Residual-Scale Output Path Separation

- 문제:
  - `conn_fusion=scaled_sum`에서 `fusion_residual_scale` sweep을 돌려도 실험명이 동일해 결과 폴더가 덮어써질 수 있었다.
- 변경:
  - `train.py:get_experiment_output_name`에 `scaled_sum` 전용 suffix를 추가했다.
    - 형식: `_rs<scale>` (예: `dist_scaled_sum_A_rs0.3_8_smooth_l1`)
  - `scripts/train_launcher_from_config.py:build_experiment_output_name`도 동일 규칙으로 맞췄다.
  - `resolve_pretrained_path`가 scale별 실험명을 사용하도록 `fusion_residual_scale` 전달 경로를 보강했다.
- 호환성:
  - `conn_fusion != scaled_sum` 실험명은 기존과 동일하다.
  - baseline(`conn_fusion=none`) 경로에는 영향이 없다.
- 신규 config:
  - `scripts/configs/drive_chase_scaled_sum_residual_ablation.yaml`
  - 목적: DRIVE+CHASE에서 `scaled_sum(A) × residual_scale(0.1/0.2/0.3/0.5)`를 label mode/aux loss 조합으로 실행.

## 2026-04-29 Update: Aggregate Report에 `scaled_sum` Residual Spec 표시

- 문제:
  - `summary_experiment_means_datasets.tex`에서 `scaled_sum/A` 행이 residual scale별로 구분되지 않아 누락/중복처럼 보였다.
- 변경 (`scripts/aggregate_results.py`):
  - 실험명 suffix `rs<scale>`를 `parse_experiment_metadata`에서 파싱해 `fusion_residual_scale` 메타로 보존.
  - 테이블 `Fusion` 라벨에서 `scaled_sum`일 때 `scaled_sum/A/rs0.x` 형식으로 표시.
  - experiment mean CSV에도 `fusion_residual_scale` 컬럼 추가.
- 효과:
  - dataset 요약 LaTeX/PDF에서 `scaled_sum` residual-scale 실험 spec이 명시적으로 드러난다.

## 2026-04-29 Update: `decoder_guided` 표기 명확화/통일

- 문제:
  - dataset 요약표에서 일부 행이 `Fusion=none`, `Dec=A`처럼 보이며(`dist_inverted` 중심) 의미 해석이 어렵고 모드 간 표기 규칙이 불일치해 보였다.
- 변경 (`scripts/aggregate_results.py`):
  - `decoder_guided` 전용 예외 표기를 제거했다.
  - `Fusion` 컬럼에 항상 `decoder_guided/<profile>`를 표시하도록 통일했다.

## 2026-04-29 Update: DGRF 실험 스크립트 정리

- `dg_residual` alias 제거 이후 혼동 방지를 위해 아래 실험 설정 파일을 제거했다.
  - `scripts/configs/chasedb1_fusion_ablations.yaml`
  - `scripts/configs/cremi_dgrf_ablations.yaml`
  - `scripts/configs/drive_dgrf_ablations.yaml`
  - `Dec` 컬럼은 일관되게 `none` 또는 SegAux weight(`0.1/0.2/0.3/0.5`)만 표시한다.
- 효과:
  - `dist`, `dist_inverted` 모두 동일한 표기 규칙을 사용하며, 실험 spec이 더 직접적으로 드러난다.

## 2026-04-29 Update: `cremi_dgrf_ablations.yaml` 재도입 (실행 복구)

- 상황:
  - 실행 중 런처가 `scripts/configs/cremi_dgrf_ablations.yaml`를 참조하는데 파일이 제거되어 스케줄 재실행/재개가 실패했다.
- 조치:
  - `scripts/configs/cremi_dgrf_ablations.yaml`를 `conn_fusion: decoder_guided` 기준으로 재생성했다(legacy `decoder_guided_residual` 미사용).
  - `multi.experiments_only: true` + 명시 조합(총 11 run)으로 DGRF ablation 스케줄을 복구했다.
  - `uv run bash scripts/train_launcher.sh --config scripts/configs/cremi_dgrf_ablations.yaml --dry_run` 통과를 확인했다.

## 2026-04-29 Update: Ablation LaTeX 옵션 반영 + 버킷 배치 고정

- 변경 (`scripts/aggregate_results.py` / `write_ablation_latex`):
  - `Fusion Objective` 테이블 헤더를 `Conn. Fusion + Fusion Spec`으로 확장
  - `scaled_sum`의 residual-scale을 `A/rs0.x` 형태로 표시
  - `Decoder / SegAux` 테이블은 `segaux`, `segaux_w0.1/0.2/0.5` 등 확장 옵션을 직접 표시
- 배치:
  - ablation 테이블 버킷 순서를 `CREMI -> DRIVE -> Other datasets`로 유지/고정

## 2026-04-29 Update: Decoder-Fusion Config 충돌 방지

- 문제:
  - `scripts/configs/decoder_fusion_multi_train.yaml`이 `conn_fusion: none`으로 설정되어 baseline 실험명(`binary_8_bce` 등)과 충돌 가능.
- 변경:
  - `conn_fusion: gate`
  - `fusion_loss_profiles: [C]`
  - `decoder_fusions: [residual_gate]`
  - `output_dir: output_decoder_fusion`
- 결과:
  - decoder-fusion 실험이 `*_gate_C_dec_residual_gate_*` 이름으로 생성되어 baseline과 분리됨.

## 2026-04-29 Update: `gpu_train_process_summary.sh` empty config-key lookup guard

- 문제:
  - 일부 GPU row에서 `ROW_CONFIG_PATH`가 비어 있을 때 associative array 조회가 빈 key로 실행되어
    `CONFIG_CURRENT_PLUS_REMAINING_COUNT: bad array subscript` 오류가 발생했다.
- 변경 (`scripts/gpu_train_process_summary.sh`):
  - `CONFIG_CURRENT_PLUS_REMAINING_COUNT[...]` 조회 전에 `row_config_path` non-empty 조건을 추가했다.
  - key가 비어 있으면 `remaining_count`를 빈 값으로 유지하고 ETA 총합 계산을 건너뛴다.
- 영향:
  - baseline/extension 학습 로직에는 영향 없고, 모니터링 스크립트의 안정성만 개선된다.

## 2026-04-29 Update: Dataset PDF Loss-wise Table Split

- 변경 (`scripts/aggregate_results.py` / `write_experiment_mean_dataset_tables_latex`):
  - `datasets.pdf`(= `*_experiment_means_datasets.pdf`) 생성 시 dataset 내부 row를 `loss` 기준으로 분할해, loss마다 별도 테이블을 출력하도록 변경했다.
  - 캡션 형식: `Cross-experiment mean summary (<dataset>, Loss=<loss>)`.
- 정렬/일관성:
  - loss 출력 순서는 기존 표 정렬과 동일한 우선순위(`bce -> smooth_l1 -> cl_dice -> gjml_sf_l1 -> gjml_sj_l1 -> unknown`)를 재사용한다.
- 영향 범위:
  - baseline 학습/평가 경로에는 영향 없고, fork 집계 PDF 표현 방식만 변경된다.

## 2026-04-29 Update: Ablations PDF Mean 제거 + Category별 Best-run 표시

- 변경 (`scripts/aggregate_results.py` / `write_ablation_latex`):
  - 기존 category별 marginal mean/std 집계를 제거했다.
  - 각 dataset × category value 조합에서 run들을 모아 `best_dice` 우선(동점 시 `best_jac`)으로 best-run 1개를 선택해 표시한다.
  - 테이블 metric 셀은 mean±std(`shortstack`) 대신 단일 값(Dice/IoU)으로 출력한다.
- 표 제목:
  - `Ablation Studies (Marginal Averages)` -> `Ablation Studies (Category-grouped Best Runs)`
- 영향 범위:
  - `ablations.pdf` 렌더링 정책만 변경, 학습/평가 및 dataset summary 경로에는 영향 없음.

## 2026-04-29 Update: Exclude ambiguous `scaled_sum/A` rows without `rs` in aggregation

- 문제:
  - 과거 실험명(`...scaled_sum_A...`)에는 `_rs<scale>` suffix가 없어 `fusion_residual_scale`를 역추적할 수 없는 행이 남아 있었다.
  - 특히 `decoder_fusion=none` 경로에서 `scaled_sum/A` 행이 `rs` 미지정(`nan`)으로 집계되어 spec 해석이 모호해졌다.
- 변경 (`scripts/aggregate_results.py`):
  - `should_exclude_ambiguous_scaled_sum_row` 정책 함수를 추가했다.
  - 조건(`conn_fusion=scaled_sum`, `fusion_loss_profile=A`, `decoder_fusion=none`, `fusion_residual_scale is None`)을 만족하는 run은 aggregate output에서 제외한다.
- 영향:
  - baseline/학습 로직에는 영향 없고, 집계 리포트에서 residual-scale 미명시 행만 제거된다.

## 2026-04-29 Update: Ablation PDF에 Best-run Drop-one Delta 표 추가

- 변경 (`scripts/aggregate_results.py` / `write_ablation_latex`):
  - 각 dataset의 best run(Dice 우선, IoU tie-break) 기준으로,
    `label_mode`, `loss`, `fusion`, `dec` 중 한 축만 변경한 대안(best alternative)을 찾아
    `ΔDice`, `ΔIoU`를 표시하는 표를 추가했다.
  - 캡션: `Best-run drop-one deltas (<bucket>)`
- 표시 규칙:
  - 대안이 없으면 값/델타를 `-`로 표시
  - 델타는 signed(`+/-`) 4자리 소수로 표시

## 2026-04-29 Update: Drop-one `BEST/-Loss` loss pair 고정

- 변경 (`scripts/aggregate_results.py` / `write_ablation_latex`):
  - `Best-run drop-one deltas`의 `BEST` 행은 전체 loss 후보 중 최고값이 아니라 `gjml_sf_l1`, `smooth_l1` 두 loss 안에서만 선택한다.
  - `-Loss` 행은 "다른 loss 중 최고"가 아니라 `BEST`에서 채택되지 않은 반대쪽 loss만 비교 대상으로 사용한다.
- 목적:
  - ablation 해석을 `smooth_l1` vs `gjml_sf_l1`의 직접 비교로 고정하고, `bce` 등 다른 loss가 `BEST/-Loss` 의미를 흐리지 않도록 한다.

## 2026-04-29 Update: Drop-one `-Fusion` plain baseline 규칙 + `trash` dataset 제외

- 변경 (`scripts/aggregate_results.py`):
  - `Best-run drop-one deltas`의 `-Fusion` 행은 best와 같은 `label_mode`, `conn_num`, `conn_layout`, `loss`를 유지한 채 `Fusion=none`, `Dec=none`인 plain baseline으로 비교한다.
  - 같은 drop-one 대안 선택에서 `conn_num/conn_layout`은 공통으로 고정해, `-Loss`/`-Fusion` 등이 다른 connectivity 설정으로 이동하지 않도록 했다.
  - dataset-level 리포트에서 `trash`는 dataset으로 취급하지 않도록 제외했다.
    - 제외 대상: per-dataset CSV, `datasets.pdf`, `ablation.pdf`, cross-experiment LaTeX의 dataset grouping
- 목적:
  - `-Fusion`이 decoder-guided/segaux best에서도 실제 no-fusion baseline을 가리키도록 만들고,
    `trash` 실험 보관용 디렉터리가 dataset 통계에 섞이지 않도록 한다.

## 2026-04-29 Update: `summary_ablation.pdf` drop-one default-axis 처리

- 변경 (`scripts/aggregate_results.py` / `write_ablation_latex`):
  - `Best-run drop-one deltas`에서 best가 이미 기본값을 쓰는 축은 드롭 불가로 처리한다.
  - 적용 규칙:
    - `Label Mode=binary`면 `-Label Mode`는 `-`
    - `Fusion=none`면 `-Fusion`은 `-`
    - `Dec=none`면 `-Dec`는 `-`
- 목적:
  - `best`가 이미 plain baseline인데 `-Fusion`이 자기 자신(`+0.0000`)으로 다시 선택되는 비의미적 표기를 제거한다.

## 2026-04-29 Update: Drop-one 표 마지막 행을 고정 plain baseline으로 표시

- 변경 (`scripts/aggregate_results.py` / `write_ablation_latex`):
  - `Best-run drop-one deltas` 각 dataset table의 마지막 행에 고정 baseline row를 추가했다.
  - 기준 run은 `binary`, `conn_num=8`, `conn_layout=default`, `Fusion=none`, `Dec=none`, `Loss=bce` 조합이다.
- 표시:
  - variant 라벨은 `BASE`
  - Dice/IoU는 `BEST` 절대값이 아니라 `BEST` 대비 delta로 계속 표시한다.
- 목적:
  - drop-one 대안들과 별개로, 가장 기본 baseline과의 차이를 각 dataset 표 맨 아래에서 바로 비교할 수 있게 한다.

## 2026-04-29 Update: `binary + decoder_guided` 전용 launcher config 추가

- 배경:
  - 기존 DGRF ablation config(`scripts/configs/cremi_dgrf_ablations.yaml`)는 `dist/dist_inverted` 조합만 명시하고 있었다.
  - `binary + decoder_guided`는 코드상 가능하지만, baseline 보존 원칙 때문에 기존 ablation 스케줄에는 포함되지 않았다.
- 변경:
  - `scripts/configs/drive_cremi_dgrf_binary.yaml` 추가
  - scope: `dataset=[drive, cremi]`, `label_modes=[binary]`, `conn_fusion=decoder_guided`, `fusion_loss_profile=A`
- 의도:
  - 기존 DGRF ablation 파일 의미를 바꾸지 않고, `binary` DGRF를 별도 스케줄로 명시적으로 실행 가능하게 유지한다.

## 2026-04-29 Update: DRIVE `binary + decoder_guided + segaux_w0.5` 전용 config 추가

- 추가:
  - `scripts/configs/drive_dgrf_binary_segaux_w0.5.yaml`
- 스펙:
  - dataset: `drive`
  - `conn_num=8`, `conn_layout=standard8`
  - `label_mode=binary`
  - `conn_fusion=decoder_guided`, `fusion_loss_profile=A`
  - `use_seg_aux=true`, `seg_aux_weight=0.5`
- 의도:
  - DRIVE에서 report 기준 `Fusion=decoder_guided/A`, `Dec=segaux_w0.5`에 해당하는 단일 run을 재현 가능한 별도 launcher config로 분리한다.

## 2026-04-29 Update: CREMI `binary + decoder_guided/A` 전용 config 추가

- 추가:
  - `scripts/configs/cremi_dgrf_binary.yaml`
- 스펙:
  - dataset: `cremi`
  - `conn_num=8`, `conn_layout=standard8`
  - `label_mode=binary`
  - `conn_fusion=decoder_guided`, `fusion_loss_profile=A`
  - `use_seg_aux` 미사용
- 의도:
  - `drive_cremi_dgrf_binary.yaml`의 `cremi` 단일 실행판으로, CREMI base DGRF binary run을 별도 지정 가능하게 유지한다.

## 2026-04-29 Update: CREMI `dist + decoder_guided/A + smooth_l1` 전용 config 추가

- 추가:
  - `scripts/configs/cremi_dgrf_dist_smooth_l1.yaml`
- 스펙:
  - dataset: `cremi`
  - `conn_num=8`, `conn_layout=standard8`
  - `label_mode=dist`
  - `dist_aux_loss=smooth_l1`
  - `conn_fusion=decoder_guided`, `fusion_loss_profile=A`
  - `use_seg_aux` 미사용
- 의도:
  - `summary_ablation.pdf`에서 누락 확인된 CREMI `dist / decoder_guided / smooth_l1` 조합을 별도 재현 가능한 launcher config로 분리한다.

## 2026-04-29 Update: 다른 single-class dataset에도 동일 DGRF+SegAux 규칙 확장

- 추가:
  - `scripts/configs/other_datasets_dgrf_binary_segaux_w0.5.yaml`
- 포함 대상:
  - `chase`
  - `cremi`
  - `isic2018`
  - `octa500-3M`
  - `octa500-6M`
- 공통 스펙:
  - `conn_num=8`, `conn_layout=standard8`
  - `label_mode=binary`
  - `conn_fusion=decoder_guided`, `fusion_loss_profile=A`
  - `use_seg_aux=true`, `seg_aux_weight=0.5`
- 제외:
  - `retouch`는 현재 multi-class 경로라 `conn_fusion` 적용 대상이 아니므로 포함하지 않았다.

## 2026-04-29 Update: Dataset Summary PDF에서 `CHASE`/`ISIC` 섹션 분리

- `scripts/aggregate_results.py`의 dataset summary grouping 순서를 다음처럼 조정했다.
  - `CREMI`
  - `DRIVE`
  - `CHASE`
  - `ISIC`
  - `OCTA3M&6M`
  - `Other datasets`
- 목적:
  - `summary_experiment_means_datasets.pdf`에서 `chase`와 `isic`가 `Other datasets` 아래에 섞이지 않고 독립 제목으로 보이게 한다.
- 영향 범위:
  - dataset summary LaTeX/PDF 섹션 제목
  - 동일 grouping 함수를 재사용하는 dataset CSV/정렬 순서

## 2026-04-29 Update: Aggregate Default Scope를 `DRIVE`/`CHASE`/`OCTA`로 제한

- 변경 (`scripts/aggregate_results.py`):
  - combined aggregate 출력의 기본 dataset scope를 `drive`, `chase(chasedb1 포함)`, `octa500-3M/6M` 계열로 제한했다.
  - `--all` 플래그를 추가해, 이 옵션을 줄 때만 모든 counted dataset을 종합한다.
  - default mode 출력 stem은 기존대로 유지하고, `--all` mode는 `_all` suffix를 붙여 기본 집계 산출물과 충돌하지 않게 했다.
  - dataset summary / ablation LaTeX에는 dataset 또는 table 경계마다 `\clearpage`를 넣어 서로 다른 dataset이 같은 PDF 페이지를 공유하지 않도록 했다.
- 영향 범위:
  - cross-experiment aggregate CSV/LaTeX
  - dataset summary CSV/LaTeX/PDF
  - ablation LaTeX/PDF
- 비영향 범위:
  - per-root fold summary CSV
  - `_smoke` / `trash` 제외 정책
