# Testing Notes (Condensed)

Last updated: 2026-04-24

## Scope

- 이 문서는 현재 유효한 핵심 검증 결과만 기록합니다.
- 상세 이력 로그는 Git 기록을 기준으로 합니다.

## 2026-04-24: 24to8 제거 검증

### A. 정적 검증

- `.venv/bin/python -m py_compile train.py model/DconnNet.py scripts/train_launcher_from_config.py scripts/aggregate_results.py` 통과
- `bash -n scripts/train_launcher.sh scripts/gpu_train_process_summary.sh` 통과

### B. 런처 동작 검증

- dry-run 생성 커맨드에서 `--direction_grouping`, `--direction_fusion` 인자가 더 이상 출력되지 않음
- YAML에 제거된 키(`direction_grouping`, `direction_fusion`)를 넣으면 명시적 에러가 발생하도록 검증

### C. 집계/모니터링 검증

- `scripts/aggregate_results.py` 출력에서 direction 관련 컬럼(`direction_grouping`, `direction_fusion`)이 제거됨
- `scripts/gpu_train_process_summary.sh` 출력 컬럼에서 direction 항목이 제거됨

### D. 산출물 정리

- `output/` 하위 `*24to8*`, `*coarse24to8*` 디렉터리 제거 완료
- 제거 후 남은 실험 경로만 대상으로 집계가 동작함

## Residual Risks

- 현재 워크트리에 기존 변경사항이 포함되어 있어, 후속 머지 시 충돌 가능성은 남아 있음
- `pytest` 기반 자동화 테스트가 없는 항목은 향후 회귀 테스트 보강이 필요함
