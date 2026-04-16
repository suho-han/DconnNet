#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python: ${PYTHON_BIN}" >&2
  exit 127
fi

CONN_NUM="8"
LABEL_MODE="binary"
DIST_AUX_LOSS="smooth_l1"
DIST_SF_L1_GAMMA="1.0"
FOLDS="5"
TARGET_FOLD=""
CHECKPOINT_OUTPUT_DIR="output"
TEST_OUTPUT_DIR="output/test_only_best_model"

fold_scope() {
  local folds="$1"
  if [[ "${folds}" == "1" ]]; then
    echo "1fold"
  else
    echo "${folds}folds"
  fi
}

experiment_name() {
  local label_mode="$1"
  local conn_num="$2"
  local dist_aux_loss="$3"
  if [[ "${label_mode}" == "binary" ]]; then
    echo "binary_${conn_num}_bce"
  else
    echo "${label_mode}_${conn_num}_${dist_aux_loss}"
  fi
}

print_help() {
  cat <<EOF
Usage: bash scripts/chasedb1_test_best.sh [options] [extra train.py args]

CHASE-DB1 best-model test-only launcher.
Runs evaluation using only models/<fold>/best_model.pth checkpoints.

Primary options:
  --conn_num <int>                            (default: 8; supported: 8, 24)
  --label_mode <binary|dist|dist_inverted> (default: binary)
  --dist_aux_loss <smooth_l1|gjml_sf_l1>     (default: smooth_l1)
  --dist_sf_l1_gamma <float>                 (default: 1.0)
  --folds <int>                              (default: 5)
  --target_fold <int>                        (optional 1-based fold to test)
  --checkpoint_output_dir <path>             (default: output)
  --test_output_dir <path>                   (default: output/test_only_best_model)

Checkpoint lookup:
  <checkpoint_output_dir>/<fold_scope>/<experiment>/models/<fold>/best_model.pth

Test output layout:
  <test_output_dir>/<fold_scope>/<experiment>/

Notes:
  - Existing training outputs are not overwritten; test artifacts go to a separate root.
  - binary mode uses experiment name binary_<conn_num>_bce.
  - dist modes use <label_mode>_<conn_num>_<dist_aux_loss>.
  - Additional args after the primary options are passed through to train.py.

Examples:
  bash scripts/chasedb1_test_best.sh
  bash scripts/chasedb1_test_best.sh --folds 1
  bash scripts/chasedb1_test_best.sh --label_mode dist --dist_aux_loss gjml_sf_l1
  bash scripts/chasedb1_test_best.sh --target_fold 3 --checkpoint_output_dir output --test_output_dir output/test_only_best_model
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help|-help)
      print_help
      exit 0
      ;;
    --conn_num)
      CONN_NUM="$2"
      shift 2
      ;;
    --conn_num=*)
      CONN_NUM="${1#*=}"
      shift
      ;;
    --label_mode)
      LABEL_MODE="$2"
      shift 2
      ;;
    --label_mode=*)
      LABEL_MODE="${1#*=}"
      shift
      ;;
    --dist_aux_loss)
      DIST_AUX_LOSS="$2"
      shift 2
      ;;
    --dist_aux_loss=*)
      DIST_AUX_LOSS="${1#*=}"
      shift
      ;;
    --dist_sf_l1_gamma)
      DIST_SF_L1_GAMMA="$2"
      shift 2
      ;;
    --dist_sf_l1_gamma=*)
      DIST_SF_L1_GAMMA="${1#*=}"
      shift
      ;;
    --folds)
      FOLDS="$2"
      shift 2
      ;;
    --folds=*)
      FOLDS="${1#*=}"
      shift
      ;;
    --target_fold)
      TARGET_FOLD="$2"
      shift 2
      ;;
    --target_fold=*)
      TARGET_FOLD="${1#*=}"
      shift
      ;;
    --checkpoint_output_dir)
      CHECKPOINT_OUTPUT_DIR="$2"
      shift 2
      ;;
    --checkpoint_output_dir=*)
      CHECKPOINT_OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --test_output_dir)
      TEST_OUTPUT_DIR="$2"
      shift 2
      ;;
    --test_output_dir=*)
      TEST_OUTPUT_DIR="${1#*=}"
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [[ "${CONN_NUM}" != "8" && "${CONN_NUM}" != "24" ]]; then
  echo "Unsupported --conn_num: ${CONN_NUM} (supported: 8, 24)" >&2
  exit 2
fi

if [[ -n "${TARGET_FOLD}" ]]; then
  if (( TARGET_FOLD < 1 || TARGET_FOLD > FOLDS )); then
    echo "--target_fold must be within [1, --folds]" >&2
    exit 2
  fi
fi

FOLD_SCOPE="$(fold_scope "${FOLDS}")"
EXPERIMENT_NAME="$(experiment_name "${LABEL_MODE}" "${CONN_NUM}" "${DIST_AUX_LOSS}")"
CHECKPOINT_ROOT="${CHECKPOINT_OUTPUT_DIR%/}/${FOLD_SCOPE}/${EXPERIMENT_NAME}"
TEST_ROOT="${TEST_OUTPUT_DIR%/}"

cd "${REPO_ROOT}"

if [[ -n "${TARGET_FOLD}" ]]; then
  FOLD_LIST=("${TARGET_FOLD}")
else
  FOLD_LIST=()
  for ((fold = 1; fold <= FOLDS; fold++)); do
    FOLD_LIST+=("${fold}")
  done
fi

RUN_EXIT_CODE="0"
set +e
for fold in "${FOLD_LIST[@]}"; do
  CKPT_PATH="${CHECKPOINT_ROOT}/models/${fold}/best_model.pth"
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Missing checkpoint: ${CKPT_PATH}" >&2
    RUN_EXIT_CODE="1"
    break
  fi

  echo "[INFO] fold=${fold} checkpoint=${CKPT_PATH}"
  "${PYTHON_BIN}" train.py \
    --dataset chase \
    --data_root data/chase \
    --resize 960 960 \
    --num-class 1 \
    --batch-size 4 \
    --epochs 1 \
    --lr 0.0038 \
    --lr-update poly \
    --folds "${FOLDS}" \
    --target_fold "${fold}" \
    --conn_num "${CONN_NUM}" \
    --label_mode "${LABEL_MODE}" \
    --dist_aux_loss "${DIST_AUX_LOSS}" \
    --dist_sf_l1_gamma "${DIST_SF_L1_GAMMA}" \
    --pretrained "${CKPT_PATH}" \
    --output_dir "${TEST_ROOT}" \
    --test_only \
    "$@"
  RUN_EXIT_CODE="$?"
  if [[ "${RUN_EXIT_CODE}" != "0" ]]; then
    break
  fi
done
set -e

if [[ "${RUN_EXIT_CODE}" == "0" ]]; then
  ALERT_STATUS="DONE"
else
  ALERT_STATUS="FAILED"
fi

if [[ -f "${REPO_ROOT}/scripts/telegram_alert.py" ]]; then
  ALERT_JOB="chasedb1_test_best(conn=${CONN_NUM},label=${LABEL_MODE},folds=${FOLDS})"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/telegram_alert.py" \
    --job "${ALERT_JOB}" \
    --status "${ALERT_STATUS}" || true
fi

exit "${RUN_EXIT_CODE}"
