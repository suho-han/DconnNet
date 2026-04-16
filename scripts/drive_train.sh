#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

send_telegram_alert_on_exit() {
  EXIT_CODE="$?"

  if [ "${EXIT_CODE}" -eq 0 ]; then
    ALERT_STATUS="DONE"
  else
    ALERT_STATUS="FAILED"
  fi

  if [ -x "${PYTHON_BIN}" ] && [ -f "${REPO_ROOT}/scripts/telegram_alert.py" ]; then
    ALERT_JOB="drive_train(conn=${CONN_NUM:-unknown},label=${LABEL_MODE:-unknown},epochs=${EPOCHS:-${EPOCHS_OVERRIDE:-unset}})"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/telegram_alert.py" \
      --job "${ALERT_JOB}" \
      --status "${ALERT_STATUS}" || true
  fi
}

trap 'send_telegram_alert_on_exit' EXIT

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Missing python: ${PYTHON_BIN}" >&2
  exit 127
fi

CONN_NUM="8"
LABEL_MODE="binary"
DIST_AUX_LOSS="smooth_l1"
DIST_SF_L1_GAMMA="1.0"
EPOCHS_OVERRIDE=""
DEVICE_OVERRIDE=""

print_help() {
  cat <<EOF
Usage: sh scripts/drive_train.sh [options] [extra train.py args]

Unified DRIVE training launcher.
Put primary options before extra train.py args.

Primary options:
  --device <int>                        (optional GPU index; passed to train.py --device)
  --conn_num <int>                      (default: 8; supported: 8, 24)
  --label_mode <binary|dist|dist_inverted> (default: binary)
  --dist_aux_loss <smooth_l1|gjml_sf_l1>          (default: smooth_l1)
  --dist_sf_l1_gamma <float>            (default: 1.0)
  --epochs <int>                        (optional override)

Output layout:
  binary 5-fold run  -> output/5folds/binary_<conn_num>_bce/
  binary hold-out    -> output/1fold/binary_<conn_num>_bce/
  dist 5-fold run    -> output/5folds/<label_mode>_<conn_num>_<dist_aux_loss>/
  dist hold-out      -> output/1fold/<label_mode>_<conn_num>_<dist_aux_loss>/

Notes:
  --dist_aux_loss is only used by dist/dist_inverted.
  binary mode keeps the upstream BCE/Dice-style path and is named with _bce.

Fold selection:
  pass --folds through to train.py as an extra arg
  example: sh scripts/drive_train.sh --folds 1

Default epochs policy (when --epochs is not set):
  label_mode=binary, conn_num=8   -> 130
  label_mode=binary, conn_num=24  -> 390
  label_mode=dist_*               -> 260

Examples:
  bash scripts/drive_train.sh
  bash scripts/drive_train.sh --conn_num 24
  bash scripts/drive_train.sh --label_mode dist --dist_aux_loss gjml_sf_l1 --dist_sf_l1_gamma 1.0
  bash scripts/drive_train.sh --label_mode dist_inverted --epochs 30 --folds 1
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help|-help)
      print_help
      exit 0
      ;;
    --conn_num)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --conn_num" >&2
        exit 2
      fi
      CONN_NUM="$2"
      shift 2
      ;;
    --conn_num=*)
      CONN_NUM="${1#*=}"
      shift
      ;;
    --device)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --device" >&2
        exit 2
      fi
      DEVICE_OVERRIDE="$2"
      shift 2
      ;;
    --device=*)
      DEVICE_OVERRIDE="${1#*=}"
      shift
      ;;
    --label_mode)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --label_mode" >&2
        exit 2
      fi
      LABEL_MODE="$2"
      shift 2
      ;;
    --label_mode=*)
      LABEL_MODE="${1#*=}"
      shift
      ;;
    --dist_aux_loss)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --dist_aux_loss" >&2
        exit 2
      fi
      DIST_AUX_LOSS="$2"
      shift 2
      ;;
    --dist_aux_loss=*)
      DIST_AUX_LOSS="${1#*=}"
      shift
      ;;
    --dist_sf_l1_gamma)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --dist_sf_l1_gamma" >&2
        exit 2
      fi
      DIST_SF_L1_GAMMA="$2"
      shift 2
      ;;
    --dist_sf_l1_gamma=*)
      DIST_SF_L1_GAMMA="${1#*=}"
      shift
      ;;
    --epochs)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --epochs" >&2
        exit 2
      fi
      EPOCHS_OVERRIDE="$2"
      shift 2
      ;;
    --epochs=*)
      EPOCHS_OVERRIDE="${1#*=}"
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [ "${CONN_NUM}" != "8" ] && [ "${CONN_NUM}" != "24" ]; then
  echo "Unsupported --conn_num: ${CONN_NUM} (supported: 8, 24)" >&2
  exit 2
fi

if [ -n "${DEVICE_OVERRIDE}" ]; then
  case "${DEVICE_OVERRIDE}" in
    ''|*[!0-9]*)
      echo "Unsupported --device: ${DEVICE_OVERRIDE} (expected non-negative integer)" >&2
      exit 2
      ;;
  esac
fi

if [ -n "${EPOCHS_OVERRIDE}" ]; then
  EPOCHS="${EPOCHS_OVERRIDE}"
elif [ "${LABEL_MODE}" = "dist" ] || [ "${LABEL_MODE}" = "dist_inverted" ]; then
  EPOCHS="260"
elif [ "${CONN_NUM}" = "24" ]; then
  EPOCHS="390"
else
  EPOCHS="130"
fi

cd "${REPO_ROOT}"

DEVICE_ARGS=""
if [ -n "${DEVICE_OVERRIDE}" ]; then
  DEVICE_ARGS="--device ${DEVICE_OVERRIDE}"
fi

"${PYTHON_BIN}" train.py \
  --dataset 'drive' \
  --data_root 'data/DRIVE' \
  --resize 960 960 \
  --num-class 1 \
  --batch-size 4 \
  --epochs "${EPOCHS}" \
  --lr 0.0038 \
  --lr-update 'poly' \
  --folds 1 \
  --conn_num "${CONN_NUM}" \
  --label_mode "${LABEL_MODE}" \
  --dist_aux_loss "${DIST_AUX_LOSS}" \
  --dist_sf_l1_gamma "${DIST_SF_L1_GAMMA}" \
  ${DEVICE_ARGS} \
  "$@"
