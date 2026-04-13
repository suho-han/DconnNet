#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Missing python: ${PYTHON_BIN}" >&2
  exit 127
fi

CONN_NUM="8"
LABEL_MODE="binary"
DIST_AUX_LOSS="smooth_l1"
DIST_SF_L1_GAMMA="1.0"
EPOCHS_OVERRIDE=""

print_help() {
  cat <<EOF
Usage: sh scripts/chasedb1_train.sh [options] [extra train.py args]

Unified CHASE-DB1 training launcher.
Put primary options before extra train.py args.

Primary options:
  --conn_num <int>                      (default: 8)
  --label_mode <binary|dist_signed|dist_inverted> (default: binary)
  --dist_aux_loss <smooth_l1|gjml_sf_l1>          (default: smooth_l1)
  --dist_sf_l1_gamma <float>            (default: 1.0)
  --epochs <int>                        (optional override)

Default epochs policy (when --epochs is not set):
  label_mode=binary, conn_num=8   -> 130
  label_mode=binary, conn_num=25  -> 390
  label_mode=dist_*               -> 260

Examples:
  sh scripts/chasedb1_train.sh
  sh scripts/chasedb1_train.sh --conn_num 25
  sh scripts/chasedb1_train.sh --label_mode dist_signed --dist_aux_loss gjml_sf_l1 --dist_sf_l1_gamma 1.0
  sh scripts/chasedb1_train.sh --label_mode dist_inverted --epochs 30 --folds 1
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

if [ -n "${EPOCHS_OVERRIDE}" ]; then
  EPOCHS="${EPOCHS_OVERRIDE}"
elif [ "${LABEL_MODE}" = "dist_signed" ] || [ "${LABEL_MODE}" = "dist_inverted" ]; then
  EPOCHS="260"
elif [ "${CONN_NUM}" = "25" ]; then
  EPOCHS="390"
else
  EPOCHS="130"
fi

cd "${REPO_ROOT}"

"${PYTHON_BIN}" train.py \
  --dataset 'chase' \
  --data_root 'data/chase' \
  --resize 960 960 \
  --num-class 1 \
  --batch-size 4 \
  --epochs "${EPOCHS}" \
  --lr 0.0038 \
  --lr-update 'poly' \
  --folds 5 \
  --conn_num "${CONN_NUM}" \
  --label_mode "${LABEL_MODE}" \
  --dist_aux_loss "${DIST_AUX_LOSS}" \
  --dist_sf_l1_gamma "${DIST_SF_L1_GAMMA}" \
  "$@"
