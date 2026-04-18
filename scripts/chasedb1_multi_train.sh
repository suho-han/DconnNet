#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/configs/chasedb1_multi_train.yaml"

echo "[DEPRECATED] scripts/chasedb1_multi_train.sh is deprecated and will be removed in the next release. Use scripts/train_launcher.sh --config \"${CONFIG_PATH}\" [--device N] [--dry_run]" >&2
exec bash "${SCRIPT_DIR}/train_launcher.sh" --config "${CONFIG_PATH}" "$@"
