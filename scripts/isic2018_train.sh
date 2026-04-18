#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/configs/isic2018_train.yaml"

echo "[DEPRECATED] scripts/isic2018_train.sh is deprecated and will be removed in the next release. Use scripts/train_launcher.sh --config \"${CONFIG_PATH}\" [--device N] [--dry_run]" >&2
exec bash "${SCRIPT_DIR}/train_launcher.sh" --config "${CONFIG_PATH}" "$@"
