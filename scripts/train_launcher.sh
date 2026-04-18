#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
HELPER_SCRIPT="${SCRIPT_DIR}/train_launcher_from_config.py"

if [[ ! -f "${HELPER_SCRIPT}" ]]; then
  echo "Missing launcher helper: ${HELPER_SCRIPT}" >&2
  exit 127
fi

CONFIG_PATH=""
PREV=""
for ARG in "$@"; do
  if [[ "${PREV}" == "--config" ]]; then
    CONFIG_PATH="${ARG}"
    break
  fi
  if [[ "${ARG}" == --config=* ]]; then
    CONFIG_PATH="${ARG#--config=}"
    break
  fi
  PREV="${ARG}"
done

if [[ -n "${CONFIG_PATH}" && ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 2
fi

exec "${PYTHON_BIN}" "${HELPER_SCRIPT}" "$@"
