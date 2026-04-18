#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/gpu_train_process_summary.sh [--all]

Options:
  --all   Include GPU compute processes that are not train.py.
  -h, --help
EOF
}

trim() {
  local value="$*"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

normalize_output_dir() {
  local output_dir="$1"
  if [[ -z "${output_dir}" ]]; then
    printf 'output'
    return 0
  fi
  if [[ "${output_dir}" == "/" ]]; then
    printf '/'
    return 0
  fi
  output_dir="${output_dir%/}"
  if [[ -z "${output_dir}" ]]; then
    printf '.'
    return 0
  fi
  printf '%s' "${output_dir}"
}

build_experiment_name() {
  local label_mode="$1"
  local conn_num="$2"
  local dist_aux_loss="$3"
  local direction_grouping="$4"
  local direction_fusion="$5"
  local base_name

  if [[ "${label_mode}" == "binary" ]]; then
    base_name="binary_${conn_num}_bce"
  else
    base_name="${label_mode}_${conn_num}_${dist_aux_loss}"
  fi

  if [[ "${direction_grouping}" != "none" ]]; then
    base_name="${base_name}_${direction_grouping}_${direction_fusion}"
  fi

  printf '%s' "${base_name}"
}

format_duration_hms() {
  local total_seconds="$1"
  local days hours minutes seconds remainder
  if ((total_seconds < 0)); then
    total_seconds=0
  fi
  days=$((total_seconds / 86400))
  remainder=$((total_seconds % 86400))
  hours=$((remainder / 3600))
  remainder=$((remainder % 3600))
  minutes=$((remainder / 60))
  seconds=$((remainder % 60))
  if ((days > 0)); then
    printf '%dd %02d:%02d:%02d' "${days}" "${hours}" "${minutes}" "${seconds}"
  else
    printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
  fi
}

compute_eta_fields() {
  local csv_path="$1"
  local total_epochs_raw="$2"
  local parsed rc marker last_epoch elapsed_sec valid_rows total_epochs remaining_epochs eta_sec now_epoch finish_epoch eta_duration eta_finish eta_status progress

  if [[ ! -r "${csv_path}" ]]; then
    printf 'n/a\tn/a\tn/a\tno_csv'
    return 0
  fi

  if ! parsed="$(awk -F',' '
    function trim(s) {
      gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", s)
      return s
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        key = trim($i)
        idx[key] = i
      }
      if (!("epoch" in idx) || !("elapsed_hms" in idx)) {
        exit 10
      }
      epoch_idx = idx["epoch"]
      elapsed_idx = idx["elapsed_hms"]
      next
    }
    {
      epoch_raw = trim($(epoch_idx))
      elapsed_raw = trim($(elapsed_idx))
      if (epoch_raw == "" || elapsed_raw == "") {
        next
      }
      if (epoch_raw !~ /^[0-9]+$/) {
        next
      }
      count = split(elapsed_raw, t, ":")
      if (count != 3) {
        next
      }
      if (t[1] !~ /^[0-9]+$/ || t[2] !~ /^[0-9]+$/ || t[3] !~ /^[0-9]+$/) {
        next
      }
      if ((t[2] + 0) >= 60 || (t[3] + 0) >= 60) {
        next
      }

      epoch = epoch_raw + 0
      elapsed_sec = (t[1] * 3600) + (t[2] * 60) + t[3]
      valid_rows += 1

      if (epoch > best_epoch || (epoch == best_epoch && elapsed_sec > best_elapsed_sec)) {
        best_epoch = epoch
        best_elapsed_sec = elapsed_sec
      }
    }
    END {
      if (valid_rows == 0) {
        exit 11
      }
      printf "OK,%d,%d,%d\n", best_epoch, best_elapsed_sec, valid_rows
    }
  ' "${csv_path}" 2>/dev/null)"; then
    rc=$?
    case "${rc}" in
      10)
        printf 'n/a\tn/a\tn/a\tmissing_cols'
        ;;
      11)
        printf 'n/a\tn/a\tn/a\tno_valid_rows'
        ;;
      *)
        printf 'n/a\tn/a\tn/a\tparse_error'
        ;;
    esac
    return 0
  fi

  IFS=',' read -r marker last_epoch elapsed_sec valid_rows <<< "${parsed}"
  if [[ "${marker}" != "OK" ]] || [[ -z "${last_epoch}" ]] || [[ -z "${elapsed_sec}" ]]; then
    printf 'n/a\tn/a\tn/a\tparse_error'
    return 0
  fi
  if ((last_epoch <= 0)); then
    printf 'n/a\tn/a\tn/a\tinvalid_epoch'
    return 0
  fi

  total_epochs=0
  if [[ "${total_epochs_raw}" =~ ^[0-9]+$ ]] && ((total_epochs_raw > 0)); then
    total_epochs="${total_epochs_raw}"
  else
    total_epochs="${last_epoch}"
  fi

  remaining_epochs=$((total_epochs - last_epoch))
  if ((remaining_epochs > 0)); then
    eta_status="running"
  else
    eta_status="completed_or_overrun"
    remaining_epochs=0
  fi

  eta_sec="$(awk -v elapsed="${elapsed_sec}" -v epoch="${last_epoch}" -v remain="${remaining_epochs}" 'BEGIN { printf "%d", ((elapsed / epoch) * remain) + 0.5 }')"
  eta_duration="$(format_duration_hms "${eta_sec}")"

  now_epoch="$(date +%s)"
  finish_epoch=$((now_epoch + eta_sec))
  if ! eta_finish="$(date -d "@${finish_epoch}" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null)"; then
    if ! eta_finish="$(date -r "${finish_epoch}" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null)"; then
      eta_finish="n/a"
    fi
  fi

  progress="${last_epoch}/${total_epochs}"
  printf '%s\t%s\t%s\t%s' "${progress}" "${eta_duration}" "${eta_finish}" "${eta_status}"
}

extract_flag() {
  local key="$1"
  local default_value="$2"
  shift 2

  local -a argv=("$@")
  local i token next
  local long_key="--${key}"
  local long_key_with_equal="--${key}="

  for ((i = 0; i < ${#argv[@]}; i++)); do
    token="${argv[i]}"
    if [[ "${token}" == "${long_key}" ]]; then
      if ((i + 1 < ${#argv[@]})); then
        next="${argv[i + 1]}"
        printf '%s' "${next}"
        return 0
      fi
      break
    fi
    if [[ "${token}" == "${long_key_with_equal}"* ]]; then
      printf '%s' "${token#${long_key_with_equal}}"
      return 0
    fi
  done

  printf '%s' "${default_value}"
}

is_train_py_process() {
  local -a argv=("$@")
  local token
  for token in "${argv[@]}"; do
    case "${token}" in
      train.py|*/train.py)
        return 0
        ;;
    esac
  done
  return 1
}

INCLUDE_ALL=0
while (($#)); do
  case "$1" in
    --all)
      INCLUDE_ALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found in PATH." >&2
  exit 127
fi

if ! GPU_QUERY="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)"; then
  echo "[ERROR] Failed to query GPU metadata via nvidia-smi." >&2
  exit 1
fi

if ! PROC_QUERY="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null)"; then
  echo "[ERROR] Failed to query GPU compute processes via nvidia-smi." >&2
  exit 1
fi
if [[ -z "${PROC_QUERY}" ]]; then
  echo "No active GPU compute processes found."
  exit 0
fi

declare -A GPU_INDEX_BY_UUID=()
while IFS=',' read -r raw_index raw_uuid; do
  index="$(trim "${raw_index}")"
  uuid="$(trim "${raw_uuid}")"
  if [[ -n "${uuid}" ]]; then
    GPU_INDEX_BY_UUID["${uuid}"]="${index}"
  fi
done <<< "${GPU_QUERY}"

declare -a rows=()
declare -A seen_pid_gpu=()

while IFS=',' read -r raw_pid raw_gpu_uuid raw_mem; do
  pid="$(trim "${raw_pid}")"
  gpu_uuid="$(trim "${raw_gpu_uuid}")"
  gpu_mem_mb="$(trim "${raw_mem}")"

  [[ -n "${pid}" ]] || continue
  [[ "${pid}" =~ ^[0-9]+$ ]] || continue

  key="${pid}:${gpu_uuid}"
  if [[ -n "${seen_pid_gpu["${key}"]+x}" ]]; then
    continue
  fi
  seen_pid_gpu["${key}"]=1

  cmdline_path="/proc/${pid}/cmdline"
  if [[ ! -r "${cmdline_path}" ]]; then
    continue
  fi

  if ! mapfile -d '' -t argv < "${cmdline_path}" 2>/dev/null; then
    continue
  fi
  ((${#argv[@]} > 0)) || continue

  if ! is_train_py_process "${argv[@]}" && ((INCLUDE_ALL == 0)); then
    continue
  fi

  dataset="$(extract_flag "dataset" "retouch-Spectrailis" "${argv[@]}")"
  conn_num="$(extract_flag "conn_num" "8" "${argv[@]}")"
  label_mode="$(extract_flag "label_mode" "binary" "${argv[@]}")"
  dist_aux_loss="$(extract_flag "dist_aux_loss" "smooth_l1" "${argv[@]}")"
  direction_grouping="$(extract_flag "direction_grouping" "none" "${argv[@]}")"
  direction_fusion="$(extract_flag "direction_fusion" "weighted_sum" "${argv[@]}")"
  epochs="$(extract_flag "epochs" "45" "${argv[@]}")"
  output_dir_raw="$(extract_flag "output_dir" "output/" "${argv[@]}")"
  device="$(extract_flag "device" "" "${argv[@]}")"
  gpu_index="${GPU_INDEX_BY_UUID["${gpu_uuid}"]:-unknown}"
  output_dir_norm="$(normalize_output_dir "${output_dir_raw}")"

  if [[ "${output_dir_norm}" = /* ]]; then
    output_base="${output_dir_norm}"
  else
    output_base="${REPO_ROOT}/${output_dir_norm}"
  fi
  experiment_name="$(build_experiment_name "${label_mode}" "${conn_num}" "${dist_aux_loss}" "${direction_grouping}" "${direction_fusion}")"
  results_csv="${output_base}/${dataset}/${experiment_name}/results.csv"
  eta_fields="$(compute_eta_fields "${results_csv}" "${epochs}")"
  IFS=$'\t' read -r epoch_progress eta_duration eta_finish eta_status <<< "${eta_fields}"

  if [[ -z "${device}" ]]; then
    device="${gpu_index}"
  fi

  printf -v row '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    "${pid}" \
    "${gpu_index}" \
    "${dataset}" \
    "${conn_num}" \
    "${label_mode}" \
    "${dist_aux_loss}" \
    "${direction_grouping}" \
    "${direction_fusion}" \
    "${device}" \
    "${gpu_mem_mb}" \
    "${epoch_progress}" \
    "${eta_duration}" \
    "${eta_finish}" \
    "${eta_status}"
  rows+=("${row}")
done <<< "${PROC_QUERY}"

if ((${#rows[@]} == 0)); then
  echo "No active train.py GPU process found."
  if ((INCLUDE_ALL == 0)); then
    echo "Tip: run with --all to include non-train GPU compute processes."
  fi
  exit 0
fi

{
  printf 'PID\tGPU\tDATASET\tCONN_NUM\tLABEL_MODE\tDIST_AUX_LOSS\tDIRECTION_GROUPING\tDIRECTION_FUSION\tDEVICE\tGPU_MEM_MB\tEPOCH_PROGRESS\tETA_DURATION\tETA_FINISH\tETA_STATUS\n'
  printf '%s\n' "${rows[@]}"
} | {
  if command -v column >/dev/null 2>&1; then
    column -t -s $'\t'
  else
    cat
  fi
}
