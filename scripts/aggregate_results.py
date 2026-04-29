#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")


LATEX_TABLE_ARRAYSTRETCH = "1.5"
# NOTE: "decoder_guided" is a fork-specific fusion objective used in
# experiment names like "..._decoder_guided_A...".
KNOWN_CONN_FUSIONS = ("conv_residual", "scaled_sum", "gate", "decoder_guided")
ALLOWED_LABEL_MODES = ("binary", "dist", "dist_inverted")
CONN_FUSION_SORT_ORDER = {
    "none": 0,
    "conv_residual": 1,
    "gate": 2,
    "scaled_sum": 3,
    "decoder_guided": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate DconnNet results and export CSV/LaTeX/PDF."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="output",
        help=(
            "Result root directory. Supported layouts: "
            "<input-root>/<experiment>/<fold>/<input-name>, "
            "<input-root>/<experiment>/final_results_<fold>.csv, "
            "<input-root>/<scope>/<experiment>/final_results_<fold>.csv "
            "(for example output/5folds/<experiment>/final_results_<fold>.csv), "
            "<input-root>/<fold>/<input-name>, or <input-root>/final_results_<fold>.csv"
        ),
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="1",
        help="Comma-separated fold ids to aggregate (example: 1,2,3,4,5).",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="final_results_{fold}.csv",
        help=(
            "Input CSV file name pattern in each fold directory/root. "
            "If missing, legacy results_{fold}.csv is also checked automatically."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/summary",
        help="Directory where summary CSV/TEX/PDF will be written.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="summary",
        help="Output file stem (without extension).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Aggregate all counted datasets into combined outputs. "
            "By default, combined outputs include only DRIVE, CHASE, and "
            "OCTA-family datasets."
        ),
    )
    parser.add_argument(
        "--sample-vis-count",
        type=int,
        default=2,
        help=(
            "Number of highest- and lowest-Dice samples to visualize from the "
            "reference model. Set to 0 to disable sample visualization export."
        ),
    )
    parser.add_argument(
        "--max-vis-models",
        type=int,
        default=5,
        help=(
            "Maximum number of models to include in the sample visualization grid. "
            "Comparing too many models (e.g., > 10) significantly increases "
            "the execution time of matplotlib rendering."
        ),
    )
    return parser.parse_args()


def _to_float(value: str) -> float:
    value = value.strip()
    if value.lower() == "nan":
        return float("nan")
    return float(value)


def _safe_mean(values: List[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def _safe_std(values: List[float], mean: float) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if len(valid) <= 1:
        return 0.0
    var = sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)
    return math.sqrt(var)


def _safe_duration_std(values: List[float], mean: float) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return float("nan")
    if len(valid) == 1:
        return 0.0
    var = sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)
    return math.sqrt(var)


def _row_float_from_header(
    row: List[str],
    header_map: Dict[str, int],
    key: str,
    default: float = float("nan"),
) -> float:
    idx = header_map.get(key)
    if idx is None or idx >= len(row):
        return default
    value = row[idx].strip()
    if value == "":
        return default
    return _to_float(value)


def _dict_row_float(
    row: Dict[str, str],
    key: str,
    default: float = float("nan"),
) -> float:
    value = str(row.get(key, "")).strip()
    if value == "":
        return default
    return _to_float(value)


def parse_hms_duration(value: str) -> float:
    value = value.strip()
    if value == "":
        return float("nan")
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid HH:MM:SS duration: {value}")
    hours, minutes, seconds = [int(part) for part in parts]
    return float(hours * 3600 + minutes * 60 + seconds)


def format_hms_duration(value: float) -> str:
    if math.isnan(float(value)):
        return ""
    total_seconds = max(0, int(round(float(value))))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _fmt_duration_csv(value: object) -> str:
    if isinstance(value, (int, float)):
        return format_hms_duration(float(value))
    text = str(value).strip()
    return text


def _fmt_duration_latex(value: object) -> str:
    text = _fmt_duration_csv(value)
    return text if text else "N/A"


def _normalize_dataset_name(dataset_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", dataset_name.lower())


def _is_counted_dataset_name(dataset_name: object) -> bool:
    return _normalize_dataset_name(str(dataset_name)) != "trash"


def _dataset_summary_group(dataset_name: str) -> Tuple[int, str]:
    normalized = _normalize_dataset_name(dataset_name)
    if normalized == "cremi":
        return 0, "CREMI"
    if normalized == "drive":
        return 1, "DRIVE"
    if normalized in {"chase", "chasedb1"}:
        return 2, "CHASE"
    if normalized.startswith("isic"):
        return 3, "ISIC"
    if "octa" in normalized and any(tag in normalized for tag in ("3m", "6m")):
        return 4, "OCTA3M&6M"
    return 5, "Other datasets"


def _dataset_group_sort_key(dataset_name: str) -> Tuple[int, str, List[object]]:
    group_order, group_name = _dataset_summary_group(dataset_name)
    return group_order, group_name, natural_sort_key(dataset_name)


def _append_dataset_group_heading(lines: List[str], dataset_name: str) -> None:
    _, group_name = _dataset_summary_group(dataset_name)
    lines.append(rf"\section*{{{escape_latex_text(group_name)}}}")


def _append_page_break(lines: List[str], has_previous_content: bool) -> bool:
    if has_previous_content:
        lines.append(r"\clearpage")
    return True


def is_octa_dataset_name(dataset_name: object) -> bool:
    normalized = _normalize_dataset_name(str(dataset_name))
    return "octa" in normalized and any(tag in normalized for tag in ("3m", "6m"))


def should_include_dataset_in_default_aggregate(dataset_name: object) -> bool:
    normalized = _normalize_dataset_name(str(dataset_name))
    return (
        normalized == "drive"
        or normalized in {"chase", "chasedb1"}
        or is_octa_dataset_name(dataset_name)
    )


def should_include_dataset_in_aggregate(dataset_name: object, include_all: bool) -> bool:
    if not _is_counted_dataset_name(dataset_name):
        return False
    if include_all:
        return True
    return should_include_dataset_in_default_aggregate(dataset_name)


def filter_rows_for_aggregate_scope(
    rows: List[Dict[str, object]],
    include_all: bool,
) -> List[Dict[str, object]]:
    return [
        row for row in rows
        if should_include_dataset_in_aggregate(row.get("dataset", "unknown"), include_all)
    ]


def _format_conn_table_label(conn_num: object, conn_layout: object = "default") -> str:
    conn_text = str(conn_num)
    if str(conn_layout) == "out8" and conn_text == "8":
        return "8'"
    return conn_text


def _conn_layout_sort_key(conn_layout: object) -> Tuple[int, str]:
    layout_text = str(conn_layout)
    order = {
        "default": 0,
        "standard8": 0,
        "full24": 0,
        "out8": 1,
    }.get(layout_text, 9)
    return order, layout_text.lower()


def parse_final_summary_csv(path: str) -> Optional[Dict[str, object]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if row is None:
            return None

    if "dice" not in row or "jac" not in row or "cldice" not in row:
        return None

    eval_epoch_raw = row.get("eval_epoch", "").strip()
    eval_epoch = int(eval_epoch_raw) if eval_epoch_raw.isdigit() else 0

    elapsed_raw = row.get("elapsed_hms", "").strip()
    elapsed_seconds = (
        parse_hms_duration(elapsed_raw) if elapsed_raw != "" else float("nan")
    )

    return {
        "best_epoch": float(eval_epoch),
        "best_dice": _to_float(str(row.get("dice", "nan"))),
        "best_jac": _to_float(str(row.get("jac", "nan"))),
        "best_cldice": _to_float(str(row.get("cldice", "nan"))),
        "best_precision": _dict_row_float(row, "precision"),
        "best_accuracy": _dict_row_float(row, "accuracy"),
        "best_betti_error_0": _to_float(str(row.get("betti_error_0", "nan"))),
        "best_betti_error_1": _to_float(str(row.get("betti_error_1", "nan"))),
        "train_elapsed_seconds": elapsed_seconds,
        "train_elapsed_hms": format_hms_duration(elapsed_seconds),
    }


def parse_fold_csv(path: str) -> Dict[str, object]:
    final_summary = parse_final_summary_csv(path)
    if final_summary is not None:
        return final_summary

    epoch_rows: Dict[int, Dict[str, float]] = {}
    summary_epoch: Optional[int] = None
    summary_dice: Optional[float] = None
    header_map: Optional[Dict[str, int]] = None
    last_elapsed_seconds = float("nan")

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [item.strip() for item in row]
            if not any(item != "" for item in row):
                continue
            if header_map is None and row[0].lower() == "epoch":
                header_map = {
                    item.lower(): idx for idx, item in enumerate(row) if item != ""
                }
                continue

            if row[0] == "":
                continue

            try:
                epoch = int(row[0])
            except ValueError:
                continue

            if (
                header_map is not None and
                "dice" in header_map and
                "jac" in header_map and
                "cldice" in header_map
            ):
                required_idx = max(
                    header_map["dice"],
                    header_map["jac"],
                    header_map["cldice"],
                )
                if len(row) > required_idx:
                    dice = _to_float(row[header_map["dice"]])
                    jac = _to_float(row[header_map["jac"]])
                    cldice = _to_float(row[header_map["cldice"]])
                    precision = _row_float_from_header(row, header_map, "precision")
                    accuracy = _row_float_from_header(row, header_map, "accuracy")
                    betti_error_0 = _row_float_from_header(row, header_map, "betti_error_0")
                    betti_error_1 = _row_float_from_header(row, header_map, "betti_error_1")
                    if (
                        "elapsed_hms" in header_map and
                        header_map["elapsed_hms"] < len(row)
                    ):
                        elapsed_seconds = parse_hms_duration(row[header_map["elapsed_hms"]])
                        if not math.isnan(elapsed_seconds):
                            last_elapsed_seconds = elapsed_seconds
                    epoch_rows[epoch] = {
                        "dice": dice,
                        "jac": jac,
                        "cldice": cldice,
                        "precision": precision,
                        "accuracy": accuracy,
                        "betti_error_0": betti_error_0,
                        "betti_error_1": betti_error_1,
                    }
                    continue

            if len(row) >= 4:
                dice = _to_float(row[1])
                jac = _to_float(row[2])
                cldice = _to_float(row[3])
                precision = float("nan")
                accuracy = float("nan")
                betti_error_0 = _to_float(row[6]) if len(row) >= 7 and row[6] != "" else float("nan")
                betti_error_1 = _to_float(row[7]) if len(row) >= 8 and row[7] != "" else float("nan")
                epoch_rows[epoch] = {
                    "dice": dice,
                    "jac": jac,
                    "cldice": cldice,
                    "precision": precision,
                    "accuracy": accuracy,
                    "betti_error_0": betti_error_0,
                    "betti_error_1": betti_error_1,
                }
            elif len(row) >= 2:
                summary_epoch = int(row[0])
                summary_dice = _to_float(row[1])

    if not epoch_rows:
        print(f"No epoch rows found in: {path}")
        return None

    if summary_epoch is None or summary_dice is None:
        best_epoch, best_metrics = max(
            epoch_rows.items(), key=lambda kv: kv[1]["dice"]
        )
        best_dice = best_metrics["dice"]
        best_jac = best_metrics["jac"]
        best_cldice = best_metrics["cldice"]
        best_precision = best_metrics.get("precision", float("nan"))
        best_accuracy = best_metrics.get("accuracy", float("nan"))
        best_betti_error_0 = best_metrics["betti_error_0"]
        best_betti_error_1 = best_metrics["betti_error_1"]
    else:
        best_epoch = summary_epoch
        best_dice = summary_dice
        if best_epoch in epoch_rows:
            best_jac = epoch_rows[best_epoch]["jac"]
            best_cldice = epoch_rows[best_epoch]["cldice"]
            best_precision = epoch_rows[best_epoch].get("precision", float("nan"))
            best_accuracy = epoch_rows[best_epoch].get("accuracy", float("nan"))
            best_betti_error_0 = epoch_rows[best_epoch]["betti_error_0"]
            best_betti_error_1 = epoch_rows[best_epoch]["betti_error_1"]
        else:
            best_jac, best_cldice = float("nan"), float("nan")
            best_precision, best_accuracy = float("nan"), float("nan")
            best_betti_error_0, best_betti_error_1 = float("nan"), float("nan")

    return {
        "best_epoch": float(best_epoch),
        "best_dice": best_dice,
        "best_jac": best_jac,
        "best_cldice": best_cldice,
        "best_precision": best_precision,
        "best_accuracy": best_accuracy,
        "best_betti_error_0": best_betti_error_0,
        "best_betti_error_1": best_betti_error_1,
        "train_elapsed_seconds": last_elapsed_seconds,
        "train_elapsed_hms": format_hms_duration(last_elapsed_seconds),
    }


def build_fold_input_name_candidates(
    input_name: str,
    fold: str,
    allow_legacy_results: bool = True,
) -> List[str]:
    candidates: List[str] = []
    if "{fold}" in input_name:
        candidates.append(input_name.format(fold=fold))
    else:
        candidates.append(input_name)

    # Keep backward compatibility with legacy per-epoch result summaries.
    candidates.append(f"final_results_{fold}.csv")
    if allow_legacy_results:
        candidates.append(f"results_{fold}.csv")
    if str(fold) == "1":
        candidates.append("final_results.csv")
        if allow_legacy_results:
            candidates.append("results.csv")

    unique_candidates: List[str] = []
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        unique_candidates.append(name)
    return unique_candidates


def resolve_fold_csv_path(
    input_root: str,
    fold: str,
    input_name: str,
    allow_legacy_results: bool = True,
) -> str:
    checked: List[str] = []
    candidate_names = build_fold_input_name_candidates(
        input_name,
        fold,
        allow_legacy_results=allow_legacy_results,
    )

    for candidate_name in candidate_names:
        fold_dir_candidate = os.path.join(input_root, fold, candidate_name)
        checked.append(fold_dir_candidate)
        if os.path.isfile(fold_dir_candidate):
            return fold_dir_candidate

    for candidate_name in candidate_names:
        root_candidate = os.path.join(input_root, candidate_name)
        checked.append(root_candidate)
        if os.path.isfile(root_candidate):
            return root_candidate

    raise FileNotFoundError(
        "Missing fold CSV for fold "
        f"{fold}. Checked: {', '.join(checked)}"
    )


def discover_available_folds(
    input_root: str,
    input_name: str,
    allow_legacy_results: bool = True,
) -> List[str]:
    discovered = set()
    if not os.path.isdir(input_root):
        return []

    single_run_candidates = ["final_results.csv"]
    if allow_legacy_results:
        single_run_candidates.append("results.csv")
    if any(os.path.isfile(os.path.join(input_root, name)) for name in single_run_candidates):
        discovered.add("1")

    for entry in os.listdir(input_root):
        entry_path = os.path.join(input_root, entry)
        if not entry.isdigit() or not os.path.isdir(entry_path):
            continue

        candidate_names = build_fold_input_name_candidates(
            input_name,
            entry,
            allow_legacy_results=allow_legacy_results,
        )
        if any(os.path.isfile(os.path.join(entry_path, name)) for name in candidate_names):
            discovered.add(entry)

    if "{fold}" in input_name:
        prefix, suffix = input_name.split("{fold}")
        for entry in os.listdir(input_root):
            if not (entry.startswith(prefix) and entry.endswith(suffix)):
                continue
            fold = entry[len(prefix): len(entry) - len(suffix) if len(suffix) > 0 else len(entry)]
            if fold.isdigit():
                discovered.add(fold)

    for entry in os.listdir(input_root):
        if not entry.startswith("final_results_") or not entry.endswith(".csv"):
            if allow_legacy_results:
                match = re.fullmatch(r"results_(\d+)\.csv", entry)
                if match is None:
                    continue
                fold = match.group(1)
            else:
                continue
        else:
            fold = entry[len("final_results_"):-len(".csv")]
        if fold.isdigit():
            discovered.add(fold)

    return sorted(discovered, key=int)


def parse_scope_fold_count(name: str) -> Optional[int]:
    match = re.fullmatch(r"(\d+)folds?", name.lower())
    if match is None:
        return None
    return int(match.group(1))


def extract_scope_info(path: str) -> Tuple[str, object]:
    for part in reversed(os.path.normpath(path).split(os.sep)):
        fold_count = parse_scope_fold_count(part)
        if fold_count is not None:
            return part.lower(), fold_count
    return "direct", "NA"


def discover_nested_target_root_infos(
    input_root: str,
    input_name: str,
    max_depth: int = 3,
) -> List[Dict[str, object]]:
    target_infos: List[Dict[str, object]] = []
    pending: List[Tuple[str, int]] = [(input_root, 0)]
    seen_dirs: Set[str] = set()

    while pending:
        current_dir, depth = pending.pop(0)
        normalized_dir = os.path.normpath(current_dir)
        if normalized_dir in seen_dirs:
            continue
        seen_dirs.add(normalized_dir)

        rel_path = os.path.relpath(current_dir, input_root)
        rel_parts = [] if rel_path in {".", ""} else rel_path.split(os.sep)
        if "_smoke" in rel_parts:
            continue

        if depth > 0:
            available_folds = discover_available_folds(current_dir, input_name)
            if available_folds:
                scope_fold_count = None
                for part in rel_parts:
                    scope_fold_count = parse_scope_fold_count(part)
                    if scope_fold_count is not None:
                        break
                target_infos.append(
                    {
                        "root": current_dir,
                        "available_folds": available_folds,
                        "scope_fold_count": scope_fold_count,
                    }
                )
                continue

        if depth >= max_depth:
            continue

        try:
            entries = sorted(os.listdir(current_dir))
        except OSError:
            continue

        for entry in entries:
            entry_path = os.path.join(current_dir, entry)
            if os.path.isdir(entry_path):
                pending.append((entry_path, depth + 1))

    return target_infos


def select_target_root_infos(
    input_root: str,
    target_infos: List[Dict[str, object]],
    requested_folds: List[str],
) -> List[Dict[str, object]]:
    if not target_infos:
        return []

    scope_names = sorted(
        {
            f"{int(info['scope_fold_count'])}-fold"
            for info in target_infos
            if isinstance(info.get("scope_fold_count"), int)
        },
        key=natural_sort_key,
    )
    if len(scope_names) > 1:
        print(
            f"[INFO] {input_root}: multiple output scopes detected; "
            f"including {', '.join(scope_names)} experiment roots for requested folds "
            f"{', '.join(requested_folds)}."
        )

    return sorted(
        target_infos,
        key=lambda item: (
            int(item["scope_fold_count"]) if isinstance(item.get("scope_fold_count"), int) else 10 ** 9,
            str(item["root"]),
        ),
    )


def discover_target_roots(input_root: str, folds: List[str], input_name: str) -> List[str]:
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input root directory not found: {input_root}")

    available_in_root = discover_available_folds(input_root, input_name)
    if available_in_root:
        return [input_root]

    target_infos = discover_nested_target_root_infos(input_root, input_name)
    if target_infos:
        return [
            str(info["root"])
            for info in select_target_root_infos(input_root, target_infos, folds)
        ]

    raise FileNotFoundError(
        f"No fold CSV files were found in '{input_root}' or its nested subdirectories."
    )


def root_has_completed_final_results(input_root: str, folds: List[str], input_name: str) -> bool:
    for fold in folds:
        try:
            resolve_fold_csv_path(
                input_root,
                fold,
                input_name,
                allow_legacy_results=True,
            )
        except FileNotFoundError:
            return False
    return True


def resolve_folds_for_root(input_root: str, requested_folds: List[str], input_name: str) -> List[str]:
    missing = []
    for fold in requested_folds:
        try:
            resolve_fold_csv_path(input_root, fold, input_name, allow_legacy_results=True)
        except FileNotFoundError:
            missing.append(fold)

    if not missing:
        return requested_folds

    auto_folds = discover_available_folds(input_root, input_name, allow_legacy_results=False)
    if not auto_folds:
        raise FileNotFoundError(
            f"No completed folds in '{input_root}'. Missing requested folds: {', '.join(missing)}"
        )
    scope_name, scope_fold_count = extract_scope_info(input_root)
    log_tag = "WARN"
    if isinstance(scope_fold_count, int) and scope_fold_count == len(auto_folds):
        log_tag = "INFO"

    print(
        f"[{log_tag}] {input_root}: missing requested folds ({', '.join(missing)}). "
        f"Using detected folds: {', '.join(auto_folds)}"
        + (f" within {scope_name} scope." if log_tag == "INFO" else "")
    )
    return auto_folds


def aggregate_root(input_root: str, folds: List[str], input_name: str) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    fold_rows: List[Dict[str, object]] = []
    for fold in folds:
        csv_path = resolve_fold_csv_path(input_root, fold, input_name, allow_legacy_results=True)
        metrics = parse_fold_csv(csv_path)
        if metrics is None:
            print(f"[WARN] {input_root} fold {fold}: failed to parse metrics from CSV: {csv_path}")
            continue
        metrics["fold"] = float(fold)
        fold_rows.append(metrics)

    epochs = [row["best_epoch"] for row in fold_rows]
    dices = [row["best_dice"] for row in fold_rows]
    jacs = [row["best_jac"] for row in fold_rows]
    cldices = [row["best_cldice"] for row in fold_rows]
    precisions = [row["best_precision"] for row in fold_rows]
    accuracies = [row["best_accuracy"] for row in fold_rows]
    betti_error_0s = [row["best_betti_error_0"] for row in fold_rows]
    betti_error_1s = [row["best_betti_error_1"] for row in fold_rows]
    elapsed_seconds = [float(row["train_elapsed_seconds"]) for row in fold_rows]

    elapsed_mean_seconds = _safe_mean(elapsed_seconds)
    elapsed_std_seconds = _safe_duration_std(elapsed_seconds, elapsed_mean_seconds)

    mean_row = {
        "best_epoch": _safe_mean(epochs),
        "best_dice": _safe_mean(dices),
        "best_jac": _safe_mean(jacs),
        "best_cldice": _safe_mean(cldices),
        "best_precision": _safe_mean(precisions),
        "best_accuracy": _safe_mean(accuracies),
        "best_betti_error_0": _safe_mean(betti_error_0s),
        "best_betti_error_1": _safe_mean(betti_error_1s),
        "train_elapsed_seconds": elapsed_mean_seconds,
        "train_elapsed_hms": format_hms_duration(elapsed_mean_seconds),
    }
    std_row = {
        "best_epoch": _safe_std(epochs, mean_row["best_epoch"]),
        "best_dice": _safe_std(dices, mean_row["best_dice"]),
        "best_jac": _safe_std(jacs, mean_row["best_jac"]),
        "best_cldice": _safe_std(cldices, mean_row["best_cldice"]),
        "best_precision": _safe_std(precisions, mean_row["best_precision"]),
        "best_accuracy": _safe_std(accuracies, mean_row["best_accuracy"]),
        "best_betti_error_0": _safe_std(betti_error_0s, mean_row["best_betti_error_0"]),
        "best_betti_error_1": _safe_std(betti_error_1s, mean_row["best_betti_error_1"]),
        "train_elapsed_seconds": elapsed_std_seconds,
        "train_elapsed_hms": format_hms_duration(elapsed_std_seconds),
    }
    return fold_rows, mean_row, std_row


def parse_sample_metrics_csv(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "batch": int(row["batch"]),
                    "sample_in_batch": int(row["sample_in_batch"]),
                    "sample_name": row["sample_name"],
                    "val_loss": _to_float(row["val_loss"]),
                    "dice": _to_float(row["dice"]),
                    "jac": _to_float(row["jac"]),
                    "cldice": _to_float(row["cldice"]),
                }
            )
    return rows


def discover_checkpoint_epochs(model_dir: str) -> List[int]:
    checkpoint_root = os.path.join(model_dir, "checkpoint_batches")
    if not os.path.isdir(checkpoint_root):
        return []

    epochs: List[int] = []
    for entry in os.listdir(checkpoint_root):
        match = re.fullmatch(r"epoch_(\d+)", entry)
        if match is not None:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def resolve_visualization_epoch(model_dir: str, target_epoch: int) -> Tuple[Optional[int], bool]:
    epochs = discover_checkpoint_epochs(model_dir)
    if not epochs:
        return None, False
    if target_epoch in epochs:
        return target_epoch, True

    nearest = min(epochs, key=lambda epoch: (abs(epoch - target_epoch), -epoch))
    return nearest, False


def find_fold_row(fold_rows: List[Dict[str, object]], fold: int) -> Optional[Dict[str, object]]:
    for row in fold_rows:
        if int(row["fold"]) == int(fold):
            return row
    return None


def build_model_candidate(summary: Dict[str, object], fold_row: Dict[str, object]) -> Dict[str, object]:
    fold = int(fold_row["fold"])
    model_dir = os.path.join(str(summary["root"]), "models", str(fold))
    if not os.path.isdir(model_dir):
        model_dir = os.path.join(str(summary["root"]), "models")
    sample_csv = os.path.join(model_dir, "test_sample_metrics.csv")
    sample_rows = parse_sample_metrics_csv(sample_csv) if os.path.isfile(sample_csv) else []
    best_epoch = int(fold_row["best_epoch"])
    vis_epoch, is_exact = resolve_visualization_epoch(model_dir, best_epoch)
    exp_meta = parse_experiment_metadata(os.path.basename(os.path.normpath(str(summary["root"]))))
    _, fold_scope_count = extract_scope_info(str(summary["root"]))
    config_folds = fold_scope_count if isinstance(fold_scope_count, int) else len(summary["folds"])

    return {
        "root": summary["root"],
        "root_name": summary["root_name"],
        "experiment": exp_meta["experiment"],
        "conn_num": exp_meta["conn_num"],
        "conn_layout": exp_meta["conn_layout"],
        "loss": exp_meta["loss"],
        "config_folds": config_folds,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_dice": float(fold_row["best_dice"]),
        "model_dir": model_dir,
        "sample_csv": sample_csv,
        "sample_rows": sample_rows,
        "vis_epoch": vis_epoch,
        "vis_epoch_exact": is_exact,
    }


def choose_visualization_models(
    root_summaries: List[Dict[str, object]],
) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    if not root_summaries:
        return None, []

    ranked_summaries = sorted(
        root_summaries,
        key=lambda item: float(item["mean_row"]["best_dice"]),
        reverse=True,
    )
    reference_summary = ranked_summaries[0]
    reference_fold_row = max(
        reference_summary["fold_rows"],
        key=lambda row: float(row["best_dice"]),
    )
    reference_model = build_model_candidate(reference_summary, reference_fold_row)
    reference_fold = int(reference_fold_row["fold"])

    models: List[Dict[str, object]] = [reference_model]
    for summary in ranked_summaries[1:]:
        fold_row = find_fold_row(summary["fold_rows"], reference_fold)
        if fold_row is None:
            fold_row = max(
                summary["fold_rows"],
                key=lambda row: float(row["best_dice"]),
            )
            print(
                f"[WARN] {summary['root_name']}: fold {reference_fold} "
                "not found for sample comparison; using its best fold instead."
            )
        models.append(build_model_candidate(summary, fold_row))

    return reference_model, models


def filter_sample_rows_for_epoch(sample_rows: List[Dict[str, object]], epoch: int) -> List[Dict[str, object]]:
    return [row for row in sample_rows if int(row["epoch"]) == int(epoch)]


def select_ranked_samples(sample_rows: List[Dict[str, object]], count: int) -> List[Dict[str, object]]:
    if count <= 0:
        return []

    selected: List[Dict[str, object]] = []
    used_names: Set[str] = set()

    top_rows = sorted(sample_rows, key=lambda row: (-float(row["dice"]), str(row["sample_name"])))
    for row in top_rows:
        sample_name = str(row["sample_name"])
        if sample_name in used_names:
            continue
        picked = dict(row)
        picked["sample_group"] = "top"
        selected.append(picked)
        used_names.add(sample_name)
        if len([item for item in selected if item["sample_group"] == "top"]) >= count:
            break

    bottom_rows = sorted(sample_rows, key=lambda row: (float(row["dice"]), str(row["sample_name"])))
    bottom_selected = 0
    for row in bottom_rows:
        sample_name = str(row["sample_name"])
        if sample_name in used_names:
            continue
        picked = dict(row)
        picked["sample_group"] = "bottom"
        selected.append(picked)
        used_names.add(sample_name)
        bottom_selected += 1
        if bottom_selected >= count:
            break

    for idx, row in enumerate(selected, start=1):
        row["sample_rank"] = idx

    return selected


def find_sample_row(sample_rows: List[Dict[str, object]], epoch: int, sample_name: str) -> Optional[Dict[str, object]]:
    for row in sample_rows:
        if int(row["epoch"]) == int(epoch) and str(row["sample_name"]) == sample_name:
            return row
    return None


def checkpoint_image_path(model_dir: str, epoch: Optional[int], kind: str, batch_idx: int) -> Optional[str]:
    if epoch is None:
        return None
    path = os.path.join(
        model_dir,
        "checkpoint_batches",
        f"epoch_{int(epoch):03d}",
        kind,
        f"batch_{int(batch_idx):04d}.png",
    )
    return path if os.path.isfile(path) else None


def load_panel_image(image_path: str, panel_kind: str) -> np.ndarray:
    with Image.open(image_path) as image:
        if panel_kind == "image":
            # Keep input images in RGB and auto-correct likely BGR-encoded snapshots.
            image = image.convert("RGB")
            image_arr = np.asarray(image)
            if image_arr.ndim == 3 and image_arr.shape[-1] == 3:
                red_mean = float(np.mean(image_arr[:, :, 0]))
                blue_mean = float(np.mean(image_arr[:, :, 2]))
                # Fundus-like images should generally not be blue-dominant.
                # If blue is clearly larger than red, treat it as BGR-encoded and swap.
                if blue_mean > red_mean * 1.10:
                    image_arr = image_arr[:, :, ::-1]
            return image_arr

        # Segmentation targets/predictions are binary-like maps; render as single-channel grayscale.
        image = image.convert("L")
        return np.asarray(image)


def render_panel(
    ax,
    image_path: Optional[str],
    title: str,
    panel_kind: str,
    row_label: Optional[str] = None,
) -> None:
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if row_label:
        ax.set_ylabel(row_label, rotation=0, labelpad=52, va="center")

    if image_path is None:
        ax.set_facecolor("#f4f4f4")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
        return

    image = load_panel_image(image_path, panel_kind)
    if panel_kind != "image":
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)


def format_model_config_summary(model: Dict[str, object], multiline: bool = False) -> str:
    separator = "\n" if multiline else " | "
    seg_aux_label = _format_segaux_column_value(
        model.get("seg_aux_weight"),
        model.get("seg_aux_variant", "none"),
    )
    fields = [
        f"Exp={model['experiment']}",
        f"Conn={_format_conn_table_label(model['conn_num'], model.get('conn_layout', 'default'))}",
        f"Fusion={_format_fusion_table_label(model.get('conn_fusion', 'none'), model.get('fusion_loss_profile', 'A'), model.get('fusion_residual_scale'))}",
        f"SegAux={seg_aux_label or 'none'}",
        f"Loss={model['loss']}",
    ]
    fields.append(f"Folds={model['config_folds']}")
    return separator.join(fields)


def build_sample_visualization_fieldnames(model_count: int) -> List[str]:
    fieldnames = [
        "sample_rank",
        "sample_group",
        "sample_name",
        "reference_dice",
        "reference_jac",
        "reference_cldice",
        "image_path",
        "gt_path",
    ]
    for idx in range(1, model_count + 1):
        fieldnames.extend(
            [
                f"model{idx}_name",
                f"model{idx}_config",
                f"model{idx}_experiment",
                f"model{idx}_conn_num",
                f"model{idx}_loss",
                f"model{idx}_folds",
                f"model{idx}_fold",
                f"model{idx}_best_epoch",
                f"model{idx}_vis_epoch",
                f"model{idx}_vis_epoch_exact",
                f"model{idx}_dice",
                f"model{idx}_pred_path",
            ]
        )
    return fieldnames


def write_sample_visualization_csv(path: str, rows: List[Dict[str, object]], model_count: int) -> None:
    fieldnames = build_sample_visualization_fieldnames(model_count)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_sample_visualization(
    output_dir: str,
    image_output_dir: str,
    output_stem: str,
    root_summaries: List[Dict[str, object]],
    sample_vis_count: int,
    max_vis_models: int = 5,
) -> None:
    if sample_vis_count <= 0:
        return

    reference_model, models = choose_visualization_models(root_summaries)
    if reference_model is None or not models:
        return

    if max_vis_models > 0 and len(models) > max_vis_models:
        models = models[:max_vis_models]

    if reference_model["vis_epoch"] is None:
        print(f"[WARN] {reference_model['model_dir']}: no checkpoint_batches found; skipping sample visualization.")
        return
    if not reference_model["sample_rows"]:
        print(f"[WARN] {reference_model['sample_csv']}: missing sample metrics; skipping sample visualization.")
        return

    reference_rows = filter_sample_rows_for_epoch(reference_model["sample_rows"], int(reference_model["best_epoch"]))
    if not reference_rows:
        print(
            f"[WARN] {reference_model['sample_csv']}: no rows for best epoch {reference_model['best_epoch']}; "
            "skipping sample visualization."
        )
        return

    selected_rows = select_ranked_samples(reference_rows, sample_vis_count)
    if not selected_rows:
        print("[WARN] No sample rows available for visualization.")
        return

    if not bool(reference_model["vis_epoch_exact"]):
        print(
            f"[WARN] {reference_model['root_name']} fold {reference_model['fold']}: "
            f"best epoch {reference_model['best_epoch']} has no saved PNG; "
            f"using nearest checkpoint epoch {reference_model['vis_epoch']}."
        )
    for model in models[1:]:
        if model["vis_epoch"] is None:
            print(f"[WARN] {model['model_dir']}: no checkpoint_batches found; prediction column will be N/A.")
            continue
        if not bool(model["vis_epoch_exact"]):
            print(
                f"[WARN] {model['root_name']} fold {model['fold']}: "
                f"best epoch {model['best_epoch']} has no saved PNG; using nearest checkpoint epoch {model['vis_epoch']}."
            )
        if not model["sample_rows"]:
            print(f"[WARN] {model['sample_csv']}: missing sample metrics; prediction column may be N/A.")

    render_rows: List[Dict[str, object]] = []
    for row in selected_rows:
        sample_name = str(row["sample_name"])
        batch_idx = int(row["batch"])
        image_path = checkpoint_image_path(str(reference_model["model_dir"]), reference_model["vis_epoch"], "image", batch_idx)
        gt_path = checkpoint_image_path(str(reference_model["model_dir"]), reference_model["vis_epoch"], "mask", batch_idx)

        render_row: Dict[str, object] = {
            "sample_rank": int(row["sample_rank"]),
            "sample_group": str(row["sample_group"]),
            "sample_name": sample_name,
            "reference_dice": float(row["dice"]),
            "reference_jac": float(row["jac"]),
            "reference_cldice": float(row["cldice"]),
            "image_path": image_path or "",
            "gt_path": gt_path or "",
        }

        for idx, model in enumerate(models, start=1):
            model_row = None
            model_pred_path = None
            model_dice = float("nan")
            model_vis_epoch = ""
            if model["vis_epoch"] is not None:
                model_vis_epoch = int(model["vis_epoch"])

            if model["vis_epoch"] is not None and model["sample_rows"]:
                model_row = find_sample_row(model["sample_rows"], int(model["best_epoch"]), sample_name)
                if model_row is not None:
                    model_dice = float(model_row["dice"])
                    model_pred_path = checkpoint_image_path(
                        str(model["model_dir"]),
                        model["vis_epoch"],
                        "pred",
                        int(model_row["batch"]),
                    )

            render_row[f"model{idx}_name"] = str(model["root_name"])
            render_row[f"model{idx}_config"] = format_model_config_summary(model)
            render_row[f"model{idx}_experiment"] = str(model["experiment"])
            render_row[f"model{idx}_conn_num"] = _format_conn_table_label(
                model["conn_num"],
                model.get("conn_layout", "default"),
            )
            render_row[f"model{idx}_loss"] = str(model["loss"])
            render_row[f"model{idx}_folds"] = model["config_folds"]
            render_row[f"model{idx}_fold"] = int(model["fold"])
            render_row[f"model{idx}_best_epoch"] = int(model["best_epoch"])
            render_row[f"model{idx}_vis_epoch"] = model_vis_epoch
            render_row[f"model{idx}_vis_epoch_exact"] = bool(model["vis_epoch_exact"])
            render_row[f"model{idx}_dice"] = model_dice
            render_row[f"model{idx}_pred_path"] = model_pred_path or ""

        render_rows.append(render_row)

    vis_csv_out = os.path.join(output_dir, f"{output_stem}_sample_visualization.csv")
    vis_png_out = os.path.join(image_output_dir, f"{output_stem}_sample_visualization.png")
    write_sample_visualization_csv(vis_csv_out, render_rows, len(models))

    fig, axes = plt.subplots(
        nrows=len(render_rows),
        ncols=2 + len(models),
        figsize=(max(14, (2 + len(models)) * 3.7), max(4, len(render_rows) * 4)),
        squeeze=False,
    )

    model_titles = [format_model_config_summary(model, multiline=True) for model in models]
    figure_title = (
        f"Reference: {format_model_config_summary(reference_model)} / fold {reference_model['fold']} "
        f"/ best epoch {reference_model['best_epoch']} (viz epoch {reference_model['vis_epoch']})"
    )
    if len(models) > 1:
        figure_title += f"\nCompared Models: {len(models)}"

    for row_idx, row in enumerate(render_rows):
        row_label = (
            f"Sample {row['sample_rank']}\n"
            f"{row['sample_name']}\n"
            f"{row['sample_group']} dice={row['reference_dice']:.4f}"
        )
        render_panel(
            axes[row_idx][0],
            row["image_path"] or None,
            "Image",
            panel_kind="image",
            row_label=row_label,
        )
        render_panel(axes[row_idx][1], row["gt_path"] or None, "GT", panel_kind="mask")
        for model_idx, title in enumerate(model_titles, start=1):
            render_panel(
                axes[row_idx][model_idx + 1],
                row.get(f"model{model_idx}_pred_path") or None,
                title,
                panel_kind="pred",
            )

    fig.suptitle(figure_title, fontsize=12)
    plt.tight_layout(rect=(0.03, 0.03, 1.0, 0.93))
    fig.savefig(vis_png_out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAMPLE_VIS] CSV: {vis_csv_out}")
    print(f"[SAMPLE_VIS] PNG: {vis_png_out}")


def write_summary_csv(path: str, fold_rows: List[Dict[str, object]], mean_row: Dict[str, object], std_row: Dict[str, object]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "fold",
                "best_epoch",
                "best_dice",
                "best_jac",
                "best_cldice",
                "best_betti_error_0",
                "best_betti_error_1",
                "train_elapsed_hms",
            ]
        )
        for row in fold_rows:
            writer.writerow(
                [
                    int(row["fold"]),
                    int(row["best_epoch"]),
                    f'{row["best_dice"]:.4f}',
                    f'{row["best_jac"]:.4f}' if not math.isnan(
                        row["best_jac"]) else "nan",
                    f'{row["best_cldice"]:.4f}' if not math.isnan(
                        row["best_cldice"]) else "nan",
                    f'{row["best_betti_error_0"]:.4f}' if not math.isnan(
                        row["best_betti_error_0"]) else "nan",
                    f'{row["best_betti_error_1"]:.4f}' if not math.isnan(
                        row["best_betti_error_1"]) else "nan",
                    _fmt_duration_csv(row.get("train_elapsed_hms", "")),
                ]
            )
        writer.writerow(
            [
                "mean",
                f'{mean_row["best_epoch"]:.4f}',
                f'{mean_row["best_dice"]:.4f}',
                f'{mean_row["best_jac"]:.4f}' if not math.isnan(
                    mean_row["best_jac"]) else "nan",
                f'{mean_row["best_cldice"]:.4f}' if not math.isnan(
                    mean_row["best_cldice"]) else "nan",
                f'{mean_row["best_betti_error_0"]:.4f}' if not math.isnan(
                    mean_row["best_betti_error_0"]) else "nan",
                f'{mean_row["best_betti_error_1"]:.4f}' if not math.isnan(
                    mean_row["best_betti_error_1"]) else "nan",
                _fmt_duration_csv(mean_row.get("train_elapsed_hms", "")),
            ]
        )
        writer.writerow(
            [
                "std",
                f'{std_row["best_epoch"]:.4f}',
                f'{std_row["best_dice"]:.4f}',
                f'{std_row["best_jac"]:.4f}' if not math.isnan(
                    std_row["best_jac"]) else "nan",
                f'{std_row["best_cldice"]:.4f}' if not math.isnan(
                    std_row["best_cldice"]) else "nan",
                f'{std_row["best_betti_error_0"]:.4f}' if not math.isnan(
                    std_row["best_betti_error_0"]) else "nan",
                f'{std_row["best_betti_error_1"]:.4f}' if not math.isnan(
                    std_row["best_betti_error_1"]) else "nan",
                _fmt_duration_csv(std_row.get("train_elapsed_hms", "")),
            ]
        )


def _fmt_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def infer_loss_tag_from_legacy_name(experiment: str) -> str:
    name = experiment.lower()
    if "cl_dice" in name:
        return "cl_dice"
    if "_gjml_sf_l1" in name:
        return "gjml_sf_l1"
    if "_gjml_sj_l1" in name:
        return "gjml_sj_l1"
    if "_smooth_l1" in name:
        return "smooth_l1"
    if "dist" in name:
        return "smooth_l1"
    if "binary" in name:
        return "bce"
    return "unknown"


def _rank_row_indices(rows: List[Dict[str, object]], key: str, higher_is_better: bool = True) -> Tuple[Set[int], Set[int]]:
    indexed_values: List[Tuple[int, float]] = []
    for idx, row in enumerate(rows):
        value = row.get(key)
        if value is None:
            continue
        value_f = float(value)
        if math.isnan(value_f):
            continue
        indexed_values.append((idx, value_f))

    if not indexed_values:
        return set(), set()

    ranked_values = sorted({value for _, value in indexed_values}, reverse=higher_is_better)
    best_value = ranked_values[0]
    second_value = ranked_values[1] if len(ranked_values) > 1 else None

    best_indices = {idx for idx, value in indexed_values if value == best_value}
    second_indices = (
        {idx for idx, value in indexed_values if second_value is not None and value == second_value}
        if second_value is not None
        else set()
    )
    return best_indices, second_indices


def _fmt_latex_ranked_value(value: float, is_best: bool, is_second: bool) -> str:
    if math.isnan(value):
        return "-"
    text = _fmt_float(value)
    if is_best:
        return rf"\textbf{{{text}}}"
    if is_second:
        return rf"\underline{{{text}}}"
    return text


def natural_sort_key(text: str) -> Tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text))


def _fmt_latex_ranked_mean_std_marginal(mean: float, std: float, is_best: bool, is_second: bool) -> str:
    if math.isnan(mean):
        return "--"
    mean_txt = f"{mean:.4f}"
    if is_best:
        mean_txt = rf"\textbf{{{mean_txt}}}"
    elif is_second:
        mean_txt = rf"\underline{{{mean_txt}}}"
    std_txt = f"{std:.4f}"
    res = rf"\shortstack{{{mean_txt}\\({std_txt})}}"
    return res


def sanitize_scope_name_for_filename(scope_name: str) -> str:
    sanitized = scope_name.strip().replace(os.sep, "_")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", sanitized).strip("_")
    return sanitized if sanitized else "unknown_scope"


def canonical_dataset_name(text: str) -> Optional[str]:
    name = text.lower()
    if name in {"chase", "chasedb1"}:
        return "chase"
    if name == "drive":
        return "drive"
    if name.startswith("isic"):
        return "isic"
    if name == "cremi":
        return "cremi"
    return None


def extract_dataset_name(root: str, input_root: str) -> str:
    input_dataset = canonical_dataset_name(os.path.basename(os.path.normpath(input_root)))
    if input_dataset is not None:
        return input_dataset

    rel_path = os.path.relpath(root, input_root)
    if rel_path == ".":
        root_dataset = canonical_dataset_name(os.path.basename(os.path.normpath(root)))
        if root_dataset is not None:
            return root_dataset
        return os.path.basename(os.path.normpath(root))

    rel_parts = os.path.normpath(rel_path).split(os.sep)
    for part in rel_parts:
        dataset_name = canonical_dataset_name(part)
        if dataset_name is not None:
            return dataset_name

    for part in rel_parts:
        if parse_scope_fold_count(part) is None:
            return part

    return os.path.basename(os.path.normpath(input_root))


def parse_experiment_metadata(root_name: str) -> Dict[str, object]:
    tokens = root_name.split("_")
    loss = "unknown"
    conn_layout: object = "default"
    seg_aux_weight: Optional[float] = None
    seg_aux_variant = "none"

    segaux_match = re.search(r"_segaux(?:_w([0-9]*\.?[0-9]+))?$", root_name)
    if segaux_match is not None:
        if segaux_match.group(1) is not None:
            seg_aux_weight = float(segaux_match.group(1))
            seg_aux_variant = f"w{segaux_match.group(1)}"
        else:
            seg_aux_weight = 0.3
            seg_aux_variant = "segaux"

    # Preferred fork naming convention:
    #   <label_mode>[_<fusion_tag>]_ <conn_num> [_<conn_layout>] _<loss> [_suffix...]
    # Example:
    #   dist_inverted_decoder_guided_A_8_gjml_sf_l1
    #   dist_inverted_decoder_guided_A_8_gjml_sf_l1_segaux
    ordered_name_match = re.fullmatch(
        r"(binary|dist|dist_inverted)(?:_(.+?))?_(\d+)(?:_(standard8|full24|out8))?_(bce|smooth_l1|cl_dice|gjml_sf_l1|gjml_sj_l1)(?:_.+)?",
        root_name,
    )
    if ordered_name_match is not None:
        ordered_label_mode = ordered_name_match.group(1)
        ordered_body = ordered_name_match.group(2)
        conn_num = int(ordered_name_match.group(3))
        explicit_layout = ordered_name_match.group(4)
        conn_layout = explicit_layout if explicit_layout is not None else "default"
        loss = ordered_name_match.group(5)
        if ordered_body:
            experiment = f"{ordered_label_mode}_{ordered_body}"
        else:
            experiment = ordered_label_mode
        tokens = experiment.split("_")
    else:
        experiment = root_name

    loss_suffixes = [
        (["gjml", "sf", "l1"], "gjml_sf_l1"),
        (["gjml", "sj", "l1"], "gjml_sj_l1"),
        (["smooth", "l1"], "smooth_l1"),
        (["cl", "dice"], "cl_dice"),
        (["bce"], "bce"),
    ]
    for suffix_tokens, loss_name in loss_suffixes:
        if tokens[-len(suffix_tokens):] == suffix_tokens:
            tokens = tokens[:-len(suffix_tokens)]
            loss = loss_name
            break

    conn_num: object = "NA" if ordered_name_match is None else int(ordered_name_match.group(3))
    if tokens:
        if len(tokens) >= 2 and tokens[-2].isdigit() and tokens[-1] in {"standard8", "full24", "out8"}:
            conn_num = int(tokens[-2])
            conn_layout = tokens[-1]
            tokens = tokens[:-2]
        elif tokens[-1].isdigit():
            conn_num = int(tokens[-1])
            tokens = tokens[:-1]

    if ordered_name_match is None:
        experiment = "_".join(tokens) if tokens else root_name
    if loss == "unknown":
        loss = infer_loss_tag_from_legacy_name(root_name)
    if experiment == "binary":
        loss = "bce"

    conn_fusion = "none"
    fusion_loss_profile = "A"
    fusion_residual_scale: Optional[float] = None
    decoder_fusion = "none"
    label_mode = infer_label_mode_from_experiment(experiment)
    experiment_prefixes = ALLOWED_LABEL_MODES
    for prefix in experiment_prefixes:
        prefix_token = f"{prefix}_"
        if not experiment.startswith(prefix_token):
            continue
        experiment_body = experiment[len(prefix_token):]
        for fusion_name in KNOWN_CONN_FUSIONS:
            fusion_token = f"{fusion_name}_"
            if not experiment_body.startswith(fusion_token):
                continue
            fusion_tail = experiment_body[len(fusion_token):]
            # Historically we used strict regex matching here, but experiment
            # names sometimes carry extra suffixes. We only need the initial
            # profile token (A/B/C) and an optional decoder fusion name.
            parts = [p for p in fusion_tail.split("_") if p]
            if not parts or parts[0] not in {"A", "B", "C"}:
                continue

            conn_fusion = fusion_name
            fusion_loss_profile = parts[0]
            next_idx = 1
            if len(parts) >= 2:
                rs_match = re.fullmatch(r"rs([0-9]*\.?[0-9]+)", parts[1])
                if rs_match is not None:
                    fusion_residual_scale = float(rs_match.group(1))
                    next_idx = 2
            if len(parts) >= (next_idx + 2) and parts[next_idx] == "dec":
                decoder_fusion = "_".join(parts[next_idx + 1:])
            break
        if conn_fusion != "none":
            break

    return {
        "experiment": experiment,
        "label_mode": label_mode,
        "conn_num": conn_num,
        "conn_layout": conn_layout,
        "loss": loss,
        "conn_fusion": conn_fusion,
        "fusion_loss_profile": fusion_loss_profile,
        "fusion_residual_scale": fusion_residual_scale,
        "decoder_fusion": decoder_fusion,
        "seg_aux_weight": seg_aux_weight,
        "seg_aux_variant": seg_aux_variant,
    }


def infer_label_mode_from_experiment(experiment_name: object) -> str:
    experiment = str(experiment_name)
    if experiment == "binary" or experiment.startswith("binary_"):
        return "binary"
    if experiment == "dist_inverted" or experiment.startswith("dist_inverted_"):
        return "dist_inverted"
    if experiment == "dist" or experiment.startswith("dist_"):
        return "dist"
    return "unknown"


def _format_fusion_table_label(
    conn_fusion: object, fusion_loss_profile: object, fusion_residual_scale: object = None
) -> str:
    conn_fusion_name = str(conn_fusion if conn_fusion is not None else "none")
    if conn_fusion_name == "none":
        return "none"
    profile_name = str(fusion_loss_profile if fusion_loss_profile is not None else "A")
    if conn_fusion_name == "scaled_sum" and fusion_residual_scale is not None:
        try:
            return f"{conn_fusion_name}/{profile_name}/rs{float(fusion_residual_scale):g}"
        except (TypeError, ValueError):
            pass
    return f"{conn_fusion_name}/{profile_name}"


def _format_seg_aux_table_label(seg_aux_weight: object) -> Optional[str]:
    if seg_aux_weight is None:
        return None
    if isinstance(seg_aux_weight, (int, float)):
        return f"{float(seg_aux_weight):g}"
    text = str(seg_aux_weight).strip()
    return text if text else None


def _format_segaux_column_value(seg_aux_weight: object, seg_aux_variant: object) -> Optional[str]:
    seg_aux_label = _format_seg_aux_table_label(seg_aux_weight)
    if seg_aux_label is None:
        return None
    seg_aux_variant_name = str(seg_aux_variant if seg_aux_variant is not None else "none")
    if seg_aux_variant_name == "segaux":
        return "segaux"
    if seg_aux_variant_name.startswith("w"):
        return seg_aux_variant_name
    return seg_aux_label


def _table_fusion_and_decoder_labels(
    conn_fusion: object,
    fusion_loss_profile: object,
    fusion_residual_scale: object,
    seg_aux_weight: object = None,
    seg_aux_variant: object = "none",
) -> Tuple[str, str]:
    conn_fusion_name = str(conn_fusion if conn_fusion is not None else "none")
    profile_name = str(fusion_loss_profile if fusion_loss_profile is not None else "A")
    seg_aux_label = _format_segaux_column_value(seg_aux_weight, seg_aux_variant)

    if seg_aux_label is not None:
        return _format_fusion_table_label(conn_fusion_name, profile_name, fusion_residual_scale), seg_aux_label

    return (
        _format_fusion_table_label(conn_fusion_name, profile_name, fusion_residual_scale),
        "none",
    )


def _latex_summary_fusion_and_decoder_labels(
    conn_fusion: object,
    fusion_loss_profile: object,
    fusion_residual_scale: object,
    seg_aux_weight: object = None,
    seg_aux_variant: object = "none",
) -> Tuple[str, str]:
    conn_fusion_name = str(conn_fusion if conn_fusion is not None else "none")
    profile_name = str(fusion_loss_profile if fusion_loss_profile is not None else "A")

    if conn_fusion_name == "scaled_sum" and profile_name == "A":
        seg_aux_label = _format_segaux_column_value(seg_aux_weight, seg_aux_variant)
        dec_label = "none"
        if seg_aux_label is not None:
            dec_label = seg_aux_label
        if fusion_residual_scale is not None:
            try:
                return f"scaled_sum/rs{float(fusion_residual_scale):g}", dec_label
            except (TypeError, ValueError):
                pass
        return "scaled_sum", dec_label

    fusion_label, decoder_label = _table_fusion_and_decoder_labels(
        conn_fusion,
        fusion_loss_profile,
        fusion_residual_scale,
        seg_aux_weight,
        seg_aux_variant,
    )
    if conn_fusion_name == "decoder_guided" and profile_name == "A":
        return "decoder_guided", decoder_label
    return fusion_label, decoder_label


def _latex_ablation_fusion_spec(
    conn_fusion: object,
    fusion_loss_profile: object,
    fusion_residual_scale: object = None,
) -> str:
    conn_fusion_name = str(conn_fusion if conn_fusion is not None else "none")
    profile_name = str(fusion_loss_profile if fusion_loss_profile is not None else "A")
    if conn_fusion_name == "decoder_guided" and profile_name == "A":
        return ""
    if conn_fusion_name == "scaled_sum" and profile_name == "A":
        if fusion_residual_scale is not None:
            try:
                return f"rs{float(fusion_residual_scale):g}"
            except (TypeError, ValueError):
                return ""
        return ""
    if conn_fusion_name == "scaled_sum" and fusion_residual_scale is not None:
        try:
            return f"{profile_name}/rs{float(fusion_residual_scale):g}"
        except (TypeError, ValueError):
            pass
    return profile_name


def _fusion_profile_sort_value(profile_name: object) -> int:
    profile = str(profile_name)
    if profile == "A":
        return 0
    if profile == "B":
        return 1
    if profile == "C":
        return 2
    return 9


def _conn_fusion_sort_value(conn_fusion_name: object) -> int:
    conn_fusion = str(conn_fusion_name)
    return CONN_FUSION_SORT_ORDER.get(conn_fusion, 9)


def _loss_sort_key(loss_name: object) -> Tuple[int, str]:
    loss = str(loss_name if loss_name is not None else "unknown")
    loss_order = {
        "bce": 0,
        "smooth_l1": 1,
        "cl_dice": 2,
        "gjml_sf_l1": 3,
        "gjml_sj_l1": 3,
        "unknown": 9,
    }.get(loss, 9)
    return (loss_order, loss.lower())


def _experiment_mean_unique_key(row: Dict[str, object]) -> Tuple[object, ...]:
    # Keep one row per experiment directory name so suffix variants
    # (for example *_segaux, *_segaux_w0.1) are not collapsed.
    experiment_source = str(row.get("experiment_source", row.get("experiment", "unknown")))
    return (
        row["dataset"],
        experiment_source,
        row["experiment"],
        row.get("label_mode", "unknown"),
        row["conn_num"],
        row.get("conn_layout", "default"),
        row.get("conn_fusion", "none"),
        row.get("fusion_loss_profile", "A"),
        row["loss"],
        row["fold_scope"],
    )


def experiment_sort_key(row: Dict[str, object]) -> Tuple[object, ...]:
    label_mode = str(row.get("label_mode", infer_label_mode_from_experiment(row.get("experiment", "unknown"))))
    label_mode_order = {
        "binary": 0,
        "dist": 1,
        "dist_inverted": 2,
    }.get(label_mode, 9)

    conn_raw = row.get("conn_num", "NA")
    conn_sort = int(conn_raw) if isinstance(conn_raw, int) else 10 ** 9
    conn_layout_sort = _conn_layout_sort_key(row.get("conn_layout", "default"))

    fusion_label, decoder_label = _table_fusion_and_decoder_labels(
        row.get("conn_fusion", "none"),
        row.get("fusion_loss_profile", "A"),
        row.get("fusion_residual_scale"),
        row.get("seg_aux_weight"),
        row.get("seg_aux_variant", "none"),
    )
    fusion_name, _, fusion_profile_name = fusion_label.partition("/")
    fusion_sort = _conn_fusion_sort_value(fusion_name)
    fusion_profile_sort = _fusion_profile_sort_value(fusion_profile_name if fusion_profile_name else "A")
    decoder_sort = (0, "") if decoder_label == "none" else (1, natural_sort_key(decoder_label))

    loss_order, loss_norm = _loss_sort_key(row.get("loss", "Unknown"))

    fold_scope = str(row.get("fold_scope", "direct"))
    fold_scope_count = row.get("fold_scope_count", "NA")
    fold_scope_sort = int(fold_scope_count) if isinstance(fold_scope_count, int) else 10 ** 9

    return (
        label_mode_order,
        natural_sort_key(label_mode),
        conn_sort,
        conn_layout_sort,
        fusion_sort,
        fusion_profile_sort,
        natural_sort_key(fusion_label),
        decoder_sort,
        loss_order,
        loss_norm,
        fold_scope_sort,
        fold_scope.lower(),
    )


def write_experiment_mean_csv(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label_mode",
                "conn_num",
                "conn_label",
                "conn_layout",
                "conn_fusion",
                "fusion_loss_profile",
                "fusion_residual_scale",
                "loss",
                "num_folds",
                "best_dice_mean",
                "best_dice_std",
                "best_jac_mean",
                "best_jac_std",
                "best_cldice_mean",
                "best_cldice_std",
                "best_precision_mean",
                "best_precision_std",
                "best_accuracy_mean",
                "best_accuracy_std",
                "best_betti_error_0_mean",
                "best_betti_error_0_std",
                "best_betti_error_1_mean",
                "best_betti_error_1_std",
                "train_elapsed_mean_hms",
                "train_elapsed_std_hms",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    str(row.get("label_mode", infer_label_mode_from_experiment(row.get("experiment", "unknown")))),
                    row["conn_num"],
                    _format_conn_table_label(row["conn_num"], row.get("conn_layout", "default")),
                    row.get("conn_layout", "default"),
                    row.get("conn_fusion", "none"),
                    row.get("fusion_loss_profile", "A"),
                    _fmt_float(
                        (
                            float(row.get("fusion_residual_scale"))
                            if row.get("fusion_residual_scale") is not None
                            else float("nan")
                        )
                    ),
                    row["loss"],
                    int(row["num_folds"]),
                    _fmt_float(row["best_dice"]),
                    _fmt_float(row.get("best_dice_std", float("nan"))),
                    _fmt_float(row["best_jac"]),
                    _fmt_float(row.get("best_jac_std", float("nan"))),
                    _fmt_float(row["best_cldice"]),
                    _fmt_float(row.get("best_cldice_std", float("nan"))),
                    _fmt_float(row.get("best_precision", float("nan"))),
                    _fmt_float(row.get("best_precision_std", float("nan"))),
                    _fmt_float(row.get("best_accuracy", float("nan"))),
                    _fmt_float(row.get("best_accuracy_std", float("nan"))),
                    _fmt_float(row["best_betti_error_0"]),
                    _fmt_float(row.get("best_betti_error_0_std", float("nan"))),
                    _fmt_float(row["best_betti_error_1"]),
                    _fmt_float(row.get("best_betti_error_1_std", float("nan"))),
                    _fmt_duration_csv(row.get("train_elapsed_hms", "")),
                    _fmt_duration_csv(row.get("train_elapsed_std_hms", "")),
                ]
            )


def write_latex(
    path: str,
    title: str,
    fold_rows: List[Dict[str, object]],
    mean_row: Dict[str, object],
    std_row: Dict[str, object],
    dataset_name: str,
) -> None:
    metric_specs = dataset_fold_metric_specs(dataset_name)
    table_columns = 3 + len(metric_specs)
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a2paper,margin=1in,portrait]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\begin{document}",
        rf"\renewcommand{{\arraystretch}}{{{LATEX_TABLE_ARRAYSTRETCH}}}",
        rf"\section*{{{title}}}",
        r"\begin{table}[H]",
        r"\centering",
        rf"\begin{{tabular}}{{{'l' + 'c' * (table_columns - 1)}}}",
        r"\toprule",
        "Fold & Best Epoch & " + " & ".join(label for _, label in metric_specs) + r" & Train Time \\",
        r"\midrule",
    ]

    for row in fold_rows:
        time_txt = _fmt_duration_latex(row.get("train_elapsed_hms", ""))
        metric_vals = []
        for key, _ in metric_specs:
            metric_vals.append(_fmt_float(float(row.get(key, float("nan")))))
        lines.append(
            f'{int(row["fold"])} & {int(row["best_epoch"])} & '
            + " & ".join(metric_vals)
            + f' & {time_txt} \\\\'
        )

    mean_metrics = [_fmt_float(float(mean_row.get(key, float("nan")))) for key, _ in metric_specs]
    std_metrics = [_fmt_float(float(std_row.get(key, float("nan")))) for key, _ in metric_specs]
    mean_time = _fmt_duration_latex(mean_row.get("train_elapsed_hms", ""))
    std_time = _fmt_duration_latex(std_row.get("train_elapsed_hms", ""))
    lines += [
        r"\midrule",
        f'Mean & {mean_row["best_epoch"]:.4f} & ' + " & ".join(mean_metrics) + f' & {mean_time} \\\\',
        f'Std & {std_row["best_epoch"]:.4f} & ' + " & ".join(std_metrics) + f' & {std_time} \\\\',
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{final result summary}",
        r"\end{table}",
        r"\end{document}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _append_experiment_mean_table_lines(lines: List[str], rows: List[Dict[str, object]], caption: str) -> None:
    dice_best, dice_second = _rank_row_indices(rows, "best_dice")
    jac_best, jac_second = _rank_row_indices(rows, "best_jac")
    cldice_best, cldice_second = _rank_row_indices(rows, "best_cldice")
    betti0_best, betti0_second = _rank_row_indices(rows, "best_betti_error_0", higher_is_better=False)
    betti1_best, betti1_second = _rank_row_indices(rows, "best_betti_error_1", higher_is_better=False)

    lines += [
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{lccccccccccc}",
        r"\toprule",
        r"No. & label\_mode & Conn & Fusion & SegAux & Loss & Dice & Jac & clDice & Err $(\beta_0)$ & Err $(\beta_1)$ & Train Time \\",
        r"\midrule",
    ]

    for idx, row in enumerate(rows):
        fusion_label, decoder_label = _latex_summary_fusion_and_decoder_labels(
            row.get("conn_fusion", "none"),
            row.get("fusion_loss_profile", "A"),
            row.get("fusion_residual_scale"),
            row.get("seg_aux_weight"),
            row.get("seg_aux_variant", "none"),
        )
        dice_txt = _fmt_latex_ranked_mean_std_marginal(
            float(row["best_dice"]),
            float(row.get("best_dice_std", float("nan"))),
            idx in dice_best,
            idx in dice_second,
        )
        jac_txt = _fmt_latex_ranked_mean_std_marginal(
            float(row["best_jac"]),
            float(row.get("best_jac_std", float("nan"))),
            idx in jac_best,
            idx in jac_second,
        )
        cldice_txt = _fmt_latex_ranked_mean_std_marginal(
            float(row["best_cldice"]),
            float(row.get("best_cldice_std", float("nan"))),
            idx in cldice_best,
            idx in cldice_second,
        )
        betti0_txt = _fmt_latex_ranked_mean_std_marginal(
            float(row["best_betti_error_0"]),
            float(row.get("best_betti_error_0_std", float("nan"))),
            idx in betti0_best,
            idx in betti0_second,
        )
        betti1_txt = _fmt_latex_ranked_mean_std_marginal(
            float(row["best_betti_error_1"]),
            float(row.get("best_betti_error_1_std", float("nan"))),
            idx in betti1_best,
            idx in betti1_second,
        )
        mean_time = _fmt_duration_latex(row.get("train_elapsed_hms", ""))
        std_time = _fmt_duration_latex(row.get("train_elapsed_std_hms", ""))
        time_txt = rf"\shortstack{{{mean_time}\\({std_time})}}"

        lines.append(
            f'{idx + 1} '
            f'& {escape_latex_text(str(row.get("label_mode", infer_label_mode_from_experiment(row.get("experiment", "unknown")))))} '
            f'& {escape_latex_text(_format_conn_table_label(row["conn_num"], row.get("conn_layout", "default")))} '
            f'& {escape_latex_text(fusion_label)} '
            f'& {escape_latex_text(decoder_label)} '
            f'& {escape_latex_text(str(row["loss"]))} '
            f'& {dice_txt} & {jac_txt} & {cldice_txt} & {betti0_txt} & {betti1_txt} & {time_txt} \\\\'
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        r"\end{table}",
    ]


def dataset_metric_specs(dataset_name: str) -> List[Tuple[str, str, bool]]:
    name = dataset_name.lower()
    if name in {"chase", "drive"}:
        return [
            ("best_dice", "Dice", True),
            ("best_jac", "IoU", True),
            ("best_cldice", "clDice", True),
            ("best_betti_error_0", "B0 error", False),
            ("best_betti_error_1", "B1 error", False),
        ]
    if name in {"isic", "cremi"}:
        return [
            ("best_dice", "Dice", True),
            ("best_jac", "IoU", True),
            ("best_accuracy", "Accuracy", True),
            ("best_precision", "Precision", True),
        ]

    return [
        ("best_dice", "Dice", True),
        ("best_jac", "IoU", True),
        ("best_cldice", "clDice", True),
        ("best_betti_error_0", "B0 error", False),
        ("best_betti_error_1", "B1 error", False),
    ]


def dataset_fold_metric_specs(dataset_name: str) -> List[Tuple[str, str]]:
    name = dataset_name.lower()
    if name in {"isic", "cremi"}:
        return [
            ("best_dice", "Dice"),
            ("best_jac", "IoU"),
            ("best_accuracy", "Accuracy"),
            ("best_precision", "Precision"),
        ]

    return [
        ("best_dice", "Dice"),
        ("best_jac", "IoU"),
        ("best_cldice", "clDice"),
        ("best_betti_error_0", "B0 error"),
        ("best_betti_error_1", "B1 error"),
    ]


def _append_dataset_mean_table_lines(
    lines: List[str],
    rows: List[Dict[str, object]],
    dataset_name: str,
    include_std: bool = True,
) -> None:
    metric_specs = dataset_metric_specs(dataset_name)
    rank_map = {
        key: _rank_row_indices(rows, key, higher_is_better=higher_is_better)
        for key, _, higher_is_better in metric_specs
    }

    table_columns = 6 + len(metric_specs)
    lines += [
        r"\begin{table}[H]",
        r"\centering",
        rf"\begin{{tabular}}{{{'l' + 'c' * (table_columns - 1)}}}",
        r"\toprule",
        r"No. & label\_mode & Conn & Fusion & SegAux & Loss & " + " & ".join(label for _, label, _ in metric_specs) + r" \\",
        r"\midrule",
    ]

    for idx, row in enumerate(rows):
        fusion_label, decoder_label = _latex_summary_fusion_and_decoder_labels(
            row.get("conn_fusion", "none"),
            row.get("fusion_loss_profile", "A"),
            row.get("fusion_residual_scale"),
            row.get("seg_aux_weight"),
            row.get("seg_aux_variant", "none"),
        )
        metric_cells: List[str] = []
        for key, _, _ in metric_specs:
            best_indices, second_indices = rank_map[key]
            if include_std:
                metric_cells.append(
                    _fmt_latex_ranked_mean_std_marginal(
                        float(row.get(key, float("nan"))),
                        float(row.get(f"{key}_std", float("nan"))),
                        idx in best_indices,
                        idx in second_indices,
                    )
                )
            else:
                metric_cells.append(
                    _fmt_latex_ranked_value(
                        float(row.get(key, float("nan"))),
                        idx in best_indices,
                        idx in second_indices,
                    )
                )

        lines.append(
            f'{idx + 1} '
            f'& {escape_latex_text(str(row.get("label_mode", infer_label_mode_from_experiment(row.get("experiment", "unknown")))))} '
            f'& {escape_latex_text(_format_conn_table_label(row["conn_num"], row.get("conn_layout", "default")))} '
            f'& {escape_latex_text(fusion_label)} '
            f'& {escape_latex_text(decoder_label)} '
            f'& {escape_latex_text(str(row["loss"]))} '
            f'& ' + " & ".join(metric_cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]


def write_experiment_mean_latex(path: str, title: str, rows: List[Dict[str, object]]) -> None:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a2paper,margin=1in,portrait]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\begin{document}",
        rf"\renewcommand{{\arraystretch}}{{{LATEX_TABLE_ARRAYSTRETCH}}}",
    ]

    rows_by_dataset: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        dataset_name = str(row.get("dataset", "unknown_dataset"))
        if not _is_counted_dataset_name(dataset_name):
            continue
        rows_by_dataset.setdefault(dataset_name, []).append(row)

    has_previous_dataset = False
    if len(rows_by_dataset) == 1:
        dataset_name, dataset_rows = next(iter(rows_by_dataset.items()))
        has_previous_dataset = _append_page_break(lines, has_previous_dataset)
        _append_dataset_mean_table_lines(lines, dataset_rows, dataset_name)
        lines += [
            rf"\caption{{Cross-experiment mean summary ({escape_latex_text(dataset_name)})}}",
            r"\end{table}",
        ]
    else:
        grouped_rows: Dict[int, List[Tuple[str, List[Dict[str, object]]]]] = {}
        for dataset_name, dataset_rows in rows_by_dataset.items():
            group_order, _ = _dataset_summary_group(dataset_name)
            grouped_rows.setdefault(group_order, []).append((dataset_name, dataset_rows))

        for group_order in sorted(grouped_rows):
            group_items = sorted(grouped_rows[group_order], key=lambda item: _dataset_group_sort_key(item[0]))
            if not group_items:
                continue
            for dataset_name, dataset_rows in group_items:
                if len(dataset_rows) == 0:
                    continue
                has_previous_dataset = _append_page_break(lines, has_previous_dataset)
                _append_dataset_group_heading(lines, dataset_name)
                _append_dataset_mean_table_lines(lines, dataset_rows, dataset_name)
                lines += [
                    rf"\caption{{Cross-experiment mean summary ({escape_latex_text(dataset_name)})}}",
                    r"\end{table}",
                ]

    lines += [
        r"\end{document}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_ablation_latex(path: str, rows: List[Dict[str, object]]) -> None:
    datasets = set()
    for row in rows:
        dataset_name = str(row.get("dataset", "unknown"))
        if not _is_counted_dataset_name(dataset_name):
            continue
        datasets.add(dataset_name)
    dataset_list = sorted(list(datasets), key=natural_sort_key)
    dataset_buckets: List[Tuple[str, List[str]]] = []
    core_bucket = [ds for ds in ["cremi", "drive"] if ds in dataset_list]
    if core_bucket:
        dataset_buckets.append(("CREMI, DRIVE", core_bucket))
    other_bucket = [ds for ds in dataset_list if ds not in {"cremi", "drive"}]
    if other_bucket:
        dataset_buckets.append(("Other datasets", other_bucket))

    def row_label_mode(row: Dict[str, object]) -> str:
        label_mode = str(row.get("label_mode", ""))
        if label_mode in ALLOWED_LABEL_MODES:
            return label_mode
        return infer_label_mode_from_experiment(row.get("experiment", "unknown"))

    def fusion_objective_spec(row: Dict[str, object]) -> Tuple[str, str]:
        conn_fusion = str(row.get("conn_fusion", "none"))
        residual_scale = row.get("fusion_residual_scale")
        return (
            conn_fusion,
            _latex_ablation_fusion_spec(
                conn_fusion,
                row.get("fusion_loss_profile", "A"),
                residual_scale,
            ),
        )

    def decoder_option_spec(row: Dict[str, object]) -> Tuple[str]:
        _fusion_label, dec_label = _table_fusion_and_decoder_labels(
            row.get("conn_fusion", "none"),
            row.get("fusion_loss_profile", "A"),
            row.get("fusion_residual_scale"),
            row.get("seg_aux_weight"),
            row.get("seg_aux_variant", "none"),
        )
        return (str(dec_label),)

    def run_rank_key(row: Dict[str, object]) -> Tuple[object, ...]:
        dice = float(row.get("best_dice", float("nan")))
        iou = float(row.get("best_jac", float("nan")))
        dice_key = dice if not math.isnan(dice) else float("-inf")
        iou_key = iou if not math.isnan(iou) else float("-inf")
        return (
            dice_key,
            iou_key,
            tuple(-x if isinstance(x, int) else x for x in experiment_sort_key(row)),
            natural_sort_key(str(row.get("experiment", "unknown"))),
        )

    def _fmt_signed_delta(value: float) -> str:
        if math.isnan(value):
            return "-"
        return f"{value:+.4f}"

    def _fmt_latex_delta_colored(value: float) -> str:
        return _fmt_latex_delta_colored_with_direction(value, higher_is_better=True)

    def _fmt_latex_delta_colored_with_direction(value: float, higher_is_better: bool) -> str:
        if math.isnan(value):
            return "-"
        signed_value = f"{value:+.4f}"
        if value == 0:
            return signed_value
        is_improvement = value > 0 if higher_is_better else value < 0
        if is_improvement:
            return rf"\textcolor{{green!60!black}}{{{signed_value}}}"
        return rf"\textcolor{{red!70!black}}{{{signed_value}}}"

    variable_groups = [
        {
            "name": "Label Mode",
            "headers": ["Label Mode"],
            "func": lambda r: (row_label_mode(r),),
            "filter": lambda r: row_label_mode(r) in ALLOWED_LABEL_MODES,
        },
        {
            "name": "Loss",
            "headers": ["Loss"],
            "func": lambda r: (str(r.get("loss", "unknown")),),
            "filter": lambda r: str(r.get("loss", "unknown")) != "bce",
        },
        {
            "name": "Fusion + Decoder/SegAux",
            "headers": ["Conn. Fusion", "Fusion Spec", "SegAux"],
            "func": lambda r: (
                fusion_objective_spec(r)[0],
                fusion_objective_spec(r)[1],
                decoder_option_spec(r)[0],
            ),
            "sort_key": lambda tup: (
                _conn_fusion_sort_value(tup[0]),
                _fusion_profile_sort_value(
                    "A" if tup[0] == "decoder_guided" and tup[1] == "" else str(tup[1]).split("/", 1)[0]
                ),
                natural_sort_key(str(tup[0])),
                natural_sort_key(str(tup[1])),
                natural_sort_key(str(tup[2])),
            ),
        },
    ]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a2paper,margin=1in,portrait]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\usepackage{multirow}",
        r"\usepackage[table]{xcolor}",
        r"\begin{document}",
        rf"\renewcommand{{\arraystretch}}{{{LATEX_TABLE_ARRAYSTRETCH}}}",
        r"\section*{Ablation Studies (Category-grouped Best Runs)}",
    ]
    has_previous_table = False

    def get_ranks(values: List[float], higher_is_better: bool = True) -> Tuple[Set[int], Set[int]]:
        valid_vals = [v for v in values if not math.isnan(v)]
        if not valid_vals:
            return set(), set()
        ranked = sorted(set(valid_vals), reverse=higher_is_better)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        best_idx = {i for i, v in enumerate(values) if v == best}
        second_idx = {i for i, v in enumerate(values) if v == second} if second is not None else set()
        return best_idx, second_idx

    # Emit all tables for a bucket first so table numbering becomes:
    # core bucket(1,2,3,...) then other bucket(...).
    for bucket_name, bucket_datasets in dataset_buckets:
        for v_group in variable_groups:
            var_name = v_group["name"]
            headers = v_group["headers"]
            var_func = v_group["func"]
            row_filter = v_group.get("filter", lambda r: True)

            val_set = set()
            for row in rows:
                if row_filter(row):
                    val_set.add(var_func(row))

            def tuple_sort_key(tup):
                custom_sort_key = v_group.get("sort_key")
                if custom_sort_key is not None:
                    return custom_sort_key(tup)
                return [natural_sort_key(str(x)) for x in tup]

            val_list = sorted(list(val_set), key=tuple_sort_key)
            selected_rows: Dict[str, Dict[Tuple[object, ...], Dict[str, object]]] = {
                ds: {} for ds in bucket_datasets
            }

            for row in rows:
                if not row_filter(row):
                    continue
                ds = str(row.get("dataset", "unknown"))
                if ds not in selected_rows:
                    continue
                val = var_func(row)
                existing = selected_rows[ds].get(val)
                if existing is None or run_rank_key(row) > run_rank_key(existing):
                    selected_rows[ds][val] = row

            dataset_metrics = {ds: {"dice": [], "iou": []} for ds in bucket_datasets}

            for ds in bucket_datasets:
                for v in val_list:
                    selected = selected_rows[ds].get(v)
                    if selected is None:
                        dataset_metrics[ds]["dice"].append(float("nan"))
                        dataset_metrics[ds]["iou"].append(float("nan"))
                        continue
                    dataset_metrics[ds]["dice"].append(float(selected.get("best_dice", float("nan"))))
                    dataset_metrics[ds]["iou"].append(float(selected.get("best_jac", float("nan"))))

            dataset_ranks = {}
            for ds in bucket_datasets:
                dice_b, dice_s = get_ranks(dataset_metrics[ds]["dice"], True)
                iou_b, iou_s = get_ranks(dataset_metrics[ds]["iou"], True)
                dataset_ranks[ds] = {"dice_best": dice_b, "dice_second": dice_s, "iou_best": iou_b, "iou_second": iou_s}

            has_previous_table = _append_page_break(lines, has_previous_table)
            lines.extend([
                r"\begin{table}[H]",
                r"\centering",
            ])

            num_var_cols = len(headers) + 1
            col_str = "l" * num_var_cols + "cc" * len(bucket_datasets)
            lines.append(rf"\begin{{tabular}}{{{col_str}}}")
            lines.append(r"\toprule")

            header1 = [r"\multirow{2}{*}{No.}"]
            for h in headers:
                if isinstance(h, str) and h.startswith("LATEX:"):
                    header_txt = h[len("LATEX:"):]
                else:
                    header_txt = escape_latex_text(str(h))
                header1.append(rf"\multirow{{2}}{{*}}{{{header_txt}}}")
            for ds in bucket_datasets:
                header1.append(rf"\multicolumn{{2}}{{c}}{{{escape_latex_text(ds)}}}")
            lines.append(" & ".join(header1) + r" \\")

            cmidrules = []
            current_col = num_var_cols + 1
            for _ in bucket_datasets:
                cmidrules.append(rf"\cmidrule(lr){{{current_col}-{current_col+1}}}")
                current_col += 2
            if cmidrules:
                lines.append(" ".join(cmidrules))

            header2 = [""] * num_var_cols
            for _ in bucket_datasets:
                header2.extend(["Dice", "IoU"])
            lines.append(" & ".join(header2) + r" \\")
            lines.append(r"\midrule")

            for idx, v in enumerate(val_list):
                row_str_parts = [str(idx + 1)]
                for x in v:
                    x_str = str(x)
                    if x_str.startswith("LATEX:"):
                        row_str_parts.append(x_str[len("LATEX:"):])
                    else:
                        row_str_parts.append(escape_latex_text(x_str))
                for ds in bucket_datasets:
                    selected = selected_rows[ds].get(v)
                    dice = float("nan") if selected is None else float(selected.get("best_dice", float("nan")))
                    iou = float("nan") if selected is None else float(selected.get("best_jac", float("nan")))
                    dice_txt = _fmt_latex_ranked_value(
                        dice, idx in dataset_ranks[ds]["dice_best"], idx in dataset_ranks[ds]["dice_second"]
                    )
                    iou_txt = _fmt_latex_ranked_value(
                        iou, idx in dataset_ranks[ds]["iou_best"], idx in dataset_ranks[ds]["iou_second"]
                    )

                    row_str_parts.append(dice_txt)
                    row_str_parts.append(iou_txt)
                lines.append(" & ".join(row_str_parts) + r" \\")

            lines.extend([
                r"\bottomrule",
                r"\end{tabular}",
                rf"\caption{{Ablation on {escape_latex_text(var_name)} ({escape_latex_text(bucket_name)})}}",
                r"\end{table}",
            ])

    # Additional table: from each dataset best run, switch one element at a time
    # and report the best available alternative with +/- deltas.
    drop_axes = [
        ("label_mode", "Label Mode", lambda r: row_label_mode(r)),
        ("loss", "Loss", lambda r: str(r.get("loss", "unknown"))),
        ("dec", "SegAux", lambda r: decoder_option_spec(r)[0]),
        ("fusion", "Fusion", lambda r: str(fusion_objective_spec(r))),
    ]
    preferred_drop_one_losses = ("gjml_sf_l1", "smooth_l1")
    preferred_drop_one_loss_set = set(preferred_drop_one_losses)
    drop_metric_specs = [
        ("best_dice", "Dice", True),
        ("best_jac", "IoU", True),
        ("best_cldice", "clDice", True),
        ("best_betti_error_0", r"Err $(\beta_0)$", False),
        ("best_betti_error_1", r"Err $(\beta_1)$", False),
    ]

    def _loss_toggle_target(loss_name: object) -> Optional[str]:
        loss = str(loss_name if loss_name is not None else "unknown")
        if loss == "gjml_sf_l1":
            return "smooth_l1"
        if loss == "smooth_l1":
            return "gjml_sf_l1"
        return None

    def _option_cells(row: Optional[Dict[str, object]]) -> Tuple[str, str, str, str, str]:
        if row is None:
            return ("-", "-", "-", "-", "-")
        fusion_label, decoder_label = _latex_summary_fusion_and_decoder_labels(
            row.get("conn_fusion", "none"),
            row.get("fusion_loss_profile", "A"),
            row.get("fusion_residual_scale"),
            row.get("seg_aux_weight"),
            row.get("seg_aux_variant", "none"),
        )
        return (
            row_label_mode(row),
            _format_conn_table_label(row.get("conn_num", "unknown"), row.get("conn_layout", "default")),
            fusion_label,
            decoder_label,
            str(row.get("loss", "unknown")),
        )

    def _same_conn_config(a: Dict[str, object], b: Dict[str, object]) -> bool:
        return (
            a.get("conn_num", "unknown") == b.get("conn_num", "unknown")
            and str(a.get("conn_layout", "default")) == str(b.get("conn_layout", "default"))
        )

    def _is_plain_binary8_baseline(row: Dict[str, object]) -> bool:
        return (
            row_label_mode(row) == "binary"
            and row.get("conn_num", "unknown") == 8
            and str(row.get("conn_layout", "default")) == "default"
            and str(row.get("conn_fusion", "none")) == "none"
            and decoder_option_spec(row)[0] == "none"
            and str(row.get("loss", "unknown")) == "bce"
        )

    for bucket_name, bucket_datasets in dataset_buckets:
        # Split drop-one delta tables per dataset.
        table_dataset_groups: List[Tuple[str, List[str]]] = []
        if bucket_name == "CREMI, DRIVE":
            if "cremi" in bucket_datasets:
                table_dataset_groups.append(("CREMI", ["cremi"]))
            if "drive" in bucket_datasets:
                table_dataset_groups.append(("DRIVE", ["drive"]))
        else:
            for ds in bucket_datasets:
                table_dataset_groups.append((ds, [ds]))

        for table_label, table_datasets in table_dataset_groups:
            rows_out: List[
                Tuple[
                    str,
                    str,
                    Tuple[str, str, str, str, str],
                    List[float],
                    List[float],
                ]
            ] = []
            for ds in table_datasets:
                ds_rows = [r for r in rows if str(r.get("dataset", "unknown")) == ds]
                if len(ds_rows) == 0:
                    continue
                preferred_best_rows = [
                    r for r in ds_rows
                    if str(r.get("loss", "unknown")) in preferred_drop_one_loss_set
                ]
                best_row = max(preferred_best_rows or ds_rows, key=run_rank_key)
                best_metric_values = [
                    float(best_row.get(metric_key, float("nan")))
                    for metric_key, _metric_label, _higher_is_better in drop_metric_specs
                ]
                rows_out.append((
                    ds,
                    "BEST",
                    _option_cells(best_row),
                    best_metric_values,
                    [float("nan")] * len(drop_metric_specs),
                ))

                axis_values = {axis_key: axis_func(best_row) for axis_key, _, axis_func in drop_axes}
                for axis_key, axis_label, axis_func in drop_axes:
                    # Drop-one tables should remove an active choice from the best run.
                    # If the best row already uses the default/plain setting, there is
                    # nothing to drop for that axis.
                    if axis_key == "label_mode" and axis_values["label_mode"] == "binary":
                        rows_out.append((
                            ds,
                            f"-{axis_label}",
                            _option_cells(None),
                            [float("nan")] * len(drop_metric_specs),
                            [float("nan")] * len(drop_metric_specs),
                        ))
                        continue
                    if axis_key == "fusion" and axis_values["fusion"] == str(("none", "A")):
                        rows_out.append((
                            ds,
                            f"-{axis_label}",
                            _option_cells(None),
                            [float("nan")] * len(drop_metric_specs),
                            [float("nan")] * len(drop_metric_specs),
                        ))
                        continue
                    if axis_key == "dec" and axis_values["dec"] == "none":
                        rows_out.append((
                            ds,
                            f"-{axis_label}",
                            _option_cells(None),
                            [float("nan")] * len(drop_metric_specs),
                            [float("nan")] * len(drop_metric_specs),
                        ))
                        continue

                    alt_candidates = []
                    for cand in ds_rows:
                        if not _same_conn_config(cand, best_row):
                            continue

                        if axis_key == "loss":
                            target_loss = _loss_toggle_target(axis_values["loss"])
                            if target_loss is None:
                                continue
                            if str(cand.get("loss", "unknown")) != target_loss:
                                continue
                            if row_label_mode(cand) == axis_values["label_mode"] and \
                               str(fusion_objective_spec(cand)) == axis_values["fusion"] and \
                               decoder_option_spec(cand)[0] == axis_values["dec"]:
                                alt_candidates.append(cand)
                            continue

                        if axis_key == "label_mode":
                            # Special rule: when best is dist, dropped label mode
                            # must compare against binary+bce.
                            if axis_values["label_mode"] in {"dist", "dist_inverted"}:
                                if row_label_mode(cand) != "binary":
                                    continue
                                if str(cand.get("loss", "unknown")) != "bce":
                                    continue
                                if str(fusion_objective_spec(cand)) == axis_values["fusion"] and \
                                   decoder_option_spec(cand)[0] == axis_values["dec"]:
                                    alt_candidates.append(cand)
                                continue
                            if row_label_mode(cand) == axis_values["label_mode"]:
                                continue
                            if str(cand.get("loss", "unknown")) == axis_values["loss"] and \
                               str(fusion_objective_spec(cand)) == axis_values["fusion"] and \
                               decoder_option_spec(cand)[0] == axis_values["dec"]:
                                alt_candidates.append(cand)
                            continue

                        if axis_key == "fusion":
                            # Special rule: dropped fusion compares against the
                            # plain no-fusion, no-dec baseline with the same
                            # label mode, connectivity, and loss.
                            if str(cand.get("conn_fusion", "none")) != "none":
                                continue
                            if decoder_option_spec(cand)[0] != "none":
                                continue
                            if row_label_mode(cand) == axis_values["label_mode"] and \
                               str(cand.get("loss", "unknown")) == axis_values["loss"]:
                                alt_candidates.append(cand)
                            continue

                        if axis_key == "dec":
                            # Special rule: if best uses decoder option, dropped variant
                            # should compare against the plain dec=none configuration.
                            best_dec = axis_values["dec"]
                            cand_dec = decoder_option_spec(cand)[0]
                            if best_dec != "none":
                                if cand_dec != "none":
                                    continue
                                if row_label_mode(cand) == axis_values["label_mode"] and \
                                   str(cand.get("loss", "unknown")) == axis_values["loss"] and \
                                   str(fusion_objective_spec(cand)) == axis_values["fusion"]:
                                    alt_candidates.append(cand)
                                continue

                        if axis_func(cand) == axis_values[axis_key]:
                            continue
                        same_other_axes = True
                        for other_axis_key, _, other_axis_func in drop_axes:
                            if other_axis_key == axis_key:
                                continue
                            if other_axis_func(cand) != axis_values[other_axis_key]:
                                same_other_axes = False
                                break
                        if same_other_axes:
                            alt_candidates.append(cand)

                    if len(alt_candidates) == 0:
                        rows_out.append((
                            ds,
                            f"-{axis_label}",
                            _option_cells(None),
                            [float("nan")] * len(drop_metric_specs),
                            [float("nan")] * len(drop_metric_specs),
                        ))
                        continue

                    alt = max(alt_candidates, key=run_rank_key)
                    alt_metric_values = [
                        float(alt.get(metric_key, float("nan")))
                        for metric_key, _metric_label, _higher_is_better in drop_metric_specs
                    ]
                    alt_metric_deltas = []
                    for idx, _spec in enumerate(drop_metric_specs):
                        alt_value = alt_metric_values[idx]
                        best_value = best_metric_values[idx]
                        if math.isnan(alt_value) or math.isnan(best_value):
                            alt_metric_deltas.append(float("nan"))
                        else:
                            alt_metric_deltas.append(alt_value - best_value)
                    rows_out.append((ds, f"-{axis_label}", _option_cells(alt), alt_metric_values, alt_metric_deltas))

                baseline_candidates = [r for r in ds_rows if _is_plain_binary8_baseline(r)]
                if len(baseline_candidates) == 0:
                    rows_out.append((
                        ds,
                        "BASE",
                        _option_cells(None),
                        [float("nan")] * len(drop_metric_specs),
                        [float("nan")] * len(drop_metric_specs),
                    ))
                else:
                    baseline_row = max(baseline_candidates, key=run_rank_key)
                    baseline_metric_values = [
                        float(baseline_row.get(metric_key, float("nan")))
                        for metric_key, _metric_label, _higher_is_better in drop_metric_specs
                    ]
                    baseline_metric_deltas = []
                    for idx, _spec in enumerate(drop_metric_specs):
                        baseline_value = baseline_metric_values[idx]
                        best_value = best_metric_values[idx]
                        if math.isnan(baseline_value) or math.isnan(best_value):
                            baseline_metric_deltas.append(float("nan"))
                        else:
                            baseline_metric_deltas.append(baseline_value - best_value)
                    rows_out.append(
                        (
                            ds,
                            "BASE",
                            _option_cells(baseline_row),
                            baseline_metric_values,
                            baseline_metric_deltas,
                        )
                    )

            if len(rows_out) == 0:
                continue

            has_previous_table = _append_page_break(lines, has_previous_table)
            lines.extend([
                r"\begin{table}[H]",
                r"\centering",
                r"\begin{tabular}{lllcccccccccc}",
                r"\toprule",
                r"No. & Dataset & Variant"
                r" & label\_mode & Conn & Fusion & SegAux & Loss"
                r" & " + " & ".join(metric_label for _metric_key, metric_label, _higher_is_better in drop_metric_specs) + r" \\",
                r"\midrule",
            ])
            for row_idx, (ds, variant, opts, metric_values, metric_deltas) in enumerate(rows_out):
                metric_cells: List[str] = []
                for metric_idx, (_metric_key, _metric_label, higher_is_better) in enumerate(drop_metric_specs):
                    value = metric_values[metric_idx]
                    delta = metric_deltas[metric_idx]
                    if variant == "BEST":
                        metric_cells.append(_fmt_latex_ranked_value(value, False, False))
                    else:
                        metric_cells.append(_fmt_latex_delta_colored_with_direction(delta, higher_is_better))
                lines.append(
                    f"{row_idx + 1} & {escape_latex_text(ds)} & {escape_latex_text(variant)} "
                    f"& {escape_latex_text(opts[0])} & {escape_latex_text(opts[1])} & {escape_latex_text(opts[2])} & {escape_latex_text(opts[3])} & {escape_latex_text(opts[4])} "
                    f"& " + " & ".join(metric_cells) + r" \\"
                )
            lines.extend([
                r"\bottomrule",
                r"\end{tabular}",
                rf"\caption{{Best-run drop-one deltas ({escape_latex_text(table_label)})}}",
                r"\end{table}",
            ])

    lines.append(r"\end{document}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_experiment_mean_dataset_tables_latex(
    path: str,
    title: str,
    rows_by_dataset: Dict[str, List[Dict[str, object]]],
) -> None:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a2paper,margin=1in,portrait]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\begin{document}",
    ]

    grouped_rows: Dict[int, List[Tuple[str, List[Dict[str, object]]]]] = {}
    for dataset_name, rows in rows_by_dataset.items():
        if not _is_counted_dataset_name(dataset_name):
            continue
        group_order, _ = _dataset_summary_group(dataset_name)
        grouped_rows.setdefault(group_order, []).append((dataset_name, rows))

    has_previous_dataset = False
    for group_order in sorted(grouped_rows):
        group_items = sorted(grouped_rows[group_order], key=lambda item: _dataset_group_sort_key(item[0]))
        if not group_items:
            continue
        for dataset_name, rows in group_items:
            if len(rows) == 0:
                continue
            has_previous_dataset = _append_page_break(lines, has_previous_dataset)
            _append_dataset_group_heading(lines, dataset_name)
            rows_by_loss: Dict[str, List[Dict[str, object]]] = {}
            for row in rows:
                loss = str(row.get("loss", "unknown"))
                rows_by_loss.setdefault(loss, []).append(row)

            ordered_losses = sorted(rows_by_loss.keys(), key=_loss_sort_key)
            for loss in ordered_losses:
                loss_rows = rows_by_loss[loss]
                if len(loss_rows) == 0:
                    continue
                _append_dataset_mean_table_lines(lines, loss_rows, dataset_name, include_std=False)
                lines.extend([
                    rf"\caption{{Cross-experiment mean summary ({escape_latex_text(dataset_name)}, Loss={escape_latex_text(loss)})}}",
                    r"\end{table}",
                ])

    lines += [
        r"\end{document}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def escape_latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for key, value in replacements.items():
        escaped = escaped.replace(key, value)

    # Use \times for patterns like 1x1
    escaped = re.sub(r'(\d)x(\d)', r'$\1\\times \2$', escaped)
    # Use en-dash for octa500-3M etc to satisfy chktex
    escaped = re.sub(r'([a-zA-Z0-9])-([a-zA-Z0-9])', r'\1--\2', escaped)

    return escaped


def build_pdf(tex_path: str, pdf_out_path: Optional[str] = None) -> str:
    out_dir = os.path.dirname(os.path.abspath(tex_path))
    tex_file = os.path.basename(tex_path)
    tex_stem, _ = os.path.splitext(tex_file)
    generated_pdf_path = os.path.join(out_dir, f"{tex_stem}.pdf")
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_file,
    ]
    subprocess.run(cmd, cwd=out_dir, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    target_pdf_path = (
        os.path.abspath(pdf_out_path)
        if pdf_out_path is not None
        else generated_pdf_path
    )
    if os.path.abspath(generated_pdf_path) != target_pdf_path:
        os.makedirs(os.path.dirname(target_pdf_path), exist_ok=True)
        shutil.move(generated_pdf_path, target_pdf_path)
    return target_pdf_path


def root_display_name(root: str, input_root: str) -> str:
    rel_path = os.path.relpath(root, input_root)
    if rel_path == ".":
        return os.path.basename(os.path.normpath(root))
    return rel_path


def root_output_label(root: str, input_root: str) -> str:
    return root_display_name(root, input_root).replace(os.sep, "_")


def move_non_pdf_files_to_dump(output_dir: str, dump_dir: str) -> None:
    for entry in os.listdir(output_dir):
        src_path = os.path.join(output_dir, entry)
        if not os.path.isfile(src_path):
            continue
        if entry.lower().endswith(".pdf") or entry.lower().endswith(".png"):
            continue

        dst_path = os.path.join(dump_dir, entry)
        if os.path.exists(dst_path):
            if os.path.isfile(dst_path):
                os.remove(dst_path)
            else:
                shutil.rmtree(dst_path)
        shutil.move(src_path, dst_path)


def normalize_aggregate_stem(output_stem: str) -> str:
    stem = str(output_stem).strip()
    if stem.endswith("_experiment_means"):
        return stem
    return f"{stem}_experiment_means"


def aggregate_output_stem(output_stem: str, include_all: bool) -> str:
    stem = normalize_aggregate_stem(output_stem)
    if include_all:
        return f"{stem}_all"
    return stem


def should_exclude_ambiguous_scaled_sum_row(exp_meta: Dict[str, object]) -> bool:
    """Drop legacy scaled_sum rows without explicit residual-scale spec.

    These runs were produced before `_rs<scale>` was appended to the
    experiment name, so the residual scale cannot be recovered reliably.
    """
    conn_fusion = str(exp_meta.get("conn_fusion", "none"))
    profile = str(exp_meta.get("fusion_loss_profile", "A"))
    residual_scale = exp_meta.get("fusion_residual_scale")
    return (
        conn_fusion == "scaled_sum"
        and profile == "A"
        and residual_scale is None
    )


def remove_legacy_duplicate_aggregate_pdfs(output_dir: str, agg_stem: str) -> None:
    # If agg_stem is already "..._experiment_means", old behavior could also
    # emit "..._experiment_means_experiment_means_*". Remove those duplicates.
    legacy_stem = f"{agg_stem}_experiment_means"
    duplicate_pdf_paths = [
        os.path.join(output_dir, f"{legacy_stem}_datasets.pdf"),
        os.path.join(output_dir, f"{legacy_stem}_ablation.pdf"),
    ]
    for path in duplicate_pdf_paths:
        if os.path.isfile(path):
            os.remove(path)


def main() -> None:
    args = parse_args()
    requested_folds = [item.strip() for item in args.folds.split(",") if item.strip()]
    if not requested_folds:
        raise ValueError("No folds were provided.")

    os.makedirs(args.output_dir, exist_ok=True)
    dump_dir = os.path.join(args.output_dir, "dump")
    dump_csv_dir = os.path.join(dump_dir, "csv")
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(dump_csv_dir, exist_ok=True)
    move_non_pdf_files_to_dump(args.output_dir, dump_dir)
    target_roots = discover_target_roots(args.input_root, requested_folds, args.input_name)
    experiment_mean_rows: List[Dict[str, object]] = []
    root_summaries: List[Dict[str, object]] = []

    for root in target_roots:
        if not root_has_completed_final_results(root, requested_folds, args.input_name):
            display_name = root_display_name(root, args.input_root)
            print(
                f"[WARN] {root}: no final_results_*.csv found for requested folds; "
                f"skipping unfinished experiment ({display_name})."
            )
            continue

        folds = resolve_folds_for_root(root, requested_folds, args.input_name)
        fold_rows, mean_row, std_row = aggregate_root(root, folds, args.input_name)
        experiment_root_name = os.path.basename(os.path.normpath(root))
        display_name = root_display_name(root, args.input_root)
        scope_name, scope_fold_count = extract_scope_info(root)
        root_summaries.append(
            {
                "root": root,
                "root_name": display_name,
                "folds": folds,
                "fold_rows": fold_rows,
                "mean_row": mean_row,
                "std_row": std_row,
            }
        )
        exp_meta = parse_experiment_metadata(experiment_root_name)
        if should_exclude_ambiguous_scaled_sum_row(exp_meta):
            print(
                f"[WARN] {root}: ambiguous scaled_sum/A run without explicit rs suffix; "
                "excluding from aggregate outputs."
            )
            continue
        experiment_mean_rows.append(
            {
                "dataset": extract_dataset_name(root, args.input_root),
                "experiment_source": experiment_root_name,
                "experiment": exp_meta["experiment"],
                "label_mode": exp_meta["label_mode"],
                "conn_num": exp_meta["conn_num"],
                "conn_layout": exp_meta["conn_layout"],
                "conn_fusion": exp_meta["conn_fusion"],
                "fusion_loss_profile": exp_meta["fusion_loss_profile"],
                "fusion_residual_scale": exp_meta.get("fusion_residual_scale"),
                "decoder_fusion": exp_meta.get("decoder_fusion", "none"),
                "seg_aux_weight": exp_meta.get("seg_aux_weight"),
                "seg_aux_variant": exp_meta.get("seg_aux_variant", "none"),
                "loss": exp_meta["loss"],
                "fold_scope": scope_name,
                "fold_scope_count": scope_fold_count,
                "num_folds": float(len(folds)),
                "best_dice": mean_row["best_dice"],
                "best_dice_std": std_row["best_dice"],
                "best_jac": mean_row["best_jac"],
                "best_jac_std": std_row["best_jac"],
                "best_cldice": mean_row["best_cldice"],
                "best_cldice_std": std_row["best_cldice"],
                "best_precision": mean_row["best_precision"],
                "best_precision_std": std_row["best_precision"],
                "best_accuracy": mean_row["best_accuracy"],
                "best_accuracy_std": std_row["best_accuracy"],
                "best_betti_error_0": mean_row["best_betti_error_0"],
                "best_betti_error_0_std": std_row["best_betti_error_0"],
                "best_betti_error_1": mean_row["best_betti_error_1"],
                "best_betti_error_1_std": std_row["best_betti_error_1"],
                "train_elapsed_seconds": mean_row["train_elapsed_seconds"],
                "train_elapsed_hms": mean_row["train_elapsed_hms"],
                "train_elapsed_std_seconds": std_row["train_elapsed_seconds"],
                "train_elapsed_std_hms": std_row["train_elapsed_hms"],
            }
        )

        if len(target_roots) == 1:
            output_stem = args.output_stem
        else:
            output_stem = f"{root_output_label(root, args.input_root)}_{args.output_stem}"

        csv_out = os.path.join(dump_csv_dir, f"{output_stem}.csv")

        write_summary_csv(csv_out, fold_rows, mean_row, std_row)

    if len(target_roots) == 1:
        maybe_write_sample_visualization(
            dump_dir,
            args.output_dir,
            args.output_stem,
            root_summaries,
            args.sample_vis_count,
            args.max_vis_models,
        )

    if len(experiment_mean_rows) >= 1:
        unique_rows: Dict[Tuple, Dict[str, object]] = {}
        for row in experiment_mean_rows:
            key = _experiment_mean_unique_key(row)
            unique_rows[key] = row
        experiment_mean_rows = list(unique_rows.values())

        experiment_mean_rows = sorted(
            experiment_mean_rows,
            key=experiment_sort_key,
        )
        aggregate_rows = filter_rows_for_aggregate_scope(
            experiment_mean_rows,
            include_all=args.all,
        )
        if len(aggregate_rows) == 0:
            scope_label = "all counted datasets" if args.all else "default dataset scope"
            print(f"[WARN] No aggregate rows matched {scope_label}; skipping combined outputs.")
            return

        agg_stem = aggregate_output_stem(args.output_stem, include_all=args.all)
        agg_csv_out = os.path.join(dump_csv_dir, f"{agg_stem}.csv")
        agg_tex_out = os.path.join(dump_dir, f"{agg_stem}.tex")
        log_prefix = "[ALL]" if args.all else "[DEFAULT]"

        write_experiment_mean_csv(agg_csv_out, aggregate_rows)
        write_experiment_mean_latex(
            agg_tex_out,
            "Cross-experiment Mean Summary",
            aggregate_rows,
        )

        print(f"{log_prefix} CSV: {agg_csv_out}")
        print(f"{log_prefix} LaTeX: {agg_tex_out}")

        rows_by_dataset: Dict[str, List[Dict[str, object]]] = {}
        for row in aggregate_rows:
            dataset_name = str(row.get("dataset", "unknown_dataset"))
            if not _is_counted_dataset_name(dataset_name):
                continue
            rows_by_dataset.setdefault(dataset_name, []).append(row)

        for dataset_name, dataset_rows in sorted(
            rows_by_dataset.items(),
            key=lambda item: _dataset_group_sort_key(item[0]),
        ):
            dataset_suffix = sanitize_scope_name_for_filename(dataset_name)
            dataset_csv_out = os.path.join(dump_csv_dir, f"{agg_stem}_{dataset_suffix}.csv")
            write_experiment_mean_csv(dataset_csv_out, dataset_rows)
            print(f"{log_prefix}:{dataset_name} CSV: {dataset_csv_out}")

        if len(rows_by_dataset) >= 1:
            dataset_tex_out = os.path.join(dump_dir, f"{agg_stem}_datasets.tex")
            dataset_pdf_out = os.path.join(args.output_dir, f"{agg_stem}_datasets.pdf")

            write_experiment_mean_dataset_tables_latex(
                dataset_tex_out,
                "Cross-experiment Mean Summary by Dataset",
                rows_by_dataset,
            )
            build_pdf(dataset_tex_out, dataset_pdf_out)

            print(f"{log_prefix}:DATASETS LaTeX: {dataset_tex_out}")
            print(f"{log_prefix}:DATASETS PDF: {dataset_pdf_out}")

            ablation_tex_out = os.path.join(dump_dir, f"{agg_stem}_ablation.tex")
            ablation_pdf_out = os.path.join(args.output_dir, f"{agg_stem}_ablation.pdf")
            write_ablation_latex(ablation_tex_out, aggregate_rows)
            build_pdf(ablation_tex_out, ablation_pdf_out)

            print(f"{log_prefix}:ABLATION LaTeX: {ablation_tex_out}")
            print(f"{log_prefix}:ABLATION PDF: {ablation_pdf_out}")

            remove_legacy_duplicate_aggregate_pdfs(args.output_dir, agg_stem)


if __name__ == "__main__":
    main()
