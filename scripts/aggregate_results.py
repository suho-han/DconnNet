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


DIRECTION_GROUPING_SUFFIXES: List[Tuple[List[str], str]] = [
    (["coarse24to8"], "24to8"),
    (["24to8"], "24to8"),
]

DIRECTION_FUSION_SUFFIXES: List[Tuple[List[str], str]] = [
    (["attention", "gating"], "attention_gating"),
    (["weighted", "sum"], "weighted_sum"),
    (["conv1x1"], "conv1x1"),
    (["mean"], "mean"),
]


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
        "--sample-vis-count",
        type=int,
        default=2,
        help=(
            "Number of highest- and lowest-Dice samples to visualize from the "
            "reference model. Set to 0 to disable sample visualization export."
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

        if depth > 0:
            available_folds = discover_available_folds(current_dir, input_name)
            if available_folds:
                rel_parts = os.path.relpath(current_dir, input_root).split(os.sep)
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
        "loss": exp_meta["loss"],
        "direction_grouping": exp_meta["direction_grouping"],
        "direction_fusion": exp_meta["direction_fusion"],
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
    fields = [
        f"Exp={model['experiment']}",
        f"Conn={model['conn_num']}",
        f"Loss={model['loss']}",
    ]
    if str(model.get("direction_grouping", "none")) != "none":
        fields.append(f"Group={model.get('direction_grouping', 'none')}")
        fields.append(f"Fusion={model.get('direction_fusion', 'weighted_sum')}")
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
                f"model{idx}_direction_grouping",
                f"model{idx}_direction_fusion",
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
) -> None:
    if sample_vis_count <= 0:
        return

    reference_model, models = choose_visualization_models(root_summaries)
    if reference_model is None or not models:
        return
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
            render_row[f"model{idx}_conn_num"] = model["conn_num"]
            render_row[f"model{idx}_loss"] = str(model["loss"])
            render_row[f"model{idx}_direction_grouping"] = str(model.get("direction_grouping", "none"))
            render_row[f"model{idx}_direction_fusion"] = str(model.get("direction_fusion", "weighted_sum"))
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
    text = _fmt_float(value)
    if text == "nan":
        return text
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
    std_txt = f"{std:.4f}"
    res = rf"{mean_txt} \pm {std_txt}"
    if is_best:
        return rf"\textbf{{${res}$}}"
    if is_second:
        return rf"\underline{{${res}$}}"
    return rf"${res}$"


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


def _strip_direction_suffix_tokens(tokens: List[str]) -> Tuple[List[str], str, str]:
    direction_grouping = "none"
    direction_fusion = "weighted_sum"

    for fusion_suffix_tokens, fusion_name in DIRECTION_FUSION_SUFFIXES:
        if len(tokens) < len(fusion_suffix_tokens):
            continue
        if tokens[-len(fusion_suffix_tokens):] != fusion_suffix_tokens:
            continue

        prefix_tokens = tokens[:-len(fusion_suffix_tokens)]
        for grouping_suffix_tokens, grouping_name in DIRECTION_GROUPING_SUFFIXES:
            if len(prefix_tokens) < len(grouping_suffix_tokens):
                continue
            if prefix_tokens[-len(grouping_suffix_tokens):] != grouping_suffix_tokens:
                continue
            return (
                prefix_tokens[:-len(grouping_suffix_tokens)],
                grouping_name,
                fusion_name,
            )

    return tokens, direction_grouping, direction_fusion


def parse_experiment_metadata(root_name: str) -> Dict[str, object]:
    tokens = root_name.split("_")
    tokens, direction_grouping, direction_fusion = _strip_direction_suffix_tokens(tokens)
    loss = "unknown"

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

    conn_num: object = "NA"
    if tokens and tokens[-1].isdigit():
        conn_num = int(tokens[-1])
        tokens = tokens[:-1]

    experiment = "_".join(tokens) if tokens else root_name
    if loss == "unknown":
        loss = infer_loss_tag_from_legacy_name(root_name)
    if experiment == "binary":
        loss = "bce"

    return {
        "experiment": experiment,
        "conn_num": conn_num,
        "loss": loss,
        "direction_grouping": direction_grouping,
        "direction_fusion": direction_fusion,
    }


def experiment_sort_key(row: Dict[str, object]) -> Tuple[object, ...]:
    experiment = str(row["experiment"])
    exp_order = {
        "binary": 0,
        "dist": 1,
        "dist_inverted": 2,
    }.get(experiment, 9)

    conn_raw = row.get("conn_num", "NA")
    conn_sort = int(conn_raw) if isinstance(conn_raw, int) else 10 ** 9

    loss = str(row.get("loss", "Unknown"))
    loss_order = {
        "bce": 0,
        "smooth_l1": 1,
        "cl_dice": 2,
        "gjml_sf_l1": 3,
        "gjml_sj_l1": 3,
        "unknown": 9,
    }.get(loss, 9)

    direction_grouping = str(row.get("direction_grouping", "none"))
    direction_fusion = str(row.get("direction_fusion", "weighted_sum"))
    direction_grouping_order = {
        "none": 0,
        "24to8": 1,
    }.get(direction_grouping, 9)
    direction_fusion_order = {
        "weighted_sum": 0,
        "mean": 1,
        "conv1x1": 2,
        "attention_gating": 3,
    }.get(direction_fusion, 9)

    fold_scope = str(row.get("fold_scope", "direct"))
    fold_scope_count = row.get("fold_scope_count", "NA")
    fold_scope_sort = int(fold_scope_count) if isinstance(fold_scope_count, int) else 10 ** 9

    return (
        exp_order,
        natural_sort_key(experiment),
        conn_sort,
        loss_order,
        loss.lower(),
        direction_grouping_order,
        direction_grouping.lower(),
        direction_fusion_order,
        direction_fusion.lower(),
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
                "loss",
                "direction_grouping",
                "direction_fusion",
                "num_folds",
                "best_dice_mean",
                "best_jac_mean",
                "best_cldice_mean",
                "best_precision_mean",
                "best_accuracy_mean",
                "best_betti_error_0_mean",
                "best_betti_error_1_mean",
                "train_elapsed_mean_hms",
            ]
        )
        for row in rows:
            grouping = str(row.get("direction_grouping", "none"))
            fusion = str(row.get("direction_fusion", "weighted_sum"))
            if grouping == "none":
                grouping = "-"
                fusion = "-"
            writer.writerow(
                [
                    row["experiment"],
                    row["conn_num"],
                    row["loss"],
                    grouping,
                    fusion,
                    int(row["num_folds"]),
                    _fmt_float(row["best_dice"]),
                    _fmt_float(row["best_jac"]),
                    _fmt_float(row["best_cldice"]),
                    _fmt_float(row.get("best_precision", float("nan"))),
                    _fmt_float(row.get("best_accuracy", float("nan"))),
                    _fmt_float(row["best_betti_error_0"]),
                    _fmt_float(row["best_betti_error_1"]),
                    _fmt_duration_csv(row.get("train_elapsed_hms", "")),
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
        r"\usepackage[a4paper,margin=1in,landscape]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\begin{document}",
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
        r"label\_mode & Conn & Loss & Grouping & Fusion & \#Folds & Dice & Jac & clDice & Err $(\beta_0)$ & Err $(\beta_1)$ & Train Time \\",
        r"\midrule",
    ]

    for idx, row in enumerate(rows):
        dice_txt = _fmt_latex_ranked_value(float(row["best_dice"]), idx in dice_best, idx in dice_second)
        jac_txt = _fmt_latex_ranked_value(float(row["best_jac"]), idx in jac_best, idx in jac_second)
        cldice_txt = _fmt_latex_ranked_value(
            float(row["best_cldice"]),
            idx in cldice_best,
            idx in cldice_second,
        )
        betti0_txt = _fmt_latex_ranked_value(
            float(row["best_betti_error_0"]),
            idx in betti0_best,
            idx in betti0_second,
        )
        betti1_txt = _fmt_latex_ranked_value(
            float(row["best_betti_error_1"]),
            idx in betti1_best,
            idx in betti1_second,
        )
        time_txt = _fmt_duration_latex(row.get("train_elapsed_hms", ""))

        grouping = str(row.get("direction_grouping", "none"))
        fusion = str(row.get("direction_fusion", "weighted_sum"))
        if grouping == "none":
            grouping = "--"
            fusion = "--"

        lines.append(
            f'{escape_latex_text(str(row["experiment"]))} & {escape_latex_text(str(row["conn_num"]))} '
            f'& {escape_latex_text(str(row["loss"]))} '
            f'& {escape_latex_text(grouping)} '
            f'& {escape_latex_text(fusion)} '
            f'& {int(row["num_folds"])} & {dice_txt} & {jac_txt} & {cldice_txt} & {betti0_txt} & {betti1_txt} & {time_txt} \\\\'
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
        r"label\_mode & Conn & Loss & Grouping & Fusion & \#Folds & " + " & ".join(label for _, label, _ in metric_specs) + r" \\",
        r"\midrule",
    ]

    for idx, row in enumerate(rows):
        metric_cells: List[str] = []
        for key, _, _ in metric_specs:
            best_indices, second_indices = rank_map[key]
            metric_cells.append(
                _fmt_latex_ranked_value(
                    float(row.get(key, float("nan"))),
                    idx in best_indices,
                    idx in second_indices,
                )
            )

        grouping = str(row.get("direction_grouping", "none"))
        fusion = str(row.get("direction_fusion", "weighted_sum"))
        if grouping == "none":
            grouping = "--"
            fusion = "--"

        lines.append(
            f'{escape_latex_text(str(row["experiment"]))} & {escape_latex_text(str(row["conn_num"]))} '
            f'& {escape_latex_text(str(row["loss"]))} '
            f'& {escape_latex_text(grouping)} '
            f'& {escape_latex_text(fusion)} '
            f'& {int(row["num_folds"])} & ' + " & ".join(metric_cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]


def write_experiment_mean_latex(path: str, title: str, rows: List[Dict[str, object]]) -> None:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=1in,landscape]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\begin{document}",
    ]

    rows_by_dataset: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        dataset_name = str(row.get("dataset", "unknown_dataset"))
        rows_by_dataset.setdefault(dataset_name, []).append(row)

    if len(rows_by_dataset) == 1:
        dataset_name, dataset_rows = next(iter(rows_by_dataset.items()))
        _append_dataset_mean_table_lines(lines, dataset_rows, dataset_name)
        lines += [
            rf"\caption{{Cross-experiment mean summary ({escape_latex_text(dataset_name)})}}",
            r"\end{table}",
        ]
    else:
        for dataset_name, dataset_rows in sorted(rows_by_dataset.items(), key=lambda item: natural_sort_key(item[0])):
            if len(dataset_rows) == 0:
                continue
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
        datasets.add(str(row.get("dataset", "unknown")))
    dataset_list = sorted(list(datasets), key=natural_sort_key)

    variable_groups = [
        {
            "name": "Label Mode",
            "headers": ["Label Mode"],
            "func": lambda r: (str(r.get("experiment", "unknown")),),
        },
        {
            "name": "Loss",
            "headers": ["Loss"],
            "func": lambda r: (str(r.get("loss", "unknown")),),
            "filter": lambda r: str(r.get("loss", "unknown")) != "bce",
        },
        {
            "name": "Connectivity and Grouping-Fusion",
            "headers": ["Connectivity", "Grouping -- Fusion"],
            "func": lambda r: (
                str(r.get("conn_num", "unknown")),
                "--" if str(r.get("direction_grouping", "none")) == "none" else f"{r.get('direction_grouping')} -- {r.get('direction_fusion')}"
            ),
        }
    ]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=1in,landscape]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\usepackage{multirow}",
        r"\begin{document}",
        r"\section*{Ablation Studies (Marginal Averages)}",
    ]

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
            return [natural_sort_key(str(x)) for x in tup]

        val_list = sorted(list(val_set), key=tuple_sort_key)

        stats = {ds: {v: {"dice": [], "iou": []} for v in val_list} for ds in dataset_list}

        for row in rows:
            if not row_filter(row):
                continue
            ds = str(row.get("dataset", "unknown"))
            val = var_func(row)
            dice = float(row.get("best_dice", float("nan")))
            iou = float(row.get("best_jac", float("nan")))
            if not math.isnan(dice):
                stats[ds][val]["dice"].append(dice)
            if not math.isnan(iou):
                stats[ds][val]["iou"].append(iou)

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

        avg_stats = {
            ds: {
                v: {
                    "dice_mean": float("nan"),
                    "dice_std": float("nan"),
                    "iou_mean": float("nan"),
                    "iou_std": float("nan"),
                }
                for v in val_list
            }
            for ds in dataset_list
        }
        dataset_metrics = {ds: {"dice": [], "iou": []} for ds in dataset_list}

        for ds in dataset_list:
            for v in val_list:
                dice_list = stats[ds][v]["dice"]
                iou_list = stats[ds][v]["iou"]
                avg_dice = _safe_mean(dice_list)
                avg_iou = _safe_mean(iou_list)
                std_dice = _safe_std(dice_list, avg_dice)
                std_iou = _safe_std(iou_list, avg_iou)
                
                avg_stats[ds][v]["dice_mean"] = avg_dice
                avg_stats[ds][v]["dice_std"] = std_dice
                avg_stats[ds][v]["iou_mean"] = avg_iou
                avg_stats[ds][v]["iou_std"] = std_iou
                
                dataset_metrics[ds]["dice"].append(avg_dice)
                dataset_metrics[ds]["iou"].append(avg_iou)

        dataset_ranks = {}
        for ds in dataset_list:
            dice_b, dice_s = get_ranks(dataset_metrics[ds]["dice"], True)
            iou_b, iou_s = get_ranks(dataset_metrics[ds]["iou"], True)
            dataset_ranks[ds] = {"dice_best": dice_b, "dice_second": dice_s, "iou_best": iou_b, "iou_second": iou_s}

        lines.extend([
            r"\begin{table}[H]",
            r"\centering",
        ])

        num_var_cols = len(headers)
        col_str = "l" * num_var_cols + "cc" * len(dataset_list)
        lines.append(rf"\begin{{tabular}}{{{col_str}}}")
        lines.append(r"\toprule")

        header1 = [rf"\multirow{{2}}{{*}}{{{escape_latex_text(h)}}}" for h in headers]
        for ds in dataset_list:
            header1.append(rf"\multicolumn{{2}}{{c}}{{{escape_latex_text(ds)}}}")
        lines.append(" & ".join(header1) + r" \\")

        cmidrules = []
        current_col = num_var_cols + 1
        for ds in dataset_list:
            cmidrules.append(rf"\cmidrule(lr){{{current_col}-{current_col+1}}}")
            current_col += 2
        if cmidrules:
            lines.append(" ".join(cmidrules))

        header2 = [""] * num_var_cols
        for ds in dataset_list:
            header2.extend(["Dice", "IoU"])
        lines.append(" & ".join(header2) + r" \\")
        lines.append(r"\midrule")

        for idx, v in enumerate(val_list):
            row_str_parts = [escape_latex_text(str(x)) for x in v]
            for ds in dataset_list:
                dice_mean = avg_stats[ds][v]["dice_mean"]
                dice_std = avg_stats[ds][v]["dice_std"]
                iou_mean = avg_stats[ds][v]["iou_mean"]
                iou_std = avg_stats[ds][v]["iou_std"]

                dice_txt = _fmt_latex_ranked_mean_std_marginal(
                    dice_mean, dice_std, idx in dataset_ranks[ds]["dice_best"], idx in dataset_ranks[ds]["dice_second"]
                )
                iou_txt = _fmt_latex_ranked_mean_std_marginal(
                    iou_mean, iou_std, idx in dataset_ranks[ds]["iou_best"], idx in dataset_ranks[ds]["iou_second"]
                )

                row_str_parts.append(dice_txt)
                row_str_parts.append(iou_txt)
            lines.append(" & ".join(row_str_parts) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{Ablation on {escape_latex_text(var_name)}}}",
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
        r"\usepackage[a4paper,margin=1in,landscape]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\begin{document}",
    ]

    for dataset_name, rows in sorted(rows_by_dataset.items(), key=lambda item: natural_sort_key(item[0])):
        if len(rows) == 0:
            continue
        _append_dataset_mean_table_lines(lines, rows, dataset_name)
        lines.extend([
            rf"\caption{{Cross-experiment mean summary ({escape_latex_text(dataset_name)})}}",
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


def main() -> None:
    args = parse_args()
    requested_folds = [item.strip() for item in args.folds.split(",") if item.strip()]
    if not requested_folds:
        raise ValueError("No folds were provided.")

    os.makedirs(args.output_dir, exist_ok=True)
    dump_dir = os.path.join(args.output_dir, "dump")
    os.makedirs(dump_dir, exist_ok=True)
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
        experiment_mean_rows.append(
            {
                "dataset": extract_dataset_name(root, args.input_root),
                "experiment": exp_meta["experiment"],
                "conn_num": exp_meta["conn_num"],
                "loss": exp_meta["loss"],
                "direction_grouping": exp_meta["direction_grouping"],
                "direction_fusion": exp_meta["direction_fusion"],
                "fold_scope": scope_name,
                "fold_scope_count": scope_fold_count,
                "num_folds": float(len(folds)),
                "best_dice": mean_row["best_dice"],
                "best_jac": mean_row["best_jac"],
                "best_cldice": mean_row["best_cldice"],
                "best_precision": mean_row["best_precision"],
                "best_accuracy": mean_row["best_accuracy"],
                "best_betti_error_0": mean_row["best_betti_error_0"],
                "best_betti_error_1": mean_row["best_betti_error_1"],
                "train_elapsed_seconds": mean_row["train_elapsed_seconds"],
                "train_elapsed_hms": mean_row["train_elapsed_hms"],
            }
        )

        if len(target_roots) == 1:
            output_stem = args.output_stem
            title = "Final Metrics"
        else:
            output_stem = f"{root_output_label(root, args.input_root)}_{args.output_stem}"
            title = f"Final Metrics ({escape_latex_text(display_name)})"
        dataset_name = extract_dataset_name(root, args.input_root)

        csv_out = os.path.join(dump_dir, f"{output_stem}.csv")
        tex_out = os.path.join(dump_dir, f"{output_stem}.tex")
        pdf_out = os.path.join(args.output_dir, f"{output_stem}.pdf")

        write_summary_csv(csv_out, fold_rows, mean_row, std_row)
        write_latex(tex_out, title, fold_rows, mean_row, std_row, dataset_name)
        if len(target_roots) == 1:
            build_pdf(tex_out, pdf_out)

    maybe_write_sample_visualization(
        dump_dir,
        args.output_dir,
        args.output_stem,
        root_summaries,
        args.sample_vis_count,
    )

    if len(experiment_mean_rows) >= 1:
        unique_rows: Dict[Tuple, Dict[str, object]] = {}
        for row in experiment_mean_rows:
            key = (
                row["dataset"],
                row["experiment"],
                row["conn_num"],
                row["loss"],
                row["direction_grouping"],
                row["direction_fusion"],
                row["fold_scope"],
            )
            unique_rows[key] = row
        experiment_mean_rows = list(unique_rows.values())

        experiment_mean_rows = sorted(
            experiment_mean_rows,
            key=experiment_sort_key,
        )
        agg_stem = f"{args.output_stem}_experiment_means"
        agg_csv_out = os.path.join(dump_dir, f"{agg_stem}.csv")
        agg_tex_out = os.path.join(dump_dir, f"{agg_stem}.tex")

        write_experiment_mean_csv(agg_csv_out, experiment_mean_rows)
        write_experiment_mean_latex(
            agg_tex_out,
            "Cross-experiment Mean Summary",
            experiment_mean_rows,
        )

        print(f"[ALL] CSV: {agg_csv_out}")
        print(f"[ALL] LaTeX: {agg_tex_out}")

        rows_by_dataset: Dict[str, List[Dict[str, object]]] = {}
        for row in experiment_mean_rows:
            dataset_name = str(row.get("dataset", "unknown_dataset"))
            rows_by_dataset.setdefault(dataset_name, []).append(row)

        for dataset_name, dataset_rows in sorted(
            rows_by_dataset.items(),
            key=lambda item: natural_sort_key(item[0]),
        ):
            dataset_suffix = sanitize_scope_name_for_filename(dataset_name)
            dataset_csv_out = os.path.join(dump_dir, f"{agg_stem}_{dataset_suffix}.csv")
            dataset_csv_summary_out = os.path.join(args.output_dir, f"{agg_stem}_{dataset_suffix}.csv")
            write_experiment_mean_csv(dataset_csv_out, dataset_rows)
            write_experiment_mean_csv(dataset_csv_summary_out, dataset_rows)
            print(f"[ALL:{dataset_name}] CSV: {dataset_csv_out}")
            print(f"[ALL:{dataset_name}] CSV (summary): {dataset_csv_summary_out}")

        if len(rows_by_dataset) >= 1:
            dataset_tex_out = os.path.join(dump_dir, f"{agg_stem}_datasets.tex")
            dataset_pdf_out = os.path.join(args.output_dir, f"{agg_stem}_datasets.pdf")

            write_experiment_mean_dataset_tables_latex(
                dataset_tex_out,
                "Cross-experiment Mean Summary by Dataset",
                rows_by_dataset,
            )
            build_pdf(dataset_tex_out, dataset_pdf_out)

            print(f"[ALL:DATASETS] LaTeX: {dataset_tex_out}")
            print(f"[ALL:DATASETS] PDF: {dataset_pdf_out}")

            ablation_tex_out = os.path.join(dump_dir, f"{agg_stem}_ablation.tex")
            ablation_pdf_out = os.path.join(args.output_dir, f"{agg_stem}_ablation.pdf")
            write_ablation_latex(ablation_tex_out, experiment_mean_rows)
            build_pdf(ablation_tex_out, ablation_pdf_out)

            print(f"[ALL:ABLATION] LaTeX: {ablation_tex_out}")
            print(f"[ALL:ABLATION] PDF: {ablation_pdf_out}")


if __name__ == "__main__":
    main()
