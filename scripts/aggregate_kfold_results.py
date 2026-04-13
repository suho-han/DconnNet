#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import subprocess
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate K-fold DconnNet results and export CSV/LaTeX/PDF."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="output",
        help=(
            "Fold root directory. Supported layouts: "
            "<input-root>/<fold>/<input-name> or <input-root>/results_<fold>.csv"
        ),
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated fold ids to aggregate (example: 1,2,3,4,5).",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="results.csv",
        help="Input CSV file name in each fold directory.",
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
        default="kfold_summary",
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


def parse_fold_csv(path: str) -> Dict[str, float]:
    epoch_rows: Dict[int, Tuple[float, float, float]] = {}
    summary_epoch: Optional[int] = None
    summary_dice: Optional[float] = None
    use_extended_epoch_format = False

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header_skipped = False
        for row in reader:
            row = [item.strip() for item in row if item.strip() != ""]
            if not row:
                continue
            if not header_skipped and row[0].lower() == "epoch":
                lowered = [item.lower() for item in row]
                use_extended_epoch_format = (
                    "train_loss" in lowered and
                    "val_loss" in lowered and
                    "dice" in lowered
                )
                header_skipped = True
                continue

            if len(row) >= 4:
                epoch = int(row[0])
                if use_extended_epoch_format and len(row) >= 6:
                    dice = _to_float(row[3])
                    jac = _to_float(row[4])
                    cldice = _to_float(row[5])
                else:
                    dice = _to_float(row[1])
                    jac = _to_float(row[2])
                    cldice = _to_float(row[3])
                epoch_rows[epoch] = (dice, jac, cldice)
            elif len(row) >= 2:
                summary_epoch = int(row[0])
                summary_dice = _to_float(row[1])

    if not epoch_rows:
        raise ValueError(f"No epoch rows found in: {path}")

    if summary_epoch is None or summary_dice is None:
        best_epoch, (best_dice, best_jac, best_cldice) = max(
            epoch_rows.items(), key=lambda kv: kv[1][0]
        )
    else:
        best_epoch = summary_epoch
        best_dice = summary_dice
        if best_epoch in epoch_rows:
            _, best_jac, best_cldice = epoch_rows[best_epoch]
        else:
            best_jac, best_cldice = float("nan"), float("nan")

    return {
        "best_epoch": float(best_epoch),
        "best_dice": best_dice,
        "best_jac": best_jac,
        "best_cldice": best_cldice,
    }


def resolve_fold_csv_path(input_root: str, fold: str, input_name: str) -> str:
    default_path = os.path.join(input_root, fold, input_name)
    if os.path.isfile(default_path):
        return default_path

    candidates = []
    if "{fold}" in input_name:
        candidates.append(os.path.join(input_root, input_name.format(fold=fold)))
    candidates.append(os.path.join(input_root, f"results_{fold}.csv"))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    checked = [default_path] + candidates
    raise FileNotFoundError(
        "Missing fold CSV for fold "
        f"{fold}. Checked: {', '.join(checked)}"
    )


def discover_available_folds(input_root: str, input_name: str) -> List[str]:
    discovered = set()
    if not os.path.isdir(input_root):
        return []

    for entry in os.listdir(input_root):
        entry_path = os.path.join(input_root, entry)
        if entry.isdigit() and os.path.isfile(os.path.join(entry_path, input_name)):
            discovered.add(entry)

    for entry in os.listdir(input_root):
        if not entry.startswith("results_") or not entry.endswith(".csv"):
            continue
        fold = entry[len("results_"):-len(".csv")]
        if fold.isdigit():
            discovered.add(fold)

    return sorted(discovered, key=int)


def discover_target_roots(input_root: str, folds: List[str], input_name: str) -> List[str]:
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input root directory not found: {input_root}")

    available_in_root = discover_available_folds(input_root, input_name)
    if available_in_root:
        return [input_root]

    targets: List[str] = []
    for entry in sorted(os.listdir(input_root)):
        entry_path = os.path.join(input_root, entry)
        if not os.path.isdir(entry_path):
            continue
        if discover_available_folds(entry_path, input_name):
            targets.append(entry_path)

    if targets:
        return targets

    raise FileNotFoundError(
        f"No fold CSV files were found in '{input_root}' or its direct subdirectories."
    )


def resolve_folds_for_root(input_root: str, requested_folds: List[str], input_name: str) -> List[str]:
    missing = []
    for fold in requested_folds:
        try:
            resolve_fold_csv_path(input_root, fold, input_name)
        except FileNotFoundError:
            missing.append(fold)

    if not missing:
        return requested_folds

    auto_folds = discover_available_folds(input_root, input_name)
    if not auto_folds:
        raise FileNotFoundError(
            f"No usable folds in '{input_root}'. Missing requested folds: {', '.join(missing)}"
        )
    print(
        f"[WARN] {input_root}: missing requested folds ({', '.join(missing)}). "
        f"Using detected folds: {', '.join(auto_folds)}"
    )
    return auto_folds


def aggregate_root(input_root: str, folds: List[str], input_name: str) -> Tuple[List[Dict[str, float]], Dict[str, float], Dict[str, float]]:
    fold_rows: List[Dict[str, float]] = []
    for fold in folds:
        csv_path = resolve_fold_csv_path(input_root, fold, input_name)
        metrics = parse_fold_csv(csv_path)
        metrics["fold"] = float(fold)
        fold_rows.append(metrics)

    epochs = [row["best_epoch"] for row in fold_rows]
    dices = [row["best_dice"] for row in fold_rows]
    jacs = [row["best_jac"] for row in fold_rows]
    cldices = [row["best_cldice"] for row in fold_rows]

    mean_row = {
        "best_epoch": _safe_mean(epochs),
        "best_dice": _safe_mean(dices),
        "best_jac": _safe_mean(jacs),
        "best_cldice": _safe_mean(cldices),
    }
    std_row = {
        "best_epoch": _safe_std(epochs, mean_row["best_epoch"]),
        "best_dice": _safe_std(dices, mean_row["best_dice"]),
        "best_jac": _safe_std(jacs, mean_row["best_jac"]),
        "best_cldice": _safe_std(cldices, mean_row["best_cldice"]),
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


def find_fold_row(fold_rows: List[Dict[str, float]], fold: int) -> Optional[Dict[str, float]]:
    for row in fold_rows:
        if int(row["fold"]) == int(fold):
            return row
    return None


def build_model_candidate(summary: Dict[str, object], fold_row: Dict[str, float]) -> Dict[str, object]:
    fold = int(fold_row["fold"])
    model_dir = os.path.join(str(summary["root"]), "models", str(fold))
    sample_csv = os.path.join(model_dir, "test_sample_metrics.csv")
    sample_rows = parse_sample_metrics_csv(sample_csv) if os.path.isfile(sample_csv) else []
    best_epoch = int(fold_row["best_epoch"])
    vis_epoch, is_exact = resolve_visualization_epoch(model_dir, best_epoch)

    return {
        "root": summary["root"],
        "root_name": summary["root_name"],
        "fold": fold,
        "best_epoch": best_epoch,
        "best_dice": float(fold_row["best_dice"]),
        "model_dir": model_dir,
        "sample_csv": sample_csv,
        "sample_rows": sample_rows,
        "vis_epoch": vis_epoch,
        "vis_epoch_exact": is_exact,
    }


def choose_visualization_models(root_summaries: List[Dict[str, object]]) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    if not root_summaries:
        return None, None

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
    model1 = build_model_candidate(reference_summary, reference_fold_row)

    model2: Optional[Dict[str, object]] = None
    if len(ranked_summaries) >= 2:
        compare_summary = ranked_summaries[1]
        compare_fold_row = find_fold_row(compare_summary["fold_rows"], int(reference_fold_row["fold"]))
        if compare_fold_row is None:
            compare_fold_row = max(
                compare_summary["fold_rows"],
                key=lambda row: float(row["best_dice"]),
            )
            print(
                f"[WARN] {compare_summary['root_name']}: fold {int(reference_fold_row['fold'])} "
                "not found for sample comparison; using its best fold instead."
            )
        model2 = build_model_candidate(compare_summary, compare_fold_row)
    else:
        ranked_folds = sorted(
            reference_summary["fold_rows"],
            key=lambda row: float(row["best_dice"]),
            reverse=True,
        )
        if len(ranked_folds) >= 2:
            compare_fold_row = ranked_folds[1]
            model2 = build_model_candidate(reference_summary, compare_fold_row)
            print(
                "[WARN] Only one experiment root detected; "
                "Model 2 uses the second-best fold from the same experiment."
            )

    return model1, model2


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
            # Keep input images in RGB to avoid OpenCV-style BGR confusion.
            image = image.convert("RGB")
            return np.asarray(image)

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


def write_sample_visualization_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "sample_rank",
        "sample_group",
        "sample_name",
        "reference_dice",
        "reference_jac",
        "reference_cldice",
        "image_path",
        "gt_path",
        "model1_name",
        "model1_fold",
        "model1_best_epoch",
        "model1_vis_epoch",
        "model1_vis_epoch_exact",
        "model1_dice",
        "model1_pred_path",
        "model2_name",
        "model2_fold",
        "model2_best_epoch",
        "model2_vis_epoch",
        "model2_vis_epoch_exact",
        "model2_dice",
        "model2_pred_path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_sample_visualization(
    output_dir: str,
    output_stem: str,
    root_summaries: List[Dict[str, object]],
    sample_vis_count: int,
) -> None:
    if sample_vis_count <= 0:
        return

    model1, model2 = choose_visualization_models(root_summaries)
    if model1 is None:
        return
    if model1["vis_epoch"] is None:
        print(f"[WARN] {model1['model_dir']}: no checkpoint_batches found; skipping sample visualization.")
        return
    if not model1["sample_rows"]:
        print(f"[WARN] {model1['sample_csv']}: missing sample metrics; skipping sample visualization.")
        return

    reference_rows = filter_sample_rows_for_epoch(model1["sample_rows"], int(model1["best_epoch"]))
    if not reference_rows:
        print(
            f"[WARN] {model1['sample_csv']}: no rows for best epoch {model1['best_epoch']}; "
            "skipping sample visualization."
        )
        return

    selected_rows = select_ranked_samples(reference_rows, sample_vis_count)
    if not selected_rows:
        print("[WARN] No sample rows available for visualization.")
        return

    if not bool(model1["vis_epoch_exact"]):
        print(
            f"[WARN] {model1['root_name']} fold {model1['fold']}: "
            f"best epoch {model1['best_epoch']} has no saved PNG; using nearest checkpoint epoch {model1['vis_epoch']}."
        )
    if model2 is not None and model2["vis_epoch"] is not None and not bool(model2["vis_epoch_exact"]):
        print(
            f"[WARN] {model2['root_name']} fold {model2['fold']}: "
            f"best epoch {model2['best_epoch']} has no saved PNG; using nearest checkpoint epoch {model2['vis_epoch']}."
        )

    render_rows: List[Dict[str, object]] = []
    for row in selected_rows:
        sample_name = str(row["sample_name"])
        batch_idx = int(row["batch"])
        image_path = checkpoint_image_path(str(model1["model_dir"]), model1["vis_epoch"], "image", batch_idx)
        gt_path = checkpoint_image_path(str(model1["model_dir"]), model1["vis_epoch"], "mask", batch_idx)
        model1_pred_path = checkpoint_image_path(str(model1["model_dir"]), model1["vis_epoch"], "pred", batch_idx)

        model2_row = None
        model2_pred_path = None
        model2_dice = float("nan")
        model2_name = ""
        model2_fold = ""
        model2_best_epoch = ""
        model2_vis_epoch = ""
        model2_vis_epoch_exact = ""

        if model2 is not None and model2["vis_epoch"] is not None and model2["sample_rows"]:
            model2_row = find_sample_row(model2["sample_rows"], int(model2["best_epoch"]), sample_name)
            model2_name = str(model2["root_name"])
            model2_fold = int(model2["fold"])
            model2_best_epoch = int(model2["best_epoch"])
            model2_vis_epoch = int(model2["vis_epoch"])
            model2_vis_epoch_exact = bool(model2["vis_epoch_exact"])
            if model2_row is not None:
                model2_dice = float(model2_row["dice"])
                model2_pred_path = checkpoint_image_path(
                    str(model2["model_dir"]),
                    model2["vis_epoch"],
                    "pred",
                    int(model2_row["batch"]),
                )

        render_rows.append(
            {
                "sample_rank": int(row["sample_rank"]),
                "sample_group": str(row["sample_group"]),
                "sample_name": sample_name,
                "reference_dice": float(row["dice"]),
                "reference_jac": float(row["jac"]),
                "reference_cldice": float(row["cldice"]),
                "image_path": image_path or "",
                "gt_path": gt_path or "",
                "model1_name": str(model1["root_name"]),
                "model1_fold": int(model1["fold"]),
                "model1_best_epoch": int(model1["best_epoch"]),
                "model1_vis_epoch": int(model1["vis_epoch"]),
                "model1_vis_epoch_exact": bool(model1["vis_epoch_exact"]),
                "model1_dice": float(row["dice"]),
                "model1_pred_path": model1_pred_path or "",
                "model2_name": model2_name,
                "model2_fold": model2_fold,
                "model2_best_epoch": model2_best_epoch,
                "model2_vis_epoch": model2_vis_epoch,
                "model2_vis_epoch_exact": model2_vis_epoch_exact,
                "model2_dice": model2_dice,
                "model2_pred_path": model2_pred_path or "",
            }
        )

    vis_csv_out = os.path.join(output_dir, f"{output_stem}_sample_visualization.csv")
    vis_png_out = os.path.join(output_dir, f"{output_stem}_sample_visualization.png")
    write_sample_visualization_csv(vis_csv_out, render_rows)

    fig, axes = plt.subplots(
        nrows=len(render_rows),
        ncols=4,
        figsize=(16, max(4, len(render_rows) * 4)),
        squeeze=False,
    )

    model1_title = "Model 1"
    model2_title = "Model 2"
    figure_title = (
        f"Reference: {model1['root_name']} / fold {model1['fold']} / best epoch {model1['best_epoch']} "
        f"(viz epoch {model1['vis_epoch']})"
    )
    if model2 is not None:
        figure_title += (
            f"\nCompare: {model2['root_name']} / fold {model2['fold']} / best epoch {model2['best_epoch']}"
            f" (viz epoch {model2['vis_epoch']})"
        )

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
        render_panel(axes[row_idx][2], row["model1_pred_path"] or None, model1_title, panel_kind="pred")
        render_panel(axes[row_idx][3], row["model2_pred_path"] or None, model2_title, panel_kind="pred")

    fig.suptitle(figure_title, fontsize=12)
    plt.tight_layout(rect=(0.03, 0.03, 1.0, 0.93))
    fig.savefig(vis_png_out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAMPLE_VIS] CSV: {vis_csv_out}")
    print(f"[SAMPLE_VIS] PNG: {vis_png_out}")


def write_summary_csv(path: str, fold_rows: List[Dict[str, float]], mean_row: Dict[str, float], std_row: Dict[str, float]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_epoch", "best_dice",
                        "best_jac", "best_cldice"])
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
            ]
        )


def _fmt_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def infer_loss_label(experiment: str) -> str:
    name = experiment.lower()
    if "_gjml_sf_l1" in name:
        return "GJML+SF-L1"
    if "_gjml_sj_l1" in name:
        return "GJML+SJ-L1"
    if "dist" in name:
        return "Smooth-L1"
    if "binary" in name:
        return "BCE"
    return "Unknown"


def _rank_row_indices(rows: List[Dict[str, object]], key: str) -> Tuple[Set[int], Set[int]]:
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

    unique_desc = sorted({value for _, value in indexed_values}, reverse=True)
    best_value = unique_desc[0]
    second_value = unique_desc[1] if len(unique_desc) > 1 else None

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


def parse_experiment_metadata(root_name: str) -> Dict[str, object]:
    raw = root_name
    lowered = raw.lower()
    loss = infer_loss_label(lowered)

    # Keep experiment name clean by removing known loss suffix tags first.
    for suffix in ("_gjml_sf_l1", "_gjml_sj_l1"):
        if lowered.endswith(suffix):
            raw = raw[: -len(suffix)]
            lowered = raw.lower()
            break

    conn_num: object = "NA"
    match = re.search(r"_(\d+)$", raw)
    if match is not None:
        conn_num = int(match.group(1))
        experiment = raw[:match.start()]
    else:
        experiment = raw

    return {
        "experiment": experiment,
        "conn_num": conn_num,
        "loss": loss,
    }


def experiment_sort_key(row: Dict[str, object]) -> Tuple[object, ...]:
    experiment = str(row["experiment"])
    exp_order = {
        "binary": 0,
        "dist_signed": 1,
        "dist_inverted": 2,
    }.get(experiment, 9)

    conn_raw = row.get("conn_num", "NA")
    conn_sort = int(conn_raw) if isinstance(conn_raw, int) else 10 ** 9

    loss = str(row.get("loss", "Unknown"))
    loss_order = {
        "BCE": 0,
        "Smooth-L1": 1,
        "GJML+SF-L1": 2,
        "GJML+SJ-L1": 2,
        "Unknown": 9,
    }.get(loss, 9)

    return (exp_order, natural_sort_key(experiment), conn_sort, loss_order, loss.lower())


def write_experiment_mean_csv(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "conn_num",
                "loss",
                "num_folds",
                "best_dice_mean",
                "best_jac_mean",
                "best_cldice_mean",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["experiment"],
                    row["conn_num"],
                    row["loss"],
                    int(row["num_folds"]),
                    _fmt_float(row["best_dice"]),
                    _fmt_float(row["best_jac"]),
                    _fmt_float(row["best_cldice"]),
                ]
            )


def write_latex(path: str, title: str, fold_rows: List[Dict[str, float]], mean_row: Dict[str, float], std_row: Dict[str, float]) -> None:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\begin{document}",
        rf"\section*{{{title}}}",
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Fold & Best Epoch & Dice & Jac & clDice \\",
        r"\midrule",
    ]

    for row in fold_rows:
        jac_txt = f'{row["best_jac"]:.4f}' if not math.isnan(
            row["best_jac"]) else "nan"
        cld_txt = f'{row["best_cldice"]:.4f}' if not math.isnan(
            row["best_cldice"]) else "nan"
        lines.append(
            f'{int(row["fold"])} & {int(row["best_epoch"])} & {row["best_dice"]:.4f} & {jac_txt} & {cld_txt} \\\\'
        )

    mean_jac = f'{mean_row["best_jac"]:.4f}' if not math.isnan(
        mean_row["best_jac"]) else "nan"
    mean_cld = f'{mean_row["best_cldice"]:.4f}' if not math.isnan(
        mean_row["best_cldice"]) else "nan"
    std_jac = f'{std_row["best_jac"]:.4f}' if not math.isnan(
        std_row["best_jac"]) else "nan"
    std_cld = f'{std_row["best_cldice"]:.4f}' if not math.isnan(
        std_row["best_cldice"]) else "nan"
    lines += [
        r"\midrule",
        f'Mean & {mean_row["best_epoch"]:.4f} & {mean_row["best_dice"]:.4f} & {mean_jac} & {mean_cld} \\\\',
        f'Std & {std_row["best_epoch"]:.4f} & {std_row["best_dice"]:.4f} & {std_jac} & {std_cld} \\\\',
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{K-fold final result summary}",
        r"\end{table}",
        r"\end{document}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_experiment_mean_latex(path: str, title: str, rows: List[Dict[str, object]]) -> None:
    dice_best, dice_second = _rank_row_indices(rows, "best_dice")
    jac_best, jac_second = _rank_row_indices(rows, "best_jac")
    cldice_best, cldice_second = _rank_row_indices(rows, "best_cldice")

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\begin{document}",
        rf"\section*{{{title}}}",
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Experiment & Conn & Loss & \#Folds & Dice (Mean) & Jac (Mean) & clDice (Mean) \\",
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
        lines.append(
            f'{escape_latex_text(str(row["experiment"]))} & {escape_latex_text(str(row["conn_num"]))} '
            f'& {escape_latex_text(str(row["loss"]))} & {int(row["num_folds"])} & {dice_txt} '
            f'& {jac_txt} & {cldice_txt} \\\\'
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Cross-experiment mean summary from k-fold runs}",
        r"\end{table}",
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
    return escaped


def build_pdf(tex_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(tex_path))
    tex_file = os.path.basename(tex_path)
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_file,
    ]
    subprocess.run(cmd, cwd=out_dir, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def main() -> None:
    args = parse_args()
    requested_folds = [item.strip() for item in args.folds.split(",") if item.strip()]
    if not requested_folds:
        raise ValueError("No folds were provided.")

    os.makedirs(args.output_dir, exist_ok=True)
    target_roots = discover_target_roots(args.input_root, requested_folds, args.input_name)
    experiment_mean_rows: List[Dict[str, object]] = []
    root_summaries: List[Dict[str, object]] = []

    for root in target_roots:
        folds = resolve_folds_for_root(root, requested_folds, args.input_name)
        fold_rows, mean_row, std_row = aggregate_root(root, folds, args.input_name)
        root_name = os.path.basename(os.path.normpath(root))
        root_summaries.append(
            {
                "root": root,
                "root_name": root_name,
                "folds": folds,
                "fold_rows": fold_rows,
                "mean_row": mean_row,
                "std_row": std_row,
            }
        )
        exp_meta = parse_experiment_metadata(root_name)
        experiment_mean_rows.append(
            {
                "experiment": exp_meta["experiment"],
                "conn_num": exp_meta["conn_num"],
                "loss": exp_meta["loss"],
                "num_folds": float(len(folds)),
                "best_dice": mean_row["best_dice"],
                "best_jac": mean_row["best_jac"],
                "best_cldice": mean_row["best_cldice"],
            }
        )

        if len(target_roots) == 1:
            output_stem = args.output_stem
            title = "K-fold Final Metrics"
        else:
            output_stem = f"{root_name}_{args.output_stem}"
            title = f"K-fold Final Metrics ({escape_latex_text(root_name)})"

        csv_out = os.path.join(args.output_dir, f"{output_stem}.csv")
        tex_out = os.path.join(args.output_dir, f"{output_stem}.tex")
        pdf_out = os.path.join(args.output_dir, f"{output_stem}.pdf")

        write_summary_csv(csv_out, fold_rows, mean_row, std_row)
        write_latex(tex_out, title, fold_rows, mean_row, std_row)
        build_pdf(tex_out)

        print(f"[{root}] CSV: {csv_out}")
        print(f"[{root}] LaTeX: {tex_out}")
        print(f"[{root}] PDF: {pdf_out}")

    maybe_write_sample_visualization(
        args.output_dir,
        args.output_stem,
        root_summaries,
        args.sample_vis_count,
    )

    if len(experiment_mean_rows) > 1:
        experiment_mean_rows = sorted(
            experiment_mean_rows,
            key=experiment_sort_key,
        )
        agg_stem = f"{args.output_stem}_experiment_means"
        agg_csv_out = os.path.join(args.output_dir, f"{agg_stem}.csv")
        agg_tex_out = os.path.join(args.output_dir, f"{agg_stem}.tex")
        agg_pdf_out = os.path.join(args.output_dir, f"{agg_stem}.pdf")

        write_experiment_mean_csv(agg_csv_out, experiment_mean_rows)
        write_experiment_mean_latex(
            agg_tex_out,
            "Cross-experiment Mean Summary",
            experiment_mean_rows,
        )
        build_pdf(agg_tex_out)

        print(f"[ALL] CSV: {agg_csv_out}")
        print(f"[ALL] LaTeX: {agg_tex_out}")
        print(f"[ALL] PDF: {agg_pdf_out}")


if __name__ == "__main__":
    main()
