#!/usr/bin/env python3
"""Config-first train launcher.

Usage:
  scripts/train_launcher.sh --config <yaml> [--device N] [--batch-size N] [--dry_run] [--smoke]
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

CANONICAL_8_OFFSETS = [
    (1, 1),
    (1, 0),
    (1, -1),
    (0, 1),
    (0, -1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
]

OUTER_8_OFFSETS = [
    (-2, -2),
    (-2, 0),
    (-2, 2),
    (0, -2),
    (0, 2),
    (2, -2),
    (2, 0),
    (2, 2),
]

CONNECTIVITY_LAYOUTS = {
    "standard8": {
        "name": "standard8",
        "channel_count": 8,
        "kernel_size": 3,
        "include_center": False,
        "offsets": CANONICAL_8_OFFSETS,
    },
    "full24": {
        "name": "full24",
        "channel_count": 24,
        "kernel_size": 5,
        "include_center": False,
        "offsets": [],  # Not needed for launcher
    },
    "out8": {
        "name": "out8",
        "channel_count": 8,
        "kernel_size": 5,
        "include_center": False,
        "offsets": OUTER_8_OFFSETS,
    },
}


def default_connectivity_layout_name(conn_num: int) -> str:
    if conn_num == 8:
        return "standard8"
    if conn_num == 24:
        return "full24"
    raise ValueError(f"Unsupported conn_num {conn_num}, only 8 and 24 are supported")


def normalize_conn_layout(conn_num: int, conn_layout: str | None = None) -> str:
    layout_name = (
        default_connectivity_layout_name(conn_num)
        if conn_layout is None
        else str(conn_layout)
    )
    if layout_name not in CONNECTIVITY_LAYOUTS:
        raise ValueError(
            f"Unsupported conn_layout {layout_name}, "
            f"supported: {sorted(CONNECTIVITY_LAYOUTS)}"
        )

    if conn_num == 8 and layout_name not in {"standard8", "out8"}:
        raise ValueError("conn_num=8 supports only conn_layout in {'standard8', 'out8'}")
    if conn_num == 24 and layout_name != "full24":
        raise ValueError("conn_num=24 supports only conn_layout='full24'")
    return layout_name


def is_default_connectivity_layout(conn_num: int, conn_layout: str) -> bool:
    return (
        normalize_conn_layout(conn_num, conn_layout)
        == default_connectivity_layout_name(conn_num)
    )

is_default_conn_layout = is_default_connectivity_layout
default_conn_layout_name = default_connectivity_layout_name

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "PyYAML is required for config launcher. Install dependencies to provide `yaml` module."
    ) from exc

ALLOWED_DATASETS = {
    "chase",
    "cremi",
    "drive",
    "isic2018",
    "octa500",
    "retouch",
    "retouch-Cirrus",
    "retouch-Spectrailis",
    "retouch-Topcon",
}
ALLOWED_MODES = {"single", "multi"}
ALLOWED_OCTA_VARIANTS = {"3M", "6M"}
ALLOWED_RETOUCH_DEVICES = {"Cirrus", "Spectrailis", "Topcon"}
ALLOWED_MONITOR_METRICS = {"val_dice", "val_loss"}
ALLOWED_LABEL_MODES = {"binary", "dist", "dist_inverted"}
ALLOWED_CONN_FUSIONS = {"none", "gate", "scaled_sum", "conv_residual", "decoder_guided"}
ALLOWED_FUSION_LOSS_PROFILES = {"A", "B", "C"}
DATASET_PRESETS = {
    "chase": {
        "dataset": "chase",
        "data_root": "data/chase",
        "resize": (960, 960),
        "batch_size": 4,
    },
    "cremi": {
        "dataset": "cremi",
        "data_root": "data/CREMI",
        "resize": (256, 256),
        "batch_size": 4,
    },
    "drive": {
        "dataset": "drive",
        "data_root": "data/DRIVE",
        "resize": (512, 512),
        "batch_size": 4,
    },
    "isic2018": {
        "dataset": "isic",
        "data_root": "data/ISIC2018",
        "resize": (224, 320),
        "batch_size": 64,
    },
}


def normalize_conn_fusion(value: Any) -> str:
    conn_fusion = str(value if value is not None else "none")
    if conn_fusion not in ALLOWED_CONN_FUSIONS:
        raise ValueError(
            f"Unsupported conn_fusion: {conn_fusion} "
            f"(supported: {sorted(ALLOWED_CONN_FUSIONS)})"
        )
    return conn_fusion


def normalize_fusion_loss_profile(value: Any) -> str:
    profile = str(value if value is not None else "A").upper()
    if profile not in ALLOWED_FUSION_LOSS_PROFILES:
        raise ValueError(
            f"Unsupported fusion_loss_profile: {profile} "
            f"(supported: {sorted(ALLOWED_FUSION_LOSS_PROFILES)})"
        )
    return profile


def normalize_label_mode(value: Any) -> str:
    label_mode = str(value if value is not None else "binary")
    if label_mode not in ALLOWED_LABEL_MODES:
        raise ValueError(
            f"Unsupported label_mode: {label_mode} "
            f"(supported: {sorted(ALLOWED_LABEL_MODES)})"
        )
    return label_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Config-first dataset launcher."
    )
    parser.add_argument("--config", required=True, help="Path to launcher YAML config")
    parser.add_argument("--device", type=int, default=None, help="Optional device override")
    parser.add_argument("--dry_run", action="store_true", help="Print generated commands only")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a minimal smoke run (default: 1 run, 1 epoch, batch_size=1, "
            "target_fold=1, output_dir=output_smoke)."
        ),
    )
    parser.add_argument(
        "--smoke_limit",
        type=int,
        default=1,
        help="When --smoke is set, limit to first N generated run(s).",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="Run generated train.py command(s) in test-only mode",
    )
    parser.add_argument(
        "--pretrained",
        nargs="?",
        const="__AUTO__",
        default=None,
        help=(
            "Optional pretrained path for test-only runs. "
            "If omitted after flag, auto-resolve from output_dir/dataset/experiment_name."
        ),
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None, help="Optional batch size override")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        f = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {path}")
    with f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def apply_smoke_overrides(config: dict[str, Any], mode: str) -> None:
    if mode == "single":
        block_key = "single"
    elif mode == "multi":
        block_key = "multi"
    else:
        raise ValueError(f"Unsupported mode for smoke overrides: {mode}")

    block_cfg = config.get(block_key)
    if block_cfg is None:
        block_cfg = {}
        config[block_key] = block_cfg
    if not isinstance(block_cfg, dict):
        raise ValueError(f"{block_key} block must be a mapping for smoke overrides")

    # Keep smoke outputs outside the default `output/` tree to avoid polluting aggregation.
    block_cfg["output_dir"] = "output_smoke"
    block_cfg["epochs"] = 1
    block_cfg["save_best_only"] = True
    # In train.py, early stopping is disabled when patience <= 0.
    block_cfg["early_stopping_patience"] = 0


def ensure_isic_dataset_ready(repo_root: Path) -> None:
    image_dir = repo_root / "data" / "ISIC2018" / "image"
    label_dir = repo_root / "data" / "ISIC2018" / "label"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError(
            "Missing prepared ISIC2018 npy dataset under data/ISIC2018/{image,label}. "
            "Run scripts/prepare_isic2018_npy.py first."
        )


def as_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return list(value)
    return [value]


def get_with_alias(cfg: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def normalize_datasets_for_mode(dataset_value: Any, mode: str) -> list[str]:
    if mode == "single":
        if isinstance(dataset_value, list):
            raise ValueError("single mode requires `dataset` to be a string, not a list")
        if not isinstance(dataset_value, str):
            raise ValueError("single mode requires `dataset` to be a string")
        return [dataset_value]

    if isinstance(dataset_value, str):
        datasets = [dataset_value]
    elif isinstance(dataset_value, list):
        datasets = list(dataset_value)
    else:
        raise ValueError("multi mode requires `dataset` to be a string or list of strings")

    if not datasets:
        raise ValueError("multi mode requires at least one dataset")
    if any(not isinstance(dataset, str) for dataset in datasets):
        raise ValueError("multi mode requires all `dataset` entries to be strings")
    return datasets


def format_dataset_summary(datasets: list[str]) -> str:
    if len(datasets) == 1:
        return datasets[0]
    return "[" + ",".join(datasets) + "]"


def default_single_epochs(dataset: str, conn_num: int, label_mode: str) -> int:
    if dataset.startswith("retouch"):
        return 50
    if dataset == "isic2018":
        return 500
    if label_mode in {"dist", "dist_inverted"}:
        return 260
    return 390 if conn_num == 24 else 130


def resolve_preset(
    dataset: str,
    octa_variant: str | None = None,
    retouch_device: str | None = None,
    retouch_data_root: str | None = None,
) -> dict[str, Any]:
    if dataset not in {"octa500", "retouch"}:
        return DATASET_PRESETS[dataset]
    if octa_variant not in ALLOWED_OCTA_VARIANTS:
        if dataset == "octa500":
            raise ValueError(f"Unsupported octa variant: {octa_variant} (supported: 3M, 6M)")

    if dataset == "octa500":
        return {
            "dataset": f"octa500-{octa_variant}",
            "data_root": f"data/OCTA500_{octa_variant}",
            "resize": (512, 512),
            "batch_size": 16,
            "num_class": 1,
            "lr": 0.0038,
            "lr_update": "poly",
            "use_sdl": False,
        }

    if retouch_device not in ALLOWED_RETOUCH_DEVICES:
        raise ValueError(
            f"Unsupported retouch device: {retouch_device} "
            f"(supported: {sorted(ALLOWED_RETOUCH_DEVICES)})"
        )
    data_root = retouch_data_root or "data/retouch"
    lr = 0.0008 if retouch_device == "Topcon" else 0.00085
    return {
        "dataset": f"retouch-{retouch_device}",
        "data_root": data_root,
        "resize": (256, 256),
        "batch_size": 8,
        "num_class": 4,
        "lr": lr,
        "lr_update": "poly",
        "use_sdl": False,
    }


def parse_retouch_devices(value: Any, default: list[str]) -> list[str]:
    devices = [str(v) for v in as_list(value, default)]
    unsupported = sorted({device for device in devices if device not in ALLOWED_RETOUCH_DEVICES})
    if unsupported:
        raise ValueError(
            f"Unsupported retouch device(s): {unsupported} "
            f"(supported: {sorted(ALLOWED_RETOUCH_DEVICES)})"
        )
    return devices


def parse_training_control_overrides(block_cfg: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if "monitor_metric" in block_cfg:
        monitor_metric = str(block_cfg["monitor_metric"])
        if monitor_metric not in ALLOWED_MONITOR_METRICS:
            raise ValueError(
                f"Unsupported monitor_metric: {monitor_metric} "
                f"(supported: {sorted(ALLOWED_MONITOR_METRICS)})"
            )
        overrides["monitor_metric"] = monitor_metric

    if "early_stopping_patience" in block_cfg:
        patience = int(block_cfg["early_stopping_patience"])
        if patience < 0:
            raise ValueError("early_stopping_patience must be >= 0")
        overrides["early_stopping_patience"] = patience

    if "early_stopping_min_delta" in block_cfg:
        min_delta = float(block_cfg["early_stopping_min_delta"])
        if min_delta < 0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        overrides["early_stopping_min_delta"] = min_delta

    if "early_stopping_tie_eps" in block_cfg:
        tie_eps = float(block_cfg["early_stopping_tie_eps"])
        if tie_eps < 0:
            raise ValueError("early_stopping_tie_eps must be >= 0")
        overrides["early_stopping_tie_eps"] = tie_eps

    if "early_stopping_stop_interval" in block_cfg:
        stop_interval = int(block_cfg["early_stopping_stop_interval"])
        if stop_interval < 1:
            raise ValueError("early_stopping_stop_interval must be >= 1")
        overrides["early_stopping_stop_interval"] = stop_interval

    if "tie_break_with_loss" in block_cfg:
        tie_break_with_loss = block_cfg["tie_break_with_loss"]
        if not isinstance(tie_break_with_loss, bool):
            raise ValueError("tie_break_with_loss must be boolean")
        overrides["tie_break_with_loss"] = tie_break_with_loss

    if "save_best_only" in block_cfg:
        save_best_only = block_cfg["save_best_only"]
        if not isinstance(save_best_only, bool):
            raise ValueError("save_best_only must be boolean")
        overrides["save_best_only"] = save_best_only

    return overrides


def parse_bool_config(config: dict[str, Any], key: str, default: bool = False) -> bool:
    if key not in config:
        return default
    value = config[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def _normalize_experiment_datasets(exp_cfg: dict[str, Any], default_datasets: list[str]) -> list[str]:
    exp_datasets_raw = exp_cfg.get("datasets", exp_cfg.get("dataset"))
    if exp_datasets_raw is None:
        return list(default_datasets)
    datasets = normalize_datasets_for_mode(exp_datasets_raw, mode="multi")
    unknown = sorted(set(datasets) - set(default_datasets))
    if unknown:
        raise ValueError(
            "multi.experiments specifies dataset(s) not present in top-level `dataset`: "
            + ", ".join(unknown)
        )
    return datasets


REMOVED_DECODER_FUSION_KEYS = ("decoder_fusion", "decoder_fusions", "lambda_vote_aux")


def reject_removed_decoder_fusion_keys(config: dict[str, Any], context: str) -> None:
    for key in REMOVED_DECODER_FUSION_KEYS:
        if key in config:
            raise ValueError(
                f"{context} uses removed key `{key}`. "
                "Explicit decoder_fusion support has been removed."
            )

def build_experiment_output_name(
    conn_num: int,
    label_mode: str,
    dist_aux_loss: str,
    conn_layout: str | None = None,
    conn_fusion: str = "none",
    fusion_loss_profile: str = "A",
    fusion_residual_scale: float = 0.2,
    use_seg_aux: bool = False,
    seg_aux_weight: float = 0.3,
) -> str:
    layout_name = normalize_conn_layout(conn_num, conn_layout)
    layout_suffix = "" if is_default_conn_layout(conn_num, layout_name) else f"_{layout_name}"
    conn_fusion_name = normalize_conn_fusion(conn_fusion)
    fusion_profile_name = normalize_fusion_loss_profile(fusion_loss_profile)
    if conn_fusion_name == "none":
        if label_mode == "binary":
            base_name = f"binary_{conn_num}{layout_suffix}_bce"
        else:
            base_name = f"{label_mode}_{conn_num}{layout_suffix}_{dist_aux_loss}"
    else:
        fusion_tag = f"{conn_fusion_name}_{fusion_profile_name}"
        if conn_fusion_name == "scaled_sum":
            scale_str = f"{float(fusion_residual_scale):.6f}".rstrip("0").rstrip(".")
            fusion_tag = f"{fusion_tag}_rs{scale_str}"
        if label_mode == "binary":
            base_name = f"binary_{fusion_tag}_{conn_num}{layout_suffix}_bce"
        else:
            base_name = f"{label_mode}_{fusion_tag}_{conn_num}{layout_suffix}_{dist_aux_loss}"
    
    if use_seg_aux:
        if abs(seg_aux_weight - 0.3) > 1e-6:
            base_name = f"{base_name}_segaux_w{seg_aux_weight}"
        else:
            base_name = f"{base_name}_segaux"
            
    return base_name


def resolve_pretrained_path(
    pretrained_spec: str | None,
    output_dir: str,
    dataset: str,
    conn_num: int,
    label_mode: str,
    dist_aux_loss: str,
    conn_layout: str | None = None,
    conn_fusion: str = "none",
    fusion_loss_profile: str = "A",
    fusion_residual_scale: float = 0.2,
    use_seg_aux: bool = False,
    seg_aux_weight: float = 0.3,
) -> str:
    exp_name = build_experiment_output_name(
        conn_num=conn_num,
        label_mode=label_mode,
        dist_aux_loss=dist_aux_loss,
        conn_layout=conn_layout,
        conn_fusion=conn_fusion,
        fusion_loss_profile=fusion_loss_profile,
        fusion_residual_scale=fusion_residual_scale,
        use_seg_aux=use_seg_aux,
        seg_aux_weight=seg_aux_weight,
    )
    models_dir = Path(output_dir) / dataset / exp_name / "models"

    if pretrained_spec:
        try:
            resolved = pretrained_spec.format(
                dataset=dataset,
                conn_num=conn_num,
                label_mode=label_mode,
                dist_aux_loss=dist_aux_loss,
                conn_layout=normalize_conn_layout(conn_num, conn_layout),
                conn_fusion=normalize_conn_fusion(conn_fusion),
                fusion_loss_profile=normalize_fusion_loss_profile(fusion_loss_profile),
                use_seg_aux=use_seg_aux,
                seg_aux_weight=seg_aux_weight,
                experiment_name=exp_name,
            )
        except KeyError as exc:
            raise ValueError(
                f"Unsupported pretrained format key: {exc} "
                "(supported: dataset, conn_num, label_mode, dist_aux_loss, conn_layout, "
                "conn_fusion, fusion_loss_profile, experiment_name)"
            ) from exc
        resolved_path = Path(resolved)
        if resolved_path.is_absolute():
            return str(resolved_path)

        normalized = resolved.replace("\\", "/")
        if normalized.startswith("output/") or normalized.startswith("./") or normalized.startswith("../"):
            return resolved
        if normalized.startswith("models/"):
            normalized = normalized[len("models/"):]
        return str(models_dir / normalized)

    return str(models_dir / "best_model.pth")


def build_train_cmd(
    pybin: str,
    repo_root: Path,
    preset: dict[str, Any],
    conn_num: int,
    label_mode: str,
    dist_aux_loss: str,
    conn_layout: str,
    dist_sf_l1_gamma: float,
    device: int,
    epochs: int,
    folds: int,
    target_fold: int | None,
    training_ctrl_overrides: dict[str, Any] | None = None,
    output_dir: str | None = None,
    pretrained: str | None = None,
    test_only: bool = False,
    conn_fusion: str = "none",
    fusion_loss_profile: str = "A",
    fusion_lambda_inner: float = 0.2,
    fusion_lambda_outer: float = 0.05,
    fusion_lambda_fused: float = 0.3,
    fusion_residual_scale: float = 0.2,
    use_seg_aux: bool = False,
    seg_aux_weight: float = 0.3,
    fusion_gate_reg_weight: float = 0.01,
    conn_aux_c3_weight: float = 0.3,
    conn_aux_c5_weight: float = 0.2,
) -> list[str]:
    resize_h, resize_w = preset["resize"]
    cmd = [
        pybin,
        str(repo_root / "train.py"),
        "--dataset",
        str(preset["dataset"]),
        "--data_root",
        str(preset["data_root"]),
        "--resize",
        str(resize_h),
        str(resize_w),
        "--num-class",
        str(preset.get("num_class", 1)),
        "--batch-size",
        str(preset["batch_size"]),
        "--epochs",
        str(epochs),
        "--lr",
        str(preset.get("lr", 0.0038)),
        "--lr-update",
        str(preset.get("lr_update", "poly")),
        "--folds",
        str(folds),
        "--conn_num",
        str(conn_num),
        "--conn_layout",
        str(conn_layout),
        "--label_mode",
        str(label_mode),
        "--dist_aux_loss",
        str(dist_aux_loss),
        "--dist_sf_l1_gamma",
        str(dist_sf_l1_gamma),
        "--conn_fusion",
        str(conn_fusion),
        "--fusion_loss_profile",
        str(fusion_loss_profile),
        "--fusion_lambda_inner",
        str(fusion_lambda_inner),
        "--fusion_lambda_outer",
        str(fusion_lambda_outer),
        "--fusion_lambda_fused",
        str(fusion_lambda_fused),
        "--fusion_residual_scale",
        str(fusion_residual_scale),
        "--device",
        str(device),
    ]
    if use_seg_aux:
        cmd.append("--use_seg_aux")
        cmd.extend(["--seg_aux_weight", str(seg_aux_weight)])
    if conn_fusion == "decoder_guided":
        cmd.extend(["--fusion_gate_reg_weight", str(fusion_gate_reg_weight)])
        cmd.extend(["--conn_aux_c3_weight", str(conn_aux_c3_weight)])
        cmd.extend(["--conn_aux_c5_weight", str(conn_aux_c5_weight)])
    if preset.get("use_sdl", False):
        cmd.append("--use_SDL")
    if target_fold is not None:
        cmd.extend(["--target_fold", str(target_fold)])
    if output_dir:
        cmd.extend(["--output_dir", str(output_dir)])
    if pretrained:
        cmd.extend(["--pretrained", str(pretrained)])
    if test_only:
        cmd.append("--test_only")
    if training_ctrl_overrides:
        if "monitor_metric" in training_ctrl_overrides:
            cmd.extend(["--monitor_metric", str(training_ctrl_overrides["monitor_metric"])])
        if "early_stopping_patience" in training_ctrl_overrides:
            cmd.extend(
                ["--early_stopping_patience", str(training_ctrl_overrides["early_stopping_patience"])]
            )
        if "early_stopping_min_delta" in training_ctrl_overrides:
            cmd.extend(
                ["--early_stopping_min_delta", str(training_ctrl_overrides["early_stopping_min_delta"])]
            )
        if "early_stopping_tie_eps" in training_ctrl_overrides:
            cmd.extend(
                ["--early_stopping_tie_eps", str(training_ctrl_overrides["early_stopping_tie_eps"])]
            )
        if "early_stopping_stop_interval" in training_ctrl_overrides:
            cmd.extend(
                [
                    "--early_stopping_stop_interval",
                    str(training_ctrl_overrides["early_stopping_stop_interval"]),
                ]
            )
        if "tie_break_with_loss" in training_ctrl_overrides:
            cmd.append(
                "--tie_break_with_loss"
                if bool(training_ctrl_overrides["tie_break_with_loss"])
                else "--no_tie_break_with_loss"
            )
        if "save_best_only" in training_ctrl_overrides:
            cmd.append(
                "--save_best_only"
                if bool(training_ctrl_overrides["save_best_only"])
                else "--no_save_best_only"
            )
    return cmd


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY_RUN] {' '.join(shlex.quote(p) for p in cmd)}")
        return
    subprocess.run(cmd, check=True)


def is_run_completed(run: dict[str, Any]) -> bool:
    output_dir = str(run.get("output_dir", "output"))
    preset = run["preset"]
    dataset = str(preset["dataset"])
    exp_name = build_experiment_output_name(
        conn_num=int(run["conn_num"]),
        label_mode=str(run["label_mode"]),
        dist_aux_loss=str(run["dist_aux_loss"]),
        conn_layout=str(run["conn_layout"]),
        conn_fusion=str(run.get("conn_fusion", "none")),
        fusion_loss_profile=str(run.get("fusion_loss_profile", "A")),
        fusion_residual_scale=float(run.get("fusion_residual_scale", 0.2)),
        use_seg_aux=bool(run.get("use_seg_aux", False)),
        seg_aux_weight=float(run.get("seg_aux_weight", 0.3)),
    )
    exp_dir = Path(output_dir) / dataset / exp_name

    target_fold = run.get("target_fold")
    fold_ids = [int(target_fold)] if target_fold is not None else list(range(1, int(run["folds"]) + 1))

    for fold in fold_ids:
        final_csv = exp_dir / f"final_results_{fold}.csv"
        if final_csv.exists():
            return True

    expected_epoch = int(run["epochs"])
    final_epoch_model = exp_dir / "models" / f"{expected_epoch}_model.pth"
    if final_epoch_model.exists():
        return True

    return False


def send_telegram_alert(
    pybin: str,
    repo_root: Path,
    status: str,
    summary: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    alert_script = repo_root / "scripts" / "telegram_alert.py"
    if not alert_script.exists():
        return
    subprocess.run(
        [
            pybin,
            str(alert_script),
            "--env-file",
            str(repo_root / ".env"),
            "--job",
            "train_launcher_config",
            "--status",
            status,
            "--summary",
            summary,
            "--message",
            summary,
        ],
        check=False,
    )


def build_single_schedule(config: dict[str, Any], device: int) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    single_cfg = config.get("single") or {}
    if not isinstance(single_cfg, dict):
        raise ValueError("single block must be a mapping")
    reject_removed_decoder_fusion_keys(config, "top-level config")
    reject_removed_decoder_fusion_keys(single_cfg, "single config")

    conn_num = int(single_cfg.get("conn_num", 8))
    conn_layout = normalize_conn_layout(conn_num, single_cfg.get("conn_layout", config.get("conn_layout")))
    conn_fusion = normalize_conn_fusion(single_cfg.get("conn_fusion", config.get("conn_fusion", "none")))
    fusion_loss_profile = normalize_fusion_loss_profile(
        single_cfg.get("fusion_loss_profile", config.get("fusion_loss_profile", "A"))
    )
    fusion_lambda_inner = float(single_cfg.get("fusion_lambda_inner", config.get("fusion_lambda_inner", 0.2)))
    fusion_lambda_outer = float(single_cfg.get("fusion_lambda_outer", config.get("fusion_lambda_outer", 0.05)))
    fusion_lambda_fused = float(single_cfg.get("fusion_lambda_fused", config.get("fusion_lambda_fused", 0.3)))
    fusion_residual_scale = float(single_cfg.get("fusion_residual_scale", config.get("fusion_residual_scale", 0.2)))
    label_mode = normalize_label_mode(single_cfg.get("label_mode", "binary"))
    dist_aux_loss = str(single_cfg.get("dist_aux_loss", "smooth_l1"))
    folds = int(single_cfg.get("folds", 1))
    training_ctrl_overrides = parse_training_control_overrides(single_cfg)
    test_only = parse_bool_config(single_cfg, "test_only", parse_bool_config(config, "test_only", False))
    output_dir = str(single_cfg.get("output_dir", config.get("output_dir", "output")))
    pretrained_spec = single_cfg.get("pretrained", config.get("pretrained"))
    if pretrained_spec is not None and not isinstance(pretrained_spec, str):
        raise ValueError("pretrained must be a string path")

    batch_size = get_with_alias(
        single_cfg,
        ("batch_size", "batch-size"),
        get_with_alias(config, ("batch_size", "batch-size")),
    )

    if (dataset == "retouch" or dataset.startswith("retouch-")) and conn_num != 8:
        raise ValueError("retouch single mode currently supports conn_num=8 only")

    epochs = int(single_cfg.get("epochs", default_single_epochs(dataset, conn_num, label_mode)))

    if dataset == "octa500":
        variant = str(single_cfg.get("octa_variant", "6M"))
        preset = resolve_preset(dataset, variant)
    elif dataset == "retouch" or dataset.startswith("retouch-"):
        # ... (handled in loop below)
        preset = None
    else:
        preset = resolve_preset(dataset)

    if preset is not None and batch_size is not None:
        preset = dict(preset)
        preset["batch_size"] = int(batch_size)

    def validate_single_layout_compatibility(active_preset: dict[str, Any]) -> None:
        if active_preset.get("num_class", 1) != 1 and conn_layout != "standard8":
            raise ValueError("single config supports non-standard conn_layout only for single-class runs")
        if conn_fusion != "none":
            if active_preset.get("num_class", 1) != 1:
                raise ValueError("conn_fusion supports only single-class runs")
            if conn_num != 8:
                raise ValueError("conn_fusion supports only conn_num=8")
            if conn_layout != "standard8":
                raise ValueError("conn_fusion supports only conn_layout=standard8")

    if dataset == "octa500":
        target_folds = [None]
    elif dataset == "retouch" or dataset.startswith("retouch-"):
        if dataset.startswith("retouch-"):
            retouch_devices = [dataset.split("-", 1)[1]]
        else:
            retouch_devices = parse_retouch_devices(
                single_cfg.get("retouch_device"),
                ["Spectrailis"],
            )
        retouch_data_root = str(single_cfg.get("retouch_data_root", "data/retouch"))
        if folds != 3:
            raise ValueError("retouch single mode requires folds=3")
        target_folds_cfg = as_list(single_cfg.get("target_folds"), [1, 2, 3])
        target_folds = [int(v) for v in target_folds_cfg]
        for fold in target_folds:
            if fold < 1 or fold > 3:
                raise ValueError("retouch target_folds must be within [1, 3]")
        runs: list[dict[str, Any]] = []
        for retouch_device in retouch_devices:
            preset = resolve_preset(
                "retouch",
                retouch_device=retouch_device,
                retouch_data_root=retouch_data_root,
            )
            validate_single_layout_compatibility(preset)
            if batch_size is not None:
                preset = dict(preset)
                preset["batch_size"] = int(batch_size)
            for target_fold in target_folds:
                runs.append(
                    {
                        "preset": preset,
                        "conn_num": conn_num,
                        "conn_layout": conn_layout,
                        "label_mode": label_mode,
                        "dist_aux_loss": dist_aux_loss,
                        "conn_fusion": conn_fusion,
                        "fusion_loss_profile": fusion_loss_profile,
                        "fusion_lambda_inner": fusion_lambda_inner,
                        "fusion_lambda_outer": fusion_lambda_outer,
                        "fusion_lambda_fused": fusion_lambda_fused,
                        "fusion_residual_scale": fusion_residual_scale,
                        "epochs": epochs,
                        "folds": folds,
                        "target_fold": target_fold,
                        "device": device,
                        "training_ctrl_overrides": training_ctrl_overrides,
                        "output_dir": output_dir,
                        "test_only": test_only,
                        "pretrained": resolve_pretrained_path(
                            pretrained_spec=pretrained_spec,
                            output_dir=output_dir,
                            dataset=str(preset["dataset"]),
                            conn_num=conn_num,
                            label_mode=label_mode,
                            dist_aux_loss=dist_aux_loss,
                            conn_layout=conn_layout,
                            conn_fusion=conn_fusion,
                            fusion_loss_profile=fusion_loss_profile,
                            fusion_residual_scale=fusion_residual_scale,
                        ) if test_only else None,
                    }
                )
        return runs
    else:
        target_folds = [None]

    validate_single_layout_compatibility(preset)

    return [
        {
            "preset": preset,
            "conn_num": conn_num,
            "conn_layout": conn_layout,
            "label_mode": label_mode,
            "dist_aux_loss": dist_aux_loss,
            "conn_fusion": conn_fusion,
            "fusion_loss_profile": fusion_loss_profile,
            "fusion_lambda_inner": fusion_lambda_inner,
            "fusion_lambda_outer": fusion_lambda_outer,
            "fusion_lambda_fused": fusion_lambda_fused,
            "fusion_residual_scale": fusion_residual_scale,
            "epochs": epochs,
            "folds": folds,
            "target_fold": target_folds[0],
            "device": device,
            "training_ctrl_overrides": training_ctrl_overrides,
            "output_dir": output_dir,
            "test_only": test_only,
            "pretrained": resolve_pretrained_path(
                pretrained_spec=pretrained_spec,
                output_dir=output_dir,
                dataset=str(preset["dataset"]),
                conn_num=conn_num,
                label_mode=label_mode,
                dist_aux_loss=dist_aux_loss,
                conn_layout=conn_layout,
                conn_fusion=conn_fusion,
                fusion_loss_profile=fusion_loss_profile,
                fusion_residual_scale=fusion_residual_scale,
            ) if test_only else None,
        }
    ]


def build_multi_schedule(config: dict[str, Any], device: int, datasets: list[str]) -> list[dict[str, Any]]:
    multi_cfg = config.get("multi") or {}
    if not isinstance(multi_cfg, dict):
        raise ValueError("multi block must be a mapping")
    reject_removed_decoder_fusion_keys(config, "top-level config")
    reject_removed_decoder_fusion_keys(multi_cfg, "multi config")

    epochs = int(multi_cfg.get("epochs", int(os.getenv("MULTI_TRAIN_EPOCHS", "500"))))
    folds = int(multi_cfg.get("folds", int(os.getenv("MULTI_TRAIN_FOLDS", "1"))))
    training_ctrl_overrides = parse_training_control_overrides(multi_cfg)
    test_only = parse_bool_config(multi_cfg, "test_only", parse_bool_config(config, "test_only", False))
    output_dir = str(multi_cfg.get("output_dir", config.get("output_dir", "output")))
    pretrained_spec = multi_cfg.get("pretrained", config.get("pretrained"))
    if pretrained_spec is not None and not isinstance(pretrained_spec, str):
        raise ValueError("pretrained must be a string path")

    batch_size = get_with_alias(
        multi_cfg,
        ("batch_size", "batch-size"),
        get_with_alias(config, ("batch_size", "batch-size")),
    )

    conn_values = [int(v) for v in as_list(get_with_alias(multi_cfg, ("conn_nums", "conn_num")), [8, 24])]
    conn_layouts_raw = as_list(get_with_alias(multi_cfg, ("conn_layouts", "conn_layout")), [None])
    conn_fusions = [
        normalize_conn_fusion(v)
        for v in as_list(get_with_alias(multi_cfg, ("conn_fusions", "conn_fusion")), ["none"])
    ]
    fusion_loss_profiles = [
        normalize_fusion_loss_profile(v)
        for v in as_list(get_with_alias(multi_cfg, ("fusion_loss_profiles", "fusion_loss_profile")), ["A"])
    ]
    fusion_lambda_inner = float(get_with_alias(multi_cfg, ("fusion_lambda_inner",), config.get("fusion_lambda_inner", 0.2)))
    fusion_lambda_outer = float(get_with_alias(multi_cfg, ("fusion_lambda_outer",), config.get("fusion_lambda_outer", 0.05)))
    fusion_lambda_fused = float(get_with_alias(multi_cfg, ("fusion_lambda_fused",), config.get("fusion_lambda_fused", 0.3)))
    default_residual_scale = float(get_with_alias(multi_cfg, ("fusion_residual_scale",), config.get("fusion_residual_scale", 0.2)))
    fusion_residual_scales = [
        float(v)
        for v in as_list(get_with_alias(multi_cfg, ("fusion_residual_scales", "fusion_residual_scale")), [default_residual_scale])
    ]
    fusion_matrix_cfg = multi_cfg.get("fusion_matrix")
    fusion_grid: list[dict[str, Any]] = []
    if fusion_matrix_cfg is not None:
        if not isinstance(fusion_matrix_cfg, list) or not fusion_matrix_cfg:
            raise ValueError("multi.fusion_matrix must be a non-empty list")
        for idx, entry in enumerate(fusion_matrix_cfg):
            if not isinstance(entry, dict):
                raise ValueError(f"multi.fusion_matrix[{idx}] must be a mapping")
            reject_removed_decoder_fusion_keys(entry, f"multi.fusion_matrix[{idx}]")
            if "conn_fusion" not in entry:
                raise ValueError(f"multi.fusion_matrix[{idx}] requires `conn_fusion`")
            conn_fusion = normalize_conn_fusion(entry.get("conn_fusion"))
            profile_values = [
                normalize_fusion_loss_profile(v)
                for v in as_list(get_with_alias(entry, ("fusion_loss_profiles", "fusion_loss_profile")), ["A"])
            ]
            entry_default_residual_scale = float(
                get_with_alias(entry, ("fusion_residual_scale",), default_residual_scale)
            )
            entry_residual_scales = [
                float(v)
                for v in as_list(
                    get_with_alias(entry, ("fusion_residual_scales", "fusion_residual_scale")),
                    [entry_default_residual_scale],
                )
            ]
            if conn_fusion == "none":
                profile_values = ["A"]
                entry_residual_scales = [entry_default_residual_scale]
            elif conn_fusion == "scaled_sum":
                pass
            else:
                if "fusion_residual_scales" in entry and len(entry_residual_scales) > 1:
                    raise ValueError(
                        f"multi.fusion_matrix[{idx}] uses fusion_residual_scales, "
                        "but only scaled_sum supports residual scale sweeps"
                    )
                entry_residual_scales = [entry_default_residual_scale]

            fusion_grid.append(
                {
                    "conn_fusion": conn_fusion,
                    "fusion_loss_profiles": profile_values,
                    "fusion_residual_scales": entry_residual_scales,
                }
            )
    else:
        for conn_fusion in conn_fusions:
            profile_values = fusion_loss_profiles if conn_fusion != "none" else ["A"]
            residual_scales = (
                fusion_residual_scales
                if conn_fusion == "scaled_sum"
                else [default_residual_scale]
            )
            fusion_grid.append(
                {
                    "conn_fusion": conn_fusion,
                    "fusion_loss_profiles": profile_values,
                    "fusion_residual_scales": residual_scales,
                }
            )

    label_modes = [
        normalize_label_mode(v)
        for v in as_list(get_with_alias(multi_cfg, ("label_modes", "label_mode")), ["binary", "dist", "dist_inverted"])
    ]
    dist_aux_values = [str(v) for v in as_list(get_with_alias(multi_cfg, ("dist_aux_losses", "dist_aux_loss")), ["gjml_sf_l1", "smooth_l1"])]
    binary_dist_aux = str(get_with_alias(multi_cfg, ("binary_dist_aux_loss",), "smooth_l1"))

    experiments = multi_cfg.get("experiments")
    experiments_only = parse_bool_config(multi_cfg, "experiments_only", default=False)
    if experiments_only and not experiments:
        raise ValueError("multi.experiments_only requires multi.experiments to be set")

    runs: list[dict[str, Any]] = []
    grid_datasets = [] if experiments_only else datasets
    for dataset in grid_datasets:
        dataset_conn_values = list(conn_values)
        if dataset == "octa500":
            default_variants = os.getenv("MULTI_OCTA_VARIANTS", "3M 6M").replace(",", " ").split()
            octa_variants = [str(v) for v in as_list(multi_cfg.get("octa_variants"), default_variants)]
            retouch_devices = [None]
            retouch_folds = [None]
        elif dataset == "retouch" or dataset.startswith("retouch-"):
            if dataset.startswith("retouch-"):
                retouch_devices = [dataset.split("-", 1)[1]]
            else:
                retouch_devices = parse_retouch_devices(
                    multi_cfg.get("retouch_devices"),
                    ["Cirrus", "Spectrailis", "Topcon"],
                )
            retouch_folds_cfg = as_list(multi_cfg.get("retouch_target_folds"), [1, 2, 3])
            retouch_folds = [int(v) for v in retouch_folds_cfg]
            if folds != 3:
                raise ValueError("retouch multi mode requires folds=3")
            if conn_values != [8]:
                print(
                    "[INFO] retouch multi-class path currently supports conn_num=8; forcing conn sweep to [8]",
                    file=sys.stderr,
                )
            dataset_conn_values = [8]
            for fold in retouch_folds:
                if fold < 1 or fold > 3:
                    raise ValueError("retouch_target_folds must be within [1, 3]")
            octa_variants = [None]
        else:
            octa_variants = [None]
            retouch_devices = [None]
            retouch_folds = [None]

        for variant in octa_variants:
            for retouch_device in retouch_devices:
                if dataset == "retouch" or dataset.startswith("retouch-"):
                    retouch_data_root = str(multi_cfg.get("retouch_data_root", "data/retouch"))
                    preset = resolve_preset(
                        "retouch",
                        retouch_device=retouch_device,
                        retouch_data_root=retouch_data_root,
                    )
                else:
                    preset = resolve_preset(dataset, variant)

                if batch_size is not None:
                    preset = dict(preset)
                    preset["batch_size"] = int(batch_size)
                for label_mode in label_modes:
                    aux_losses = [binary_dist_aux] if label_mode == "binary" else dist_aux_values
                    for conn_num in dataset_conn_values:
                        conn_layouts = [normalize_conn_layout(conn_num, value) for value in conn_layouts_raw]
                        if preset.get("num_class", 1) != 1:
                            invalid = [layout for layout in conn_layouts if layout != "standard8"]
                            if invalid:
                                raise ValueError("multi-class runs support only conn_layout=standard8")
                            invalid_fusions = [
                                str(grid_entry["conn_fusion"])
                                for grid_entry in fusion_grid
                                if str(grid_entry["conn_fusion"]) != "none"
                            ]
                            if invalid_fusions:
                                raise ValueError("conn_fusion supports only single-class runs")
                        for dist_aux_loss in aux_losses:
                            for conn_layout in conn_layouts:
                                for fusion_cfg in fusion_grid:
                                    conn_fusion = str(fusion_cfg["conn_fusion"])
                                    if conn_fusion != "none":
                                        if conn_num != 8:
                                            raise ValueError("conn_fusion supports only conn_num=8")
                                        if conn_layout != "standard8":
                                            raise ValueError("conn_fusion supports only conn_layout=standard8")
                                    profile_values = [str(v) for v in fusion_cfg["fusion_loss_profiles"]]
                                    residual_scales = [float(v) for v in fusion_cfg["fusion_residual_scales"]]
                                    for fusion_loss_profile in profile_values:
                                        for fusion_residual_scale in residual_scales:
                                            for target_fold in retouch_folds:
                                                runs.append(
                                                    {
                                                        "preset": preset,
                                                        "conn_num": conn_num,
                                                        "conn_layout": conn_layout,
                                                        "label_mode": label_mode,
                                                        "dist_aux_loss": dist_aux_loss,
                                                        "conn_fusion": conn_fusion,
                                                        "fusion_loss_profile": fusion_loss_profile,
                                                        "fusion_lambda_inner": fusion_lambda_inner,
                                                        "fusion_lambda_outer": fusion_lambda_outer,
                                                        "fusion_lambda_fused": fusion_lambda_fused,
                                                        "fusion_residual_scale": fusion_residual_scale,
                                                        "epochs": epochs,
                                                        "folds": folds,
                                                        "target_fold": target_fold,
                                                        "device": device,
                                                        "training_ctrl_overrides": training_ctrl_overrides,
                                                        "output_dir": output_dir,
                                                        "test_only": test_only,
                                                        "pretrained": resolve_pretrained_path(
                                                            pretrained_spec=pretrained_spec,
                                                            output_dir=output_dir,
                                                            dataset=str(preset["dataset"]),
                                                            conn_num=conn_num,
                                                            label_mode=label_mode,
                                                            dist_aux_loss=dist_aux_loss,
                                                            conn_layout=conn_layout,
                                                            conn_fusion=conn_fusion,
                                                            fusion_loss_profile=fusion_loss_profile,
                                                            fusion_residual_scale=fusion_residual_scale,
                                                        ) if test_only else None,
                                                    }
                                                )
    # Handle explicit experiments list if present
    if experiments:
        if not isinstance(experiments, list):
            raise ValueError("multi.experiments must be a list")
        for exp_idx, exp_cfg in enumerate(experiments):
            if not isinstance(exp_cfg, dict):
                raise ValueError(f"multi.experiments[{exp_idx}] must be a mapping")
            reject_removed_decoder_fusion_keys(exp_cfg, f"multi.experiments[{exp_idx}]")

            exp_datasets = _normalize_experiment_datasets(exp_cfg, default_datasets=datasets)
            for dataset in exp_datasets:
                if dataset == "octa500":
                    octa_variant = str(exp_cfg.get("octa_variant", "6M"))
                    preset = resolve_preset(dataset, octa_variant=octa_variant)
                elif dataset == "retouch" or dataset.startswith("retouch-"):
                    if dataset.startswith("retouch-"):
                        retouch_device = dataset.split("-", 1)[1]
                    else:
                        retouch_device = str(exp_cfg.get("retouch_device", "Spectrailis"))
                    retouch_data_root = str(exp_cfg.get("retouch_data_root", multi_cfg.get("retouch_data_root", "data/retouch")))
                    preset = resolve_preset(
                        "retouch",
                        retouch_device=retouch_device,
                        retouch_data_root=retouch_data_root,
                    )
                else:
                    preset = resolve_preset(dataset)

                exp_batch_size = exp_cfg.get("batch_size")
                if exp_batch_size is not None:
                    preset = dict(preset)
                    preset["batch_size"] = int(exp_batch_size)
                elif batch_size is not None:
                    preset = dict(preset)
                    preset["batch_size"] = int(batch_size)

                exp_conn_num = int(exp_cfg.get("conn_num", conn_values[0] if conn_values else 8))
                exp_label_mode = normalize_label_mode(
                    exp_cfg.get("label_mode", label_modes[0] if label_modes else "binary")
                )
                if "dist_aux_loss" in exp_cfg and exp_cfg["dist_aux_loss"] is not None:
                    exp_dist_aux_loss = str(exp_cfg["dist_aux_loss"])
                else:
                    exp_dist_aux_loss = binary_dist_aux if exp_label_mode == "binary" else (dist_aux_values[0] if dist_aux_values else "smooth_l1")
                exp_conn_layout = normalize_conn_layout(exp_conn_num, exp_cfg.get("conn_layout", "standard8"))
                exp_conn_fusion = normalize_conn_fusion(exp_cfg.get("conn_fusion", "none"))
                exp_fusion_loss_profile = normalize_fusion_loss_profile(exp_cfg.get("fusion_loss_profile", "A"))

                if exp_conn_fusion != "none":
                    if exp_conn_num != 8:
                        raise ValueError("multi.experiments with conn_fusion requires conn_num=8")
                    if exp_conn_layout != "standard8":
                        raise ValueError("multi.experiments with conn_fusion requires conn_layout=standard8")

                exp_use_seg_aux = bool(exp_cfg.get("use_seg_aux", False))
                exp_seg_aux_weight = float(exp_cfg.get("seg_aux_weight", 0.3))
                exp_device = int(exp_cfg.get("device", device))

                runs.append(
                    {
                        "preset": preset,
                        "conn_num": exp_conn_num,
                        "conn_layout": exp_conn_layout,
                        "label_mode": exp_label_mode,
                        "dist_aux_loss": exp_dist_aux_loss,
                        "conn_fusion": exp_conn_fusion,
                        "fusion_loss_profile": exp_fusion_loss_profile,
                        "fusion_lambda_inner": float(exp_cfg.get("fusion_lambda_inner", fusion_lambda_inner)),
                        "fusion_lambda_outer": float(exp_cfg.get("fusion_lambda_outer", fusion_lambda_outer)),
                        "fusion_lambda_fused": float(exp_cfg.get("fusion_lambda_fused", fusion_lambda_fused)),
                        "fusion_residual_scale": float(exp_cfg.get("fusion_residual_scale", default_residual_scale)),
                        "use_seg_aux": exp_use_seg_aux,
                        "seg_aux_weight": exp_seg_aux_weight,
                        "fusion_gate_reg_weight": float(exp_cfg.get("fusion_gate_reg_weight", 0.01)),
                        "conn_aux_c3_weight": float(exp_cfg.get("conn_aux_c3_weight", 0.3)),
                        "conn_aux_c5_weight": float(exp_cfg.get("conn_aux_c5_weight", 0.2)),
                        "epochs": int(exp_cfg.get("epochs", epochs)),
                        "folds": int(exp_cfg.get("folds", folds)),
                        "target_fold": None,
                        "device": exp_device,
                        "training_ctrl_overrides": training_ctrl_overrides,
                        "output_dir": output_dir,
                        "test_only": test_only,
                        "pretrained": resolve_pretrained_path(
                            pretrained_spec=pretrained_spec,
                            output_dir=output_dir,
                            dataset=str(preset["dataset"]),
                            conn_num=exp_conn_num,
                            label_mode=exp_label_mode,
                            dist_aux_loss=exp_dist_aux_loss,
                            conn_layout=exp_conn_layout,
                            conn_fusion=exp_conn_fusion,
                            fusion_loss_profile=exp_fusion_loss_profile,
                            fusion_residual_scale=float(exp_cfg.get("fusion_residual_scale", default_residual_scale)),
                            use_seg_aux=exp_use_seg_aux,
                            seg_aux_weight=exp_seg_aux_weight,
                        ) if test_only else None,
                    }
                )

    return runs


def validate_config_shape(config: dict[str, Any]) -> list[str]:
    removed_keys = [key for key in ("direction_grouping", "direction_fusion") if key in config]
    if removed_keys:
        raise ValueError(
            "Unsupported config key(s): "
            + ", ".join(sorted(removed_keys))
            + ". Direction-grouping options were removed from this fork."
        )

    dataset_value = config.get("dataset")
    mode = config.get("mode")

    if mode not in ALLOWED_MODES:
        raise ValueError(f"Unsupported mode: {mode} (supported: {sorted(ALLOWED_MODES)})")
    datasets = normalize_datasets_for_mode(dataset_value, mode)

    unsupported_datasets = sorted({dataset for dataset in datasets if dataset not in ALLOWED_DATASETS})
    if unsupported_datasets:
        raise ValueError(
            f"Unsupported dataset(s): {unsupported_datasets} "
            f"(supported: {sorted(ALLOWED_DATASETS)})"
        )

    return datasets


def main() -> int:
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    status = "FAILED"
    summary = f"launcher failed: {config_path.name}"

    try:
        config = load_config(config_path)
        datasets = validate_config_shape(config)

        if args.device is not None:
            config["device"] = int(args.device)
        if args.test_only:
            config["test_only"] = True
        if args.pretrained is not None:
            config["pretrained"] = None if args.pretrained == "__AUTO__" else str(args.pretrained)
        if args.batch_size is not None:
            config["batch_size"] = int(args.batch_size)

        mode = config["mode"]
        device = int(config.get("device", 0))
        dist_sf_l1_gamma = float(config.get("dist_sf_l1_gamma", 1.0))

        if args.smoke:
            apply_smoke_overrides(config, mode)
            if args.batch_size is None:
                config["batch_size"] = 1
            if args.smoke_limit < 1:
                raise ValueError("--smoke_limit must be >= 1")

        if "isic2018" in datasets:
            ensure_isic_dataset_ready(repo_root)

        runs: list[dict[str, Any]] = []
        if mode == "single":
            config["dataset"] = datasets[0]
            runs = build_single_schedule(config, device)
        else:
            runs = build_multi_schedule(config, device, datasets)

        skip_completed = parse_bool_config(config, "skip_completed", default=False)
        if mode == "multi":
            skip_completed = parse_bool_config(config.get("multi", {}), "skip_completed", default=skip_completed)
        elif mode == "single":
            skip_completed = parse_bool_config(config.get("single", {}), "skip_completed", default=skip_completed)
        if skip_completed:
            kept_runs: list[dict[str, Any]] = []
            skipped_runs = 0
            for run in runs:
                if is_run_completed(run):
                    skipped_runs += 1
                    continue
                kept_runs.append(run)
            if skipped_runs:
                print(f"[INFO] skip_completed enabled: skipped {skipped_runs} completed run(s).", file=sys.stderr)
            runs = kept_runs

        if args.smoke:
            for run in runs:
                run["epochs"] = 1
                run["target_fold"] = 1
                run["output_dir"] = "output_smoke"
            runs = runs[: args.smoke_limit]

        pybin = sys.executable
        for run in runs:
            if run.get("test_only") and run.get("pretrained") and not args.dry_run:
                if not Path(str(run["pretrained"])).exists():
                    raise ValueError(f"Missing pretrained checkpoint for test_only: {run['pretrained']}")
            cmd = build_train_cmd(
                pybin=pybin,
                repo_root=repo_root,
                preset=run["preset"],
                conn_num=run["conn_num"],
                conn_layout=run["conn_layout"],
                label_mode=run["label_mode"],
                dist_aux_loss=run["dist_aux_loss"],
                dist_sf_l1_gamma=dist_sf_l1_gamma,
                device=run["device"],
                epochs=run["epochs"],
                folds=run["folds"],
                target_fold=run.get("target_fold"),
                training_ctrl_overrides=run.get("training_ctrl_overrides"),
                output_dir=run.get("output_dir"),
                pretrained=run.get("pretrained"),
                test_only=bool(run.get("test_only", False)),
                conn_fusion=str(run.get("conn_fusion", "none")),
                fusion_loss_profile=str(run.get("fusion_loss_profile", "A")),
                fusion_lambda_inner=float(run.get("fusion_lambda_inner", 0.2)),
                fusion_lambda_outer=float(run.get("fusion_lambda_outer", 0.05)),
                fusion_lambda_fused=float(run.get("fusion_lambda_fused", 0.3)),
                fusion_residual_scale=float(run.get("fusion_residual_scale", 0.2)),
                use_seg_aux=bool(run.get("use_seg_aux", False)),
                seg_aux_weight=float(run.get("seg_aux_weight", 0.3)),
                fusion_gate_reg_weight=float(run.get("fusion_gate_reg_weight", 0.01)),
                conn_aux_c3_weight=float(run.get("conn_aux_c3_weight", 0.3)),
                conn_aux_c5_weight=float(run.get("conn_aux_c5_weight", 0.2)),
            )
            run_cmd(cmd, args.dry_run)

        run_count = len(runs)
        dataset_summary = format_dataset_summary(datasets)
        print(
            f"[INFO] Completed {mode} schedule for dataset(s)={dataset_summary} "
            f"with {run_count} run(s).",
            file=sys.stderr,
        )
        status = "DONE"
        summary = f"launcher done: dataset={dataset_summary}, mode={mode}, runs={run_count}"
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    finally:
        send_telegram_alert(
            pybin=sys.executable,
            repo_root=repo_root,
            status=status,
            summary=summary,
            dry_run=args.dry_run or args.smoke,
        )


if __name__ == "__main__":
    raise SystemExit(main())
