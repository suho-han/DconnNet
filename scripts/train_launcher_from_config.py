#!/usr/bin/env python3
"""Config-first train launcher.

Usage:
  scripts/train_launcher.sh --config <yaml> [--device N] [--dry_run]
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

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
ALLOWED_DIRECTION_GROUPING = {"none", "24to8"}
ALLOWED_DIRECTION_FUSION = {"mean", "weighted_sum", "conv1x1", "attention_gating"}
ALLOWED_OCTA_VARIANTS = {"3M", "6M"}
ALLOWED_RETOUCH_DEVICES = {"Cirrus", "Spectrailis", "Topcon"}
ALLOWED_MONITOR_METRICS = {"val_dice", "val_loss"}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Config-first dataset launcher."
    )
    parser.add_argument("--config", required=True, help="Path to launcher YAML config")
    parser.add_argument("--device", type=int, default=None, help="Optional device override")
    parser.add_argument("--dry_run", action="store_true", help="Print generated commands only")
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
    if label_mode in {"dist", "dist_inverted", "dist_signed"}:
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


def build_experiment_output_name(
    conn_num: int,
    label_mode: str,
    dist_aux_loss: str,
    direction_grouping: str,
    direction_fusion: str,
) -> str:
    if label_mode == "binary":
        base_name = f"binary_{conn_num}_bce"
    else:
        base_name = f"{label_mode}_{conn_num}_{dist_aux_loss}"
    if direction_grouping != "none":
        base_name = f"{base_name}_{direction_grouping}_{direction_fusion}"
    return base_name


def resolve_pretrained_path(
    pretrained_spec: str | None,
    output_dir: str,
    dataset: str,
    conn_num: int,
    label_mode: str,
    dist_aux_loss: str,
    direction_grouping: str,
    direction_fusion: str,
) -> str:
    exp_name = build_experiment_output_name(
        conn_num=conn_num,
        label_mode=label_mode,
        dist_aux_loss=dist_aux_loss,
        direction_grouping=direction_grouping,
        direction_fusion=direction_fusion,
    )
    models_dir = Path(output_dir) / dataset / exp_name / "models"

    if pretrained_spec:
        try:
            resolved = pretrained_spec.format(
                dataset=dataset,
                conn_num=conn_num,
                label_mode=label_mode,
                dist_aux_loss=dist_aux_loss,
                direction_grouping=direction_grouping,
                direction_fusion=direction_fusion,
                experiment_name=exp_name,
            )
        except KeyError as exc:
            raise ValueError(
                f"Unsupported pretrained format key: {exc} "
                "(supported: dataset, conn_num, label_mode, dist_aux_loss, "
                "direction_grouping, direction_fusion, experiment_name)"
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
    dist_sf_l1_gamma: float,
    direction_grouping: str,
    direction_fusion: str,
    device: int,
    epochs: int,
    folds: int,
    target_fold: int | None,
    training_ctrl_overrides: dict[str, Any] | None = None,
    output_dir: str | None = None,
    pretrained: str | None = None,
    test_only: bool = False,
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
        "--label_mode",
        str(label_mode),
        "--dist_aux_loss",
        str(dist_aux_loss),
        "--dist_sf_l1_gamma",
        str(dist_sf_l1_gamma),
        "--direction_grouping",
        str(direction_grouping),
        "--direction_fusion",
        str(direction_fusion),
        "--device",
        str(device),
    ]
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


def normalize_direction_conn_policy(direction_grouping: str, conn_values: list[int]) -> list[int]:
    if direction_grouping != "24to8":
        return conn_values
    if conn_values != [8]:
        print(
            "[INFO] 24to8 requires canonical 8-direction branch; forcing conn sweep to [8]",
            file=sys.stderr,
        )
    return [8]


def build_single_schedule(config: dict[str, Any], device: int) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    single_cfg = config.get("single") or {}
    if not isinstance(single_cfg, dict):
        raise ValueError("single block must be a mapping")

    conn_num = int(single_cfg.get("conn_num", 8))
    label_mode = str(single_cfg.get("label_mode", "binary"))
    dist_aux_loss = str(single_cfg.get("dist_aux_loss", "smooth_l1"))
    folds = int(single_cfg.get("folds", 1))
    training_ctrl_overrides = parse_training_control_overrides(single_cfg)
    test_only = parse_bool_config(single_cfg, "test_only", parse_bool_config(config, "test_only", False))
    output_dir = str(single_cfg.get("output_dir", config.get("output_dir", "output")))
    pretrained_spec = single_cfg.get("pretrained", config.get("pretrained"))
    if pretrained_spec is not None and not isinstance(pretrained_spec, str):
        raise ValueError("pretrained must be a string path")

    batch_size = single_cfg.get("batch_size", config.get("batch_size"))

    direction_grouping = config.get("direction_grouping", "none")
    if direction_grouping == "24to8" and conn_num == 24:
        raise ValueError("single + 24to8 requires conn_num=8")
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
            if batch_size is not None:
                preset = dict(preset)
                preset["batch_size"] = int(batch_size)
            for target_fold in target_folds:
                runs.append(
                    {
                        "preset": preset,
                        "conn_num": conn_num,
                        "label_mode": label_mode,
                        "dist_aux_loss": dist_aux_loss,
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
                            direction_grouping=direction_grouping,
                            direction_fusion=str(config.get("direction_fusion", "weighted_sum")),
                        ) if test_only else None,
                    }
                )
        return runs
    else:
        target_folds = [None]

    return [
        {
            "preset": preset,
            "conn_num": conn_num,
            "label_mode": label_mode,
            "dist_aux_loss": dist_aux_loss,
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
                direction_grouping=direction_grouping,
                direction_fusion=str(config.get("direction_fusion", "weighted_sum")),
            ) if test_only else None,
        }
    ]


def build_multi_schedule(config: dict[str, Any], device: int, datasets: list[str]) -> list[dict[str, Any]]:
    multi_cfg = config.get("multi") or {}
    if not isinstance(multi_cfg, dict):
        raise ValueError("multi block must be a mapping")

    epochs = int(multi_cfg.get("epochs", int(os.getenv("MULTI_TRAIN_EPOCHS", "500"))))
    folds = int(multi_cfg.get("folds", int(os.getenv("MULTI_TRAIN_FOLDS", "1"))))
    training_ctrl_overrides = parse_training_control_overrides(multi_cfg)
    test_only = parse_bool_config(multi_cfg, "test_only", parse_bool_config(config, "test_only", False))
    output_dir = str(multi_cfg.get("output_dir", config.get("output_dir", "output")))
    pretrained_spec = multi_cfg.get("pretrained", config.get("pretrained"))
    if pretrained_spec is not None and not isinstance(pretrained_spec, str):
        raise ValueError("pretrained must be a string path")

    batch_size = multi_cfg.get("batch_size", config.get("batch_size"))

    conn_values = [int(v) for v in as_list(multi_cfg.get("conn_nums"), [8, 24])]
    direction_grouping = config.get("direction_grouping", "none")
    conn_values = normalize_direction_conn_policy(direction_grouping, conn_values)

    label_modes = [str(v) for v in as_list(multi_cfg.get("label_modes"), ["binary", "dist", "dist_inverted"])]
    dist_aux_values = [str(v) for v in as_list(multi_cfg.get("dist_aux_losses"), ["gjml_sf_l1", "smooth_l1"])]
    binary_dist_aux = str(multi_cfg.get("binary_dist_aux_loss", "smooth_l1"))

    runs: list[dict[str, Any]] = []
    for dataset in datasets:
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
                        for dist_aux_loss in aux_losses:
                            for target_fold in retouch_folds:
                                runs.append(
                                    {
                                        "preset": preset,
                                        "conn_num": conn_num,
                                        "label_mode": label_mode,
                                        "dist_aux_loss": dist_aux_loss,
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
                                            direction_grouping=direction_grouping,
                                            direction_fusion=str(config.get("direction_fusion", "weighted_sum")),
                                        ) if test_only else None,
                                    }
                                )
    return runs


def normalize_direction_fusions(direction_fusion_value: Any) -> list[str]:
    direction_fusions = [str(v) for v in as_list(direction_fusion_value, ["weighted_sum"])]
    if not direction_fusions:
        raise ValueError("direction_fusion must include at least one fusion mode")
    unsupported_fusions = sorted(
        {fusion for fusion in direction_fusions if fusion not in ALLOWED_DIRECTION_FUSION}
    )
    if unsupported_fusions:
        raise ValueError(
            f"Unsupported direction_fusion: {unsupported_fusions} "
            f"(supported: {sorted(ALLOWED_DIRECTION_FUSION)})"
        )
    return direction_fusions


def validate_config_shape(config: dict[str, Any]) -> tuple[list[str], list[str]]:
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

    direction_grouping = config.get("direction_grouping", "none")
    direction_fusion = config.get("direction_fusion", "weighted_sum")

    if direction_grouping not in ALLOWED_DIRECTION_GROUPING:
        raise ValueError(
            f"Unsupported direction_grouping: {direction_grouping} "
            f"(supported: {sorted(ALLOWED_DIRECTION_GROUPING)})"
        )
    direction_fusions = normalize_direction_fusions(direction_fusion)
    return datasets, direction_fusions


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
        datasets, direction_fusions = validate_config_shape(config)

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
        direction_grouping = str(config.get("direction_grouping", "none"))
        direction_fusion = str(config.get("direction_fusion", "weighted_sum"))
        dist_sf_l1_gamma = float(config.get("dist_sf_l1_gamma", 1.0))

        if "isic2018" in datasets:
            ensure_isic_dataset_ready(repo_root)

        runs: list[dict[str, Any]] = []
        if mode == "single":
            config["dataset"] = datasets[0]
            for fusion in direction_fusions:
                fusion_config = dict(config)
                fusion_config["direction_fusion"] = fusion
                fusion_runs = build_single_schedule(fusion_config, device)
                for run in fusion_runs:
                    run["direction_fusion"] = fusion
                runs.extend(fusion_runs)
        else:
            for fusion in direction_fusions:
                fusion_config = dict(config)
                fusion_config["direction_fusion"] = fusion
                fusion_runs = build_multi_schedule(fusion_config, device, datasets)
                for run in fusion_runs:
                    run["direction_fusion"] = fusion
                runs.extend(fusion_runs)

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
                label_mode=run["label_mode"],
                dist_aux_loss=run["dist_aux_loss"],
                dist_sf_l1_gamma=dist_sf_l1_gamma,
                direction_grouping=direction_grouping,
                direction_fusion=str(run.get("direction_fusion", direction_fusion)),
                device=run["device"],
                epochs=run["epochs"],
                folds=run["folds"],
                target_fold=run.get("target_fold"),
                training_ctrl_overrides=run.get("training_ctrl_overrides"),
                output_dir=run.get("output_dir"),
                pretrained=run.get("pretrained"),
                test_only=bool(run.get("test_only", False)),
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
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    raise SystemExit(main())
