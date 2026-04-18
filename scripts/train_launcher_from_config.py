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

ALLOWED_DATASETS = {"chase", "drive", "isic2018", "octa500"}
ALLOWED_MODES = {"single", "multi"}
ALLOWED_DIRECTION_GROUPING = {"none", "coarse24to8"}
ALLOWED_DIRECTION_FUSION = {"mean", "weighted_sum", "conv1x1", "attention_gating"}
ALLOWED_OCTA_VARIANTS = {"3M", "6M"}

DATASET_PRESETS = {
    "chase": {
        "dataset": "chase",
        "data_root": "data/chase",
        "resize": (960, 960),
        "batch_size": 4,
    },
    "drive": {
        "dataset": "drive",
        "data_root": "data/DRIVE",
        "resize": (960, 960),
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


def default_single_epochs(dataset: str, conn_num: int, label_mode: str) -> int:
    if dataset == "isic2018":
        return 500
    if label_mode in {"dist", "dist_inverted", "dist_signed"}:
        return 260
    return 390 if conn_num == 24 else 130


def resolve_preset(dataset: str, octa_variant: str | None = None) -> dict[str, Any]:
    if dataset != "octa500":
        return DATASET_PRESETS[dataset]
    if octa_variant not in ALLOWED_OCTA_VARIANTS:
        raise ValueError(f"Unsupported octa variant: {octa_variant} (supported: 3M, 6M)")
    return {
        "dataset": f"octa500-{octa_variant}",
        "data_root": f"data/OCTA500_{octa_variant}",
        "resize": (512, 512),
        "batch_size": 16,
    }


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
) -> list[str]:
    resize_h, resize_w = preset["resize"]
    return [
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
        "1",
        "--batch-size",
        str(preset["batch_size"]),
        "--epochs",
        str(epochs),
        "--lr",
        "0.0038",
        "--lr-update",
        "poly",
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
            "--job",
            "train_launcher_config",
            "--status",
            status,
            "--summary",
            summary,
        ],
        check=False,
    )


def normalize_direction_conn_policy(direction_grouping: str, conn_values: list[int]) -> list[int]:
    if direction_grouping != "coarse24to8":
        return conn_values
    if conn_values != [8]:
        print(
            "[INFO] coarse24to8 requires canonical 8-direction branch; forcing conn sweep to [8]",
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

    direction_grouping = config.get("direction_grouping", "none")
    if direction_grouping == "coarse24to8" and conn_num == 24:
        raise ValueError("single + coarse24to8 requires conn_num=8")

    epochs = int(single_cfg.get("epochs", default_single_epochs(dataset, conn_num, label_mode)))

    if dataset == "octa500":
        variant = str(single_cfg.get("octa_variant", "6M"))
        preset = resolve_preset(dataset, variant)
    else:
        preset = resolve_preset(dataset)

    return [
        {
            "preset": preset,
            "conn_num": conn_num,
            "label_mode": label_mode,
            "dist_aux_loss": dist_aux_loss,
            "epochs": epochs,
            "folds": folds,
            "device": device,
        }
    ]


def build_multi_schedule(config: dict[str, Any], device: int) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    multi_cfg = config.get("multi") or {}
    if not isinstance(multi_cfg, dict):
        raise ValueError("multi block must be a mapping")

    epochs = int(multi_cfg.get("epochs", int(os.getenv("MULTI_TRAIN_EPOCHS", "500"))))
    folds = int(multi_cfg.get("folds", int(os.getenv("MULTI_TRAIN_FOLDS", "1"))))

    conn_values = [int(v) for v in as_list(multi_cfg.get("conn_nums"), [8, 24])]
    direction_grouping = config.get("direction_grouping", "none")
    conn_values = normalize_direction_conn_policy(direction_grouping, conn_values)

    label_modes = [str(v) for v in as_list(multi_cfg.get("label_modes"), ["binary", "dist", "dist_inverted"])]
    dist_aux_values = [str(v) for v in as_list(multi_cfg.get("dist_aux_losses"), ["gjml_sf_l1", "smooth_l1"])]
    binary_dist_aux = str(multi_cfg.get("binary_dist_aux_loss", "smooth_l1"))

    if dataset == "octa500":
        default_variants = os.getenv("MULTI_OCTA_VARIANTS", "3M 6M").replace(",", " ").split()
        octa_variants = [str(v) for v in as_list(multi_cfg.get("octa_variants"), default_variants)]
    else:
        octa_variants = [None]

    runs: list[dict[str, Any]] = []
    for variant in octa_variants:
        preset = resolve_preset(dataset, variant)
        for label_mode in label_modes:
            aux_losses = [binary_dist_aux] if label_mode == "binary" else dist_aux_values
            for conn_num in conn_values:
                for dist_aux_loss in aux_losses:
                    runs.append(
                        {
                            "preset": preset,
                            "conn_num": conn_num,
                            "label_mode": label_mode,
                            "dist_aux_loss": dist_aux_loss,
                            "epochs": epochs,
                            "folds": folds,
                            "device": device,
                        }
                    )
    return runs


def validate_config_shape(config: dict[str, Any]) -> None:
    dataset = config.get("dataset")
    mode = config.get("mode")

    if dataset not in ALLOWED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset} (supported: {sorted(ALLOWED_DATASETS)})")
    if mode not in ALLOWED_MODES:
        raise ValueError(f"Unsupported mode: {mode} (supported: {sorted(ALLOWED_MODES)})")

    direction_grouping = config.get("direction_grouping", "none")
    direction_fusion = config.get("direction_fusion", "weighted_sum")

    if direction_grouping not in ALLOWED_DIRECTION_GROUPING:
        raise ValueError(
            f"Unsupported direction_grouping: {direction_grouping} "
            f"(supported: {sorted(ALLOWED_DIRECTION_GROUPING)})"
        )
    if direction_fusion not in ALLOWED_DIRECTION_FUSION:
        raise ValueError(
            f"Unsupported direction_fusion: {direction_fusion} "
            f"(supported: {sorted(ALLOWED_DIRECTION_FUSION)})"
        )


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
        validate_config_shape(config)

        if args.device is not None:
            config["device"] = int(args.device)

        dataset = config["dataset"]
        mode = config["mode"]
        device = int(config.get("device", 0))
        direction_grouping = str(config.get("direction_grouping", "none"))
        direction_fusion = str(config.get("direction_fusion", "weighted_sum"))
        dist_sf_l1_gamma = float(config.get("dist_sf_l1_gamma", 1.0))

        if dataset == "isic2018":
            ensure_isic_dataset_ready(repo_root)

        if mode == "single":
            runs = build_single_schedule(config, device)
        else:
            runs = build_multi_schedule(config, device)

        pybin = sys.executable
        for run in runs:
            cmd = build_train_cmd(
                pybin=pybin,
                repo_root=repo_root,
                preset=run["preset"],
                conn_num=run["conn_num"],
                label_mode=run["label_mode"],
                dist_aux_loss=run["dist_aux_loss"],
                dist_sf_l1_gamma=dist_sf_l1_gamma,
                direction_grouping=direction_grouping,
                direction_fusion=direction_fusion,
                device=run["device"],
                epochs=run["epochs"],
                folds=run["folds"],
            )
            run_cmd(cmd, args.dry_run)

        run_count = len(runs)
        print(f"[INFO] Completed {mode} schedule with {run_count} run(s).", file=sys.stderr)
        status = "DONE"
        summary = f"launcher done: dataset={dataset}, mode={mode}, runs={run_count}"
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
