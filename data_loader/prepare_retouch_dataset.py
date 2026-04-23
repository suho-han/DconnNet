#!/usr/bin/env python3
"""Prepare RETOUCH training volumes into PNG slice folders for K-fold usage.

Output structure:
    <dst-root>/<Device>/all/TRAINxxx/orig/*.png
    <dst-root>/<Device>/all/TRAINxxx/mask/*.png

Only RETOUCH training volumes are converted in this script.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class DeviceSource:
    source_parent: str
    source_set_name: str


DEVICE_SOURCES: dict[str, DeviceSource] = {
    "Cirrus": DeviceSource(
        source_parent="TrainingCirrus",
        source_set_name="RETOUCH-TrainingSet-Cirrus",
    ),
    "Spectrailis": DeviceSource(
        # RETOUCH original folder uses "Spectralis". We keep "Spectrailis"
        # for output compatibility with existing training arguments.
        source_parent="TrainingSpectralis",
        source_set_name="RETOUCH-TrainingSet-Spectralis",
    ),
    "Topcon": DeviceSource(
        source_parent="TrainingTopcon",
        source_set_name="RETOUCH-TrainingSet-Topcon",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RETOUCH training MHD volumes to PNG slices under "
            "<dst>/<Device>/all/TRAINxxx/{orig,mask}."
        )
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("data/retouch"),
        help="Root path that contains RETOUCH raw directories.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("data/retouch"),
        help="Root path where converted device/all tree will be created.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=sorted(DEVICE_SOURCES.keys()),
        default=sorted(DEVICE_SOURCES.keys()),
        help="Device names to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG pairs if already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan inputs and report planned outputs without writing PNG files.",
    )
    parser.add_argument(
        "--max-volumes-per-device",
        type=int,
        default=None,
        help="Optional cap used for quick smoke checks.",
    )
    return parser.parse_args()


def to_uint8_or_raise(arr: np.ndarray, array_name: str, volume_id: str) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr

    arr_min = int(arr.min())
    arr_max = int(arr.max())
    if arr_min < 0 or arr_max > 255:
        raise ValueError(
            f"{volume_id} {array_name} range is [{arr_min}, {arr_max}] and "
            "cannot be safely stored as uint8 PNG without changing values."
        )
    return arr.astype(np.uint8)


def oct_to_uint8(arr: np.ndarray, volume_id: str) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max <= arr_min:
        return np.zeros(arr.shape, dtype=np.uint8)

    # Device-specific OCT volumes can be uint16; convert to uint8 PNG via
    # per-volume min-max normalization for a unified loader input format.
    scaled = (arr - arr_min) * 255.0 / (arr_max - arr_min)
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def read_volume(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    # Shape is expected as (Z, H, W)
    return sitk.GetArrayFromImage(image)


def find_training_volumes(src_root: Path, device_name: str) -> list[Path]:
    source = DEVICE_SOURCES[device_name]
    train_root = src_root / source.source_parent / source.source_set_name
    if not train_root.is_dir():
        raise FileNotFoundError(f"Missing RETOUCH training directory: {train_root}")

    volumes = sorted(
        p for p in train_root.iterdir() if p.is_dir() and p.name.startswith("TRAIN")
    )
    if not volumes:
        raise FileNotFoundError(f"No TRAIN* volume directories found under: {train_root}")
    return volumes


def convert_device(
    src_root: Path,
    dst_root: Path,
    device_name: str,
    overwrite: bool,
    dry_run: bool,
    max_volumes: int | None,
) -> dict[str, int]:
    volumes = find_training_volumes(src_root, device_name)
    if max_volumes is not None:
        volumes = volumes[:max_volumes]

    counts = {
        "volumes": 0,
        "slices_total": 0,
        "pairs_written": 0,
        "pairs_skipped_existing": 0,
    }

    for vol_dir in volumes:
        volume_id = vol_dir.name
        oct_path = vol_dir / "oct.mhd"
        ref_path = vol_dir / "reference.mhd"
        if not oct_path.is_file() or not ref_path.is_file():
            raise FileNotFoundError(
                f"{volume_id} is missing required files: {oct_path.name}, {ref_path.name}"
            )

        oct_arr = read_volume(oct_path)
        ref_arr = read_volume(ref_path)

        if oct_arr.shape != ref_arr.shape:
            raise ValueError(
                f"{volume_id} shape mismatch: oct={oct_arr.shape}, reference={ref_arr.shape}"
            )
        if oct_arr.ndim != 3:
            raise ValueError(f"{volume_id} expects (Z,H,W), got shape={oct_arr.shape}")

        oct_arr = oct_to_uint8(oct_arr, volume_id)
        ref_arr = to_uint8_or_raise(ref_arr, "reference", volume_id)

        out_volume_root = dst_root / device_name / "all" / volume_id
        out_orig = out_volume_root / "orig"
        out_mask = out_volume_root / "mask"
        out_orig.mkdir(parents=True, exist_ok=True)
        out_mask.mkdir(parents=True, exist_ok=True)

        z_slices = int(oct_arr.shape[0])
        counts["volumes"] += 1
        counts["slices_total"] += z_slices

        for z_idx in range(z_slices):
            filename = f"{z_idx:04d}.png"
            out_orig_file = out_orig / filename
            out_mask_file = out_mask / filename

            if (
                not overwrite
                and out_orig_file.is_file()
                and out_mask_file.is_file()
            ):
                counts["pairs_skipped_existing"] += 1
                continue

            if not dry_run:
                ok_img = cv2.imwrite(str(out_orig_file), oct_arr[z_idx])
                ok_mask = cv2.imwrite(str(out_mask_file), ref_arr[z_idx])
                if not ok_img or not ok_mask:
                    raise RuntimeError(
                        f"Failed to write PNG pair for {device_name}/{volume_id}/{filename}"
                    )
            counts["pairs_written"] += 1

    return counts


def main() -> None:
    args = parse_args()

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()

    if args.max_volumes_per_device is not None and args.max_volumes_per_device <= 0:
        raise ValueError("--max-volumes-per-device must be positive")

    print(f"[INFO] src_root={src_root}")
    print(f"[INFO] dst_root={dst_root}")
    print(f"[INFO] devices={args.devices}")
    print(f"[INFO] mode={'dry-run' if args.dry_run else 'write'}")

    overall = {
        "volumes": 0,
        "slices_total": 0,
        "pairs_written": 0,
        "pairs_skipped_existing": 0,
    }

    for device_name in args.devices:
        counts = convert_device(
            src_root=src_root,
            dst_root=dst_root,
            device_name=device_name,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            max_volumes=args.max_volumes_per_device,
        )

        print(
            "[DONE] "
            f"{device_name}: "
            f"volumes={counts['volumes']}, "
            f"slices={counts['slices_total']}, "
            f"written={counts['pairs_written']}, "
            f"skipped_existing={counts['pairs_skipped_existing']}"
        )
        for key in overall:
            overall[key] += counts[key]

    print(
        "[SUMMARY] "
        f"volumes={overall['volumes']}, "
        f"slices={overall['slices_total']}, "
        f"written={overall['pairs_written']}, "
        f"skipped_existing={overall['pairs_skipped_existing']}"
    )


if __name__ == "__main__":
    main()
