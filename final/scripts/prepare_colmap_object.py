#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], dry_run: bool) -> int:
    print(" ".join(command))
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def count_registered_images(model_dir: Path) -> int:
    images_txt = model_dir / "images.txt"
    if not images_txt.exists():
        return 0
    count = 0
    for line in images_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.split()[0].isdigit() and len(line.split()) >= 10:
            count += 1
    return count


def convert_model_to_text(model_dir: Path, dry_run: bool) -> int:
    text_dir = model_dir.parent / "text"
    if dry_run:
        print(f"Would convert COLMAP model {model_dir} -> {text_dir}")
        return 0
    if text_dir.exists():
        shutil.rmtree(text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)
    return run(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(text_dir),
            "--output_type",
            "TXT",
        ],
        dry_run=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run COLMAP for object-A multi-view reconstruction.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--camera-model",
        default="SIMPLE_PINHOLE",
        help="Use SIMPLE_PINHOLE so 2DGS can read the model without undistortion.",
    )
    parser.add_argument("--min-registered-images", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    images = Path(args.images).resolve()
    workspace = Path(args.workspace).resolve()
    database = workspace / "database.db"
    sparse = workspace / "sparse"
    undistorted = workspace / "undistorted"
    if not images.exists() and not args.dry_run:
        print(f"Missing image directory: {images}", file=sys.stderr)
        return 2
    if shutil.which("colmap") is None and not args.dry_run:
        print("COLMAP is not installed or not on PATH.", file=sys.stderr)
        return 2
    if not args.dry_run:
        workspace.mkdir(parents=True, exist_ok=True)
        if sparse.exists():
            shutil.rmtree(sparse)
        sparse.mkdir(parents=True, exist_ok=True)
        if database.exists():
            database.unlink()
        if undistorted.exists():
            shutil.rmtree(undistorted)

    image_count = len(list(images.glob("*"))) if images.exists() else 0
    use_sequential = image_count >= 8

    feature_cmd = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--ImageReader.camera_model",
        args.camera_model,
        "--ImageReader.single_camera",
        "1",
    ]
    if use_sequential:
        match_cmd = [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(database),
            "--SequentialMatching.overlap",
            "20",
            "--SequentialMatching.quadratic_overlap",
            "1",
        ]
    else:
        match_cmd = ["colmap", "exhaustive_matcher", "--database_path", str(database)]

    mapper_cmd = [
        "colmap",
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--output_path",
        str(sparse),
        "--Mapper.multiple_models",
        "0",
        "--Mapper.min_num_matches",
        "15",
        "--Mapper.init_min_num_inliers",
        "50",
        "--Mapper.abs_pose_min_num_inliers",
        "15",
    ]

    for command in (feature_cmd, match_cmd, mapper_cmd):
        code = run(command, args.dry_run)
        if code != 0:
            return code

    model_dir = sparse / "0"
    if args.dry_run:
        print(f"COLMAP sparse model target: {model_dir}")
        print(f"2DGS source target: {undistorted}")
        return 0

    if not model_dir.exists():
        print("COLMAP did not produce sparse/0.", file=sys.stderr)
        return 2

    if convert_model_to_text(model_dir, dry_run=False) != 0:
        return 2

    registered = count_registered_images(sparse / "text")
    print(f"COLMAP registered {registered} / {image_count} images")

    if args.camera_model.upper() in {"OPENCV", "RADIAL", "SIMPLE_RADIAL"}:
        undistorted.mkdir(parents=True, exist_ok=True)
        code = run(
            [
                "colmap",
                "image_undistorter",
                "--image_path",
                str(images),
                "--input_path",
                str(model_dir),
                "--output_path",
                str(undistorted),
                "--output_type",
                "COLMAP",
            ],
            dry_run=False,
        )
        if code != 0:
            return code
        train_source = undistorted
    else:
        undistorted.mkdir(parents=True, exist_ok=True)
        images_link = undistorted / "images"
        if images_link.exists() or images_link.is_symlink():
            images_link.unlink()
        images_link.symlink_to(images, target_is_directory=True)
        sparse_out = undistorted / "sparse"
        if sparse_out.exists():
            shutil.rmtree(sparse_out)
        sparse_out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_dir, sparse_out / "0")
        train_source = undistorted

    if registered < args.min_registered_images:
        print(
            f"Warning: only {registered} images registered; 2DGS quality will be poor. "
            f"Capture 80+ views with more baseline/overlap if possible.",
            file=sys.stderr,
        )

    print(f"COLMAP sparse model: {model_dir}")
    print(f"2DGS source root: {train_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
