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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare object captures from an uploaded MP4 video.")
    parser.add_argument("--video", required=True, help="Input MP4 path.")
    parser.add_argument("--object-a-images", required=True, help="Output directory for COLMAP/2DGS frames.")
    parser.add_argument("--object-c-image", required=True, help="Output PNG candidate for Magic123.")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate for object A.")
    parser.add_argument("--max-frames", type=int, default=150, help="Maximum extracted frames for object A.")
    parser.add_argument("--object-c-time", default="00:00:02", help="Timestamp used for the object C candidate frame.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    video = Path(args.video)
    object_a = Path(args.object_a_images)
    object_c = Path(args.object_c_image)
    if shutil.which("ffmpeg") is None and not args.dry_run:
        print("ffmpeg is not installed or not on PATH.", file=sys.stderr)
        return 2
    if not video.exists() and not args.dry_run:
        print(f"Missing video: {video}", file=sys.stderr)
        return 2
    if not args.dry_run:
        object_a.mkdir(parents=True, exist_ok=True)
        object_c.parent.mkdir(parents=True, exist_ok=True)

    # Extract a bounded number of sharp-enough, evenly spaced frames for COLMAP.
    frame_pattern = object_a / "frame_%04d.png"
    extract_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={args.fps}",
        "-frames:v",
        str(args.max_frames),
        str(frame_pattern),
    ]
    code = run(extract_cmd, args.dry_run)
    if code != 0:
        return code

    # Pick one clean frame as the Magic123 input candidate. Remove background manually or with rembg afterwards.
    object_c_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(args.object_c_time),
        "-i",
        str(video),
        "-frames:v",
        "1",
        str(object_c),
    ]
    code = run(object_c_cmd, args.dry_run)
    if code != 0:
        return code

    print(f"Object A frames: {object_a}")
    print(f"Object C candidate frame: {object_c}")
    print("For object C, remove the background and overwrite foreground.png before running Magic123.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
