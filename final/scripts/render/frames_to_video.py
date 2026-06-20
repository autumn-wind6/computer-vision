#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode rendered frames into an MP4 video.")
    parser.add_argument("--frames", required=True)
    parser.add_argument("--pattern", default="%05d.png")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None and not args.dry_run:
        print("ffmpeg is not installed or not on PATH.", file=sys.stderr)
        return 2
    if not args.dry_run:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(Path(args.frames) / args.pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        args.output,
    ]
    print(" ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
