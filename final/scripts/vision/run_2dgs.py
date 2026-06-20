#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_or_print(command: list[str], dry_run: bool) -> int:
    print(" ".join(command))
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrapper for 2D Gaussian Splatting training/rendering.")
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train")
    train.add_argument("--repo", required=True)
    train.add_argument("--source", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resolution", type=int, default=4)
    train.add_argument("--iterations", type=int, default=7000)
    train.add_argument("--dry-run", action="store_true")

    render = sub.add_parser("render")
    render.add_argument("--repo", required=True)
    render.add_argument("--source", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    repo = Path(args.repo)
    script = repo / ("train.py" if args.mode == "train" else "render.py")
    if not script.exists() and not args.dry_run:
        print(f"Missing 2DGS script: {script}", file=sys.stderr)
        return 2

    if args.mode == "train":
        command = [
            sys.executable,
            str(script),
            "-s",
            args.source,
            "-m",
            args.output,
            "--resolution",
            str(args.resolution),
            "--iterations",
            str(args.iterations),
        ]
    else:
        command = [sys.executable, str(script), "-s", args.source, "-m", args.output]
    return run_or_print(command, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
