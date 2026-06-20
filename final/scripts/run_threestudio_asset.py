#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch threestudio text-to-3D asset generation.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/dreamfusion-if.yaml")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    repo = Path(args.repo)
    launch = repo / "launch.py"
    if not launch.exists() and not args.dry_run:
        print(f"Missing threestudio launch.py: {launch}", file=sys.stderr)
        return 2
    if not args.dry_run:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(launch),
        "--config",
        args.config,
        "--train",
        "--gpu",
        args.gpu,
        f"system.prompt_processor.prompt={args.prompt}",
        f"trial_dir={args.output}",
        *args.extra,
    ]
    print(" ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False, cwd=repo).returncode


if __name__ == "__main__":
    raise SystemExit(main())
