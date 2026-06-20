#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvfinal.lerobot_prep import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare CALVIN split directories as local LeRobot dataset roots.")
    parser.add_argument("--dataset-root", required=True, help="Root containing splitA/splitB/splitC/splitD.")
    parser.add_argument("--splits", nargs="+", required=True, help="Splits to expose or merge.")
    parser.add_argument("--output", required=True, help="Prepared LeRobot dataset root.")
    parser.add_argument("--copy", action="store_true", help="Copy single split instead of symlinking it.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing prepared output.")
    args = parser.parse_args()

    output = prepare_dataset(
        Path(args.dataset_root),
        args.splits,
        Path(args.output),
        copy=args.copy,
        force=args.force,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
