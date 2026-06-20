#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cvfinal.paths import ensure_workspace


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the external large-file workspace.")
    parser.add_argument("--data-root", required=True, help="Large-file root, e.g. /workspace/cv_final_data")
    args = parser.parse_args()

    created = ensure_workspace(args.data_root)
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
