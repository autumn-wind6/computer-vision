#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def add_path(zipf: zipfile.ZipFile, path: Path) -> None:
    if path.is_file():
        zipf.write(path, arcname=path.name)
        return
    for child in path.rglob("*"):
        if child.is_file():
            zipf.write(child, arcname=str(Path(path.name) / child.relative_to(path)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Package selected checkpoints and fusion exports.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--strict", action="store_true", help="Fail if any input is missing.")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    missing = [Path(raw) for raw in args.inputs if not Path(raw).exists()]
    if args.strict and missing:
        for path in missing:
            print(f"Missing input: {path}")
        return 2
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for raw in args.inputs:
            path = Path(raw)
            if not path.exists():
                print(f"Skipping missing input: {path}")
                continue
            add_path(zipf, path)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
