#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvfinal.paths import load_json
from cvfinal.ply import read_ascii_xyzrgb_ply, transform_vertices, write_ascii_xyzrgb_ply


def expanded(path: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(path)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge transformed ASCII XYZRGB PLY assets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    merged = []
    for asset in config["assets"]:
        path = expanded(asset["path"])
        if not path.exists():
            if args.allow_missing:
                print(f"Skipping missing asset {asset['name']}: {path}")
                continue
            print(f"Missing asset {asset['name']}: {path}", file=sys.stderr)
            return 2
        vertices = read_ascii_xyzrgb_ply(path, asset.get("default_color", [180, 180, 180]))
        transformed = transform_vertices(
            vertices,
            scale=float(asset.get("scale", 1.0)),
            rotation_deg=asset.get("rotation_deg", [0.0, 0.0, 0.0]),
            translation=asset.get("translation", [0.0, 0.0, 0.0]),
        )
        print(f"{asset['name']}: {len(transformed)} vertices")
        merged.extend(transformed)
    write_ascii_xyzrgb_ply(args.output, merged)
    print(f"Wrote {len(merged)} vertices to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
