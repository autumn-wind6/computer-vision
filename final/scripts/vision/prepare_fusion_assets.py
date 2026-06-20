#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cvfinal.paths import load_json
from cvfinal.ply import Vertex, read_ascii_xyzrgb_ply, write_ascii_xyzrgb_ply


ASSET_ENV_KEYS = {
    "background_counter": ["BACKGROUND_PLY", "BACKGROUND_COUNTER_PLY"],
    "object_a_real_2dgs": ["OBJECT_A_PLY"],
    "object_b_text_to_3d": ["OBJECT_B_PLY"],
    "object_c_image_to_3d": ["OBJECT_C_PLY"],
}

ASSET_SEARCH_DIRS = {
    "background_counter": ["runs/2dgs/background_counter"],
    "object_a_real_2dgs": ["runs/2dgs/object_a"],
    "object_b_text_to_3d": ["runs/aigc/object_b_text_to_3d"],
    "object_c_image_to_3d": ["runs/aigc/object_c_image_to_3d"],
}

SH_C0 = 0.28209479177387814


def expanded(path: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(path)))


def clamp_color(value: int | float) -> int:
    return max(0, min(255, int(round(float(value)))))


def color_from_float(value: object) -> int:
    raw = float(value)
    if 0.0 <= raw <= 1.0:
        raw *= 255.0
    return clamp_color(raw)


def env_source(asset_name: str) -> Path | None:
    for key in ASSET_ENV_KEYS.get(asset_name, []):
        raw = os.environ.get(key)
        if raw:
            return expanded(raw)
    return None


def discover_source(asset_name: str, data_root: Path, target: Path) -> Path | None:
    candidates: list[Path] = []
    for relative in ASSET_SEARCH_DIRS.get(asset_name, []):
        root = data_root / relative
        if root.exists():
            candidates.extend(path for path in root.rglob("*.ply") if path.resolve() != target.resolve())
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.stat().st_size, path.stat().st_mtime), reverse=True)
    return candidates[0]


def vertices_from_plyfile(source: Path, default_color: Iterable[int]) -> list[Vertex]:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("Install plyfile first, or provide ASCII PLY files.") from exc

    ply = PlyData.read(source)
    if "vertex" not in ply:
        raise ValueError(f"{source} has no vertex element")
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    for required in ("x", "y", "z"):
        if required not in names:
            raise ValueError(f"{source} must contain x y z properties")

    default = list(default_color)
    if len(default) != 3:
        raise ValueError("default_color must have three channels")

    has_rgb = all(name in names for name in ("red", "green", "blue"))
    has_short_rgb = all(name in names for name in ("r", "g", "b"))
    has_dc = all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2"))

    rows: list[Vertex] = []
    for item in vertex:
        if has_rgb:
            red = color_from_float(item["red"])
            green = color_from_float(item["green"])
            blue = color_from_float(item["blue"])
        elif has_short_rgb:
            red = color_from_float(item["r"])
            green = color_from_float(item["g"])
            blue = color_from_float(item["b"])
        elif has_dc:
            red = clamp_color((float(item["f_dc_0"]) * SH_C0 + 0.5) * 255.0)
            green = clamp_color((float(item["f_dc_1"]) * SH_C0 + 0.5) * 255.0)
            blue = clamp_color((float(item["f_dc_2"]) * SH_C0 + 0.5) * 255.0)
        else:
            red, green, blue = [clamp_color(v) for v in default]
        rows.append(Vertex(float(item["x"]), float(item["y"]), float(item["z"]), red, green, blue))
    return rows


def normalize_ply(source: Path, target: Path, default_color: Iterable[int]) -> None:
    with open(source, "rb") as f:
        header = f.read(256)
    try:
        rows = vertices_from_plyfile(source, default_color)
    except RuntimeError:
        if b"format ascii 1.0" not in header:
            raise
        rows = read_ascii_xyzrgb_ply(source, default_color)
    write_ascii_xyzrgb_ply(target, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect and normalize scene assets to ASCII XYZRGB PLY files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=os.environ.get("CVFINAL_DATA_ROOT", "/workspace/cv_final_data"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    config = load_json(args.config)
    missing: list[str] = []
    for asset in config["assets"]:
        name = asset["name"]
        target = expanded(asset["path"])
        default_color = asset.get("default_color", [180, 180, 180])
        source = env_source(name)
        if source is None:
            source = discover_source(name, data_root, target)

        if target.exists() and not args.force:
            print(f"{name}: using existing {target}")
            continue

        if source is None or not source.exists():
            env_hint = "/".join(ASSET_ENV_KEYS.get(name, [])) or f"{name.upper()}_PLY"
            missing.append(f"{name}: no source PLY found; set {env_hint}=<path-to-ply>")
            continue

        if source.suffix.lower() != ".ply":
            missing.append(f"{name}: source must be a PLY file for automatic fusion, got {source}")
            continue

        print(f"{name}: {source} -> {target}")
        normalize_ply(source, target, default_color)

    if missing:
        print("Missing fusion assets:", file=sys.stderr)
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
