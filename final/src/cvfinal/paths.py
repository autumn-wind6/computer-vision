from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


WORKSPACE_DIRS = [
    "datasets",
    "datasets/calvin_lerobot",
    "datasets/mipnerf360",
    "captures/object_a/images",
    "captures/object_a/colmap",
    "captures/object_c",
    "third_party",
    "runs/2dgs",
    "runs/aigc",
    "runs/act",
    "exports/ply",
    "exports/fusion",
    "exports/videos",
    "logs",
    "weights",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    path = Path(expanded)
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def ensure_workspace(data_root: str | Path) -> list[Path]:
    root = resolve_path(data_root)
    created: list[Path] = []
    for rel in WORKSPACE_DIRS:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created
