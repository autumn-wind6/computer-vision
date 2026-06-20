#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def command_version(command: str, version_args: list[str]) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, *version_args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return f"found at {executable}, version check failed: {exc}"
    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else "version unavailable"
    return f"{executable} :: {first_line}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local/cloud tools needed by the final project.")
    parser.add_argument("--data-root", default="/workspace/cv_final_data")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if required CUDA tools are missing.")
    args = parser.parse_args()

    checks = {
        "python": command_version("python", ["--version"]) or command_version("python3", ["--version"]),
        "git": command_version("git", ["--version"]),
        "mamba": command_version("mamba", ["--version"]),
        "ffmpeg": command_version("ffmpeg", ["-version"]),
        "colmap": command_version("colmap", ["-h"]),
        "blender": command_version("blender", ["--version"]),
        "nvcc": command_version("nvcc", ["--version"]),
        "nvidia-smi": command_version("nvidia-smi", []),
    }
    data_root = Path(args.data_root)
    payload = {
        "data_root": str(data_root),
        "data_root_exists": data_root.exists(),
        "checks": checks,
        "cuda_ready": bool(checks["nvcc"] and checks["nvidia-smi"]),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    required_when_strict = ["python", "git", "mamba", "blender", "nvcc", "nvidia-smi"]
    missing = [name for name in required_when_strict if not checks.get(name)]
    if args.strict and missing:
        print(
            "Missing tools required for the full server run: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
