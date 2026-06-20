#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Magic123 single-image-to-3D asset generation.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iters", type=int, default=3000, help="Coarse-stage training iterations.")
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--text",
        default="A high-resolution DSLR image of a small foreground object",
        help="Text prompt paired with the input image.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo)
    main_py = repo / "main.py"
    image = Path(args.image)
    output = Path(args.output)
    workspace = output / "coarse"

    if not main_py.exists() and not args.dry_run:
        print(f"Missing Magic123 entrypoint: {main_py}", file=sys.stderr)
        return 2
    if not image.exists() and not args.dry_run:
        print(f"Missing foreground image: {image}", file=sys.stderr)
        return 2
    if not args.dry_run:
        workspace.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(main_py),
        "-O",
        "--text",
        args.text,
        "--sd_version",
        "1.5",
        "--image",
        str(image),
        "--workspace",
        str(workspace),
        "--optim",
        "adam",
        "--iters",
        str(args.iters),
        "--guidance",
        "SD",
        "zero123",
        "--lambda_guidance",
        "1.0",
        "40",
        "--guidance_scale",
        "100",
        "5",
        "--latent_iter_ratio",
        "0",
        "--normal_iter_ratio",
        "0.2",
        "--t_range",
        "0.2",
        "0.6",
        "--bg_radius",
        "-1",
        "--save_mesh",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(" ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False, cwd=repo, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
