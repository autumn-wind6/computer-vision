#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvfinal.paths import load_json
from cvfinal.ply import Vertex, read_ascii_xyzrgb_ply, write_ascii_xyzrgb_ply


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print(" ".join(command))
    result = subprocess.run(command, check=False, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def validate_configs() -> None:
    required = [
        ROOT / "configs" / "paths.example.json",
        ROOT / "configs" / "2dgs_background_counter.json",
        ROOT / "configs" / "fusion_transforms.example.json",
        ROOT / "configs" / "act_env_b.json",
        ROOT / "configs" / "act_env_abc.json",
    ]
    for path in required:
        load_json(path)
        print(f"valid json: {path.relative_to(ROOT)}")


def validate_ply_merge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        a = tmpdir / "a.ply"
        b = tmpdir / "b.ply"
        out = tmpdir / "merged.ply"
        write_ascii_xyzrgb_ply(a, [Vertex(0, 0, 0, 255, 0, 0)])
        write_ascii_xyzrgb_ply(b, [Vertex(1, 0, 0, 0, 255, 0)])
        cfg = tmpdir / "fusion.json"
        cfg.write_text(
            json.dumps(
                {
                    "assets": [
                        {"name": "a", "path": str(a), "scale": 1, "rotation_deg": [0, 0, 0], "translation": [0, 0, 0]},
                        {"name": "b", "path": str(b), "scale": 2, "rotation_deg": [0, 0, 0], "translation": [1, 0, 0]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        run([sys.executable, "scripts/merge_scene_assets.py", "--config", str(cfg), "--output", str(out)])
        vertices = read_ascii_xyzrgb_ply(out)
        assert len(vertices) == 2, len(vertices)
        assert abs(vertices[1].x - 3.0) < 1e-8, vertices[1]
        print("ply merge ok")


def validate_dry_runs() -> None:
    run(
        [
            sys.executable,
            "scripts/run_2dgs.py",
            "train",
            "--repo",
            "/missing/2dgs",
            "--source",
            "/data/scene",
            "--output",
            "/tmp/out",
            "--dry-run",
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        split_b = tmpdir / "calvin_lerobot" / "splitB"
        (split_b / "data").mkdir(parents=True)
        (split_b / "meta").mkdir(parents=True)
        run(
            [
                sys.executable,
                "scripts/prepare_lerobot_dataset.py",
                "--dataset-root",
                str(tmpdir / "calvin_lerobot"),
                "--splits",
                "splitB",
                "--output",
                str(tmpdir / "calvin_prepared" / "env_b"),
                "--force",
            ]
        )
    run(
        [
            sys.executable,
            "scripts/run_act_experiment.py",
            "train",
            "--config",
            "configs/act_env_b.json",
            "--output",
            "/tmp/cvfinal_act_smoke",
            "--dry-run",
        ]
    )
    run(
        [
            sys.executable,
            "scripts/prepare_captures_from_video.py",
            "--video",
            "/tmp/object_video.mp4",
            "--object-a-images",
            "/tmp/object_a/images",
            "--object-c-image",
            "/tmp/object_c/foreground.png",
            "--dry-run",
        ]
    )


def main() -> int:
    validate_configs()
    validate_ply_merge()
    validate_dry_runs()
    print("smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
