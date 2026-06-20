#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cvfinal.paths import load_json


def expanded(path: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(path)))


def nonempty(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(path.iterdir())
    return False


def add_required(items: list[dict], label: str, path: Path, kind: str = "path") -> None:
    items.append({"label": label, "path": str(path), "kind": kind, "ok": nonempty(path)})


def add_optional(items: list[dict], label: str, path: Path, kind: str = "path") -> None:
    items.append({"label": label, "path": str(path), "kind": kind, "ok": nonempty(path)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether all server-side final artifacts exist before shutting down the cloud machine.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--data-root", default=os.environ.get("CVFINAL_DATA_ROOT", "/workspace/cv_final_data"))
    parser.add_argument("--fusion-config", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser()
    data_root = Path(args.data_root).expanduser()
    fusion_config = Path(args.fusion_config) if args.fusion_config else project_root / "configs/fusion_transforms.example.json"
    os.environ.setdefault("CVFINAL_DATA_ROOT", str(data_root))

    required: list[dict] = []
    optional: list[dict] = []

    add_required(required, "ACT splitB best checkpoint", data_root / "runs/act/env_b/checkpoints/best")
    add_required(required, "ACT splitA+B+C best checkpoint", data_root / "runs/act/env_abc/checkpoints/best")
    add_required(required, "ACT splitB model eval on splitD", data_root / "runs/act/env_b_eval_d")
    add_required(required, "ACT splitA+B+C model eval on splitD", data_root / "runs/act/env_abc_eval_d")
    add_required(required, "Packaged best weights", data_root / "weights/cvfinal_best_weights.zip")
    add_required(required, "Merged fusion PLY", data_root / "exports/fusion/counter_with_assets_ascii.ply")
    add_required(required, "Fusion walkthrough video", data_root / "exports/videos/counter_fusion_walkthrough.mp4")
    add_required(required, "Final server artifact bundle", data_root / "exports/cvfinal_server_artifacts.zip")

    config = load_json(fusion_config)
    for asset in config["assets"]:
        add_required(required, f"Fusion source PLY: {asset['name']}", expanded(asset["path"]))

    add_optional(optional, "ACT metric summary JSON", project_root / "results/act_metrics_summary.json")
    add_optional(optional, "ACT metric summary Markdown", project_root / "results/act_metrics_summary.md")
    report_assets = data_root / "exports/report_assets"
    keyframes = sorted(report_assets.glob("fusion_keyframe_*.png")) if report_assets.exists() else []
    optional.append(
        {
            "label": "Fusion video keyframes",
            "path": str(report_assets),
            "kind": "png-sequence",
            "ok": len(keyframes) >= 3,
            "count": len(keyframes),
        }
    )

    payload = {
        "project_root": str(project_root),
        "data_root": str(data_root),
        "required": required,
        "optional": optional,
    }

    output_json = Path(args.output_json) if args.output_json else data_root / "exports/final_server_audit.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    missing = [item for item in required if not item["ok"]]
    warnings = [item for item in optional if not item["ok"]]

    print(f"Audit JSON: {output_json}")
    if missing:
        print("Missing required server artifacts:", file=sys.stderr)
        for item in missing:
            print(f"- {item['label']}: {item['path']}", file=sys.stderr)
    if warnings:
        print("Optional report helpers not found:", file=sys.stderr)
        for item in warnings:
            print(f"- {item['label']}: {item['path']}", file=sys.stderr)
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
