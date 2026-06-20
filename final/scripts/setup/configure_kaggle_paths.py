#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def update_json(path: Path, data_root: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["dataset_root"] = f"{data_root}/datasets/calvin_lerobot"
    data["prepared_root"] = f"{data_root}/datasets/calvin_prepared"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update config paths for a Kaggle /kaggle/working data root.")
    parser.add_argument("--data-root", default="/kaggle/working/cv_final_data")
    args = parser.parse_args()

    for name in ("configs/act_env_b.json", "configs/act_env_abc.json"):
        update_json(Path(name), args.data_root.rstrip("/"))
        print(f"updated {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
