#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def patch_episodes_stats(root: Path) -> None:
    lengths = {
        json.loads(line)["episode_index"]: json.loads(line).get("length", 0)
        for line in open(root / "meta/episodes.jsonl") if line.strip()
    }
    rows = []
    for line in open(root / "meta/episodes_stats.jsonl"):
        row = json.loads(line)
        count = lengths[row["episode_index"]]
        for stats in row["stats"].values():
            if "count" not in stats:
                stats["count"] = [count]
            elif not isinstance(stats["count"], list):
                stats["count"] = [stats["count"]]
        rows.append(json.dumps(row, ensure_ascii=False))
    (root / "meta/episodes_stats.jsonl").write_text("\n".join(rows) + "\n")


def rebuild_episodes_metadata(root: Path, data_file_size_in_mb: int) -> list[dict]:
    from lerobot.scripts.convert_dataset_v21_to_v30 import (
        DEFAULT_CHUNK_SIZE,
        get_parquet_file_size_in_mb,
        get_parquet_num_frames,
        update_chunk_file_indices,
    )

    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    num_frames = 0
    paths_to_cat: list[Path] = []
    episodes_metadata: list[dict] = []

    for ep_idx, ep_path in enumerate(sorted((root / "data").glob("*/*.parquet"))):
        ep_size = get_parquet_file_size_in_mb(ep_path)
        ep_frames = get_parquet_num_frames(ep_path)
        if size_in_mb + ep_size >= data_file_size_in_mb and paths_to_cat:
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb = 0
            paths_to_cat = []

        episodes_metadata.append(
            {
                "episode_index": ep_idx,
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": num_frames,
                "dataset_to_index": num_frames + ep_frames,
            }
        )
        size_in_mb += ep_size
        num_frames += ep_frames
        paths_to_cat.append(ep_path)

    return episodes_metadata


def finish_conversion(root: Path, data_file_size_in_mb: int = 100) -> None:
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_episodes_metadata

    root = root.resolve()
    new_root = root.parent / f"{root.name}_v30"
    if not new_root.is_dir():
        raise FileNotFoundError(f"Missing converted directory: {new_root}")

    patch_episodes_stats(root)
    episodes_dir = new_root / "meta/episodes"
    if episodes_dir.exists():
        shutil.rmtree(episodes_dir)

    episodes_metadata = rebuild_episodes_metadata(root, data_file_size_in_mb)
    convert_episodes_metadata(root, new_root, episodes_metadata, None)

    old_root = root.parent / f"{root.name}_old"
    if old_root.exists():
        shutil.rmtree(old_root)
    shutil.move(str(root), str(old_root))
    shutil.move(str(new_root), str(root))
    print(f"Finished v3.0 conversion: {root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Finish LeRobot v2.1->v3.0 conversion after data step.")
    parser.add_argument("--root", required=True, help="Prepared dataset root (v2.1 source)")
    parser.add_argument("--data-file-size-in-mb", type=int, default=100)
    args = parser.parse_args()
    finish_conversion(Path(args.root), args.data_file_size_in_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
