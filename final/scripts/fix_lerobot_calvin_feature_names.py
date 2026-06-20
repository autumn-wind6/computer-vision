#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RENAME_FEATURES = {
    "actions": "action",
    "state": "observation.state",
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
}


def rename_table_columns(table: pa.Table, mapping: dict[str, str]) -> pa.Table:
    for old, new in mapping.items():
        if old in table.column_names:
            idx = table.schema.get_field_index(old)
            table = table.set_column(idx, new, table.column(old))
    return table


def rename_stats_dict(stats: dict) -> dict:
    out = {}
    for key, value in stats.items():
        out[RENAME_FEATURES.get(key, key)] = value
    return out


def fix_info(info_path: Path) -> None:
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    for old, new in RENAME_FEATURES.items():
        if old in features:
            features[new] = features.pop(old)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")


def fix_stats(stats_path: Path) -> None:
    if not stats_path.exists():
        return
    stats = json.loads(stats_path.read_text())
    stats_path.write_text(json.dumps(rename_stats_dict(stats), ensure_ascii=False, indent=2) + "\n")


def fix_data_parquets(root: Path) -> None:
    for path in sorted((root / "data").glob("*/*.parquet")):
        table = pq.read_table(path)
        table = rename_table_columns(table, RENAME_FEATURES)
        pq.write_table(table, path)


def fix_episodes_parquets(root: Path) -> None:
    episodes_root = root / "meta" / "episodes"
    if not episodes_root.exists():
        return
    col_mapping = {
        f"stats/{old}/{suffix}": f"stats/{new}/{suffix}"
        for old, new in RENAME_FEATURES.items()
        for suffix in ("min", "max", "mean", "std", "count")
    }
    for path in sorted(episodes_root.glob("*/*.parquet")):
        table = pq.read_table(path)
        for old_col, new_col in col_mapping.items():
            if old_col in table.column_names:
                idx = table.schema.get_field_index(old_col)
                table = table.set_column(idx, new_col, table.column(old_col))
        pq.write_table(table, path)


def fix_root(root: Path) -> None:
    root = root.resolve()
    print(f"Fixing feature names in {root}")
    fix_info(root / "meta" / "info.json")
    fix_stats(root / "meta" / "stats.json")
    fix_data_parquets(root)
    fix_episodes_parquets(root)
    print(f"Done: {root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rename CALVIN LeRobot features to LeRobot v3 ACT conventions.")
    parser.add_argument("--root", action="append", required=True, help="Prepared dataset root (repeatable)")
    args = parser.parse_args()
    for root in args.root:
        fix_root(Path(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
