from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    _safe_unlink(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())


def validate_split_root(split_root: Path) -> None:
    missing = [str(split_root / child) for child in ("data", "meta") if not (split_root / child).exists()]
    if missing:
        raise FileNotFoundError("Missing LeRobot split directories:\n" + "\n".join(missing))


def prepare_single_split(dataset_root: Path, split: str, output: Path, copy: bool = False, force: bool = False) -> Path:
    src = dataset_root / split
    validate_split_root(src)
    if output.exists() and not force:
        return output
    if force:
        _safe_unlink(output)
    output.mkdir(parents=True, exist_ok=True)
    for child in ("data", "meta", "videos"):
        child_src = src / child
        if child_src.exists():
            _link_or_copy(child_src, output / child, copy)
    return output


def _get_chunks_size(info: dict[str, Any], default: int = 1000) -> int:
    return int(info.get("chunks_size") or info.get("chunksize") or default)


def _source_parquet(split_root: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    return split_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _target_parquet(output: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    return output / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _replace_column(table: Any, name: str, values: Any) -> Any:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table
    return table.set_column(idx, name, values)


def _offset_int_fields(row: dict[str, Any], fields: tuple[str, ...], offset: int) -> None:
    for field in fields:
        if field in row and isinstance(row[field], int):
            row[field] += offset


def prepare_merged_splits(
    dataset_root: Path,
    splits: list[str],
    output: Path,
    copy: bool = False,
    force: bool = False,
) -> Path:
    if not splits:
        raise ValueError("At least one split is required")
    if len(splits) == 1:
        return prepare_single_split(dataset_root, splits[0], output, copy=copy, force=force)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Merging multiple LeRobot splits requires pyarrow: pip install pyarrow") from exc

    if output.exists() and not force:
        return output
    if force:
        _safe_unlink(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)

    first_info: dict[str, Any] | None = None
    first_meta: Path | None = None
    chunks_size = 1000
    total_frames = 0
    total_tasks = 0
    total_episodes = 0
    out_episodes: list[dict[str, Any]] = []
    out_episode_stats: list[dict[str, Any]] = []
    out_tasks: list[dict[str, Any]] = []

    for split in splits:
        split_root = dataset_root / split
        validate_split_root(split_root)
        meta = split_root / "meta"
        info = read_json(meta / "info.json")
        first_info = first_info or dict(info)
        first_meta = first_meta or meta
        chunks_size = _get_chunks_size(first_info)

        episodes = read_jsonl(meta / "episodes.jsonl")
        episode_stats = {row.get("episode_index"): row for row in read_jsonl(meta / "episodes_stats.jsonl")}
        tasks = read_jsonl(meta / "tasks.jsonl")
        task_offset = total_tasks
        for task in tasks:
            new_task = dict(task)
            _offset_int_fields(new_task, ("task_index", "index"), task_offset)
            new_task["source_split"] = split
            out_tasks.append(new_task)
        total_tasks += len(tasks)

        source_chunks_size = _get_chunks_size(info, chunks_size)
        for episode in episodes:
            source_episode_index = int(episode["episode_index"])
            new_episode_index = total_episodes
            source_file = _source_parquet(split_root, source_episode_index, source_chunks_size)
            target_file = _target_parquet(output, new_episode_index, chunks_size)
            if not source_file.exists():
                raise FileNotFoundError(f"Missing parquet episode file: {source_file}")
            target_file.parent.mkdir(parents=True, exist_ok=True)

            table = pq.read_table(source_file)
            n_rows = table.num_rows
            if "episode_index" in table.column_names:
                table = _replace_column(table, "episode_index", pa.array([new_episode_index] * n_rows, type=table["episode_index"].type))
            if "index" in table.column_names:
                table = _replace_column(table, "index", pa.array(range(total_frames, total_frames + n_rows), type=table["index"].type))
            if "task_index" in table.column_names and task_offset:
                original = table["task_index"].to_pylist()
                table = _replace_column(table, "task_index", pa.array([int(v) + task_offset for v in original], type=table["task_index"].type))
            if copy:
                pq.write_table(table, target_file)
            else:
                pq.write_table(table, target_file)

            new_episode = dict(episode)
            new_episode["episode_index"] = new_episode_index
            new_episode["source_split"] = split
            new_episode["source_episode_index"] = source_episode_index
            if "dataset_from_index" in new_episode:
                new_episode["dataset_from_index"] = total_frames
            if "dataset_to_index" in new_episode:
                new_episode["dataset_to_index"] = total_frames + n_rows
            _offset_int_fields(new_episode, ("task_index",), task_offset)
            out_episodes.append(new_episode)

            stats = episode_stats.get(source_episode_index)
            if stats is not None:
                new_stats = dict(stats)
                new_stats["episode_index"] = new_episode_index
                new_stats["source_split"] = split
                new_stats["source_episode_index"] = source_episode_index
                _offset_int_fields(new_stats, ("task_index",), task_offset)
                out_episode_stats.append(new_stats)

            total_frames += n_rows
            total_episodes += 1

    assert first_info is not None
    info = dict(first_info)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = total_tasks
    info["total_chunks"] = (total_episodes + chunks_size - 1) // chunks_size
    info["chunks_size"] = chunks_size
    info["source_splits"] = splits
    info["splits"] = {"train": f"0:{total_episodes}"}
    write_json(output / "meta" / "info.json", info)
    write_jsonl(output / "meta" / "episodes.jsonl", out_episodes)
    write_jsonl(output / "meta" / "episodes_stats.jsonl", out_episode_stats)
    write_jsonl(output / "meta" / "tasks.jsonl", out_tasks)

    if first_meta is not None:
        for name in ("modality.json", "stats.json"):
            src = first_meta / name
            if src.exists():
                shutil.copy2(src, output / "meta" / name)
    write_json(output / "meta" / "conversion.json", {"source_dataset_root": str(dataset_root), "source_splits": splits})
    return output


def prepare_dataset(dataset_root: Path, splits: list[str], output: Path, copy: bool = False, force: bool = False) -> Path:
    if len(splits) == 1:
        return prepare_single_split(dataset_root, splits[0], output, copy=copy, force=force)
    return prepare_merged_splits(dataset_root, splits, output, copy=copy, force=force)
