#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def resolve_endpoint(explicit: str | None) -> str:
    endpoint = (explicit or os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com").rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    return endpoint


def split_paths_for(split: str, repo_files: list[str]) -> list[str]:
    return order_split_files([path for path in repo_files if path.startswith(f"{split}/")])


def missing_split_files(target: Path, split: str, repo_files: list[str]) -> list[str]:
    missing: list[str] = []
    for rel_path in split_paths_for(split, repo_files):
        dest = target / rel_path
        if not dest.exists() or dest.stat().st_size == 0:
            missing.append(rel_path)
    return missing


def split_ready(target: Path, split: str, repo_files: list[str]) -> bool:
    split_dir = target / split
    if not (split_dir / "data").exists() or not (split_dir / "meta").exists():
        return False
    return not missing_split_files(target, split, repo_files)


def list_repo_files(repo_id: str, endpoint: str) -> list[str]:
    from huggingface_hub import HfApi

    info = HfApi(endpoint=endpoint).dataset_info(repo_id)
    return [item.rfilename for item in info.siblings if item.rfilename]


def download_one_file(
    *,
    repo_id: str,
    target: Path,
    rel_path: str,
    endpoint: str,
    force: bool,
) -> str:
    from huggingface_hub import hf_hub_download

    dest = target / rel_path
    if not force and dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=rel_path,
        repo_type="dataset",
        endpoint=endpoint,
        local_dir=str(target),
    )
    return "ok"


def order_split_files(rel_paths: list[str]) -> list[str]:
    meta = sorted(path for path in rel_paths if "/meta/" in path)
    rest = sorted(path for path in rel_paths if "/meta/" not in path)
    return meta + rest


def download_split(
    *,
    repo_id: str,
    target: Path,
    split: str,
    endpoint: str,
    repo_files: list[str],
    force: bool,
    max_workers: int,
) -> None:
    from tqdm import tqdm

    rel_paths = split_paths_for(split, repo_files)
    if not rel_paths:
        raise RuntimeError(f"No files found for {split} in {repo_id}")

    pending = missing_split_files(target, split, repo_files) if not force else rel_paths
    skipped = len(rel_paths) - len(pending)

    print(f"{split}: {skipped} cached, {len(pending)} to download")
    if not pending:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                download_one_file,
                repo_id=repo_id,
                target=target,
                rel_path=rel_path,
                endpoint=endpoint,
                force=force,
            ): rel_path
            for rel_path in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=split, unit="file"):
            future.result()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download selected CALVIN LeRobot splits from Hugging Face.")
    parser.add_argument("--repo-id", default="xiaoma26/calvin-lerobot")
    parser.add_argument("--target", required=True)
    parser.add_argument("--splits", nargs="+", default=["splitA", "splitB", "splitC", "splitD"])
    parser.add_argument("--endpoint", default=None, help="HF Hub endpoint, default: HF_ENDPOINT or https://hf-mirror.com")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel file downloads")
    parser.add_argument("--force", action="store_true", help="Re-download splits even if data/meta already exist")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        print("Install huggingface_hub first: pip install huggingface_hub", file=sys.stderr)
        return 2

    endpoint = resolve_endpoint(args.endpoint)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} to {target}")
    print(f"Endpoint: {endpoint}")
    print(f"Splits: {args.splits}")
    print(f"Max workers: {args.max_workers}")

    repo_files = list_repo_files(args.repo_id, endpoint)
    print(f"Indexed {len(repo_files)} files from Hub metadata")

    for split in args.splits:
        split_dir = target / split
        if not args.force and split_ready(target, split, repo_files):
            print(f"{split}: already complete, skipping")
            continue
        missing = missing_split_files(target, split, repo_files)
        if missing:
            print(f"{split}: {len(missing)} files missing, downloading...")
        else:
            print(f"{split}: downloading...")
        download_split(
            repo_id=args.repo_id,
            target=target,
            split=split,
            endpoint=endpoint,
            repo_files=repo_files,
            force=args.force,
            max_workers=args.max_workers,
        )
        if not split_ready(target, split, repo_files):
            still_missing = missing_split_files(target, split, repo_files)
            print(
                f"{split}: incomplete after download ({len(still_missing)} files still missing)",
                file=sys.stderr,
            )
            return 1
        print(f"{split}: ok")

    for split in args.splits:
        split_dir = target / split
        print(f"{split}: data={split_dir / 'data'} meta={split_dir / 'meta'} exists={split_dir.exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
