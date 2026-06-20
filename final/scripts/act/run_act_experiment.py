#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cvfinal.lerobot_prep import prepare_dataset
from cvfinal.paths import load_json, write_json


def make_prepared_manifest(config: dict, output: Path, prepared_dataset: Path) -> Path:
    dataset_root = Path(os.path.expandvars(config["dataset_root"])).expanduser()
    train_splits = config["train_splits"]
    eval_split = config["eval_split"]
    # Keep manifest outside LeRobot's output_dir; otherwise first train run fails
    # because lerobot refuses to write into an already-existing directory.
    prepared = output.parent / f"{config['experiment']}_prepared_manifest.json"
    write_json(
        prepared,
        {
            "dataset_repo_id": config["dataset_repo_id"],
            "dataset_root": str(dataset_root),
            "prepared_dataset": str(prepared_dataset),
            "train_splits": train_splits,
            "eval_split": eval_split,
            "note": "Prepared dataset roots expose LeRobot data/meta directly. splitB and splitD can be symlinks; splitABC is materialized with renumbered episodes.",
        },
    )
    return prepared


def resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        path = path.parent
    if (path / "model.safetensors").is_file():
        return path
    if (path / "pretrained_model" / "model.safetensors").is_file():
        return path / "pretrained_model"

    checkpoints_dir = path if path.name == "checkpoints" else path.parent
    if path.name in {"best", "last"} or checkpoints_dir.name == "checkpoints":
        for candidate in (
            checkpoints_dir / "last" / "pretrained_model",
            checkpoints_dir / "best" / "pretrained_model",
        ):
            if (candidate / "model.safetensors").is_file():
                return candidate
        numeric = sorted(
            (
                item / "pretrained_model"
                for item in checkpoints_dir.iterdir()
                if item.is_dir() and item.name.isdigit() and (item / "pretrained_model" / "model.safetensors").is_file()
            ),
            key=lambda item: int(item.parent.name),
        )
        if numeric:
            return numeric[-1]

    raise FileNotFoundError(f"Could not resolve ACT checkpoint under {checkpoint}")


def ensure_best_checkpoint(train_output: Path) -> Path:
    checkpoints_dir = train_output / "checkpoints"
    best = checkpoints_dir / "best"
    if best.exists() and any(best.iterdir()):
        return best

    pretrained = resolve_checkpoint_path(checkpoints_dir / "last")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if best.is_symlink() or best.exists():
        best.unlink()
    best.symlink_to(pretrained.resolve())
    return best


def resolve_eval_dataset(prepared_root: Path, config: dict) -> Path:
    eval_split = config["eval_split"]
    primary = prepared_root / f"{config['experiment']}_eval_{eval_split}"
    if primary.exists():
        return primary
    shared = prepared_root / f"act_env_b_eval_{eval_split}"
    if shared.exists():
        return shared
    return primary


def render_template(template: list[str], config: dict, output: Path, checkpoint: str | None, prepared_dataset: Path) -> list[str]:
    training = config["training"]
    policy = config["policy"]
    values = {
        "python": sys.executable,
        "dataset_repo_id": config["dataset_repo_id"],
        "local_repo_id": config.get("local_repo_id", config["dataset_repo_id"].replace("/", "_")),
        "dataset_root": config["dataset_root"],
        "prepared_dataset": str(prepared_dataset),
        "output": str(output),
        "checkpoint": checkpoint or str(output / "checkpoints" / "last"),
        "steps": training["steps"],
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
        "num_workers": training.get("num_workers", 4),
        "seed": training["seed"],
        "chunk_size": policy["chunk_size"],
        "experiment": config["experiment"],
    }
    return [part.format(**values) for part in template]


def main() -> int:
    os.environ.setdefault("CVFINAL_DATA_ROOT", "/workspace/cv_final_data")
    parser = argparse.ArgumentParser(description="Train/evaluate ACT on CALVIN LeRobot splits.")
    parser.add_argument("mode", choices=["train", "eval", "manifest"])
    parser.add_argument("--repo", required=False, help="Path to a LeRobot checkout. Added to PYTHONPATH if provided.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", help="Checkpoint for eval mode.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-prepare", action="store_true", help="Recreate prepared local dataset roots.")
    args = parser.parse_args()

    config = load_json(args.config)
    output = Path(args.output)
    if not args.dry_run and args.mode != "train":
        output.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(os.path.expandvars(config["dataset_root"])).expanduser()
    prepared_root = Path(os.path.expandvars(config.get("prepared_root", str(dataset_root.parent / "prepared")))).expanduser()
    if args.mode == "eval":
        splits = [config["eval_split"]]
        prepared_dataset = resolve_eval_dataset(prepared_root, config)
    else:
        splits = config["train_splits"]
        prepared_dataset = prepared_root / config["experiment"]
    try:
        if not args.dry_run and (args.force_prepare or not prepared_dataset.exists()):
            prepare_dataset(dataset_root, splits, prepared_dataset, force=args.force_prepare)
        elif not args.dry_run:
            print(f"Using existing prepared dataset: {prepared_dataset}")
        manifest = make_prepared_manifest(config, output, prepared_dataset)
    except (FileNotFoundError, RuntimeError) as exc:
        if not args.dry_run:
            print(exc, file=sys.stderr)
            return 2
        manifest = output.parent / f"{config['experiment']}_prepared_manifest.json"
        print(f"DRY RUN: would prepare {splits} at {prepared_dataset}")
        print(f"DRY RUN: would create {manifest}")

    if args.mode == "manifest":
        print(manifest)
        return 0

    if args.mode == "eval":
        checkpoint = resolve_checkpoint_path(args.checkpoint or output / "checkpoints" / "last")
        eval_script = Path(__file__).resolve().parent / "eval_act_offline.py"
        eval_episodes = config.get("eval", {}).get("n_episodes", 50)
        command = [
            sys.executable,
            str(eval_script),
            "--repo-id",
            config.get("local_repo_id", config["dataset_repo_id"]),
            "--dataset-root",
            str(prepared_dataset),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--n-episodes",
            str(eval_episodes),
            "--batch-size",
            str(config["training"].get("eval_batch_size", config["training"]["batch_size"])),
            "--num-workers",
            str(config["training"].get("num_workers", 4)),
        ]
        print(" ".join(command))
        if args.dry_run:
            return 0
        env = os.environ.copy()
        if args.repo:
            repo = Path(args.repo).resolve()
            env["PYTHONPATH"] = str(repo / "src") + os.pathsep + str(repo) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(command, check=False, env=env).returncode

    template = config["command_template"][args.mode]
    command = render_template(template, config, output, args.checkpoint, prepared_dataset)
    print(" ".join(command))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    if args.repo:
        repo = Path(args.repo).resolve()
        env["PYTHONPATH"] = str(repo / "src") + os.pathsep + str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(command, check=False, env=env).returncode
    if result == 0 and args.mode == "train":
        try:
            best = ensure_best_checkpoint(output)
            print(f"Linked best checkpoint to {best}")
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
