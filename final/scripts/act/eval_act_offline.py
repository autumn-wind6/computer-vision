#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved: dict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            moved[key] = [item.to(device, non_blocking=True) for item in value]
        else:
            moved[key] = value
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline ACT evaluation on a prepared LeRobot dataset.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    pretrained_path = args.checkpoint
    if (pretrained_path / "pretrained_model" / "model.safetensors").is_file():
        pretrained_path = pretrained_path / "pretrained_model"
    if not (pretrained_path / "model.safetensors").is_file():
        print(f"Missing model checkpoint under {pretrained_path}", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    meta = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    policy_cfg.pretrained_path = str(pretrained_path)
    policy_cfg.device = args.device

    episode_ids = list(range(min(args.n_episodes, meta.total_episodes)))
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)
    dataset = LeRobotDataset(
        args.repo_id,
        root=dataset_root,
        episodes=episode_ids,
        delta_timestamps=delta_timestamps,
    )
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )

    policy = make_policy(cfg=policy_cfg, ds_meta=meta)
    policy.to(device)
    policy.eval()
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(pretrained_path),
        dataset_stats=meta.stats,
    )

    total_l1 = 0.0
    total_mse = 0.0
    total_elems = 0
    batch_rows: list[dict[str, float | int]] = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader, start=1):
            if hasattr(policy, "reset"):
                policy.reset()
            batch = move_batch_to_device(batch, device)
            batch = preprocessor(batch)

            target = batch[ACTION]
            pred = policy.predict_action_chunk(batch)
            valid = (~batch["action_is_pad"]).unsqueeze(-1)
            abs_err = (pred - target).abs() * valid
            sq_err = ((pred - target) ** 2) * valid
            elems = int(valid.sum().item() * target.shape[-1])
            if elems == 0:
                continue

            batch_l1 = float(abs_err.sum().item() / elems)
            batch_mse = float(sq_err.sum().item() / elems)
            total_l1 += abs_err.sum().item()
            total_mse += sq_err.sum().item()
            total_elems += elems
            batch_rows.append(
                {
                    "step": step,
                    "action_l1": batch_l1,
                    "action_mse": batch_mse,
                    "policy_l1_loss": batch_l1,
                }
            )

    if total_elems == 0:
        print("No valid action frames found in evaluation dataset.", file=sys.stderr)
        return 2

    summary = {
        "repo_id": args.repo_id,
        "dataset_root": str(dataset_root),
        "checkpoint": str(pretrained_path),
        "n_episodes": len(episode_ids),
        "n_batches": len(batch_rows),
        "action_l1": total_l1 / total_elems,
        "action_mse": total_mse / total_elems,
        "device": args.device,
    }

    metrics_json = output / "metrics.json"
    metrics_csv = output / "metrics.csv"
    metrics_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "action_l1", "action_mse", "policy_l1_loss"])
        writer.writeheader()
        writer.writerows(batch_rows)
        writer.writerow(
            {
                "step": "summary",
                "action_l1": summary["action_l1"],
                "action_mse": summary["action_mse"],
                "policy_l1_loss": summary["action_l1"],
            }
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(metrics_json)
    print(metrics_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
