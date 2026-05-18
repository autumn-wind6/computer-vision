from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import build_loaders
from losses import build_loss
from models import UNet

NUM_CLASSES = 3


def seed_everything(seed: int) -> None:
    """固定随机种子，保证数据划分和训练初始化更可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_confusion_matrix(confusion: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> None:
    """累积 3x3 混淆矩阵，用于整轮计算 pet/border/background IoU 和 mIoU。"""
    pred = logits.argmax(dim=1).view(-1)
    target = target.view(-1)
    valid = (target >= 0) & (target < confusion.size(0))
    # indices = 真值类别 * 类别数 + 预测类别，可以一次性统计所有像素的落点。
    indices = confusion.size(0) * target[valid] + pred[valid]
    confusion += torch.bincount(indices, minlength=confusion.numel()).view_as(confusion).cpu()


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, float]:
    """由混淆矩阵计算 pixel accuracy 和每类 IoU。"""
    intersection = confusion.diag()
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = intersection.float() / union.clamp_min(1).float()
    accuracy = intersection.sum().float() / confusion.sum().clamp_min(1).float()
    return {
        "pixel_acc": accuracy.item(),
        "pet_iou": iou[0].item(),
        "border_iou": iou[1].item(),
        "bg_iou": iou[2].item(),
        "miou": iou.mean().item(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """跑完一个 epoch；optimizer 为空表示验证/测试，否则表示训练。"""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    progress = tqdm(loader, leave=False, desc="train" if training else "eval")
    for images, masks in progress:
        # images: [B, 3, H, W]；masks: [B, H, W]，像素值为 0/1/2。
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(training):
            # logits: [B, 3, H, W]，每个像素三个类别分数。
            logits = model(images)
            loss = criterion(logits, masks)

        if training:
            # 标准 PyTorch 训练三步：清梯度、反向传播、更新参数。
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        # 指标在 CPU 上累积，避免显存占用，并且不用参与梯度计算。
        update_confusion_matrix(confusion, logits.detach().cpu(), masks.detach().cpu())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def save_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    """每轮训练后保存 history.csv，方便写报告和画曲线。"""
    fieldnames = [
        "epoch",
        "train_loss",
        "train_miou",
        "train_pet_iou",
        "train_border_iou",
        "train_bg_iou",
        "val_loss",
        "val_miou",
        "val_pet_iou",
        "val_border_iou",
        "val_bg_iou",
        "val_pixel_acc",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_history(path: Path) -> list[dict[str, float | int]]:
    """读取已完成的 history.csv，用于中断后继续训练。"""
    if not path.exists():
        return []

    rows: list[dict[str, float | int]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed: dict[str, float | int] = {}
            for key, value in row.items():
                if key == "epoch":
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def parse_args() -> argparse.Namespace:
    """集中定义所有命令行参数，三组损失实验只需要改 --loss 和 --output-dir。"""
    parser = argparse.ArgumentParser(description="Train a U-Net on Oxford-IIIT Pet foreground segmentation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/task3/combo"))
    parser.add_argument("--loss", choices=["ce", "dice", "combo"], default="combo")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=600)
    parser.add_argument("--max-val-samples", type=int, default=200)
    parser.add_argument("--max-test-samples", type=int, default=200)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 正式实验：读取 Oxford-IIIT Pet，构造轻量级 train/val/test DataLoader。
    train_loader, val_loader, test_loader, data_root = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        prepare_data=args.prepare_data,
    )

    # 保存本次实验配置，报告中可以直接说明数据量、输入尺寸、损失函数等参数。
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["device"] = str(device)
    config["data_root"] = str(data_root)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # 从零初始化 U-Net；题目要求不能用预训练权重，所以这里没有加载任何 pretrained 参数。
    model = UNet(num_classes=NUM_CLASSES, base_channels=args.base_channels).to(device)
    # 根据 --loss 选择 ce / dice / combo 三种损失。
    criterion = build_loss(args.loss, num_classes=NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = args.output_dir / "history.csv"
    last_path = args.output_dir / "last.pt"
    history = load_history(history_path)
    completed_epochs = int(history[-1]["epoch"]) if history else 0
    best_val_miou = max((float(row["val_miou"]) for row in history), default=-1.0)

    if completed_epochs > 0:
        if not last_path.exists():
            raise FileNotFoundError(
                f"{history_path} exists but {last_path} is missing; "
                "please use a new --output-dir or restore last.pt."
            )
        last_state = torch.load(last_path, map_location=device)
        model.load_state_dict(last_state["model"])
        optimizer.load_state_dict(last_state["optimizer"])
        print(f"Resuming {args.output_dir} from epoch {completed_epochs + 1}/{args.epochs}")

    if completed_epochs >= args.epochs:
        print(f"{args.output_dir} already has {completed_epochs} completed epochs.")

    for epoch in range(completed_epochs + 1, args.epochs + 1):
        # 先训练一轮，再在验证集上评估，最后保存日志和模型。
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_miou": train_metrics["miou"],
            "train_pet_iou": train_metrics["pet_iou"],
            "train_border_iou": train_metrics["border_iou"],
            "train_bg_iou": train_metrics["bg_iou"],
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "val_pet_iou": val_metrics["pet_iou"],
            "val_border_iou": val_metrics["border_iou"],
            "val_bg_iou": val_metrics["bg_iou"],
            "val_pixel_acc": val_metrics["pixel_acc"],
        }
        history.append(row)
        save_history(history_path, history)

        # last.pt 保存最后一轮；best.pt 保存验证集 mIoU 最优的一轮。
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_miou": val_metrics["miou"],
            "config": config,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_metrics["miou"] > best_val_miou:
            best_val_miou = val_metrics["miou"]
            torch.save(checkpoint, args.output_dir / "best.pt")

        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_miou={train_metrics['miou']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_miou={val_metrics['miou']:.4f} "
            f"val_pet_iou={val_metrics['pet_iou']:.4f}"
        )

    # 训练结束后加载验证集最优模型，在测试集上得到最终对比指标。
    best_state = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_state["model"])
    test_metrics = run_epoch(model, test_loader, criterion, device)
    summary = {
        "best_val_miou": best_val_miou,
        "test_loss": test_metrics["loss"],
        "test_miou": test_metrics["miou"],
        "test_pet_iou": test_metrics["pet_iou"],
        "test_border_iou": test_metrics["border_iou"],
        "test_bg_iou": test_metrics["bg_iou"],
        "test_pixel_acc": test_metrics["pixel_acc"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
