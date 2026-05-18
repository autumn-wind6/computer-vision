from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import time

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import NUM_CLASSES, build_pet_loaders
from models import build_model, split_parameter_groups


def parse_args() -> argparse.Namespace:
    # 所有实验超参数都通过命令行传入，run_experiments.ps1 会调用这些参数跑多组对比。
    parser = argparse.ArgumentParser(description="Train Oxford-IIIT Pet classifier.")
    parser.add_argument("--data-dir", default="data/oxford-iiit-pet")
    parser.add_argument("--output-dir", default="runs/task1/debug")
    # arch 控制模型结构：resnet18/resnet34/se_resnet18/se_resnet34。
    parser.add_argument("--arch", default="resnet18")
    # 加上 --pretrained 就使用 ImageNet 预训练；不加就是随机初始化。
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    # lr_head 给最后分类层 fc 使用；lr_backbone 给前面的预训练特征提取层使用。
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> int:
    # logits 是模型输出的每类分数，argmax 取分数最高的类别作为预测结果。
    return int((logits.argmax(dim=1) == targets).sum().item())


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """跑一个 epoch。

    传入 optimizer 时表示训练模式，会反向传播和更新参数；
    optimizer=None 时表示验证/测试模式，只计算 loss 和 accuracy。
    """

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    # 只有训练阶段需要梯度；验证/测试阶段关闭梯度可以节省显存和时间。
    with torch.set_grad_enabled(is_train):
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_train:
                # 清空上一轮 batch 的梯度。
                optimizer.zero_grad(set_to_none=True)

            # 前向传播：输入图片，得到 37 个类别的分数。
            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                # 反向传播计算梯度，然后 optimizer.step() 更新模型参数。
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            # 累加总 loss 和正确预测数量，最后换算成整个 epoch 的平均值。
            total_loss += float(loss.item()) * batch_size
            total_correct += accuracy(logits.detach(), targets)
            total_count += batch_size
            progress.set_postfix(
                loss=f"{total_loss / total_count:.4f}",
                acc=f"{total_correct / total_count:.4f}",
            )

    return total_loss / total_count, total_correct / total_count


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
) -> None:
    # 保存模型权重和优化器状态，方便之后复现或继续训练。
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    # 固定随机种子，让数据划分和随机初始化尽量可复现。
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 保存本次实验配置，报告里可以根据 config.json 追溯每次实验参数。
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)

    # 读取 Oxford-IIIT Pet 的训练、验证、测试数据。
    train_loader, val_loader, test_loader = build_pet_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        download=not args.no_download,
    )

    device = torch.device(args.device)
    # 根据 arch 和 pretrained 创建模型；pretrained=True 时加载 ImageNet 权重。
    model = build_model(args.arch, pretrained=args.pretrained, num_classes=NUM_CLASSES).to(device)
    # 多分类任务使用交叉熵损失。
    criterion = nn.CrossEntropyLoss()
    # split_parameter_groups 会把 fc 和 backbone 分成两组，实现不同学习率微调。
    optimizer = AdamW(
        split_parameter_groups(model, args.lr_backbone, args.lr_head, args.weight_decay)
    )
    # 余弦退火学习率调度器：训练过程中学习率逐渐变小。
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history_path = output_dir / "history.csv"
    best_val_acc = -1.0
    best_metrics: dict = {}
    last_metrics: dict = {}

    with history_path.open("w", newline="", encoding="utf-8") as file:
        # history.csv 记录每一轮的 train/val/test loss 和 accuracy，报告画曲线就用它。
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "test_loss",
                "test_acc",
                "lr_backbone",
                "lr_head",
                "seconds",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            started = time()
            # 1. 用训练集更新参数。
            train_loss, train_acc = run_one_epoch(
                model, train_loader, criterion, device, optimizer=optimizer
            )
            # 2. 用验证集选择最佳模型。
            val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device)
            # 3. 用测试集记录当前模型表现，方便实验对比。
            test_loss, test_acc = run_one_epoch(model, test_loader, criterion, device)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr_backbone": optimizer.param_groups[0]["lr"],
                "lr_head": optimizer.param_groups[1]["lr"],
                "seconds": time() - started,
            }
            # 每个 epoch 写入一行，训练中断时也能看到已经完成的结果。
            writer.writerow(row)
            file.flush()
            last_metrics = row

            print(
                f"epoch {epoch:03d}/{args.epochs:03d} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
            )

            if val_acc > best_val_acc:
                # 验证集准确率更高时，保存为 best.pt。
                best_val_acc = val_acc
                best_metrics = row
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, row)

            # 更新学习率，下一轮 epoch 使用新的 lr。
            scheduler.step()

    # last.pt 保存最后一轮模型；best.pt 保存验证集最好的模型。
    save_checkpoint(output_dir / "last.pt", model, optimizer, args.epochs, last_metrics)
    print("best metrics:")
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
