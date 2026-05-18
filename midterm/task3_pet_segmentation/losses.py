from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """手动实现 Dice Loss，用来衡量预测区域和真实区域的重叠程度。"""

    def __init__(self, num_classes: int = 3, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        # smooth 防止分母为 0，也能让训练初期数值更稳定。
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W]，先 softmax 得到每个像素属于每类的概率。
        probs = logits.softmax(dim=1)
        # target: [B, H, W]，转换成 one-hot 后变为 [B, C, H, W]，才能和 probs 对齐相乘。
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        # Dice = 2 * 交集 / (预测面积 + 真实面积)，这里对 batch 和空间维度求和。
        intersection = (probs * target_one_hot).sum(dims)
        union = probs.sum(dims) + target_one_hot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # Dice 越大越好；训练时需要最小化损失，所以返回 1 - Dice。
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """组合损失：CE 提供稳定逐像素分类监督，Dice 缓解三分类像素不平衡。"""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, num_classes: int = 3) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def build_loss(name: str, num_classes: int = 3) -> nn.Module:
    """根据命令行参数选择三种实验损失之一。"""
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "dice":
        return DiceLoss(num_classes=num_classes)
    if name == "combo":
        return CombinedLoss(num_classes=num_classes)
    raise ValueError("loss must be one of: ce, dice, combo")


@torch.no_grad()
def mean_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> tuple[float, list[float]]:
    """单独计算 mIoU 的辅助函数；训练主流程里使用 confusion matrix 版本统计整轮指标。"""
    pred = logits.argmax(dim=1)
    ious: list[float] = []
    for class_id in range(num_classes):
        pred_mask = pred == class_id
        target_mask = target == class_id
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        ious.append(float("nan") if union == 0 else intersection / union)

    valid = [value for value in ious if value == value]
    return sum(valid) / max(1, len(valid)), ious
