from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGE_SIZE = 224
NUM_CLASSES = 37


def build_transforms(train: bool) -> transforms.Compose:
    """构造图像预处理。

    训练集使用随机裁剪、翻转和颜色扰动做数据增强；
    验证集/测试集不能随机增强，否则每次评估结果会不稳定，所以只做中心裁剪。
    """

    if train:
        return transforms.Compose(
            [
                # 先统一缩放到 256x256，再随机裁剪到模型输入需要的 224x224。
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                # ToTensor 把 PIL 图片转成 PyTorch 张量，像素值范围变成 0~1。
                transforms.ToTensor(),
                # 使用 ImageNet 的均值和方差归一化，和预训练 ResNet 的输入分布保持一致。
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            # 验证/测试也必须使用同样的归一化方式。
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_pet_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    val_ratio: float = 0.15,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    # OxfordIIITPet 的 category 标签就是 37 个猫狗品种类别，正好对应本任务的分类目标。
    trainval_train = datasets.OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        # 训练子集使用带随机增强的 transform。
        transform=build_transforms(train=True),
        download=download,
    )
    trainval_val = datasets.OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        # 验证子集来自同一个 trainval split，但 transform 不使用随机增强。
        transform=build_transforms(train=False),
        download=download,
    )
    test = datasets.OxfordIIITPet(
        root=data_dir,
        split="test",
        target_types="category",
        # 官方 test split 只用于最终测试。
        transform=build_transforms(train=False),
        download=download,
    )

    # torchvision 的 OxfordIIITPet 没有单独验证集，这里从 trainval 中固定划出 15% 做验证。
    val_size = int(len(trainval_train) * val_ratio)
    # 固定随机种子，保证每次划分出来的训练/验证集合一致。
    indices = torch.randperm(len(trainval_train), generator=torch.Generator().manual_seed(42))
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    train_set = Subset(trainval_train, train_indices)
    val_set = Subset(trainval_val, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        # 训练集 shuffle=True 可以打乱样本顺序，有利于优化。
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        # 验证/测试不需要打乱，方便复现实验结果。
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader
