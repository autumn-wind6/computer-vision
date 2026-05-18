from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as F


IMAGE_SIZE = 128
# ImageNet 均值/方差。虽然本任务不使用预训练权重，但这个归一化方式很常见，
# 可以让输入数值范围更稳定，训练时梯度也更平滑。
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _is_readable_dir(path: Path) -> bool:
    """检查目录是否存在且可读取；这里捕获 PermissionError 是为了兼容本机数据目录权限异常。"""
    try:
        return path.is_dir() and any(path.iterdir())
    except PermissionError:
        return False


def _extract_tar(archive: Path, destination: Path) -> None:
    """把 Oxford-IIIT Pet 官方压缩包解压到一个可访问的新目录。"""
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination)


def resolve_pet_root(data_dir: str | Path, prepare_data: bool = False) -> Path:
    """找到真正包含 images/annotations 的数据集根目录。"""
    data_dir = Path(data_dir)
    # torchvision 下载和手动解压可能产生不同目录结构，所以这里把常见布局都试一遍。
    candidates = [
        data_dir / "oxford-iiit-pet",
        data_dir,
        data_dir.parent / "oxford-iiit-pet-prepared",
        data_dir.parent / "oxford-iiit-pet-prepared" / "oxford-iiit-pet",
    ]
    for candidate in candidates:
        if _is_readable_dir(candidate / "images") and _is_readable_dir(candidate / "annotations"):
            return candidate

    archive_root = data_dir / "oxford-iiit-pet"
    if not (archive_root / "images.tar.gz").exists():
        archive_root = data_dir

    # 如果已有目录不可读，就从 images.tar.gz / annotations.tar.gz 重新解压一份。
    prepared_parent = data_dir.parent / "oxford-iiit-pet-prepared"
    if prepare_data or (archive_root / "images.tar.gz").exists():
        _extract_tar(archive_root / "images.tar.gz", prepared_parent)
        _extract_tar(archive_root / "annotations.tar.gz", prepared_parent)
        for prepared_root in (prepared_parent, prepared_parent / "oxford-iiit-pet"):
            if _is_readable_dir(prepared_root / "images") and _is_readable_dir(prepared_root / "annotations"):
                return prepared_root

    raise FileNotFoundError(
        "Could not find readable Oxford-IIIT Pet images/annotations. "
        "Pass --prepare-data with a data directory containing images.tar.gz and annotations.tar.gz."
    )


class SegmentationTransform:
    """图像和 mask 必须做完全一致的几何变换，否则像素标注会对不上。"""

    def __init__(self, train: bool, image_size: int = IMAGE_SIZE) -> None:
        self.train = train
        self.image_size = image_size

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        mask = mask.convert("L")

        # 图像用双线性插值更平滑；mask 是类别编号，必须用最近邻插值，避免产生 1/2/3 之外的类别值。
        image = F.resize(image, (self.image_size, self.image_size), interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resize(mask, (self.image_size, self.image_size), interpolation=F.InterpolationMode.NEAREST)

        # 轻量数据增强：水平翻转。图像和 mask 同时翻转，保证标注仍然正确。
        if self.train and torch.rand(()) < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # image_tensor: [3, H, W]，float；mask_tensor: [H, W]，long。
        image_tensor = F.normalize(F.to_tensor(image), mean=MEAN, std=STD)
        mask_tensor = F.pil_to_tensor(mask).squeeze(0).long()

        # Oxford trimap 的原始类别：1=pet，2=border，3=background。
        # 训练时转换为 0/1/2 三分类标签，避免 CrossEntropyLoss 的类别索引从 1 开始。
        class_mask = (mask_tensor - 1).clamp(min=0, max=2).long()
        return image_tensor, class_mask


class OxfordPetSegmentation(Dataset):
    """自定义 Dataset：返回一张 RGB 图片和对应的二值分割 mask。"""

    def __init__(self, root: str | Path, split: str, transform: Callable | None = None) -> None:
        if split not in {"trainval", "test"}:
            raise ValueError("split must be 'trainval' or 'test'")

        self.root = Path(root)
        self.transform = transform
        list_file = self.root / "annotations" / f"{split}.txt"
        if not list_file.exists():
            raise FileNotFoundError(f"Missing split file: {list_file}")

        self.samples: list[tuple[Path, Path]] = []
        for line in list_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            # split 文件每行第一个字段是样本名，例如 Abyssinian_1。
            # 图片在 images/name.jpg，分割标注在 annotations/trimaps/name.png。
            name = line.split()[0]
            image_path = self.root / "images" / f"{name}.jpg"
            mask_path = self.root / "annotations" / "trimaps" / f"{name}.png"
            if image_path.exists() and mask_path.exists():
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"No segmentation samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        if self.transform is not None:
            return self.transform(image, mask)
        return F.to_tensor(image), torch.as_tensor(mask)


def _limited_subset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    """从完整数据集中固定随机抽样，保证每次运行使用同一批轻量级样本。"""
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def build_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    image_size: int = IMAGE_SIZE,
    max_train_samples: int | None = 600,
    max_val_samples: int | None = 200,
    max_test_samples: int | None = 200,
    val_ratio: float = 0.15,
    prepare_data: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, Path]:
    root = resolve_pet_root(data_dir, prepare_data=prepare_data)
    # trainval 原始划分被我们再切成 train/val；test 使用官方 test 划分。
    trainval_train = OxfordPetSegmentation(root, "trainval", SegmentationTransform(True, image_size))
    trainval_val = OxfordPetSegmentation(root, "trainval", SegmentationTransform(False, image_size))
    test = OxfordPetSegmentation(root, "test", SegmentationTransform(False, image_size))

    # 使用固定随机种子切分验证集，方便三种损失函数做公平对比。
    val_size = max(1, int(len(trainval_train) * val_ratio))
    indices = torch.randperm(len(trainval_train), generator=torch.Generator().manual_seed(42))
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    # 默认只使用小子集训练，满足题目“无预训练时保证收敛速度”的要求。
    train_set = _limited_subset(Subset(trainval_train, train_indices), max_train_samples, seed=1)
    val_set = _limited_subset(Subset(trainval_val, val_indices), max_val_samples, seed=2)
    test_set = _limited_subset(test, max_test_samples, seed=3)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, root
