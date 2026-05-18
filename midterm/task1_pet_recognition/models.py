from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34
from torchvision.models.resnet import BasicBlock, ResNet


NUM_CLASSES = 37


class SEBlock(nn.Module):
    """SE 注意力模块：根据每个通道的重要性，给特征图的通道重新加权。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        # reduction 控制中间层压缩比例，先把通道数压小，再恢复到原通道数。
        hidden = max(channels // reduction, 4)
        # AdaptiveAvgPool2d(1) 会把每个通道压缩成 1 个数，相当于提取通道的全局信息。
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            # Sigmoid 输出 0~1 的权重，用来表示每个通道的重要程度。
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self.fc(self.pool(x)) 得到通道权重，再和原特征 x 相乘。
        return x * self.fc(self.pool(x))


class SEBasicBlock(BasicBlock):
    """在 ResNet 的 BasicBlock 中加入 SEBlock，用于注意力机制实验。"""

    def __init__(self, *args, se_reduction: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # bn2.num_features 就是当前残差块输出的通道数。
        self.se = SEBlock(self.bn2.num_features, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # identity 是残差连接的原始输入，最后会和卷积分支相加。
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 和普通 ResNet BasicBlock 的区别：这里多了一步 SE 通道注意力。
        out = self.se(out)

        # 如果输入输出维度不同，需要用 downsample 把 identity 调整到同样形状。
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：卷积分支 + 原始输入分支。
        out += identity
        out = self.relu(out)

        return out


def _set_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    # ImageNet 预训练的 ResNet 最后一层原本是 1000 类，这里替换为宠物数据集的 37 类。
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _load_imagenet_backbone(model: nn.Module, arch: Literal["resnet18", "resnet34"]) -> None:
    """给 SE-ResNet 加载普通 ResNet 的 ImageNet 预训练权重。"""

    if arch == "resnet18":
        state = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    elif arch == "resnet34":
        state = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
    else:
        raise ValueError(f"Unsupported pretrained SE backbone: {arch}")

    # fc 是最后分类层，类别数不同，不能加载 ImageNet 的 1000 类分类头。
    state = {key: value for key, value in state.items() if not key.startswith("fc.")}
    # strict=False 允许 SE-ResNet 里新增的 se.* 参数随机初始化。
    model.load_state_dict(state, strict=False)


def build_model(arch: str, pretrained: bool, num_classes: int = NUM_CLASSES) -> nn.Module:
    """根据命令行参数创建模型。

    pretrained=True 表示使用 ImageNet 预训练权重；
    pretrained=False 表示随机初始化，用于预训练消融实验。
    """

    if arch == "resnet18":
        # torchvision 的 ResNet18_Weights.IMAGENET1K_V1 就是 ImageNet 预训练参数。
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    elif arch == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
    elif arch == "se_resnet18":
        # [2, 2, 2, 2] 是 ResNet-18 每个 stage 的 BasicBlock 数量。
        model = ResNet(SEBasicBlock, [2, 2, 2, 2])
        if pretrained:
            _load_imagenet_backbone(model, "resnet18")
    elif arch == "se_resnet34":
        # [3, 4, 6, 3] 是 ResNet-34 每个 stage 的 BasicBlock 数量。
        model = ResNet(SEBasicBlock, [3, 4, 6, 3])
        if pretrained:
            _load_imagenet_backbone(model, "resnet34")
    else:
        raise ValueError(
            "arch must be one of: resnet18, resnet34, se_resnet18, se_resnet34"
        )

    # 无论是否预训练，最后都要换成 37 类分类层。
    return _set_classifier(model, num_classes)


def split_parameter_groups(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
) -> list[dict]:
    """把模型参数分成两组，方便设置不同学习率。

    backbone：ResNet 前面的卷积/残差层，通常已经有 ImageNet 预训练知识，学习率小一点。
    head：最后的 fc 分类层，是新换的随机初始化层，学习率大一点。
    """

    backbone_params = []
    head_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fc."):
            # fc.* 是最后分类头，对应 optimizer 里的 lr_head。
            head_params.append(parameter)
        else:
            # 其余参数属于 backbone，对应 optimizer 里的 lr_backbone。
            backbone_params.append(parameter)

    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]
