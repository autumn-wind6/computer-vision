from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """U-Net 的基础卷积块：Conv-BN-ReLU 重复两次，保持图像尺寸不变。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """编码器下采样块：先 2x2 最大池化把分辨率减半，再提取更高层语义特征。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """解码器上采样块：转置卷积放大特征图，再与编码器同尺度特征拼接。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        # ConvTranspose2d 负责把空间尺寸扩大 2 倍，同时把通道数减半。
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 拼接后的通道数 = 上采样后的通道数 + skip connection 的通道数。
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 如果输入尺寸不是 16 的整数倍，上采样后可能和 skip 特征差 1 个像素，用 pad 对齐。
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        # skip connection 保留浅层边缘/纹理信息，对分割边界很重要。
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """经典 U-Net：左边编码器压缩语义，右边解码器恢复空间分辨率。"""

    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        # 编码器：分辨率逐层减半，通道数逐层翻倍。
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        # 解码器：分辨率逐层恢复，并通过 skip connection 拼接对应编码器特征。
        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels, base_channels)
        # 1x1 卷积把每个像素映射为 3 个类别分数：pet / border / background。
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x1-x4 会作为 skip connection 传给解码器，帮助恢复精细边界。
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # 从最深层语义特征开始逐层上采样，输出尺寸回到输入尺寸。
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # 返回 logits，形状为 [B, 3, H, W]；后续 CrossEntropy/Dice 会使用它计算损失。
        return self.outc(x)
