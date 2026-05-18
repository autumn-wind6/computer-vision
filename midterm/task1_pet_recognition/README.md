# 任务 1：Oxford-IIIT Pet 宠物识别

本目录实现作业任务 1 的训练与对比实验，数据集使用 Oxford-IIIT Pet Dataset：
https://www.robots.ox.ac.uk/~vgg/data/pets/

代码使用 PyTorch / torchvision，支持：

- ImageNet 预训练 ResNet-18 / ResNet-34 baseline
- 随机初始化 ResNet-18 的预训练消融实验
- 训练轮数和学习率组合的超参数分析
- 在 baseline 上加入 SE-block 的注意力机制实验
- 输出每轮 train/val/test accuracy，并保存最佳模型

## 环境

```powershell
pip install torch torchvision tqdm
```

## 单次训练示例

```powershell
python .\task1_pet_recognition\train.py `
  --data-dir .\data\oxford-iiit-pet `
  --arch resnet18 `
  --pretrained `
  --epochs 20 `
  --batch-size 32 `
  --lr-head 1e-3 `
  --lr-backbone 1e-4 `
  --output-dir .\runs\task1\resnet18_pretrained
```

## 批量实验

运行预设脚本：

```powershell
.\task1_pet_recognition\run_experiments.ps1
```

脚本包含以下正式实验，输出统一保存到 `runs/task1/` 下：

| 对应要求 | 实验目录 | 模型 | 初始化 | epochs | lr_head | lr_backbone |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | `resnet18_pretrained` | ResNet-18 | ImageNet | 20 | 1e-3 | 1e-4 |
| 超参数 | `resnet18_e10_lr_mid` | ResNet-18 | ImageNet | 10 | 1e-3 | 1e-4 |
| 超参数 | `resnet18_e30_lr_mid` | ResNet-18 | ImageNet | 30 | 1e-3 | 1e-4 |
| 超参数 | `resnet18_e10_lr_small` | ResNet-18 | ImageNet | 10 | 5e-4 | 5e-5 |
| 超参数 | `resnet18_lr_small` | ResNet-18 | ImageNet | 20 | 5e-4 | 5e-5 |
| 超参数 | `resnet18_e30_lr_small` | ResNet-18 | ImageNet | 30 | 5e-4 | 5e-5 |
| 超参数 | `resnet18_e10_lr_large` | ResNet-18 | ImageNet | 10 | 3e-3 | 3e-4 |
| 超参数 | `resnet18_lr_large` | ResNet-18 | ImageNet | 20 | 3e-3 | 3e-4 |
| 超参数 | `resnet18_e30_lr_large` | ResNet-18 | ImageNet | 30 | 3e-3 | 3e-4 |
| 预训练消融 | `resnet18_scratch` | ResNet-18 | Random | 20 | 1e-3 | 1e-3 |
| 注意力机制 | `se_resnet18_pretrained` | SE-ResNet-18 | ImageNet | 20 | 1e-3 | 1e-4 |

## 输出文件

每次训练会在 `--output-dir` 下生成：

- `config.json`：本次实验配置
- `history.csv`：每轮 loss / accuracy
- `best.pt`：验证集准确率最高的模型
- `last.pt`：最后一轮模型

报告中建议使用 `history.csv` 统计 `best val acc` 和对应的 `test acc`，并绘制训练曲线。
