# 任务 3：Oxford-IIIT Pet 三分类图像分割

本目录实现一个从零搭建的经典 U-Net，用 Oxford-IIIT Pet 的 trimap 标注训练三分类语义分割模型，并对比三种损失：

- `ce`：标准 Cross-Entropy Loss
- `dice`：手动实现的 Dice Loss
- `combo`：Cross-Entropy Loss + Dice Loss

## 数据与标签

Oxford-IIIT Pet trimap 的原始标签为：

- `1`：pet
- `2`：border
- `3`：background

训练时会转换为 `0/1/2` 三分类标签。代码不使用任何预训练权重，U-Net 参数从随机初始化开始训练。

## 单次训练

已有数据压缩包位于 `data\oxford-iiit-pet\oxford-iiit-pet\images.tar.gz` 和 `annotations.tar.gz`。如果解压目录访问异常，`--prepare-data` 会重新解到 `data\oxford-iiit-pet-prepared`。

```powershell
python .\task3_pet_segmentation\train.py `
  --data-dir .\data\oxford-iiit-pet `
  --prepare-data `
  --loss combo `
  --epochs 8 `
  --batch-size 8 `
  --image-size 128 `
  --max-train-samples 600 `
  --max-val-samples 200 `
  --max-test-samples 200 `
  --output-dir .\runs\task3\combo
```

## 三组损失对比

```powershell
.\task3_pet_segmentation\run_experiments.ps1
```

每组实验输出到 `runs/task3/<loss>/`：

- `config.json`：实验配置
- `history.csv`：每轮 train/val loss、mIoU、各类 IoU
- `best.pt`：验证集 mIoU 最佳模型
- `last.pt`：最后一轮模型
- `summary.json`：最佳模型在测试子集上的 mIoU、pet/border/background IoU、像素准确率

## 报告记录建议

| 实验 | 损失函数 | 训练子集 | 验证子集 | 测试子集 | 记录指标 |
| --- | --- | --- | --- | --- | --- |
| CE | Cross-Entropy | 600 | 200 | 200 | test_miou / test_pet_iou / test_border_iou / test_bg_iou |
| Dice | Dice Loss | 600 | 200 | 200 | test_miou / test_pet_iou / test_border_iou / test_bg_iou |
| Combo | CE + Dice | 600 | 200 | 200 | test_miou / test_pet_iou / test_border_iou / test_bg_iou |

一般现象：CE 更稳定；Dice 直接优化区域重叠，对类别不平衡更敏感；组合损失通常在收敛稳定性和小区域类别 IoU 之间更均衡。
