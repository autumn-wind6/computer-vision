$ErrorActionPreference = "Stop"

# 项目根目录、训练脚本、数据目录与实验输出目录。
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
if (-not $ScriptDir) {
    $ScriptDir = (Get-Location).Path
}
$Root = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
if (-not $Root) {
    throw "无法解析项目根目录。请在 task1_pet_recognition 目录下执行: .\run_experiments.ps1"
}
$Train = Join-Path $ScriptDir "train.py"
$Data = Join-Path $Root "data\oxford-iiit-pet"
$Runs = Join-Path $Root "runs\task1"

# Learning-rate groups.
# small: conservative fine-tuning
# mid: baseline learning rate
# large: more aggressive fine-tuning
$LrSmallHead = "5e-4"
$LrSmallBackbone = "5e-5"
$LrMidHead = "1e-3"
$LrMidBackbone = "1e-4"
$LrLargeHead = "3e-3"
$LrLargeBackbone = "3e-4"

# (1) Baseline: ImageNet-pretrained ResNet-18, 20 epochs, mid learning rate.
python $Train --data-dir $Data --arch resnet18 --pretrained --epochs 20 --batch-size 32 --lr-head $LrMidHead --lr-backbone $LrMidBackbone --output-dir (Join-Path $Runs "resnet18_pretrained")

# (2) Hyperparameter analysis: epochs 10/20/30 x small/mid/large learning rates.
# Mid learning rate. The 20-epoch mid run is the baseline above.
python $Train --data-dir $Data --arch resnet18 --pretrained --epochs 10 --batch-size 32 --lr-head $LrMidHead --lr-backbone $LrMidBackbone --output-dir (Join-Path $Runs "resnet18_e10_lr_mid")
python $Train --data-dir $Data --arch resnet18 --pretrained --epochs 30 --batch-size 32 --lr-head $LrMidHead --lr-backbone $LrMidBackbone --output-dir (Join-Path $Runs "resnet18_e30_lr_mid")

# Small learning rate.
python $Train --data-dir $Data --arch resnet18 --pretrained --epochs 10 --batch-size 32 --lr-head $LrSmallHead --lr-backbone $LrSmallBackbone --output-dir (Join-Path $Runs "resnet18_e10_lr_small")
python $Train --data-dir $Data --arch resnet18 --pretrained --epochs 20 --batch-size 32 --lr-head $LrSmallHead --lr-backbone $LrSmallBackbone --output-dir (Join-Path $Runs "resnet18_lr_small")
python $Train --data-dir $Data --arch resnet18 --pretrained --epochs 30 --batch-size 32 --lr-head $LrSmallHead --lr-backbone $LrSmallBackbone --output-dir (Join-Path $Runs "resnet18_e30_lr_small")

# Large learning rate.
python $Train --data-dir $Data --arch resnet18 --pretrained --epochs 10 --batch-size 32 --lr-head $LrLargeHead --lr-backbone $LrLargeBackbone --output-dir (Join-Path $Runs "resnet18_e10_lr_large")
python $Train --data-dir $Data --arch resnet18 --pretrained --epochs 20 --batch-size 32 --lr-head $LrLargeHead --lr-backbone $LrLargeBackbone --output-dir (Join-Path $Runs "resnet18_lr_large")
python $Train --data-dir $Data --arch resnet18 --pretrained --epochs 30 --batch-size 32 --lr-head $LrLargeHead --lr-backbone $LrLargeBackbone --output-dir (Join-Path $Runs "resnet18_e30_lr_large")

# (3) Pretraining ablation: no --pretrained, so ResNet-18 starts from random initialization.
python $Train --data-dir $Data --arch resnet18 --epochs 20 --batch-size 32 --lr-head 1e-3 --lr-backbone 1e-3 --output-dir (Join-Path $Runs "resnet18_scratch")

# (4) Attention mechanism: SE-blocks are added to the ResNet-18 BasicBlock.
python $Train --data-dir $Data --arch se_resnet18 --pretrained --epochs 20 --batch-size 32 --lr-head $LrMidHead --lr-backbone $LrMidBackbone --output-dir (Join-Path $Runs "se_resnet18_pretrained")
