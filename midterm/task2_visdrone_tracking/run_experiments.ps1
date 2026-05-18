$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$DataRoot = Join-Path $Root "data"
$TaskRoot = $PSScriptRoot
$RunRoot = Join-Path $Root "runs\task2\detect"
$OutputRoot = Join-Path $Root "runs\task2\outputs"
$Model = Join-Path $RunRoot "visdrone_yolov8s_lowvram\weights\best.pt"
$Video = Join-Path $DataRoot "video.mp4"

python "$PSScriptRoot\data.py" `
  --visdrone-root (Join-Path $DataRoot "VisDrone") `
  --output-yaml (Join-Path $TaskRoot "visdrone_det.yaml")

python "$PSScriptRoot\train.py" `
  --data (Join-Path $TaskRoot "visdrone_det.yaml") `
  --epochs 30 `
  --batch 6 `
  --device 0 `
  --project $RunRoot `
  --name "visdrone_yolov8s_lowvram"

python "$PSScriptRoot\track_video.py" `
  --model $Model `
  --video $Video `
  --output (Join-Path $OutputRoot "result_tracking_custom.mp4") `
  --csv-output (Join-Path $OutputRoot "tracking_log.csv")

python "$PSScriptRoot\analyze_occlusion.py" `
  --model $Model `
  --video $Video `
  --start-frame 165 `
  --frames 150 `
  --right-half-only `
  --output-dir (Join-Path $OutputRoot "silver_suv_analysis")
