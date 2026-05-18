$ErrorActionPreference = "Stop"

# Run the three loss-function comparison experiments required by task 3.
$Root = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $Root "data\oxford-iiit-pet"
$RunRoot = Join-Path $Root "runs\task3"

$common = @(
  "--data-dir", $DataDir,
  "--prepare-data",
  "--epochs", "8",
  "--batch-size", "8",
  "--image-size", "128",
  "--num-workers", "0",
  "--max-train-samples", "600",
  "--max-val-samples", "200",
  "--max-test-samples", "200"
)

python "$PSScriptRoot\train.py" @common --loss ce --output-dir (Join-Path $RunRoot "ce")
python "$PSScriptRoot\train.py" @common --loss dice --output-dir (Join-Path $RunRoot "dice")
python "$PSScriptRoot\train.py" @common --loss combo --output-dir (Join-Path $RunRoot "combo")
